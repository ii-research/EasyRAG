"""
Pure FiD Evaluation Script with T5Gemma2 Backbone
==================================================

Evaluates pure FiD models (no source pointer prediction) trained with T5Gemma2.

Paper: "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering"
Authors: Izacard & Grave (2021)

T5Gemma2 Backbone: Zhang et al. (2025)

Metrics:
- Exact Match (EM): Primary metric
- F1: Token-level F1 score (optional)

No provenance metrics since pure FiD doesn't predict source pointers.

Usage:
    # Evaluate on validation set
    python evaluate_fid_pure_t5gemma.py --checkpoint checkpoints/fid_pure_t5gemma/step_5000

    # Quick test
    python evaluate_fid_pure_t5gemma.py --checkpoint checkpoints/fid_pure_t5gemma/step_5000 --max_samples 50

    # Verbose output
    python evaluate_fid_pure_t5gemma.py --checkpoint checkpoints/fid_pure_t5gemma/step_5000 --verbose
"""

import argparse
import json
import os
import re
import string
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

from transformers import AutoModelForSeq2SeqLM, AutoProcessor
from transformers.modeling_outputs import BaseModelOutput
from torch.nn.parallel import data_parallel

# Web Demo state reporting (optional - only used when running from web UI)
try:
    from web_demo.utils.state_io import update_step_state, StepStatus
    HAS_WEB_DEMO = True
except ImportError:
    HAS_WEB_DEMO = False


def report_evaluation_progress(current: int, total: int, em: float, f1: float):
    """Report evaluation progress to web demo (if available)."""
    if not HAS_WEB_DEMO:
        return
    try:
        progress = (current / total) * 100 if total > 0 else 0
        message = f"Evaluating {current}/{total} | EM: {em:.1f}% | F1: {f1:.1f}%"
        update_step_state(
            step_name="evaluate",
            progress=progress,
            message=message,
            status=StepStatus.RUNNING.value,
            extra={
                "current": current,
                "total": total,
                "exact_match": em,
                "f1": f1,
                "algorithm": "fid_pure",
                "model": "t5gemma"
            }
        )
    except Exception:
        pass  # Silently ignore web demo errors


class EncoderWrapper(nn.Module):
    """Wrapper for T5Gemma2 encoder to work with data_parallel."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return output.last_hidden_state


class FiDPureT5GemmaEvaluator:
    """
    Pure FiD evaluator with T5Gemma2 backbone (no source pointer, answer-only output).

    Metrics:
    - Exact Match (EM): Normalized string comparison
    - F1: Token-level F1 (optional)
    """

    def __init__(self, args: argparse.Namespace):
        """Initialize evaluator."""
        self.args = args
        self.device = torch.device(args.device)
        self.results = {}

        # Load model and tokenizer
        self._load_model()

    def _load_model(self) -> None:
        """Load T5Gemma2 model from checkpoint."""
        print(f"Loading T5Gemma2 model from {self.args.checkpoint}...")

        # Use AutoProcessor for T5Gemma2
        try:
            self.processor = AutoProcessor.from_pretrained(self.args.checkpoint)
            self.tokenizer = self.processor.tokenizer
        except Exception:
            # Fallback: load from base model if checkpoint doesn't have processor files
            print("  Processor not found in checkpoint, loading from base model...")
            base_model = "google/t5gemma-2-270m-270m"
            self.processor = AutoProcessor.from_pretrained(base_model)
            self.tokenizer = self.processor.tokenizer

        # Load model with BF16
        if self.args.bf16:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.args.checkpoint,
                torch_dtype=torch.bfloat16
            )
            print("  Loaded with BF16 precision")
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.checkpoint)
            print("  Loaded with FP32 precision")

        # T5Gemma2 fix: Set decoder_start_token_id if not set
        if self.model.config.decoder_start_token_id is None:
            self.model.config.decoder_start_token_id = self.model.config.bos_token_id or 2
            print(f"  Set decoder_start_token_id to {self.model.config.decoder_start_token_id}")

        self.model.to(self.device)
        self.model.eval()

        # Multi-GPU support
        self.n_gpu = torch.cuda.device_count()
        self.use_multi_gpu = self.args.multi_gpu and self.n_gpu > 1

        if self.use_multi_gpu:
            print(f"Using {self.n_gpu} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model)
        else:
            print(f"Using single device: {self.device}")

        # Load config if available
        config_path = os.path.join(self.args.checkpoint, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
            print(f"Loaded config: compression_k={self.config.get('compression_k', 250)}, "
                  f"num_passages={self.config.get('num_passages', 100)}")
            # Use config values as defaults
            if self.args.compression_k is None:
                self.args.compression_k = self.config.get("compression_k", 250)
            if self.args.num_passages is None:
                self.args.num_passages = self.config.get("num_passages", 100)
        else:
            self.config = {}
            if self.args.compression_k is None:
                self.args.compression_k = 250
            if self.args.num_passages is None:
                self.args.num_passages = 100

    def get_base_model(self):
        """Get the underlying model (handles DataParallel wrapper)."""
        if self.use_multi_gpu:
            return self.model.module
        return self.model

    def _get_encoder(self):
        """Get T5Gemma2 encoder."""
        base_model = self.get_base_model()
        if hasattr(base_model, 'encoder'):
            return base_model.encoder
        elif hasattr(base_model, 'get_encoder'):
            return base_model.get_encoder()
        else:
            raise AttributeError("Cannot find encoder in model")

    def _normalize_answer(self, s: str) -> str:
        """Normalize answer for exact match comparison."""
        # Remove articles
        s = re.sub(r'\b(a|an|the)\b', ' ', s.lower())
        # Remove punctuation
        s = ''.join(ch for ch in s if ch not in string.punctuation)
        # Normalize whitespace
        return ' '.join(s.split())

    def _compute_f1(self, prediction: str, ground_truth: str) -> float:
        """Compute token-level F1 score."""
        pred_tokens = self._normalize_answer(prediction).split()
        gold_tokens = self._normalize_answer(ground_truth).split()

        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            return int(pred_tokens == gold_tokens)

        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def encode_and_fuse(
        self,
        input_texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode and fuse passages (no compression for pure FiD).

        Args:
            input_texts: List of passage texts

        Returns:
            fused_hidden: [1, n_passages * seq_len, hidden_dim]
            fused_mask: [1, n_passages * seq_len]
        """
        encoder = self._get_encoder()
        n_passages = len(input_texts)
        k = self.args.compression_k

        # Tokenize
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            max_length=self.args.max_input_length,
            truncation=True,
            padding="max_length"
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        passage_length = input_ids.shape[1]

        # Encode with optional BF16
        with torch.no_grad():
            if self.args.bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if self.use_multi_gpu and self.n_gpu > 1:
                        encoder_wrapper = EncoderWrapper(encoder).to(self.device)
                        device_ids = list(range(self.n_gpu))
                        hidden_states = data_parallel(
                            encoder_wrapper,
                            (input_ids, attention_mask),
                            device_ids=device_ids
                        )
                    else:
                        encoder_output = encoder(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True
                        )
                        hidden_states = encoder_output.last_hidden_state
            else:
                if self.use_multi_gpu and self.n_gpu > 1:
                    encoder_wrapper = EncoderWrapper(encoder).to(self.device)
                    device_ids = list(range(self.n_gpu))
                    hidden_states = data_parallel(
                        encoder_wrapper,
                        (input_ids, attention_mask),
                        device_ids=device_ids
                    )
                else:
                    encoder_output = encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    hidden_states = encoder_output.last_hidden_state

        hidden_dim = hidden_states.shape[-1]

        # Reshape: [n_passages, seq_len, hidden] -> [1, n_passages, seq_len, hidden]
        hidden_states = hidden_states.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

        # Apply compression (k tokens per passage)
        actual_k = min(k, passage_length)
        compressed = hidden_states[:, :, :actual_k, :].contiguous()
        comp_mask = attention_mask[:, :, :actual_k]

        # Reshape for decoder: [1, n_passages * k, hidden_dim]
        fused_hidden = compressed.reshape(1, n_passages * actual_k, hidden_dim)
        fused_mask = comp_mask.reshape(1, n_passages * actual_k)

        return fused_hidden, fused_mask

    def generate(
        self,
        input_texts: List[str]
    ) -> str:
        """
        Generate answer from input passages.

        Args:
            input_texts: List of passage texts

        Returns:
            Generated answer string
        """
        # Encode and fuse
        fused_hidden, fused_mask = self.encode_and_fuse(input_texts)

        # Generate with V10 parameters to prevent repetition
        with torch.no_grad():
            outputs = self.get_base_model().generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=fused_hidden),
                attention_mask=fused_mask,
                max_new_tokens=self.args.max_output_length,
                num_beams=self.args.num_beams,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode - pure FiD outputs answer directly
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated

    def evaluate_task(
        self,
        task: str,
        samples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate on a single task.

        Args:
            task: Task name
            samples: List of evaluation samples

        Returns:
            Dict with EM, F1, and count metrics
        """
        correct = 0
        total = 0
        f1_scores = []

        if self.args.max_samples:
            samples = samples[:self.args.max_samples]

        pbar = tqdm(samples, desc=f"  {task}", leave=False)

        for sample in pbar:
            input_texts = sample["input_texts"]
            gold_answer = sample["answer"]

            # Generate
            pred_answer = self.generate(input_texts)

            # Compute metrics
            gold_norm = self._normalize_answer(gold_answer)
            pred_norm = self._normalize_answer(pred_answer)

            em = 1 if gold_norm == pred_norm else 0
            correct += em
            total += 1

            f1 = self._compute_f1(pred_answer, gold_answer)
            f1_scores.append(f1)

            # Verbose output
            if self.args.verbose:
                status = "O" if em else "X"
                print(f"\n{status} Sample {total}:")
                print(f"  Query: {sample['query'][:80]}...")
                print(f"  Gold:  {gold_answer}")
                print(f"  Pred:  {pred_answer}")
                print(f"  EM: {em}, F1: {f1:.4f}")

            pbar.set_postfix({
                "EM": f"{correct/total:.4f}" if total > 0 else "0",
                "F1": f"{sum(f1_scores)/len(f1_scores):.4f}" if f1_scores else "0"
            })

            # Report progress to web demo (every 10 samples)
            if total % 10 == 0:
                em_pct = correct / total * 100 if total > 0 else 0
                f1_pct = sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0
                report_evaluation_progress(total, len(samples), em_pct, f1_pct)

        metrics = {
            "em": correct / total if total > 0 else 0,
            "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
            "correct": correct,
            "total": total
        }

        return metrics

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """
        Run evaluation on all tasks.

        Returns:
            Dict of metrics per task
        """
        print(f"\n{'='*60}")
        print("Pure FiD T5Gemma2 Evaluation (Answer-Only, No Provenance)")
        print(f"{'='*60}")
        print(f"Checkpoint: {self.args.checkpoint}")
        print(f"Compression k: {self.args.compression_k}")
        print(f"Num beams: {self.args.num_beams}")
        print(f"Max samples per task: {self.args.max_samples or 'all'}")
        print(f"BF16: {self.args.bf16}")
        print()

        # Load validation data
        val_data = self._load_validation_data()

        if not val_data:
            print("No validation data found!")
            return {}

        results = {}

        for task, samples in val_data.items():
            print(f"\nEvaluating {task} ({len(samples)} samples)...")
            metrics = self.evaluate_task(task, samples)
            results[task] = metrics

            print(f"  {task}: EM={metrics['em']:.4f}, F1={metrics['f1']:.4f} "
                  f"({metrics['correct']}/{metrics['total']})")

        # Compute average
        if results:
            avg_em = sum(m["em"] for m in results.values()) / len(results)
            avg_f1 = sum(m["f1"] for m in results.values()) / len(results)
            total_correct = sum(m["correct"] for m in results.values())
            total_samples = sum(m["total"] for m in results.values())

            print(f"\n{'='*60}")
            print("Overall Results")
            print(f"{'='*60}")
            print(f"Average EM: {avg_em:.4f}")
            print(f"Average F1: {avg_f1:.4f}")
            print(f"Total: {total_correct}/{total_samples}")

            results["_average"] = {
                "em": avg_em,
                "f1": avg_f1,
                "correct": total_correct,
                "total": total_samples
            }

        # Save results
        self._save_results(results)

        return results

    def _load_validation_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load precomputed validation data from single parquet file."""
        val_data = {}
        path = self.args.val_path

        # If path is a directory, look for all_tasks_dev.parquet
        if os.path.isdir(path):
            data_path = os.path.join(path, "all_tasks_dev.parquet")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"all_tasks_dev.parquet not found in {path}")
        else:
            data_path = path

        # Load single parquet file and group by task
        print(f"Loading validation data from {data_path}...")
        df = pd.read_parquet(data_path, engine='fastparquet')
        for task in df['task'].unique():
            task_df = df[df['task'] == task]
            val_data[task] = task_df.to_dict('records')
            print(f"  Loaded {task}: {len(val_data[task])} samples")

        return val_data

    def _save_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """Save evaluation results."""
        # Save to checkpoint directory
        results_path = os.path.join(self.args.checkpoint, "eval_results_pure_fid_t5gemma.json")

        # Convert to serializable format
        serializable = {}
        for task, metrics in results.items():
            serializable[task] = {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in metrics.items()
            }

        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)

        print(f"\nResults saved to {results_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pure FiD T5Gemma2 Evaluation (No Source Pointer)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")

    # Validation data
    parser.add_argument("--val_path", type=str,
                        default="kilt_data/precomputed_fid/",
                        help="Path to precomputed validation data (directory with *_dev.parquet)")

    # Model settings (override config if provided)
    parser.add_argument("--compression_k", type=int, default=None,
                        help="Compression k (default: from config, or 250)")
    parser.add_argument("--num_passages", type=int, default=None,
                        help="Number of passages (default: from config, or 100)")

    # Input/Output
    parser.add_argument("--max_input_length", type=int, default=250,
                        help="Max input tokens per passage")
    parser.add_argument("--max_output_length", type=int, default=64,
                        help="Max output tokens")

    # Decoding
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Beam size for generation (paper: 1 = greedy)")

    # Evaluation
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per task (None = all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output per sample")

    # Device
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda/cpu)")
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Use all available GPUs")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use BF16 precision (default: True for T5Gemma2)")
    parser.add_argument("--no_bf16", action="store_true",
                        help="Disable BF16, use FP32")

    args = parser.parse_args()

    # Handle bf16 flag
    if args.no_bf16:
        args.bf16 = False

    return args


def main():
    """Main entry point."""
    args = parse_args()

    evaluator = FiDPureT5GemmaEvaluator(args)
    results = evaluator.evaluate()


if __name__ == "__main__":
    main()
