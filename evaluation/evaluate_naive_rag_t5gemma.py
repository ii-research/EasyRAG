"""
Naive RAG Baseline Evaluation Script with T5Gemma2
==================================================

Evaluates a T5Gemma2 model trained with simple RAG: concatenate all passages into one context.

- Naive RAG: Retrieve passages, concatenate them, use standard encoder-decoder
- Metrics: EM (Exact Match), F1 (no KILT-score since no source pointer)

Input format:
    question: What is the capital of France?
    context: [1] France is a country in Western Europe. Its capital is Paris...
    [2] Paris is the largest city in France...

Output format:
    Paris

Usage:
    # Evaluate on validation set
    python evaluate_naive_rag_t5gemma.py --checkpoint checkpoints/naive_rag_t5gemma/step_10000 \
        --data_path kilt_data/precomputed_v5/all_tasks_dev.parquet

    # Quick test
    python evaluate_naive_rag_t5gemma.py --checkpoint checkpoints/naive_rag_t5gemma/step_10000 \
        --data_path kilt_data/precomputed_v5/all_tasks_dev.parquet --max_samples 50
"""

import argparse
import json
import os
import re
import string
from typing import Dict, List, Any, Optional
from collections import Counter

import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

from transformers import AutoModelForSeq2SeqLM, AutoProcessor

# Web Demo state reporting (optional)
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
                "algorithm": "naive_rag",
                "model": "t5gemma"
            }
        )
    except Exception:
        pass


class NaiveRAGT5GemmaEvaluator:
    """
    Naive RAG evaluator with T5Gemma2 (concatenate passages, standard encoder-decoder).

    Metrics:
    - Exact Match (EM): Normalized string comparison
    - F1: Token-level F1 score
    """

    def __init__(self, args: argparse.Namespace):
        """Initialize evaluator."""
        self.args = args
        self.device = torch.device(args.device)
        self.results = {}

        self._load_model()

    def _load_model(self) -> None:
        """Load T5Gemma2 model from checkpoint."""
        print(f"Loading T5Gemma2 model from {self.args.checkpoint}...")

        # Try to load processor from checkpoint
        try:
            self.processor = AutoProcessor.from_pretrained(self.args.checkpoint)
            self.tokenizer = self.processor.tokenizer
        except Exception:
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

        # Fix decoder_start_token_id
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
            print(f"Loaded config: algorithm={self.config.get('algorithm', 'naive_rag')}, "
                  f"num_passages={self.config.get('num_passages', 10)}")
            if self.args.num_passages is None:
                self.args.num_passages = self.config.get("num_passages", 10)
        else:
            self.config = {}
            if self.args.num_passages is None:
                self.args.num_passages = 10

    def get_base_model(self):
        """Get the underlying model (handles DataParallel wrapper)."""
        if self.use_multi_gpu:
            return self.model.module
        return self.model

    def _normalize_answer(self, s: str) -> str:
        """Normalize answer for exact match comparison."""
        s = re.sub(r'\b(a|an|the)\b', ' ', s.lower())
        s = ''.join(ch for ch in s if ch not in string.punctuation)
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

    def _extract_context_from_input_text(self, input_text: str) -> str:
        """Extract context part from FiD-format input text."""
        match = re.search(r'context:\s*(.+)$', input_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return input_text.split(':', 1)[-1].strip() if ':' in input_text else input_text

    def _format_naive_rag_input(self, query: str, input_texts: List[str]) -> str:
        """Format input for Naive RAG: concatenate passages into single context."""
        passages = input_texts[:self.args.num_passages]

        contexts = []
        for i, text in enumerate(passages):
            context = self._extract_context_from_input_text(text)
            contexts.append(f"[{i+1}] {context}")

        context_str = "\n".join(contexts)
        return f"question: {query}\ncontext: {context_str}"

    def generate(self, query: str, input_texts: List[str]) -> str:
        """Generate answer from question + concatenated passages."""
        formatted_input = self._format_naive_rag_input(query, input_texts)

        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            max_length=self.args.max_input_length,
            truncation=True
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.get_base_model().generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.args.max_output_length,
                num_beams=self.args.num_beams,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Filter out <unused...> tokens that T5Gemma sometimes generates
        import re
        generated = re.sub(r'<unused\d+>', '', generated).strip()
        return generated

    def evaluate_task(
        self,
        task: str,
        samples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate on a single task."""
        correct = 0
        total = 0
        f1_scores = []

        if self.args.max_samples:
            samples = samples[:self.args.max_samples]

        pbar = tqdm(samples, desc=f"  {task}", leave=False)

        for sample in pbar:
            query = sample["query"]
            input_texts = sample["input_texts"]
            gold_answer = sample["answer"]

            # Generate with concatenated passages
            pred_answer = self.generate(query, input_texts)

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
                print(f"  Query: {query[:80]}...")
                print(f"  Gold:  {gold_answer}")
                print(f"  Pred:  {pred_answer}")
                print(f"  EM: {em}, F1: {f1:.4f}")

            pbar.set_postfix({
                "EM": f"{correct/total:.4f}" if total > 0 else "0",
                "F1": f"{sum(f1_scores)/len(f1_scores):.4f}" if f1_scores else "0"
            })

            # Report progress
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
        """Run evaluation on all tasks."""
        print(f"\n{'='*60}")
        print("Naive RAG T5Gemma2 Baseline Evaluation (Concatenated Passages)")
        print(f"{'='*60}")
        print(f"Checkpoint: {self.args.checkpoint}")
        print(f"Num passages: {self.args.num_passages}")
        print(f"Max input length: {self.args.max_input_length}")
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
            # Filter by task if specified
            if self.args.task and task != self.args.task:
                continue

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
        """Load validation data from parquet file."""
        val_data = {}
        path = self.args.data_path

        if os.path.isdir(path):
            data_path = os.path.join(path, "all_tasks_dev.parquet")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"all_tasks_dev.parquet not found in {path}")
        else:
            data_path = path

        print(f"Loading validation data from {data_path}...")
        df = pd.read_parquet(data_path, engine='fastparquet')
        for task in df['task'].unique():
            task_df = df[df['task'] == task]
            val_data[task] = task_df.to_dict('records')
            print(f"  Loaded {task}: {len(val_data[task])} samples")

        return val_data

    def _save_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """Save evaluation results."""
        if self.args.output:
            results_path = self.args.output
        else:
            results_path = os.path.join(self.args.checkpoint, "eval_results_naive_rag_t5gemma.json")

        os.makedirs(os.path.dirname(results_path) if os.path.dirname(results_path) else ".", exist_ok=True)

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
        description="Naive RAG T5Gemma2 Baseline Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")

    # Validation data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to validation data")
    parser.add_argument("--task", type=str, default=None,
                        help="Evaluate single task only")

    # RAG settings
    parser.add_argument("--num_passages", type=int, default=None,
                        help="Number of passages (default: from config, or 10)")
    parser.add_argument("--max_input_length", type=int, default=512,
                        help="Max input tokens")
    parser.add_argument("--max_output_length", type=int, default=64,
                        help="Max output tokens")

    # Decoding
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Beam size for generation")

    # Evaluation
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per task")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results JSON")

    # Device
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device")
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Use all available GPUs")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use BF16 (default: True)")
    parser.add_argument("--no_bf16", action="store_true",
                        help="Disable BF16")

    args = parser.parse_args()

    if args.no_bf16:
        args.bf16 = False

    return args


def main():
    """Main entry point."""
    args = parse_args()

    evaluator = NaiveRAGT5GemmaEvaluator(args)
    results = evaluator.evaluate()


if __name__ == "__main__":
    main()
