"""
FiD-Light T5Gemma2 Evaluation Script
=====================================

Evaluates trained FiD-Light model with T5Gemma2 backbone using KILT Score metric.

KILT Score = Answer Accuracy × Provenance Accuracy

Where:
- Answer Accuracy: Exact match between predicted and gold answer
- Provenance Accuracy: Whether predicted source pointer points to gold provenance passage

Usage:
    python evaluate_fidlight_t5gemma.py --checkpoint checkpoints/fidlight_t5gemma_270m/final --task nq

Papers:
- FiD-Light: Hofstatter et al. (2023)
- T5Gemma2: Zhang et al. (2025)
"""

import argparse
import os
import re
import json
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoProcessor
from transformers.modeling_outputs import BaseModelOutput
from torch.nn.parallel import data_parallel

# Web Demo state reporting (optional - only used when running from web UI)
try:
    from web_demo.utils.state_io import update_step_state, StepStatus
    HAS_WEB_DEMO = True
except ImportError:
    HAS_WEB_DEMO = False


def report_evaluation_progress(current: int, total: int, em: float, kilt: float):
    """Report evaluation progress to web demo (if available)."""
    if not HAS_WEB_DEMO:
        return
    try:
        progress = (current / total) * 100 if total > 0 else 0
        message = f"Evaluating {current}/{total} | EM: {em:.1f}% | KILT: {kilt:.1f}%"
        update_step_state(
            step_name="evaluate",
            progress=progress,
            message=message,
            status=StepStatus.RUNNING.value,
            extra={
                "current": current,
                "total": total,
                "exact_match": em,
                "kilt_score": kilt,
                "algorithm": "fidlight",
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
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state


class FiDLightT5GemmaEvaluator:
    """FiD-Light model evaluator with T5Gemma2 backbone and KILT Score computation."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        compression_k: int = 64,
        num_beams: int = 4,
        max_output_length: int = 64,
        repetition_penalty: float = 1.2,
        multi_gpu: bool = False,
    ):
        """
        Initialize the evaluator.

        Args:
            checkpoint_path: Path to model checkpoint
            device: 'cuda' or 'cpu'
            compression_k: Compression factor (first k tokens per passage)
            num_beams: Beam search width
            max_output_length: Max tokens to generate
            repetition_penalty: Penalty for repeated tokens (1.0 = no penalty)
            multi_gpu: Use all available GPUs for encoding
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.k = compression_k
        self.num_beams = num_beams
        self.max_output_length = max_output_length
        self.repetition_penalty = repetition_penalty
        self.multi_gpu = multi_gpu
        self.n_gpu = torch.cuda.device_count() if multi_gpu else 1
        self.num_passages = None  # Will be set via set_num_passages()

        print(f"Loading T5Gemma2 model from {checkpoint_path}...")

        # Use AutoProcessor from original model (checkpoint doesn't save processor files)
        base_model_name = "google/t5gemma-2-270m-270m"
        self.processor = AutoProcessor.from_pretrained(base_model_name)
        self.tokenizer = self.processor.tokenizer

        # Load model weights from checkpoint
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
        )

        # T5Gemma2 fix: Set decoder_start_token_id if not set
        if self.model.config.decoder_start_token_id is None:
            self.model.config.decoder_start_token_id = self.model.config.bos_token_id or 2
            print(f"  Set decoder_start_token_id to {self.model.config.decoder_start_token_id}")

        # ===== V9: Simple format like T5-Base =====
        print(f"  V9: Simple format (no chat template)")
        print(f"  decoder_start_token_id: {self.model.config.decoder_start_token_id}")

        self.model.to(device)
        self.model.eval()

        # Get encoder reference
        self.encoder = self._get_encoder()

        if multi_gpu and self.n_gpu > 1:
            print(f"Using {self.n_gpu} GPUs for encoding")
        print(f"Model loaded on {device}")

    def set_num_passages(self, num_passages: Optional[int]):
        """Set number of passages to use (truncates input_texts if needed)."""
        self.num_passages = num_passages
        if num_passages:
            print(f"Using {num_passages} passages per query (truncating if needed)")

    def _get_encoder(self):
        """Get encoder from model (handles different model architectures)."""
        # Try direct attribute access first (T5Gemma2 style)
        if hasattr(self.model, 'encoder'):
            return self.model.encoder
        # Try get_encoder method (T5 style)
        elif hasattr(self.model, 'get_encoder'):
            return self.model.get_encoder()
        # For wrapped models, try to access base model
        elif hasattr(self.model, 'model'):
            base = self.model.model
            if hasattr(base, 'encoder'):
                return base.encoder
            elif hasattr(base, 'get_encoder'):
                return base.get_encoder()
        raise AttributeError("Cannot find encoder in model")

    def generate(
        self,
        input_texts: List[str],
        max_input_length: int = 384,
    ) -> str:
        """
        Generate answer for a single sample.

        Args:
            input_texts: List of formatted passage strings
            max_input_length: Max tokens per passage

        Returns:
            Generated text (e.g., "index: 3 text: Paris")
        """
        # Truncate to num_passages if set (for models trained with fewer passages)
        if self.num_passages and len(input_texts) > self.num_passages:
            input_texts = input_texts[:self.num_passages]
        n_passages = len(input_texts)

        # Tokenize all passages
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Encode (multi-GPU if available)
        with torch.no_grad():
            if self.multi_gpu and self.n_gpu > 1:
                encoder_wrapper = EncoderWrapper(self.encoder).to(self.device)
                device_ids = list(range(self.n_gpu))
                last_hidden_state = data_parallel(
                    encoder_wrapper,
                    (input_ids, attention_mask),
                    device_ids=device_ids
                )
            else:
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                last_hidden_state = encoder_outputs.last_hidden_state
            total_sequences, seq_len, hidden_dim = last_hidden_state.shape

            # Compress
            actual_k = min(self.k, seq_len)
            compressed_states = last_hidden_state[:, :actual_k, :]
            compressed_mask = attention_mask[:, :actual_k]

            # Fuse (reshape for batch_size=1)
            fused_hidden_states = compressed_states.reshape(1, n_passages * actual_k, hidden_dim)
            fused_attention_mask = compressed_mask.reshape(1, n_passages * actual_k)

            # ===== V9: Simple generation like T5-Base =====
            outputs = self.model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=fused_hidden_states),
                attention_mask=fused_attention_mask,
                max_new_tokens=self.max_output_length,
                num_beams=self.num_beams,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode output
        full_output = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()

        return full_output

    def extract_first_answer(self, generated_text: str) -> str:
        """
        Extract the first valid 'index: X text: Y' answer from generated text.

        This handles cases where the model repeats the pattern multiple times.

        Args:
            generated_text: Raw generated text (may contain multiple index:...text:... patterns)

        Returns:
            First valid answer in format "index: X text: Y"
        """
        import re
        text = generated_text.strip()

        # Find the first complete "index: X text: Y" pattern
        pattern = r'(index:\s*[\d,\s]+\s*text:\s*[^i]*?)(?=index:|$)'
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return match.group(1).strip()

        # If no match, try to extract just the first text: content
        if "text:" in text:
            parts = text.split("text:", 1)
            if len(parts) > 1:
                answer_part = parts[1].split("index:")[0].strip()
                idx_match = re.search(r'index:\s*([\d,\s]+)', parts[0])
                if idx_match:
                    return f"index: {idx_match.group(1).strip()} text: {answer_part}"
                return f"index: 1 text: {answer_part}"

        # Fallback: return as-is
        return text

    def parse_output(self, output: str) -> Tuple[List[int], str]:
        """
        Parse model output to extract indices and answer.

        Expected format: "index: 1,3,5 text: Paris"

        Returns:
            (list of indices, answer text)
        """
        indices = []
        answer = ""

        # Extract indices (accept 1-100 range, actual validation done in check_provenance)
        index_match = re.search(r"index:\s*([0-9,\s]+)", output)
        if index_match:
            indices_str = index_match.group(1)
            for idx_str in indices_str.split(","):
                try:
                    idx = int(idx_str.strip())
                    if 1 <= idx <= 100:  # Loose filter, check_provenance validates against actual passage count
                        indices.append(idx)
                except ValueError:
                    continue

        # Extract answer text (stop at next "index:" if present to handle repetition)
        text_match = re.search(r"text:\s*(.+?)(?=\s*index:|$)", output, re.DOTALL | re.IGNORECASE)
        if text_match:
            answer = text_match.group(1).strip()

        return indices, answer

    def normalize_answer(self, s: str) -> str:
        """Normalize answer for comparison."""
        import string

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            return ''.join(ch for ch in text if ch not in string.punctuation)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def exact_match(self, prediction: str, gold: str) -> bool:
        """Check if normalized prediction matches gold answer."""
        return self.normalize_answer(prediction) == self.normalize_answer(gold)

    def compute_f1(self, prediction: str, gold: str) -> float:
        """Compute token-level F1 score."""
        pred_tokens = self.normalize_answer(prediction).split()
        gold_tokens = self.normalize_answer(gold).split()

        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            return int(pred_tokens == gold_tokens)

        common = set(pred_tokens) & set(gold_tokens)
        num_common = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def check_provenance(
        self,
        predicted_indices: List[int],
        retrieved_wiki_ids: List[str],
        gold_provenance_ids: List[str],
    ) -> bool:
        """
        Check if any predicted index points to a gold provenance passage.

        Args:
            predicted_indices: 1-based indices from model output
            retrieved_wiki_ids: List of wikipedia_ids for the 40 passages
            gold_provenance_ids: Gold provenance wikipedia_ids

        Returns:
            True if at least one predicted passage is in gold provenance
        """
        if not predicted_indices or not gold_provenance_ids:
            return False

        gold_set = set(gold_provenance_ids)

        for idx in predicted_indices:
            if 1 <= idx <= len(retrieved_wiki_ids):
                wiki_id = retrieved_wiki_ids[idx - 1]  # Convert to 0-based
                if wiki_id in gold_set:
                    return True

        return False


def load_validation_data(data_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load precomputed validation data from Parquet file."""
    print(f"Loading validation data from {data_path}...")
    table = pq.read_table(data_path)

    samples = []
    for i in range(len(table)):
        sample = {
            "id": table["id"][i].as_py(),
            "task": table["task"][i].as_py(),
            "query": table["query"][i].as_py(),
            "answer": table["answer"][i].as_py(),
            "retrieved_wiki_ids": table["retrieved_wiki_ids"][i].as_py(),
            "input_texts": table["input_texts"][i].as_py(),
            "gold_provenance_ids": table["gold_provenance_ids"][i].as_py(),
        }
        samples.append(sample)

        if max_samples and len(samples) >= max_samples:
            break

    print(f"Loaded {len(samples)} samples")
    return samples


def evaluate(
    evaluator: FiDLightT5GemmaEvaluator,
    samples: List[Dict],
    verbose: bool = False,
    extract_first_answer: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate model on samples and compute metrics.

    Args:
        evaluator: FiDLightT5GemmaEvaluator instance
        samples: List of validation samples
        verbose: Print sample-level results
        extract_first_answer: V8 option to extract only first answer from repeated outputs

    Returns:
        Dictionary of metrics
    """
    results = {
        "total": 0,
        "exact_match": 0,
        "f1_sum": 0.0,
        "provenance_correct": 0,
        "kilt_score": 0,  # Samples where both answer AND provenance are correct
        "per_task": defaultdict(lambda: {
            "total": 0,
            "exact_match": 0,
            "f1_sum": 0.0,
            "provenance_correct": 0,
            "kilt_score": 0,
        }),
    }

    for sample in tqdm(samples, desc="Evaluating"):
        task = sample["task"]

        # Generate prediction
        output = evaluator.generate(sample["input_texts"])

        # V8: Optionally extract only the first answer
        if extract_first_answer:
            output = evaluator.extract_first_answer(output)

        predicted_indices, predicted_answer = evaluator.parse_output(output)

        # Compute metrics
        gold_answer = sample["answer"]
        em = evaluator.exact_match(predicted_answer, gold_answer)
        f1 = evaluator.compute_f1(predicted_answer, gold_answer)

        prov_correct = evaluator.check_provenance(
            predicted_indices,
            sample["retrieved_wiki_ids"],
            sample["gold_provenance_ids"],
        )

        # KILT Score: both answer and provenance must be correct
        kilt = 1 if (em and prov_correct) else 0

        # Update totals
        results["total"] += 1
        results["exact_match"] += int(em)
        results["f1_sum"] += f1
        results["provenance_correct"] += int(prov_correct)
        results["kilt_score"] += kilt

        # Update per-task
        results["per_task"][task]["total"] += 1
        results["per_task"][task]["exact_match"] += int(em)
        results["per_task"][task]["f1_sum"] += f1
        results["per_task"][task]["provenance_correct"] += int(prov_correct)
        results["per_task"][task]["kilt_score"] += kilt

        # Report progress to web demo (every 10 samples)
        if results["total"] % 10 == 0:
            n = results["total"]
            em_pct = results["exact_match"] / n * 100 if n > 0 else 0
            kilt_pct = results["kilt_score"] / n * 100 if n > 0 else 0
            report_evaluation_progress(n, len(samples), em_pct, kilt_pct)

        if verbose:
            print(f"\n[{sample['id']}] Task: {task}")
            print(f"  Query: {sample['query'][:80]}...")
            print(f"  Gold: {gold_answer}")
            print(f"  Pred: {predicted_answer}")
            print(f"  Indices: {predicted_indices}")
            print(f"  EM: {em}, F1: {f1:.3f}, Prov: {prov_correct}, KILT: {kilt}")

    # Compute final metrics
    n = results["total"]
    metrics = {
        "total_samples": n,
        "backbone": "t5gemma2-270m-270m",
        "answer_accuracy": results["exact_match"] / n * 100 if n > 0 else 0,
        "answer_f1": results["f1_sum"] / n * 100 if n > 0 else 0,
        "provenance_accuracy": results["provenance_correct"] / n * 100 if n > 0 else 0,
        "kilt_score": results["kilt_score"] / n * 100 if n > 0 else 0,
        "per_task": {},
    }

    for task, task_results in results["per_task"].items():
        n_task = task_results["total"]
        metrics["per_task"][task] = {
            "total_samples": n_task,
            "answer_accuracy": task_results["exact_match"] / n_task * 100 if n_task > 0 else 0,
            "answer_f1": task_results["f1_sum"] / n_task * 100 if n_task > 0 else 0,
            "provenance_accuracy": task_results["provenance_correct"] / n_task * 100 if n_task > 0 else 0,
            "kilt_score": task_results["kilt_score"] / n_task * 100 if n_task > 0 else 0,
        }

    return metrics


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Print evaluation metrics in a formatted table."""
    print("\n" + "=" * 70)
    print("FiD-Light T5Gemma2 Evaluation Results")
    print("=" * 70)

    print(f"\nBackbone: {metrics.get('backbone', 'unknown')}")
    print(f"\nOverall Metrics ({metrics['total_samples']} samples):")
    print("-" * 40)
    print(f"  Answer Accuracy (EM):   {metrics['answer_accuracy']:.2f}%")
    print(f"  Answer F1:              {metrics['answer_f1']:.2f}%")
    print(f"  Provenance Accuracy:    {metrics['provenance_accuracy']:.2f}%")
    print(f"  KILT Score:             {metrics['kilt_score']:.2f}%")

    if metrics["per_task"]:
        print("\nPer-Task Metrics:")
        print("-" * 70)
        print(f"{'Task':<20} {'Samples':>8} {'EM':>8} {'F1':>8} {'Prov':>8} {'KILT':>8}")
        print("-" * 70)

        for task, task_metrics in sorted(metrics["per_task"].items()):
            print(f"{task:<20} {task_metrics['total_samples']:>8} "
                  f"{task_metrics['answer_accuracy']:>7.2f}% "
                  f"{task_metrics['answer_f1']:>7.2f}% "
                  f"{task_metrics['provenance_accuracy']:>7.2f}% "
                  f"{task_metrics['kilt_score']:>7.2f}%")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FiD-Light T5Gemma2 model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/fidlight_t5gemma_270m/final",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="kilt_data/precomputed_v5/all_tasks_dev.parquet",
        help="Path to precomputed validation data",
    )
    parser.add_argument(
        "--num_passages",
        type=int,
        default=None,
        help="Limit number of passages per query (default: use all from data). "
             "Set to 20 if model was trained with --num_passages 20",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Specific task to evaluate (default: all tasks)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples to evaluate (for testing)",
    )
    parser.add_argument(
        "--compression_k",
        type=int,
        default=64,
        help="Compression factor",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Beam search width",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="Repetition penalty for generation (1.0 = no penalty)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print sample-level results",
    )
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Use all available GPUs for encoding",
    )
    parser.add_argument(
        "--extract_first_answer",
        action="store_true",
        help="V8: Extract only the first answer from generated output (handles repetition)",
    )

    args = parser.parse_args()

    # Load data
    if args.task:
        data_path = f"kilt_data/precomputed_v5/{args.task}_dev.parquet"
    else:
        data_path = args.data_path

    samples = load_validation_data(data_path, args.max_samples)

    # Initialize evaluator
    evaluator = FiDLightT5GemmaEvaluator(
        checkpoint_path=args.checkpoint,
        compression_k=args.compression_k,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        multi_gpu=args.multi_gpu,
    )

    # Set num_passages if specified (for models trained with fewer passages)
    if args.num_passages:
        evaluator.set_num_passages(args.num_passages)

    # Evaluate
    metrics = evaluate(
        evaluator, samples,
        verbose=args.verbose,
        extract_first_answer=args.extract_first_answer
    )

    # Print results
    print_metrics(metrics)

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
