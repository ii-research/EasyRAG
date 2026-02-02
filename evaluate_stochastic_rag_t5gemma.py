"""
Stochastic RAG Evaluation Script with T5Gemma2 Backbone
========================================================

Evaluates trained Stochastic RAG model using KILT Score metric.

Key differences from evaluate_stochastic_rag.py:
- Uses AutoModelForSeq2SeqLM and AutoTokenizer (T5Gemma2)
- T5Gemma2 has different hidden_size (1024 for 270M vs T5-base's 768)
- Native BF16 support (recommended for T5Gemma2)
- Requires transformers>=5.0.0rc1

Key difference from FiD-Light: Uses learned reranker to select passages
before generation, rather than just using retrieval order.

KILT Score = Answer Accuracy × Provenance Accuracy

Usage:
    python evaluate_stochastic_rag_t5gemma.py --checkpoint checkpoints/stochastic_rag_t5gemma/final --task nq --bf16

Paper: "Stochastic RAG: End-to-End Retrieval-Augmented Generation through Expected Utility Maximization"
       Zamani & Bendersky (SIGIR 2024)
Backbone: T5Gemma2 - Zhang et al. (2025)
"""

import argparse
import os
import re
import json
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
                "algorithm": "stochastic_rag",
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


class Reranker(nn.Module):
    """
    Passage Reranker for Stochastic RAG.

    Scores each passage based on its compressed representation.
    """
    def __init__(self, hidden_dim: int = 1024, scoring_type: str = "linear"):
        super().__init__()
        self.scoring_type = scoring_type

        if scoring_type == "linear":
            self.scorer = nn.Linear(hidden_dim, 1)
        elif scoring_type == "mlp":
            self.scorer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        else:
            raise ValueError(f"Unknown scoring_type: {scoring_type}")

    def forward(self, passage_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Score passages.

        Args:
            passage_embeddings: [n_passages, hidden_dim] or [batch, n_passages, hidden_dim]

        Returns:
            scores: [n_passages] or [batch, n_passages]
        """
        scores = self.scorer(passage_embeddings).squeeze(-1)
        return scores


class StochasticRAGEvaluator:
    """Stochastic RAG model evaluator with KILT Score computation (T5Gemma2)."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        compression_k: int = 64,
        num_passages: int = 10,
        num_beams: int = 4,
        max_output_length: int = 64,
        multi_gpu: bool = False,
        bf16: bool = False,
    ):
        """
        Initialize the evaluator.

        Args:
            checkpoint_path: Path to model checkpoint
            device: 'cuda' or 'cpu'
            compression_k: Compression factor (first k tokens per passage)
            num_passages: Number of passages to select for generation
            num_beams: Beam search width
            max_output_length: Max tokens to generate
            multi_gpu: Use all available GPUs for encoding
            bf16: Use BF16 precision (recommended for T5Gemma2)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.k = compression_k
        self.num_passages = num_passages
        self.num_beams = num_beams
        self.max_output_length = max_output_length
        self.multi_gpu = multi_gpu
        self.n_gpu = torch.cuda.device_count() if multi_gpu else 1
        self.bf16 = bf16

        # Load generator (using Auto* classes for T5Gemma2)
        print(f"Loading generator from {checkpoint_path}...")
        print(f"  (Requires transformers>=5.0.0rc1 for T5Gemma2)")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        if bf16:
            print(f"  Loading with BF16 precision (recommended for T5Gemma2)")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16
            )
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

        self.model.to(device)
        self.model.eval()

        # Load reranker
        reranker_path = os.path.join(checkpoint_path, "reranker.pt")
        if os.path.exists(reranker_path):
            print(f"Loading reranker from {reranker_path}...")
            reranker_state = torch.load(reranker_path, map_location=device)

            # Debug: print state dict keys
            print(f"  Reranker state_dict keys: {list(reranker_state.keys())}")

            # T5Gemma2 may use different attribute names for hidden dimension
            config = self.model.config
            hidden_dim = getattr(config, 'd_model', None) or \
                         getattr(config, 'hidden_size', None) or \
                         getattr(config, 'encoder_hidden_size', None)

            if hidden_dim is None:
                # Detect via test forward pass
                print(f"  Detecting encoder hidden_dim via test forward pass...")
                encoder = self._get_encoder()
                with torch.no_grad():
                    test_input = torch.ones(1, 10, dtype=torch.long, device=device)
                    test_output = encoder(input_ids=test_input)
                    hidden_dim = test_output.last_hidden_state.shape[-1]

            print(f"  Model hidden_dim: {hidden_dim}")

            self.reranker = Reranker(hidden_dim=hidden_dim)
            self.reranker.load_state_dict(reranker_state)
            self.reranker.to(device)
            self.reranker.eval()
            self.use_reranker = True

            # Debug: verify reranker weights
            for name, param in self.reranker.named_parameters():
                print(f"  {name}: shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")

            print(f"Reranker loaded successfully (hidden_dim={hidden_dim})")
        else:
            print(f"Warning: No reranker found at {reranker_path}")
            print("Will use retrieval order (same as FiD-Light)")
            self.reranker = None
            self.use_reranker = False

        if multi_gpu and self.n_gpu > 1:
            print(f"Using {self.n_gpu} GPUs for encoding")
        print(f"Model loaded on {device}")

    def _get_encoder(self):
        """Get the encoder from T5Gemma2 model."""
        if hasattr(self.model, 'encoder'):
            return self.model.encoder
        elif hasattr(self.model, 'get_encoder'):
            return self.model.get_encoder()
        else:
            raise AttributeError("Cannot access encoder from model")

    def generate(
        self,
        input_texts: List[str],
        max_input_length: int = 384,
    ) -> Tuple[str, List[int]]:
        """
        Generate answer for a single sample.

        Args:
            input_texts: List of 40 formatted passage strings
            max_input_length: Max tokens per passage

        Returns:
            (generated_text, selected_indices) - indices are 1-based
        """
        n_candidates = len(input_texts)

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

        # BF16 autocast context
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.bf16 else torch.nullcontext()

        with torch.no_grad(), autocast_ctx:
            # Encode (multi-GPU if available)
            encoder = self._get_encoder()
            if self.multi_gpu and self.n_gpu > 1:
                encoder_wrapper = EncoderWrapper(encoder).to(self.device)
                device_ids = list(range(self.n_gpu))
                last_hidden_state = data_parallel(
                    encoder_wrapper,
                    (input_ids, attention_mask),
                    device_ids=device_ids
                )
            else:
                encoder_outputs = encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                last_hidden_state = encoder_outputs.last_hidden_state

            # Compress
            actual_k = min(self.k, last_hidden_state.shape[1])
            compressed_states = last_hidden_state[:, :actual_k, :]  # [n_candidates, k, hidden]

            # Select passages using reranker or retrieval order
            if self.use_reranker:
                # Use first token (CLS-like) for scoring - same as training!
                passage_embeddings = last_hidden_state[:, 0, :]  # [n_candidates, hidden]
                scores = self.reranker(passage_embeddings)  # [n_candidates]

                # Store debug info for verbose output
                self._last_score_stats = {
                    "min": scores.min().item(),
                    "max": scores.max().item(),
                    "std": scores.std().item(),
                    "mean": scores.mean().item(),
                    "top5": scores[:5].tolist(),
                    "all_scores": scores.tolist(),
                }

                # Select top-k passages
                k = min(self.num_passages, n_candidates)
                _, selected_idx = torch.topk(scores, k)

                # Debug: raw selection before sorting
                raw_selected = (selected_idx + 1).tolist()

                selected_idx = selected_idx.sort().values  # Keep order

                # Store raw selection too
                self._last_score_stats["raw_topk_indices"] = raw_selected
                selected_indices = (selected_idx + 1).tolist()  # 1-based
            else:
                # Use retrieval order (top passages)
                self._last_score_stats = None
                k = min(self.num_passages, n_candidates)
                selected_idx = torch.arange(k, device=self.device)
                selected_indices = list(range(1, k + 1))

            # Get selected compressed states
            selected_states = compressed_states[selected_idx]  # [k, actual_k, hidden]
            selected_mask = attention_mask[selected_idx, :actual_k]

            # Fuse
            fused_hidden_states = selected_states.reshape(1, -1, selected_states.shape[-1])
            fused_attention_mask = selected_mask.reshape(1, -1)

            # Generate
            outputs = self.model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=fused_hidden_states),
                attention_mask=fused_attention_mask,
                max_length=self.max_output_length,
                num_beams=self.num_beams,
                early_stopping=True,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text, selected_indices

    def parse_output(self, output: str) -> Tuple[List[int], str]:
        """
        Parse model output to extract indices and answer.

        Expected format: "index: 1,3,5 text: Paris"

        Returns:
            (list of indices, answer text)
        """
        indices = []
        answer = ""

        # Extract indices
        index_match = re.search(r"index:\s*([0-9,\s]+)", output)
        if index_match:
            indices_str = index_match.group(1)
            for idx_str in indices_str.split(","):
                try:
                    idx = int(idx_str.strip())
                    if 1 <= idx <= 40:
                        indices.append(idx)
                except ValueError:
                    continue

        # Extract answer text
        text_match = re.search(r"text:\s*(.+)", output, re.DOTALL)
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

    def check_selected_provenance(
        self,
        selected_indices: List[int],
        retrieved_wiki_ids: List[str],
        gold_provenance_ids: List[str],
    ) -> bool:
        """
        Check if any reranker-selected passage is a gold provenance passage.

        This measures the reranker's ability to select relevant documents.

        Args:
            selected_indices: 1-based indices selected by reranker
            retrieved_wiki_ids: List of wikipedia_ids for the 40 passages
            gold_provenance_ids: Gold provenance wikipedia_ids

        Returns:
            True if at least one selected passage is in gold provenance
        """
        if not selected_indices or not gold_provenance_ids:
            return False

        gold_set = set(gold_provenance_ids)

        for idx in selected_indices:
            if 1 <= idx <= len(retrieved_wiki_ids):
                wiki_id = retrieved_wiki_ids[idx - 1]
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
    evaluator: StochasticRAGEvaluator,
    samples: List[Dict],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate model on samples and compute metrics.

    Args:
        evaluator: StochasticRAGEvaluator instance
        samples: List of validation samples
        verbose: Print sample-level results

    Returns:
        Dictionary of metrics
    """
    results = {
        "total": 0,
        "exact_match": 0,
        "f1_sum": 0.0,
        "provenance_correct": 0,  # From model output (source pointer)
        "selected_provenance_correct": 0,  # From reranker selection
        "kilt_score": 0,
        "per_task": defaultdict(lambda: {
            "total": 0,
            "exact_match": 0,
            "f1_sum": 0.0,
            "provenance_correct": 0,
            "selected_provenance_correct": 0,
            "kilt_score": 0,
        }),
    }

    for sample in tqdm(samples, desc="Evaluating"):
        task = sample["task"]

        # Generate prediction (with reranker selection)
        output, selected_indices = evaluator.generate(sample["input_texts"])
        predicted_indices, predicted_answer = evaluator.parse_output(output)

        # Compute metrics
        gold_answer = sample["answer"]
        em = evaluator.exact_match(predicted_answer, gold_answer)
        f1 = evaluator.compute_f1(predicted_answer, gold_answer)

        # Provenance from model output (source pointer)
        prov_correct = evaluator.check_provenance(
            predicted_indices,
            sample["retrieved_wiki_ids"],
            sample["gold_provenance_ids"],
        )

        # Provenance from reranker selection
        selected_prov_correct = evaluator.check_selected_provenance(
            selected_indices,
            sample["retrieved_wiki_ids"],
            sample["gold_provenance_ids"],
        )

        # KILT Score: both answer and provenance must be correct
        # For Stochastic RAG, use reranker's selection as provenance (per original paper definition)
        kilt = 1 if (em and selected_prov_correct) else 0

        # Update totals
        results["total"] += 1
        results["exact_match"] += int(em)
        results["f1_sum"] += f1
        results["provenance_correct"] += int(prov_correct)
        results["selected_provenance_correct"] += int(selected_prov_correct)
        results["kilt_score"] += kilt

        # Update per-task
        results["per_task"][task]["total"] += 1
        results["per_task"][task]["exact_match"] += int(em)
        results["per_task"][task]["f1_sum"] += f1
        results["per_task"][task]["provenance_correct"] += int(prov_correct)
        results["per_task"][task]["selected_provenance_correct"] += int(selected_prov_correct)
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
            # Show gold provenance indices (which of the 40 passages contain gold provenance)
            gold_prov_indices = []
            gold_set = set(sample.get("gold_provenance_ids", []))
            for i, wiki_id in enumerate(sample.get("retrieved_wiki_ids", []), 1):
                if wiki_id in gold_set:
                    gold_prov_indices.append(i)
            print(f"  Gold provenance indices: {gold_prov_indices if gold_prov_indices else 'None in top-40'}")
            print(f"  Selected by reranker: {selected_indices}")
            # Print reranker score stats if available
            if hasattr(evaluator, '_last_score_stats') and evaluator._last_score_stats is not None:
                stats = evaluator._last_score_stats
                print(f"  Reranker scores: min={stats['min']:.4f}, max={stats['max']:.4f}, std={stats['std']:.4f}, mean={stats['mean']:.4f}")
                print(f"  First 5 scores: {[f'{s:.4f}' for s in stats['top5']]}")
                print(f"  Raw top-k selection (before sort): {stats.get('raw_topk_indices', 'N/A')}")
                # Show all 40 scores if std is very low
                if stats['std'] < 0.01:
                    print(f"  WARNING: std is very low! All scores: {[f'{s:.4f}' for s in stats.get('all_scores', [])]}")
            else:
                print(f"  (No reranker stats - use_reranker={evaluator.use_reranker if hasattr(evaluator, 'use_reranker') else 'unknown'})")
            print(f"  Source pointer: {predicted_indices}")
            print(f"  EM: {em}, F1: {f1:.3f}, Prov: {prov_correct}, SelProv: {selected_prov_correct}, KILT: {kilt}")

    # Compute final metrics
    n = results["total"]
    metrics = {
        "total_samples": n,
        "answer_accuracy": results["exact_match"] / n * 100 if n > 0 else 0,
        "answer_f1": results["f1_sum"] / n * 100 if n > 0 else 0,
        "provenance_accuracy": results["provenance_correct"] / n * 100 if n > 0 else 0,
        "selected_provenance_accuracy": results["selected_provenance_correct"] / n * 100 if n > 0 else 0,
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
            "selected_provenance_accuracy": task_results["selected_provenance_correct"] / n_task * 100 if n_task > 0 else 0,
            "kilt_score": task_results["kilt_score"] / n_task * 100 if n_task > 0 else 0,
        }

    return metrics


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Print evaluation metrics in a formatted table."""
    print("\n" + "=" * 80)
    print("Stochastic RAG (T5Gemma2) Evaluation Results")
    print("=" * 80)

    print(f"\nOverall Metrics ({metrics['total_samples']} samples):")
    print("-" * 50)
    print(f"  Answer Accuracy (EM):        {metrics['answer_accuracy']:.2f}%")
    print(f"  Answer F1:                   {metrics['answer_f1']:.2f}%")
    print(f"  Provenance Accuracy:         {metrics['provenance_accuracy']:.2f}%")
    print(f"  Reranker Selection Accuracy: {metrics['selected_provenance_accuracy']:.2f}%")
    print(f"  KILT Score:                  {metrics['kilt_score']:.2f}%")

    if metrics["per_task"]:
        print("\nPer-Task Metrics:")
        print("-" * 80)
        print(f"{'Task':<20} {'N':>6} {'EM':>8} {'F1':>8} {'Prov':>8} {'SelProv':>8} {'KILT':>8}")
        print("-" * 80)

        for task, task_metrics in sorted(metrics["per_task"].items()):
            print(f"{task:<20} {task_metrics['total_samples']:>6} "
                  f"{task_metrics['answer_accuracy']:>7.2f}% "
                  f"{task_metrics['answer_f1']:>7.2f}% "
                  f"{task_metrics['provenance_accuracy']:>7.2f}% "
                  f"{task_metrics['selected_provenance_accuracy']:>7.2f}% "
                  f"{task_metrics['kilt_score']:>7.2f}%")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stochastic RAG model (T5Gemma2)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/stochastic_rag_t5gemma/final",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="kilt_data/precomputed/all_tasks_dev.parquet",
        help="Path to precomputed validation data",
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
        "--num_passages",
        type=int,
        default=10,
        help="Number of passages to select for generation",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Beam search width",
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
        "--bf16",
        action="store_true",
        help="Use BF16 precision (recommended for T5Gemma2)",
    )

    args = parser.parse_args()

    # Load data
    if args.task:
        # Support both directory and specific file
        if os.path.isdir(args.data_path):
            data_path = os.path.join(args.data_path, f"{args.task}_dev.parquet")
        else:
            data_path = f"kilt_data/precomputed/{args.task}_dev.parquet"
    else:
        data_path = args.data_path

    samples = load_validation_data(data_path, args.max_samples)

    # Initialize evaluator
    evaluator = StochasticRAGEvaluator(
        checkpoint_path=args.checkpoint,
        compression_k=args.compression_k,
        num_passages=args.num_passages,
        num_beams=args.num_beams,
        multi_gpu=args.multi_gpu,
        bf16=args.bf16,
    )

    # Evaluate
    metrics = evaluate(evaluator, samples, verbose=args.verbose)

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
