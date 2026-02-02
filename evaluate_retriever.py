"""
GTR Retriever Evaluation Script
================================

Evaluates retriever performance using Recall@K and computes ΔT.

Paper: "FiD-Light: Efficient and Effective Retrieval-Augmented Text Generation"
       Hofstatter et al. (2023) - Appendix B, Table 5

Metrics:
- Recall@40: Fraction of queries where at least one gold passage is in top-40
- ΔT: Relative difference between train and dev recall (measures overfitting)
      ΔT = (R@40_train - R@40_dev) / R@40_train

Usage:
    # Evaluate fine-tuned model
    python evaluate_retriever.py \
        --model_path checkpoints/gtr_kilt_finetuned \
        --index_path kilt_data/gtr_faiss_index_finetuned \
        --tasks all

    # Compare with zero-shot baseline
    python evaluate_retriever.py \
        --model_path sentence-transformers/gtr-t5-base \
        --index_path kilt_data/gtr_faiss_index_full \
        --tasks nq
"""

import argparse
import json
import os
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from kilt_loader import load_filtered_kilt_task
from gtr_retriever import GTRRetriever
from multitask_loader import extract_provenance_ids


# KILT tasks (3 QA tasks only)
ALL_TASKS = [
    "nq",
    "hotpotqa",
    "triviaqa_support_only",
]


def compute_recall_at_k(
    retriever: GTRRetriever,
    samples: List[Dict],
    k: int = 40,
    batch_size: int = 256,
) -> float:
    """
    Compute Recall@K for a set of samples.

    Recall@K = fraction of queries where at least one gold passage is in top-K

    Args:
        retriever: GTRRetriever instance
        samples: List of KILT samples with provenance
        k: Cutoff for recall
        batch_size: Batch size for retrieval

    Returns:
        Recall@K value (0.0 to 1.0)
    """
    # Filter samples with valid provenance
    valid_samples = []
    for sample in samples:
        query = sample.get("input", "")
        provenance_ids = extract_provenance_ids(sample)
        if query and provenance_ids:
            valid_samples.append({
                "query": query,
                "provenance_ids": provenance_ids,
            })

    if not valid_samples:
        return 0.0

    # Process in batches
    hits = 0
    num_batches = (len(valid_samples) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(valid_samples))

        batch_samples = valid_samples[start_idx:end_idx]
        batch_queries = [s["query"] for s in batch_samples]

        # Batch retrieval
        batch_retrieved = retriever.batch_retrieve(
            batch_queries,
            top_k=k,
            return_text=False,
            batch_size=batch_size
        )

        # Check hits
        for sample_info, retrieved in zip(batch_samples, batch_retrieved):
            provenance_ids = sample_info["provenance_ids"]
            retrieved_ids = {r.get("wikipedia_id", "") for r in retrieved}

            # Hit if any gold passage is in retrieved
            if provenance_ids & retrieved_ids:
                hits += 1

    recall = hits / len(valid_samples)
    return recall


def evaluate_task(
    task_name: str,
    retriever: GTRRetriever,
    k: int = 40,
    max_samples: Optional[int] = None,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """
    Evaluate retriever on a single KILT task.

    Returns train and dev recall for ΔT computation.

    Args:
        task_name: KILT task name
        retriever: GTRRetriever instance
        k: Cutoff for recall
        max_samples: Max samples per split (for testing)
        batch_size: Batch size for retrieval

    Returns:
        Dict with train_recall, dev_recall, delta_t
    """
    results = {
        "task": task_name,
        "train_recall": None,
        "dev_recall": None,
        "delta_t": None,
        "train_samples": 0,
        "dev_samples": 0,
    }

    # Evaluate on train set
    try:
        train_samples = load_filtered_kilt_task(task_name, split="train")
        if max_samples:
            train_samples = train_samples[:max_samples]
        results["train_samples"] = len(train_samples)

        print(f"    Evaluating train ({len(train_samples)} samples)...")
        results["train_recall"] = compute_recall_at_k(
            retriever, train_samples, k=k, batch_size=batch_size
        )
    except FileNotFoundError:
        print(f"    Train data not found for {task_name}")

    # Evaluate on dev set
    try:
        dev_samples = load_filtered_kilt_task(task_name, split="validation")
        if max_samples:
            dev_samples = dev_samples[:max_samples]
        results["dev_samples"] = len(dev_samples)

        print(f"    Evaluating dev ({len(dev_samples)} samples)...")
        results["dev_recall"] = compute_recall_at_k(
            retriever, dev_samples, k=k, batch_size=batch_size
        )
    except FileNotFoundError:
        print(f"    Dev data not found for {task_name}")

    # Compute ΔT: (train - dev) / train
    # Negative ΔT means dev is higher than train (unusual)
    # Small absolute ΔT is good (low overfitting)
    if results["train_recall"] and results["dev_recall"] and results["train_recall"] > 0:
        results["delta_t"] = (results["train_recall"] - results["dev_recall"]) / results["train_recall"]

    return results


def print_results_table(results: List[Dict[str, Any]], k: int = 40) -> None:
    """Print results in Table 5 format."""
    print()
    print("=" * 80)
    print(f"Retriever Evaluation Results (Recall@{k})")
    print("=" * 80)
    print()
    print(f"{'Task':<20} {'Train R@' + str(k):>12} {'Dev R@' + str(k):>12} {'ΔT':>10}")
    print("-" * 60)

    for r in results:
        train_str = f"{r['train_recall']:.2%}" if r['train_recall'] is not None else "N/A"
        dev_str = f"{r['dev_recall']:.2%}" if r['dev_recall'] is not None else "N/A"
        delta_str = f"{r['delta_t']:.1%}" if r['delta_t'] is not None else "N/A"

        print(f"{r['task']:<20} {train_str:>12} {dev_str:>12} {delta_str:>10}")

    print("-" * 60)

    # Compute averages
    train_recalls = [r['train_recall'] for r in results if r['train_recall'] is not None]
    dev_recalls = [r['dev_recall'] for r in results if r['dev_recall'] is not None]
    delta_ts = [r['delta_t'] for r in results if r['delta_t'] is not None]

    if train_recalls:
        avg_train = np.mean(train_recalls)
        avg_dev = np.mean(dev_recalls) if dev_recalls else None
        avg_delta = np.mean(delta_ts) if delta_ts else None

        avg_train_str = f"{avg_train:.2%}"
        avg_dev_str = f"{avg_dev:.2%}" if avg_dev else "N/A"
        avg_delta_str = f"{avg_delta:.1%}" if avg_delta else "N/A"

        print(f"{'AVERAGE':<20} {avg_train_str:>12} {avg_dev_str:>12} {avg_delta_str:>10}")

    print("=" * 80)
    print()
    print("ΔT interpretation:")
    print("  - Negative ΔT: Dev recall > Train recall (unusual)")
    print("  - Small |ΔT|: Low overfitting (good)")
    print("  - Large positive ΔT: Overfitting to train set (bad)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GTR retriever on KILT tasks"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned model (None = use index's original model)"
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="kilt_data/gtr_faiss_index_full",
        help="Path to Faiss index"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated task names or 'all'"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=40,
        help="Cutoff for Recall@K (paper: 40)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples per split (for testing)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for retrieval"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Parse tasks
    if args.tasks.lower() == "all":
        tasks = ALL_TASKS
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]

    print("=" * 60)
    print("GTR Retriever Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path or 'Default (from index)'}")
    print(f"Index: {args.index_path}")
    print(f"Tasks: {tasks}")
    print(f"Recall@{args.k}")
    if args.max_samples:
        print(f"Max samples per split: {args.max_samples}")
    print()

    # Load retriever
    # Note: For evaluation, we need to use the same model that created the index
    # If a different model is specified, we'd need to re-encode queries with it
    print("Loading retriever...")
    retriever = GTRRetriever(index_path=args.index_path)

    # If a custom model path is specified, load that model for query encoding
    if args.model_path and args.model_path != "sentence-transformers/gtr-t5-base":
        print(f"Loading custom model from {args.model_path}...")
        from sentence_transformers import SentenceTransformer
        retriever.model = SentenceTransformer(args.model_path, device=retriever.device)
        print("Custom model loaded for query encoding")

    print()

    # Evaluate each task
    all_results = []

    for task in tasks:
        print(f"Evaluating: {task}")

        results = evaluate_task(
            task_name=task,
            retriever=retriever,
            k=args.k,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
        )

        all_results.append(results)
        print()

    # Print results table
    print_results_table(all_results, k=args.k)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "model_path": args.model_path,
                "index_path": args.index_path,
                "k": args.k,
                "results": all_results,
            }, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
