"""
GTR Retriever Training Data Generator
======================================

Generates training triples for GTR retriever fine-tuning on KILT tasks.

Paper: "FiD-Light: Efficient and Effective Retrieval-Augmented Text Generation"
       Hofstatter et al. (2023) - Appendix B

Training data format (per paper):
- query: The input query from KILT sample
- positive: Known relevant passage from provenance
- negative: Randomly sampled from GTR zero-shot top-100 (excluding positives)

Usage:
    python generate_retrieval_training_data.py \
        --output_path kilt_data/retrieval_training_data.jsonl \
        --tasks all

    # Quick test with limited samples
    python generate_retrieval_training_data.py \
        --output_path kilt_data/retrieval_training_data_test.jsonl \
        --tasks nq \
        --max_samples_per_task 1000
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import os
import random
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict

from tqdm import tqdm

from utils.kilt_loader import load_filtered_kilt_task, KILTWikipediaArrow
from utils.gtr_retriever import GTRRetriever
from utils.multitask_loader import extract_provenance_ids, extract_answer


# KILT tasks for retriever training (3 QA tasks only)
ALL_TASKS = [
    "nq",
    "hotpotqa",
    "triviaqa_support_only",
]


def format_passage_text(article: Dict[str, Any], max_paragraphs: int = 5, max_chars: int = 2000) -> str:
    """
    Format a Wikipedia article for retriever training.

    Matches KILT official format: "{title} {text}" (no labels).

    Args:
        article: Wikipedia article dict
        max_paragraphs: Max paragraphs to include
        max_chars: Max total characters

    Returns:
        Formatted text: "{title} {paragraphs}"
    """
    title = article.get("wikipedia_title", "")
    paragraphs = article.get("text", [])

    if isinstance(paragraphs, list):
        text = " ".join(paragraphs[:max_paragraphs])
    else:
        text = str(paragraphs)

    # No labels, just title + text (KILT official format)
    formatted = f"{title} {text}"
    if len(formatted) > max_chars:
        formatted = formatted[:max_chars]

    return formatted


def generate_training_triples(
    task_name: str,
    retriever: GTRRetriever,
    wiki: KILTWikipediaArrow,
    max_samples: Optional[int] = None,
    top_k_negatives: int = 100,
    batch_size: int = 256,
    seed: int = 42,
    filtered_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate training triples for a single KILT task.

    Per paper Appendix B:
    "We created passage retrieval training triples containing a query,
    a known relevant passage, and a sampled negative passage
    (randomly sampled from the top-100 GTR zero-shot rankings for the query)."

    Args:
        task_name: KILT task name
        retriever: GTRRetriever (zero-shot model for negative mining)
        wiki: KILTWikipediaArrow for article lookup
        max_samples: Max samples to process (None = all)
        top_k_negatives: Retrieve top-k for negative sampling (paper: 100)
        batch_size: Batch size for retrieval
        seed: Random seed

    Returns:
        List of training triples
    """
    random.seed(seed)

    # Load task data
    try:
        samples = load_filtered_kilt_task(task_name, split="train", filtered_dir=filtered_dir)
    except FileNotFoundError as e:
        print(f"  Skipping {task_name}: {e}")
        return []

    if max_samples:
        samples = samples[:max_samples]

    print(f"  Processing {task_name}: {len(samples)} samples")

    # Pre-extract valid samples with provenance
    valid_samples = []
    for sample in samples:
        query = sample.get("input", "")
        provenance_ids = extract_provenance_ids(sample)

        if query and provenance_ids:
            valid_samples.append({
                "sample": sample,
                "query": query,
                "provenance_ids": provenance_ids,
            })

    print(f"    Valid samples with provenance: {len(valid_samples)}")

    if not valid_samples:
        return []

    # Process in batches
    triples = []
    stats = {
        "total": len(valid_samples),
        "success": 0,
        "no_positive": 0,
        "no_negative": 0,
    }

    num_batches = (len(valid_samples) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"    {task_name}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(valid_samples))

        batch_samples = valid_samples[start_idx:end_idx]
        batch_queries = [s["query"] for s in batch_samples]

        # Batch retrieval for negative mining
        batch_retrieved = retriever.batch_retrieve(
            batch_queries,
            top_k=top_k_negatives,
            return_text=False,  # Only need IDs for negative sampling
            batch_size=batch_size
        )

        # Process each sample in batch
        for sample_info, retrieved in zip(batch_samples, batch_retrieved):
            query = sample_info["query"]
            provenance_ids = sample_info["provenance_ids"]

            # Get positive passage from provenance
            positive_text = None
            positive_id = None
            for prov_id in provenance_ids:
                article = wiki.get_by_id(prov_id)
                if article:
                    positive_text = format_passage_text(article)
                    positive_id = prov_id
                    break

            if not positive_text:
                stats["no_positive"] += 1
                continue

            # Sample negative from top-100 (excluding positives)
            negative_candidates = []
            for r in retrieved:
                wiki_id = r.get("wikipedia_id", "")
                if wiki_id and wiki_id not in provenance_ids:
                    negative_candidates.append(wiki_id)

            if not negative_candidates:
                stats["no_negative"] += 1
                continue

            # Random sample one negative
            negative_id = random.choice(negative_candidates)
            negative_article = wiki.get_by_id(negative_id)

            if not negative_article:
                stats["no_negative"] += 1
                continue

            negative_text = format_passage_text(negative_article)

            # Create triple
            triples.append({
                "query": query,
                "positive": positive_text,
                "negative": negative_text,
                "task": task_name,
                "positive_id": positive_id,
                "negative_id": negative_id,
            })
            stats["success"] += 1

    print(f"    Generated {stats['success']} triples "
          f"(no_positive: {stats['no_positive']}, no_negative: {stats['no_negative']})")

    return triples


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for GTR retriever fine-tuning"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="kilt_data/retrieval_training_data.jsonl",
        help="Output path for training triples (jsonl format)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated task names or 'all'"
    )
    parser.add_argument(
        "--max_samples_per_task",
        type=int,
        default=None,
        help="Max samples per task (for testing)"
    )
    parser.add_argument(
        "--top_k_negatives",
        type=int,
        default=100,
        help="Top-k for negative sampling (paper: 100)"
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="kilt_data/gtr_faiss_index_full",
        help="Path to GTR Faiss index (should be zero-shot index)"
    )
    parser.add_argument(
        "--wiki_arrow_path",
        type=str,
        default=None,
        help="Path to Wikipedia Arrow dataset"
    )
    parser.add_argument(
        "--filtered_dir",
        type=str,
        default=None,
        help="Path to filtered KILT data directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for retrieval"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--use_gpu_index",
        action="store_true",
        help="Force GPU Faiss index (disable mmap, slower startup but much faster search)"
    )

    args = parser.parse_args()

    # Parse tasks
    if args.tasks.lower() == "all":
        tasks = ALL_TASKS
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]

    print("=" * 60)
    print("GTR Retriever Training Data Generation")
    print("=" * 60)
    print(f"Tasks: {tasks}")
    print(f"Output: {args.output_path}")
    print(f"Top-k negatives: {args.top_k_negatives}")
    if args.max_samples_per_task:
        print(f"Max samples per task: {args.max_samples_per_task}")
    print()

    # Load retriever (zero-shot for negative mining)
    print("Loading GTR retriever (zero-shot)...")
    # use_mmap=False enables GPU Faiss index (slower startup, much faster search)
    use_mmap = False if args.use_gpu_index else None  # None = auto-detect
    retriever = GTRRetriever(index_path=args.index_path, use_mmap=use_mmap)

    # Load Wikipedia
    print("Loading Wikipedia Arrow dataset...")
    wiki = KILTWikipediaArrow(arrow_path=args.wiki_arrow_path)
    print()

    # Store filtered_dir for later use
    filtered_dir = args.filtered_dir

    # Generate triples for each task
    all_triples = []
    task_counts = {}

    for task in tasks:
        print(f"Processing task: {task}")

        triples = generate_training_triples(
            task_name=task,
            retriever=retriever,
            wiki=wiki,
            max_samples=args.max_samples_per_task,
            top_k_negatives=args.top_k_negatives,
            batch_size=args.batch_size,
            seed=args.seed,
            filtered_dir=filtered_dir,
        )

        task_counts[task] = len(triples)
        all_triples.extend(triples)
        print()

    # Save to jsonl
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print(f"Saving {len(all_triples)} triples to {args.output_path}...")
    with open(args.output_path, "w") as f:
        for triple in all_triples:
            f.write(json.dumps(triple, ensure_ascii=False) + "\n")

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total triples: {len(all_triples)}")
    print()
    print("Per-task counts:")
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        print(f"  {task}: {count:,}")
    print()
    print(f"Output saved to: {args.output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
