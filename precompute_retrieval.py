"""
FiD-Light Retrieval Precomputation Script
==========================================

Precomputes retrieval results for all KILT training samples to eliminate
real-time retrieval overhead during training.

Key features:
1. GTR-T5-Base retrieval for top-40 passages per query
2. Gold passage injection when retrieval misses provenance (Table 1 Row 3)
3. Complete input_texts and target_text generation
4. Arrow format output for efficient training data loading

Paper: "FiD-Light: Efficient and Effective Retrieval-Augmented Text Generation"
       Hofstatter et al. (2023)

Usage:
    # Single task
    python precompute_retrieval.py --tasks nq --output_dir kilt_data/precomputed/

    # All tasks
    python precompute_retrieval.py --tasks all --output_dir kilt_data/precomputed/

    # Quick test (100 samples)
    python precompute_retrieval.py --tasks nq --max_samples 100 --output_dir kilt_data/precomputed/
"""

import argparse
import os
import json
import random
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from kilt_loader import load_filtered_kilt_task, KILTWikipediaArrow
from gtr_retriever import GTRRetriever
from multitask_loader import extract_provenance_ids, extract_answer


# KILT tasks for FiD-Light training (3 QA tasks only)
ALL_TASKS = [
    "nq",
    "hotpotqa",
    "triviaqa_support_only",
]


def format_passage_text(article: Dict[str, Any], max_paragraphs: int = 5, max_chars: int = 1500) -> str:
    """
    Format a Wikipedia article for FiD-Light input.

    Args:
        article: Wikipedia article dict
        max_paragraphs: Max paragraphs to include
        max_chars: Max characters

    Returns:
        Formatted text: "{title} {paragraphs}"
    """
    title = article.get("wikipedia_title", "")
    paragraphs = article.get("text", [])

    if isinstance(paragraphs, list):
        text = " ".join(paragraphs[:max_paragraphs])
    else:
        text = str(paragraphs)

    combined = f"{title} {text}"
    if len(combined) > max_chars:
        combined = combined[:max_chars]

    return combined


def precompute_task(
    task_name: str,
    retriever: GTRRetriever,
    wiki: KILTWikipediaArrow,
    num_passages: int = 40,
    max_chars_per_passage: int = 1500,
    inject_gold: bool = True,
    max_samples: Optional[int] = None,
    verbose: bool = True,
    batch_size: int = 256,
    split: str = "train",
    filtered_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Precompute retrieval results for a single KILT task using BATCH retrieval.

    Args:
        task_name: KILT task name (e.g., "nq", "fever")
        retriever: GTRRetriever instance
        split: Data split ("train" or "validation")
        wiki: KILTWikipediaArrow instance for gold passage lookup
        num_passages: Number of passages to retrieve per query
        max_chars_per_passage: Max characters per passage
        inject_gold: Whether to inject gold passages when retrieval misses
        max_samples: Max samples to process (None = all)
        verbose: Print progress
        batch_size: Batch size for retrieval (GPU encoding)
        filtered_dir: Path to filtered KILT data directory

    Returns:
        List of precomputed samples
    """
    # Load task data
    try:
        samples = load_filtered_kilt_task(task_name, split=split, filtered_dir=filtered_dir)
    except FileNotFoundError as e:
        print(f"  Skipping {task_name}: {e}")
        return []

    if max_samples:
        samples = samples[:max_samples]

    if verbose:
        print(f"  Processing {task_name}: {len(samples)} samples (batch_size={batch_size})")

    # Pre-extract all queries, answers, and provenance
    valid_samples = []
    queries = []
    for sample in samples:
        query = sample.get("input", "")
        answer = extract_answer(sample)
        if query and answer:
            valid_samples.append({
                "sample": sample,
                "query": query,
                "answer": answer,
                "provenance_ids": extract_provenance_ids(sample),
                "sample_id": sample.get("id", f"{task_name}_{len(valid_samples)}")
            })
            queries.append(query)

    if verbose:
        print(f"    Valid samples: {len(valid_samples)} (skipped {len(samples) - len(valid_samples)} without answer)")

    results = []
    stats = {
        "total": len(valid_samples),
        "with_provenance": 0,
        "gold_injected": 0,
        "retrieval_hit": 0,
    }

    # Process in batches
    num_batches = (len(queries) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"  {task_name}", disable=not verbose):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(queries))

        batch_queries = queries[start_idx:end_idx]
        batch_samples = valid_samples[start_idx:end_idx]

        # BATCH RETRIEVAL - GPU accelerated!
        batch_retrieved = retriever.batch_retrieve(
            batch_queries,
            top_k=num_passages,
            return_text=True,
            batch_size=batch_size
        )

        # Process each result in the batch
        for i, (sample_info, retrieved) in enumerate(zip(batch_samples, batch_retrieved)):
            query = sample_info["query"]
            answer = sample_info["answer"]
            sample_id = sample_info["sample_id"]
            provenance_ids = sample_info["provenance_ids"]

            if provenance_ids:
                stats["with_provenance"] += 1

            # Check which retrieved passages match provenance
            matching_indices = []
            for j, passage in enumerate(retrieved):
                wiki_id = passage.get("wikipedia_id", "")
                if wiki_id in provenance_ids:
                    matching_indices.append(j + 1)  # 1-based

            if matching_indices:
                stats["retrieval_hit"] += 1

            # Inject gold passages if needed (Table 1 Row 3)
            gold_injected = False
            original_recall = len(matching_indices)

            if inject_gold and not matching_indices and provenance_ids:
                gold_injected = True
                stats["gold_injected"] += 1

                # Random position injection (1-40) to avoid position bias
                used_positions = set()
                for gold_id in list(provenance_ids)[:3]:
                    article = wiki.get_by_id(gold_id)
                    if article:
                        # Random position, avoiding already used positions
                        available = [i for i in range(num_passages) if i not in used_positions]
                        if not available:
                            break
                        replace_idx = random.choice(available)
                        used_positions.add(replace_idx)

                        gold_passage = {
                            "wikipedia_id": gold_id,
                            "title": article.get("wikipedia_title", ""),
                            "text": format_passage_text(article, max_chars=max_chars_per_passage),
                            "score": 0.0,
                            "rank": replace_idx + 1,
                        }
                        if replace_idx < len(retrieved):
                            retrieved[replace_idx] = gold_passage
                            matching_indices.append(replace_idx + 1)

            # Generate input_texts
            input_texts = []
            retrieved_wiki_ids = []

            for j, passage in enumerate(retrieved):
                title = passage.get("title", "")
                text = passage.get("text", "")
                wiki_id = passage.get("wikipedia_id", "")

                context = f"{title} {text}" if title else text
                if len(context) > max_chars_per_passage:
                    context = context[:max_chars_per_passage]

                formatted = f"query: {query} index: {j+1} context: {context}"
                input_texts.append(formatted)
                retrieved_wiki_ids.append(wiki_id)

            # Generate target
            sorted_indices = sorted(matching_indices)[:3]
            indices_str = ",".join(str(idx) for idx in sorted_indices) if sorted_indices else "1"
            target_text = f"index: {indices_str} text: {answer}"

            results.append({
                "id": sample_id,
                "task": task_name,
                "query": query,
                "answer": answer,
                "retrieved_wiki_ids": retrieved_wiki_ids,
                "matching_indices": sorted_indices,
                "input_texts": input_texts,
                "target_text": target_text,
                "gold_injected": gold_injected,
                "original_recall": original_recall,
                "gold_provenance_ids": list(provenance_ids),  # For evaluation
            })

    if verbose:
        hit_rate = stats["retrieval_hit"] / stats["with_provenance"] * 100 if stats["with_provenance"] > 0 else 0
        print(f"    Total: {stats['total']}, With provenance: {stats['with_provenance']}, "
              f"Retrieval hit: {stats['retrieval_hit']} ({hit_rate:.1f}%), "
              f"Gold injected: {stats['gold_injected']}")

    return results


def save_to_arrow(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save precomputed results to Arrow/Parquet format.

    Args:
        results: List of precomputed samples
        output_path: Output file path (.parquet)
    """
    if not results:
        print(f"  No results to save for {output_path}")
        return

    # Convert to columnar format
    columns = {
        "id": [],
        "task": [],
        "query": [],
        "answer": [],
        "retrieved_wiki_ids": [],
        "matching_indices": [],
        "input_texts": [],
        "target_text": [],
        "gold_injected": [],
        "original_recall": [],
        "gold_provenance_ids": [],
    }

    for r in results:
        for key in columns:
            columns[key].append(r[key])

    # Create Arrow table
    table = pa.table({
        "id": pa.array(columns["id"]),
        "task": pa.array(columns["task"]),
        "query": pa.array(columns["query"]),
        "answer": pa.array(columns["answer"]),
        "retrieved_wiki_ids": pa.array(columns["retrieved_wiki_ids"]),
        "matching_indices": pa.array(columns["matching_indices"]),
        "input_texts": pa.array(columns["input_texts"]),
        "target_text": pa.array(columns["target_text"]),
        "gold_injected": pa.array(columns["gold_injected"]),
        "original_recall": pa.array(columns["original_recall"]),
        "gold_provenance_ids": pa.array(columns["gold_provenance_ids"]),
    })

    # Save as Parquet
    pq.write_table(table, output_path)
    print(f"  Saved {len(results)} samples to {output_path}")


def load_precomputed(path: str) -> pa.Table:
    """Load precomputed data from Parquet file."""
    return pq.read_table(path)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute retrieval results for FiD-Light training"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="nq",
        help="Comma-separated task names or 'all' for all tasks"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="kilt_data/precomputed",
        help="Output directory for precomputed data"
    )
    parser.add_argument(
        "--num_passages",
        type=int,
        default=40,
        help="Number of passages to retrieve per query"
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=1500,
        help="Max characters per passage"
    )
    parser.add_argument(
        "--inject_gold",
        action="store_true",
        default=True,
        help="Inject gold passages when retrieval misses (Table 1 Row 3)"
    )
    parser.add_argument(
        "--no_inject_gold",
        action="store_true",
        help="Disable gold passage injection"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples per task (for testing)"
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="kilt_data/gtr_faiss_index",
        help="Path to GTR Faiss index"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned GTR model (default: gtr-t5-base)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for GPU retrieval (default 256)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="all",
        help="Data splits to precompute: 'all' (train+validation), 'train', 'validation', or comma-separated"
    )
    parser.add_argument(
        "--use_multi_gpu",
        action="store_true",
        help="Use all available GPUs for encoding (recommended)"
    )
    parser.add_argument(
        "--preload_wiki",
        action="store_true",
        help="Preload all Wikipedia articles to memory (~10-15GB) to eliminate IO bottleneck"
    )
    parser.add_argument(
        "--use_mmap",
        action="store_true",
        help="Use mmap for Faiss index (fast load, but slower search). Default is full load + GPU for batch processing."
    )
    parser.add_argument(
        "--wiki_arrow_path",
        type=str,
        default=None,
        help="Path to Wikipedia Arrow dataset (default: kilt_data/kilt_wikipedia_arrow)"
    )
    parser.add_argument(
        "--filtered_dir",
        type=str,
        default=None,
        help="Path to filtered KILT data directory (default: kilt_data/filtered)"
    )

    args = parser.parse_args()

    # Parse tasks
    if args.tasks.lower() == "all":
        tasks = ALL_TASKS
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]

    inject_gold_base = args.inject_gold and not args.no_inject_gold

    # Parse splits
    if args.splits.lower() == "all":
        splits = ["train", "validation"]
    else:
        splits = [s.strip() for s in args.splits.split(",")]

    print("=" * 60)
    print("FiD-Light Retrieval Precomputation")
    print("=" * 60)
    print(f"Tasks: {tasks}")
    print(f"Splits: {splits}")
    print(f"Passages per query: {args.num_passages}")
    print(f"Inject gold passages (train only): {inject_gold_base}")
    print(f"Batch size: {args.batch_size}")
    print(f"Index path: {args.index_path}")
    print(f"Model path: {args.model_path or 'gtr-t5-base (default)'}")
    print(f"Multi-GPU: {args.use_multi_gpu}")
    print(f"Preload Wiki: {args.preload_wiki}")
    print(f"Use mmap: {args.use_mmap} (default: False for GPU index)")
    print(f"Wiki Arrow path: {args.wiki_arrow_path or 'default'}")
    print(f"Filtered dir: {args.filtered_dir or 'default'}")
    print(f"Output directory: {args.output_dir}")
    if args.max_samples:
        print(f"Max samples per task: {args.max_samples}")
    print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load retriever
    # Default: full load + GPU index for batch processing (use_mmap=False)
    # Use --use_mmap for fast startup but slower search
    print("Loading GTR retriever...")
    retriever = GTRRetriever(
        index_path=args.index_path,
        model_path=args.model_path,
        use_multi_gpu=args.use_multi_gpu,
        preload_wiki=args.preload_wiki,
        use_mmap=args.use_mmap,  # Default False for GPU index
        wiki_arrow_path=args.wiki_arrow_path,
    )

    # Use retriever's wiki (already preloaded if --preload_wiki was set)
    wiki = retriever.wiki
    if wiki is None:
        print("Loading Wikipedia Arrow dataset...")
        wiki = KILTWikipediaArrow(arrow_path=args.wiki_arrow_path)
    print()

    # Process each split and task
    total_samples = 0

    for split in splits:
        print(f"\n{'=' * 60}")
        print(f"Processing split: {split.upper()}")
        print(f"{'=' * 60}")

        # For validation, never inject gold passages (we test retrieval quality)
        inject_gold = inject_gold_base and (split != "validation")
        if split == "validation":
            print("  (Gold injection disabled for validation)")

        all_results = []

        for task in tasks:
            print(f"Processing task: {task}")

            results = precompute_task(
                task_name=task,
                retriever=retriever,
                wiki=wiki,
                num_passages=args.num_passages,
                max_chars_per_passage=args.max_chars,
                inject_gold=inject_gold,
                max_samples=args.max_samples,
                batch_size=args.batch_size,
                split=split,
                filtered_dir=args.filtered_dir,
            )

            if results:
                # Save per-task file (use split name in filename)
                split_suffix = "dev" if split == "validation" else "train"
                task_path = os.path.join(args.output_dir, f"{task}_{split_suffix}.parquet")
                save_to_arrow(results, task_path)
                all_results.extend(results)

            print()

        # Save combined file for this split
        if all_results:
            split_suffix = "dev" if split == "validation" else "train"
            combined_path = os.path.join(args.output_dir, f"all_tasks_{split_suffix}.parquet")
            save_to_arrow(all_results, combined_path)
            total_samples += len(all_results)

    # Cleanup
    retriever.close()

    print("=" * 60)
    print("Precomputation complete!")
    print(f"Total samples: {total_samples}")
    print(f"Splits processed: {splits}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
