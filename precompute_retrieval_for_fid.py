"""
FiD (Fusion-in-Decoder) Retrieval Precomputation Script
=======================================================

Precomputes retrieval results for FiD paper reproduction.

Paper: "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering"
Authors: Izacard & Grave (2021)

Key differences from FiD-Light precomputation:
1. Input format: "question: {q} title: {title} context: {text}" (separate title field)
2. No "index: X" in input (no source pointer prediction)
3. Target: answer only (not "index: X text: Y")
4. Default 100 passages (paper setting)
5. No gold passage injection (pure retrieval evaluation)

Features:
- Multi-GPU batch encoding
- Checkpoint/resume support (save every N batches)
- Preload Wikipedia to memory for faster processing

Usage:
    # Full precomputation (100 passages, all tasks)
    python precompute_retrieval_for_fid.py --tasks all --output_dir kilt_data/precomputed_fid/

    # Quick test
    python precompute_retrieval_for_fid.py --tasks nq --max_samples 100 --output_dir kilt_data/precomputed_fid/

    # Resume from checkpoint
    python precompute_retrieval_for_fid.py --tasks nq --resume --output_dir kilt_data/precomputed_fid/

    # Use 40 passages (faster, slightly lower accuracy)
    python precompute_retrieval_for_fid.py --tasks all --num_passages 40 --output_dir kilt_data/precomputed_fid_40/
"""

import argparse
import os
import json
import pickle
from typing import Dict, List, Any, Optional
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from kilt_loader import load_filtered_kilt_task, KILTWikipediaArrow
from gtr_retriever import GTRRetriever
from multitask_loader import extract_provenance_ids, extract_answer


# KILT tasks for FiD training (same as FiD-Light)
ALL_TASKS = [
    "nq",
    "hotpotqa",
    "triviaqa_support_only",
]


def format_fid_input(query: str, title: str, text: str) -> str:
    """
    Format input for FiD paper.

    Paper format: "question: {query} title: {title} context: {text}"

    Args:
        query: Question text
        title: Wikipedia article title
        text: Passage text (without title)

    Returns:
        Formatted input string
    """
    return f"question: {query} title: {title} context: {text}"


def extract_passage_content(article: Dict[str, Any], max_paragraphs: int = 5, max_chars: int = 1000) -> str:
    """
    Extract passage content WITHOUT title.

    Args:
        article: Wikipedia article dict
        max_paragraphs: Max paragraphs to include
        max_chars: Max characters for content

    Returns:
        Text content (without title)
    """
    paragraphs = article.get("text", [])

    if isinstance(paragraphs, list):
        text = " ".join(paragraphs[:max_paragraphs])
    else:
        text = str(paragraphs)

    if len(text) > max_chars:
        text = text[:max_chars]

    return text


def precompute_task_fid(
    task_name: str,
    retriever: GTRRetriever,
    wiki: KILTWikipediaArrow,
    num_passages: int = 100,
    max_chars_per_passage: int = 1000,
    max_samples: Optional[int] = None,
    verbose: bool = True,
    batch_size: int = 256,
    split: str = "train",
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 1000,
    resume: bool = False,
    filtered_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Precompute retrieval results for FiD paper format.

    Args:
        task_name: KILT task name
        retriever: GTRRetriever instance
        wiki: KILTWikipediaArrow instance
        num_passages: Number of passages per query (default 100 per paper)
        max_chars_per_passage: Max characters per passage
        max_samples: Max samples to process
        verbose: Print progress
        batch_size: Batch size for retrieval
        split: Data split ("train" or "validation")
        checkpoint_dir: Directory for checkpoints
        checkpoint_every: Save checkpoint every N samples
        resume: Resume from checkpoint
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

    # Check for checkpoint
    results = []
    start_batch = 0

    if checkpoint_dir and resume:
        checkpoint_path = os.path.join(checkpoint_dir, f"{task_name}_{split}_checkpoint.pkl")
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)
            results = checkpoint["results"]
            start_batch = checkpoint["next_batch"]
            print(f"    Resuming from batch {start_batch} ({len(results)} samples already processed)")

    stats = {
        "total": len(valid_samples),
        "with_provenance": 0,
        "retrieval_hit": 0,
    }

    # Process in batches
    num_batches = (len(queries) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(start_batch, num_batches), desc=f"  {task_name}", disable=not verbose, initial=start_batch, total=num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(queries))

        batch_queries = queries[start_idx:end_idx]
        batch_samples = valid_samples[start_idx:end_idx]

        # BATCH RETRIEVAL - GPU accelerated
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

            # Check retrieval hit (for stats only, not used in FiD)
            matching_indices = []
            for j, passage in enumerate(retrieved):
                wiki_id = passage.get("wikipedia_id", "")
                if wiki_id in provenance_ids:
                    matching_indices.append(j + 1)

            if matching_indices:
                stats["retrieval_hit"] += 1

            # Generate input_texts in FiD format
            input_texts = []
            retrieved_wiki_ids = []
            titles = []

            for j, passage in enumerate(retrieved):
                title = passage.get("title", "")
                text = passage.get("text", "")
                wiki_id = passage.get("wikipedia_id", "")

                # FiD format: "question: {q} title: {title} context: {text}"
                # Note: text may already contain title from retriever, extract just content
                # If text starts with title, remove it
                if text.startswith(title):
                    text = text[len(title):].strip()

                # Truncate text if needed
                if len(text) > max_chars_per_passage:
                    text = text[:max_chars_per_passage]

                formatted = format_fid_input(query, title, text)
                input_texts.append(formatted)
                retrieved_wiki_ids.append(wiki_id)
                titles.append(title)

            # Target is just the answer (no index prefix!)
            target_text = answer

            results.append({
                "id": sample_id,
                "task": task_name,
                "query": query,
                "answer": answer,
                "retrieved_wiki_ids": retrieved_wiki_ids,
                "titles": titles,  # Store titles separately for reference
                "matching_indices": sorted(matching_indices)[:3] if matching_indices else [],
                "input_texts": input_texts,
                "target_text": target_text,
                "gold_provenance_ids": list(provenance_ids),
            })

        # Save checkpoint
        if checkpoint_dir and (batch_idx + 1) % (checkpoint_every // batch_size + 1) == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{task_name}_{split}_checkpoint.pkl")
            with open(checkpoint_path, "wb") as f:
                pickle.dump({
                    "results": results,
                    "next_batch": batch_idx + 1,
                    "stats": stats,
                }, f)
            if verbose:
                print(f"\n    Checkpoint saved at batch {batch_idx + 1} ({len(results)} samples)")

    if verbose:
        hit_rate = stats["retrieval_hit"] / stats["with_provenance"] * 100 if stats["with_provenance"] > 0 else 0
        print(f"    Total: {stats['total']}, With provenance: {stats['with_provenance']}, "
              f"Retrieval hit: {stats['retrieval_hit']} ({hit_rate:.1f}%)")

    # Remove checkpoint after completion
    if checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, f"{task_name}_{split}_checkpoint.pkl")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            if verbose:
                print(f"    Checkpoint removed (task completed)")

    return results


def save_to_parquet_fid(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save precomputed results to Parquet format.

    Schema for FiD:
    - id: sample ID
    - task: task name
    - query: question text
    - answer: gold answer (also used as target_text)
    - retrieved_wiki_ids: list of Wikipedia IDs
    - titles: list of passage titles
    - matching_indices: indices matching gold provenance (for analysis)
    - input_texts: list of formatted inputs
    - target_text: answer only (no index prefix)
    - gold_provenance_ids: gold Wikipedia IDs (for analysis)
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
        "titles": [],
        "matching_indices": [],
        "input_texts": [],
        "target_text": [],
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
        "titles": pa.array(columns["titles"]),
        "matching_indices": pa.array(columns["matching_indices"]),
        "input_texts": pa.array(columns["input_texts"]),
        "target_text": pa.array(columns["target_text"]),
        "gold_provenance_ids": pa.array(columns["gold_provenance_ids"]),
    })

    # Save as Parquet
    pq.write_table(table, output_path)
    print(f"  Saved {len(results)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute retrieval results for FiD (Izacard & Grave, 2021)"
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
        default="kilt_data/precomputed_fid",
        help="Output directory for precomputed data"
    )
    parser.add_argument(
        "--num_passages",
        type=int,
        default=100,
        help="Number of passages to retrieve per query (paper default: 100)"
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=1000,
        help="Max characters per passage content (excluding title)"
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
        help="Batch size for GPU retrieval"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="all",
        help="Data splits: 'all' (train+validation), 'train', 'validation', or comma-separated"
    )
    parser.add_argument(
        "--use_multi_gpu",
        action="store_true",
        help="Use all available GPUs for encoding"
    )
    parser.add_argument(
        "--preload_wiki",
        action="store_true",
        help="Preload all Wikipedia articles to memory (~10-15GB)"
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=5000,
        help="Save checkpoint every N samples"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
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
    parser.add_argument(
        "--use_mmap",
        action="store_true",
        help="Use memory-mapped Faiss index (slower but less memory)"
    )

    args = parser.parse_args()

    # Parse tasks
    if args.tasks.lower() == "all":
        tasks = ALL_TASKS
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]

    # Parse splits
    if args.splits.lower() == "all":
        splits = ["train", "validation"]
    else:
        splits = [s.strip() for s in args.splits.split(",")]

    print("=" * 60)
    print("FiD Retrieval Precomputation (Izacard & Grave, 2021)")
    print("=" * 60)
    print(f"Tasks: {tasks}")
    print(f"Splits: {splits}")
    print(f"Passages per query: {args.num_passages}")
    print(f"Max chars per passage: {args.max_chars}")
    print(f"Batch size: {args.batch_size}")
    print(f"Index path: {args.index_path}")
    print(f"Model path: {args.model_path or 'gtr-t5-base (default)'}")
    print(f"Wiki Arrow path: {args.wiki_arrow_path or '(default)'}")
    print(f"Filtered dir: {args.filtered_dir or '(default)'}")
    print(f"Multi-GPU: {args.use_multi_gpu}")
    print(f"Preload Wiki: {args.preload_wiki}")
    print(f"Use mmap: {args.use_mmap}")
    print(f"Checkpoint every: {args.checkpoint_every} samples")
    print(f"Resume: {args.resume}")
    print(f"Output directory: {args.output_dir}")
    if args.max_samples:
        print(f"Max samples per task: {args.max_samples}")
    print()
    print("Format: question: {q} title: {title} context: {text}")
    print("Target: answer only (no index prefix)")
    print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load retriever
    print("Loading GTR retriever...")
    retriever = GTRRetriever(
        index_path=args.index_path,
        model_path=args.model_path,
        wiki_arrow_path=args.wiki_arrow_path,
        use_multi_gpu=args.use_multi_gpu,
        preload_wiki=args.preload_wiki,
        use_mmap=args.use_mmap,
    )

    # Use retriever's wiki
    wiki = retriever.wiki
    if wiki is None:
        print("Loading Wikipedia Arrow dataset...")
        wiki = KILTWikipediaArrow(arrow_path=args.wiki_arrow_path)
    print()

    # Process each split
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print(f"{'='*60}\n")

        all_results = []

        for task in tasks:
            print(f"Processing task: {task}")

            results = precompute_task_fid(
                task_name=task,
                retriever=retriever,
                wiki=wiki,
                num_passages=args.num_passages,
                max_chars_per_passage=args.max_chars,
                max_samples=args.max_samples,
                batch_size=args.batch_size,
                split=split,
                checkpoint_dir=args.output_dir,
                checkpoint_every=args.checkpoint_every,
                resume=args.resume,
                filtered_dir=args.filtered_dir,
            )

            if results:
                # Save per-task file
                split_suffix = "dev" if split == "validation" else "train"
                task_path = os.path.join(args.output_dir, f"{task}_{split_suffix}.parquet")
                save_to_parquet_fid(results, task_path)
                all_results.extend(results)

            print()

        # Save combined file for this split
        if all_results:
            split_suffix = "dev" if split == "validation" else "train"
            combined_path = os.path.join(args.output_dir, f"all_tasks_{split_suffix}.parquet")
            save_to_parquet_fid(all_results, combined_path)

    # Cleanup
    retriever.close()

    print("=" * 60)
    print("Precomputation complete!")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
