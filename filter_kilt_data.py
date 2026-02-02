#!/usr/bin/env python
"""
KILT Dataset Filter
===================
Filter out samples without provenance to generate datasets usable for FiD-Light training.

Provenance Coverage Analysis Results:
-------------------------------------
Task                      Train Coverage   Validation Coverage
nq                        88.1%            100%
aidayago2                 100%             100%
cweb                      No train set     100%
eli5                      0% (!)           100%
fever                     67.9%            100%
hotpotqa                  77.3%            100%
structured_zeroshot       89.3%            100%
trex                      100%             100%
triviaqa_support_only     85.5%            100%
wned                      No train set     100%
wow                       85.2%            100%

Note: eli5 training set has no provenance at all; only validation set or skip this task.

Usage:
    # Filter all tasks
    python filter_kilt_data.py

    # Filter specific tasks
    python filter_kilt_data.py --tasks nq fever hotpotqa

    # Filter only training set
    python filter_kilt_data.py --splits train
"""

import argparse
import json
import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Default directories
DEFAULT_CACHE_DIR = "./kilt_data"
DEFAULT_OUTPUT_DIR = "./kilt_data/filtered"
DEFAULT_TRIVIAQA_FIXED_DIR = "./kilt_data/triviaqa_fixed"

# Available tasks (3 QA tasks only)
KILT_TASK_NAMES = [
    "nq", "hotpotqa", "triviaqa_support_only"
]


def has_valid_provenance(example: Dict[str, Any]) -> bool:
    """
    Check if sample has valid provenance.

    Valid provenance requires:
    1. At least one output contains a non-empty provenance list
    2. At least one provenance entry contains a wikipedia_id
    """
    for output in example.get('output', []):
        provenance_list = output.get('provenance', [])
        if provenance_list:
            # Check for valid wikipedia_id
            for prov in provenance_list:
                if prov.get('wikipedia_id'):
                    return True
    return False


def load_triviaqa_from_fixed(split: str, fixed_dir: str = DEFAULT_TRIVIAQA_FIXED_DIR) -> List[Dict]:
    """
    Load TriviaQA data from fix_triviaqa.py output intermediate files.

    Args:
        split: Data split (train, validation, test)
        fixed_dir: fix_triviaqa.py output directory

    Returns:
        List of samples
    """
    file_path = os.path.join(fixed_dir, f"triviaqa_support_only_{split}.jsonl")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"TriviaQA fixed file does not exist: {file_path}\n"
            f"Please run first: python fix_triviaqa.py --output_dir {fixed_dir}"
        )

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    return data


def filter_dataset(
    task_name: str,
    split: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
    triviaqa_fixed_dir: str = DEFAULT_TRIVIAQA_FIXED_DIR
) -> tuple:
    """
    Filter data for specified task and split.

    For TriviaQA, load from fix_triviaqa.py output intermediate files.
    For other tasks, load from HuggingFace.

    Returns:
        (filtered_data, stats) - Filtered data and statistics
    """
    # TriviaQA: Load from local fixed files (fix_triviaqa.py output)
    if task_name == "triviaqa_support_only":
        try:
            data = load_triviaqa_from_fixed(split, triviaqa_fixed_dir)
            print(f"Loaded TriviaQA from local: {len(data)} samples")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return [], {"error": str(e)}
    else:
        # Other tasks: Load from HuggingFace
        from datasets import load_dataset

        try:
            dataset = load_dataset(
                "facebook/kilt_tasks",
                name=task_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load {task_name}: {e}")
            return [], {"error": str(e)}

        if split not in dataset:
            return [], {"error": f"Split {split} does not exist"}

        data = list(dataset[split])

    total = len(data)

    filtered = []
    for example in tqdm(data, desc=f"Filtering {task_name}/{split}"):
        if has_valid_provenance(example):
            # Convert to serializable dict
            filtered.append({
                'id': example['id'],
                'input': example['input'],
                'output': example['output'],
                'meta': example.get('meta', {})
            })

    stats = {
        "total": total,
        "kept": len(filtered),
        "removed": total - len(filtered),
        "keep_rate": len(filtered) / total * 100 if total > 0 else 0
    }

    return filtered, stats


def save_filtered_data(
    data: List[Dict],
    task_name: str,
    split: str,
    output_dir: str = DEFAULT_OUTPUT_DIR
) -> str:
    """Save filtered data in JSONL format."""
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{task_name}_{split}.jsonl")

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Filter KILT dataset, keeping samples with provenance")
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=KILT_TASK_NAMES,
        help=f"List of tasks to filter, default all. Available: {KILT_TASK_NAMES}"
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'validation'],
        help="Splits to filter, default train and validation"
    )
    parser.add_argument(
        '--cache-dir',
        default=DEFAULT_CACHE_DIR,
        help="KILT data cache directory"
    )
    parser.add_argument(
        '--output-dir',
        default=DEFAULT_OUTPUT_DIR,
        help="Filtered data output directory"
    )
    parser.add_argument(
        '--skip-100-percent',
        action='store_true',
        help="Skip datasets with 100% provenance coverage (avoid reprocessing)"
    )
    parser.add_argument(
        '--triviaqa-fixed-dir',
        default=DEFAULT_TRIVIAQA_FIXED_DIR,
        help="fix_triviaqa.py output directory (TriviaQA intermediate files)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("KILT Data Filter - Keep Samples with Provenance")
    print("=" * 60)
    print(f"Tasks: {args.tasks}")
    print(f"Splits: {args.splits}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Summary statistics
    summary = []

    for task_name in args.tasks:
        print(f"\n{'='*40}")
        print(f"Processing task: {task_name}")
        print('='*40)

        for split in args.splits:
            print(f"\n--- {split} ---")

            filtered_data, stats = filter_dataset(
                task_name, split, args.cache_dir, args.triviaqa_fixed_dir
            )

            if "error" in stats:
                print(f"Skipped: {stats['error']}")
                continue

            # Save filtered data
            if filtered_data:
                output_path = save_filtered_data(
                    filtered_data, task_name, split, args.output_dir
                )
                print(f"Total: {stats['total']}")
                print(f"Kept: {stats['kept']} ({stats['keep_rate']:.1f}%)")
                print(f"Removed: {stats['removed']}")
                print(f"Saved to: {output_path}")
            else:
                print(f"Warning: No valid samples to keep!")

            summary.append({
                "task": task_name,
                "split": split,
                **stats
            })

    # Print summary
    print("\n" + "=" * 60)
    print("Filter Summary")
    print("=" * 60)
    print(f"{'Task':<25} {'Split':<12} {'Kept':<10} {'Removed':<10} {'Keep Rate':<10}")
    print("-" * 60)

    total_kept = 0
    total_removed = 0

    for s in summary:
        if "error" not in s:
            print(f"{s['task']:<25} {s['split']:<12} {s['kept']:<10} {s['removed']:<10} {s['keep_rate']:.1f}%")
            total_kept += s.get('kept', 0)
            total_removed += s.get('removed', 0)

    print("-" * 60)
    print(f"{'Total':<37} {total_kept:<10} {total_removed:<10}")
    print(f"\nFiltered data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
