"""
Batch evaluation script for Naive RAG T5Gemma2 checkpoints (Parallel version).

Evaluates all step_* checkpoints in parallel (one checkpoint per GPU).

Usage:
    python evaluate_naive_rag_t5gemma_all_checkpoints.py \
        --checkpoint_dir checkpoints/naive_rag_t5gemma_v1 \
        --val_path kilt_data/precomputed_v5/

    # With step range
    python evaluate_naive_rag_t5gemma_all_checkpoints.py \
        --checkpoint_dir checkpoints/naive_rag_t5gemma_v1 \
        --val_path kilt_data/precomputed_v5/ \
        --start_step 2000 --end_step 10000

    # Custom number of passages
    python evaluate_naive_rag_t5gemma_all_checkpoints.py \
        --checkpoint_dir checkpoints/naive_rag_t5gemma_v1 \
        --val_path kilt_data/precomputed_v5/ \
        --num_passages 10
"""

import argparse
import os
import re
import subprocess
import sys
import json
import time
import glob
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# Web Demo state reporting (optional)
try:
    from web_demo.utils.state_io import update_step_state, StepStatus
    HAS_WEB_DEMO = True
except ImportError:
    HAS_WEB_DEMO = False


def report_evaluation_progress(current: int, total: int, checkpoint_name: str, results: dict = None):
    """Report evaluation progress to web demo (if available)."""
    if not HAS_WEB_DEMO:
        return
    try:
        progress = (current / total) * 100 if total > 0 else 0
        message = f"Checkpoint {current}/{total}: {checkpoint_name}"
        extra = {
            "current": current,
            "total": total,
            "checkpoint": checkpoint_name,
            "algorithm": "naive_rag",
            "model": "t5gemma"
        }
        if results:
            extra["exact_match"] = results.get("em", 0) * 100
            extra["f1"] = results.get("f1", 0) * 100
        update_step_state(
            step_name="evaluate_all",
            progress=progress,
            message=message,
            status=StepStatus.RUNNING.value,
            extra=extra
        )
    except Exception:
        pass


def find_checkpoints(
    checkpoint_dir: str,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,
) -> List[str]:
    """Find all step_* checkpoints in directory within step range."""
    checkpoints = []
    for name in os.listdir(checkpoint_dir):
        if name.startswith("step_") or name == "final":
            path = os.path.join(checkpoint_dir, name)
            if os.path.isdir(path):
                # Check if it has model files
                if os.path.exists(os.path.join(path, "config.json")):
                    if name == "final":
                        if not end_step:
                            checkpoints.append(path)
                    else:
                        match = re.search(r'step_(\d+)', name)
                        if match:
                            step = int(match.group(1))
                            if start_step and step < start_step:
                                continue
                            if end_step and step > end_step:
                                continue
                            checkpoints.append(path)

    def get_step(path):
        name = os.path.basename(path)
        if name == "final":
            return float('inf')
        match = re.search(r'step_(\d+)', name)
        return int(match.group(1)) if match else 0

    checkpoints.sort(key=get_step)
    return checkpoints


def evaluate_single_checkpoint(args_tuple):
    """Evaluate a single checkpoint on specified GPU. Returns (name, results)."""
    checkpoint_path, val_path, output_path, gpu_id, max_samples, num_passages = args_tuple
    checkpoint_name = os.path.basename(checkpoint_path)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, "evaluate_naive_rag_t5gemma.py",
        "--checkpoint", checkpoint_path,
        "--data_path", val_path,
        "--output", output_path,
        "--num_passages", str(num_passages),
        "--device", "cuda",
        "--bf16",
    ]

    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])

    print(f"[GPU {gpu_id}] Starting: {checkpoint_name}")
    start_time = time.time()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )

    for line in process.stdout:
        print(f"[GPU {gpu_id}] {line}", end="")

    process.wait()
    elapsed = time.time() - start_time

    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)

        avg = results.get("_average", {})
        em = avg.get("em", 0) * 100
        f1 = avg.get("f1", 0) * 100
        print(f"[GPU {gpu_id}] Finished: {checkpoint_name} in {elapsed:.1f}s - EM: {em:.2f}%, F1: {f1:.2f}%")
        return checkpoint_name, results

    print(f"[GPU {gpu_id}] Failed: {checkpoint_name}")
    return checkpoint_name, None


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation of Naive RAG T5Gemma2 checkpoints (parallel)"
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing step_* checkpoints")
    parser.add_argument("--val_path", type=str,
                        default="kilt_data/precomputed_v5/",
                        help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results (default: checkpoint_dir/eval_results)")
    parser.add_argument("--start_step", type=int, default=None,
                        help="Start from this step (inclusive)")
    parser.add_argument("--end_step", type=int, default=None,
                        help="End at this step (inclusive)")
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs to use (default: all available)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per task")
    parser.add_argument("--num_passages", type=int, default=10,
                        help="Number of passages to use (default: 10)")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint_dir, "eval_results")
    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    available_gpus = torch.cuda.device_count()
    num_gpus = args.num_gpus if args.num_gpus else available_gpus
    print(f"Using {num_gpus} GPUs for parallel evaluation")
    print(f"Using {args.num_passages} passages per sample")

    checkpoints = find_checkpoints(args.checkpoint_dir, args.start_step, args.end_step)

    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return

    if args.start_step or args.end_step:
        range_str = f"step {args.start_step or 'start'} to {args.end_step or 'end'}"
        print(f"Found {len(checkpoints)} checkpoints in range {range_str}:")
    else:
        print(f"Found {len(checkpoints)} checkpoints:")

    for cp in checkpoints:
        print(f"  {os.path.basename(cp)}")

    tasks = []
    cached_results = {}

    for checkpoint_path in checkpoints:
        checkpoint_name = os.path.basename(checkpoint_path)
        output_path = os.path.join(args.output_dir, f"{checkpoint_name}_eval.json")

        if os.path.exists(output_path):
            print(f"  [cached] {checkpoint_name}")
            with open(output_path) as f:
                cached_results[checkpoint_name] = json.load(f)
        else:
            tasks.append((checkpoint_path, args.val_path, output_path))

    print(f"\nNeed to evaluate: {len(tasks)} checkpoints")
    print(f"Already cached: {len(cached_results)} checkpoints")

    if not tasks:
        all_results = cached_results
    else:
        tasks_with_gpu = [
            (cp, val, out, i % num_gpus, args.max_samples, args.num_passages)
            for i, (cp, val, out) in enumerate(tasks)
        ]

        print(f"\nStarting parallel evaluation on {num_gpus} GPUs...")
        print("=" * 80)

        all_results = dict(cached_results)
        completed_count = len(cached_results)
        total_count = len(tasks_with_gpu) + len(cached_results)

        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = {executor.submit(evaluate_single_checkpoint, t): t for t in tasks_with_gpu}

            for future in as_completed(futures):
                name, results = future.result()
                if results:
                    all_results[name] = results
                    completed_count += 1
                    report_evaluation_progress(completed_count, total_count, name, results)

    print(f"\n{'='*80}")
    print("SUMMARY: Naive RAG T5Gemma2 All Checkpoint Results")
    print(f"{'='*80}")
    print(f"{'Checkpoint':<20} {'NQ EM':>10} {'HotpotQA':>10} {'TriviaQA':>10} {'Avg EM':>10} {'Avg F1':>10}")
    print("-" * 80)

    def get_step(name):
        if name == "final":
            return float('inf')
        match = re.search(r'step_(\d+)', name)
        return int(match.group(1)) if match else 0

    sorted_names = sorted(all_results.keys(), key=get_step)

    best_em = 0
    best_checkpoint = None

    for name in sorted_names:
        r = all_results[name]
        nq = r.get("nq", {}).get("em", 0) * 100
        hotpot = r.get("hotpotqa", {}).get("em", 0) * 100
        trivia = r.get("triviaqa_support_only", {}).get("em", 0) * 100
        avg = r.get("_average", {})
        avg_em = avg.get("em", 0) * 100
        avg_f1 = avg.get("f1", 0) * 100

        print(f"{name:<20} {nq:>9.2f}% {hotpot:>9.2f}% {trivia:>9.2f}% {avg_em:>9.2f}% {avg_f1:>9.2f}%")

        if avg_em > best_em:
            best_em = avg_em
            best_checkpoint = name

    print("=" * 80)
    print(f"\nBest EM: {best_checkpoint} ({best_em:.2f}%)")

    combined_path = os.path.join(args.output_dir, "all_eval_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Combined results saved to {combined_path}")


if __name__ == "__main__":
    main()
