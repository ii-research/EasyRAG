"""
Batch evaluation script for Pure FiD checkpoints (Parallel version).

Evaluates all step_* checkpoints in parallel (one checkpoint per GPU).

Usage:
    python evaluate_fid_pure_all_checkpoints.py \
        --checkpoint_dir checkpoints/fid_pure \
        --val_path kilt_data/precomputed_fid/

    # With step range
    python evaluate_fid_pure_all_checkpoints.py \
        --checkpoint_dir checkpoints/fid_pure \
        --val_path kilt_data/precomputed_fid/ \
        --start_step 2000 --end_step 5000
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

# Web Demo state reporting (optional - only used when running from web UI)
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
            "algorithm": "fid_pure",
            "model": "t5base"
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
        pass  # Silently ignore web demo errors


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
                    # Extract step number
                    if name == "final":
                        if not end_step:  # Only include final if no end_step filter
                            checkpoints.append(path)
                    else:
                        match = re.search(r'step_(\d+)', name)
                        if match:
                            step = int(match.group(1))
                            # Filter by step range
                            if start_step and step < start_step:
                                continue
                            if end_step and step > end_step:
                                continue
                            checkpoints.append(path)

    # Sort by step number
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
    checkpoint_path, val_path, output_path, gpu_id, max_samples, bf16 = args_tuple
    checkpoint_name = os.path.basename(checkpoint_path)

    # Set CUDA device for this process
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, "evaluate_fid_pure.py",
        "--checkpoint", checkpoint_path,
        "--val_path", val_path,
        "--device", "cuda",
    ]

    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])

    if bf16:
        cmd.append("--bf16")

    print(f"[GPU {gpu_id}] Starting: {checkpoint_name}")
    start_time = time.time()

    # Run with real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )

    # Stream output with GPU prefix
    for line in process.stdout:
        print(f"[GPU {gpu_id}] {line}", end="")

    process.wait()
    elapsed = time.time() - start_time

    # Load results from checkpoint directory
    results_path = os.path.join(checkpoint_path, "eval_results_pure_fid.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

        # Also save to output_path for caching
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        avg = results.get("_average", {})
        em = avg.get("em", 0) * 100
        f1 = avg.get("f1", 0) * 100
        print(f"[GPU {gpu_id}] Finished: {checkpoint_name} in {elapsed:.1f}s - EM: {em:.2f}%, F1: {f1:.2f}%")
        return checkpoint_name, results

    print(f"[GPU {gpu_id}] Failed: {checkpoint_name}")
    return checkpoint_name, None


def load_all_eval_results(output_dir: str) -> Dict[str, Any]:
    """Load all existing eval JSON files from output directory."""
    all_results = {}

    pattern = os.path.join(output_dir, "*_eval.json")
    for json_path in glob.glob(pattern):
        try:
            with open(json_path) as f:
                results = json.load(f)
            # Extract checkpoint name from filename
            filename = os.path.basename(json_path)
            checkpoint_name = filename.replace("_eval.json", "")
            all_results[checkpoint_name] = results
        except Exception as e:
            print(f"Warning: Could not load {json_path}: {e}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation of Pure FiD checkpoints (parallel)"
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing step_* checkpoints")
    parser.add_argument("--val_path", type=str,
                        default="kilt_data/precomputed_fid/",
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
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use BF16 precision")
    parser.add_argument("--no_bf16", action="store_true",
                        help="Disable BF16")

    args = parser.parse_args()

    if args.no_bf16:
        args.bf16 = False

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint_dir, "eval_results")
    os.makedirs(args.output_dir, exist_ok=True)

    # Detect GPUs
    import torch
    available_gpus = torch.cuda.device_count()
    num_gpus = args.num_gpus if args.num_gpus else available_gpus
    print(f"Using {num_gpus} GPUs for parallel evaluation")

    # Find checkpoints
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

    # Filter out already evaluated checkpoints
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
        # Assign GPUs round-robin
        tasks_with_gpu = [
            (cp, val, out, i % num_gpus, args.max_samples, args.bf16)
            for i, (cp, val, out) in enumerate(tasks)
        ]

        print(f"\nStarting parallel evaluation on {num_gpus} GPUs...")
        print("=" * 80)

        # Run in parallel
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

                    # Report progress to web demo
                    report_evaluation_progress(completed_count, total_count, name, results)

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY: Pure FiD All Checkpoint Results")
    print(f"{'='*80}")
    print(f"{'Checkpoint':<20} {'NQ EM':>10} {'HotpotQA':>10} {'TriviaQA':>10} {'Avg EM':>10} {'Avg F1':>10}")
    print("-" * 80)

    # Sort by step number
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

    # Save combined results
    combined_path = os.path.join(args.output_dir, "all_eval_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Combined results saved to {combined_path}")


if __name__ == "__main__":
    main()
