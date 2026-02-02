"""
Batch Evaluation Script for FiD-Light T5Gemma2 Checkpoints
===========================================================

Evaluates all T5Gemma2 checkpoints in parallel (one checkpoint per GPU).

Usage:
    python evaluate_fidlight_t5gemma_all_checkpoints.py \
        --checkpoint_dir $DATA_DIR/checkpoints/fidlight_t5gemma_270m \
        --data_path $KILT_DATA_DIR/precomputed_v5/all_tasks_dev.parquet \
        --output_dir results/fidlight_t5gemma_all_checkpoints \
        --start_step 2000 --end_step 4200
"""

import argparse
import os
import re
import json
import glob
from typing import Dict, List, Any, Optional
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

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
            "algorithm": "fidlight",
            "model": "t5gemma"
        }
        if results:
            extra["exact_match"] = results.get("overall", {}).get("answer_accuracy", 0)
            extra["kilt_score"] = results.get("overall", {}).get("kilt_score", 0)
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
    """Find all valid checkpoints in directory within step range."""
    checkpoints = []

    # Look for step_* directories
    pattern = os.path.join(checkpoint_dir, "step_*")
    for path in glob.glob(pattern):
        if os.path.isdir(path):
            # Check if it has model files
            if (os.path.exists(os.path.join(path, "config.json")) or
                os.path.exists(os.path.join(path, "pytorch_model.bin")) or
                os.path.exists(os.path.join(path, "model.safetensors"))):

                # Extract step number
                name = os.path.basename(path)
                match = re.search(r'step_(\d+)', name)
                if match:
                    step = int(match.group(1))
                    # Filter by step range
                    if start_step and step < start_step:
                        continue
                    if end_step and step > end_step:
                        continue
                checkpoints.append(path)

    # Also check for 'final' checkpoint (only if no end_step filter)
    if not end_step:
        final_path = os.path.join(checkpoint_dir, "final")
        if os.path.isdir(final_path):
            checkpoints.append(final_path)

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
    checkpoint_path, data_path, output_path, gpu_id, max_samples, num_passages = args_tuple
    checkpoint_name = os.path.basename(checkpoint_path)

    # Set CUDA device for this process
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Use T5Gemma2 evaluation script
    cmd = [
        sys.executable, "evaluate_fidlight_t5gemma.py",
        "--checkpoint", checkpoint_path,
        "--data_path", data_path,
        "--output", output_path,
    ]

    # Add optional arguments
    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])
    if num_passages:
        cmd.extend(["--num_passages", str(num_passages)])

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

    # Load results
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        print(f"[GPU {gpu_id}] Finished: {checkpoint_name} in {elapsed:.1f}s - KILT: {results['kilt_score']:.2f}%")
        return checkpoint_name, results

    print(f"[GPU {gpu_id}] Failed: {checkpoint_name}")
    return checkpoint_name, None


def main():
    parser = argparse.ArgumentParser(description="Evaluate all FiD-Light T5Gemma2 checkpoints")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing T5Gemma2 checkpoints",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to validation data parquet",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/fidlight_t5gemma_all_checkpoints",
        help="Directory to save individual results",
    )
    parser.add_argument(
        "--start_step",
        type=int,
        default=None,
        help="Start from this step (inclusive)",
    )
    parser.add_argument(
        "--end_step",
        type=int,
        default=None,
        help="End at this step (inclusive)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples per evaluation (for quick testing)",
    )
    parser.add_argument(
        "--num_passages",
        type=int,
        default=None,
        help="Limit passages per query (set to 20 if model trained with --num_passages 20)",
    )

    args = parser.parse_args()

    # Detect GPUs
    import torch
    available_gpus = torch.cuda.device_count()
    num_gpus = args.num_gpus if args.num_gpus else available_gpus
    print(f"Using {num_gpus} GPUs for parallel evaluation")

    # Find checkpoints
    checkpoints = find_checkpoints(args.checkpoint_dir, args.start_step, args.end_step)

    if args.start_step or args.end_step:
        range_str = f"step {args.start_step or 'start'} to {args.end_step or 'end'}"
        print(f"Found {len(checkpoints)} T5Gemma2 checkpoints in range {range_str}:")
    else:
        print(f"Found {len(checkpoints)} T5Gemma2 checkpoints:")

    for cp in checkpoints:
        print(f"  - {os.path.basename(cp)}")

    if not checkpoints:
        print("No checkpoints found!")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

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
            tasks.append((checkpoint_path, args.data_path, output_path, args.max_samples, args.num_passages))

    print(f"\nNeed to evaluate: {len(tasks)} checkpoints")
    print(f"Already cached: {len(cached_results)} checkpoints")

    if not tasks:
        all_results = cached_results
    else:
        # Assign GPUs round-robin
        tasks_with_gpu = [
            (cp, dp, op, i % num_gpus, ms, np)
            for i, (cp, dp, op, ms, np) in enumerate(tasks)
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

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: FiD-Light T5Gemma2 All Checkpoint Results")
    print("=" * 80)
    print(f"{'Checkpoint':<20} {'EM':>10} {'F1':>10} {'Prov':>10} {'KILT':>10}")
    print("-" * 80)

    # Sort by step number for display
    def get_step(name):
        if name == "final":
            return float('inf')
        match = re.search(r'step_(\d+)', name)
        return int(match.group(1)) if match else 0

    sorted_names = sorted(all_results.keys(), key=get_step)

    best_kilt = 0
    best_checkpoint = None
    best_em = 0
    best_em_checkpoint = None

    for name in sorted_names:
        r = all_results[name]
        print(f"{name:<20} {r['answer_accuracy']:>9.2f}% {r['answer_f1']:>9.2f}% "
              f"{r['provenance_accuracy']:>9.2f}% {r['kilt_score']:>9.2f}%")

        if r['kilt_score'] > best_kilt:
            best_kilt = r['kilt_score']
            best_checkpoint = name

        if r['answer_accuracy'] > best_em:
            best_em = r['answer_accuracy']
            best_em_checkpoint = name

    print("=" * 80)
    print(f"\nBest KILT Score: {best_checkpoint} ({best_kilt:.2f}%)")
    print(f"Best EM Score:   {best_em_checkpoint} ({best_em:.2f}%)")

    # Save summary
    summary = {
        "backbone": "t5gemma2-270m-270m",
        "checkpoints": all_results,
        "best_kilt_checkpoint": best_checkpoint,
        "best_kilt_score": best_kilt,
        "best_em_checkpoint": best_em_checkpoint,
        "best_em_score": best_em,
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Per-task comparison for best checkpoint
    if best_checkpoint and "per_task" in all_results[best_checkpoint]:
        print(f"\nPer-Task Results for Best Checkpoint ({best_checkpoint}):")
        print("-" * 70)
        print(f"{'Task':<25} {'EM':>10} {'F1':>10} {'Prov':>10} {'KILT':>10}")
        print("-" * 70)
        for task, metrics in sorted(all_results[best_checkpoint]["per_task"].items()):
            print(f"{task:<25} {metrics['answer_accuracy']:>9.2f}% {metrics['answer_f1']:>9.2f}% "
                  f"{metrics['provenance_accuracy']:>9.2f}% {metrics['kilt_score']:>9.2f}%")


if __name__ == "__main__":
    main()
