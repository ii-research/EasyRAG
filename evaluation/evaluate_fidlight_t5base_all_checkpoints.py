"""
Batch Evaluation Script for FiD-Light T5-Base Checkpoints
==========================================================

Evaluates all T5-Base checkpoints in parallel (one checkpoint per GPU).
Generates plots showing loss curve and eval metrics over training.

Usage:
    python evaluate_fidlight_t5base_all_checkpoints.py \
        --checkpoint_dir $DATA_DIR/checkpoints/fidlight_v5_bf16 \
        --data_path $KILT_DATA_DIR/precomputed_v5/all_tasks_dev.parquet \
        --output_dir results/fidlight_t5base_all_checkpoints \
        --start_step 2500 --end_step 4700
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
import numpy as np

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
            "model": "t5base"
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
    checkpoint_path, data_path, output_path, gpu_id = args_tuple
    checkpoint_name = os.path.basename(checkpoint_path)

    # Set CUDA device for this process
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, "evaluate_fidlight.py",
        "--checkpoint", checkpoint_path,
        "--data_path", data_path,
        "--output", output_path,
    ]

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


def load_loss_history(checkpoint_dir: str) -> Dict[int, float]:
    """Load loss history from checkpoints and training_state files."""
    loss_by_step = {}

    # Method 1: Read from loss_history.npy in each checkpoint
    for checkpoint_path in glob.glob(os.path.join(checkpoint_dir, "step_*")):
        step_match = re.search(r'step_(\d+)', checkpoint_path)
        if not step_match:
            continue
        step = int(step_match.group(1))

        # Try loss_history.npy
        loss_file = os.path.join(checkpoint_path, "loss_history.npy")
        if os.path.exists(loss_file):
            try:
                losses = np.load(loss_file)
                if len(losses) > 0:
                    # Use the last loss value for this checkpoint
                    loss_by_step[step] = float(losses[-1])
            except Exception as e:
                print(f"Warning: Could not load {loss_file}: {e}")

        # Try training_state.json
        state_file = os.path.join(checkpoint_path, "training_state.json")
        if os.path.exists(state_file) and step not in loss_by_step:
            try:
                with open(state_file) as f:
                    state = json.load(f)
                if "loss_history" in state and state["loss_history"]:
                    loss_by_step[step] = float(state["loss_history"][-1])
            except Exception as e:
                print(f"Warning: Could not load {state_file}: {e}")

    return loss_by_step


def load_all_eval_results(output_dir: str) -> Dict[str, Any]:
    """
    Load all existing eval JSON files from output directory.

    This ensures that plots include ALL evaluated checkpoints,
    even if current run only covers a subset of steps.
    """
    all_results = {}

    pattern = os.path.join(output_dir, "*_eval.json")
    for json_path in glob.glob(pattern):
        try:
            with open(json_path) as f:
                results = json.load(f)
            # Extract checkpoint name from filename (e.g., "step_2500_eval.json" -> "step_2500")
            filename = os.path.basename(json_path)
            checkpoint_name = filename.replace("_eval.json", "")
            all_results[checkpoint_name] = results
        except Exception as e:
            print(f"Warning: Could not load {json_path}: {e}")

    return all_results


def plot_eval_progress(
    eval_results: Dict[str, Any],
    output_path: str,
    start_step: int = 0,
):
    """
    Generate 2x2 eval metrics plot.

    Layout:
        - Top-left: Overall (EM, F1, Prov, KILT)
        - Top-right: nq task
        - Bottom-left: hotpotqa task
        - Bottom-right: triviaqa_support_only task

    Args:
        eval_results: Dict mapping checkpoint_name -> metrics dict
        output_path: Path to save the plot
        start_step: Start step for x-axis (default 0)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation")
        return

    # Extract data by task
    tasks = ['overall', 'nq', 'hotpotqa', 'triviaqa_support_only']
    task_titles = ['Overall', 'NQ', 'HotpotQA', 'TriviaQA']

    # Collect metrics for each task
    data = {task: {'steps': [], 'em': [], 'f1': [], 'prov': [], 'kilt': []} for task in tasks}

    for name, metrics in eval_results.items():
        if metrics is None:
            continue
        match = re.search(r'step_(\d+)', name)
        if not match:
            continue
        step = int(match.group(1))

        # Overall
        data['overall']['steps'].append(step)
        data['overall']['em'].append(metrics.get('answer_accuracy', 0))
        data['overall']['f1'].append(metrics.get('answer_f1', 0))
        data['overall']['prov'].append(metrics.get('provenance_accuracy', 0))
        data['overall']['kilt'].append(metrics.get('kilt_score', 0))

        # Per-task
        per_task = metrics.get('per_task', {})
        for task in ['nq', 'hotpotqa', 'triviaqa_support_only']:
            if task in per_task:
                data[task]['steps'].append(step)
                data[task]['em'].append(per_task[task].get('answer_accuracy', 0))
                data[task]['f1'].append(per_task[task].get('answer_f1', 0))
                data[task]['prov'].append(per_task[task].get('provenance_accuracy', 0))
                data[task]['kilt'].append(per_task[task].get('kilt_score', 0))

    # Sort each task's data by step
    for task in tasks:
        if data[task]['steps']:
            sorted_idx = np.argsort(data[task]['steps'])
            for key in ['steps', 'em', 'f1', 'prov', 'kilt']:
                data[task][key] = [data[task][key][i] for i in sorted_idx]

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (task, title) in enumerate(zip(tasks, task_titles)):
        ax = axes[idx]
        d = data[task]

        if d['steps']:
            ax.plot(d['steps'], d['em'], 'b-o', label='EM', markersize=4, linewidth=1.5)
            ax.plot(d['steps'], d['f1'], 'c-^', label='F1', markersize=4, linewidth=1.5)
            ax.plot(d['steps'], d['prov'], 'orange', marker='s', label='Prov', markersize=4, linewidth=1.5)
            ax.plot(d['steps'], d['kilt'], 'g-D', label='KILT', markersize=4, linewidth=1.5)

        ax.set_title(title)
        ax.set_xlabel('Step')
        ax.set_ylabel('Score (%)')
        ax.set_xlim(left=start_step)
        ax.set_ylim(0, 100)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('FiD-Light T5-Base Evaluation Progress', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Eval plot saved to {output_path}")


def plot_loss_progress(
    checkpoint_dir: str,
    output_path: str,
    start_step: int = 0,
    end_step: Optional[int] = None,
):
    """
    Generate loss curve plot from all checkpoints.

    Args:
        checkpoint_dir: Directory containing checkpoints
        output_path: Path to save the plot
        start_step: Start step for x-axis
        end_step: End step (optional)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation")
        return

    # Collect all loss values from all checkpoints
    all_losses = []
    all_steps = []

    # Find the latest checkpoint to get full loss history
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "step_*")),
                        key=lambda x: int(re.search(r'step_(\d+)', x).group(1)))

    if not checkpoints:
        print("No checkpoints found for loss plot")
        return

    # Use the latest checkpoint's full loss history
    latest_checkpoint = checkpoints[-1]

    # Try loss_history.npy first
    loss_file = os.path.join(latest_checkpoint, "loss_history.npy")
    if os.path.exists(loss_file):
        try:
            all_losses = np.load(loss_file).tolist()
        except Exception as e:
            print(f"Warning: Could not load {loss_file}: {e}")

    # Try training_state.json as fallback
    if not all_losses:
        state_file = os.path.join(latest_checkpoint, "training_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file) as f:
                    state = json.load(f)
                all_losses = state.get("loss_history", [])
            except Exception as e:
                print(f"Warning: Could not load {state_file}: {e}")

    if not all_losses:
        print("No loss history found")
        return

    # Generate step numbers (assuming loss recorded every gradient update step)
    all_steps = list(range(1, len(all_losses) + 1))

    # Filter by step range
    filtered_steps = []
    filtered_losses = []
    for s, l in zip(all_steps, all_losses):
        if s >= start_step and (end_step is None or s <= end_step):
            filtered_steps.append(s)
            filtered_losses.append(l)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(filtered_steps, filtered_losses, 'r-', linewidth=1, alpha=0.8)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('FiD-Light T5-Base Training Loss')
    ax.set_xlim(left=start_step)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate all FiD-Light T5-Base checkpoints")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing T5-Base checkpoints",
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
        default="results/fidlight_t5base_all_checkpoints",
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
        print(f"Found {len(checkpoints)} T5-Base checkpoints in range {range_str}:")
    else:
        print(f"Found {len(checkpoints)} T5-Base checkpoints:")

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
            tasks.append((checkpoint_path, args.data_path, output_path))

    print(f"\nNeed to evaluate: {len(tasks)} checkpoints")
    print(f"Already cached: {len(cached_results)} checkpoints")

    # Define plot paths
    eval_plot_path = os.path.join(args.output_dir, "eval_progress.png")
    loss_plot_path = os.path.join(args.output_dir, "loss_progress.png")

    if not tasks:
        all_results = cached_results
        # Generate plots from ALL existing results (not just current range)
        all_existing_results = load_all_eval_results(args.output_dir)
        print(f"\n[Plot] Generating plots from {len(all_existing_results)} total checkpoints...")
        plot_eval_progress(
            all_existing_results,
            eval_plot_path,
            start_step=0  # Show all from beginning
        )
        plot_loss_progress(
            args.checkpoint_dir,
            loss_plot_path,
            start_step=0,
            end_step=None  # Show full loss history
        )
    else:
        # Assign GPUs round-robin
        tasks_with_gpu = [
            (cp, dp, op, i % num_gpus)
            for i, (cp, dp, op) in enumerate(tasks)
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

                    # Generate incremental plots after each checkpoint
                    # Load ALL existing results to include previously evaluated checkpoints
                    all_existing_results = load_all_eval_results(args.output_dir)
                    print(f"\n[Plot] Generating updated plots with {len(all_existing_results)} total checkpoints...")
                    plot_eval_progress(
                        all_existing_results,
                        eval_plot_path,
                        start_step=0  # Show all from beginning
                    )
                    plot_loss_progress(
                        args.checkpoint_dir,
                        loss_plot_path,
                        start_step=0,
                        end_step=None  # Show full loss history
                    )

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: FiD-Light T5-Base All Checkpoint Results")
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
        "backbone": "t5-base",
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
