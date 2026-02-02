"""
Batch Evaluation Script for Stochastic RAG Checkpoints
=======================================================

Evaluates all Stochastic RAG checkpoints in parallel (one checkpoint per GPU).
Generates plots showing loss curve, E[U] curve, and eval metrics over training.

Usage:
    python evaluate_stochastic_rag_all_checkpoints.py \
        --checkpoint_dir $DATA_DIR/checkpoints/stochastic_rag_v5 \
        --data_path $KILT_DATA_DIR/precomputed_v5/all_tasks_dev.parquet \
        --output_dir results/stochastic_rag_v5_all_checkpoints \
        --start_step 500 --end_step 5000
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
            "algorithm": "stochastic_rag",
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
            # Check if it has model files and reranker
            has_model = (os.path.exists(os.path.join(path, "config.json")) or
                os.path.exists(os.path.join(path, "pytorch_model.bin")) or
                os.path.exists(os.path.join(path, "model.safetensors")))
            has_reranker = os.path.exists(os.path.join(path, "reranker.pt"))

            if has_model:
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
    checkpoint_path, data_path, output_path, gpu_id, num_beams = args_tuple
    checkpoint_name = os.path.basename(checkpoint_path)

    # Set CUDA device for this process
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, "evaluate_stochastic_rag.py",
        "--checkpoint", checkpoint_path,
        "--data_path", data_path,
        "--output", output_path,
        "--num_beams", str(num_beams),
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


def load_training_history(checkpoint_dir: str) -> Dict[str, List]:
    """Load loss and E[U] history from checkpoints."""
    history = {
        "loss": [],
        "utility": [],
    }

    # Find the latest checkpoint to get full history
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "step_*")),
                        key=lambda x: int(re.search(r'step_(\d+)', x).group(1)))

    if not checkpoints:
        return history

    # Use the latest checkpoint's full history
    latest_checkpoint = checkpoints[-1]

    # Try training_state.json
    state_file = os.path.join(latest_checkpoint, "training_state.json")
    if os.path.exists(state_file):
        try:
            with open(state_file) as f:
                state = json.load(f)
            history["loss"] = state.get("loss_history", [])
            history["utility"] = state.get("utility_history", [])
        except Exception as e:
            print(f"Warning: Could not load {state_file}: {e}")

    return history


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
        - Top-left: Overall (EM, F1, Prov, SelProv, KILT)
        - Top-right: nq task
        - Bottom-left: hotpotqa task
        - Bottom-right: triviaqa_support_only task
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
    data = {task: {'steps': [], 'em': [], 'f1': [], 'prov': [], 'sel_prov': [], 'kilt': []} for task in tasks}

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
        data['overall']['sel_prov'].append(metrics.get('selected_provenance_accuracy', 0))
        data['overall']['kilt'].append(metrics.get('kilt_score', 0))

        # Per-task
        per_task = metrics.get('per_task', {})
        for task in ['nq', 'hotpotqa', 'triviaqa_support_only']:
            if task in per_task:
                data[task]['steps'].append(step)
                data[task]['em'].append(per_task[task].get('answer_accuracy', 0))
                data[task]['f1'].append(per_task[task].get('answer_f1', 0))
                data[task]['prov'].append(per_task[task].get('provenance_accuracy', 0))
                data[task]['sel_prov'].append(per_task[task].get('selected_provenance_accuracy', 0))
                data[task]['kilt'].append(per_task[task].get('kilt_score', 0))

    # Sort each task's data by step
    for task in tasks:
        if data[task]['steps']:
            sorted_idx = np.argsort(data[task]['steps'])
            for key in ['steps', 'em', 'f1', 'prov', 'sel_prov', 'kilt']:
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
            ax.plot(d['steps'], d['sel_prov'], 'm-*', label='SelProv', markersize=4, linewidth=1.5)
            ax.plot(d['steps'], d['kilt'], 'g-D', label='KILT', markersize=4, linewidth=1.5)

        ax.set_title(title)
        ax.set_xlabel('Step')
        ax.set_ylabel('Score (%)')
        ax.set_xlim(left=start_step)
        ax.set_ylim(0, 100)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Stochastic RAG Evaluation Progress', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Eval plot saved to {output_path}")


def plot_training_progress(
    checkpoint_dir: str,
    output_path: str,
    start_step: int = 0,
    end_step: Optional[int] = None,
):
    """
    Generate training curves plot (Loss and E[U]).
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation")
        return

    history = load_training_history(checkpoint_dir)

    if not history["loss"]:
        print("No loss history found")
        return

    # Generate step numbers
    all_steps = list(range(1, len(history["loss"]) + 1))

    # Filter by step range
    filtered_steps = []
    filtered_losses = []
    filtered_utility = []

    for i, s in enumerate(all_steps):
        if s >= start_step and (end_step is None or s <= end_step):
            filtered_steps.append(s)
            filtered_losses.append(history["loss"][i])
            if i < len(history["utility"]):
                filtered_utility.append(history["utility"][i])

    # Create 1x2 subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(filtered_steps, filtered_losses, 'r-', linewidth=1, alpha=0.8)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlim(left=start_step)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    # E[U] plot
    if filtered_utility:
        utility_steps = filtered_steps[:len(filtered_utility)]
        ax2.plot(utility_steps, filtered_utility, 'b-', linewidth=1, alpha=0.8)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('E[U]')
        ax2.set_title('Expected Utility')
        ax2.set_xlim(left=start_step)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

    plt.suptitle('Stochastic RAG Training Progress', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate all Stochastic RAG checkpoints")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing Stochastic RAG checkpoints",
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
        default="results/stochastic_rag_all_checkpoints",
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
        "--num_beams",
        type=int,
        default=4,
        help="Beam search width for generation",
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
        print(f"Found {len(checkpoints)} Stochastic RAG checkpoints in range {range_str}:")
    else:
        print(f"Found {len(checkpoints)} Stochastic RAG checkpoints:")

    for cp in checkpoints:
        has_reranker = os.path.exists(os.path.join(cp, "reranker.pt"))
        reranker_status = "[+reranker]" if has_reranker else "[no reranker]"
        print(f"  - {os.path.basename(cp)} {reranker_status}")

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
    training_plot_path = os.path.join(args.output_dir, "training_progress.png")

    if not tasks:
        all_results = cached_results
        # Generate plots from ALL existing results
        all_existing_results = load_all_eval_results(args.output_dir)
        print(f"\n[Plot] Generating plots from {len(all_existing_results)} total checkpoints...")
        plot_eval_progress(
            all_existing_results,
            eval_plot_path,
            start_step=0
        )
        plot_training_progress(
            args.checkpoint_dir,
            training_plot_path,
            start_step=0,
            end_step=None
        )
    else:
        # Assign GPUs round-robin
        tasks_with_gpu = [
            (cp, dp, op, i % num_gpus, args.num_beams)
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
                    all_existing_results = load_all_eval_results(args.output_dir)
                    print(f"\n[Plot] Generating updated plots with {len(all_existing_results)} total checkpoints...")
                    plot_eval_progress(
                        all_existing_results,
                        eval_plot_path,
                        start_step=0
                    )
                    plot_training_progress(
                        args.checkpoint_dir,
                        training_plot_path,
                        start_step=0,
                        end_step=None
                    )

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY: Stochastic RAG All Checkpoint Results")
    print("=" * 100)
    print(f"{'Checkpoint':<15} {'EM':>8} {'F1':>8} {'Prov':>8} {'SelProv':>8} {'KILT':>8}")
    print("-" * 100)

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
        sel_prov = r.get('selected_provenance_accuracy', 0)
        print(f"{name:<15} {r['answer_accuracy']:>7.2f}% {r['answer_f1']:>7.2f}% "
              f"{r['provenance_accuracy']:>7.2f}% {sel_prov:>7.2f}% {r['kilt_score']:>7.2f}%")

        if r['kilt_score'] > best_kilt:
            best_kilt = r['kilt_score']
            best_checkpoint = name

        if r['answer_accuracy'] > best_em:
            best_em = r['answer_accuracy']
            best_em_checkpoint = name

    print("=" * 100)
    print(f"\nBest KILT Score: {best_checkpoint} ({best_kilt:.2f}%)")
    print(f"Best EM Score:   {best_em_checkpoint} ({best_em:.2f}%)")

    # Save summary
    summary = {
        "model": "stochastic_rag",
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
        print("-" * 80)
        print(f"{'Task':<25} {'EM':>8} {'F1':>8} {'Prov':>8} {'SelProv':>8} {'KILT':>8}")
        print("-" * 80)
        for task, metrics in sorted(all_results[best_checkpoint]["per_task"].items()):
            sel_prov = metrics.get('selected_provenance_accuracy', 0)
            print(f"{task:<25} {metrics['answer_accuracy']:>7.2f}% {metrics['answer_f1']:>7.2f}% "
                  f"{metrics['provenance_accuracy']:>7.2f}% {sel_prov:>7.2f}% {metrics['kilt_score']:>7.2f}%")


if __name__ == "__main__":
    main()
