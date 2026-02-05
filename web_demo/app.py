"""
FiDLight Web Demo Main Application
====================================

NiceGUI-based dashboard for monitoring and controlling
the FiD-Light training pipeline.

Usage:
    python -m web_demo.app

Features:
    - 8-step pipeline visualization
    - Real-time progress monitoring
    - Log viewing
    - Loss curve charts
    - Step configuration dialogs
    - Inference demonstration
"""

import asyncio
import re
import os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from nicegui import ui, app

from .pipeline_orchestrator import (
    PipelineOrchestrator,
    get_orchestrator,
    STEP_CONFIGS,
)
from .utils.state_io import (
    read_pipeline_state,
    PipelineState,
    StepStatus,
    PipelineStatus,
    WorkspaceConfig,
    set_current_workspace,
    get_current_workspace,
)
from .components.pipeline_overview import PipelineOverview
from .components.step_dialog import StepConfigDialog, SkipStepDialog, TrainModelConfigDialog, KiltDataConfigDialog
from .components.log_viewer import LogViewer, LossChart
from .components.inference_panel import InferencePanel
from .components.evaluate_panel import EvaluatePanel
from .components.workspace_selector import WorkspaceSelector
from .components.compare_panel import ComparePanel
from .components.web_rag_panel import WebRAGPanel


# ============================================================================
# Step Completion Data Preview Functions
# ============================================================================

def get_step_completion_extra(step_id: int, state) -> dict:
    """
    Get extra data to show in Step Results after a step completes.
    Reads output files and returns preview information.

    Args:
        step_id: The completed step ID
        state: Current pipeline state

    Returns:
        Dict with preview data (found_items, samples, etc.)
    """
    import json
    extra = {}

    # Get the step's output path from previous steps or config
    step = state.get_step(step_id)
    if not step:
        return extra

    try:
        if step_id == 4:  # Fix TriviaQA
            # Look for triviaqa_fixed files
            # Try to find from step 4's output or common locations
            possible_paths = [
                Path("/data3/iiserver32/ZHOUXUANCHEN/kilt/triviaqa_fixed"),
                Path("kilt_data/triviaqa_fixed"),
            ]

            for path in possible_paths:
                if path.exists():
                    found_items = []
                    for split in ["train", "validation", "test"]:
                        f = path / f"triviaqa_support_only_{split}.jsonl"
                        if f.exists():
                            size_mb = f.stat().st_size / (1024 * 1024)
                            found_items.append({"name": f"TriviaQA {split.title()}", "info": f"{size_mb:.1f} MB"})

                    if found_items:
                        extra["found_items"] = found_items

                        # Read a sample
                        train_file = path / "triviaqa_support_only_train.jsonl"
                        if train_file.exists():
                            with open(train_file, 'r') as f:
                                line = f.readline()
                                sample = json.loads(line)
                                q = sample.get('input', '')[:60]
                                a = sample.get('output', [{}])[0].get('answer', '')[:40] if sample.get('output') else ''
                                extra["task_samples"] = [{
                                    "task": "TriviaQA",
                                    "question": q + "..." if len(sample.get('input', '')) > 60 else q,
                                    "answer": a
                                }]
                    break

        elif step_id == 5:  # Filter Data
            possible_paths = [
                Path("kilt_data/filtered"),
                Path("/data3/iiserver32/ZHOUXUANCHEN/kilt/filtered"),
            ]

            for path in possible_paths:
                if path.exists():
                    found_items = []
                    for task, name in [("nq", "NQ"), ("hotpotqa", "HotpotQA"), ("triviaqa_support_only", "TriviaQA")]:
                        train_f = path / f"{task}_train.jsonl"
                        val_f = path / f"{task}_validation.jsonl"
                        if train_f.exists() or val_f.exists():
                            size = 0
                            count = 0
                            if train_f.exists():
                                size += train_f.stat().st_size
                                count += 1
                            if val_f.exists():
                                size += val_f.stat().st_size
                                count += 1
                            found_items.append({"name": name, "info": f"{count} file(s), {size/(1024*1024):.1f} MB"})

                    if found_items:
                        extra["found_items"] = found_items
                    break

        elif step_id == 11:  # Precompute Retrieval
            import pyarrow.parquet as pq

            # Try to find precomputed output directory
            output_path = step.output_path if step.output_path else None
            if not output_path:
                # Try common locations
                possible_paths = [
                    Path("/data3/iiserver32/ZHOUXUANCHEN/kilt/precomputed"),
                    Path("kilt_data/precomputed"),
                ]
                for p in possible_paths:
                    if p.exists():
                        output_path = str(p)
                        break

            if output_path:
                path = Path(output_path)
                if path.exists():
                    found_items = []
                    total_samples = 0

                    # Check for parquet files
                    for split, suffix in [("Train", "train"), ("Validation", "dev")]:
                        combined_file = path / f"all_tasks_{suffix}.parquet"
                        if combined_file.exists():
                            try:
                                table = pq.read_table(combined_file)
                                num_samples = len(table)
                                size_mb = combined_file.stat().st_size / (1024 * 1024)
                                found_items.append({
                                    "name": f"{split} Set",
                                    "info": f"{num_samples:,} samples ({size_mb:.1f} MB)"
                                })
                                total_samples += num_samples
                            except Exception:
                                pass

                    # Per-task files
                    for task in ["nq", "hotpotqa", "triviaqa_support_only"]:
                        train_f = path / f"{task}_train.parquet"
                        dev_f = path / f"{task}_dev.parquet"
                        if train_f.exists() or dev_f.exists():
                            task_name = {"nq": "NQ", "hotpotqa": "HotpotQA", "triviaqa_support_only": "TriviaQA"}.get(task, task)
                            count = sum(1 for f in [train_f, dev_f] if f.exists())
                            found_items.append({"name": f"  {task_name}", "info": f"{count} split(s)"})

                    if found_items:
                        extra["found_items"] = found_items
                        extra["total_samples"] = total_samples

                    # Read a sample from train
                    train_file = path / "all_tasks_train.parquet"
                    if train_file.exists():
                        try:
                            table = pq.read_table(train_file)
                            if len(table) > 0:
                                row = table.slice(0, 1).to_pydict()
                                extra["precompute_sample"] = {
                                    "query": row["query"][0][:80] + "..." if len(row["query"][0]) > 80 else row["query"][0],
                                    "answer": row["answer"][0][:50] if row.get("answer") else "",
                                    "num_passages": len(row["input_texts"][0]) if row.get("input_texts") else 0,
                                }
                        except Exception:
                            pass

    except Exception as e:
        print(f"Error getting step completion extra: {e}")

    return extra


# ============================================================================
# Loss Chart Helper Functions
# ============================================================================

def load_loss_history_from_checkpoint(output_dir: str) -> Tuple[List[int], List[float]]:
    """
    Load loss history from the latest checkpoint in output_dir.

    Args:
        output_dir: Training output directory (e.g., checkpoints/fidlight_paper/)

    Returns:
        Tuple of (steps, losses) lists
    """
    if not output_dir or not os.path.exists(output_dir):
        return [], []

    # Find checkpoint directories (step_XXXXX)
    checkpoint_dirs = []
    for name in os.listdir(output_dir):
        if name.startswith("step_") and os.path.isdir(os.path.join(output_dir, name)):
            try:
                step_num = int(name.split("_")[1])
                checkpoint_dirs.append((step_num, name))
            except (ValueError, IndexError):
                continue

    if not checkpoint_dirs:
        return [], []

    # Sort by step number and get the latest
    checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_step, latest_dir = checkpoint_dirs[0]
    checkpoint_path = os.path.join(output_dir, latest_dir)

    # Try to load loss_history.npy
    npy_path = os.path.join(checkpoint_path, "loss_history.npy")
    if os.path.exists(npy_path):
        try:
            losses = np.load(npy_path).tolist()
            # Generate step numbers (1, 2, 3, ...)
            steps = list(range(1, len(losses) + 1))
            return steps, losses
        except Exception as e:
            print(f"Failed to load loss_history.npy: {e}")

    # Fallback: try training_state.json
    json_path = os.path.join(checkpoint_path, "training_state.json")
    if os.path.exists(json_path):
        try:
            import json
            with open(json_path, "r") as f:
                state = json.load(f)
            losses = state.get("loss_history", [])
            steps = list(range(1, len(losses) + 1))
            return steps, losses
        except Exception as e:
            print(f"Failed to load training_state.json: {e}")

    return [], []


def parse_loss_from_log_line(line: str) -> Tuple[Optional[int], Optional[float]]:
    """
    Parse step and loss from a training log line.

    Expected format: "Training: ...| 61/50000 [..., loss=2.3256, ...]"

    Returns:
        Tuple of (step, loss) or (None, None) if not a training line
    """
    # Only match Training progress lines (not eval progress)
    if "Training:" not in line or "loss=" not in line:
        return None, None

    # Extract step number: "| 61/50000" pattern
    step_match = re.search(r'\|\s*(\d+)/\d+', line)
    if not step_match:
        return None, None

    # Extract loss value: "loss=2.3256" or "loss=nan" or "loss=inf" pattern
    loss_match = re.search(r'loss=([\d.eE+-]+|nan|inf)', line, re.IGNORECASE)
    if not loss_match:
        return None, None

    try:
        step = int(step_match.group(1))
        loss_str = loss_match.group(1).lower()
        if loss_str == 'nan':
            loss = float('nan')
        elif loss_str == 'inf':
            loss = float('inf')
        else:
            loss = float(loss_str)
        return step, loss
    except (ValueError, IndexError):
        return None, None


# Global state for tracking last parsed step (avoid duplicates)
last_parsed_step = [0]


# Global state
orchestrator: Optional[PipelineOrchestrator] = None
pipeline_overview: Optional[PipelineOverview] = None
log_viewer: Optional[LogViewer] = None
loss_chart: Optional[LossChart] = None
workspace_selector: Optional[WorkspaceSelector] = None


def on_start_step(step_id: int):
    """Handle start step button click."""
    config = orchestrator.get_step_config(step_id)
    if not config:
        ui.notify(f"Unknown step: {step_id}", type="negative")
        return

    # Check dependencies
    can_start, reason = orchestrator.can_start_step(step_id)
    if not can_start:
        ui.notify(reason, type="warning")
        return

    # Steps 2-5 (KILT data steps): Use special KILT data dialog
    # 2: download wiki, 3: download tasks, 4: fix_triviaqa, 5: filter
    if step_id in [2, 3, 4, 5]:
        dialog = KiltDataConfigDialog(
            step_id=step_id,
            on_confirm=on_step_config_confirm,
            on_skip=on_skip_confirm,
        )
        dialog.show()
        return

    # Step 12: Use special dialog for algorithm/model selection
    if step_id == 12:
        dialog = TrainModelConfigDialog(on_confirm=on_step_config_confirm)
        dialog.show()
        return

    # Other steps: Use standard config dialog (with "Use Existing" tab)
    dialog = StepConfigDialog(
        step_id=step_id,
        display_name=config.display_name,
        default_args=config.default_args,
        required_paths=config.required_paths,
        description=config.description,
        on_confirm=on_step_config_confirm,
        on_skip=on_skip_confirm,
    )
    dialog.show()


def on_step_config_confirm(step_id: int, args: dict):
    """Handle step config dialog confirmation."""
    success, message = orchestrator.start_step(step_id, args)
    if success:
        ui.notify(f"Step {step_id} started: {message}", type="positive")
    else:
        ui.notify(f"Failed to start step: {message}", type="negative")


def on_skip_step(step_id: int):
    """Handle skip step button click."""
    config = orchestrator.get_step_config(step_id)
    if not config:
        ui.notify(f"Unknown step: {step_id}", type="negative")
        return

    dialog = SkipStepDialog(
        step_id=step_id,
        display_name=config.display_name,
        on_confirm=on_skip_confirm,
    )
    dialog.show()


def on_skip_confirm(step_id: int, output_path: str, extra: dict = None):
    """Handle skip dialog confirmation."""
    success, message = orchestrator.skip_step(step_id, output_path, extra=extra)
    if success:
        ui.notify(message, type="info")
    else:
        ui.notify(f"Failed to skip step: {message}", type="negative")


def on_start_all():
    """Handle start all button click."""
    ui.notify("Full pipeline execution not yet implemented", type="warning")
    # TODO: Show a dialog to collect all step configs, then run sequentially


def on_stop_all():
    """Handle stop all button click."""
    success, message = orchestrator.stop_pipeline()
    if success:
        ui.notify("Pipeline stopped", type="info")
    else:
        ui.notify(f"Failed to stop: {message}", type="negative")


def on_reset_step(step_id: int):
    """Handle reset step button click."""
    success, message = orchestrator.reset_step(step_id)
    if success:
        ui.notify(message, type="info")
    else:
        ui.notify(f"Reset failed: {message}", type="negative")


def on_stop_step(step_id: int):
    """Handle stop step button click."""
    success, message = orchestrator.stop_step(step_id)
    if success:
        ui.notify("Step stopped", type="info")
    else:
        ui.notify(f"Stop failed: {message}", type="negative")


def on_workspace_change(config: WorkspaceConfig):
    """Handle workspace change."""
    global orchestrator
    if orchestrator:
        # Re-initialize orchestrator with new workspace
        orchestrator.set_workspace(config)
        ui.notify(f"Workspace updated", type="info")


@ui.page("/")
def main_page():
    """Main dashboard page."""
    global orchestrator, pipeline_overview, log_viewer, loss_chart
    global workspace_selector

    # Dark mode toggle
    dark = ui.dark_mode()
    dark.value = False

    # Header
    with ui.header().classes("items-center justify-between"):
        with ui.row().classes("items-center gap-4"):
            ui.label("EasyRAG Dashboard").classes("text-h5 text-white")
            ui.link("Evaluate", "/evaluate").classes(
                "text-white no-underline ml-4"
            )
            ui.link("Inference", "/inference").classes(
                "text-white no-underline ml-2"
            )
            ui.link("Compare", "/compare").classes(
                "text-white no-underline ml-2"
            )
            ui.link("Live Demo", "/live-demo").classes(
                "text-white no-underline ml-2"
            )

        ui.button(
            icon="dark_mode",
            on_click=lambda: dark.toggle(),
        ).props("flat color=white")

    # Main container with workspace selector at top
    with ui.column().classes("w-full h-screen"):
        # Workspace selector (at top)
        with ui.element("div").classes("w-full px-4 pt-4"):
            workspace_selector = WorkspaceSelector(
                on_workspace_change=on_workspace_change,
            )

        # Initialize orchestrator with workspace
        orchestrator = get_orchestrator()
        workspace_config = workspace_selector.get_config()
        if workspace_config:
            orchestrator.set_workspace(workspace_config)

        # Only initialize if no existing state, step count changed, or step definitions changed
        existing_state = read_pipeline_state(orchestrator.state_file)
        need_reinit = False
        if existing_state is None:
            need_reinit = True
        elif len(existing_state.steps) != 12:
            need_reinit = True
        else:
            # Check if step names match current definitions (detect swapped steps)
            from .utils.state_io import STEP_DEFINITIONS
            for i, step in enumerate(existing_state.steps):
                expected = STEP_DEFINITIONS[i]
                if step.name != expected["name"]:
                    need_reinit = True
                    print(f"Step definition changed: {step.name} -> {expected['name']}, reinitializing...")
                    break

        if need_reinit:
            orchestrator.initialize("production")

        # Main content splitter
        with ui.splitter(value=35).classes("w-full flex-grow") as splitter:
            # Left panel: Pipeline overview
            with splitter.before:
                with ui.column().classes("w-full p-4"):
                    pipeline_overview = PipelineOverview(
                        on_start_step=on_start_step,
                        on_skip_step=on_skip_step,
                        on_start_all=on_start_all,
                        on_stop_all=on_stop_all,
                        on_reset_step=on_reset_step,
                        on_stop_step=on_stop_step,
                    )

            # Right panel: Details
            with splitter.after:
                with ui.column().classes("w-full p-4 h-full"):
                    # Tabs for logs and charts (top)
                    with ui.card().classes("w-full mb-4"):
                        with ui.tabs().classes("w-full") as tabs:
                            log_tab = ui.tab("Logs", icon="terminal")
                            chart_tab = ui.tab("Loss Chart", icon="show_chart")

                        with ui.tab_panels(tabs, value=log_tab).classes("w-full"):
                            with ui.tab_panel(log_tab):
                                log_viewer = LogViewer()

                            with ui.tab_panel(chart_tab):
                                loss_chart = LossChart()

                    # Step result area (bottom) - dynamically updated
                    with ui.card().classes("w-full flex-grow"):
                        with ui.row().classes("items-center gap-2 mb-2"):
                            ui.icon("info").classes("text-xl text-green-500")
                            ui.label("Step Results").classes("text-h6")
                        step_result_container = ui.column().classes("w-full gap-2")
                        with step_result_container:
                            ui.label("Select a step to begin").classes("text-gray-500 dark:text-gray-400 w-full")

    # Track last state hash for efficient updates
    last_state_hash = [""]
    # Track last training output_dir to detect new training sessions
    last_training_output_dir = [""]

    # Start background refresh using ui.timer (runs every 2 seconds)
    async def do_refresh():
        state = orchestrator.get_state()

        # Find running step
        running_step = None
        for step in state.steps:
            if step.status == StepStatus.RUNNING.value:
                running_step = step
                break

        # Check if running step's process has actually finished
        if running_step and not orchestrator.is_step_running(running_step.id):
            # Process ended - check return code to determine success/failure
            config = orchestrator.get_step_config(running_step.id)
            if config:
                process_info = orchestrator.process_manager.get_process_info(config.name)
                if process_info:
                    # return_code is 0 for success, None means not yet set (treat as success if process stopped)
                    # negative values or positive non-zero values indicate failure
                    return_code = process_info.return_code
                    if return_code is None or return_code == 0:
                        # Success
                        from .utils.state_io import mark_step_completed, update_step_state

                        # Get output_path from step config
                        output_path = None
                        if config.output_path_key and running_step.config:
                            output_path = running_step.config.get(config.output_path_key)
                        mark_step_completed(config.name, output_path=output_path, state_file=orchestrator.state_file)

                        # Get and save preview data for Step Results display
                        # Re-read state to get updated output_path
                        state = orchestrator.get_state()
                        running_step = state.get_step(running_step.id)
                        extra = get_step_completion_extra(running_step.id, state)
                        if extra:
                            update_step_state(
                                step_name=config.name,
                                extra=extra,
                                state_file=orchestrator.state_file,
                            )

                        ui.notify(f"Step {running_step.id} completed!", type="positive")
                    else:
                        # Failed
                        from .utils.state_io import mark_step_failed
                        mark_step_failed(config.name, f"Process exited with code {return_code}", state_file=orchestrator.state_file)
                        ui.notify(f"Step {running_step.id} failed!", type="negative")
                    # Refresh state after update
                    state = orchestrator.get_state()
                    running_step = None

        if pipeline_overview:
            pipeline_overview.update_from_state(state)

        # Create state hash to detect changes
        state_hash = "|".join([f"{s.id}:{s.status}" for s in state.steps])
        if state_hash != last_state_hash[0]:
            last_state_hash[0] = state_hash
            step_result_container.clear()
            with step_result_container:
                _render_all_step_results(state.steps)

        # Update log if there's a running step or recently completed step
        lines = []
        if log_viewer:
            # Find step to show logs for (running or most recently completed)
            log_step = running_step
            if not log_step:
                # Find the most recently completed/failed step that has logs
                for step in reversed(state.steps):
                    if step.status in [StepStatus.COMPLETED.value, StepStatus.FAILED.value]:
                        log_step = step
                        break

            if log_step:
                lines = orchestrator.get_step_output(log_step.id, tail_lines=100)
                if lines:
                    log_viewer.set_lines(lines)

        # Update loss chart for training steps (step_id=9 or 12)
        if loss_chart and running_step and running_step.id in [9, 12]:
            output_dir = running_step.extra.get("output_dir") if running_step.extra else None

            # Detect new training session (different output_dir)
            if output_dir and output_dir != last_training_output_dir[0]:
                # New training session - clear chart and load history
                loss_chart.clear()
                last_parsed_step[0] = 0
                last_training_output_dir[0] = output_dir

                # Load history from checkpoint (for resume)
                hist_steps, hist_losses = load_loss_history_from_checkpoint(output_dir)
                if hist_steps:
                    loss_chart.set_data(hist_steps, hist_losses)
                    last_parsed_step[0] = hist_steps[-1]

            # Parse new loss values from log lines
            if lines:
                for line in lines:
                    step, loss = parse_loss_from_log_line(line)
                    if step is not None and loss is not None:
                        # Only add if it's a new step (avoid duplicates)
                        if step > last_parsed_step[0]:
                            loss_chart.add_point(step, loss)
                            last_parsed_step[0] = step
        elif loss_chart and (not running_step or running_step.id not in [9, 12]):
            # Not training - reset tracking for next training session
            if last_training_output_dir[0]:
                last_training_output_dir[0] = ""

    def _render_all_step_results(steps):
        """Render results for all completed/skipped/running steps."""
        has_content = False

        for step in steps:
            if step.status in [StepStatus.COMPLETED.value, StepStatus.SKIPPED.value, StepStatus.RUNNING.value]:
                has_content = True
                _render_step_result(step)
                ui.separator().classes("w-full my-2")

        if not has_content:
            ui.label("Click a step on the left to start").classes("text-gray-500 dark:text-gray-400")

    def _render_step_result(step):
        """Render single step result."""
        status_icons = {
            StepStatus.COMPLETED.value: ("✅", "text-green-600 dark:text-green-400", "bg-green-50 dark:bg-green-900/30"),
            StepStatus.RUNNING.value: ("🔄", "text-blue-600 dark:text-blue-400", "bg-blue-50 dark:bg-blue-900/30"),
            StepStatus.FAILED.value: ("❌", "text-red-600 dark:text-red-400", "bg-red-50 dark:bg-red-900/30"),
            StepStatus.SKIPPED.value: ("⏭️", "text-orange-500 dark:text-orange-400", "bg-orange-50 dark:bg-orange-900/30"),
        }
        icon, color, bg = status_icons.get(step.status, ("⏳", "text-gray-500 dark:text-gray-400", "bg-gray-50 dark:bg-gray-800"))

        with ui.element("div").classes(f"w-full p-3 rounded {bg}"):
            # Header row
            with ui.row().classes("items-center gap-2"):
                ui.label(f"{icon} {step.display_name}").classes(f"font-bold {color}")
                if step.status == StepStatus.RUNNING.value:
                    ui.label(f"{step.progress:.0f}%").classes("text-xs text-blue-600 ml-auto")

            # Message
            if step.message:
                ui.label(step.message).classes("text-xs text-gray-600 dark:text-gray-400 mt-1")

            # Step-specific content
            if step.id == 1 and step.extra and "details" in step.extra:
                # Environment check - show first 3 details
                for detail in step.extra["details"][:3]:
                    ui.label(f"• {detail}").classes("text-xs text-gray-500 dark:text-gray-400")

            elif step.id in [2, 3, 4, 5]:  # KILT data steps
                if step.status == StepStatus.SKIPPED.value:
                    ui.label("📂 Using existing data").classes("text-xs text-orange-600 dark:text-orange-400")
                elif step.status == StepStatus.COMPLETED.value:
                    ui.label("✅ Data processed").classes("text-xs text-green-600 dark:text-green-400")

                # Show found items for both SKIPPED and COMPLETED
                if step.extra and "found_items" in step.extra:
                    for item in step.extra["found_items"][:4]:  # Show max 4 items
                        ui.label(f"  • {item['name']}: {item['info']}").classes("text-xs text-gray-600 dark:text-gray-400")

                # Show Wikipedia samples for step 2
                if step.id == 2 and step.extra and "wiki_samples" in step.extra:
                    ui.label("📖 Wikipedia Samples:").classes("text-xs font-bold text-gray-700 dark:text-gray-300 mt-2")
                    for sample in step.extra["wiki_samples"][:3]:
                        ui.label(f"  📄 {sample['title']}").classes("text-xs text-blue-600 dark:text-blue-400")
                        if sample.get('text'):
                            ui.label(f"     {sample['text']}").classes("text-xs text-gray-500 dark:text-gray-400")

                # Show task samples for step 3 and 4
                if step.id in [3, 4] and step.extra and "task_samples" in step.extra:
                    ui.label("💬 QA Samples:").classes("text-xs font-bold text-gray-700 dark:text-gray-300 mt-2")
                    for sample in step.extra["task_samples"][:3]:
                        ui.label(f"  [{sample['task']}] Q: {sample['question']}").classes("text-xs text-blue-600 dark:text-blue-400")
                        if sample.get('answer'):
                            ui.label(f"     A: {sample['answer']}").classes("text-xs text-green-600 dark:text-green-400")

                if step.output_path:
                    # Show shortened path
                    path_display = step.output_path
                    if len(path_display) > 40:
                        path_display = "..." + path_display[-37:]
                    ui.label(f"📁 {path_display}").classes("text-xs text-gray-500 dark:text-gray-400")

            elif step.id == 11:  # Precompute Retrieval
                if step.status == StepStatus.SKIPPED.value:
                    ui.label("📂 Using existing precomputed data").classes("text-xs text-orange-600 dark:text-orange-400")
                elif step.status == StepStatus.COMPLETED.value:
                    ui.label("✅ Precomputation complete").classes("text-xs text-green-600 dark:text-green-400")

                # Show found items
                if step.extra and "found_items" in step.extra:
                    for item in step.extra["found_items"][:6]:
                        ui.label(f"  • {item['name']}: {item['info']}").classes("text-xs text-gray-600 dark:text-gray-400")

                # Show total samples
                if step.extra and "total_samples" in step.extra:
                    ui.label(f"📊 Total: {step.extra['total_samples']:,} samples").classes("text-xs font-bold text-blue-600 dark:text-blue-400 mt-1")

                # Show sample
                if step.extra and "precompute_sample" in step.extra:
                    sample = step.extra["precompute_sample"]
                    ui.label("📝 Sample:").classes("text-xs font-bold text-gray-700 dark:text-gray-300 mt-2")
                    ui.label(f"  Q: {sample['query']}").classes("text-xs text-blue-600 dark:text-blue-400")
                    if sample.get('answer'):
                        ui.label(f"  A: {sample['answer']}").classes("text-xs text-green-600 dark:text-green-400")
                    if sample.get('num_passages'):
                        ui.label(f"  📚 {sample['num_passages']} passages retrieved").classes("text-xs text-gray-500 dark:text-gray-400")

                if step.output_path:
                    path_display = step.output_path
                    if len(path_display) > 40:
                        path_display = "..." + path_display[-37:]
                    ui.label(f"📁 {path_display}").classes("text-xs text-gray-500 dark:text-gray-400")

            elif step.id == 12 and step.extra:  # Train model
                info_parts = []
                if "algorithm" in step.extra:
                    info_parts.append(step.extra["algorithm"])
                if "model" in step.extra:
                    info_parts.append(step.extra["model"])
                if info_parts:
                    ui.label(" | ".join(info_parts)).classes("text-xs text-gray-600 dark:text-gray-400")
                if "loss" in step.extra:
                    ui.label(f"Loss: {step.extra['loss']:.4f}").classes("text-xs font-mono text-gray-600 dark:text-gray-400")

    ui.timer(2.0, do_refresh)


@ui.page("/evaluate")
def evaluate_page():
    """Model evaluation page."""
    # Dark mode toggle
    dark = ui.dark_mode()
    dark.value = False

    # Header
    with ui.header().classes("items-center justify-between"):
        with ui.row().classes("items-center gap-4"):
            ui.link("Dashboard", "/").classes("text-white no-underline")
            ui.link("Inference", "/inference").classes("text-white no-underline ml-2")
            ui.link("Compare", "/compare").classes("text-white no-underline ml-2")
            ui.link("Live Demo", "/live-demo").classes("text-white no-underline ml-2")
            ui.label("|").classes("text-white")
            ui.label("Evaluate").classes("text-h5 text-white")

        ui.button(
            icon="dark_mode",
            on_click=lambda: dark.toggle(),
        ).props("flat color=white")

    # Main content - EvaluatePanel
    with ui.column().classes("w-full max-w-5xl mx-auto p-4"):
        EvaluatePanel()


@ui.page("/inference")
def inference_page():
    """Inference demonstration page."""
    # Dark mode toggle
    dark = ui.dark_mode()
    dark.value = False

    # Header
    with ui.header().classes("items-center justify-between"):
        with ui.row().classes("items-center gap-4"):
            ui.link("Dashboard", "/").classes("text-white no-underline")
            ui.link("Evaluate", "/evaluate").classes("text-white no-underline ml-2")
            ui.link("Compare", "/compare").classes("text-white no-underline ml-2")
            ui.link("Live Demo", "/live-demo").classes("text-white no-underline ml-2")
            ui.label("|").classes("text-white")
            ui.label("Inference").classes("text-h5 text-white")

        ui.button(
            icon="dark_mode",
            on_click=lambda: dark.toggle(),
        ).props("flat color=white")

    # Main content - Production mode paths
    with ui.column().classes("w-full max-w-4xl mx-auto p-4"):
        InferencePanel(
            default_checkpoint="checkpoints/fidlight_paper/final",
            default_index="kilt_data/gtr_faiss_index_finetuned",
            default_wiki="",  # Not needed for full retriever
        )


@ui.page("/compare")
def compare_page():
    """Model comparison page."""
    # Dark mode toggle
    dark = ui.dark_mode()
    dark.value = False

    # Header
    with ui.header().classes("items-center justify-between"):
        with ui.row().classes("items-center gap-4"):
            ui.link("Dashboard", "/").classes("text-white no-underline")
            ui.link("Evaluate", "/evaluate").classes("text-white no-underline ml-2")
            ui.link("Inference", "/inference").classes("text-white no-underline ml-2")
            ui.link("Live Demo", "/live-demo").classes("text-white no-underline ml-2")
            ui.label("|").classes("text-white")
            ui.label("Compare").classes("text-h5 text-white")

        ui.button(
            icon="dark_mode",
            on_click=lambda: dark.toggle(),
        ).props("flat color=white")

    # Main content - ComparePanel
    with ui.column().classes("w-full max-w-6xl mx-auto p-4"):
        ComparePanel(
            default_retriever="",
            default_index="kilt_data/gtr_faiss_index_finetuned",
        )


@ui.page("/live-demo")
def live_demo_page():
    """Live Demo demo page - simple RAG with web search."""
    # Dark mode toggle
    dark = ui.dark_mode()
    dark.value = False

    # Header
    with ui.header().classes("items-center justify-between"):
        with ui.row().classes("items-center gap-4"):
            ui.link("Dashboard", "/").classes("text-white no-underline")
            ui.link("Evaluate", "/evaluate").classes("text-white no-underline ml-2")
            ui.link("Inference", "/inference").classes("text-white no-underline ml-2")
            ui.link("Compare", "/compare").classes("text-white no-underline ml-2")
            ui.label("|").classes("text-white")
            ui.label("Live Demo").classes("text-h5 text-white")

        ui.button(
            icon="dark_mode",
            on_click=lambda: dark.toggle(),
        ).props("flat color=white")

    # Main content - WebRAGPanel
    with ui.column().classes("w-full max-w-5xl mx-auto p-4"):
        WebRAGPanel()


def main():
    """Run the web demo."""
    ui.run(
        title="EasyRAG Dashboard",
        port=8080,
        reload=True,  # Dev mode: auto-restart on code changes
        show=True,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
