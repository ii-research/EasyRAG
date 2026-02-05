"""
Evaluate Panel Component
========================

UI component for model evaluation with algorithm/model selection.
Supports both single checkpoint and all checkpoints evaluation.
"""

from nicegui import ui
from typing import Callable, Dict, Any, Optional
from pathlib import Path
import subprocess
import asyncio
import json


# Algorithm and model options (same as step_dialog.py)
ALGORITHM_OPTIONS = {
    "fidlight": "FiD-Light",
    "fid_pure": "FiD Pure",
    "stochastic_rag": "Stochastic RAG",
}

MODEL_OPTIONS = {
    "t5base": "T5-base",
    "t5gemma": "T5Gemma2-540M",
}

EVAL_TYPE_OPTIONS = {
    "single": "Single Checkpoint",
    "all": "All Checkpoints",
}

# Evaluation script mapping (same as pipeline_orchestrator.py)
EVAL_SCRIPTS = {
    # Single checkpoint evaluation
    ("fidlight", "t5base", "single"): "evaluation/evaluate_fidlight.py",
    ("fidlight", "t5gemma", "single"): "evaluation/evaluate_fidlight_t5gemma.py",
    ("fid_pure", "t5base", "single"): "evaluation/evaluate_fid_pure.py",
    ("fid_pure", "t5gemma", "single"): "evaluation/evaluate_fid_pure_t5gemma.py",
    ("stochastic_rag", "t5base", "single"): "evaluation/evaluate_stochastic_rag.py",
    ("stochastic_rag", "t5gemma", "single"): "evaluation/evaluate_stochastic_rag_t5gemma.py",
    # All checkpoints evaluation
    ("fidlight", "t5base", "all"): "evaluation/evaluate_fidlight_t5base_all_checkpoints.py",
    ("fidlight", "t5gemma", "all"): "evaluation/evaluate_fidlight_t5gemma_all_checkpoints.py",
    ("fid_pure", "t5base", "all"): "evaluation/evaluate_fid_pure_all_checkpoints.py",
    ("fid_pure", "t5gemma", "all"): "evaluation/evaluate_fid_pure_all_checkpoints_t5gemma.py",
    ("stochastic_rag", "t5base", "all"): "evaluation/evaluate_stochastic_rag_all_checkpoints.py",
    ("stochastic_rag", "t5gemma", "all"): "evaluation/evaluate_stochastic_rag_t5gemma_all_checkpoints.py",
}


class EvaluatePanel:
    """
    Evaluation panel with algorithm/model selection.

    Features:
    - Algorithm selection (FiD-Light, FiD Pure, Stochastic RAG)
    - Model selection (T5-base, T5Gemma2)
    - Evaluation type (Single Checkpoint, All Checkpoints)
    - Path configuration
    - Progress display
    - Results visualization
    """

    def __init__(self):
        """Initialize the evaluate panel."""
        self.selected_algorithm = "fidlight"
        self.selected_model = "t5base"
        self.selected_eval_type = "single"

        # UI references
        self.checkpoint_input = None
        self.data_path_input = None
        self.output_dir_input = None
        self.start_step_input = None
        self.end_step_input = None
        self.status_label = None
        self.progress_bar = None
        self.results_container = None
        self.log_area = None

        # State
        self.is_running = False
        self.process = None

        self._build_ui()

    def _build_ui(self):
        """Build the evaluation panel UI."""
        with ui.card().classes("w-full"):
            ui.label("Model Evaluation").classes("text-h5 mb-4")

            # Algorithm and Model selection row
            with ui.row().classes("w-full gap-4 mb-4"):
                # Algorithm selection
                with ui.column().classes("flex-1"):
                    ui.label("Algorithm").classes("font-bold mb-2")
                    algo_options = list(ALGORITHM_OPTIONS.values())
                    ui.radio(
                        algo_options,
                        value=ALGORITHM_OPTIONS["fidlight"],
                        on_change=self._on_algorithm_change,
                    ).props("dense")

                # Model selection
                with ui.column().classes("flex-1"):
                    ui.label("Model").classes("font-bold mb-2")
                    model_options = list(MODEL_OPTIONS.values())
                    ui.radio(
                        model_options,
                        value=MODEL_OPTIONS["t5base"],
                        on_change=self._on_model_change,
                    ).props("dense")

                # Evaluation type selection
                with ui.column().classes("flex-1"):
                    ui.label("Evaluation Type").classes("font-bold mb-2")
                    eval_options = list(EVAL_TYPE_OPTIONS.values())
                    ui.radio(
                        eval_options,
                        value=EVAL_TYPE_OPTIONS["single"],
                        on_change=self._on_eval_type_change,
                    ).props("dense")

            ui.separator()

            # Path configuration
            with ui.column().classes("w-full gap-2 mt-4") as self.paths_container:
                self._build_single_checkpoint_paths()

            ui.separator().classes("mt-4")

            # Action buttons
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button(
                    "Start Evaluation",
                    icon="play_arrow",
                    on_click=self._start_evaluation,
                ).props("color=primary").bind_enabled_from(
                    self, "is_running", backward=lambda x: not x
                )
                ui.button(
                    "Stop",
                    icon="stop",
                    on_click=self._stop_evaluation,
                ).props("color=negative").bind_enabled_from(self, "is_running")

        # Status and results card
        with ui.card().classes("w-full mt-4"):
            ui.label("Status").classes("text-h6 mb-2")

            with ui.row().classes("w-full items-center gap-4"):
                self.status_label = ui.label("Ready").classes("text-lg")
                self.progress_bar = ui.linear_progress(value=0).classes("flex-grow")

            # Log output
            with ui.expansion("Logs", icon="article").classes("w-full mt-4"):
                self.log_area = ui.code("").classes("w-full h-64 overflow-auto")

            # Results display
            with ui.expansion("Results", icon="analytics").classes("w-full mt-2") as self.results_expansion:
                self.results_container = ui.column().classes("w-full")

    def _build_single_checkpoint_paths(self):
        """Build path inputs for single checkpoint evaluation."""
        self.paths_container.clear()

        with self.paths_container:
            ui.label("Paths Configuration").classes("font-bold mb-2")

            with ui.row().classes("w-full items-center gap-2 mb-2"):
                ui.label("Checkpoint:").classes("w-32")
                self.checkpoint_input = ui.input(
                    placeholder="checkpoints/fidlight_paper/final",
                    value="checkpoints/fidlight_paper/final"
                ).classes("flex-grow")

            with ui.row().classes("w-full items-center gap-2"):
                ui.label("Data Path:").classes("w-32")
                self.data_path_input = ui.input(
                    placeholder="kilt_data/precomputed/all_tasks_dev.parquet",
                    value="kilt_data/precomputed/all_tasks_dev.parquet"
                ).classes("flex-grow")

    def _build_all_checkpoints_paths(self):
        """Build path inputs for all checkpoints evaluation."""
        self.paths_container.clear()

        with self.paths_container:
            ui.label("Paths Configuration").classes("font-bold mb-2")

            with ui.row().classes("w-full items-center gap-2 mb-2"):
                ui.label("Checkpoint Dir:").classes("w-32")
                self.checkpoint_input = ui.input(
                    placeholder="checkpoints/fidlight_v5_bf16",
                    value="checkpoints/fidlight_v5_bf16"
                ).classes("flex-grow")

            with ui.row().classes("w-full items-center gap-2 mb-2"):
                ui.label("Data Path:").classes("w-32")
                self.data_path_input = ui.input(
                    placeholder="kilt_data/precomputed/all_tasks_dev.parquet",
                    value="kilt_data/precomputed/all_tasks_dev.parquet"
                ).classes("flex-grow")

            with ui.row().classes("w-full items-center gap-2 mb-2"):
                ui.label("Output Dir:").classes("w-32")
                self.output_dir_input = ui.input(
                    placeholder="results/evaluation",
                    value="results/evaluation"
                ).classes("flex-grow")

            with ui.row().classes("w-full items-center gap-4"):
                with ui.row().classes("items-center gap-2"):
                    ui.label("Start Step:").classes("w-20")
                    self.start_step_input = ui.number(
                        value=0, min=0
                    ).classes("w-24")

                with ui.row().classes("items-center gap-2"):
                    ui.label("End Step:").classes("w-20")
                    self.end_step_input = ui.number(
                        value=50000, min=0
                    ).classes("w-24")

    def _on_algorithm_change(self, e):
        """Handle algorithm selection change."""
        for key, label in ALGORITHM_OPTIONS.items():
            if label == e.value:
                self.selected_algorithm = key
                break

    def _on_model_change(self, e):
        """Handle model selection change."""
        for key, label in MODEL_OPTIONS.items():
            if label == e.value:
                self.selected_model = key
                break

    def _on_eval_type_change(self, e):
        """Handle evaluation type change."""
        for key, label in EVAL_TYPE_OPTIONS.items():
            if label == e.value:
                self.selected_eval_type = key
                break

        # Rebuild paths UI
        if self.selected_eval_type == "single":
            self._build_single_checkpoint_paths()
        else:
            self._build_all_checkpoints_paths()

    def _get_script_path(self) -> Optional[str]:
        """Get the evaluation script path for current selection."""
        key = (self.selected_algorithm, self.selected_model, self.selected_eval_type)
        script_name = EVAL_SCRIPTS.get(key)
        if script_name:
            return script_name
        return None

    def _build_command(self) -> list:
        """Build the evaluation command."""
        script_name = self._get_script_path()
        if not script_name:
            return []

        cmd = ["python", script_name]

        if self.selected_eval_type == "single":
            # Single checkpoint evaluation
            cmd.extend(["--checkpoint", self.checkpoint_input.value])
            cmd.extend(["--data_path", self.data_path_input.value])
        else:
            # All checkpoints evaluation
            cmd.extend(["--checkpoint_dir", self.checkpoint_input.value])
            cmd.extend(["--data_path", self.data_path_input.value])
            cmd.extend(["--output_dir", self.output_dir_input.value])
            if self.start_step_input and self.start_step_input.value > 0:
                cmd.extend(["--start_step", str(int(self.start_step_input.value))])
            if self.end_step_input and self.end_step_input.value > 0:
                cmd.extend(["--end_step", str(int(self.end_step_input.value))])

        return cmd

    async def _start_evaluation(self):
        """Start the evaluation process."""
        if self.is_running:
            ui.notify("Evaluation already running", type="warning")
            return

        script_name = self._get_script_path()
        if not script_name:
            ui.notify("No evaluation script found for this combination", type="negative")
            return

        # Validate inputs
        if not self.checkpoint_input.value:
            ui.notify("Please enter checkpoint path", type="warning")
            return

        cmd = self._build_command()
        if not cmd:
            ui.notify("Failed to build command", type="negative")
            return

        self.is_running = True
        self.status_label.text = f"Running: {script_name}"
        self.progress_bar.value = 0
        self.log_area.content = f"$ {' '.join(cmd)}\n\n"

        ui.notify(f"Starting evaluation: {script_name}", type="positive")

        try:
            # Start subprocess
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(Path(__file__).parent.parent.parent),
            )

            # Read output
            log_lines = []
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    break

                line_text = line.decode("utf-8", errors="replace")
                log_lines.append(line_text)

                # Keep last 200 lines
                if len(log_lines) > 200:
                    log_lines = log_lines[-200:]

                self.log_area.content = "".join(log_lines)

                # Try to parse progress
                self._parse_progress(line_text)

            await self.process.wait()

            # Update status
            if self.process.returncode == 0:
                self.status_label.text = "Completed"
                self.progress_bar.value = 1.0
                ui.notify("Evaluation completed successfully", type="positive")
                self._parse_results(log_lines)
            else:
                self.status_label.text = f"Failed (exit code: {self.process.returncode})"
                ui.notify("Evaluation failed", type="negative")

        except Exception as e:
            self.status_label.text = f"Error: {str(e)}"
            ui.notify(f"Error: {str(e)}", type="negative")
        finally:
            self.is_running = False
            self.process = None

    async def _stop_evaluation(self):
        """Stop the running evaluation."""
        if self.process:
            self.process.terminate()
            self.status_label.text = "Stopped"
            ui.notify("Evaluation stopped", type="warning")

    def _parse_progress(self, line: str):
        """Try to parse progress from log line."""
        import re

        # Look for progress patterns like "50%" or "100/200"
        percent_match = re.search(r'(\d+)%', line)
        if percent_match:
            self.progress_bar.value = int(percent_match.group(1)) / 100.0

        fraction_match = re.search(r'(\d+)/(\d+)', line)
        if fraction_match:
            current = int(fraction_match.group(1))
            total = int(fraction_match.group(2))
            if total > 0:
                self.progress_bar.value = current / total

    def _parse_results(self, log_lines: list):
        """Try to parse results from log output."""
        self.results_container.clear()

        with self.results_container:
            results_found = False

            for line in log_lines:
                # Look for KILT Score results
                if "KILT Score" in line or "Answer Accuracy" in line or "EM:" in line:
                    ui.label(line.strip()).classes("font-mono text-sm")
                    results_found = True
                elif "Provenance" in line or "F1:" in line:
                    ui.label(line.strip()).classes("font-mono text-sm")
                    results_found = True

            if not results_found:
                ui.label("No results parsed. Check logs for details.").classes("text-gray-500 dark:text-gray-400")
