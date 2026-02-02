"""
Step Configuration Dialog
==========================

Modal dialog for configuring step parameters before starting.
- Shows paper default values
- Allows path selection
- Validates inputs
- Special handling for Step 8 (algorithm/model selection)
"""

from nicegui import ui
from typing import Callable, Dict, Any, Optional, List
from pathlib import Path


# Algorithm and model options
ALGORITHM_OPTIONS = {
    "fidlight": "FiD-Light (Paper Method)",
    "fid_pure": "FiD Pure (No Source Pointer)",
    "stochastic_rag": "Stochastic RAG (Sampling)",
}

MODEL_OPTIONS = {
    "t5base": "T5-base (Paper Default)",
    "t5gemma": "T5Gemma2-540M (Enhanced)",
}

# Default args for different algorithm/model combinations
# These match the actual argparse defaults in each training script
# Note: multi_gpu defaults to True (user can disable), bf16 depends on model type
TRAIN_DEFAULT_ARGS = {
    # FiD-Light T5-base (train_fidlight_paper.py defaults)
    ("fidlight", "t5base"): {
        "learning_rate": 1e-3,
        "total_steps": 50000,
        "compression_k": 64,
        "num_passages": 40,
        "temperature": 2.0,
        "batch_size": 1,
        "gradient_accumulation_steps": 128,
        "warmup_steps": 0,
        "save_steps": 5000,
        "eval_steps": 2500,
        "eval_samples": 100,
        "multi_gpu": True,
        "bf16": True,
    },
    # FiD-Light T5Gemma2 (train_fidlight_t5gemma.py defaults)
    ("fidlight", "t5gemma"): {
        "learning_rate": 1e-3,
        "total_steps": 50000,
        "compression_k": 64,
        "num_passages": 40,
        "temperature": 2.0,
        "batch_size": 1,
        "gradient_accumulation_steps": 128,
        "warmup_steps": 100,
        "optimizer": "adafactor",
        "weight_decay": 0.01,
        "save_steps": 5000,
        "eval_steps": 2500,
        "eval_samples": 100,
        "multi_gpu": True,
        "bf16": True,
    },
    # FiD Pure T5-base (train_fid_pure.py defaults - original FiD paper settings)
    ("fid_pure", "t5base"): {
        "learning_rate": 1e-4,
        "total_steps": 10000,
        "compression_k": 250,
        "num_passages": 100,
        "temperature": 2.0,
        "dropout": 0.1,
        "batch_size": 1,
        "gradient_accumulation_steps": 64,
        "warmup_steps": 0,
        "save_steps": 1000,
        "eval_steps": 500,
        "eval_samples": 100,
        "multi_gpu": True,
        "bf16": True,
    },
    # FiD Pure T5Gemma2 (train_fid_pure_t5gemma.py defaults)
    ("fid_pure", "t5gemma"): {
        "model_name": "google/t5gemma-2-270m-270m",  # Required for this script!
        "learning_rate": 1e-5,
        "total_steps": 10000,
        "compression_k": 250,
        "num_passages": 100,
        "temperature": 2.0,
        "batch_size": 1,
        "gradient_accumulation_steps": 64,
        "warmup_steps": 100,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "save_steps": 1000,
        "eval_steps": 500,
        "eval_samples": 100,
        "multi_gpu": True,
        "bf16": True,
    },
    # Stochastic RAG T5-base (train_stochastic_rag.py defaults)
    # Note: Uses lr_generator/lr_reranker instead of learning_rate
    ("stochastic_rag", "t5base"): {
        "lr_generator": 1e-3,
        "lr_reranker": 1e-4,
        "total_steps": 50000,
        "compression_k": 64,
        "n_candidates": 40,
        "num_passages": 10,
        "gumbel_tau": 1.0,
        "num_utility_samples": 10,
        "scoring_type": "linear",
        "temperature": 2.0,
        "batch_size": 1,
        "gradient_accumulation_steps": 128,
        "warmup_steps": 0,
        "save_steps": 5000,
        "eval_steps": 2500,
        "eval_samples": 100,
        "multi_gpu": True,
    },
    # Stochastic RAG T5Gemma2 (train_stochastic_rag_t5gemma.py defaults)
    ("stochastic_rag", "t5gemma"): {
        "lr_generator": 1e-5,
        "lr_reranker": 1e-4,
        "total_steps": 50000,
        "compression_k": 64,
        "n_candidates": 40,
        "num_passages": 10,
        "gumbel_tau": 1.0,
        "num_utility_samples": 10,
        "scoring_type": "linear",
        "temperature": 2.0,
        "batch_size": 1,
        "gradient_accumulation_steps": 128,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "save_steps": 5000,
        "eval_steps": 2500,
        "eval_samples": 100,
        "multi_gpu": True,
        "bf16": True,
    },
}


class StepConfigDialog:
    """
    Configuration dialog for a pipeline step.

    Features:
    - Paper default values pre-filled
    - Path selection with file picker
    - Advanced options collapsible
    - Input validation
    - "Use Existing" tab for skipping with existing output
    """

    def __init__(
        self,
        step_id: int,
        display_name: str,
        default_args: Dict[str, Any],
        required_paths: list[str],
        description: str,
        on_confirm: Callable[[int, Dict[str, Any]], None],
        on_skip: Callable[[int, str, Optional[Dict[str, Any]]], None] = None,
    ):
        self.step_id = step_id
        self.display_name = display_name
        self.default_args = default_args
        self.required_paths = required_paths
        self.description = description
        self.on_confirm = on_confirm
        self.on_skip = on_skip

        self.inputs: Dict[str, Any] = {}
        self.dialog: Optional[ui.dialog] = None

    def show(self):
        """Show the configuration dialog."""
        with ui.dialog() as self.dialog, ui.card().classes("w-[500px]"):
            # Header
            with ui.row().classes("w-full items-center mb-2"):
                ui.label(f"Step {self.step_id}: {self.display_name}").classes(
                    "text-h6"
                )

            # Description
            ui.label(self.description).classes("text-sm text-gray-600 dark:text-gray-400 mb-2")

            # Tabs for Run vs Use Existing
            with ui.tabs().classes("w-full") as tabs:
                run_tab = ui.tab("Run", icon="play_arrow")
                existing_tab = ui.tab("Use Existing", icon="folder_open")

            with ui.tab_panels(tabs, value=run_tab).classes("w-full"):
                # Run panel
                with ui.tab_panel(run_tab):
                    self._build_run_panel()

                # Use existing panel
                with ui.tab_panel(existing_tab):
                    self._build_existing_panel()

        self.dialog.open()

    def _build_run_panel(self):
        """Build the run configuration panel."""
        # Step 11: Precompute format selection
        if self.step_id == 11:
            ui.label("Precompute Format").classes("font-bold mt-2")
            with ui.column().classes("w-full gap-2 mb-4"):
                format_options = {
                    "fidlight": "FiD-Light (with source pointer, 40 passages)",
                    "fid_pure": "FiD Pure (answer only, 100 passages)",
                }
                self.format_select = ui.radio(
                    options=format_options,
                    value="fidlight",
                    on_change=self._on_precompute_format_change,
                ).props("dense")
                self.inputs["format"] = self.format_select
                ui.label(
                    "FiD-Light: for FiD-Light, Stochastic RAG | FiD Pure: for FiD Pure training"
                ).classes("text-xs text-gray-500 dark:text-gray-400")

        # Required paths section
        if self.required_paths:
            ui.label("Path Configuration").classes("font-bold mt-2")
            for path_key in self.required_paths:
                with ui.row().classes("w-full items-center gap-2"):
                    label = path_key.replace("_", " ").title()
                    ui.label(f"{label}:").classes("w-32 text-sm")

                    path_input = ui.input(
                        placeholder=f"Enter {label} path..."
                    ).classes("flex-grow")
                    self.inputs[path_key] = path_input

        # Advanced options (collapsible with scroll)
        if self.default_args:
            with ui.expansion("Advanced Options (Paper Defaults)").classes("w-full mt-4"):
                with ui.scroll_area().classes("max-h-64"):
                    for key, default_value in self.default_args.items():
                        if key in self.required_paths:
                            continue
                        # Step 11: skip "format" as it's handled by radio button above
                        if self.step_id == 11 and key == "format":
                            continue

                        with ui.row().classes("w-full items-center gap-2 mb-1"):
                            label = key.replace("_", " ").title()
                            ui.label(f"{label}:").classes("w-40 text-sm")

                            if isinstance(default_value, bool):
                                inp = ui.switch(value=default_value)
                            elif isinstance(default_value, int):
                                inp = ui.number(value=default_value).classes("w-32")
                            elif isinstance(default_value, float):
                                inp = ui.number(
                                    value=default_value, format="%.2e"
                                ).classes("w-32")
                            elif isinstance(default_value, list):
                                inp = ui.input(
                                    value=", ".join(str(v) for v in default_value)
                                ).classes("flex-grow")
                            else:
                                inp = ui.input(value=str(default_value)).classes(
                                    "flex-grow"
                                )

                            self.inputs[key] = inp

        # Buttons
        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=self.dialog.close).props("flat")
            ui.button(
                "Start",
                icon="play_arrow",
                on_click=self._on_confirm,
            ).props("color=primary")

    def _build_existing_panel(self):
        """Build the use existing data panel."""
        ui.label("Use existing output files to skip this step").classes("text-sm text-gray-600 dark:text-gray-400 mb-4")

        with ui.row().classes("w-full items-center gap-2 mb-4"):
            ui.label("Output Dir:").classes("w-24 text-sm")
            self.existing_path_input = ui.input(
                placeholder="Enter path to existing data..."
            ).classes("flex-grow")

        # Validation area
        with ui.row().classes("w-full items-center gap-2 mb-2"):
            ui.button(
                "Validate",
                icon="check_circle",
                on_click=self._validate_existing,
            ).props("outline")
            self.validation_label = ui.label("").classes("text-sm")

        self.validation_details = ui.column().classes("w-full")

        # Buttons
        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=self.dialog.close).props("flat")
            ui.button(
                "Use Existing",
                icon="check",
                on_click=self._use_existing,
            ).props("color=warning")

    def _on_confirm(self):
        """Handle confirm button click."""
        # Collect values
        args = {}

        for key, inp in self.inputs.items():
            if hasattr(inp, "value"):
                value = inp.value
                # Convert types
                if key in self.default_args:
                    default = self.default_args[key]
                    # Check bool BEFORE int (bool is subclass of int in Python)
                    if isinstance(default, bool):
                        value = bool(value)
                    elif isinstance(default, int):
                        value = int(value)
                    elif isinstance(default, float):
                        value = float(value)
                    elif isinstance(default, list):
                        value = [v.strip() for v in str(value).split(",")]
                args[key] = value

        # Validate required paths
        for path_key in self.required_paths:
            if path_key not in args or not args[path_key]:
                ui.notify(f"Please select {path_key}", type="warning")
                return

        self.dialog.close()
        self.on_confirm(self.step_id, args)

    def _validate_existing(self):
        """Validate existing output data."""
        path = self.existing_path_input.value.strip()
        if not path:
            ui.notify("Please enter a path", type="warning")
            return

        data_path = Path(path)
        if not data_path.exists():
            self.validation_label.text = "Path does not exist"
            self.validation_label.classes("text-red-500 dark:text-red-400", remove="text-gray-500 dark:text-gray-400 text-green-600 dark:text-green-400")
            self.validation_details.clear()
            return

        self.validation_details.clear()

        # Check for expected output based on step type
        found_items = []

        # Step 6: Wiki index (Arrow format)
        if self.step_id == 6:
            arrow_dir = data_path / "kilt_wikipedia_arrow" if (data_path / "kilt_wikipedia_arrow").exists() else data_path
            if arrow_dir.exists() and any(arrow_dir.glob("*.arrow")):
                arrow_files = list(arrow_dir.glob("*.arrow"))
                total_size = sum(f.stat().st_size for f in arrow_files) / (1024 * 1024 * 1024)
                found_items.append(("Arrow Index", f"{len(arrow_files)} file(s), {total_size:.1f} GB"))

        # Step 7: GTR index (FAISS)
        elif self.step_id == 7:
            index_dir = data_path / "gtr_faiss_index" if (data_path / "gtr_faiss_index").exists() else data_path
            if index_dir.exists():
                if (index_dir / "index.faiss").exists() or (index_dir / "index").exists():
                    found_items.append(("FAISS Index", "Found"))
                if (index_dir / "passages.json").exists() or (index_dir / "docstore.json").exists():
                    found_items.append(("Document Store", "Found"))

        # Step 8: Retrieval training data
        elif self.step_id == 8:
            # Check for retrieval_training_data.jsonl
            jsonl_file = data_path / "retrieval_training_data.jsonl" if (data_path / "retrieval_training_data.jsonl").exists() else data_path
            if jsonl_file.exists() and jsonl_file.is_file():
                size_mb = jsonl_file.stat().st_size / (1024 * 1024)
                # Count lines
                try:
                    with open(jsonl_file, 'r') as f:
                        line_count = sum(1 for _ in f)
                    found_items.append(("Training Data", f"{line_count:,} triplets, {size_mb:.1f} MB"))
                except:
                    found_items.append(("Training Data", f"{size_mb:.1f} MB"))
            elif data_path.is_file() and str(data_path).endswith('.jsonl'):
                size_mb = data_path.stat().st_size / (1024 * 1024)
                found_items.append(("Training Data", f"{size_mb:.1f} MB"))

        # Step 9: Trained retriever checkpoint
        elif self.step_id == 9:
            if (data_path / "pytorch_model.bin").exists() or (data_path / "model.safetensors").exists():
                found_items.append(("Model Weights", "Found"))
            if (data_path / "config.json").exists():
                found_items.append(("Config File", "Found"))

        # Step 10: Rebuilt index (finetuned FAISS)
        elif self.step_id == 10:
            index_dir = data_path / "gtr_faiss_index_finetuned" if (data_path / "gtr_faiss_index_finetuned").exists() else data_path
            if index_dir.exists():
                if (index_dir / "index.faiss").exists() or (index_dir / "index").exists():
                    found_items.append(("Finetuned FAISS Index", "Found"))

        # Step 11: Precomputed retrieval
        elif self.step_id == 11:
            precomputed_dir = data_path / "precomputed" if (data_path / "precomputed").exists() else data_path
            if precomputed_dir.exists():
                parquet_files = list(precomputed_dir.glob("*.parquet"))
                if parquet_files:
                    found_items.append(("Precomputed Files", f"{len(parquet_files)} parquet file(s)"))

        # Generic check for any step
        if not found_items:
            # Just check if directory is not empty
            if data_path.is_dir() and any(data_path.iterdir()):
                found_items.append(("Directory", "Not empty"))

        # Display results
        with self.validation_details:
            if found_items:
                for name, info in found_items:
                    ui.label(f"  {name}: {info}").classes("text-xs text-green-600")
                self.validation_label.text = "Validation passed"
                self.validation_label.classes("text-green-600 dark:text-green-400", remove="text-gray-500 dark:text-gray-400 text-red-500 dark:text-red-400")
            else:
                ui.label("Expected output files not found").classes("text-xs text-red-500")
                self.validation_label.text = "Validation failed"
                self.validation_label.classes("text-red-500 dark:text-red-400", remove="text-gray-500 dark:text-gray-400 text-green-600 dark:text-green-400")

        self.found_items = found_items

    def _use_existing(self):
        """Use existing data to skip this step."""
        path = self.existing_path_input.value.strip()
        if not path:
            ui.notify("Please enter a path", type="warning")
            return

        if not Path(path).exists():
            ui.notify("Path does not exist", type="negative")
            return

        if self.on_skip is None:
            ui.notify("This step cannot be skipped", type="warning")
            return

        # Build extra info
        extra = None
        if hasattr(self, 'found_items') and self.found_items:
            extra = {"found_items": [{"name": n, "info": i} for n, i in self.found_items]}

        self.dialog.close()
        self.on_skip(self.step_id, path, extra)

    def _on_precompute_format_change(self, e):
        """Handle precompute format change for Step 11."""
        format_value = e.value
        # Update num_passages default based on format
        # FiD-Light: 40 passages (with source pointer)
        # FiD Pure: 100 passages (answer only)
        if "num_passages" in self.inputs:
            if format_value == "fid_pure":
                self.inputs["num_passages"].value = 100
            else:
                self.inputs["num_passages"].value = 40


class SkipStepDialog:
    """Dialog for skipping a step with existing output."""

    def __init__(
        self,
        step_id: int,
        display_name: str,
        on_confirm: Callable[[int, str], None],
    ):
        self.step_id = step_id
        self.display_name = display_name
        self.on_confirm = on_confirm
        self.dialog: Optional[ui.dialog] = None

    def show(self):
        """Show the skip dialog."""
        with ui.dialog() as self.dialog, ui.card().classes("w-80"):
            ui.label(f"Skip Step {self.step_id}").classes("text-h6 mb-4")

            ui.label(
                f"To skip '{self.display_name}', please provide the path to existing output files."
            ).classes("text-sm text-gray-600 dark:text-gray-400 mb-4")

            with ui.row().classes("w-full items-center gap-2"):
                self.path_input = ui.input(
                    placeholder="Select output folder..."
                ).classes("flex-grow")

                ui.button(
                    icon="folder_open",
                    on_click=self._pick_folder,
                ).props("flat dense")

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=self.dialog.close).props("flat")
                ui.button(
                    "Skip",
                    icon="skip_next",
                    on_click=self._on_confirm,
                ).props("color=warning")

        self.dialog.open()

    async def _pick_folder(self):
        """Open folder picker."""
        # Note: Browser file picker is limited, may need to use text input
        ui.notify("Please enter the folder path manually", type="info")

    def _on_confirm(self):
        """Handle confirm."""
        path = self.path_input.value
        if not path:
            ui.notify("Please enter output path", type="warning")
            return

        if not Path(path).exists():
            ui.notify(f"Path does not exist: {path}", type="negative")
            return

        self.dialog.close()
        self.on_confirm(self.step_id, path)


class KiltDataConfigDialog:
    """
    Configuration dialog for KILT data steps (2.1, 2.2, 2.3).

    Features:
    - Step 2: Download or use existing KILT data
    - Step 3: Filter data (remove samples without provenance)
    - Step 4: Fix TriviaQA (add missing question text)
    """

    # Step info
    STEP_INFO = {
        2: {
            "title": "2.1 Download Wikipedia",
            "description": "Download KILT Wikipedia knowledge base (~35GB)",
            "script": "download_kilt_wiki.py",
            "run_label": "Start Download",
            "run_icon": "download",
        },
        3: {
            "title": "2.2 Download Task Datasets",
            "description": "Download KILT task datasets (NQ, HotpotQA, TriviaQA)",
            "script": "download_kilt_data.py",
            "run_label": "Start Download",
            "run_icon": "download",
        },
        4: {
            "title": "2.3 Fix TriviaQA",
            "description": "Fix TriviaQA data by adding missing question text (output to intermediate dir)",
            "script": "fix_triviaqa.py",
            "run_label": "Start Fix",
            "run_icon": "build",
        },
        5: {
            "title": "2.4 Filter Data",
            "description": "Filter out samples without provenance, TriviaQA reads from fixed files",
            "script": "filter_kilt_data.py",
            "run_label": "Start Filter",
            "run_icon": "filter_alt",
        },
    }

    def __init__(
        self,
        step_id: int,
        on_confirm: Callable[[int, Dict[str, Any]], None],
        on_skip: Callable[[int, str, Optional[Dict[str, Any]]], None],
    ):
        """
        Initialize the dialog.

        Args:
            step_id: Step ID (2, 3, or 4)
            on_confirm: Callback for starting the step
            on_skip: Callback for using existing data (step_id, path, extra)
        """
        self.step_id = step_id
        self.on_confirm = on_confirm
        self.on_skip = on_skip
        self.dialog: Optional[ui.dialog] = None
        self.validation_result = None
        self.validation_label = None
        self.found_items: List[tuple] = []  # List of (name, info) tuples

    def show(self):
        """Show the configuration dialog."""
        info = self.STEP_INFO.get(self.step_id, self.STEP_INFO[2])

        with ui.dialog() as self.dialog, ui.card().classes("w-[550px]"):
            # Header
            ui.label(info["title"]).classes("text-h6 mb-2")
            ui.label(info["description"]).classes("text-sm text-gray-600 dark:text-gray-400 mb-4")

            ui.separator()

            # Mode selection
            with ui.tabs().classes("w-full") as tabs:
                use_existing_tab = ui.tab("Use Existing", icon="folder")
                run_tab = ui.tab("Run Script", icon="play_arrow")

            with ui.tab_panels(tabs, value=use_existing_tab).classes("w-full"):
                # Use existing data panel
                with ui.tab_panel(use_existing_tab):
                    ui.label("Select existing data directory").classes("font-bold mt-2 mb-2")

                    with ui.row().classes("w-full items-center gap-2"):
                        self.existing_path_input = ui.input(
                            placeholder="e.g. kilt_data/",
                            value="kilt_data/"
                        ).classes("flex-grow")

                        ui.button(
                            "Validate",
                            icon="check_circle",
                            on_click=self._validate_existing_data,
                        ).props("color=primary")

                    # Validation result area
                    with ui.card().classes("w-full mt-4 bg-gray-50 dark:bg-gray-800") as self.validation_card:
                        self.validation_label = ui.label("Click \"Validate\" to check data").classes("text-sm text-gray-500 dark:text-gray-400")
                        self.validation_details = ui.column().classes("w-full mt-2")

                    # Use button
                    with ui.row().classes("w-full justify-end gap-2 mt-4"):
                        ui.button("Cancel", on_click=self.dialog.close).props("flat")
                        ui.button(
                            "Use This Data",
                            icon="check",
                            on_click=self._use_existing,
                        ).props("color=positive")

                # Run script panel
                with ui.tab_panel(run_tab):
                    self._build_run_panel(info)

        self.dialog.open()

    def _build_run_panel(self, info):
        """Build the run script panel based on step."""
        if self.step_id == 2:
            # Download Wikipedia panel
            ui.label("Download Wikipedia Knowledge Base").classes("font-bold mt-2 mb-2")

            with ui.row().classes("w-full items-center gap-2 mb-4"):
                ui.label("Output Dir:").classes("w-24 text-sm")
                self.output_dir_input = ui.input(
                    placeholder="kilt_data/",
                    value="kilt_data/"
                ).classes("flex-grow")

            ui.label("Will download kilt_knowledgesource.json (~35GB)").classes("text-xs text-gray-500 dark:text-gray-400 mt-2")
            ui.label("Note: Download may take a long time").classes("text-xs text-orange-500 mt-2")

        elif self.step_id == 3:
            # Download tasks panel
            ui.label("Download Task Datasets").classes("font-bold mt-2 mb-2")

            with ui.row().classes("w-full items-center gap-2 mb-4"):
                ui.label("Output Dir:").classes("w-24 text-sm")
                self.output_dir_input = ui.input(
                    placeholder="kilt_data/",
                    value="kilt_data/"
                ).classes("flex-grow")

            ui.label("Select Tasks:").classes("text-sm mb-2")
            self.task_checkboxes = {}
            tasks = [
                ("nq", "Natural Questions (NQ)"),
                ("hotpotqa", "HotpotQA"),
                ("triviaqa_support_only", "TriviaQA"),
            ]
            for task_id, task_name in tasks:
                cb = ui.checkbox(task_name, value=True)
                self.task_checkboxes[task_id] = cb

            ui.label("Will download train/dev datasets from HuggingFace").classes("text-xs text-gray-500 dark:text-gray-400 mt-2")

        elif self.step_id == 4:
            # Fix TriviaQA panel (step 4 now)
            ui.label("Fix TriviaQA Data").classes("font-bold mt-2 mb-2")

            with ui.row().classes("w-full items-center gap-2 mb-4"):
                ui.label("Output Dir:").classes("w-24 text-sm")
                self.output_dir_input = ui.input(
                    placeholder="kilt_data/triviaqa_fixed/",
                    value="kilt_data/triviaqa_fixed/"
                ).classes("flex-grow")

            ui.label("Will download original TriviaQA from HuggingFace and map question text (intermediate dir, read by filter step)").classes("text-xs text-gray-500 dark:text-gray-400 mt-2")

        elif self.step_id == 5:
            # Filter panel (step 5 now)
            ui.label("Filter KILT Data").classes("font-bold mt-2 mb-2")

            with ui.row().classes("w-full items-center gap-2 mb-2"):
                ui.label("Data Dir:").classes("w-24 text-sm")
                self.cache_dir_input = ui.input(
                    placeholder="kilt_data/",
                    value="kilt_data/"
                ).classes("flex-grow")

            with ui.row().classes("w-full items-center gap-2 mb-2"):
                ui.label("Output Dir:").classes("w-24 text-sm")
                self.output_dir_input = ui.input(
                    placeholder="kilt_data/filtered/",
                    value="kilt_data/filtered/"
                ).classes("flex-grow")

            with ui.row().classes("w-full items-center gap-2 mb-4"):
                ui.label("TriviaQA Fixed:").classes("w-24 text-sm")
                self.triviaqa_fixed_dir_input = ui.input(
                    placeholder="kilt_data/triviaqa_fixed/",
                    value="kilt_data/triviaqa_fixed/"
                ).classes("flex-grow")

            ui.label("Select Tasks:").classes("text-sm mb-2")
            self.task_checkboxes = {}
            tasks = [
                ("nq", "Natural Questions (NQ)"),
                ("hotpotqa", "HotpotQA"),
                ("triviaqa_support_only", "TriviaQA"),
            ]
            for task_id, task_name in tasks:
                cb = ui.checkbox(task_name, value=True)
                self.task_checkboxes[task_id] = cb

            ui.label("Filter out samples without provenance, TriviaQA reads from fixed intermediate dir").classes("text-xs text-gray-500 dark:text-gray-400 mt-2")

        # Run button
        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=self.dialog.close).props("flat")
            ui.button(
                info["run_label"],
                icon=info["run_icon"],
                on_click=self._start_run,
            ).props("color=primary")

    def _validate_existing_data(self):
        """Validate existing data based on step."""
        path = self.existing_path_input.value.strip()
        if not path:
            ui.notify("Please enter data path", type="warning")
            return

        data_path = Path(path)
        if not data_path.exists():
            self.validation_label.text = "Path does not exist"
            self.validation_label.classes("text-red-500 dark:text-red-400", remove="text-gray-500 dark:text-gray-400 text-green-600 dark:text-green-400")
            self.validation_details.clear()
            return

        self.validation_details.clear()
        found_items = []
        sample_file = None
        self.actual_data_path = str(data_path)  # Track the actual path where data was found

        if self.step_id == 2:
            # Step 2: Check for Wikipedia knowledge source only
            for ext in [".json", ".jsonl"]:
                wiki_file = data_path / f"kilt_knowledgesource{ext}"
                if wiki_file.exists():
                    size_gb = wiki_file.stat().st_size / (1024 * 1024 * 1024)
                    found_items.append(("kilt_knowledgesource" + ext, f"{size_gb:.1f} GB"))
                    # Try to read sample Wikipedia articles
                    try:
                        import json
                        samples = []
                        with open(wiki_file, 'r', encoding='utf-8') as f:
                            for i, line in enumerate(f):
                                if i >= 3:  # Read 3 samples
                                    break
                                doc = json.loads(line)
                                title = doc.get('wikipedia_title', doc.get('title', 'N/A'))
                                text = doc.get('text', [''])[0] if isinstance(doc.get('text'), list) else doc.get('text', '')[:100]
                                samples.append({"title": title, "text": text[:80] + "..." if len(text) > 80 else text})
                        self.wiki_samples = samples
                    except Exception:
                        self.wiki_samples = []
                    break

        elif self.step_id == 3:
            # Step 3: Check for task datasets (HuggingFace cache or original files)
            self.task_samples = []  # Store QA samples

            # Check HuggingFace cache directory
            hf_cache = data_path / "facebook___kilt_tasks"
            if hf_cache.exists():
                self.actual_data_path = str(hf_cache)  # Update to HF cache path
                for task_name, display_name in [
                    ("nq", "NQ"),
                    ("hotpotqa", "HotpotQA"),
                    ("triviaqa_support_only", "TriviaQA"),
                ]:
                    task_dir = hf_cache / task_name
                    if task_dir.exists():
                        found_items.append((task_name, display_name))

            # Check for original KILT files (if downloaded separately)
            original_files = [
                ("nq-train-kilt.jsonl", "NQ"),
                ("hotpotqa-train-kilt.jsonl", "HotpotQA"),
                ("triviaqa-train_id-kilt.jsonl", "TriviaQA"),
            ]
            for filename, display_name in original_files:
                file_path = data_path / filename
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    # Avoid duplicates if already found in HF cache
                    if not any(display_name in item[1] for item in found_items):
                        found_items.append((filename, f"{display_name} ({size_mb:.1f} MB)"))
                    if not sample_file:
                        sample_file = file_path
                    # Read a sample from each task
                    if len(self.task_samples) < 3:
                        try:
                            import json
                            with open(file_path, 'r', encoding='utf-8') as f:
                                line = f.readline()
                                sample = json.loads(line)
                                q = sample.get('input', '')
                                a = sample.get('output', [{}])[0].get('answer', '') if sample.get('output') else ''
                                if q:
                                    self.task_samples.append({
                                        "task": display_name,
                                        "question": q[:60] + "..." if len(q) > 60 else q,
                                        "answer": a[:40] + "..." if len(str(a)) > 40 else str(a)
                                    })
                        except Exception:
                            pass

        elif self.step_id == 4:
            # Step 4: Check for fixed TriviaQA data (intermediate, before filtering)
            fixed_dir = data_path / "triviaqa_fixed" if (data_path / "triviaqa_fixed").exists() else data_path
            self.actual_data_path = str(fixed_dir)  # Update to actual path

            # TriviaQA files with proper question text
            triviaqa_files = [
                ("triviaqa_support_only_train.jsonl", "TriviaQA Train"),
                ("triviaqa_support_only_validation.jsonl", "TriviaQA Validation"),
                ("triviaqa_support_only_test.jsonl", "TriviaQA Test"),
            ]
            for filename, display_name in triviaqa_files:
                file_path = fixed_dir / filename
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    found_items.append((display_name, f"{size_mb:.1f} MB"))
                    if not sample_file:
                        sample_file = file_path

        elif self.step_id == 5:
            # Step 5: Check for filtered data - group by task
            filtered_dir = data_path / "filtered" if (data_path / "filtered").exists() else data_path
            self.actual_data_path = str(filtered_dir)  # Update to actual path

            tasks_check = [
                ("nq", "NQ", ["nq_train.jsonl", "nq_validation.jsonl"]),
                ("hotpotqa", "HotpotQA", ["hotpotqa_train.jsonl", "hotpotqa_validation.jsonl"]),
                ("triviaqa", "TriviaQA", ["triviaqa_support_only_train.jsonl", "triviaqa_support_only_validation.jsonl"]),
            ]
            for task_id, task_name, files in tasks_check:
                task_size = 0
                task_files = 0
                for filename in files:
                    file_path = filtered_dir / filename
                    if file_path.exists():
                        task_size += file_path.stat().st_size / (1024 * 1024)
                        task_files += 1
                        if not sample_file:
                            sample_file = file_path
                if task_files > 0:
                    found_items.append((task_name, f"{task_files} file(s), {task_size:.1f} MB"))

        # Show results
        with self.validation_details:
            if found_items:
                for name, info in found_items:
                    ui.label(f"  • {name}: {info}").classes("text-xs text-gray-600 dark:text-gray-400")
            else:
                step_hints = {
                    2: "Required: kilt_knowledgesource.json or kilt_knowledgesource.jsonl",
                    3: "Required: facebook___kilt_tasks/ directory or original KILT files",
                    4: "Required: triviaqa_fixed/triviaqa_support_only_*.jsonl files (with question text)",
                    5: "Required: *_train.jsonl and *_validation.jsonl files in filtered/ directory",
                }
                ui.label("Required data not found").classes("text-sm text-red-500")
                ui.label(step_hints.get(self.step_id, "")).classes("text-xs text-gray-500 dark:text-gray-400 mt-1")

            # Show sample data
            if sample_file:
                try:
                    import json
                    with open(sample_file, 'r', encoding='utf-8') as f:
                        first_line = f.readline()
                        sample_data = json.loads(first_line)

                    ui.label("Data Sample Preview:").classes("text-sm font-bold mt-3")
                    with ui.card().classes("w-full bg-gray-100 dark:bg-gray-800 p-2"):
                        if "input" in sample_data:
                            input_text = sample_data['input']
                            if input_text:
                                display_text = f"{input_text[:100]}..." if len(input_text) > 100 else input_text
                                ui.label(f"Question: {display_text}").classes("text-xs")
                            else:
                                ui.label("Question: Empty (needs fix_triviaqa)").classes("text-xs text-orange-500")
                        if "output" in sample_data and sample_data["output"]:
                            out = sample_data["output"][0]
                            answer = out.get("answer", "N/A")
                            ui.label(f"Answer: {answer}").classes("text-xs")
                            prov = out.get("provenance", [])
                            if prov:
                                title = prov[0].get("title", "N/A")
                                ui.label(f"Source: {title}").classes("text-xs text-gray-500 dark:text-gray-400")
                            else:
                                ui.label("Source: No provenance").classes("text-xs text-orange-500")
                except Exception as e:
                    ui.label(f"Failed to read sample: {e}").classes("text-xs text-red-500")

        if found_items:
            self.validation_label.text = "Validation passed"
            self.validation_label.classes("text-green-600 dark:text-green-400", remove="text-gray-500 dark:text-gray-400 text-red-500 dark:text-red-400")
            self.validation_result = True
            # Save found items for use in _use_existing
            self.found_items = found_items
        else:
            self.validation_label.text = "No valid data found"
            self.validation_label.classes("text-red-500 dark:text-red-400", remove="text-gray-500 dark:text-gray-400 text-green-600 dark:text-green-400")
            self.validation_result = False
            self.found_items = []

    def _use_existing(self):
        """Use existing data."""
        path = self.existing_path_input.value.strip()
        if not path:
            ui.notify("Please enter data path", type="warning")
            return

        if not Path(path).exists():
            ui.notify("Path does not exist", type="negative")
            return

        # Use the actual data path (may be a subdirectory like filtered/)
        actual_path = getattr(self, 'actual_data_path', path)

        # Build extra data with found items and samples
        if self.found_items:
            extra = {
                "found_items": [{"name": name, "info": info} for name, info in self.found_items]
            }
            # Add samples for step 2 (Wikipedia) and step 3 (tasks)
            if self.step_id == 2 and hasattr(self, 'wiki_samples') and self.wiki_samples:
                extra["wiki_samples"] = self.wiki_samples
            elif self.step_id == 3 and hasattr(self, 'task_samples') and self.task_samples:
                extra["task_samples"] = self.task_samples
        else:
            extra = None

        self.dialog.close()
        self.on_skip(self.step_id, actual_path, extra)

    def _start_run(self):
        """Start running the script."""
        if self.step_id == 2:
            # Download Wikipedia
            output_dir = self.output_dir_input.value.strip()
            if not output_dir:
                ui.notify("Please enter output directory", type="warning")
                return

            args = {
                "cache-dir": output_dir,
            }

        elif self.step_id == 3:
            # Download tasks
            output_dir = self.output_dir_input.value.strip()
            if not output_dir:
                ui.notify("Please enter output directory", type="warning")
                return

            tasks = [task_id for task_id, cb in self.task_checkboxes.items() if cb.value]
            if not tasks:
                ui.notify("Please select at least one task", type="warning")
                return

            args = {
                "cache-dir": output_dir,
            }

        elif self.step_id == 4:
            # Fix TriviaQA (step 4 now)
            output_dir = self.output_dir_input.value.strip()
            if not output_dir:
                ui.notify("Please enter output directory", type="warning")
                return

            args = {
                "output_dir": output_dir,
            }

        elif self.step_id == 5:
            # Filter (step 5 now)
            cache_dir = self.cache_dir_input.value.strip()
            output_dir = self.output_dir_input.value.strip()
            triviaqa_fixed_dir = self.triviaqa_fixed_dir_input.value.strip()

            if not cache_dir:
                ui.notify("Please enter data directory", type="warning")
                return
            if not output_dir:
                ui.notify("Please enter output directory", type="warning")
                return
            if not triviaqa_fixed_dir:
                ui.notify("Please enter TriviaQA fixed directory", type="warning")
                return

            tasks = [task_id for task_id, cb in self.task_checkboxes.items() if cb.value]
            if not tasks:
                ui.notify("Please select at least one task", type="warning")
                return

            args = {
                "cache-dir": cache_dir,
                "output-dir": output_dir,
                "triviaqa-fixed-dir": triviaqa_fixed_dir,
                "tasks": tasks,
            }

        else:
            args = {}

        self.dialog.close()
        self.on_confirm(self.step_id, args)


class TrainModelConfigDialog:
    """
    Configuration dialog for Step 12: Train Model.

    Features:
    - Algorithm selection (FiD-Light, FiD Pure, Stochastic RAG)
    - Model selection (T5-base, T5Gemma2)
    - Dynamic default parameters based on selection
    - Required paths (precomputed_path, output_dir)
    - Checkpoint resume support
    """

    def __init__(
        self,
        on_confirm: Callable[[int, Dict[str, Any]], None],
    ):
        """
        Initialize the dialog.

        Args:
            on_confirm: Callback with (step_id=12, args)
        """
        self.on_confirm = on_confirm
        self.inputs: Dict[str, Any] = {}
        self.dialog: Optional[ui.dialog] = None
        self.advanced_container = None
        self.resume_path_row = None

        # Current selection
        self.selected_algorithm = "fidlight"
        self.selected_model = "t5base"
        self.resume_enabled = False

    def show(self):
        """Show the configuration dialog."""
        with ui.dialog() as self.dialog, ui.card().classes("w-[500px]"):
            # Header
            with ui.row().classes("w-full items-center mb-2"):
                ui.label("Step 9: Train Model").classes("text-h6")

            ui.label("Select algorithm and model backbone for training").classes(
                "text-sm text-gray-600 dark:text-gray-400 mb-4"
            )

            ui.separator()

            # Algorithm selection
            ui.label("Select Algorithm").classes("font-bold mt-4 mb-2")
            algo_options = list(ALGORITHM_OPTIONS.values())
            algo_select = ui.radio(
                algo_options,
                value=ALGORITHM_OPTIONS["fidlight"],
                on_change=self._on_algorithm_change,
            ).props("dense")
            self.inputs["_algo_select"] = algo_select

            # Model selection
            ui.label("Select Model Backbone").classes("font-bold mt-4 mb-2")
            model_options = list(MODEL_OPTIONS.values())
            model_select = ui.radio(
                model_options,
                value=MODEL_OPTIONS["t5base"],
                on_change=self._on_model_change,
            ).props("dense inline")
            self.inputs["_model_select"] = model_select

            ui.separator().classes("mt-4")

            # Required paths
            ui.label("Required Paths").classes("font-bold mt-4 mb-2")

            with ui.row().classes("w-full items-center gap-2 mb-2"):
                ui.label("Train Data Dir:").classes("w-28 text-sm")
                precomputed_input = ui.input(
                    placeholder="kilt_data/precomputed/"
                ).classes("flex-grow")
                ui.label("(directory, loads *_train.parquet with temperature sampling)").classes("text-xs text-gray-500 dark:text-gray-400")
                self.inputs["precomputed_path"] = precomputed_input

            with ui.row().classes("w-full items-center gap-2 mb-2"):
                ui.label("Val Data:").classes("w-28 text-sm")
                precomputed_val_input = ui.input(
                    placeholder="kilt_data/precomputed/all_tasks_dev.parquet"
                ).classes("flex-grow")
                ui.label("(file or directory)").classes("text-xs text-gray-500 dark:text-gray-400")
                self.inputs["precomputed_val_path"] = precomputed_val_input

            with ui.row().classes("w-full items-center gap-2"):
                ui.label("Output Dir:").classes("w-28 text-sm")
                output_input = ui.input(
                    placeholder="checkpoints/..."
                ).classes("flex-grow")
                self.inputs["output_dir"] = output_input

            ui.separator().classes("mt-4")

            # Resume training option
            ui.label("Resume Training").classes("font-bold mt-4 mb-2")
            with ui.row().classes("w-full items-center gap-2 mb-2"):
                resume_switch = ui.switch(
                    "Resume from checkpoint",
                    value=False,
                    on_change=self._on_resume_change,
                )
                self.inputs["_resume_switch"] = resume_switch

            # Resume checkpoint path (initially hidden)
            self.resume_path_row = ui.row().classes("w-full items-center gap-2")
            self.resume_path_row.visible = False
            with self.resume_path_row:
                ui.label("Checkpoint:").classes("w-28 text-sm")
                resume_path_input = ui.input(
                    placeholder="checkpoints/fidlight_paper/step_10000"
                ).classes("flex-grow")
                self.inputs["resume_from"] = resume_path_input

            ui.label("Tip: Select directory containing model.pt and optimizer.pt").classes(
                "text-xs text-gray-500 dark:text-gray-400 mt-1"
            ).bind_visibility_from(self.resume_path_row, "visible")

            # Advanced options (collapsible)
            with ui.expansion("Advanced Options (Paper Defaults)").classes("w-full mt-4") as exp:
                self.advanced_container = ui.column().classes("w-full")
                self._update_advanced_options()

            ui.separator().classes("mt-4")

            # Buttons
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=self.dialog.close).props("flat")
                ui.button(
                    "Start Training",
                    icon="play_arrow",
                    on_click=self._on_confirm,
                ).props("color=primary")

        self.dialog.open()

    def _get_algorithm_key(self) -> str:
        """Get the algorithm key from selection."""
        selected = self.inputs["_algo_select"].value
        for key, label in ALGORITHM_OPTIONS.items():
            if label == selected:
                return key
        return "fidlight"

    def _get_model_key(self) -> str:
        """Get the model key from selection."""
        selected = self.inputs["_model_select"].value
        for key, label in MODEL_OPTIONS.items():
            if label == selected:
                return key
        return "t5base"

    def _on_algorithm_change(self, e):
        """Handle algorithm selection change."""
        self.selected_algorithm = self._get_algorithm_key()
        self._update_advanced_options()

    def _on_model_change(self, e):
        """Handle model selection change."""
        self.selected_model = self._get_model_key()
        self._update_advanced_options()

    def _on_resume_change(self, e):
        """Handle resume switch change."""
        self.resume_enabled = e.value
        if self.resume_path_row:
            self.resume_path_row.visible = e.value

    def _update_advanced_options(self):
        """Update advanced options based on current selection."""
        if self.advanced_container is None:
            return

        # Clear existing inputs (except path inputs)
        keys_to_remove = [k for k in self.inputs if not k.startswith("_") and k not in ("precomputed_path", "precomputed_val_path", "output_dir")]
        for k in keys_to_remove:
            del self.inputs[k]

        self.advanced_container.clear()

        # Get default args for current combination
        key = (self.selected_algorithm, self.selected_model)
        defaults = TRAIN_DEFAULT_ARGS.get(key, {})

        with self.advanced_container:
            for param_key, default_value in defaults.items():
                with ui.row().classes("w-full items-center gap-2 mb-1"):
                    label = param_key.replace("_", " ").title()
                    ui.label(f"{label}:").classes("w-44 text-sm")

                    if isinstance(default_value, bool):
                        inp = ui.switch(value=default_value)
                    elif isinstance(default_value, int):
                        inp = ui.number(value=default_value).classes("w-28")
                    elif isinstance(default_value, float):
                        inp = ui.number(
                            value=default_value, format="%.2e"
                        ).classes("w-28")
                    else:
                        inp = ui.input(value=str(default_value)).classes("w-28")

                    self.inputs[param_key] = inp

    def _on_confirm(self):
        """Handle confirm button click."""
        # Validate required paths
        precomputed_path = self.inputs["precomputed_path"].value
        output_dir = self.inputs["output_dir"].value

        if not precomputed_path:
            ui.notify("Please enter precomputed data path", type="warning")
            return

        if not output_dir:
            ui.notify("Please enter output directory", type="warning")
            return

        # Validate resume path if enabled
        resume_from = None
        if self.resume_enabled:
            resume_from = self.inputs["resume_from"].value
            if not resume_from:
                ui.notify("Please enter checkpoint path", type="warning")
                return
            # Check if checkpoint directory exists
            resume_path = Path(resume_from)
            if not resume_path.exists():
                ui.notify(f"Checkpoint path does not exist: {resume_from}", type="negative")
                return
            # Check for required files (model.pt or model_*.pt)
            model_files = list(resume_path.glob("model*.pt"))
            if not model_files:
                ui.notify("model.pt file not found in checkpoint directory", type="warning")
                return

        # Get validation data path (optional, defaults to same as precomputed_path)
        precomputed_val_path = self.inputs["precomputed_val_path"].value.strip()

        # Collect values
        args = {
            "algorithm": self._get_algorithm_key(),
            "model": self._get_model_key(),
            "precomputed_path": precomputed_path,
            "output_dir": output_dir,
        }

        # Add validation path if specified
        if precomputed_val_path:
            args["precomputed_val_path"] = precomputed_val_path

        # Add resume path if enabled
        if resume_from:
            args["resume_from"] = resume_from

        # Add advanced options
        key = (args["algorithm"], args["model"])
        defaults = TRAIN_DEFAULT_ARGS.get(key, {})

        for param_key, default_value in defaults.items():
            if param_key in self.inputs:
                value = self.inputs[param_key].value
                # Convert types (check bool BEFORE int, since bool is subclass of int)
                if isinstance(default_value, bool):
                    value = bool(value)  # ui.switch returns bool directly
                elif isinstance(default_value, int):
                    value = int(value)
                elif isinstance(default_value, float):
                    value = float(value)
                args[param_key] = value

        self.dialog.close()
        self.on_confirm(12, args)  # Step ID is 12 (displayed as "9. Train Model")
