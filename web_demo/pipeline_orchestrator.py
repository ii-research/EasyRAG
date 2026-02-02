"""
Pipeline Orchestrator for FiDLight Web Demo
=============================================

Manages the 8-step training pipeline:
1. Environment Check
2. Download KILT Data
3. Build Wiki Index
4. Build GTR Index
5. Train GTR Retriever
6. Rebuild Index (with fine-tuned model)
7. Precompute Retrieval
8. Train FiD-Light

Features:
- Sequential execution of steps
- Step-by-step manual trigger
- One-click full pipeline
- Skip steps with existing output
- Resume from any step
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

from .utils.state_io import (
    PipelineState,
    StepState,
    StepStatus,
    PipelineStatus,
    read_pipeline_state,
    write_pipeline_state,
    initialize_pipeline_state,
    update_step_state,
    mark_step_completed,
    mark_step_failed,
    mark_step_skipped,
    STATE_FILE_PATH,
    STEP_DEFINITIONS,
    WorkspaceConfig,
)
from .utils.process_manager import ProcessManager, get_process_manager


PROJECT_ROOT = Path(__file__).parent.parent


# Training script mapping: (algorithm, model) -> script_name
TRAIN_SCRIPTS = {
    ("fidlight", "t5base"): "train_fidlight_paper.py",
    ("fidlight", "t5gemma"): "train_fidlight_t5gemma.py",
    ("fid_pure", "t5base"): "train_fid_pure.py",
    ("fid_pure", "t5gemma"): "train_fid_pure_t5gemma.py",
    ("stochastic_rag", "t5base"): "train_stochastic_rag.py",
    ("stochastic_rag", "t5gemma"): "train_stochastic_rag_t5gemma.py",
}

# Evaluation script mapping: (algorithm, model, eval_type) -> script_name
EVAL_SCRIPTS = {
    # Single checkpoint evaluation
    ("fidlight", "t5base", "single"): "evaluate_fidlight.py",
    ("fidlight", "t5gemma", "single"): "evaluate_fidlight_t5gemma.py",
    ("fid_pure", "t5base", "single"): "evaluate_fid_pure.py",
    ("fid_pure", "t5gemma", "single"): "evaluate_fid_pure_t5gemma.py",
    ("stochastic_rag", "t5base", "single"): "evaluate_stochastic_rag.py",
    ("stochastic_rag", "t5gemma", "single"): "evaluate_stochastic_rag_t5gemma.py",
    # All checkpoints evaluation
    ("fidlight", "t5base", "all"): "evaluate_fidlight_t5base_all_checkpoints.py",
    ("fidlight", "t5gemma", "all"): "evaluate_fidlight_t5gemma_all_checkpoints.py",
    ("fid_pure", "t5base", "all"): "evaluate_fid_pure_all_checkpoints.py",
    ("fid_pure", "t5gemma", "all"): "evaluate_fid_pure_all_checkpoints_t5gemma.py",
    ("stochastic_rag", "t5base", "all"): "evaluate_stochastic_rag_all_checkpoints.py",
    ("stochastic_rag", "t5gemma", "all"): "evaluate_stochastic_rag_t5gemma_all_checkpoints.py",
}

# Algorithm display names
ALGORITHM_NAMES = {
    "fidlight": "FiD-Light",
    "fid_pure": "FiD Pure",
    "stochastic_rag": "Stochastic RAG",
}

# Model display names
MODEL_NAMES = {
    "t5base": "T5-base",
    "t5gemma": "T5Gemma2-540M",
}



@dataclass
class StepConfig:
    """Configuration for a pipeline step."""
    id: int
    name: str
    display_name: str
    script: Optional[str]
    default_args: Dict[str, Any] = field(default_factory=dict)
    required_paths: List[str] = field(default_factory=list)
    output_path_key: Optional[str] = None
    depends_on: List[int] = field(default_factory=list)
    description: str = ""


# Step configurations with default parameters (12 steps)
STEP_CONFIGS: List[StepConfig] = [
    StepConfig(
        id=1,
        name="environment_check",
        display_name="1. Environment Check",
        script=None,  # Built-in check
        description="Verify GPU availability, CUDA version, and dependencies",
    ),
    StepConfig(
        id=2,
        name="download_wiki",
        display_name="2.1 Download Wikipedia",
        script="download_kilt_data.py",
        default_args={
            "wikipedia-only": True,
        },
        required_paths=["cache-dir"],
        output_path_key="cache-dir",
        depends_on=[1],
        description="Download KILT Wikipedia knowledge base (~35GB)",
    ),
    StepConfig(
        id=3,
        name="download_tasks",
        display_name="2.2 Download Task Datasets",
        script="download_kilt_data.py",
        default_args={
            "tasks-only": True,
        },
        required_paths=["cache-dir"],
        output_path_key="cache-dir",
        depends_on=[1],
        description="Download KILT task datasets (NQ, HotpotQA, TriviaQA)",
    ),
    StepConfig(
        id=4,
        name="fix_triviaqa",
        display_name="2.3 Fix TriviaQA",
        script="fix_triviaqa.py",
        default_args={},
        required_paths=["output_dir"],
        output_path_key="output_dir",
        depends_on=[3],
        description="Fix TriviaQA data by adding missing question text (output to intermediate dir)",
    ),
    StepConfig(
        id=5,
        name="filter_data",
        display_name="2.4 Filter Data",
        script="filter_kilt_data.py",
        default_args={
            "tasks": ["nq", "hotpotqa", "triviaqa_support_only"],
            "splits": ["train", "validation"],
        },
        required_paths=["cache-dir", "output-dir", "triviaqa-fixed-dir"],
        output_path_key="output_dir",
        depends_on=[4],
        description="Filter out samples without provenance, TriviaQA reads from fixed intermediate files",
    ),
    StepConfig(
        id=6,
        name="build_wiki_index",
        display_name="3. Build Wiki Index",
        script="build_wiki_index.py",
        default_args={
            "format": "arrow",
        },
        required_paths=["wiki-path", "output-dir"],
        output_path_key="output_dir",
        depends_on=[2],  # Only depends on Wiki download
        description="Convert Wikipedia to Arrow format for faster retrieval",
    ),
    StepConfig(
        id=7,
        name="build_gtr_index",
        display_name="4. Build GTR Index",
        script="build_gtr_index.py",
        default_args={
            "model-path": "sentence-transformers/gtr-t5-base",
            "batch-size": 512,
            "device": "cuda",
        },
        required_paths=["wiki-arrow-path", "output-dir"],
        output_path_key="output-dir",
        depends_on=[6],
        description="Encode Wikipedia with GTR model and build Faiss index",
    ),
    StepConfig(
        id=8,
        name="generate_retrieval_data",
        display_name="5. Generate Retrieval Training Data",
        script="generate_retrieval_training_data.py",
        default_args={
            "tasks": "all",
            "top_k_negatives": 100,
            "batch_size": 256,
            "use_gpu_index": True,  # Force GPU Faiss for faster search
        },
        required_paths=["index_path", "output_path", "wiki_arrow_path", "filtered_dir"],
        output_path_key="output_path",
        depends_on=[5, 7],  # Depends on filtered data and GTR index
        description="Generate training triplets (query, positive, negative) using GTR zero-shot retrieval",
    ),
    StepConfig(
        id=9,
        name="train_retriever",
        display_name="6. Train GTR Retriever",
        script="train_gtr_retriever.py",
        default_args={
            "learning_rate": 1e-3,
            "steps": 10000,
            "batch_size": 64,
            "gradient_accumulation_steps": 3,
            "warmup_steps": 1000,
            "eval_steps": 200,
            "save_steps": 1000,
            "loss_type": "inbatch_softmax",
            "temperature": 0.01,
            "no_bf16": True,  # Use fp32 for stability
        },
        required_paths=["train_data", "output_dir"],
        output_path_key="output_dir",
        depends_on=[8],  # Depends on generated training data
        description="Fine-tune GTR retriever on KILT tasks",
    ),
    StepConfig(
        id=10,
        name="rebuild_index",
        display_name="7. Rebuild Finetuned Index",
        script="build_gtr_index.py",
        default_args={
            "batch-size": 512,
        },
        required_paths=["model-path", "wiki-arrow-path", "output-dir"],
        output_path_key="output-dir",
        depends_on=[9],
        description="Rebuild Faiss index using fine-tuned GTR model",
    ),
    StepConfig(
        id=11,
        name="precompute",
        display_name="8. Precompute Retrieval Results",
        script=None,  # Script selected based on format: precompute_retrieval.py or precompute_retrieval_for_fid.py
        default_args={
            "format": "fidlight",  # "fidlight" or "fid_pure"
            "tasks": "all",  # "all" or comma-separated: "nq,hotpotqa,triviaqa_support_only"
            "splits": "all",  # "all" (train+validation), "train", "validation", or comma-separated
            "num_passages": 40,  # 40 for FiD-Light, 100 for FiD Pure
            "batch_size": 256,
            "use_mmap": False,  # Default: full load + GPU index for faster batch processing
            "use_multi_gpu": True,  # Use all GPUs for encoding
            "preload_wiki": True,  # Preload wiki for faster processing
        },
        required_paths=["index_path", "model_path", "wiki_arrow_path", "filtered_dir", "output_dir"],
        output_path_key="output_dir",
        depends_on=[10],
        description="Precompute retrieval results (FiD-Light: with index, 40 passages | FiD Pure: answer only, 100 passages)",
    ),
    StepConfig(
        id=12,
        name="train_model",
        display_name="9. Train Model",
        script=None,  # Script is selected dynamically based on algorithm and model
        default_args={
            # Algorithm and model selection (UI will load algorithm-specific defaults)
            "algorithm": "fidlight",  # fidlight, fid_pure, stochastic_rag
            "model": "t5base",  # t5base, t5gemma
            # Common parameters (will be overridden by algorithm-specific defaults)
            "multi_gpu": True,
            "bf16": True,
        },
        required_paths=["precomputed_path", "output_dir"],
        output_path_key="output_dir",
        depends_on=[11],  # Depends on precomputed results
        description="Select algorithm and model backbone for training",
    ),
]


def get_train_script(algorithm: str, model: str) -> Optional[str]:
    """Get the training script for an algorithm and model combination."""
    return TRAIN_SCRIPTS.get((algorithm, model))


def get_eval_script(algorithm: str, model: str, eval_type: str = "single") -> Optional[str]:
    """Get the evaluation script for an algorithm, model, and evaluation type."""
    return EVAL_SCRIPTS.get((algorithm, model, eval_type))


class PipelineOrchestrator:
    """
    Orchestrates the FiDLight training pipeline.

    Responsibilities:
    - Manage pipeline state
    - Start/stop steps
    - Validate step dependencies
    - Handle step transitions
    """

    def __init__(self, state_file: Path = STATE_FILE_PATH):
        """
        Initialize PipelineOrchestrator.

        Args:
            state_file: Path to pipeline state file
        """
        self.state_file = state_file
        self.process_manager = get_process_manager()
        self.step_configs = {s.id: s for s in STEP_CONFIGS}
        self.workspace_config: Optional[WorkspaceConfig] = None

    def set_workspace(self, config: WorkspaceConfig) -> None:
        """
        Set the workspace configuration.

        This updates the state file path and process manager log directory
        to use the workspace directories.

        Args:
            config: WorkspaceConfig to use
        """
        self.workspace_config = config

        # Update state file path to use workspace
        self.state_file = config.get_state_file()

        # Ensure workspace directories exist
        config.ensure_directories()

        # Update process manager to use workspace logs directory
        self.process_manager.set_log_dir(config.get_logs_dir())

    def get_workspace(self) -> Optional[WorkspaceConfig]:
        """Get the current workspace configuration."""
        return self.workspace_config

    def get_checkpoints_dir(self) -> Optional[Path]:
        """Get the checkpoints directory for the current workspace."""
        if self.workspace_config:
            return self.workspace_config.get_checkpoints_dir()
        return None

    def get_results_dir(self) -> Optional[Path]:
        """Get the results directory for the current workspace."""
        if self.workspace_config:
            return self.workspace_config.get_results_dir()
        return None

    def get_step_config(self, step_id: int) -> Optional[StepConfig]:
        """Get step configuration by ID."""
        return self.step_configs.get(step_id)

    def get_step_config_by_name(self, name: str) -> Optional[StepConfig]:
        """Get step configuration by name."""
        for config in STEP_CONFIGS:
            if config.name == name:
                return config
        return None

    def get_state(self) -> PipelineState:
        """
        Get current pipeline state.

        Returns:
            PipelineState (initialized if not exists)
        """
        state = read_pipeline_state(self.state_file)
        if state is None:
            state = initialize_pipeline_state()
            write_pipeline_state(state, self.state_file)
        return state

    def initialize(self, mode: str = "production") -> PipelineState:
        """
        Initialize or reset pipeline state.

        Args:
            mode: "demo" or "production"

        Returns:
            New PipelineState
        """
        state = initialize_pipeline_state(mode)
        write_pipeline_state(state, self.state_file)
        return state

    def can_start_step(self, step_id: int) -> tuple[bool, str]:
        """
        Check if a step can be started.

        Args:
            step_id: Step ID to check

        Returns:
            (can_start, reason)
        """
        state = self.get_state()
        config = self.get_step_config(step_id)

        if config is None:
            return False, f"Unknown step ID: {step_id}"

        step = state.get_step(step_id)
        if step is None:
            return False, f"Step not found in state: {step_id}"

        # Check if already running
        if step.status == StepStatus.RUNNING.value:
            return False, "Step is already running"

        # Check dependencies
        for dep_id in config.depends_on:
            dep_step = state.get_step(dep_id)
            if dep_step is None:
                return False, f"Dependency step {dep_id} not found"
            if dep_step.status not in (StepStatus.COMPLETED.value, StepStatus.SKIPPED.value):
                dep_config = self.get_step_config(dep_id)
                return False, f"Dependency not met: {dep_config.display_name}"

        return True, "OK"

    def validate_step_args(
        self,
        step_id: int,
        args: Dict[str, Any],
    ) -> tuple[bool, str]:
        """
        Validate step arguments.

        Args:
            step_id: Step ID
            args: Arguments to validate

        Returns:
            (is_valid, error_message)
        """
        config = self.get_step_config(step_id)
        if config is None:
            return False, f"Unknown step ID: {step_id}"

        # Check required paths
        for path_key in config.required_paths:
            if path_key not in args:
                return False, f"Missing required path: {path_key}"
            path = Path(args[path_key])
            # Output dirs will be created, skip existence check
            if path_key in ("output_dir", "output-dir", "output_path"):
                continue
            if not path.exists():
                return False, f"Path does not exist: {path}"

        return True, "OK"

    def build_command_args(
        self,
        step_id: int,
        user_args: Dict[str, Any],
    ) -> List[str]:
        """
        Build command line arguments for a step.

        Args:
            step_id: Step ID
            user_args: User-provided arguments

        Returns:
            List of command line arguments
        """
        config = self.get_step_config(step_id)
        if config is None:
            return []

        # Merge default args with user args
        args = {**config.default_args, **user_args}

        # Filter out internal keys that are not CLI arguments
        # These are used for script selection, not passed to the script
        internal_keys = {"algorithm", "model"}
        args = {k: v for k, v in args.items() if k not in internal_keys}

        # Handle resume_from -> --resume conversion
        if "resume_from" in args:
            resume_path = args.pop("resume_from")
            if resume_path:  # Only add if not empty
                args["resume"] = resume_path

        # For training step: if precomputed_path is set, also set precomputed_val_path
        # (train and dev data are typically in the same directory)
        if step_id == 12 and "precomputed_path" in args and "precomputed_val_path" not in args:
            args["precomputed_val_path"] = args["precomputed_path"]

        # Convert to command line format
        cmd_args = []
        for key, value in args.items():
            if isinstance(value, bool):
                if value:
                    cmd_args.append(f"--{key}")
            elif isinstance(value, list):
                # For nargs='+' style arguments: --key item1 item2 item3
                cmd_args.append(f"--{key}")
                for item in value:
                    cmd_args.append(str(item))
            else:
                cmd_args.extend([f"--{key}", str(value)])

        return cmd_args

    def start_step(
        self,
        step_id: int,
        args: Dict[str, Any],
        on_output: Callable[[str], None] = None,
    ) -> tuple[bool, str]:
        """
        Start a pipeline step.

        Args:
            step_id: Step ID to start
            args: Step arguments
            on_output: Callback for output lines

        Returns:
            (success, message)
        """
        config = self.get_step_config(step_id)
        if config is None:
            return False, f"Unknown step ID: {step_id}"

        # Check if can start
        can_start, reason = self.can_start_step(step_id)
        if not can_start:
            return False, reason

        # Validate args
        is_valid, error = self.validate_step_args(step_id, args)
        if not is_valid:
            return False, error

        # Special case: environment check (built-in)
        if step_id == 1:
            return self._run_environment_check()

        # Determine script to run
        script = config.script

        # Special case: Step 11 (precompute) - select script based on format
        if step_id == 11:
            precompute_format = args.get("format", "fidlight")
            if precompute_format == "fid_pure":
                script = "precompute_retrieval_for_fid.py"
            else:
                script = "precompute_retrieval.py"

        # Special case: Step 12 (train_model) - dynamically select script
        if step_id == 12:
            algorithm = args.get("algorithm", "fidlight")
            model = args.get("model", "t5base")
            script = get_train_script(algorithm, model)
            if script is None:
                return False, f"Unknown algorithm/model combination: {algorithm}/{model}"

        # Start subprocess
        if script is None:
            return False, "Step has no script defined"

        try:
            # Filter out non-CLI args
            # Step 11: "format" is used to select script, not passed to CLI
            # Step 12: "algorithm" and "model" are used to select script, not passed to CLI
            exclude_keys = set()
            if step_id == 11:
                exclude_keys.add("format")
            if step_id == 12:
                exclude_keys.update(["algorithm", "model"])
            filtered_args = {k: v for k, v in args.items() if k not in exclude_keys}
            cmd_args = self.build_command_args(step_id, filtered_args)

            # Update state to running
            start_message = "Starting..."

            # Step 11: Show format
            if step_id == 11:
                precompute_format = args.get("format", "fidlight")
                format_display = "FiD-Light (with index)" if precompute_format == "fidlight" else "FiD Pure (answer only)"
                start_message = f"Precomputing {format_display}..."

            # Step 12: Show algorithm and model
            algo_display = ALGORITHM_NAMES.get(args.get("algorithm", ""), "")
            model_display = MODEL_NAMES.get(args.get("model", ""), "")
            resume_from = args.get("resume_from", "")
            if step_id == 12 and algo_display and model_display:
                if resume_from:
                    start_message = f"Resuming {algo_display} + {model_display} from checkpoint..."
                else:
                    start_message = f"Starting {algo_display} + {model_display}..."

            # Store output_dir in extra for loss chart to load history
            extra_data = {}
            if step_id in [9, 12]:
                if "output_dir" in args:
                    extra_data["output_dir"] = args["output_dir"]
            if step_id == 12:
                if algo_display:
                    extra_data["algorithm"] = algo_display
                if model_display:
                    extra_data["model"] = model_display

            update_step_state(
                step_name=config.name,
                status=StepStatus.RUNNING.value,
                progress=0.0,
                message=start_message,
                extra=extra_data if extra_data else None,
                state_file=self.state_file,
            )

            # Update pipeline state
            state = self.get_state()
            state.status = PipelineStatus.RUNNING.value
            state.current_step = step_id
            if state.started_at is None:
                state.started_at = datetime.now().isoformat()
            state.pid = os.getpid()
            write_pipeline_state(state, self.state_file)

            # Start process
            info = self.process_manager.start_process(
                step_name=config.name,
                script=script,
                args=cmd_args,
                on_output=on_output,
            )

            return True, f"Started with PID {info.pid}"

        except Exception as e:
            mark_step_failed(config.name, str(e), self.state_file)
            return False, str(e)

    def _run_environment_check(self) -> tuple[bool, str]:
        """Run built-in environment check."""
        import sys

        results = []

        # Python version
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        results.append(f"Python: {py_version}")

        # Check PyTorch and CUDA
        try:
            import torch
            results.append(f"PyTorch: {torch.__version__}")
            if torch.cuda.is_available():
                results.append(f"CUDA: {torch.version.cuda}")
                results.append(f"GPU Count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    results.append(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                results.append("CUDA: Not available (will use CPU)")
        except ImportError:
            mark_step_failed("environment_check", "PyTorch not installed", self.state_file)
            return False, "PyTorch not installed"

        # Check transformers
        try:
            import transformers
            results.append(f"Transformers: {transformers.__version__}")
        except ImportError:
            mark_step_failed("environment_check", "Transformers not installed", self.state_file)
            return False, "Transformers not installed"

        # Check other dependencies
        deps = ["faiss", "sentence_transformers", "datasets"]
        for dep in deps:
            try:
                module = __import__(dep.replace("-", "_"))
                version = getattr(module, "__version__", "installed")
                results.append(f"{dep}: {version}")
            except ImportError:
                results.append(f"{dep}: Not installed (may be optional)")

        # Mark as completed
        message = " | ".join(results[:3])  # First 3 items for summary
        mark_step_completed("environment_check", state_file=self.state_file)
        update_step_state(
            step_name="environment_check",
            message=message,
            extra={"details": results},
            state_file=self.state_file,
        )

        return True, "\n".join(results)

    def skip_step(
        self,
        step_id: int,
        output_path: str,
        extra: Dict[str, Any] = None,
    ) -> tuple[bool, str]:
        """
        Mark a step as skipped (user has existing output).

        Args:
            step_id: Step ID to skip
            output_path: Path to existing output
            extra: Extra data (found_items, etc.)

        Returns:
            (success, message)
        """
        config = self.get_step_config(step_id)
        if config is None:
            return False, f"Unknown step ID: {step_id}"

        # Validate output path exists
        if not Path(output_path).exists():
            return False, f"Output path does not exist: {output_path}"

        # Mark as skipped
        mark_step_skipped(config.name, output_path, self.state_file, extra=extra)

        return True, f"Step {config.display_name} marked as skipped"

    def stop_step(self, step_id: int) -> tuple[bool, str]:
        """
        Stop a running step.

        Args:
            step_id: Step ID to stop

        Returns:
            (success, message)
        """
        config = self.get_step_config(step_id)
        if config is None:
            return False, f"Unknown step ID: {step_id}"

        # Stop process
        if self.process_manager.is_running(config.name):
            self.process_manager.stop_process(config.name)

        # Update state
        update_step_state(
            step_name=config.name,
            status=StepStatus.FAILED.value,
            message="Stopped by user",
            error="Stopped by user",
            state_file=self.state_file,
        )

        return True, "Step stopped"

    def stop_pipeline(self) -> tuple[bool, str]:
        """
        Stop the entire pipeline.

        Returns:
            (success, message)
        """
        # Stop all running processes
        self.process_manager.stop_all()

        # Update state
        state = self.get_state()
        state.status = PipelineStatus.FAILED.value
        state.error = "Pipeline stopped by user"
        write_pipeline_state(state, self.state_file)

        return True, "Pipeline stopped"

    def reset_step(self, step_id: int) -> tuple[bool, str]:
        """
        Reset a step and all steps that depend on it.

        Args:
            step_id: Step ID to reset

        Returns:
            (success, message)
        """
        config = self.get_step_config(step_id)
        if config is None:
            return False, f"Unknown step ID: {step_id}"

        # Find all steps that need to be reset (this step + all that depend on it)
        steps_to_reset = {step_id}

        # Build dependency graph: which steps depend on which
        def find_dependent_steps(sid):
            dependents = []
            for cfg in STEP_CONFIGS:
                if sid in cfg.depends_on:
                    dependents.append(cfg.id)
            return dependents

        # BFS to find all dependent steps
        queue = [step_id]
        while queue:
            current = queue.pop(0)
            for dependent in find_dependent_steps(current):
                if dependent not in steps_to_reset:
                    steps_to_reset.add(dependent)
                    queue.append(dependent)

        # Reset all steps
        state = self.get_state()
        reset_names = []
        for step in state.steps:
            if step.id in steps_to_reset:
                step.status = StepStatus.PENDING.value
                step.progress = 0.0
                step.message = ""
                step.started_at = None
                step.completed_at = None
                step.duration_seconds = None
                step.extra = {}
                step.output_path = None
                step.error = None
                reset_names.append(step.display_name)

        write_pipeline_state(state, self.state_file)

        return True, f"Reset {len(reset_names)} step(s): {', '.join(reset_names)}"

    def get_step_output(self, step_id: int, tail_lines: int = 100) -> List[str]:
        """
        Get output from a step's log file.

        Args:
            step_id: Step ID
            tail_lines: Number of lines from end

        Returns:
            List of log lines
        """
        config = self.get_step_config(step_id)
        if config is None:
            return []

        return self.process_manager.read_log_file(config.name, tail_lines)

    def is_step_running(self, step_id: int) -> bool:
        """Check if a step is currently running."""
        config = self.get_step_config(step_id)
        if config is None:
            return False
        return self.process_manager.is_running(config.name)

    def get_next_step(self) -> Optional[int]:
        """
        Get the next step that should be run.

        Returns:
            Step ID of next pending step, or None if all done
        """
        state = self.get_state()

        for step in state.steps:
            if step.status == StepStatus.PENDING.value:
                return step.id

        return None

    def run_full_pipeline(
        self,
        step_args: Dict[int, Dict[str, Any]],
        on_step_complete: Callable[[int, bool], None] = None,
    ) -> tuple[bool, str]:
        """
        Run the full pipeline sequentially.

        This is a blocking call that runs all steps.
        For non-blocking, use start_step() for each step.

        Args:
            step_args: Arguments for each step {step_id: {arg: value}}
            on_step_complete: Callback after each step

        Returns:
            (success, message)
        """
        for step_id in range(1, 12):
            config = self.get_step_config(step_id)
            args = step_args.get(step_id, {})

            # Check if already done
            state = self.get_state()
            step = state.get_step(step_id)
            if step and step.status in (StepStatus.COMPLETED.value, StepStatus.SKIPPED.value):
                continue

            # Start step
            success, message = self.start_step(step_id, args)
            if not success:
                return False, f"Step {config.display_name} failed: {message}"

            # Wait for completion (if has script)
            if config.script:
                while self.is_step_running(step_id):
                    import time
                    time.sleep(1)

                # Check result
                state = self.get_state()
                step = state.get_step(step_id)
                if step.status != StepStatus.COMPLETED.value:
                    return False, f"Step {config.display_name} failed"

            if on_step_complete:
                on_step_complete(step_id, True)

        # Mark pipeline as completed
        state = self.get_state()
        state.status = PipelineStatus.COMPLETED.value
        state.completed_at = datetime.now().isoformat()
        write_pipeline_state(state, self.state_file)

        return True, "Pipeline completed successfully"


# Global orchestrator instance
_orchestrator: Optional[PipelineOrchestrator] = None


def get_orchestrator() -> PipelineOrchestrator:
    """Get the global PipelineOrchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PipelineOrchestrator()
    return _orchestrator


if __name__ == "__main__":
    # Test the module
    print("Testing PipelineOrchestrator...")

    orch = PipelineOrchestrator()

    # Initialize (production mode only)
    state = orch.initialize("production")
    print(f"Initialized: {state.mode}, {len(state.steps)} steps")

    # Check step configs
    for config in STEP_CONFIGS:
        print(f"Step {config.id}: {config.display_name}")
        if config.id == 12:
            print("  Script: (dynamic based on algorithm/model)")
            print(f"  Available combinations: {list(TRAIN_SCRIPTS.keys())}")
        else:
            print(f"  Script: {config.script}")
        print(f"  Depends on: {config.depends_on}")

    # Test script mapping
    print("\nTrain script mapping:")
    for (algo, model), script in TRAIN_SCRIPTS.items():
        print(f"  {algo} + {model} -> {script}")

    # Run environment check
    success, message = orch.start_step(1, {})
    print(f"\nEnvironment check: {success}")
    print(message)

    # Check state
    state = orch.get_state()
    step = state.get_step(1)
    print(f"Step 1 status: {step.status}")

    print("\nDone!")
