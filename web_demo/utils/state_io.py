"""
State I/O Module for FiDLight Web Demo
=======================================

Provides unified state file protocol for communication between:
- Training processes (writers)
- Web UI (reader)
- Pipeline orchestrator (reader/writer)

Workspace Structure:
    workspace_dir/
    ├── workspace_config.json   (workspace settings)
    ├── pipeline_state.json     (pipeline state)
    ├── logs/                   (step logs)
    ├── checkpoints/            (model checkpoints)
    └── results/                (evaluation results)
"""

import json
import os
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


# Default paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_WORKSPACE_DIR = PROJECT_ROOT / "outputs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
STATE_FILE_PATH = PROJECT_ROOT / "pipeline_state.json"  # Legacy default
RECENT_WORKSPACES_FILE = PROJECT_ROOT / ".recent_workspaces.json"  # Recent workspaces list

# Global workspace manager
_current_workspace: Optional[Path] = None


def _load_recent_workspaces() -> List[Dict[str, Any]]:
    """Load recent workspaces list from file."""
    if not RECENT_WORKSPACES_FILE.exists():
        return []
    try:
        with open(RECENT_WORKSPACES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_recent_workspaces(workspaces: List[Dict[str, Any]]) -> None:
    """Save recent workspaces list to file."""
    try:
        with open(RECENT_WORKSPACES_FILE, "w", encoding="utf-8") as f:
            json.dump(workspaces, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"Warning: Failed to save recent workspaces: {e}")


def add_recent_workspace(workspace_path: str, name: str = None) -> None:
    """Add a workspace to the recent list."""
    workspaces = _load_recent_workspaces()

    # Remove if already exists (to move to top)
    workspaces = [w for w in workspaces if w.get("path") != workspace_path]

    # Add to front
    workspaces.insert(0, {
        "path": workspace_path,
        "name": name or Path(workspace_path).name,
        "last_used": datetime.now().isoformat(),
    })

    # Keep only last 20
    workspaces = workspaces[:20]

    _save_recent_workspaces(workspaces)


def get_recent_workspaces() -> List[Dict[str, Any]]:
    """Get list of recent workspaces (validated to exist)."""
    workspaces = _load_recent_workspaces()

    # Filter to only existing directories
    valid = []
    for w in workspaces:
        path = Path(w.get("path", ""))
        if path.exists() and path.is_dir():
            valid.append(w)

    return valid


@dataclass
class WorkspaceConfig:
    """Workspace configuration."""
    workspace_dir: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""

    # Sub-directory names (relative to workspace_dir)
    logs_dir: str = "logs"
    checkpoints_dir: str = "checkpoints"
    results_dir: str = "results"
    data_dir: str = "data"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkspaceConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def get_state_file(self) -> Path:
        """Get pipeline_state.json path."""
        return Path(self.workspace_dir) / "pipeline_state.json"

    def get_logs_dir(self) -> Path:
        """Get logs directory path."""
        return Path(self.workspace_dir) / self.logs_dir

    def get_checkpoints_dir(self) -> Path:
        """Get checkpoints directory path."""
        return Path(self.workspace_dir) / self.checkpoints_dir

    def get_results_dir(self) -> Path:
        """Get results directory path."""
        return Path(self.workspace_dir) / self.results_dir

    def get_data_dir(self) -> Path:
        """Get data directory path."""
        return Path(self.workspace_dir) / self.data_dir

    def ensure_directories(self) -> None:
        """Create all workspace directories if they don't exist."""
        Path(self.workspace_dir).mkdir(parents=True, exist_ok=True)
        self.get_logs_dir().mkdir(parents=True, exist_ok=True)
        self.get_checkpoints_dir().mkdir(parents=True, exist_ok=True)
        self.get_results_dir().mkdir(parents=True, exist_ok=True)
        self.get_data_dir().mkdir(parents=True, exist_ok=True)


def get_current_workspace() -> Optional[Path]:
    """Get the current workspace directory."""
    global _current_workspace
    return _current_workspace


def set_current_workspace(workspace_dir: Path) -> WorkspaceConfig:
    """
    Set the current workspace directory.

    Args:
        workspace_dir: Path to workspace directory

    Returns:
        WorkspaceConfig for the workspace
    """
    global _current_workspace
    _current_workspace = Path(workspace_dir)

    config_file = _current_workspace / "workspace_config.json"

    if config_file.exists():
        # Load existing config
        config = load_workspace_config(_current_workspace)
    else:
        # Create new config
        config = WorkspaceConfig(workspace_dir=str(_current_workspace))
        config.ensure_directories()
        save_workspace_config(config)

    # Add to recent workspaces list
    add_recent_workspace(str(_current_workspace))

    return config


def load_workspace_config(workspace_dir: Path) -> Optional[WorkspaceConfig]:
    """Load workspace config from directory."""
    config_file = Path(workspace_dir) / "workspace_config.json"

    if not config_file.exists():
        return None

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return WorkspaceConfig.from_dict(data)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load workspace config: {e}")
        return None


def save_workspace_config(config: WorkspaceConfig) -> bool:
    """Save workspace config to directory."""
    config_file = Path(config.workspace_dir) / "workspace_config.json"

    try:
        Path(config.workspace_dir).mkdir(parents=True, exist_ok=True)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        return True
    except (IOError, OSError) as e:
        print(f"Error: Failed to save workspace config: {e}")
        return False


def get_state_file_path() -> Path:
    """Get the current state file path (considering workspace)."""
    global _current_workspace
    if _current_workspace:
        return _current_workspace / "pipeline_state.json"
    return STATE_FILE_PATH


def list_existing_workspaces(base_dir: Path = None) -> List[Dict[str, Any]]:
    """
    List all existing workspaces in the outputs directory.

    Returns:
        List of workspace info dicts with keys: path, created_at, description
    """
    if base_dir is None:
        base_dir = PROJECT_ROOT / "outputs"

    workspaces = []

    if not base_dir.exists():
        return workspaces

    for item in sorted(base_dir.iterdir(), reverse=True):
        if item.is_dir():
            config = load_workspace_config(item)
            if config:
                workspaces.append({
                    "path": str(item),
                    "name": item.name,
                    "created_at": config.created_at,
                    "description": config.description,
                })
            elif (item / "pipeline_state.json").exists():
                # Legacy workspace without config
                workspaces.append({
                    "path": str(item),
                    "name": item.name,
                    "created_at": None,
                    "description": "(Legacy workspace)",
                })

    return workspaces


class StepStatus(str, Enum):
    """Step execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StepState:
    """State of a single pipeline step."""
    id: int
    name: str
    display_name: str
    status: str = StepStatus.PENDING.value
    progress: float = 0.0
    message: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    output_path: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepState":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PipelineState:
    """State of the entire pipeline."""
    mode: str = "production"  # "demo" or "production"
    status: str = PipelineStatus.IDLE.value
    current_step: int = 0
    total_steps: int = 12
    pid: Optional[int] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    steps: List[StepState] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": self.mode,
            "status": self.status,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "pid": self.pid,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "steps": [s.to_dict() for s in self.steps],
            "last_updated": self.last_updated,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineState":
        """Create from dictionary."""
        steps = [StepState.from_dict(s) for s in data.get("steps", [])]
        return cls(
            mode=data.get("mode", "production"),
            status=data.get("status", PipelineStatus.IDLE.value),
            current_step=data.get("current_step", 0),
            total_steps=data.get("total_steps", 10),
            pid=data.get("pid"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            steps=steps,
            last_updated=data.get("last_updated", datetime.now().isoformat()),
            error=data.get("error"),
        )

    def get_step(self, step_id: int) -> Optional[StepState]:
        """Get step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_step_by_name(self, name: str) -> Optional[StepState]:
        """Get step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None


# Step definitions (12 steps)
STEP_DEFINITIONS = [
    {"id": 1, "name": "environment_check", "display_name": "1. Environment Check"},
    {"id": 2, "name": "download_wiki", "display_name": "2.1 Download Wikipedia"},
    {"id": 3, "name": "download_tasks", "display_name": "2.2 Download Task Datasets"},
    {"id": 4, "name": "fix_triviaqa", "display_name": "2.3 Fix TriviaQA"},
    {"id": 5, "name": "filter_data", "display_name": "2.4 Filter Data"},
    {"id": 6, "name": "build_wiki_index", "display_name": "3. Build Wiki Index"},
    {"id": 7, "name": "build_gtr_index", "display_name": "4. Build GTR Index"},
    {"id": 8, "name": "generate_retrieval_data", "display_name": "5. Generate Retrieval Training Data"},
    {"id": 9, "name": "train_retriever", "display_name": "6. Train GTR Retriever"},
    {"id": 10, "name": "rebuild_index", "display_name": "7. Rebuild Finetuned Index"},
    {"id": 11, "name": "precompute", "display_name": "8. Precompute Retrieval Results"},
    {"id": 12, "name": "train_model", "display_name": "9. Train Model"},
]


def initialize_pipeline_state(mode: str = "production") -> PipelineState:
    """
    Initialize a new pipeline state with all steps.

    Args:
        mode: "demo" or "production"

    Returns:
        Initialized PipelineState
    """
    steps = [
        StepState(
            id=step["id"],
            name=step["name"],
            display_name=step["display_name"],
        )
        for step in STEP_DEFINITIONS
    ]

    return PipelineState(
        mode=mode,
        status=PipelineStatus.IDLE.value,
        current_step=0,
        total_steps=len(steps),
        steps=steps,
        last_updated=datetime.now().isoformat(),
    )


def read_pipeline_state(state_file: Path = STATE_FILE_PATH) -> Optional[PipelineState]:
    """
    Read pipeline state from file.

    Args:
        state_file: Path to state file

    Returns:
        PipelineState or None if file doesn't exist
    """
    if not state_file.exists():
        return None

    try:
        with open(state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PipelineState.from_dict(data)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to read state file: {e}")
        return None


def write_pipeline_state(state: PipelineState, state_file: Path = STATE_FILE_PATH) -> bool:
    """
    Write pipeline state to file atomically.

    Uses atomic write (write to temp, then rename) to prevent corruption.

    Args:
        state: PipelineState to write
        state_file: Path to state file

    Returns:
        True if successful
    """
    state.last_updated = datetime.now().isoformat()

    try:
        # Write to temp file first
        fd, temp_path = tempfile.mkstemp(suffix=".json", dir=state_file.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)

            # Atomic rename
            shutil.move(temp_path, state_file)
            return True
        except Exception:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    except (IOError, OSError) as e:
        print(f"Error: Failed to write state file: {e}")
        return False


def update_step_state(
    step_name: str,
    progress: float = None,
    message: str = None,
    status: str = None,
    extra: Dict[str, Any] = None,
    error: str = None,
    output_path: str = None,
    state_file: Path = STATE_FILE_PATH,
) -> bool:
    """
    Update a single step's state. Used by training scripts.

    Args:
        step_name: Name of the step (e.g., "train_fidlight")
        progress: Progress percentage (0-100)
        message: Status message
        status: Step status
        extra: Extra data (loss, lr, etc.)
        error: Error message if failed
        output_path: Path to output files
        state_file: Path to state file

    Returns:
        True if successful
    """
    state = read_pipeline_state(state_file)
    if state is None:
        print(f"Warning: State file not found, cannot update step {step_name}")
        return False

    step = state.get_step_by_name(step_name)
    if step is None:
        print(f"Warning: Step {step_name} not found in state")
        return False

    # Update fields
    if progress is not None:
        step.progress = progress
    if message is not None:
        step.message = message
    if status is not None:
        step.status = status
        if status == StepStatus.RUNNING.value and step.started_at is None:
            step.started_at = datetime.now().isoformat()
        elif status in (StepStatus.COMPLETED.value, StepStatus.FAILED.value):
            step.completed_at = datetime.now().isoformat()
            if step.started_at:
                start = datetime.fromisoformat(step.started_at)
                end = datetime.fromisoformat(step.completed_at)
                step.duration_seconds = (end - start).total_seconds()
    if extra is not None:
        step.extra.update(extra)
    if error is not None:
        step.error = error
    if output_path is not None:
        step.output_path = output_path

    return write_pipeline_state(state, state_file)


def mark_step_completed(
    step_name: str,
    output_path: str = None,
    state_file: Path = STATE_FILE_PATH,
) -> bool:
    """
    Mark a step as completed.

    Args:
        step_name: Name of the step
        output_path: Path to output files
        state_file: Path to state file

    Returns:
        True if successful
    """
    return update_step_state(
        step_name=step_name,
        progress=100.0,
        status=StepStatus.COMPLETED.value,
        message="Completed",
        output_path=output_path,
        state_file=state_file,
    )


def mark_step_failed(
    step_name: str,
    error: str,
    state_file: Path = STATE_FILE_PATH,
) -> bool:
    """
    Mark a step as failed.

    Args:
        step_name: Name of the step
        error: Error message
        state_file: Path to state file

    Returns:
        True if successful
    """
    return update_step_state(
        step_name=step_name,
        status=StepStatus.FAILED.value,
        error=error,
        state_file=state_file,
    )


def mark_step_skipped(
    step_name: str,
    output_path: str,
    state_file: Path = STATE_FILE_PATH,
    extra: Dict[str, Any] = None,
) -> bool:
    """
    Mark a step as skipped (user provided existing output).

    Args:
        step_name: Name of the step
        output_path: Path to existing output files
        state_file: Path to state file
        extra: Extra data (found_items, etc.)

    Returns:
        True if successful
    """
    return update_step_state(
        step_name=step_name,
        progress=100.0,
        status=StepStatus.SKIPPED.value,
        message="Skipped (using existing output)",
        output_path=output_path,
        extra=extra,
        state_file=state_file,
    )


# Convenience function for training scripts
def create_progress_callback(step_name: str, total_steps: int, state_file: Path = STATE_FILE_PATH):
    """
    Create a progress callback function for training scripts.

    Usage in training script:
        from web_demo.utils.state_io import create_progress_callback

        progress_callback = create_progress_callback("train_fidlight", total_steps=50000)

        for step in range(total_steps):
            # ... training code ...
            progress_callback(step, loss=loss, lr=lr)

    Args:
        step_name: Name of the step
        total_steps: Total number of training steps
        state_file: Path to state file

    Returns:
        Callback function
    """
    last_update = [0]  # Use list for mutable closure
    update_interval = max(1, total_steps // 1000)  # Update at most 1000 times

    def callback(current_step: int, **extra):
        # Only update every N steps to avoid excessive I/O
        if current_step - last_update[0] < update_interval and current_step < total_steps - 1:
            return

        last_update[0] = current_step
        progress = (current_step + 1) / total_steps * 100

        # Build message
        parts = [f"Step {current_step + 1}/{total_steps}"]
        if "loss" in extra:
            parts.append(f"Loss: {extra['loss']:.4f}")
        if "lr" in extra:
            parts.append(f"LR: {extra['lr']:.2e}")
        message = " | ".join(parts)

        update_step_state(
            step_name=step_name,
            progress=progress,
            message=message,
            status=StepStatus.RUNNING.value,
            extra=extra,
            state_file=state_file,
        )

    return callback


if __name__ == "__main__":
    # Test the module
    print("Testing state_io module...")

    # Initialize state
    state = initialize_pipeline_state(mode="demo")
    print(f"Initialized state: {state.mode}, {len(state.steps)} steps")

    # Write state
    success = write_pipeline_state(state)
    print(f"Write state: {'OK' if success else 'FAILED'}")

    # Read state
    loaded = read_pipeline_state()
    print(f"Read state: {loaded.mode if loaded else 'FAILED'}")

    # Update step
    success = update_step_state(
        step_name="train_fidlight",
        progress=50.0,
        message="Training step 25000/50000",
        status=StepStatus.RUNNING.value,
        extra={"loss": 0.0234, "lr": 1e-3},
    )
    print(f"Update step: {'OK' if success else 'FAILED'}")

    # Verify update
    loaded = read_pipeline_state()
    step = loaded.get_step_by_name("train_fidlight")
    print(f"Step progress: {step.progress}%, message: {step.message}")

    print("Done!")
