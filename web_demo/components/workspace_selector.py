"""
Workspace Selector Component
============================

UI component for selecting or creating a workspace directory.
All pipeline outputs (state, logs, checkpoints, results) will be saved to the workspace.
"""

from nicegui import ui
from typing import Callable, Optional
from pathlib import Path
from datetime import datetime

from ..utils.state_io import (
    WorkspaceConfig,
    get_recent_workspaces,
    set_current_workspace,
    load_workspace_config,
    get_current_workspace,
    PROJECT_ROOT,
)


class WorkspaceSelector:
    """
    Workspace selector component.

    Features:
    - Create new workspace with auto-generated name
    - Select existing workspace from dropdown
    - Show current workspace path and info
    - Callback when workspace changes
    """

    def __init__(
        self,
        on_workspace_change: Optional[Callable[[WorkspaceConfig], None]] = None,
        default_workspace: Optional[str] = None,
    ):
        """
        Initialize workspace selector.

        Args:
            on_workspace_change: Callback when workspace is selected/created
            default_workspace: Default workspace path to use
        """
        self.on_workspace_change = on_workspace_change
        self.current_config: Optional[WorkspaceConfig] = None

        # UI references
        self.workspace_input = None
        self.workspace_select = None
        self.info_label = None
        self.card = None

        # Initialize with default or auto-generate
        if default_workspace:
            self._set_workspace(default_workspace)
        else:
            # Check for existing workspaces
            existing = get_recent_workspaces()
            if existing:
                # Use most recent workspace
                self._set_workspace(existing[0]["path"])
            else:
                # Create new workspace with timestamp
                new_path = PROJECT_ROOT / "outputs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self._set_workspace(str(new_path))

        self._build_ui()

    def _set_workspace(self, workspace_path: str) -> bool:
        """Set the current workspace."""
        try:
            self.current_config = set_current_workspace(Path(workspace_path))
            if self.on_workspace_change and self.current_config:
                self.on_workspace_change(self.current_config)
            return True
        except Exception as e:
            print(f"Error setting workspace: {e}")
            return False

    def _build_ui(self):
        """Build the workspace selector UI."""
        with ui.card().classes("w-full mb-4") as self.card:
            with ui.row().classes("w-full items-center gap-4"):
                ui.icon("folder_open").classes("text-2xl text-blue-500")
                ui.label("Workspace").classes("text-lg font-bold")

                # Workspace path input (default path is already set in __init__)
                self.workspace_input = ui.input(
                    placeholder="Workspace directory...",
                    value=str(self.current_config.workspace_dir) if self.current_config else "",
                ).classes("flex-grow").props("dense outlined")

                # Existing workspaces dropdown
                existing = get_recent_workspaces()
                if existing:
                    options = {w["path"]: w["name"] for w in existing}
                    self.workspace_select = ui.select(
                        options=options,
                        value=str(self.current_config.workspace_dir) if self.current_config else None,
                        on_change=self._on_select_change,
                    ).classes("w-48").props("dense outlined label='Recent'")

                # Apply button
                ui.button(
                    "Apply",
                    icon="check",
                    on_click=self._apply_workspace,
                ).props("dense color=primary")

            # Info row
            with ui.row().classes("w-full items-center gap-2 mt-2"):
                self.info_label = ui.label().classes("text-sm text-gray-500 dark:text-gray-400")
                self._update_info_label()

    def _update_info_label(self):
        """Update the info label with workspace details."""
        if self.current_config:
            workspace_path = Path(self.current_config.workspace_dir)
            if workspace_path.exists():
                # Count files/dirs
                state_exists = (workspace_path / "pipeline_state.json").exists()
                checkpoints_dir = self.current_config.get_checkpoints_dir()
                num_checkpoints = len(list(checkpoints_dir.glob("*"))) if checkpoints_dir.exists() else 0

                info_parts = []
                if state_exists:
                    info_parts.append("Has state")
                if num_checkpoints > 0:
                    info_parts.append(f"{num_checkpoints} checkpoints")
                if self.current_config.description:
                    info_parts.append(self.current_config.description)

                self.info_label.text = " | ".join(info_parts) if info_parts else "Empty workspace"
            else:
                self.info_label.text = "New workspace (will be created)"
        else:
            self.info_label.text = "No workspace selected"

    def _on_select_change(self, e):
        """Handle workspace selection from dropdown."""
        if e.value:
            self.workspace_input.value = e.value

    def _create_new_workspace(self):
        """Create a new workspace with timestamp."""
        new_path = PROJECT_ROOT / "outputs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.workspace_input.value = str(new_path)
        ui.notify(f"New workspace path generated", type="info")

    def _apply_workspace(self):
        """Apply the selected/entered workspace."""
        workspace_path = self.workspace_input.value.strip()
        if not workspace_path:
            ui.notify("Please enter a workspace path", type="warning")
            return

        if self._set_workspace(workspace_path):
            self._update_info_label()

            # Update dropdown if it exists
            if self.workspace_select:
                existing = get_recent_workspaces()
                self.workspace_select.options = {w["path"]: w["name"] for w in existing}
                self.workspace_select.value = workspace_path

            ui.notify(f"Workspace set to: {workspace_path}", type="positive")
        else:
            ui.notify("Failed to set workspace", type="negative")

    def get_config(self) -> Optional[WorkspaceConfig]:
        """Get the current workspace config."""
        return self.current_config

    def get_checkpoints_dir(self) -> Optional[Path]:
        """Get the checkpoints directory for the current workspace."""
        if self.current_config:
            return self.current_config.get_checkpoints_dir()
        return None

    def get_results_dir(self) -> Optional[Path]:
        """Get the results directory for the current workspace."""
        if self.current_config:
            return self.current_config.get_results_dir()
        return None

    def get_logs_dir(self) -> Optional[Path]:
        """Get the logs directory for the current workspace."""
        if self.current_config:
            return self.current_config.get_logs_dir()
        return None
