"""
Pipeline Overview Component
============================

Displays the 8-step training pipeline as cards with:
- Step name and status
- Progress bar
- Start/Skip buttons
"""

from nicegui import ui
from typing import Callable, Optional
from ..utils.state_io import StepStatus


class StepCard:
    """A single step card in the pipeline overview."""

    STATUS_ICONS = {
        StepStatus.PENDING.value: "hourglass_empty",
        StepStatus.RUNNING.value: "sync",
        StepStatus.COMPLETED.value: "check_circle",
        StepStatus.FAILED.value: "error",
        StepStatus.SKIPPED.value: "skip_next",
    }

    STATUS_COLORS = {
        StepStatus.PENDING.value: "grey",
        StepStatus.RUNNING.value: "blue",
        StepStatus.COMPLETED.value: "green",
        StepStatus.FAILED.value: "red",
        StepStatus.SKIPPED.value: "orange",
    }

    def __init__(
        self,
        step_id: int,
        name: str,
        display_name: str,
        on_start: Callable[[int], None],
        on_skip: Callable[[int], None],
        on_reset: Callable[[int], None] = None,
        on_stop: Callable[[int], None] = None,
    ):
        self.step_id = step_id
        self.name = name
        self.display_name = display_name
        self.on_start = on_start
        self.on_skip = on_skip
        self.on_reset = on_reset
        self.on_stop = on_stop

        self.status = StepStatus.PENDING.value
        self.progress = 0.0
        self.message = ""

        self._build_ui()

    def _build_ui(self):
        """Build the card UI."""
        with ui.card().classes("w-full p-2 mb-2") as self.card:
            with ui.row().classes("w-full items-center"):
                # Step number
                ui.badge(str(self.step_id)).classes("mr-2")

                # Status icon
                self.status_icon = ui.icon(
                    self.STATUS_ICONS[self.status]
                ).classes("mr-2")

                # Step name
                with ui.column().classes("flex-grow"):
                    ui.label(self.display_name).classes("font-bold text-sm dark:text-gray-200")
                    self.message_label = ui.label(self.message).classes(
                        "text-xs text-gray-500 dark:text-gray-400"
                    )

                # Buttons
                with ui.row().classes("gap-1"):
                    self.start_btn = ui.button(
                        icon="play_arrow",
                        on_click=lambda: self.on_start(self.step_id),
                    ).props("flat dense").classes("text-blue")
                    self.start_btn.tooltip("Start")

                    # Stop button (shown when running)
                    self.stop_btn = ui.button(
                        icon="stop",
                        on_click=lambda: self.on_stop(self.step_id) if self.on_stop else None,
                    ).props("flat dense").classes("text-red")
                    self.stop_btn.visible = False
                    self.stop_btn.tooltip("Stop")

                    self.skip_btn = ui.button(
                        icon="skip_next",
                        on_click=lambda: self.on_skip(self.step_id),
                    ).props("flat dense").classes("text-orange")
                    self.skip_btn.tooltip("Skip with existing data")

                    # Reset button (shown for completed/skipped/failed/running steps)
                    self.reset_btn = ui.button(
                        icon="restart_alt",
                        on_click=lambda: self.on_reset(self.step_id) if self.on_reset else None,
                    ).props("flat dense").classes("text-grey")
                    self.reset_btn.visible = False
                    self.reset_btn.tooltip("Reset this and subsequent steps")

            # Progress bar
            self.progress_bar = ui.linear_progress(
                value=self.progress / 100
            ).classes("w-full mt-1")

    def update(self, status: str, progress: float, message: str):
        """Update the card state."""
        self.status = status
        self.progress = progress
        self.message = message

        # Update UI
        self.status_icon.name = self.STATUS_ICONS.get(status, "help")
        self.status_icon.classes(
            replace=f"text-{self.STATUS_COLORS.get(status, 'grey')}"
        )

        self.message_label.text = message
        self.progress_bar.value = progress / 100

        # Update button states
        # Remove previous border and background classes
        self.card.classes(remove="border-l-4 border-blue-500 border-red-500 border-green-500 border-orange-500 bg-gray-50 dark:bg-gray-800")

        if status == StepStatus.RUNNING.value:
            self.start_btn.visible = False
            self.stop_btn.visible = True
            self.skip_btn.disable()
            self.reset_btn.visible = True  # Allow reset for stuck steps
            self.card.classes(add="border-l-4 border-blue-500")
        elif status == StepStatus.COMPLETED.value:
            # Allow re-running completed steps for debugging
            self.start_btn.visible = True
            self.start_btn.enable()
            self.stop_btn.visible = False
            self.skip_btn.enable()
            self.reset_btn.visible = True
            self.card.classes(add="border-l-4 border-green-500 bg-gray-50 dark:bg-gray-800")
        elif status == StepStatus.SKIPPED.value:
            # Allow re-configuring skipped steps
            self.start_btn.visible = True
            self.start_btn.enable()
            self.stop_btn.visible = False
            self.skip_btn.enable()
            self.reset_btn.visible = True
            self.card.classes(add="border-l-4 border-orange-500 bg-gray-50 dark:bg-gray-800")
        elif status == StepStatus.FAILED.value:
            self.start_btn.visible = True
            self.start_btn.enable()
            self.stop_btn.visible = False
            self.skip_btn.enable()
            self.reset_btn.visible = True
            self.card.classes(add="border-l-4 border-red-500")
        else:
            # PENDING state
            self.start_btn.visible = True
            self.start_btn.enable()
            self.stop_btn.visible = False
            self.skip_btn.enable()
            self.reset_btn.visible = False


class PipelineOverview:
    """
    Pipeline overview component showing all steps.
    """

    def __init__(
        self,
        on_start_step: Callable[[int], None],
        on_skip_step: Callable[[int], None],
        on_start_all: Callable[[], None],
        on_stop_all: Callable[[], None],
        on_reset_step: Callable[[int], None] = None,
        on_stop_step: Callable[[int], None] = None,
    ):
        self.on_start_step = on_start_step
        self.on_skip_step = on_skip_step
        self.on_start_all = on_start_all
        self.on_stop_all = on_stop_all
        self.on_reset_step = on_reset_step
        self.on_stop_step = on_stop_step

        self.step_cards: dict[int, StepCard] = {}
        self._build_ui()

    def _build_ui(self):
        """Build the overview UI."""
        from ..pipeline_orchestrator import STEP_CONFIGS

        with ui.card().classes("w-full"):
            # Header
            with ui.row().classes("w-full items-center mb-4"):
                ui.label("Training Pipeline").classes("text-h6")
                ui.space()
                ui.button(
                    "Start All",
                    icon="play_arrow",
                    on_click=self.on_start_all,
                ).props("color=primary")
                ui.button(
                    "Stop",
                    icon="stop",
                    on_click=self.on_stop_all,
                ).props("color=negative outline")

            # Current step status
            self.overall_label = ui.label("Ready").classes(
                "text-sm text-gray-500 dark:text-gray-400"
            )

            ui.separator()

            # Step cards (no scroll, show all)
            with ui.column().classes("w-full"):
                for config in STEP_CONFIGS:
                    card = StepCard(
                        step_id=config.id,
                        name=config.name,
                        display_name=config.display_name,
                        on_start=self.on_start_step,
                        on_skip=self.on_skip_step,
                        on_reset=self.on_reset_step,
                        on_stop=self.on_stop_step,
                    )
                    self.step_cards[config.id] = card

    def update_step(self, step_id: int, status: str, progress: float, message: str):
        """Update a single step."""
        if step_id in self.step_cards:
            self.step_cards[step_id].update(status, progress, message)
        self._update_overall()

    def update_from_state(self, state):
        """Update all steps from pipeline state."""
        for step in state.steps:
            self.update_step(
                step.id,
                step.status,
                step.progress,
                step.message,
            )
        self._update_overall()

    def _update_overall(self):
        """Update overall status display."""
        # Find running step
        running_step = None
        for step_id, card in self.step_cards.items():
            if card.status == StepStatus.RUNNING.value:
                running_step = (step_id, card.display_name)
                break

        # Count completed steps
        completed = sum(
            1 for card in self.step_cards.values()
            if card.status in (StepStatus.COMPLETED.value, StepStatus.SKIPPED.value)
        )
        total = len(self.step_cards)

        if running_step:
            self.overall_label.text = f"Current: {running_step[1]}"
        elif completed == total:
            self.overall_label.text = "All complete"
        elif completed > 0:
            self.overall_label.text = f"Completed {completed}/{total} steps"
        else:
            self.overall_label.text = "Ready"
