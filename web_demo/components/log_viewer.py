"""
Log Viewer Component
=====================

Real-time scrolling log display with:
- Auto-scroll to latest
- Search/filter
- Copy to clipboard
"""

from nicegui import ui
from typing import List, Optional
import asyncio


class LogViewer:
    """
    Real-time log viewer component.

    Features:
    - Displays log lines in monospace font
    - Auto-scrolls to bottom
    - Search/filter functionality
    - Pause auto-scroll
    """

    def __init__(self, max_lines: int = 500):
        """
        Initialize LogViewer.

        Args:
            max_lines: Maximum number of lines to keep in memory
        """
        self.max_lines = max_lines
        self.lines: List[str] = []
        self.auto_scroll = True
        self.filter_text = ""

        self._build_ui()

    def _build_ui(self):
        """Build the log viewer UI."""
        with ui.card().classes("w-full"):
            # Header
            with ui.row().classes("w-full items-center mb-2"):
                ui.label("Log Output").classes("text-h6")
                ui.space()

                # Search
                self.search_input = ui.input(
                    placeholder="Filter...",
                    on_change=self._on_filter_change,
                ).classes("w-40").props("dense")

                # Auto-scroll toggle
                self.scroll_switch = ui.switch(
                    "Auto-scroll",
                    value=self.auto_scroll,
                    on_change=self._on_scroll_toggle,
                )

                # Clear button
                ui.button(
                    icon="clear_all",
                    on_click=self.clear,
                ).props("flat dense")

            # Log content area
            with ui.scroll_area().classes("h-64 w-full bg-gray-900") as self.scroll_area:
                self.log_container = ui.column().classes("w-full p-2")
                # Use code element for log display (better compatibility)
                self.log_content = ui.code("", language=None).classes(
                    "text-green-400 font-mono text-xs w-full"
                ).style("background: transparent; white-space: pre-wrap; word-break: break-all;")

    def _on_filter_change(self, e):
        """Handle filter text change."""
        self.filter_text = e.value.lower() if e.value else ""
        self._refresh_display()

    def _on_scroll_toggle(self, e):
        """Handle auto-scroll toggle."""
        self.auto_scroll = e.value

    def add_line(self, line: str):
        """Add a log line."""
        self.lines.append(line)

        # Trim if too many lines
        if len(self.lines) > self.max_lines:
            self.lines = self.lines[-self.max_lines:]

        self._refresh_display()

    def add_lines(self, lines: List[str]):
        """Add multiple log lines."""
        self.lines.extend(lines)

        # Trim if too many lines
        if len(self.lines) > self.max_lines:
            self.lines = self.lines[-self.max_lines:]

        self._refresh_display()

    def set_lines(self, lines: List[str]):
        """Set all log lines (replaces existing)."""
        self.lines = lines[-self.max_lines:]
        self._refresh_display()

    def clear(self):
        """Clear all log lines."""
        self.lines = []
        self._refresh_display()

    def _refresh_display(self):
        """Refresh the displayed content."""
        # Filter lines
        if self.filter_text:
            display_lines = [
                line for line in self.lines
                if self.filter_text in line.lower()
            ]
        else:
            display_lines = self.lines

        # Update content (ui.code uses set_content method)
        content = "\n".join(display_lines)
        self.log_content.set_content(content)

        # Auto-scroll if enabled
        if self.auto_scroll:
            self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        """Scroll to bottom of log area."""
        ui.run_javascript(
            """
            const area = document.querySelector('.q-scrollarea__container');
            if (area) area.scrollTop = area.scrollHeight;
            """
        )


class LossChart:
    """
    Real-time loss curve chart using ECharts.
    """

    def __init__(self):
        """Initialize LossChart."""
        self.steps: List[int] = []
        self.losses: List[float] = []
        self.chart: Optional[ui.echart] = None

        self._build_ui()

    def _build_ui(self):
        """Build the chart UI."""
        with ui.card().classes("w-full"):
            ui.label("Training Loss").classes("text-h6 mb-2")

            self.chart = ui.echart({
                "xAxis": {
                    "type": "value",
                    "name": "Step",
                    "nameLocation": "middle",
                    "nameGap": 25,
                },
                "yAxis": {
                    "type": "value",
                    "name": "Loss",
                    "nameLocation": "middle",
                    "nameGap": 40,
                },
                "series": [{
                    "type": "line",
                    "data": [],
                    "smooth": True,
                    "lineStyle": {"color": "#3b82f6"},
                    "areaStyle": {"color": "rgba(59, 130, 246, 0.1)"},
                }],
                "tooltip": {
                    "trigger": "axis",
                },
                "grid": {
                    "left": 60,
                    "right": 20,
                    "top": 20,
                    "bottom": 40,
                },
            }).classes("h-64")

    def add_point(self, step: int, loss: float):
        """Add a data point."""
        self.steps.append(step)
        self.losses.append(loss)
        self._update_chart()

    def set_data(self, steps: List[int], losses: List[float]):
        """Set all data points."""
        self.steps = steps
        self.losses = losses
        self._update_chart()

    def clear(self):
        """Clear all data."""
        self.steps = []
        self.losses = []
        self._update_chart()

    def _update_chart(self):
        """Update the chart with current data."""
        data = [[s, l] for s, l in zip(self.steps, self.losses)]
        self.chart.options["series"][0]["data"] = data
        self.chart.update()
