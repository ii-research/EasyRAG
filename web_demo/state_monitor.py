"""
State Monitor for FiDLight Web Demo
====================================

Monitors pipeline_state.json for changes and notifies UI.

Features:
- File watching with debouncing
- Change detection
- Callback notifications
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass

from .utils.state_io import PipelineState, read_pipeline_state, STATE_FILE_PATH


@dataclass
class StateChange:
    """Represents a state change."""
    old_state: Optional[PipelineState]
    new_state: PipelineState
    changed_steps: list[int]


class StateMonitor:
    """
    Monitors pipeline state file for changes.

    Uses polling (not file system watching) for cross-platform compatibility.
    """

    def __init__(
        self,
        state_file: Path = STATE_FILE_PATH,
        poll_interval: float = 1.0,
    ):
        """
        Initialize StateMonitor.

        Args:
            state_file: Path to state file
            poll_interval: Seconds between polls
        """
        self.state_file = state_file
        self.poll_interval = poll_interval

        self.last_state: Optional[PipelineState] = None
        self.last_modified: float = 0
        self.callbacks: list[Callable[[StateChange], None]] = []
        self._running = False

    def add_callback(self, callback: Callable[[StateChange], None]):
        """Add a callback for state changes."""
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callable[[StateChange], None]):
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    async def start(self):
        """Start monitoring in background."""
        self._running = True
        while self._running:
            await self._check_for_changes()
            await asyncio.sleep(self.poll_interval)

    def stop(self):
        """Stop monitoring."""
        self._running = False

    async def _check_for_changes(self):
        """Check for state file changes."""
        if not self.state_file.exists():
            return

        # Check file modification time
        try:
            mtime = self.state_file.stat().st_mtime
            if mtime <= self.last_modified:
                return
            self.last_modified = mtime
        except OSError:
            return

        # Read new state
        new_state = read_pipeline_state(self.state_file)
        if new_state is None:
            return

        # Detect changes
        changed_steps = self._detect_changed_steps(self.last_state, new_state)

        if changed_steps or self.last_state is None:
            change = StateChange(
                old_state=self.last_state,
                new_state=new_state,
                changed_steps=changed_steps,
            )

            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(change)
                except Exception as e:
                    print(f"State callback error: {e}")

        self.last_state = new_state

    def _detect_changed_steps(
        self,
        old_state: Optional[PipelineState],
        new_state: PipelineState,
    ) -> list[int]:
        """Detect which steps have changed."""
        if old_state is None:
            return [s.id for s in new_state.steps]

        changed = []
        for new_step in new_state.steps:
            old_step = old_state.get_step(new_step.id)
            if old_step is None:
                changed.append(new_step.id)
            elif (
                old_step.status != new_step.status
                or old_step.progress != new_step.progress
                or old_step.message != new_step.message
            ):
                changed.append(new_step.id)

        return changed

    def get_current_state(self) -> Optional[PipelineState]:
        """Get the last known state."""
        return self.last_state


# Global monitor instance
_monitor: Optional[StateMonitor] = None


def get_monitor() -> StateMonitor:
    """Get the global StateMonitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = StateMonitor()
    return _monitor


if __name__ == "__main__":
    # Test the monitor
    import asyncio

    def on_change(change: StateChange):
        print(f"State changed: {len(change.changed_steps)} steps modified")
        for step_id in change.changed_steps:
            step = change.new_state.get_step(step_id)
            print(f"  Step {step_id}: {step.status} - {step.progress}%")

    async def main():
        monitor = StateMonitor()
        monitor.add_callback(on_change)

        print("Monitoring state file... (Ctrl+C to stop)")
        await monitor.start()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped")
