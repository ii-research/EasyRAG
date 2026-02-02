"""
Process Manager for FiDLight Web Demo
======================================

Manages subprocess lifecycle for training scripts:
- Start training processes in background
- Monitor process status
- Stop/kill processes
- Handle process output and logging
"""

import os
import sys
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue, Empty

import psutil


PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class ProcessInfo:
    """Information about a running process."""
    pid: int
    step_name: str
    script: str
    args: List[str]
    started_at: str
    log_file: Optional[Path] = None
    process: Optional[subprocess.Popen] = None
    return_code: Optional[int] = None
    is_running: bool = True


class ProcessManager:
    """
    Manages training subprocess lifecycle.

    Features:
    - Start processes in background (survives browser close)
    - Monitor process status
    - Capture output to log files
    - Stop processes gracefully or forcefully
    """

    def __init__(self, log_dir: Path = None):
        """
        Initialize ProcessManager.

        Args:
            log_dir: Directory for log files. Defaults to PROJECT_ROOT/logs
        """
        self.log_dir = log_dir or PROJECT_ROOT / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.processes: Dict[str, ProcessInfo] = {}
        self._lock = threading.Lock()
        self._output_threads: Dict[str, threading.Thread] = {}
        self._output_queues: Dict[str, Queue] = {}

    def set_log_dir(self, log_dir: Path) -> None:
        """
        Set the log directory.

        Args:
            log_dir: New directory for log files
        """
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def start_process(
        self,
        step_name: str,
        script: str,
        args: List[str] = None,
        env: Dict[str, str] = None,
        cwd: Path = None,
        on_output: Callable[[str], None] = None,
    ) -> ProcessInfo:
        """
        Start a training process in the background.

        Args:
            step_name: Name of the step (for identification)
            script: Script path (relative to PROJECT_ROOT or absolute)
            args: Command line arguments
            env: Environment variables to add
            cwd: Working directory
            on_output: Callback for output lines

        Returns:
            ProcessInfo with process details

        Raises:
            RuntimeError: If a process for this step is already running
        """
        with self._lock:
            # Check if already running
            if step_name in self.processes and self.is_running(step_name):
                raise RuntimeError(f"Process for step '{step_name}' is already running")

            # Resolve script path
            script_path = Path(script)
            if not script_path.is_absolute():
                script_path = PROJECT_ROOT / script

            if not script_path.exists():
                raise FileNotFoundError(f"Script not found: {script_path}")

            # Prepare command
            cmd = [sys.executable, str(script_path)]
            if args:
                cmd.extend(args)

            # Prepare environment
            process_env = os.environ.copy()
            if env:
                process_env.update(env)

            # Create log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"{step_name}_{timestamp}.log"

            # Prepare working directory
            work_dir = cwd or PROJECT_ROOT

            # Start process
            # Use PIPE for output capture, but also write to log file
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=process_env,
                cwd=str(work_dir),
                bufsize=1,
                universal_newlines=True,
                # Don't create new process group on Windows to allow proper termination
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )

            # Create process info
            info = ProcessInfo(
                pid=process.pid,
                step_name=step_name,
                script=str(script_path),
                args=args or [],
                started_at=datetime.now().isoformat(),
                log_file=log_file,
                process=process,
                is_running=True,
            )

            self.processes[step_name] = info

            # Start output capture thread
            output_queue = Queue()
            self._output_queues[step_name] = output_queue

            output_thread = threading.Thread(
                target=self._capture_output,
                args=(step_name, process, log_file, output_queue, on_output),
                daemon=True,
            )
            output_thread.start()
            self._output_threads[step_name] = output_thread

            return info

    def _capture_output(
        self,
        step_name: str,
        process: subprocess.Popen,
        log_file: Path,
        output_queue: Queue,
        on_output: Callable[[str], None] = None,
    ):
        """Capture process output to log file and queue."""
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                for line in process.stdout:
                    # Write to log file
                    f.write(line)
                    f.flush()

                    # Add to queue (for UI)
                    output_queue.put(line.rstrip())

                    # Call callback if provided
                    if on_output:
                        try:
                            on_output(line.rstrip())
                        except Exception as e:
                            print(f"Output callback error: {e}")

            # Wait for process to finish
            process.wait()

        except Exception as e:
            print(f"Output capture error for {step_name}: {e}")

        finally:
            # Mark process as finished
            with self._lock:
                if step_name in self.processes:
                    self.processes[step_name].return_code = process.returncode
                    self.processes[step_name].is_running = False

    def is_running(self, step_name: str) -> bool:
        """
        Check if a process is still running.

        Args:
            step_name: Name of the step

        Returns:
            True if running
        """
        if step_name not in self.processes:
            return False

        info = self.processes[step_name]

        # Check our recorded state first
        if not info.is_running:
            return False

        # Also check the actual process
        if info.process is not None:
            poll = info.process.poll()
            if poll is not None:
                info.return_code = poll
                info.is_running = False
                return False

        # Check using psutil in case process object is lost
        try:
            p = psutil.Process(info.pid)
            if p.status() == psutil.STATUS_ZOMBIE:
                info.is_running = False
                return False
            return p.is_running()
        except psutil.NoSuchProcess:
            info.is_running = False
            return False

    def get_process_info(self, step_name: str) -> Optional[ProcessInfo]:
        """
        Get process info for a step.

        Args:
            step_name: Name of the step

        Returns:
            ProcessInfo or None
        """
        # Update running status
        self.is_running(step_name)
        return self.processes.get(step_name)

    def get_all_processes(self) -> Dict[str, ProcessInfo]:
        """
        Get all process infos.

        Returns:
            Dict of step_name -> ProcessInfo
        """
        # Update all running statuses
        for step_name in list(self.processes.keys()):
            self.is_running(step_name)
        return self.processes.copy()

    def get_output(self, step_name: str, max_lines: int = 100) -> List[str]:
        """
        Get recent output lines from a process.

        Args:
            step_name: Name of the step
            max_lines: Maximum number of lines to return

        Returns:
            List of output lines
        """
        if step_name not in self._output_queues:
            return []

        queue = self._output_queues[step_name]
        lines = []

        # Drain queue
        while not queue.empty() and len(lines) < max_lines * 10:
            try:
                lines.append(queue.get_nowait())
            except Empty:
                break

        # Return last N lines
        return lines[-max_lines:]

    def read_log_file(self, step_name: str, tail_lines: int = 100) -> List[str]:
        """
        Read tail of log file.

        Args:
            step_name: Name of the step
            tail_lines: Number of lines from end

        Returns:
            List of log lines
        """
        info = self.processes.get(step_name)
        if not info or not info.log_file or not info.log_file.exists():
            return []

        try:
            with open(info.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                return [line.rstrip() for line in lines[-tail_lines:]]
        except Exception as e:
            print(f"Error reading log file: {e}")
            return []

    def stop_process(self, step_name: str, timeout: float = 10.0) -> bool:
        """
        Stop a process gracefully.

        Sends SIGTERM, waits for timeout, then sends SIGKILL if needed.

        Args:
            step_name: Name of the step
            timeout: Seconds to wait before force kill

        Returns:
            True if successfully stopped
        """
        if step_name not in self.processes:
            return True

        info = self.processes[step_name]
        if not info.is_running:
            return True

        try:
            # Get process and children
            try:
                parent = psutil.Process(info.pid)
                children = parent.children(recursive=True)
            except psutil.NoSuchProcess:
                info.is_running = False
                return True

            # Send SIGTERM to parent
            parent.terminate()

            # Wait for graceful shutdown
            gone, alive = psutil.wait_procs([parent] + children, timeout=timeout)

            # Force kill any remaining
            for p in alive:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass

            info.is_running = False
            info.return_code = -signal.SIGTERM

            return True

        except Exception as e:
            print(f"Error stopping process {step_name}: {e}")
            return False

    def stop_all(self, timeout: float = 10.0) -> bool:
        """
        Stop all running processes.

        Args:
            timeout: Seconds to wait before force kill

        Returns:
            True if all stopped successfully
        """
        success = True
        for step_name in list(self.processes.keys()):
            if self.is_running(step_name):
                if not self.stop_process(step_name, timeout):
                    success = False
        return success

    def cleanup(self):
        """Clean up resources."""
        self.stop_all(timeout=5.0)

    def find_running_process_by_pid(self, pid: int) -> Optional[str]:
        """
        Find step name by PID.

        Args:
            pid: Process ID

        Returns:
            Step name or None
        """
        for step_name, info in self.processes.items():
            if info.pid == pid and info.is_running:
                return step_name
        return None

    def recover_process(self, step_name: str, pid: int) -> bool:
        """
        Recover a process that was started in a previous session.

        This allows the Web UI to reconnect to a running process
        after a browser refresh.

        Args:
            step_name: Name of the step
            pid: Process ID to recover

        Returns:
            True if process is found and running
        """
        try:
            process = psutil.Process(pid)
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                # Create ProcessInfo for the recovered process
                info = ProcessInfo(
                    pid=pid,
                    step_name=step_name,
                    script="(recovered)",
                    args=[],
                    started_at=datetime.fromtimestamp(process.create_time()).isoformat(),
                    is_running=True,
                )
                self.processes[step_name] = info
                return True
        except psutil.NoSuchProcess:
            pass
        return False


# Global process manager instance
_process_manager: Optional[ProcessManager] = None


def get_process_manager() -> ProcessManager:
    """Get the global ProcessManager instance."""
    global _process_manager
    if _process_manager is None:
        _process_manager = ProcessManager()
    return _process_manager


if __name__ == "__main__":
    # Test the module
    import tempfile

    print("Testing ProcessManager...")

    # Create a test script
    test_script = Path(tempfile.mktemp(suffix=".py"))
    test_script.write_text("""
import time
import sys

print("Starting test process...")
for i in range(5):
    print(f"Step {i+1}/5")
    sys.stdout.flush()
    time.sleep(1)
print("Done!")
""")

    try:
        pm = ProcessManager()

        # Start process
        info = pm.start_process(
            step_name="test",
            script=str(test_script),
        )
        print(f"Started process with PID: {info.pid}")

        # Monitor for a bit
        for _ in range(10):
            time.sleep(0.5)
            if not pm.is_running("test"):
                break
            output = pm.get_output("test")
            if output:
                print(f"Output: {output[-1]}")

        # Check final status
        info = pm.get_process_info("test")
        print(f"Return code: {info.return_code}")
        print(f"Is running: {info.is_running}")

        # Read log
        log_lines = pm.read_log_file("test")
        print(f"Log file has {len(log_lines)} lines")

    finally:
        test_script.unlink()
        pm.cleanup()

    print("Done!")
