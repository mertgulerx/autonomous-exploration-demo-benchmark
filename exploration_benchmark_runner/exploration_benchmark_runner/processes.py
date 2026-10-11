# Copyright 2026 Mert Güler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ownership-based cleanup, including launch children in their own sessions.

An inherited random ownership token closes the fork/reparent race. PID start time
is checked before every signal; executable names are never cleanup selectors.
"""

import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
import psutil


class ProcessScope:
    def __init__(self, token=None):
        self.token = token or uuid.uuid4().hex
        self.tracked = {}
        self.roots = []
        self.logs = []
        self.commands = []

    def start(self, command, environment, log, cwd=None):
        env = dict(environment, BENCHMARK_PROCESS_SCOPE=self.token)
        handle = open(log, "a", buffering=1)
        try:
            process = subprocess.Popen(
                command,
                env=env,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            handle.close()
            raise
        self.logs.append(handle)
        self.roots.append(process)
        self.commands.append(list(command))
        self.tracked[process.pid] = psutil.Process(process.pid).create_time()
        return process

    def discover(self):
        # Descendants may outlive ros2 launch, or create new process groups.
        for process in psutil.process_iter(["pid", "uids", "create_time", "status"]):
            try:
                if (
                    process.pid == os.getpid()
                    or process.info["uids"].real != os.getuid()
                ):
                    continue
                if process.info["status"] == psutil.STATUS_ZOMBIE:
                    continue
                environment = process.environ()
                if (
                    environment.get("BENCHMARK_PROCESS_SCOPE") == self.token
                    or environment.get("BENCHMARK_WORKER_OWNER") == self.token
                ):
                    self.tracked[process.pid] = process.info["create_time"]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return self.live()

    def live(self):
        result = []
        for pid, started in tuple(self.tracked.items()):
            try:
                process = psutil.Process(pid)
                if (
                    process.create_time() == started
                    and process.status() != psutil.STATUS_ZOMBIE
                ):
                    result.append(process)
            except psutil.NoSuchProcess:
                pass
        return result

    def explorer(self, package, executable):
        for process in self.discover():
            try:
                command = process.cmdline()
                if any(
                    argument.endswith(f"/lib/{package}/{executable}")
                    for argument in command
                ):
                    return process
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return None

    def stop(self):
        self.discover()
        for sig, timeout in (
            (signal.SIGINT, 8),
            (signal.SIGTERM, 3),
            (signal.SIGKILL, 2),
        ):
            # Give ROS launch its own graceful handler first, then signal all
            # remaining identities. Do not signal arbitrary process groups.
            live = self.live()
            owned = {process.pid for process in live}
            for process in live:
                try:
                    if sig == signal.SIGINT and any(
                        parent.pid in owned for parent in process.parents()
                    ):
                        # ROS launch forwards SIGINT to its children. Sending it
                        # twice can interrupt their own cleanup and ROS contexts.
                        continue
                    if process.create_time() == self.tracked[process.pid]:
                        process.send_signal(sig)
                except psutil.NoSuchProcess:
                    pass
            deadline = time.monotonic() + timeout
            while self.live() and time.monotonic() < deadline:
                self.discover()
                time.sleep(0.05)
            if not self.live():
                break
        for root in self.roots:
            try:
                root.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
        for handle in self.logs:
            handle.close()
        self.logs.clear()
        remaining = self.live()
        if remaining:
            raise RuntimeError(
                f"Owned processes survived shutdown: {[p.pid for p in remaining]}"
            )


class DomainLease:
    """Local cross-session leases; keep the flock held until context disposal."""

    def __init__(self, pool, observer, count, directory=None):
        import fcntl

        pool = list(dict.fromkeys(pool))
        if any(
            isinstance(d, bool) or not isinstance(d, int) or not 0 <= d <= 101
            for d in pool
        ):
            raise ValueError("Local domain pool must contain integers from 0 to 101")
        self.handles, self.domains = [], []
        directory = Path(
            directory or f"/tmp/exploration-benchmark-{os.getuid()}/domains"
        )
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            for domain in pool:
                if domain == observer:
                    continue
                handle = open(directory / str(domain), "a+")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                self.handles.append(handle)
                self.domains.append(domain)
                if len(self.domains) == count:
                    return
            raise ValueError(f"Not enough unreserved ROS domains ({count} required)")
        except BaseException:
            self.close()
            raise

    def close(self):
        for handle in self.handles:
            handle.close()
        self.handles.clear()
