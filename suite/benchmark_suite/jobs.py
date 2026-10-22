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

import multiprocessing as mp
from multiprocessing.connection import Connection
import os
from pathlib import Path
import sqlite3
from typing import Any

import psutil


def execute(connection: Connection, operation: str, payload: dict[str, Any], token: str) -> None:
    os.environ["BENCHMARK_SUITE_JOB_OWNER"] = token
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    try:
        if operation == "compact":
            from .datasets import compact

            workspace = Path(payload["workspace"])
            with sqlite3.connect(
                f"file:{workspace / 'suite.sqlite3'}?mode=ro", uri=True
            ) as database:
                directories = [
                    row[0]
                    for row in database.execute(
                        "SELECT directory FROM attempts WHERE campaign_id=? AND retained=1 AND discarded=0 ORDER BY id",
                        (payload["campaign_id"],),
                    )
                ]
            result = compact(workspace, payload["campaign_id"], directories)
        elif operation == "commit":
            from .datasets import commit_compaction

            commit_compaction(payload)
            result = {"committed": True}
        elif operation == "report":
            from .reports import generate

            result = generate(payload)
        elif operation == "run_map":
            from .run_map import build

            result = build(payload)
        elif operation == "study_export":
            from .exports import export_study

            os.nice(10)
            result = export_study(payload)
        elif operation == "run_record":
            from .run_record import measure

            os.nice(10)  # live runs keep priority (RESULTS_OVERHAUL A2)
            result = measure(payload)
        else:
            raise ValueError("Unknown analysis operation: " + operation)
        connection.send({"ok": True, "result": result})
    except Exception as error:
        connection.send(
            {
                "ok": False,
                "detail": f"{type(error).__name__}: {error}",
                "errno": getattr(error, "errno", None),
            }
        )
    finally:
        connection.close()


class Job:
    def __init__(self, operation: str, payload: dict[str, Any], token: str):
        context = mp.get_context("spawn")
        self.connection, sender = context.Pipe(duplex=False)
        self.process = context.Process(
            target=execute, args=(sender, operation, payload, token), name="suite-analysis"
        )
        previous = os.environ.get("BENCHMARK_SUITE_JOB_OWNER")
        try:
            os.environ["BENCHMARK_SUITE_JOB_OWNER"] = token
            self.process.start()
            self.pid = self.process.pid
            self.created = psutil.Process(self.pid).create_time()
        except BaseException:
            if self.process.pid is not None:
                self.process.terminate()
                self.process.join(5)
            self.connection.close()
            raise
        finally:
            if previous is None:
                os.environ.pop("BENCHMARK_SUITE_JOB_OWNER", None)
            else:
                os.environ["BENCHMARK_SUITE_JOB_OWNER"] = previous
            sender.close()
        self.result: dict[str, Any] | None = None

    def poll(self) -> dict[str, Any] | None:
        if self.result is None and self.connection.poll():
            try:
                self.result = self.connection.recv()
            except EOFError:
                self.result = {
                    "ok": False,
                    "detail": "Analysis process exited without a commit receipt",
                }
        if self.process.is_alive():
            return None
        self.process.join()
        return self.result or {
            "ok": False,
            "detail": "Analysis process exited without a commit receipt",
        }

    def close(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(2)
        self.connection.close()
        self.process.close()


def recover(token: str, pid: int | None, created: float | None) -> None:
    for process in psutil.process_iter(["pid", "uids", "create_time"]):
        try:
            if process.info["uids"].real != os.getuid() or process.pid == os.getpid():
                continue
            owned = process.pid == pid and process.info["create_time"] == created
            if owned or (token and process.environ().get("BENCHMARK_SUITE_JOB_OWNER") == token):
                process.terminate()
                try:
                    process.wait(5)
                except psutil.TimeoutExpired:
                    process.kill()
                    process.wait(2)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
