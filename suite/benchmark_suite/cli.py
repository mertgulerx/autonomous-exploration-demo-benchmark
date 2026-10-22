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

import argparse
import json
from pathlib import Path
import signal
import sys
import time
from typing import Any

import yaml

from .contracts import ContractError
from .protocol import Client
from .deployment import DEFAULT_WORKSPACE, control_socket


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="benchmark-suite")
    command.add_argument(
        "--workspace", type=Path, default=DEFAULT_WORKSPACE, help=argparse.SUPPRESS
    )
    command.add_argument("--socket", type=Path)
    operations = command.add_subparsers(dest="operation", required=True)
    engine = operations.add_parser("engine", help="Run the persistent engine without a GUI")
    engine.add_argument("--root", type=Path, default=Path.cwd())
    engine.add_argument("--synthetic", action="store_true", help="Use marked synthetic workers")
    engine.add_argument(
        "--without-diagnostics",
        action="store_true",
        help="Disable optional goal/ground-truth instrumentation for overhead measurement",
    )
    operations.add_parser("gui")
    operations.add_parser("status")
    operations.add_parser("catalog")
    for name in ("validate", "submit"):
        operation = operations.add_parser(name)
        operation.add_argument("file", type=Path)
    for name in ("start", "pause", "resume", "end"):
        operations.add_parser(name).add_argument("campaign_id")
    operations.add_parser("stop").add_argument("campaign_id")
    restart = operations.add_parser(
        "restart",
        help="Run a copy of the campaign now or next in the queue; the source is unchanged",
    )
    restart.add_argument("campaign_id")
    restart.add_argument("--request-id")
    query = operations.add_parser("query")
    query.add_argument(
        "table",
        choices=(
            "campaigns",
            "experiments",
            "attempts",
            "library",
            "reports",
            "deletions",
            "runs",
        ),
    )
    query.add_argument("--filters", default="{}")
    query.add_argument("--offset", type=int, default=0)
    query.add_argument("--limit", type=int, default=100)
    export = operations.add_parser("export", help="Export a study (results/studies/<study_id>)")
    export.add_argument("study_id")
    export.add_argument("--wait", action="store_true", help="Wait until the export has finished")
    operations.add_parser(
        "verify-export", help="Check an export against its SHA256SUMS"
    ).add_argument("directory", type=Path)
    call = operations.add_parser("call", help="Access any versioned protocol operation")
    call.add_argument("method")
    call.add_argument("--params", default="{}")
    call.add_argument("--request-id")
    return command


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    workspace = arguments.workspace.expanduser().resolve()
    socket = arguments.socket or control_socket(workspace)
    try:
        if arguments.operation == "engine":
            from .engine import Engine
            from .library import Library

            library = Library(arguments.root, native=not arguments.synthetic)
            if arguments.synthetic:
                from .backends import SyntheticBackend

                backend: Any = SyntheticBackend(steps=100)
            else:
                from .native import NativeBackend

                backend = NativeBackend(
                    arguments.root, diagnostics=not arguments.without_diagnostics
                )
            engine = Engine(workspace, backend, library.resolve_campaign, library)
            for value in (signal.SIGINT, signal.SIGTERM):
                signal.signal(value, lambda signum, frame: setattr(engine, "closed", True))
            engine.run(socket)
            return 0
        if arguments.operation == "gui":
            from .gui import run

            return run(socket)
        if arguments.operation == "verify-export":
            from .storage import verify_sums

            print(json.dumps(verify_sums(arguments.directory), indent=2))
            return 0
        if arguments.operation == "export":
            client = Client(socket)
            try:
                result = client.call("study.export", {"study_id": arguments.study_id})
                while arguments.wait:
                    time.sleep(1.0)
                    result = client.call("study.export_status", {"study_id": arguments.study_id})
                    if result.get("state") != "running":
                        break
            finally:
                client.close()
            print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
            return 0 if result.get("state") != "failed" else 1
        method, params = "engine.status", {}
        if arguments.operation == "catalog":
            method = "catalog.get"
        elif arguments.operation in ("validate", "submit"):
            from .yaml_io import load_mapping

            method = "campaign.preview" if arguments.operation == "validate" else "campaign.submit"
            params = {"specification": load_mapping(arguments.file.read_text())}
        elif arguments.operation in ("start", "pause", "resume", "end"):
            method, params = (
                "campaign." + arguments.operation,
                {"campaign_id": arguments.campaign_id},
            )
        elif arguments.operation in ("stop", "restart"):
            method, params = (
                "campaign.stop" if arguments.operation == "stop" else "campaign.restart_copy",
                {"campaign_id": arguments.campaign_id},
            )
        elif arguments.operation == "query":
            method, params = (
                "results.query",
                {
                    "table": arguments.table,
                    "filters": json.loads(arguments.filters),
                    "offset": arguments.offset,
                    "limit": arguments.limit,
                },
            )
        elif arguments.operation == "call":
            method, params = arguments.method, json.loads(arguments.params)
        client = Client(socket)
        try:
            result = client.call(method, params, getattr(arguments, "request_id", None))
        finally:
            client.close()
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ContractError, OSError, ValueError, yaml.YAMLError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
