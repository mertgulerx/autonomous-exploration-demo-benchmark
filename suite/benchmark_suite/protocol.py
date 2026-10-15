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

from concurrent.futures import Future
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import socket
import struct
import threading
from typing import Any
import uuid

from .contracts import PROTOCOL_VERSION, ContractError, canonical, mapping

MAX_MESSAGE = 4 * 1024 * 1024


def encode_response(result: dict[str, Any]) -> bytes:
    encoded = (canonical(result) + "\n").encode()
    if len(encoded) > MAX_MESSAGE:
        identity = result.get("request_id")
        identity = identity if isinstance(identity, str) and len(identity) <= 160 else None
        encoded = (
            canonical(
                {
                    "ok": False,
                    "request_id": identity,
                    "error": {
                        "code": "response_too_large",
                        "field": "page",
                        "message": "Reduce page size or use dataset export; response exceeds 4 MiB",
                    },
                }
            )
            + "\n"
        ).encode()
    return encoded


def receive(stream: Any) -> dict[str, Any]:
    line = stream.readline(MAX_MESSAGE + 1)
    if not line:
        raise EOFError("Control connection closed")
    if len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
        raise ContractError("request", "Control message exceeds 4 MiB")
    try:
        return mapping(json.loads(line), "request")
    except (ValueError, UnicodeError) as exc:
        raise ContractError("request", "Invalid JSON control message") from exc


@dataclass(slots=True)
class PendingRequest:
    request: dict[str, Any]
    future: Future[dict[str, Any]]


class Server:
    def __init__(self, path: Path, requests: queue.Queue[PendingRequest], max_clients: int = 8):
        self.path, self.requests = path, requests
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.exists():
            path.unlink()
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.bind(str(path))
        os.chmod(path, 0o600)
        self.socket.listen(max_clients)
        self.socket.settimeout(0.2)
        self.closed = threading.Event()
        self.capacity = threading.BoundedSemaphore(max_clients)
        self.clients: set[socket.socket] = set()
        self.threads: set[threading.Thread] = set()
        self.pending: set[Future[dict[str, Any]]] = set()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._accept, name="suite-socket", daemon=True)
        self.thread.start()

    def _accept(self) -> None:
        while not self.closed.is_set():
            try:
                client, _ = self.socket.accept()
            except socket.timeout:
                continue
            except OSError:
                if self.closed.is_set():
                    break
                raise
            credentials = client.getsockopt(
                socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
            )
            _, uid, _ = struct.unpack("3i", credentials)
            if uid != os.getuid() or not self.capacity.acquire(blocking=False):
                client.close()
                continue
            client.settimeout(30)
            thread = threading.Thread(
                target=self._client, args=(client,), daemon=True, name="suite-client"
            )
            with self.lock:
                self.clients.add(client)
                self.threads.add(thread)
            thread.start()

    def _client(self, client: socket.socket) -> None:
        try:
            with client.makefile("rb") as stream:
                while not self.closed.is_set():
                    try:
                        request = receive(stream)
                        future: Future[dict[str, Any]] = Future()
                        with self.lock:
                            self.pending.add(future)
                        try:
                            self.requests.put(PendingRequest(request, future), timeout=1)
                            result = future.result(timeout=120)
                        finally:
                            with self.lock:
                                self.pending.discard(future)
                    except EOFError:
                        break
                    except ContractError as exc:
                        result = {"ok": False, "error": exc.dictionary()}
                    except queue.Full:
                        result = {
                            "ok": False,
                            "error": {"code": "busy", "message": "Engine queue is full"},
                        }
                    client.sendall(encode_response(result))
        except (OSError, TimeoutError):
            pass
        finally:
            client.close()
            self.capacity.release()
            with self.lock:
                self.clients.discard(client)
                self.threads.discard(threading.current_thread())

    def close(self) -> None:
        self.closed.set()
        self.socket.close()
        with self.lock:
            clients, threads, pending = (
                tuple(self.clients),
                tuple(self.threads),
                tuple(self.pending),
            )
        for future in pending:
            if not future.done():
                future.set_result(
                    {
                        "ok": False,
                        "error": {"code": "engine_shutdown", "message": "Engine is shutting down"},
                    }
                )
        for client in clients:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            client.close()
        self.thread.join(timeout=1)
        for thread in threads:
            thread.join(timeout=1)
        self.path.unlink(missing_ok=True)


class Client:
    def __init__(self, path: Path, timeout: float = 120):
        self.path, self.timeout = path, timeout
        self.lock = threading.Lock()
        self.connections: set[socket.socket] = set()
        self.closed = False

    def call(
        self, method: str, params: dict[str, Any] | None = None, request_id: str | None = None
    ) -> Any:
        request = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id or str(uuid.uuid4()),
            "method": method,
            "params": params or {},
        }
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            with self.lock:
                if self.closed:
                    raise OSError("Client closed")
                self.connections.add(connection)
            try:
                connection.settimeout(self.timeout)
                connection.connect(str(self.path))
                with self.lock:
                    if self.closed:
                        raise OSError("Client closed")
                connection.sendall((canonical(request) + "\n").encode())
                with connection.makefile("rb") as stream:
                    response = receive(stream)
            finally:
                with self.lock:
                    self.connections.discard(connection)
        if not response.get("ok"):
            error = response.get("error", {})
            raise ContractError(
                error.get("field", method),
                error.get("message", "Request failed"),
                error.get("code", "runtime_error"),
            )
        return response["result"]

    def close(self) -> None:
        with self.lock:
            self.closed = True
            connections = tuple(self.connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
