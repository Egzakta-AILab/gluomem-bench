"""Process-local pre-format diagnostics and default-deny native egress.

Install before importing native SDKs, never in a shared application process.
Original exceptions/records stay in memory; durable output is categorical only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import socket
import threading
import warnings
from pathlib import Path
from urllib.parse import urlsplit

_ERROR_TYPES = {
    cls: cls.__name__
    for cls in (
        Exception,
        RuntimeError,
        ValueError,
        TypeError,
        KeyError,
        AssertionError,
        OSError,
        TimeoutError,
        ConnectionError,
        ConnectionRefusedError,
        ConnectionResetError,
        BrokenPipeError,
        FileNotFoundError,
        PermissionError,
    )
}
_STAGES = frozenset(
    (
        "unknown",
        "rpc",
        "init",
        "add",
        "cognify",
        "search",
        "source_pins",
        "close",
        "diagnostic_admission",
        "embedding_connection_preflight",
        "graph_query",
    )
)
_CLOCKS = frozenset(
    ("unknown", "operation_deadline", "legacy_rpc_timeout", "native_embedding_preflight")
)


def clean_diagnostic(value):
    """Project untrusted RPC fields to static labels, source hashes and integers."""
    value = value if type(value) is dict else {}
    result = {
        "protocol": "cognee-safe-failure@v2",
        "stage": value.get("stage")
        if type(value.get("stage")) is str and value.get("stage") in _STAGES
        else "unknown",
        "clock": value.get("clock")
        if type(value.get("clock")) is str and value.get("clock") in _CLOCKS
        else "unknown",
        "causes": [],
    }
    stage_origin = value.get("stage_origin")
    if type(stage_origin) is dict:
        cleaned = clean_diagnostic({"causes": [{"origin": stage_origin}]})["causes"][0]
        if "origin" in cleaned:
            result["stage_origin"] = cleaned["origin"]
    causes = value.get("causes") if type(value.get("causes")) is list else []
    for cause in causes[:8]:
        if type(cause) is not dict:
            continue
        row = {
            "error_class": cause.get("error_class")
            if cause.get("error_class") in _ERROR_TYPES.values()
            else "unknown"
        }
        for key in ("errno", "winerror"):
            if type(cause.get(key)) is int:
                row[key] = cause[key]
        origin = cause.get("origin") if type(cause.get("origin")) is dict else {}
        digest, line = origin.get("source_sha256"), origin.get("line")
        if (
            type(digest) is str
            and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest)
        ):
            row["origin"] = {
                "source_sha256": digest,
                "line": line if type(line) is int and line > 0 else None,
            }
        result["causes"].append(row)
    return result


def native_diagnostic(exc, stage="unknown", clock="unknown"):
    """Never format an exception or serialize its class name, frames or arguments."""
    value = {"stage": stage, "clock": clock, "causes": []}
    seen = set()
    while isinstance(exc, BaseException) and id(exc) not in seen and len(seen) < 8:
        seen.add(id(exc))
        row = {"error_class": _ERROR_TYPES.get(type(exc), "unknown")}
        if type(exc) in _ERROR_TYPES and isinstance(exc, OSError):
            for key in ("errno", "winerror"):
                number = getattr(exc, key, None)
                if type(number) is int:
                    row[key] = number
        tb = BaseException.__traceback__.__get__(exc)
        while tb:
            origin = NativeBoundary.origin(tb.tb_frame.f_code.co_filename, tb.tb_lineno)
            row["origin"] = origin
            # A text/name match alone cannot establish this pinned native prerequisite.
            if (
                origin["source_sha256"]
                == "4804470f76196270601602a60e0a9550f89f96bd8b8ce67b2479887cbd461cdc"
                and tb.tb_frame.f_code.co_name == "test_embedding_connection"
            ):
                value["stage"] = "embedding_connection_preflight"
                value["stage_origin"] = origin
                if type(exc) is TimeoutError:
                    value["clock"] = "native_embedding_preflight"
            tb = tb.tb_next
        value["causes"].append(row)
        exc = BaseException.__cause__.__get__(exc) or BaseException.__context__.__get__(exc)
    return clean_diagnostic(value)


class NativeBoundary:
    def __init__(self, receipt):
        self.receipt = Path(receipt)
        self.lock = threading.RLock()
        self.events = []
        self.benign = []
        self.startup = True
        self.causes = []
        self.allowed = set()
        self.connections = {"approved_proxy": 0, "denied": 0}

    def capture(self, category, cause=None, origin=None, *, diagnostic=None):
        with self.lock:
            row = {"category": category}
            error = cause
            if isinstance(cause, logging.LogRecord):
                error = cause.exc_info
            elif type(cause) is tuple and len(cause) == 2:
                error = (
                    cause[0]
                    if isinstance(cause[0], BaseException)
                    else (cause[1].get("exc_info") if type(cause[1]) is dict else None)
                )
            if type(error) is tuple and len(error) == 3:
                error = error[1]
            if diagnostic is not None:
                row["diagnostic"] = clean_diagnostic(diagnostic)
            elif isinstance(error, BaseException):
                row["diagnostic"] = native_diagnostic(error)
            if origin is not None:
                row["origin"] = origin
            digest = (origin or {}).get("source_sha256")
            line = (origin or {}).get("line")
            # Explicit review: API migration notice, deprecated Field env kwargs,
            # deprecated json_encoders. No SQL or failure warning is allowlisted.
            reviewed = self.startup and (
                (
                    category == "structured_warning"
                    and digest == "a47784a889e2fc8a041ba02523ef29c65f20aeaa726dee70f079d99adb5c3c99"
                    and line == 637
                    and origin.get("message_sha256")
                    == "2392b847aa39876b322358272871f3b81d32f7d83f19204d729e0852e0081a1b"
                )
                or (
                    category == "python_warning"
                    and digest == "44a0e2f85105ea874a271ec16a49fb5815447843d93e1927a241fda2c3a2f75d"
                    and line in (47, 65, 66, 67)
                )
                or (
                    category == "python_warning"
                    and digest == "759371011c1c94390c786da3fd2d7a312f05bc303d19d9cd87a5426a212216b0"
                    and line == 300
                )
            )
            (self.benign if reviewed else self.events).append(row)
            if cause is not None:
                self.causes.append(cause)

    @staticmethod
    def origin(filename, lineno, message=None):
        try:
            digest = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        except OSError:
            digest = None
        return {
            "source_sha256": digest,
            "line": lineno,
            "message_sha256": hashlib.sha256(message.encode()).hexdigest()
            if type(message) is str
            else None,
        }

    def snapshot(self):
        with self.lock:
            return {
                "protocol": "native-preformat-boundary@v1",
                "blocking_diagnostics": len(self.events),
                "events": list(self.events),
                "connections": dict(self.connections),
                "benign_startup_diagnostics": len(self.benign),
                "benign_startup_receipts": list(self.benign),
            }

    def save(self):
        self.receipt.write_bytes((json.dumps(self.snapshot(), sort_keys=True) + "\n").encode())

    def check(self):
        self.save()
        if self.events:
            error = RuntimeError("native diagnostic admission rejected")
            error.native_diagnostic = next(
                (row["diagnostic"] for row in self.events if "diagnostic" in row),
                clean_diagnostic({"stage": "diagnostic_admission"}),
            )
            raise error

    def approve_proxies(self, settings):
        for key in ("llm_url", "embedding_url"):
            value = urlsplit(settings[key])
            if (
                value.scheme != "http"
                or value.hostname not in ("127.0.0.1", "::1")
                or not value.port
            ):
                raise ValueError("native transport requires exact scoped loopback proxy")
            self.allowed.add((value.hostname, value.port))

    def install(self, *, network=False):
        boundary = self

        # Logger.handle is before handlers, including nonpropagating children.
        def handle(logger, record):
            if record.levelno >= logging.WARNING:
                boundary.capture(
                    "logging_error" if record.levelno >= logging.ERROR else "logging_warning",
                    record,
                    boundary.origin(record.pathname, record.lineno, record.msg),
                )
            # Native logging is diagnostic-only: no record ever reaches formatters.

        logging.Logger.handle = handle

        def showwarning(message, category, filename, lineno, file=None, line=None):
            boundary.capture(
                "python_warning",
                (message, category, filename, lineno),
                boundary.origin(filename, lineno),
            )

        warnings.showwarning = showwarning
        # Native logging config ignores several SAWarning producers. Capture
        # those before Python warning filters, without enabling all ignored
        # library deprecation warnings or touching native database algorithms.
        from sqlalchemy.exc import SAWarning

        original_warn = warnings.warn

        def warn(message, category=None, stacklevel=1, source=None, **kwargs):
            actual_category = type(message) if isinstance(message, Warning) else category
            if isinstance(actual_category, type) and issubclass(actual_category, SAWarning):
                import sys

                frame = sys._getframe(1)
                boundary.capture(
                    "sql_warning",
                    message,
                    boundary.origin(frame.f_code.co_filename, frame.f_lineno),
                )
                return
            return original_warn(
                message, category, stacklevel=stacklevel + 1, source=source, **kwargs
            )

        warnings.warn = warn
        import structlog

        def process_event(logger, method_name, event, event_kw):
            if method_name in ("warning", "warn", "error", "exception", "critical", "fatal"):
                import sys

                frame = sys._getframe(1)
                while frame and ("structlog" in frame.f_code.co_filename):
                    frame = frame.f_back
                origin = (
                    boundary.origin(frame.f_code.co_filename, frame.f_lineno, event)
                    if frame
                    else None
                )
                boundary.capture(
                    "structured_warning"
                    if method_name in ("warning", "warn")
                    else "structured_error",
                    (event, event_kw),
                    origin,
                )
            # Raising DropEvent before processors prevents repr/traceback/SQL formatting.
            raise structlog.DropEvent

        structlog._base.BoundLoggerBase._process_event = process_event
        if not network:
            return
        connect = socket.socket.connect
        listen = socket.socket.listen

        def guarded_listen(sock, *args):
            result = listen(sock, *args)
            address = sock.getsockname()
            if isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1"):
                self.allowed.add((address[0], address[1]))
            return result

        def guarded_connect(sock, address):
            permitted = isinstance(address, tuple) and (address[0], address[1]) in self.allowed
            with self.lock:
                self.connections["approved_proxy" if permitted else "denied"] += 1
            if not permitted:
                import sys

                frame = sys._getframe(1)
                origins = []
                while frame and len(origins) < 20:
                    origins.append(boundary.origin(frame.f_code.co_filename, frame.f_lineno))
                    frame = frame.f_back
                self.capture("denied_connection", address, {"stack": origins})
                raise OSError("native outbound denied")
            return connect(sock, address)

        def guarded_connect_ex(sock, address):
            try:
                guarded_connect(sock, address)
                return 0
            except OSError:
                return 10013

        socket.socket.listen = guarded_listen
        socket.socket.connect = guarded_connect
        socket.socket.connect_ex = guarded_connect_ex
