"""Offline child transport fixture. All SDK-shaped objects are authored here."""

import asyncio
import importlib
import importlib.util
import os
import socket
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

source = Path(sys.argv[1])
mode = sys.argv[2]
sys.path.insert(0, str(source))
# Deny external connections; register only this interpreter's asyncio socketpair.

listen, connect = socket.socket.listen, socket.socket.connect
allowed = set()


def registered(sock, *args):
    result = listen(sock, *args)
    address = sock.getsockname()
    if address[0] == "127.0.0.1":
        allowed.add(address)
    return result


def guarded(sock, address):
    if address not in allowed:
        raise AssertionError("fixture external socket denied")
    return connect(sock, address)


socket.socket.listen, socket.socket.connect = registered, guarded
socket.socket.connect_ex = lambda *a: (_ for _ in ()).throw(AssertionError("fixture socket denied"))


def fake(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


class SAWarning(Warning):
    pass


class BoundLoggerBase:
    def _process_event(self, *args):
        pass


class DropEvent(Exception):
    pass


fake("sqlalchemy")
fake("sqlalchemy.exc", SAWarning=SAWarning)
fake("structlog", _base=NS(BoundLoggerBase=BoundLoggerBase), DropEvent=DropEvent)


async def add(**kwargs):
    if mode == "preflight":
        raise TimeoutError("PRIVATE_EMBEDDING_ENDPOINT")


async def cognify(**kwargs):
    if mode == "sleep":
        await asyncio.sleep(60)
    else:
        try:
            raise OSError(5, "PRIVATE_SQL_QUERY_BODY")
        except OSError as cause:
            raise RuntimeError("PRIVATE_GRAPH_ERROR") from cause


fake("cognee", add=add, cognify=cognify)
fake("cognee.infrastructure")
fake("cognee.infrastructure.databases")
fake("cognee.infrastructure.databases.vector")
fake(
    "cognee.infrastructure.databases.vector.create_vector_engine",
    _create_vector_engine=NS(cache_clear=lambda: None),
)
fake("cognee.infrastructure.databases.vector.embeddings")
fake(
    "cognee.infrastructure.databases.vector.embeddings.config",
    EmbeddingConfig=lambda **kw: NS(**kw),
)
graph = importlib.import_module("gluomem_bench.bench.cognee_graph")

# No SDK pin claims: this child has only explicit authored modules.
graph._native_source_pins = lambda: {"authored_fixture": True}
graph._base._native_environment = lambda settings: None

os.environ["LLM_ARGS"] = "{}"
if len(sys.argv) > 3 and sys.argv[3] == "legacy":
    import importlib.metadata

    importlib.metadata.version = lambda name: graph._base.COGNEE_VERSION
    asyncio.run(graph._base._cognee_worker())
else:
    asyncio.run(graph._graph_worker())
