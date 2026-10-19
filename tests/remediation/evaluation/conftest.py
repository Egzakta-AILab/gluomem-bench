"""Deny non-fixture sockets and databases in the offline driver tests."""
import socket
import sqlite3

import pytest


@pytest.fixture(autouse=True)
def offline_boundary(monkeypatch):
    allowed = set()
    denied = []
    bind, connect, connect_ex = socket.socket.bind, socket.socket.connect, socket.socket.connect_ex

    def fixture_bind(sock, address):
        result = bind(sock, address)
        if isinstance(address, tuple):
            actual = sock.getsockname()
            if actual[0] in ('127.0.0.1', '::1'):
                allowed.add(actual[:2])
        return result

    def fixture_connect(sock, address, *, operation=connect):
        if not isinstance(address, tuple) or address[:2] not in allowed:
            denied.append('non-fixture-socket')
            raise OSError('offline fixture socket denied')
        return operation(sock, address)

    def no_database(*args, **kwargs):
        denied.append('database')
        raise OSError('offline database connection denied')

    monkeypatch.setattr(socket.socket, 'bind', fixture_bind)
    monkeypatch.setattr(socket.socket, 'connect', fixture_connect)
    monkeypatch.setattr(socket.socket, 'connect_ex',
                        lambda sock, address: fixture_connect(sock, address, operation=connect_ex))
    monkeypatch.setattr(sqlite3, 'connect', no_database)
    yield
    assert denied == []
