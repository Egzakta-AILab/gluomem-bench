"""Test-only isolated worker bootstrap: exact fixture sockets only."""
import json
import runpy
import socket
import sqlite3
import sys
from pathlib import Path
from urllib.parse import urlsplit

worker = sys.argv[1]
request = Path(sys.argv[sys.argv.index('--request') + 1])
data = json.loads(request.read_bytes())
candidate = json.loads(Path(data['candidate']['path']).read_bytes())
plan = json.loads(Path(candidate['plan']['path']).read_bytes())
endpoint = urlsplit(plan['profile']['base_url'])
allowed = {tuple(data['address']), (endpoint.hostname, endpoint.port)}
counts = {'allowed_fixture_connections': 0, 'denied_connections': 0, 'denied_databases': 0}


def no_database(*args, **kwargs):
    counts['denied_databases'] += 1
    raise OSError('offline database connection denied')


sqlite3.connect = no_database
original_bind, original_connect, original_connect_ex = socket.socket.bind, socket.socket.connect, socket.socket.connect_ex


def bind(sock, address):
    result = original_bind(sock, address)
    if isinstance(address, tuple):
        actual = sock.getsockname()
        if actual[0] in ('127.0.0.1', '::1'):
            allowed.add(actual[:2])
    return result


def connect(sock, address, *, operation=original_connect):
    if not isinstance(address, tuple) or address[:2] not in allowed:
        counts['denied_connections'] += 1
        raise OSError('offline fixture connection denied')
    counts['allowed_fixture_connections'] += 1
    return operation(sock, address)


socket.socket.bind = bind
socket.socket.connect = connect
socket.socket.connect_ex = lambda sock, address: connect(sock, address, operation=original_connect_ex)
sys.argv = sys.argv[1:]
try:
    runpy.run_path(worker, run_name='__main__')
finally:
    (request.parent/'offline-sockets.json').write_bytes(json.dumps(counts).encode())
