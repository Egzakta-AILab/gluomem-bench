"""Authored line-RPC process; no SDK, no network, no database."""

import json
import socket
import sys
import time


def denied(*args, **kwargs):
    raise AssertionError("offline child socket denied")


socket.socket.connect = socket.socket.connect_ex = denied
for line in sys.stdin:
    request = json.loads(line)
    if request["operation"] == "init":
        time.sleep(0.02)
    elif request["operation"] == "add":
        time.sleep(60 if sys.argv[1] == "timeout" else 0.02)
    print(json.dumps({"ok": True, "result": "authored"}), flush=True)
    if request["operation"] == "close":
        break
