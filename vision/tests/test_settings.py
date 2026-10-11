import os
import sys
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "orin"))
import settings

CONF = """# Copyright
< PARAM TYPE=FILE NAME=CPU_ONLINE >
< POWER_MODEL ID=0 NAME=15W >
CPU_ONLINE CORE_0 1
< POWER_MODEL ID=1 NAME=25W >
CPU_ONLINE CORE_0 1
< POWER_MODEL ID=2 NAME=MAXN_SUPER >
CPU_ONLINE CORE_0 1
< POWER_MODEL ID=3 NAME=7W >
< PM_CONFIG DEFAULT=0 >
"""

m = settings.parse_nvpmodel_conf(CONF)
assert m == {"15W": 0, "25W": 1, "MAXN SUPER": 2, "7W": 3}, m
print("OK parse conf", m)
assert settings.parse_nvpmodel_query("NV Power Mode: MAXN_SUPER\n2\n") == ("MAXN_SUPER", 2)
print("OK parse query")

# dry run on non-Jetson
s = settings.Settings(conf_path="/nonexistent/nvpmodel.conf")
ok, msg = s.set("power", "15W")
assert not ok and "dry-run" in msg, msg
g = s.get()
assert g["dry_run"] and g["power_modes"] == [] and g["camera"] == "0.75x" and "auto" in g["options"]["camera"] and g["planner"] is True
print("OK dry-run:", msg)

# validation and on_change
calls = []
s = settings.Settings(on_change=lambda k, v: calls.append((k, v)), conf_path="/nonexistent")
for k, v in [("camera", "2x"), ("orient", "sideways"), ("planner", "maybe"), ("planner", 3), ("bogus", 1), ("camera", None)]:
    ok, msg = s.set(k, v)
    assert not ok, (k, v)
assert calls == []
assert s.set("camera", "0.5x")[0] and s.set("orient", "portrait")[0] and s.set("planner", "off")[0] and s.set("planner", True)[0]
assert calls == [("camera", "0.5x"), ("orient", "portrait"), ("planner", False), ("planner", True)], calls
assert s.get()["camera"] == "0.5x" and s.get()["orient"] == "portrait"
ok, msg = s.set("camera", "0.5x")
assert ok and "already" in msg and len(calls) == 4, (msg, calls)   # same value: no-op, no on_change
assert s.set("planner", True)[0] and len(calls) == 4
assert s.set("camera", "auto")[0] and calls[-1] == ("camera", "auto")
print("OK validation + on_change + same-value no-op")

# jetson path with mocked conf + subprocess
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_nvpmodel.conf")
with open(path, "w") as f:
    f.write(CONF)
try:
    def fake_run(cmd, **kw):
        if cmd[-1] == "-q":
            return mock.Mock(returncode=0, stdout="NV Power Mode: 15W\n0\n", stderr="")
        fake_run.last = (cmd, kw)
        return mock.Mock(returncode=0, stdout="NVPM WARN: reboot required\n", stderr="")
    with mock.patch.object(settings.subprocess, "run", fake_run):
        s = settings.Settings(conf_path=path)
        assert s.get()["power"] == "15W" and s.get()["power_modes"] == ["15W", "25W", "MAXN SUPER", "7W"]
        assert not s.set("power", "; rm -rf /")[0]
        assert not s.set("power", "99W")[0]
        ok, msg = s.set("power", "MAXN SUPER")
        assert ok and "reboot" in msg, msg
        assert fake_run.last[0] == ["sudo", "-n", "/usr/sbin/nvpmodel", "-m", "2"] and fake_run.last[1]["input"] == "NO\n"
        fake_run.last = None
        ok, msg = s.set("power", "MAXN SUPER")
        assert ok and "already" in msg and fake_run.last is None, msg   # same mode: nvpmodel not run again
finally:
    os.remove(path)
print("OK power switch (mocked)")
