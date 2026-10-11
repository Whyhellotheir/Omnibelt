"""Runtime settings for the Omnibelt Orin server (camera lens, power mode, planner, orientation).

Power switching shells out to nvpmodel via sudo. Install this sudoers line (visudo -f /etc/sudoers.d/omnibelt-nvpmodel),
replacing <user> with the account running the server:

    <user> ALL=(root) NOPASSWD: /usr/sbin/nvpmodel

To restrict it to the two forms this module uses (-q and -m <id>) list them explicitly instead:

    <user> ALL=(root) NOPASSWD: /usr/sbin/nvpmodel -q, /usr/sbin/nvpmodel -m 0, /usr/sbin/nvpmodel -m 1, /usr/sbin/nvpmodel -m 2, /usr/sbin/nvpmodel -m 3

Only IDs discovered in /etc/nvpmodel.conf for whitelisted display names are ever passed to nvpmodel; client input never reaches argv.
If /etc/nvpmodel.conf is missing (not a Jetson) the power setting runs in dry-run mode and always returns ok=False.
"""
import os
import re
import subprocess
import threading

NVP_CONF = "/etc/nvpmodel.conf"
NVP_BIN = "/usr/sbin/nvpmodel"
CAMERAS = ("1x", "0.75x", "0.5x")
ORIENTS = ("auto", "portrait", "landscape")
POWER_DISPLAY = ("7W", "15W", "25W", "MAXN SUPER")  # whitelist of display names

_MODEL_RE = re.compile(r"<\s*POWER_MODEL\s+ID=(\d+)\s+NAME=([^\s>]+)\s*>")


def _norm(s):
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def parse_nvpmodel_conf(text):
    """Return {display_name: id} for whitelisted modes found in nvpmodel.conf text (matched by name)."""
    by_norm = {_norm(d): d for d in POWER_DISPLAY}
    out = {}
    for m in _MODEL_RE.finditer(text):
        disp = by_norm.get(_norm(m.group(2)))
        if disp and disp not in out:
            out[disp] = int(m.group(1))
    return out


def parse_nvpmodel_query(text):
    """Parse `nvpmodel -q` output ('NV Power Mode: 15W' then the id on its own line) -> (name, id)."""
    name, mid = None, None
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("nv power mode"):
            name = line.split(":", 1)[1].strip()
        elif line.isdigit():
            mid = int(line)
    return name, mid


class Settings:
    def __init__(self, on_change=None, conf_path=NVP_CONF):
        self.lock = threading.Lock()
        self.on_change = on_change
        self.values = {"camera": "1x", "planner": True, "orient": "auto", "power": None}
        self.modes = {}
        self.dry_run = True
        try:
            with open(conf_path, "r", errors="replace") as f:
                self.modes = parse_nvpmodel_conf(f.read())
            self.dry_run = not self.modes
        except OSError:
            pass
        if not self.dry_run:
            self.values["power"] = self._read_power()

    def _read_power(self):
        try:
            r = subprocess.run(["sudo", "-n", NVP_BIN, "-q"], capture_output=True, text=True, timeout=10)
            name, mid = parse_nvpmodel_query(r.stdout)
        except Exception:
            return None
        for d, i in self.modes.items():
            if mid == i or (name and _norm(name) == _norm(d)):
                return d
        return name

    def get(self):
        with self.lock:
            d = dict(self.values)
            d["power_modes"] = list(self.modes)
            d["dry_run"] = self.dry_run
            d["options"] = {"camera": list(CAMERAS), "orient": list(ORIENTS), "planner": [True, False],
                            "power": list(self.modes)}
            return d

    def set(self, key, value):
        if key == "camera":
            if value not in CAMERAS:
                return False, f"camera must be one of {', '.join(CAMERAS)}"
        elif key == "orient":
            if value not in ORIENTS:
                return False, f"orientation must be one of {', '.join(ORIENTS)}"
        elif key == "planner":
            if isinstance(value, str) and value.lower() in ("on", "off", "true", "false"):
                value = value.lower() in ("on", "true")
            if not isinstance(value, bool):
                return False, "planner must be on or off"
        elif key == "power":
            return self._set_power(value)
        else:
            return False, f"unknown setting {key!r}"
        with self.lock:
            self.values[key] = value
        if self.on_change:
            try:
                self.on_change(key, value)
            except Exception as e:
                return False, f"{key} change failed: {e}"
        return True, f"{key} set to {value}"

    def _set_power(self, value):
        if self.dry_run:
            return False, "power switching unavailable: no /etc/nvpmodel.conf (not a Jetson), dry-run only"
        if not isinstance(value, str) or value not in self.modes:
            return False, f"power must be one of {', '.join(self.modes) or 'none available'}"
        mid = self.modes[value]  # whitelisted id from conf, never client input
        try:
            r = subprocess.run(["sudo", "-n", NVP_BIN, "-m", str(mid)], input="NO\n", timeout=20,
                               capture_output=True, text=True)
        except Exception as e:
            return False, f"nvpmodel failed: {e}"
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0 and "reboot" not in out.lower():
            return False, "nvpmodel failed: " + (out.strip()[-200:] or f"exit {r.returncode}")
        with self.lock:
            self.values["power"] = value
        if "reboot" in out.lower():
            return True, f"power {value} requested; a reboot is required to apply it"
        return True, f"power set to {value}"
