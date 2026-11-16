#!/usr/bin/env python3
"""Record the Pixel's power rails for a fixed time under a label and save the totals (stdlib only).

How the rails are read. On the study phone (Pixel 8, user build) `adb shell dumpsys powerstats`
only lists the rails, and the power-stats HAL refuses a dump from the shell. Perfetto's
`android.power` data source can read them without root: it samples the cumulative energy counter
of every rail (microwatt-seconds since boot). This script records one Perfetto trace for the
requested time, pulls it, deletes it on the phone, and reports for every rail
    energy = last counter - first counter,   mean power = energy / (last - first sample time).
Rail energy does not depend on the battery current, so it is valid while charging over USB.

What the numbers are, and are not:
  * Each rail is one supply of the board, not one app: it includes everything using that supply.
  * "all_rails" is the sum of the monitored rails. It is not the phone's total power: supplies
    without a monitor are missing, and a monitored LDO fed from a monitored buck is counted twice.
    Compare modes rail by rail, or group by group, on the same phone.
  * Groups (camera, gpu, cpu, tpu, display, memory) are sums of rails chosen by name, see GROUPS.

Commands sent to the phone (nothing else): `adb shell perfetto` (records only the android.power
data source), `adb pull` and `adb shell rm` of this script's own trace file, and read-only
`adb shell getprop` / `adb shell dumpsys battery|power|activity activities` for the run record.
It never installs, starts, stops or sends input to an app and never touches app data.

Usage:
  python3 analysis/power_measure.py --mode tracking_only   --label run1 --seconds 300
  python3 analysis/power_measure.py --mode tracking_verify --label run1 --seconds 300 --verifications 15
  python3 analysis/power_measure.py --parse analysis/power_runs/x.pftrace     # re-read a saved trace

Start the app and put it in the state to measure first; keep the brightness fixed between runs.
The result is <out-dir>/<mode>__<label>__<time>.json plus the raw trace next to it.
`paper_figures.py --power-dir <out-dir>` turns the runs into table T2.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile

TOOL = "power_measure.py"
FORMAT_VERSION = 1
MODES = ("tracking_only", "tracking_verify", "other")
PHONE_TRACE_DIR = "/data/misc/perfetto-traces"
POLL_MS = 1000
APP_PACKAGE = "edu.cs6301.findback"

# Group -> substrings of rail names (upper case). A rail belongs to the first group that matches.
GROUPS = [
    ("camera", ("VDD_CAM",)),
    ("gpu", ("VDD_G3D",)),
    ("cpu", ("VDD_CPUCL",)),
    ("tpu", ("VDD_TPU",)),
    ("display", ("DISP",)),
    ("memory", ("_MEM",)),
]

PERFETTO_CONFIG = """buffers: {{ size_kb: 8192 fill_policy: RING_BUFFER }}
data_sources: {{
  config {{
    name: "android.power"
    android_power_config {{
      battery_poll_ms: {poll}
      collect_power_rails: true
    }}
  }}
}}
duration_ms: {ms}
"""


# ----------------------------------------------------------------------------- trace parsing

def _varint(buf, i):
    x = shift = 0
    while True:
        c = buf[i]
        i += 1
        x |= (c & 0x7F) << shift
        if not c & 0x80:
            return x, i
        shift += 7


def _fields(buf):
    """Yield (field number, wire type, value) of one protobuf message. Length-delimited values
    are bytes; varints are ints; fixed32/64 values are raw bytes."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 1:
            value, i = buf[i:i + 8], i + 8
        elif wire == 2:
            length, i = _varint(buf, i)
            value, i = buf[i:i + length], i + length
        elif wire == 5:
            value, i = buf[i:i + 4], i + 4
        else:
            raise ValueError("unsupported protobuf wire type %d" % wire)
        if i > n:
            raise ValueError("truncated protobuf message")
        yield field, wire, value


# Field numbers from the Perfetto trace protos (protos/perfetto/trace/trace_packet.proto,
# power/power_rails.proto, clock_snapshot.proto).
F_TRACE_PACKET = 1
F_PACKET_CLOCK_SNAPSHOT = 6
F_PACKET_POWER_RAILS = 40
F_RAILS_DESCRIPTOR, F_RAILS_ENERGY = 1, 2
F_DESC_INDEX, F_DESC_NAME, F_DESC_SUBSYS = 1, 2, 3
F_ENERGY_INDEX, F_ENERGY_TIMESTAMP_MS, F_ENERGY_UWS = 1, 2, 3
F_SNAPSHOT_CLOCKS = 1
F_CLOCK_ID, F_CLOCK_TIMESTAMP = 1, 2
CLOCK_REALTIME, CLOCK_BOOTTIME = 1, 6


def parse_trace(data):
    """Rails and their samples from a Perfetto trace.

    Returns {'rails': {index: (name, subsystem)}, 'samples': {index: [(timestamp_ms, energy_uws)]},
             'boot_to_epoch_ms': float or None}.
    timestamp_ms is the rail's own sample time in milliseconds since boot. boot_to_epoch_ms, from
    the trace's first clock snapshot, converts it to the phone's wall clock (the clock that stamps
    events.jsonl).
    """
    rails, samples, boot_to_epoch = {}, {}, None
    for field, wire, packet in _fields(data):
        if field != F_TRACE_PACKET or wire != 2:
            continue
        for pf, pw, pv in _fields(packet):
            if pw != 2:
                continue
            if pf == F_PACKET_POWER_RAILS:
                for rf, rw, rv in _fields(pv):
                    if rw != 2:
                        continue
                    msg = {f: v for f, _, v in _fields(rv)}
                    if rf == F_RAILS_DESCRIPTOR:
                        rails[msg.get(F_DESC_INDEX, 0)] = (
                            msg.get(F_DESC_NAME, b"").decode("utf-8", "replace"),
                            msg.get(F_DESC_SUBSYS, b"").decode("utf-8", "replace"),
                        )
                    elif rf == F_RAILS_ENERGY:
                        if F_ENERGY_TIMESTAMP_MS in msg and F_ENERGY_UWS in msg:
                            samples.setdefault(msg.get(F_ENERGY_INDEX, 0), []).append(
                                (msg[F_ENERGY_TIMESTAMP_MS], msg[F_ENERGY_UWS]))
            elif pf == F_PACKET_CLOCK_SNAPSHOT and boot_to_epoch is None:
                clocks = {}
                for sf, sw, sv in _fields(pv):
                    if sf == F_SNAPSHOT_CLOCKS and sw == 2:
                        c = {f: v for f, w, v in _fields(sv) if w == 0}
                        if F_CLOCK_ID in c and F_CLOCK_TIMESTAMP in c:
                            clocks[c[F_CLOCK_ID]] = c[F_CLOCK_TIMESTAMP]
                if CLOCK_REALTIME in clocks and CLOCK_BOOTTIME in clocks:
                    boot_to_epoch = (clocks[CLOCK_REALTIME] - clocks[CLOCK_BOOTTIME]) / 1e6
    for s in samples.values():
        s.sort()
    return {"rails": rails, "samples": samples, "boot_to_epoch_ms": boot_to_epoch}


def group_of(rail_name):
    upper = rail_name.upper()
    for group, needles in GROUPS:
        if any(n in upper for n in needles):
            return group
    return "other"


def summarise(parsed):
    """Per-rail, per-group and all-rail energy over the common window of the trace.

    Raises ValueError when the trace has no usable rail data or a counter runs backwards.
    """
    rails_out, problems = [], []
    for index in sorted(parsed["samples"]):
        s = parsed["samples"][index]
        name, subsystem = parsed["rails"].get(index, ("rail_%d" % index, ""))
        if len(s) < 2:
            problems.append("%s: only %d sample(s)" % (name, len(s)))
            continue
        if any(b[1] < a[1] for a, b in zip(s, s[1:])):
            raise ValueError("energy counter of %s decreases; the trace is not usable" % name)
        t0, e0 = s[0]
        t1, e1 = s[-1]
        dt = (t1 - t0) / 1000.0
        if dt <= 0:
            problems.append("%s: no time between samples" % name)
            continue
        energy_j = (e1 - e0) / 1e6
        rails_out.append({
            "index": index, "name": name, "subsystem": subsystem, "group": group_of(name),
            "samples": len(s), "t_first_ms": t0, "t_last_ms": t1, "duration_s": dt,
            "energy_j": energy_j, "mean_mw": energy_j / dt * 1000.0,
        })
    if not rails_out:
        raise ValueError("no power-rail samples in the trace (is collect_power_rails supported?)")
    duration = max(r["duration_s"] for r in rails_out)
    groups = {}
    for r in rails_out:
        g = groups.setdefault(r["group"], {"energy_j": 0.0, "mean_mw": 0.0, "rails": []})
        g["energy_j"] += r["energy_j"]
        g["mean_mw"] += r["mean_mw"]
        g["rails"].append(r["name"])
    first = min(r["t_first_ms"] for r in rails_out)
    last = max(r["t_last_ms"] for r in rails_out)
    b2e = parsed.get("boot_to_epoch_ms")
    return {
        "duration_s": duration,
        "rails": rails_out,
        "groups": groups,
        "all_rails": {
            "energy_j": sum(r["energy_j"] for r in rails_out),
            "mean_mw": sum(r["mean_mw"] for r in rails_out),
            "n_rails": len(rails_out),
            "note": "sum of the monitored rails; not the phone's total power (see the script's docstring)",
        },
        "phone_start_ms": (b2e + first) if b2e is not None else None,
        "phone_end_ms": (b2e + last) if b2e is not None else None,
        "problems": problems,
    }


# ----------------------------------------------------------------------------- phone

class AdbError(RuntimeError):
    pass


def adb(args, serial=None, timeout=60, stdin_text=None):
    cmd = ["adb"] + (["-s", serial] if serial else []) + list(args)
    try:
        p = subprocess.run(cmd, input=stdin_text, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise AdbError("adb not found on PATH")
    except subprocess.TimeoutExpired:
        raise AdbError("timed out: " + " ".join(cmd))
    if p.returncode != 0:
        raise AdbError("%s failed (%d): %s" % (" ".join(cmd), p.returncode, (p.stderr or p.stdout).strip()))
    return p.stdout


def phone_state(serial):
    """Read-only context for the run record. Every value is None when it cannot be read."""
    def grab(args, pattern):
        try:
            m = re.search(pattern, adb(["shell"] + args, serial), re.M)
            return m.group(1).strip() if m else None
        except AdbError:
            return None
    battery = {}
    try:
        text = adb(["shell", "dumpsys", "battery"], serial)
        for key, pat in (("ac_powered", r"AC powered: (\w+)"), ("usb_powered", r"USB powered: (\w+)"),
                         ("level", r"^\s*level: (\d+)"), ("temperature_tenth_c", r"^\s*temperature: (\d+)"),
                         ("status", r"^\s*status: (\d+)")):
            m = re.search(pat, text, re.M)
            battery[key] = m.group(1) if m else None
    except AdbError:
        pass
    return {
        "foreground": grab(["dumpsys", "activity", "activities"], r"topResumedActivity=ActivityRecord\{\S+ \S+ (\S+)"),
        "wakefulness": grab(["dumpsys", "power"], r"mWakefulness=(\w+)"),
        "screen_brightness": grab(["dumpsys", "display"], r"^\s*mScreenBrightness=([0-9.]+)"),
        "battery": battery,
    }


def warn_if_not_app(state, mode, when):
    """paper_figures.py uses a tracking run for T2 only if the app was in the foreground at both ends."""
    fg = state.get("foreground")
    if mode != "other" and not (fg and APP_PACKAGE in fg):
        print("WARNING: %s the recording the foreground app is %s (screen: %s), not %s.\n"
              "         paper_figures.py will leave this run out of T2." % (when, fg or "unknown", state.get("wakefulness"), APP_PACKAGE),
              file=sys.stderr)


def record_trace(seconds, serial, local_path):
    """Record `seconds` of the android.power data source and pull the trace to local_path."""
    stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    remote = "%s/findback_power_%s_%d.pftrace" % (PHONE_TRACE_DIR, stamp, os.getpid())
    config = PERFETTO_CONFIG.format(poll=POLL_MS, ms=int(seconds * 1000))
    try:
        adb(["shell", "perfetto", "--txt", "-c", "-", "-o", remote], serial, timeout=seconds + 90, stdin_text=config)
        adb(["pull", remote, local_path], serial)
    finally:
        try:
            adb(["shell", "rm", "-f", remote], serial)
        except AdbError as e:
            print("WARNING: could not delete %s on the phone: %s" % (remote, e), file=sys.stderr)


def print_summary(run):
    print("window %.1f s, %d rails" % (run["duration_s"], run["all_rails"]["n_rails"]))
    print("%-22s %-12s %-8s %10s %10s" % ("rail", "subsystem", "group", "energy J", "mean mW"))
    for r in run["rails"]:
        print("%-22s %-12s %-8s %10.3f %10.1f" % (r["name"], r["subsystem"], r["group"], r["energy_j"], r["mean_mw"]))
    print("groups:")
    for g in sorted(run["groups"]):
        print("  %-8s %10.3f J %10.1f mW   (%s)" % (g, run["groups"][g]["energy_j"], run["groups"][g]["mean_mw"],
                                                    ", ".join(run["groups"][g]["rails"])))
    print("all monitored rails: %.3f J, %.1f mW (not the phone's total power)" % (
        run["all_rails"]["energy_j"], run["all_rails"]["mean_mw"]))
    for p in run["problems"]:
        print("WARNING:", p)


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=MODES, help="what the phone is doing; table T2 uses tracking_only and tracking_verify")
    ap.add_argument("--label", default="run", help="free text to tell runs apart (letters, digits, - and _)")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--verifications", type=int, help="verifications done during the run, if you counted them")
    ap.add_argument("--note", default="")
    ap.add_argument("--out-dir", default=os.path.join(here, "power_runs"))
    ap.add_argument("--serial", help="adb serial when more than one device is connected")
    ap.add_argument("--parse", metavar="TRACE", help="summarise a saved .pftrace and exit (no phone needed)")
    a = ap.parse_args(argv)

    if a.parse:
        with open(a.parse, "rb") as f:
            print_summary(summarise(parse_trace(f.read())))
        return 0
    if not a.mode:
        ap.error("--mode is required")
    if a.seconds <= 2 * POLL_MS / 1000.0:
        ap.error("--seconds must be more than %g" % (2 * POLL_MS / 1000.0))
    label = re.sub(r"[^A-Za-z0-9_-]+", "-", a.label).strip("-") or "run"

    try:
        device = {
            "model": adb(["shell", "getprop", "ro.product.model"], a.serial).strip(),
            "build": adb(["shell", "getprop", "ro.build.fingerprint"], a.serial).strip(),
            "build_type": adb(["shell", "getprop", "ro.build.type"], a.serial).strip(),
        }
        before = phone_state(a.serial)
        warn_if_not_app(before, a.mode, "before")
        os.makedirs(a.out_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        base = os.path.join(a.out_dir, "%s__%s__%s" % (a.mode, label, stamp))
        print("Recording %.0f s of power rails (mode %s, label %s)..." % (a.seconds, a.mode, label))
        with tempfile.TemporaryDirectory() as tmp:
            tmp_trace = os.path.join(tmp, "trace.pftrace")
            record_trace(a.seconds, a.serial, tmp_trace)
            with open(tmp_trace, "rb") as f:
                data = f.read()
        after = phone_state(a.serial)
        warn_if_not_app(after, a.mode, "after")
        run = summarise(parse_trace(data))
    except (AdbError, ValueError) as e:
        print("ERROR:", e, file=sys.stderr)
        return 1

    with open(base + ".pftrace", "wb") as f:
        f.write(data)
    run.update({
        "tool": TOOL, "format_version": FORMAT_VERSION, "synthetic": False,
        "mode": a.mode, "label": label, "note": a.note,
        "requested_seconds": a.seconds, "poll_ms": POLL_MS,
        "method": "Perfetto android.power data source, collect_power_rails, cumulative energy counters",
        "verifications": a.verifications,
        "device": device, "state_before": before, "state_after": after,
        "host_time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "trace_file": os.path.basename(base + ".pftrace"),
    })
    with open(base + ".json", "w") as f:
        json.dump(run, f, indent=1)
    print_summary(run)
    print("Saved", base + ".json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
