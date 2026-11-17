#!/usr/bin/env python3
"""Summary of the FindBack v0.7 detailed trace, one report per session.

Reads a data folder pulled from the phone (FindBack/docs/PLAN_v0.7_trace.md, section 5):

  events.jsonl                 the event log (the v0.7 events: ui_screen, ui_action, ui_search,
                               ui_message, capture_rejected, lock_attempt, device_state, perf,
                               app_pause, app_resume, trace_error, ar_config)
  trace/<session>/pose.csv     10 Hz camera pose, tracking, map pose, speed, screen
  trace/<session>/sensors.csv  accelerometer, gyroscope, magnetometer, barometer, light
  trace/<session>/depth.csv    2 Hz depth image statistics, feature points, planes, light estimate
  trace/<session>/guidance.csv 5 Hz during a search: distance and direction to the item

For each session with a trace it prints a Markdown report and writes it, with two plots, to the
output folder (default analysis/out/, git-ignored: a trajectory is a floor plan of the home).
A session without a trace gets the event sections only when asked for with --session.

Usage:
  python3 analysis/trace_summary.py findback_data
  python3 analysis/trace_summary.py findback_data --session ab12cd34 --out /tmp/fb_out
  python3 analysis/trace_summary.py findback_data --no-plots

Thresholds for the Findings section are the named constants below. Nothing here changes a file
in the data folder.
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "out")

# ----------------------------------------------------------------------------- Findings thresholds
TRACKING_LOST_SHARE_MAX = 0.05     # tracking lost more than 5% of the traced time
DEPTH_USABLE_WALKING_MIN = 0.50    # mean usable depth share below 50% while walking
NO_DEPTH_SHARE_MAX = 0.20          # more than 20% of depth samples without a depth image
FPS_P5_MIN = 20.0                  # 5th percentile of the 10 s fps windows below 20
THERMAL_STATUS_MODERATE = 2        # PowerManager.THERMAL_STATUS_MODERATE
REFUSED_PER_SAVED_MAX = 3.0        # more than 3 refused registration taps per saved item
LOCK_SUCCESS_MIN = 0.50            # fewer than half of the Lock presses succeed
TRACE_DROPPED_MAX = 0              # any trace line dropped
FRAME_MS_P95_MAX = 50.0            # median 95th-percentile GL frame time above 50 ms
DEPTH_CHECK_MAX_M = 0.10           # scan depth self-check: median error above 10 cm

# ----------------------------------------------------------------------------- definitions
STILL_MPS = 0.05        # below: still
WALK_MPS = 0.30         # at or above: walking; in between: slow
TURN_BINS_DPS = (10.0, 30.0, 60.0)   # turn-rate classes: <10, 10-30, 30-60, >=60 deg/s
MAX_ROW_GAP_S = 0.5     # a longer gap between pose rows is a pause, not time on a screen
WALK_JUMP_M = 0.5       # a longer step between two 10 Hz pose rows is a tracking jump
LIGHT_MATCH_S = 60.0    # a verification uses the last lux sample at most this old
MIN_CORR_N = 5          # rank correlation needs this many verifications
MB = 1024.0 * 1024.0

TRACE_DIR = "trace"
TRACE_COLUMNS = {
    "pose": ["t", "ns", "tracking", "failure", "wx", "wy", "wz", "qx", "qy", "qz", "qw", "located", "mx", "my",
             "mz", "myaw", "metersSinceFix", "speed", "turnRate", "screen", "room", "query"],
    "sensors": ["t", "ns", "type", "x", "y", "z", "accuracy"],
    "depth": ["t", "ns", "ok", "w", "h", "validShare", "usableShare", "median", "p10", "p90", "centre", "points",
              "planes", "planeArea", "lightIntensity", "speed", "turnRate"],
    "guidance": ["t", "ns", "query", "mode", "resolvable", "distance", "bearingDeg", "elevationDeg", "routeM",
                 "straightM", "inView", "ringX", "ringY"],
}
TEXT_COLUMNS = {"tracking", "failure", "screen", "room", "query", "type", "mode"}
NOMINAL_RATES_HZ = {"pose": 10.0, "depth": 2.0, "guidance": 5.0,
                    "acc": 50.0, "gyro": 50.0, "mag": 10.0, "baro": 1.0, "light": 5.0}


# ----------------------------------------------------------------------------- small helpers

def num(x):
    """A finite float, else None (booleans are not numbers)."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def pct(values, q):
    """Percentile q (0-100) of the finite values, linear interpolation; None when there are none."""
    a = np.asarray([v for v in (num(x) for x in values) if v is not None], dtype=float)
    return float(np.percentile(a, q)) if a.size else None


def median(values):
    return pct(values, 50)


def fmt(v, digits=2, unit=""):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return f"{int(v)}{unit}"
    return f"{float(v):.{digits}f}{unit}"


def share(v):
    return "n/a" if v is None else f"{100.0 * v:.1f}%"


def cm(metres):
    return None if metres is None else 100.0 * metres


def duration_text(seconds):
    if seconds is None:
        return "n/a"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h} h {m:02d} min {sec:02d} s" if h else f"{m} min {sec:02d} s"


def table(header, rows):
    """A Markdown table (list of lines). Cells are converted with str()."""
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in r) + " |")
    return out


def read_events(path):
    """events.jsonl as a list of dicts; unreadable lines are counted and skipped."""
    rows, bad = [], 0
    if not os.path.exists(path):
        return rows, bad
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(e, dict):
                rows.append(e)
            else:
                bad += 1
    return rows, bad


def read_trace_csv(path, kind):
    """One trace file as a DataFrame with numeric columns coerced; empty (with columns) if missing.

    A truncated last line (the app was killed while writing) is skipped.
    """
    cols = TRACE_COLUMNS[kind]
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame({c: pd.Series(dtype="object" if c in TEXT_COLUMNS else "float64") for c in cols})
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, on_bad_lines="skip")
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame({c: pd.Series(dtype="object" if c in TEXT_COLUMNS else "float64") for c in cols})
    # The app writes whole lines only, so a file that does not end with a line break ends with a
    # line cut off when the app was killed: drop it.
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) != b"\n" and len(df):
            df = df.iloc[:-1]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        if c in TEXT_COLUMNS:
            out[c] = df[c].astype(object).where(df[c] != "", None)
        else:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    out = out[out["t"].notna() & out["ns"].notna()].copy()
    return out.sort_values("ns", kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------------------- loading

class Session:
    """One session: its events (in file order) and its trace tables (None without a trace)."""

    def __init__(self, sid, events, trace_dir):
        self.id = sid
        self.events = events
        self.trace_dir = trace_dir if trace_dir and os.path.isdir(trace_dir) else None
        self.tables = {}
        self.file_bytes = {}
        if self.trace_dir:
            for kind in TRACE_COLUMNS:
                p = os.path.join(self.trace_dir, kind + ".csv")
                self.tables[kind] = read_trace_csv(p, kind)
                self.file_bytes[kind] = os.path.getsize(p) if os.path.exists(p) else 0

    @property
    def has_trace(self):
        return self.trace_dir is not None

    def of(self, *types):
        return [e for e in self.events if e.get("type") in types]

    def first(self, typ):
        for e in self.events:
            if e.get("type") == typ:
                return e
        return None

    def table(self, kind):
        return self.tables.get(kind)


def load(data_dir):
    """All sessions of a data folder: {session id: Session}, plus the count of unreadable lines."""
    events, bad = read_events(os.path.join(data_dir, "events.jsonl"))
    by_session = {}
    for e in events:
        by_session.setdefault(e.get("session"), []).append(e)
    trace_root = os.path.join(data_dir, TRACE_DIR)
    traced = set(os.listdir(trace_root)) if os.path.isdir(trace_root) else set()
    traced = {d for d in traced if os.path.isdir(os.path.join(trace_root, d))}
    sessions = {}
    for sid in set(by_session) | traced:
        if sid is None:
            continue
        sessions[sid] = Session(sid, by_session.get(sid, []), os.path.join(trace_root, sid))
    return sessions, bad


def session_order(sessions):
    def key(s):
        st = s.first("session_start")
        t = num(st.get("t")) if st else None
        if t is None:
            ts = [num(e.get("t")) for e in s.events if num(e.get("t")) is not None]
            t = min(ts) if ts else float("inf")
        return (t, str(s.id))
    return sorted(sessions.values(), key=key)


# ----------------------------------------------------------------------------- pose helpers

def pose_intervals(pose):
    """Per pose row: the time to the next row in seconds (0 across a pause) and continuity."""
    if pose is None or pose.empty:
        return np.zeros(0), np.zeros(0, dtype=bool)
    t = pose["t"].to_numpy(dtype=float)
    dt = np.diff(t, append=t[-1] if len(t) else 0.0) / 1000.0  # ms differences first: exact
    cont = (dt > 0) & (dt <= MAX_ROW_GAP_S)
    return np.where(cont, dt, 0.0), cont


def motion_class(speed):
    if speed is None or not math.isfinite(speed):
        return None
    if speed < STILL_MPS:
        return "still"
    if speed < WALK_MPS:
        return "slow"
    return "walking"


def turn_class(rate):
    if rate is None or not math.isfinite(rate):
        return None
    lo = 0.0
    for b in TURN_BINS_DPS:
        if rate < b:
            return f"{lo:g}-{b:g} deg/s"
        lo = b
    return f">={TURN_BINS_DPS[-1]:g} deg/s"


def pause_intervals(s):
    """[(pause t, resume t)] in epoch ms from app_pause and app_resume (an open pause is left out)."""
    out, start = [], None
    for e in sorted(s.of("app_pause", "app_resume"), key=lambda e: num(e.get("t")) or 0):
        t = num(e.get("t"))
        if t is None:
            continue
        if e["type"] == "app_pause":
            start = t
        elif start is not None:
            out.append((start, t))
            start = None
    return out


def overlap_ms(a, b, intervals):
    return sum(max(0.0, min(b, y) - max(a, x)) for x, y in intervals)


# ----------------------------------------------------------------------------- sections

def section_session(s, M):
    lines = []
    st = s.first("session_start") or {}
    ar = s.first("ar_config") or {}
    times = [num(e.get("t")) for e in s.events if num(e.get("t")) is not None]
    for kind, df in s.tables.items():
        if df is not None and not df.empty:
            times += [float(df["t"].min()), float(df["t"].max())]
    t0 = num(st.get("t")) if st else None
    if t0 is None and times:
        t0 = min(times)
    t1 = max(times) if times else None
    dur = (t1 - t0) / 1000.0 if t0 is not None and t1 is not None else None
    pauses = pause_intervals(s)
    paused_s = sum(b - a for a, b in pauses) / 1000.0
    M["session"] = {"start_ms": t0, "end_ms": t1, "duration_s": dur, "paused_s": paused_s,
                    "app_version": st.get("appVersion"), "trace_on": st.get("traceOn")}
    lines.append(f"- Session `{s.id}`: {duration_text(dur)} from the first to the last record"
                 + (f", of which {duration_text(paused_s)} paused ({len(pauses)} pauses)" if pauses else ""))
    lines.append(f"- App {st.get('appVersion') or 'n/a'} (commit {st.get('appCommit') or 'n/a'}), "
                 f"ARCore {st.get('arcoreVersion') or 'n/a'}, {st.get('device') or 'device n/a'}, "
                 f"Android {st.get('android') or 'n/a'}, display {st.get('displayPx') or 'n/a'}, "
                 f"detailed trace {st.get('traceOn') if 'traceOn' in st else 'n/a'}")
    cc = ar.get("cameraConfig") if isinstance(ar.get("cameraConfig"), dict) else {}
    if ar:
        lines.append(f"- AR: CPU image {cc.get('cpuImage', 'n/a')}, texture {cc.get('texture', 'n/a')}, "
                     f"{cc.get('fpsMin', '?')}-{cc.get('fpsMax', '?')} fps, depth {ar.get('depthMode', 'n/a')}, "
                     f"light estimation {ar.get('lightEstimationMode', 'n/a')}")
    if st.get("studyTag"):
        lines.append(f"- Study tag `{st.get('studyTag')}`, capture mode {st.get('captureMode')}")

    # Time per screen from ui_screen, paused time left out.
    screens = sorted(s.of("ui_screen"), key=lambda e: num(e.get("t")) or 0)
    per = Counter()
    for i, e in enumerate(screens):
        a = num(e.get("t"))
        b = num(screens[i + 1].get("t")) if i + 1 < len(screens) else t1
        if a is None or b is None or b <= a:
            continue
        per[e.get("to")] += (b - a - overlap_ms(a, b, pauses)) / 1000.0
    M["screen_s"] = dict(per)
    if per:
        total = sum(per.values()) or 1.0
        lines += ["", "Time per screen:", ""]
        lines += table(["Screen", "Time", "Share"],
                       [(k, duration_text(v), share(v / total)) for k, v in per.most_common()])
    actions = s.of("ui_action")
    plain = Counter(e.get("action") for e in actions if not str(e.get("action", "")).startswith("dialog:"))
    dialogs = Counter((e.get("action"), e.get("detail")) for e in actions if str(e.get("action", "")).startswith("dialog:"))
    M["actions"] = dict(plain)
    M["dialog_answers"] = {f"{a} -> {d}": n for (a, d), n in dialogs.items()}
    if plain:
        lines += ["", "Buttons and controls (`ui_action`):", ""]
        lines += table(["Action", "Count"], plain.most_common())
    if dialogs:
        lines += ["", "Dialog answers:", ""]
        lines += table(["Dialog", "Answer", "Count"], [(a[len("dialog:"):], d, n) for (a, d), n in dialogs.most_common()])
    searches = s.of("ui_search")
    if searches:
        empty = sum(1 for e in searches if (num(e.get("results")) or 0) == 0 and (num(e.get("chars")) or 0) > 0)
        M["searches"] = {"n": len(searches), "no_result": empty}
        lines += ["", f"Items searches: {len(searches)} (median {fmt(median([e.get('chars') for e in searches]), 0)} "
                      f"characters, {empty} with no result)"]
    if not (plain or dialogs or per):
        lines.append("- No `ui_screen` or `ui_action` events (the trace was off, or an older app version).")
    return lines


def section_messages(s, M):
    msgs = s.of("ui_message")
    c = Counter(e.get("text") for e in msgs)
    M["messages"] = dict(c)
    if not c:
        return ["No banner messages logged."]
    where = {}
    for e in msgs:
        where.setdefault(e.get("text"), Counter())[e.get("screen")] += 1
    rows = [(n, text, ", ".join(f"{k}" for k, _ in where[text].most_common(2) if k)) for text, n in c.most_common()]
    return table(["Count", "Message", "Screens"], rows)


def section_rejections(s, M):
    lines = []
    rej = s.of("capture_rejected")
    by = Counter((e.get("stage"), e.get("reason")) for e in rej)
    saved = len(s.of("register"))
    refused_reg = sum(n for (stage, _), n in by.items() if stage == "register")
    M["rejected"] = {f"{a}/{b}": n for (a, b), n in by.items()}
    M["saved"] = saved
    M["refused_register"] = refused_reg
    if by:
        rows = []
        for (stage, reason), n in by.most_common():
            vals = [e.get("value") for e in rej if e.get("stage") == stage and e.get("reason") == reason]
            v = median(vals)
            rows.append((stage, reason, n, fmt(v, 2) if v is not None else ""))
        lines += table(["Stage", "Reason", "Count", "Median value"], rows)
    else:
        lines.append("No refused captures logged.")
    lines.append("")
    lines.append(f"Saved items (`register`): {saved}; refused registration taps: {refused_reg}"
                 + (f" ({refused_reg / saved:.1f} per saved item)" if saved else ""))
    hits = s.of("hit_rejected")
    if hits:
        lines.append(f"`hit_rejected` (depth over 5 m): {len(hits)}, median {fmt(median([e.get('distance') for e in hits]), 1)} m")
    locks = s.of("lock_attempt")
    ok = [e for e in locks if e.get("ok") is True]
    M["locks"] = {"attempts": len(locks), "ok": len(ok),
                  "success_rate": (len(ok) / len(locks)) if locks else None,
                  "presses_per_lock": (len(locks) / len(ok)) if ok else None,
                  "reasons": dict(Counter(e.get("reason") for e in locks if e.get("ok") is not True))}
    lines.append("")
    if locks:
        lines.append(f"Lock presses: {len(locks)}, succeeded {len(ok)} ({share(len(ok) / len(locks))}); "
                     f"presses per successful lock {fmt(M['locks']['presses_per_lock'], 1)}")
        rows = []
        for kind in sorted({e.get("kind") for e in locks}, key=str):
            k = [e for e in locks if e.get("kind") == kind]
            kok = [e for e in k if e.get("ok") is True]
            spread = median([e.get("posSpreadM") for e in kok])
            rows.append((kind, len(k), len(kok), share(len(kok) / len(k)),
                         fmt(median([e.get("durationMs") for e in kok]), 0),
                         fmt(None if spread is None else 100 * spread, 1)))
        lines += [""] + table(["Kind", "Presses", "Succeeded", "Rate", "Median ms (ok)", "Median spread cm (ok)"], rows)
        reasons = M["locks"]["reasons"]
        if reasons:
            lines += ["", "Failed presses by reason: " + ", ".join(f"{k} {v}" for k, v in Counter(reasons).most_common())]
        firsts = [e for e in locks if num(e.get("attemptInBar")) == 1]
        if firsts:
            lines.append(f"First press in the bar succeeded: {sum(1 for e in firsts if e.get('ok') is True)}/{len(firsts)}")
    else:
        lines.append("No `lock_attempt` events.")
    return lines


def section_tracking(s, M):
    pose = s.table("pose")
    if pose is None:
        M["tracking"] = None
        lost_events = s.of("tracking_lost")
        return ["No trace." + (f" `tracking_lost` events (1 s or longer): {len(lost_events)}" if lost_events else "")]
    if pose.empty:
        M["tracking"] = None
        return ["`pose.csv` is empty."]
    dt, cont = pose_intervals(pose)
    tracking = (pose["tracking"].fillna("") == "TRACKING").to_numpy()
    traced = float(dt.sum())
    lost_dt = float(dt[~tracking].sum())
    # A loss starts where a tracking row is followed (without a pause) by a non-tracking row.
    starts = int(np.sum(tracking[:-1] & ~tracking[1:] & cont[:-1])) if len(tracking) > 1 else 0
    reasons = Counter()
    for r, d, trk in zip(pose["failure"].fillna("NONE"), dt, tracking):
        if not trk:
            reasons[r] += d
    M["tracking"] = {"traced_s": traced, "lost_s": lost_dt, "losses": starts,
                     "lost_share": (lost_dt / traced) if traced > 0 else None, "by_reason_s": dict(reasons)}
    lines = [f"Traced time {duration_text(traced)} (pose rows: {len(pose)}). Tracking lost {starts} times, "
             f"{fmt(lost_dt, 1)} s in total ({share(M['tracking']['lost_share'])} of the traced time)."]
    if reasons:
        lines += [""] + table(["Reason", "Lost time s", "Share of lost time"],
                              [(k, fmt(v, 1), share(v / lost_dt) if lost_dt else "n/a") for k, v in reasons.most_common()])
    ev = s.of("tracking_lost")
    if ev:
        lines.append("")
        lines.append(f"`tracking_lost` events (pauses of 1 s or longer): {len(ev)}, by reason: "
                     + ", ".join(f"{k} {v}" for k, v in Counter(e.get('reason') for e in ev).most_common()))
    return lines


def section_motion(s, M):
    pose = s.table("pose")
    if pose is None or pose.empty:
        M["motion"] = None
        return ["No trace."]
    dt, cont = pose_intervals(pose)
    trk = (pose["tracking"].fillna("") == "TRACKING").to_numpy()
    x = pose["wx"].to_numpy(dtype=float)
    z = pose["wz"].to_numpy(dtype=float)
    step = np.hypot(np.diff(x), np.diff(z)) if len(x) > 1 else np.zeros(0)
    ok = cont[:-1] & trk[:-1] & trk[1:] & np.isfinite(step) & (step < WALK_JUMP_M) if len(x) > 1 else np.zeros(0, bool)
    walked = float(step[ok].sum()) if len(step) else 0.0
    speed = pose["speed"].to_numpy(dtype=float)
    classes = Counter()
    for v, d, t in zip(speed, dt, trk):
        c = motion_class(v)
        if c is not None and t:
            classes[c] += d
    total = sum(classes.values())
    M["motion"] = {"walked_m": walked, "speed_p10": pct(speed, 10), "speed_p50": pct(speed, 50),
                   "speed_p90": pct(speed, 90), "speed_max": pct(speed, 100),
                   "share": {k: (v / total if total else None) for k, v in classes.items()}}
    lines = [f"Walked {fmt(walked, 1)} m (horizontal, while tracking). Speed m/s: p10 {fmt(M['motion']['speed_p10'], 2)}, "
             f"median {fmt(M['motion']['speed_p50'], 2)}, p90 {fmt(M['motion']['speed_p90'], 2)}, "
             f"max {fmt(M['motion']['speed_max'], 2)}."]
    if total:
        lines.append(f"Time still (< {STILL_MPS} m/s) {share(classes['still'] / total)}, "
                     f"slow {share(classes['slow'] / total)}, walking (>= {WALK_MPS} m/s) {share(classes['walking'] / total)}.")
    turn = pose["turnRate"].to_numpy(dtype=float)
    lines.append(f"Turn rate deg/s: median {fmt(pct(turn, 50), 1)}, p90 {fmt(pct(turn, 90), 1)}.")
    return lines


def section_depth(s, M):
    depth = s.table("depth")
    if depth is None:
        M["depth"] = None
        kf = s.of("scan_keyframe")
        if kf:
            return [f"No trace. Scan keyframes: median valid depth share {share(median([e.get('validDepth') for e in kf]))}."]
        return ["No trace."]
    if depth.empty:
        M["depth"] = None
        return ["`depth.csv` is empty."]
    okmask = depth["ok"] == 1
    with_img = depth[okmask]
    no_depth = float((~okmask).mean())
    usable = float(with_img["usableShare"].mean()) if len(with_img) else None
    by_motion, by_turn = {}, {}
    rows_m, rows_t = [], []
    mc = with_img["speed"].map(lambda v: motion_class(v) if pd.notna(v) else None)
    for c in ("still", "slow", "walking"):
        sub = with_img[mc == c]
        if len(sub):
            by_motion[c] = float(sub["usableShare"].mean())
        rows_m.append((c, len(sub), share(by_motion.get(c)),
                       share(float(sub["validShare"].mean())) if len(sub) else "n/a",
                       fmt(median(sub["median"]), 2)))
    tc = with_img["turnRate"].map(lambda v: turn_class(v) if pd.notna(v) else None)
    labels = [turn_class(b - 1e-9) for b in TURN_BINS_DPS] + [turn_class(TURN_BINS_DPS[-1])]
    for c in labels:
        sub = with_img[tc == c]
        if len(sub):
            by_turn[c] = float(sub["usableShare"].mean())
        rows_t.append((c, len(sub), share(by_turn.get(c))))
    # Share of samples without a depth image, by motion too (depth may stop when standing still).
    allmc = depth["speed"].map(lambda v: motion_class(v) if pd.notna(v) else None)
    nod_motion = {c: float((~okmask[allmc == c]).mean()) for c in ("still", "slow", "walking") if (allmc == c).any()}
    M["depth"] = {"samples": len(depth), "no_depth_share": no_depth, "usable_share": usable,
                  "usable_by_motion": by_motion, "usable_by_turn": by_turn, "no_depth_by_motion": nod_motion,
                  "median_m": median(with_img["median"]), "centre_m": median(with_img["centre"])}
    lines = [f"Depth samples {len(depth)}; without a depth image {share(no_depth)}. "
             f"Usable depth share (0.2-4 m, samples with an image): mean {share(usable)}; "
             f"median depth {fmt(M['depth']['median_m'], 2)} m, centre {fmt(M['depth']['centre_m'], 2)} m."]
    lines += ["", "By motion:", ""] + table(["Motion", "Samples", "Usable", "Valid", "Median depth m"], rows_m)
    if nod_motion:
        lines += ["", "No depth image by motion: " + ", ".join(f"{k} {share(v)}" for k, v in nod_motion.items())]
    lines += ["", "By turn rate:", ""] + table(["Turn rate", "Samples", "Usable"], rows_t)
    return lines


def section_scene(s, M):
    depth = s.table("depth")
    if depth is None or depth.empty:
        M["scene"] = None
        return ["No trace."]
    pts, area, planes = depth["points"], depth["planeArea"], depth["planes"]
    M["scene"] = {"points_p10": pct(pts, 10), "points_p50": pct(pts, 50), "points_p90": pct(pts, 90),
                  "area_p10": pct(area, 10), "area_p50": pct(area, 50), "area_p90": pct(area, 90),
                  "planes_p50": pct(planes, 50)}
    m = M["scene"]
    return [f"Feature points per frame: p10 {fmt(m['points_p10'], 0)}, median {fmt(m['points_p50'], 0)}, "
            f"p90 {fmt(m['points_p90'], 0)}.",
            f"Tracked planes: median {fmt(m['planes_p50'], 0)}; plane area m²: p10 {fmt(m['area_p10'], 2)}, "
            f"median {fmt(m['area_p50'], 2)}, p90 {fmt(m['area_p90'], 2)}."]


def light_samples(s):
    sens = s.table("sensors")
    if sens is None or sens.empty:
        return pd.DataFrame(columns=["t", "x"])
    return sens[sens["type"] == "light"][["t", "x"]].dropna().sort_values("t").reset_index(drop=True)


def spearman(a, b):
    a = pd.Series(a, dtype=float).rank()
    b = pd.Series(b, dtype=float).rank()
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def section_light(s, M):
    lux = light_samples(s)
    depth = s.table("depth")
    lines = []
    M["light"] = {"lux_p10": pct(lux["x"], 10), "lux_p50": pct(lux["x"], 50), "lux_p90": pct(lux["x"], 90),
                  "intensity_p50": pct(depth["lightIntensity"], 50) if depth is not None and not depth.empty else None,
                  "verify": []}
    if s.has_trace:
        lines.append(f"Ambient light (lux, {len(lux)} samples): p10 {fmt(M['light']['lux_p10'], 0)}, "
                     f"median {fmt(M['light']['lux_p50'], 0)}, p90 {fmt(M['light']['lux_p90'], 0)}. "
                     f"ARCore pixel intensity: median {fmt(M['light']['intensity_p50'], 3)}.")
    else:
        lines.append("No trace.")
    ver = s.of("verify")
    if ver and len(lux):
        lt = lux["t"].to_numpy(dtype=float)
        lx = lux["x"].to_numpy(dtype=float)
        rows = []
        for e in ver:
            t = num(e.get("t"))
            sim = num(e.get("bestSimilarity"))
            if t is None or sim is None:
                continue
            i = int(np.searchsorted(lt, t, side="right")) - 1
            v = lx[i] if i >= 0 and (t - lt[i]) / 1000.0 <= LIGHT_MATCH_S else None
            rows.append((e.get("query"), e.get("attempt"), sim, v, e.get("outcome")))
        M["light"]["verify"] = [{"similarity": r[2], "lux": r[3]} for r in rows]
        paired = [(r[2], r[3]) for r in rows if r[3] is not None]
        if rows:
            lines += ["", "Verifications and the light at capture:", ""]
            lines += table(["Query", "Attempt", "Similarity", "Lux", "Outcome"],
                           [(q, a, fmt(sim, 3), fmt(v, 0), o) for q, a, sim, v, o in rows])
        if len(paired) >= MIN_CORR_N:
            rho = spearman([p[0] for p in paired], [p[1] for p in paired])
            M["light"]["spearman"] = rho
            lines.append("")
            lines.append(f"Rank correlation of similarity with lux (n={len(paired)}): {fmt(rho, 2)}")
    elif ver:
        lines.append(f"{len(ver)} verifications, but no lux samples to compare them with.")
    return lines


def battery_drain(states):
    """Battery % per hour over consecutive device_state pairs that are both not charging."""
    rows = sorted([e for e in states if num(e.get("t")) is not None and num(e.get("batteryPct")) is not None],
                  key=lambda e: num(e.get("t")))
    drop, hours = 0.0, 0.0
    for a, b in zip(rows, rows[1:]):
        if a.get("charging") is False and b.get("charging") is False:
            h = (num(b["t"]) - num(a["t"])) / 3.6e6
            if 0 < h < 2:
                drop += num(a["batteryPct"]) - num(b["batteryPct"])
                hours += h
    return (drop / hours if hours >= 0.05 else None), hours


def latency_rows(s):
    rows = []
    for typ, keys in (("register", ("convertMs", "detectMs", "embedMs")),
                      ("verify", ("convertMs", "detectMs", "embedMs", "totalMs"))):
        ev = s.of(typ)
        if not ev:
            continue
        for k in keys:
            vals = [e.get("latency", {}).get(k) for e in ev if isinstance(e.get("latency"), dict)]
            if any(num(v) is not None for v in vals):
                rows.append((typ, k, len(vals), fmt(median(vals), 0), fmt(pct(vals, 90), 0)))
    kf = [e for e in s.of("scan_keyframe") if isinstance(e.get("latency"), dict)]
    names = sorted({k for e in kf for k in e["latency"]})
    for k in names:
        vals = [e["latency"].get(k) for e in kf]
        rows.append(("scan_keyframe", k, len(vals), fmt(median(vals), 0), fmt(pct(vals, 90), 0)))
    for e in s.of("model_loaded"):
        rows.append(("model_loaded", e.get("model") or "", 1, fmt(num(e.get("loadMs")), 0), ""))
    return rows


def section_performance(s, M):
    lines = []
    perf = s.of("perf")
    fps = [e.get("fps") for e in perf]
    p95 = [e.get("frameMsP95") for e in perf]
    states = s.of("device_state")
    drain, drain_h = battery_drain(states)
    thermal = [num(e.get("thermalStatus")) for e in states if num(e.get("thermalStatus")) is not None]
    dropped_vals = [num(e.get("traceDropped")) for e in s.of("perf", "app_pause", "device_state")
                    if num(e.get("traceDropped")) is not None]
    M["perf"] = {"windows": len(perf), "fps_p50": median(fps), "fps_p5": pct(fps, 5),
                 "frame_ms_mean_p50": median([e.get("frameMsMean") for e in perf]),
                 "frame_ms_p95_p50": median(p95), "frame_ms_p95_max": pct(p95, 100),
                 "frame_ms_max": pct([e.get("frameMsMax") for e in perf], 100),
                 "trace_ms_mean_p50": median([e.get("traceMsMean") for e in perf]),
                 "ml_busy_mean": (float(np.mean([v for v in (num(e.get("mlBusyShare")) for e in perf) if v is not None]))
                                  if any(num(e.get("mlBusyShare")) is not None for e in perf) else None),
                 "battery_drain_pct_h": drain, "battery_hours": drain_h,
                 "thermal_max": max(thermal) if thermal else None,
                 "battery_temp_max": pct([e.get("batteryTempC") for e in states], 100),
                 "trace_dropped": max(dropped_vals) if dropped_vals else None,
                 "trace_errors": [(e.get("site"), e.get("error")) for e in s.of("trace_error")]}
    m = M["perf"]
    if perf:
        lines.append(f"GL frames ({len(perf)} windows of 10 s): fps median {fmt(m['fps_p50'], 1)}, 5th percentile "
                     f"{fmt(m['fps_p5'], 1)}; frame time mean {fmt(m['frame_ms_mean_p50'], 2)} ms (median window), "
                     f"p95 {fmt(m['frame_ms_p95_p50'], 2)} ms (median window, worst {fmt(m['frame_ms_p95_max'], 2)}), "
                     f"max {fmt(m['frame_ms_max'], 1)} ms. Trace time on the GL thread {fmt(m['trace_ms_mean_p50'], 3)} "
                     f"ms per frame (median window). Capture being analysed in {share(m['ml_busy_mean'])} of frames.")
    else:
        lines.append("No `perf` events.")
    lat = latency_rows(s)
    if lat:
        lines += ["", "ML latencies (ms):", ""] + table(["Event", "Step", "n", "Median", "p90"], lat)
    lines.append("")
    if states:
        cur = [num(e.get("currentNowMa")) for e in states if e.get("charging") is False]
        lines.append(f"Battery: drain {fmt(drain, 1)} %/h over {fmt(drain_h, 2)} h not charging"
                     f"; median current {fmt(median(cur), 0)} mA (not charging); max battery temperature "
                     f"{fmt(m['battery_temp_max'], 1)} °C; max thermal status {fmt(m['thermal_max'], 0)} "
                     f"(0 none, 1 light, 2 moderate, 3 severe); {len(states)} `device_state` samples.")
    else:
        lines.append("No `device_state` events.")
    lines.append(f"Trace lines dropped: {fmt(m['trace_dropped'], 0)}.")
    if m["trace_errors"]:
        lines.append("Trace errors: " + "; ".join(f"{a}: {b}" for a, b in m["trace_errors"]))
    if s.has_trace:
        # Rows per traced second: the pose rows' time, which runs exactly while the app is resumed.
        rows = []
        traced = (M.get("tracking") or {}).get("traced_s")
        guided = None
        g = s.table("guidance")
        if g is not None and len(g) > 1:
            dt = np.diff(g["t"].to_numpy(dtype=float)) / 1000.0
            guided = float(dt[dt <= MAX_ROW_GAP_S * 2].sum())
        for kind, df in s.tables.items():
            n = 0 if df is None else len(df)
            secs = guided if kind == "guidance" else traced
            rows.append((kind + ".csv", n, f"{s.file_bytes.get(kind, 0) / MB:.2f}", fmt(n / secs if secs else None, 1),
                         fmt(NOMINAL_RATES_HZ.get(kind), 0)))
            if kind == "sensors" and df is not None and not df.empty:
                for typ, grp in df.groupby("type"):
                    rows.append((f"  {typ}", len(grp), "", fmt(len(grp) / traced if traced else None, 1),
                                 fmt(NOMINAL_RATES_HZ.get(typ), 0)))
        total_mb = sum(s.file_bytes.values()) / MB
        hours = (traced or 0) / 3600.0
        lines += ["", f"Trace files: {total_mb:.2f} MB" + (f" ({total_mb / hours:.1f} MB per traced hour)" if hours > 0.01 else "")
                  + ". Rows/s: rows per traced second (guidance: per second of search).", ""]
        lines += table(["File", "Rows", "MB", "Rows/s", "Nominal Hz"], rows)
        M["files"] = {"total_mb": total_mb, "rows": {k: (0 if v is None else len(v)) for k, v in s.tables.items()}}
    return lines


def section_scan(s, M):
    starts, ends, kfs, checks = s.of("scan_start"), s.of("scan_end"), s.of("scan_keyframe"), s.of("scan_depth_check")
    if not (starts or ends or kfs):
        M["scan"] = None
        return None
    dropped = Counter()
    for e in kfs:
        d = e.get("dropped")
        if isinstance(d, dict):
            for k, v in d.items():
                if num(v) is not None:
                    dropped[k] += int(num(v))
    det = sum(int(num(e.get("detections")) or 0) for e in kfs)
    kept = sum(int(num(e.get("kept")) or 0) for e in kfs)
    dist = [e.get("distanceM") for e in checks]
    M["scan"] = {"scans": len(starts), "keyframes": len(kfs), "detections": det, "kept": kept,
                 "dropped": dict(dropped), "depth_check_p50_m": median(dist), "depth_check_p90_m": pct(dist, 90),
                 "skipped_no_depth": sum(int(num(e.get("skippedNoDepth")) or 0) for e in ends)}
    m = M["scan"]
    lines = [f"Scans {len(starts)}, keyframes {len(kfs)} (skipped without depth: {m['skipped_no_depth']}); "
             f"detections {det}, kept {kept}.",
             "Dropped by reason: " + (", ".join(f"{k} {v}" for k, v in dropped.most_common()) or "none") + ".",
             f"Depth check (pipeline point vs hit test at the image centre): median "
             f"{fmt(cm(m['depth_check_p50_m']), 1)} cm, p90 {fmt(cm(m['depth_check_p90_m']), 1)} cm "
             f"(n={sum(num(v) is not None for v in dist)})."]
    for e in ends:
        lines.append(f"- scan ended ({e.get('reason')}): {fmt((num(e.get('durationMs')) or 0) / 1000, 0)} s, "
                     f"{fmt(num(e.get('metersWalked')), 1)} m walked, {e.get('keyframes')} keyframes")
    return lines


def findings(M):
    """Flags from the fixed thresholds at the top of the file."""
    out = []
    tr = M.get("tracking") or {}
    if tr.get("lost_share") is not None and tr["lost_share"] > TRACKING_LOST_SHARE_MAX:
        out.append(f"Tracking lost {share(tr['lost_share'])} of the traced time (over {share(TRACKING_LOST_SHARE_MAX)}).")
    dp = M.get("depth") or {}
    walk = (dp.get("usable_by_motion") or {}).get("walking")
    if walk is not None and walk < DEPTH_USABLE_WALKING_MIN:
        out.append(f"Usable depth while walking {share(walk)} (below {share(DEPTH_USABLE_WALKING_MIN)}).")
    if dp.get("no_depth_share") is not None and dp["no_depth_share"] > NO_DEPTH_SHARE_MAX:
        out.append(f"No depth image in {share(dp['no_depth_share'])} of the depth samples (over {share(NO_DEPTH_SHARE_MAX)}).")
    pf = M.get("perf") or {}
    if pf.get("fps_p5") is not None and pf["fps_p5"] < FPS_P5_MIN:
        out.append(f"fps 5th percentile {fmt(pf['fps_p5'], 1)} (below {FPS_P5_MIN:g}).")
    if pf.get("frame_ms_p95_p50") is not None and pf["frame_ms_p95_p50"] > FRAME_MS_P95_MAX:
        out.append(f"GL frame time p95 {fmt(pf['frame_ms_p95_p50'], 1)} ms in the median window (over {FRAME_MS_P95_MAX:g} ms).")
    if pf.get("thermal_max") is not None and pf["thermal_max"] >= THERMAL_STATUS_MODERATE:
        out.append(f"Thermal status reached {int(pf['thermal_max'])} (moderate or above).")
    if pf.get("trace_dropped") is not None and pf["trace_dropped"] > TRACE_DROPPED_MAX:
        out.append(f"{int(pf['trace_dropped'])} trace lines dropped.")
    if pf.get("trace_errors"):
        out.append(f"{len(pf['trace_errors'])} trace errors (see Performance).")
    saved, refused = M.get("saved") or 0, M.get("refused_register") or 0
    if saved and refused / saved > REFUSED_PER_SAVED_MAX:
        out.append(f"{refused} refused registration taps for {saved} saved items ({refused / saved:.1f} per item, over {REFUSED_PER_SAVED_MAX:g}).")
    elif not saved and refused > REFUSED_PER_SAVED_MAX:
        out.append(f"{refused} refused registration taps and no item saved.")
    sc = M.get("scan") or {}
    if sc.get("depth_check_p50_m") is not None and sc["depth_check_p50_m"] > DEPTH_CHECK_MAX_M:
        out.append(f"Scan depth check median {fmt(cm(sc['depth_check_p50_m']), 1)} cm (over {cm(DEPTH_CHECK_MAX_M):g} cm).")
    lk = M.get("locks") or {}
    if lk.get("success_rate") is not None and lk["success_rate"] < LOCK_SUCCESS_MIN:
        out.append(f"Only {share(lk['success_rate'])} of the Lock presses succeeded (below {share(LOCK_SUCCESS_MIN)}).")
    return out


# ----------------------------------------------------------------------------- report

def summarize(s):
    """All sections of one session: (markdown lines, metrics dict, findings list)."""
    M = {"session_id": s.id, "has_trace": s.has_trace}
    parts = [("Session", section_session(s, M)),
             ("Messages", section_messages(s, M)),
             ("Rejected captures and locks", section_rejections(s, M)),
             ("Tracking", section_tracking(s, M)),
             ("Motion", section_motion(s, M)),
             ("Depth", section_depth(s, M)),
             ("Scene", section_scene(s, M)),
             ("Light", section_light(s, M)),
             ("Performance", section_performance(s, M))]
    scan = section_scan(s, M)
    if scan is not None:
        parts.append(("Scan", scan))
    F = findings(M)
    M["findings"] = F
    md = [f"# FindBack trace summary: session {s.id}", ""]
    if not s.has_trace:
        md += ["This session has no trace folder (the setting was off, or an app version before v0.7): "
               "only the sections that come from events.jsonl have content.", ""]
    for title, lines in parts:
        md += [f"## {title}", ""] + lines + [""]
    md += ["## Findings", ""] + ([f"- {f}" for f in F] if F else ["- None of the fixed thresholds was crossed."]) + [""]
    return md, M, F


def plot_session(s, out_dir):
    """Top-view trajectory (by tracking state and by usable depth) and speed and depth over time.

    Returns the paths written; nothing without a pose table.
    """
    pose = s.table("pose")
    if pose is None or pose.empty:
        return []
    try:
        # The Figure API needs no pyplot and no backend switch, and the explicit layout ignores
        # whatever another module set in rcParams.
        from matplotlib.figure import Figure
    except ImportError:
        return []
    written = []
    depth = s.table("depth")
    trk = pose[pose["wx"].notna()].copy()
    fig = Figure(figsize=(10, 5), layout="constrained")
    axes = fig.subplots(1, 2)
    ok = trk["tracking"] == "TRACKING"
    axes[0].plot(trk["wx"], trk["wz"], color="0.8", lw=0.5, zorder=1)
    axes[0].scatter(trk.loc[ok, "wx"], trk.loc[ok, "wz"], s=2, c="tab:green", label="tracking", zorder=2)
    lost = pose[pose["tracking"] != "TRACKING"]
    if len(lost) and len(trk):
        # A lost row has no pose: mark it at the last tracked position before it.
        idx = np.searchsorted(trk["ns"].to_numpy(), lost["ns"].to_numpy(), side="right") - 1
        idx = idx[idx >= 0]
        if len(idx):
            axes[0].scatter(trk["wx"].to_numpy()[idx], trk["wz"].to_numpy()[idx], s=12, c="tab:red", marker="x",
                            label="tracking lost", zorder=3)
    axes[0].set_title("Top view by tracking state")
    if depth is not None and not depth.empty and len(trk):
        d = depth[depth["ok"] == 1][["ns", "usableShare"]].dropna().sort_values("ns")
        p = trk[["ns", "wx", "wz"]].sort_values("ns")
        if len(d) and len(p):
            d = d.assign(ns=d["ns"].astype("int64"))
            p = p.assign(ns=p["ns"].astype("int64"))
            j = pd.merge_asof(d, p, on="ns", direction="nearest", tolerance=int(0.5e9)).dropna()
            axes[1].plot(trk["wx"], trk["wz"], color="0.85", lw=0.5, zorder=1)
            sc = axes[1].scatter(j["wx"], j["wz"], c=j["usableShare"], cmap="viridis", vmin=0, vmax=1, s=10, zorder=2)
            fig.colorbar(sc, ax=axes[1], label="usable depth share")
    axes[1].set_title("Top view by usable depth")
    for ax in axes:
        ax.set_xlabel("world x (m)")
        ax.set_ylabel("world z (m)")
        ax.set_aspect("equal", adjustable="datalim")
    axes[0].legend(loc="best", fontsize=8)
    p1 = os.path.join(out_dir, f"trace_{s.id}_trajectory.png")
    fig.savefig(p1, dpi=120)
    written.append(p1)

    t0 = float(pose["t"].min())
    fig = Figure(figsize=(10, 5), layout="constrained")
    axes = fig.subplots(2, 1, sharex=True)
    axes[0].plot((pose["t"] - t0) / 60000.0, pose["speed"], lw=0.6)
    axes[0].axhline(WALK_MPS, color="0.6", lw=0.5, ls="--")
    axes[0].set_ylabel("speed (m/s)")
    if depth is not None and not depth.empty:
        axes[1].plot((depth["t"] - t0) / 60000.0, depth["usableShare"], lw=0.6, color="tab:purple")
        nod = depth[depth["ok"] != 1]
        axes[1].scatter((nod["t"] - t0) / 60000.0, np.zeros(len(nod)), s=4, c="tab:red", label="no depth image")
        if len(nod):
            axes[1].legend(loc="best", fontsize=8)
    axes[1].set_ylabel("usable depth share")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("minutes since the first pose row")
    p2 = os.path.join(out_dir, f"trace_{s.id}_timeline.png")
    fig.savefig(p2, dpi=120)
    written.append(p2)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--session", action="append", default=[], help="only this session (repeatable)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="folder for the Markdown and the plots (default analysis/out)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.data_dir):
        print(f"No such folder: {args.data_dir}", file=sys.stderr)
        return 2
    sessions, bad = load(args.data_dir)
    if args.session:
        missing = [x for x in args.session if x not in sessions]
        for x in missing:
            print(f"Session {x} is not in {args.data_dir}", file=sys.stderr)
        chosen = [sessions[x] for x in args.session if x in sessions]
        if not chosen:
            return 1
    else:
        chosen = [s for s in session_order(sessions) if s.has_trace]
        untraced = sum(1 for s in sessions.values() if not s.has_trace)
        if not chosen:
            print(f"No session with a trace in {args.data_dir} ({untraced} sessions without one).")
            return 0
        if untraced:
            print(f"({untraced} sessions without a trace are left out; name one with --session to see its events.)\n")
    if bad:
        print(f"({bad} unreadable lines in events.jsonl were skipped.)\n")
    os.makedirs(args.out, exist_ok=True)
    for s in chosen:
        md, M, F = summarize(s)
        text = "\n".join(md)
        path = os.path.join(args.out, f"trace_summary_{s.id}.md")
        plots = [] if args.no_plots else plot_session(s, args.out)
        if plots:
            text += "\n## Plots\n\n" + "\n".join(f"![{os.path.basename(p)}]({os.path.basename(p)})" for p in plots) + "\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(text)
        print(f"(written to {path})\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
