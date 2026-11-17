#!/usr/bin/env python3
"""Tests for trace_summary.py (FindBack v0.7 detailed trace, plan section 6 item 2).

Run:  python3 analysis/test_trace_summary.py          (one line per test, exit code 1 on failure)
      python3 -m pytest analysis/test_trace_summary.py

Every trace here is built by the test itself in a temporary folder that is removed afterwards.
No real data is read and nothing is written outside the temporary folder.
"""
import contextlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fb_load as L  # noqa: E402
import trace_summary as T  # noqa: E402

T0 = 1_800_000_000_000  # epoch ms of the hand-built session start
NS0 = 5_000_000_000_000  # elapsedRealtimeNanos at the start


class tmp:
    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="fb_trace_test_")
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


# ----------------------------------------------------------------------------- building traces

def ev(sid, dt_ms, typ, **fields):
    return dict({"t": T0 + dt_ms, "session": sid, "type": typ}, **fields)


def write_events(root, events):
    with open(os.path.join(root, "events.jsonl"), "a", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def csv_cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


def write_csv(root, sid, kind, rows, raw_tail=None):
    d = os.path.join(root, "trace", sid)
    os.makedirs(d, exist_ok=True)
    cols = T.TRACE_COLUMNS[kind]
    with open(os.path.join(d, kind + ".csv"), "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(csv_cell(r.get(c)) for c in cols) + "\n")
        if raw_tail:
            f.write(raw_tail)


def pose_row(i, tracking=True, x=0.0, z=0.0, speed=None, turn=None, failure=None, screen="READY", dt_ms=100):
    """Row i of a 10 Hz pose trace (dt_ms apart)."""
    r = {"t": T0 + i * dt_ms, "ns": NS0 + i * dt_ms * 1_000_000, "tracking": "TRACKING" if tracking else "PAUSED",
         "failure": failure, "located": 1, "screen": screen, "speed": speed, "turnRate": turn}
    if tracking:
        r.update(wx=x, wy=1.4, wz=z, qx=0, qy=0, qz=0, qw=1)
    return r


def depth_row(i, ok=True, usable=0.8, valid=0.9, speed=0.0, turn=0.0, points=120, planes=2, area=1.5, light=0.4):
    """Row i of a 2 Hz depth trace."""
    r = {"t": T0 + i * 500, "ns": NS0 + i * 500_000_000, "ok": 1 if ok else 0, "points": points, "planes": planes,
         "planeArea": area, "lightIntensity": light, "speed": speed, "turnRate": turn}
    if ok:
        r.update(w=160, h=90, validShare=valid, usableShare=usable, median=1.2, p10=0.6, p90=2.5, centre=1.1)
    return r


def summary_of(root, sid):
    sessions, _ = T.load(root)
    return T.summarize(sessions[sid])


# ----------------------------------------------------------------------------- sections

def test_session_screens_actions_and_versions():
    with tmp() as root:
        sid = "s1"
        write_events(root, [
            ev(sid, 0, "session_start", appVersion="0.7", appCommit="abc1234", arcoreVersion="1.48", traceOn=True,
               device="Pixel 8", android="17", displayPx="1080x2400", studyTag=""),
            ev(sid, 500, "ar_config", cameraConfig={"cpuImage": "1920x1080", "texture": "1920x1080", "fpsMin": 30,
                                                     "fpsMax": 30}, depthMode="AUTOMATIC", lightEstimationMode="AMBIENT_INTENSITY"),
            ev(sid, 1000, "ui_screen", **{"from": None, "to": "NOT_LOCATED", "msInPrevious": None}),
            ev(sid, 11000, "ui_screen", **{"from": "NOT_LOCATED", "to": "READY", "msInPrevious": 10000}),
            # 5 s paused while READY: not counted as time on the screen.
            ev(sid, 21000, "app_pause", sessionMs=21000, traceBytes=10, traceDropped=0),
            ev(sid, 26000, "app_resume", sessionMs=26000),
            ev(sid, 41000, "ui_screen", **{"from": "READY", "to": "FINDING", "msInPrevious": 30000}),
            ev(sid, 2000, "ui_action", action="LOCATE", screen="NOT_LOCATED", detail=None),
            ev(sid, 3000, "ui_action", action="LOCK", screen="LOCKING", detail=None),
            ev(sid, 4000, "ui_action", action="LOCK", screen="LOCKING", detail=None),
            ev(sid, 42000, "ui_action", action="dialog:found_it", screen="FINDING", detail="Keep looking"),
            ev(sid, 43000, "ui_action", action="dialog:found_it", screen="FINDING", detail="Yes, it's here"),
            ev(sid, 44000, "ui_search", chars=4, results=0, total=7, roomFilter=None),
            ev(sid, 51000, "query_end", query="q", groundTruth="present"),
        ])
        write_csv(root, sid, "pose", [pose_row(i) for i in range(10)])
        md, M, F = summary_of(root, sid)
        text = "\n".join(md)
        assert M["session"]["app_version"] == "0.7"
        assert close(M["session"]["duration_s"], 51.0)
        assert close(M["session"]["paused_s"], 5.0)
        assert close(M["screen_s"]["NOT_LOCATED"], 10.0)
        assert close(M["screen_s"]["READY"], 25.0)  # 30 s minus the 5 s pause
        assert close(M["screen_s"]["FINDING"], 10.0)  # to the last record
        assert M["actions"] == {"LOCATE": 1, "LOCK": 2}
        assert M["dialog_answers"] == {"dialog:found_it -> Keep looking": 1, "dialog:found_it -> Yes, it's here": 1}
        assert M["searches"] == {"n": 1, "no_result": 1}
        assert "abc1234" in text and "1920x1080" in text and "AMBIENT_INTENSITY" in text


def test_messages_most_frequent_first():
    with tmp() as root:
        sid = "s1"
        msgs = ["Press Done to end this search"] * 3 + ["Camera image not ready; tap again."] + ["Hold, still"] * 2
        write_events(root, [ev(sid, 0, "session_start")] +
                     [ev(sid, 10 + i, "ui_message", text=m, screen="FINDING") for i, m in enumerate(msgs)])
        md, M, _ = summary_of(root, sid)
        assert M["messages"] == {"Press Done to end this search": 3, "Camera image not ready; tap again.": 1, "Hold, still": 2}
        i = md.index("## Messages")
        rows = [r for r in md[i:] if r.startswith("| ") and not r.startswith("| Count") and not r.startswith("| ---")]
        assert rows[0].startswith("| 3 | Press Done")
        assert rows[1].startswith("| 2 | Hold, still")


def test_rejected_captures_and_lock_rates():
    with tmp() as root:
        sid = "s1"
        events = [ev(sid, 0, "session_start")]
        events += [ev(sid, 10 + i, "capture_rejected", stage="register", reason="no_hit", value=None) for i in range(5)]
        events += [ev(sid, 20, "capture_rejected", stage="register", reason="depth_far", value=7.5),
                   ev(sid, 21, "capture_rejected", stage="register", reason="depth_far", value=9.5),
                   ev(sid, 22, "capture_rejected", stage="verify", reason="not_in_view", value=0.95)]
        events += [ev(sid, 30 + i, "register", record=f"r{i}") for i in range(2)]
        # Lock: 3 presses for the first spot lock (2 timeouts), 1 for the second, 1 cancelled.
        events += [ev(sid, 40, "lock_attempt", spot="f1", kind="use", ok=False, reason="timeout_rolled", samples=3,
                      durationMs=3001, attemptInBar=1),
                   ev(sid, 41, "lock_attempt", spot="f1", kind="use", ok=False, reason="timeout_tracking", samples=0,
                      durationMs=3001, attemptInBar=2),
                   ev(sid, 42, "lock_attempt", spot="f1", kind="use", ok=True, reason=None, samples=10, posSpreadM=0.004,
                      durationMs=400, attemptInBar=3),
                   ev(sid, 43, "lock_attempt", spot=None, kind="add", ok=True, reason=None, samples=10, posSpreadM=0.002,
                      durationMs=350, attemptInBar=1),
                   ev(sid, 44, "lock_attempt", spot="f1", kind="use", ok=False, reason="cancelled", samples=2,
                      durationMs=150, attemptInBar=1)]
        write_events(root, events)
        md, M, F = summary_of(root, sid)
        assert M["rejected"] == {"register/no_hit": 5, "register/depth_far": 2, "verify/not_in_view": 1}
        assert M["saved"] == 2 and M["refused_register"] == 7
        assert M["locks"]["attempts"] == 5 and M["locks"]["ok"] == 2
        assert close(M["locks"]["success_rate"], 0.4)
        assert close(M["locks"]["presses_per_lock"], 2.5)
        assert M["locks"]["reasons"] == {"timeout_rolled": 1, "timeout_tracking": 1, "cancelled": 1}
        text = "\n".join(md)
        assert "| register | depth_far | 2 | 8.50 |" in text  # median value
        # 7 refused for 2 saved is 3.5 per item (over 3), and 40% lock success (under 50%).
        assert any("refused registration taps" in f for f in F)
        assert any("Lock presses succeeded" in f for f in F)


def test_tracking_losses_time_and_reasons():
    with tmp() as root:
        sid = "s1"
        write_events(root, [ev(sid, 0, "session_start")])
        rows = [pose_row(i) for i in range(100)]
        rows += [pose_row(i, tracking=False, failure="INSUFFICIENT_FEATURES") for i in range(100, 110)]
        rows += [pose_row(i) for i in range(110, 150)]
        rows += [pose_row(i, tracking=False, failure="EXCESSIVE_MOTION") for i in range(150, 155)]
        rows += [pose_row(i) for i in range(155, 200)]
        # A pause of 60 s, then 1 s more: the gap is not traced time.
        rows += [pose_row(i) for i in range(800, 810)]
        write_csv(root, sid, "pose", rows)
        _, M, F = summary_of(root, sid)
        tr = M["tracking"]
        assert tr["losses"] == 2
        assert close(tr["lost_s"], 1.5)
        assert close(tr["traced_s"], 19.9 + 0.9, 1e-9)  # 200 rows (19.9 s) and 10 rows (0.9 s)
        assert close(tr["by_reason_s"]["INSUFFICIENT_FEATURES"], 1.0)
        assert close(tr["by_reason_s"]["EXCESSIVE_MOTION"], 0.5)
        assert close(tr["lost_share"], 1.5 / 20.8)
        assert any("Tracking lost" in f for f in F)  # 7.2% > 5%


def test_motion_walked_speed_and_classes():
    with tmp() as root:
        sid = "s1"
        write_events(root, [ev(sid, 0, "session_start")])
        rows = [pose_row(i, x=0.0, speed=0.0) for i in range(50)]                     # 5 s still
        rows += [pose_row(50 + i, x=0.05 * (i + 1), speed=0.5) for i in range(100)]   # 10 s at 0.5 m/s: 5 m
        rows += [pose_row(150, x=5.0 + 3.0, speed=30.0)]                             # a 3 m jump: not walked
        rows += [pose_row(151 + i, x=8.0, speed=0.1) for i in range(49)]              # 4.9 s slow
        write_csv(root, sid, "pose", rows)
        _, M, _ = summary_of(root, sid)
        m = M["motion"]
        assert close(m["walked_m"], 5.0, 1e-6)
        assert close(m["speed_p10"], 0.0)
        assert close(m["speed_p50"], 0.5)  # 100 of the 200 rows walk
        assert close(m["speed_max"], 30.0)
        # Shares of the 19.9 s between the first and the last row (the last row has no interval).
        sh = m["share"]
        assert close(sh["still"], 5.0 / 19.9, 1e-9)
        assert close(sh["walking"], 10.1 / 19.9, 1e-9)  # the jump row counts at its own speed
        assert close(sh["slow"], 4.8 / 19.9, 1e-9)


def test_depth_by_motion_and_turn_rate_with_missing_images():
    with tmp() as root:
        sid = "s1"
        write_events(root, [ev(sid, 0, "session_start")])
        rows = [depth_row(i, usable=0.9, speed=0.0, turn=5.0) for i in range(10)]
        rows += [depth_row(10 + i, usable=0.3, speed=0.6, turn=40.0) for i in range(10)]
        rows += [depth_row(20 + i, ok=False, speed=0.6, turn=80.0) for i in range(5)]
        write_csv(root, sid, "depth", rows)
        write_csv(root, sid, "pose", [pose_row(i) for i in range(10)])
        _, M, F = summary_of(root, sid)
        d = M["depth"]
        assert d["samples"] == 25
        assert close(d["no_depth_share"], 5 / 25)
        assert close(d["usable_share"], 0.6)
        assert close(d["usable_by_motion"]["still"], 0.9)
        assert close(d["usable_by_motion"]["walking"], 0.3)
        assert "slow" not in d["usable_by_motion"]
        assert close(d["usable_by_turn"]["0-10 deg/s"], 0.9)
        assert close(d["usable_by_turn"]["30-60 deg/s"], 0.3)
        assert close(d["no_depth_by_motion"]["walking"], 5 / 15)
        assert any("Usable depth while walking" in f for f in F)
        assert not any("No depth image in" in f for f in F)  # 20% is not over 20%


def test_scene_and_light_against_verifications():
    with tmp() as root:
        sid = "s1"
        events = [ev(sid, 0, "session_start")]
        # Lux 50 until 10 s, 400 after; five verifications, the bright ones score higher.
        sims = [(2000, 0.60), (4000, 0.62), (12000, 0.80), (14000, 0.82), (16000, 0.85)]
        events += [ev(sid, t, "verify", query="q1", attempt=i + 1, bestSimilarity=s, outcome="X") for i, (t, s) in enumerate(sims)]
        # A verification long after the last lux sample gets no lux.
        events += [ev(sid, 200000, "verify", query="q1", attempt=9, bestSimilarity=0.5, outcome="X")]
        write_events(root, events)
        write_csv(root, sid, "sensors", [
            {"t": T0 + 0, "ns": NS0, "type": "light", "x": 50, "accuracy": 3},
            {"t": T0 + 10000, "ns": NS0 + 10_000_000_000, "type": "light", "x": 400, "accuracy": 3},
            {"t": T0 + 5, "ns": NS0 + 5_000_000, "type": "acc", "x": 0.1, "y": 9.8, "z": 0.2, "accuracy": 3},
        ])
        write_csv(root, sid, "depth", [depth_row(i, points=p, area=a) for i, (p, a) in enumerate([(10, 0.0), (100, 1.0), (200, 2.0)])])
        _, M, _ = summary_of(root, sid)
        assert close(M["scene"]["points_p50"], 100.0)
        assert close(M["scene"]["area_p50"], 1.0)
        assert close(M["light"]["lux_p50"], 225.0)
        v = M["light"]["verify"]
        assert [r["lux"] for r in v] == [50.0, 50.0, 400.0, 400.0, 400.0, None]
        # Ranks: lux ties (1.5, 1.5, 4, 4, 4) against similarities in increasing order.
        assert M["light"]["spearman"] > 0.8


def test_performance_battery_thermal_and_dropped():
    with tmp() as root:
        sid = "s1"
        events = [ev(sid, 0, "session_start")]
        fps = [30] * 8 + [15, 12] + [29] * 10  # 20 windows: the 5th percentile is 14.85
        events += [ev(sid, 10000 * (i + 1), "perf", frames=f * 10, fps=f, frameMsMean=12.0, frameMsP95=10.0 + i,
                      frameMsMax=40.0, mlBusyShare=0.1, traceQueue=0, traceDropped=0, traceMsMean=0.02) for i, f in enumerate(fps)]
        # 80% -> 75% in 30 min not charging: 10 %/h. The charging stretch after it does not count.
        events += [ev(sid, 0, "device_state", batteryPct=80, charging=False, thermalStatus=0, batteryTempC=30),
                   ev(sid, 1_800_000, "device_state", batteryPct=75, charging=False, thermalStatus=2, batteryTempC=38),
                   ev(sid, 3_600_000, "device_state", batteryPct=90, charging=True, thermalStatus=1, batteryTempC=36),
                   ev(sid, 3_600_001, "app_pause", sessionMs=1, traceBytes=100, traceDropped=12)]
        events += [ev(sid, 50, "register", latency={"convertMs": 30, "detectMs": 80, "embedMs": 20}),
                   ev(sid, 60, "verify", query="q", bestSimilarity=0.8, latency={"totalMs": 400, "embedMs": 50}),
                   ev(sid, 70, "model_loaded", model="mobilenet", loadMs=900)]
        write_events(root, events)
        write_csv(root, sid, "pose", [pose_row(i) for i in range(20)])
        md, M, F = summary_of(root, sid)
        p = M["perf"]
        assert p["windows"] == 20
        assert close(p["fps_p50"], 29.0)
        assert close(p["fps_p5"], 14.85)
        assert close(p["frame_ms_p95_max"], 29.0)
        assert close(p["frame_ms_p95_p50"], 19.5)
        assert close(p["battery_drain_pct_h"], 10.0)
        assert p["thermal_max"] == 2
        assert p["trace_dropped"] == 12
        text = "\n".join(md)
        assert "| register | detectMs | 1 | 80 |" in text
        assert "| model_loaded | mobilenet | 1 | 900 |" in text
        assert any("fps 5th percentile" in f for f in F)
        assert any("Thermal status reached 2" in f for f in F)
        assert any("12 trace lines dropped" in f for f in F)


def test_scan_section_only_when_a_scan_ran():
    with tmp() as root:
        sid = "s1"
        events = [ev(sid, 0, "session_start"),
                  ev(sid, 1, "scan_start", scan="sc"),
                  ev(sid, 2, "scan_keyframe", scan="sc", detections=5, kept=3, dropped={"person": 1, "large": 1},
                     latency={"detectMs": 70}),
                  ev(sid, 3, "scan_keyframe", scan="sc", detections=4, kept=4, dropped={}, latency={"detectMs": 90}),
                  ev(sid, 4, "scan_depth_check", scan="sc", distanceM=0.02),
                  ev(sid, 5, "scan_depth_check", scan="sc", distanceM=0.04),
                  ev(sid, 6, "scan_end", scan="sc", reason="stop", durationMs=30000, metersWalked=4.0, keyframes=2,
                     skippedNoDepth=3)]
        write_events(root, events)
        md, M, _ = summary_of(root, sid)
        s = M["scan"]
        assert s["keyframes"] == 2 and s["detections"] == 9 and s["kept"] == 7
        assert s["dropped"] == {"person": 1, "large": 1}
        assert close(s["depth_check_p50_m"], 0.03)
        assert s["skipped_no_depth"] == 3
        assert "## Scan" in md
    with tmp() as root:
        write_events(root, [ev("s2", 0, "session_start")])
        md, M, _ = summary_of(root, "s2")
        assert M["scan"] is None and "## Scan" not in md


def test_thresholds_are_strict_bounds():
    base = {"tracking": {"lost_share": T.TRACKING_LOST_SHARE_MAX}, "depth": {"usable_by_motion": {"walking": T.DEPTH_USABLE_WALKING_MIN},
            "no_depth_share": T.NO_DEPTH_SHARE_MAX},
            "perf": {"fps_p5": T.FPS_P5_MIN, "thermal_max": T.THERMAL_STATUS_MODERATE - 1, "trace_dropped": 0,
                     "frame_ms_p95_p50": T.FRAME_MS_P95_MAX, "trace_errors": []},
            "saved": 2, "refused_register": int(2 * T.REFUSED_PER_SAVED_MAX), "locks": {"success_rate": T.LOCK_SUCCESS_MIN},
            "scan": {"depth_check_p50_m": T.DEPTH_CHECK_MAX_M}}
    assert T.findings(base) == []  # every value exactly at its bound: nothing flagged
    over = json.loads(json.dumps(base))
    over["tracking"]["lost_share"] += 0.001
    over["depth"]["usable_by_motion"]["walking"] -= 0.001
    over["depth"]["no_depth_share"] += 0.001
    over["perf"]["fps_p5"] -= 0.1
    over["perf"]["thermal_max"] = T.THERMAL_STATUS_MODERATE
    over["perf"]["trace_dropped"] = 1
    over["perf"]["frame_ms_p95_p50"] += 0.1
    over["refused_register"] += 1
    over["locks"]["success_rate"] -= 0.01
    over["scan"]["depth_check_p50_m"] += 0.001
    F = T.findings(over)
    assert len(F) == 10, F
    assert T.findings({}) == []  # nothing known, nothing flagged
    assert T.findings({"saved": 0, "refused_register": 4}) == ["4 refused registration taps and no item saved."]


# ----------------------------------------------------------------------------- whole runs

def test_session_without_a_trace():
    with tmp() as root, tmp() as out:
        write_events(root, [ev("old", 0, "session_start", device="Pixel 8"),
                            ev("old", 100, "register", latency={"convertMs": 1, "detectMs": 2, "embedMs": 3}),
                            ev("old", 200, "tracking_lost", reason="EXCESSIVE_MOTION")])
        sessions, _ = T.load(root)
        assert not sessions["old"].has_trace
        md, M, F = T.summarize(sessions["old"])
        assert M["tracking"] is None and M["motion"] is None and M["depth"] is None
        assert F == []
        assert any("no trace folder" in line for line in md)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert T.main([root, "--out", out]) == 0
        assert "No session with a trace" in buf.getvalue()
        assert os.listdir(out) == []
        with contextlib.redirect_stdout(io.StringIO()):
            assert T.main([root, "--session", "old", "--out", out]) == 0
        assert sorted(os.listdir(out)) == ["trace_summary_old.md"]


def test_main_writes_markdown_and_plots_for_traced_sessions_only():
    with tmp() as root, tmp() as out:
        write_events(root, [ev("a", 0, "session_start"), ev("b", 0, "session_start")])
        write_csv(root, "a", "pose", [pose_row(i, x=0.01 * i, speed=0.1) for i in range(30)] +
                  [pose_row(30, tracking=False, failure="INSUFFICIENT_LIGHT")])
        write_csv(root, "a", "depth", [depth_row(i) for i in range(5)] + [depth_row(5, ok=False)])
        write_csv(root, "a", "sensors", [])
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            assert T.main([root, "--out", out]) == 0
        assert "1 sessions without a trace" in buf.getvalue()
        assert sorted(os.listdir(out)) == ["trace_a_timeline.png", "trace_a_trajectory.png", "trace_summary_a.md"]
        md = open(os.path.join(out, "trace_summary_a.md"), encoding="utf-8").read()
        for title in ("Session", "Messages", "Rejected captures and locks", "Tracking", "Motion", "Depth", "Scene",
                      "Light", "Performance", "Findings", "Plots"):
            assert f"## {title}" in md, title
        assert "guidance.csv" in md  # a missing file is listed with 0 rows
        with contextlib.redirect_stdout(io.StringIO()):
            assert T.main([root, "--out", out, "--session", "nope"]) == 1


def test_a_cut_off_last_line_is_dropped():
    with tmp() as root:
        write_events(root, [ev("s", 0, "session_start")])
        write_csv(root, "s", "pose", [pose_row(i) for i in range(10)], raw_tail="1800000001000,5000001000000000,TRACK")
        sessions, _ = T.load(root)
        pose = sessions["s"].table("pose")
        assert len(pose) == 10
        assert set(pose["tracking"]) == {"TRACKING"}
        # Empty fields are missing values, not zeros.
        assert math.isnan(pose["metersSinceFix"].iloc[0])
        assert pose["failure"].iloc[0] is None


def v07_events(session, t):
    """One of every new v0.7 event, as the app logs them, starting at epoch ms t."""
    rows = [
        ("ar_config", dict(cameraConfig={"cpuImage": "1920x1080", "texture": "1920x1080", "fpsMin": 30, "fpsMax": 30},
                           depthMode="AUTOMATIC", depthSupported=True, lightEstimationMode="AMBIENT_INTENSITY", traceOn=True)),
        ("app_resume", dict(sessionMs=120)),
        ("device_state", dict(trigger="start", sessionMs=130, batteryPct=80.0, batteryTempC=31.5, charging=False,
                              thermalStatus=0, availMemMb=3000, lowMemory=False, freeStorageMb=50000, traceBytes=0, traceDropped=0)),
        ("ui_screen", {"from": None, "to": "READY", "msInPrevious": None}),
        ("ui_action", dict(action="ITEMS", screen="READY", detail=None)),
        ("ui_action", dict(action="dialog:item_sheet", screen="READY", detail="Find")),
        ("ui_search", dict(chars=3, results=1, total=2, roomFilter=None)),
        ("ui_message", dict(text="Press Done to end this search", screen="FINDING")),
        ("capture_rejected", dict(stage="verify", reason="not_in_view", value=0.95, query="qA")),
        ("lock_attempt", dict(spot="f1", kind="use", ok=True, reason=None, samples=10, posSpreadM=0.003, yawSpreadDeg=0.4,
                              durationMs=420, attemptInBar=1, rejectedRolled=0, rejectedTracking=0)),
        ("perf", dict(frames=300, fps=30.0, frameMsMean=11.0, frameMsP95=18.0, frameMsMax=40.0, mlBusyShare=0.0,
                      traceQueue=3, traceDropped=0, traceMsMean=0.02, traceMsMax=0.4, windowS=10.0)),
        ("trace_error", dict(site="sensor:baro", error="not available on this device")),
        ("app_pause", dict(sessionMs=9000, traceBytes=12345, traceDropped=0)),
    ]
    return [dict({"t": t + i, "session": session, "type": typ}, **f) for i, (typ, f) in enumerate(rows)]


def _scrub(x):
    """For comparing loader output: drop the loader's bookkeeping (_seq, _folder, ..., and a
    capture's `seq`, its position in the merged log, which more events of course shift)."""
    if isinstance(x, dict):
        return {k: _scrub(v) for k, v in x.items() if not str(k).startswith("_") and k != "seq"}
    if isinstance(x, (list, tuple)):
        return [_scrub(v) for v in x]
    return x


def test_fb_load_and_evaluate_ignore_the_new_events():
    """The existing pipeline gives the same queries and the same report with the v0.7 events in the log."""
    import test_analysis as TA  # the hand-built E3 log of the existing tests

    with tmp() as a, tmp() as b:
        TA.handbuilt_e3(a)
        TA.handbuilt_e3(b)
        with open(os.path.join(b, "events.jsonl"), encoding="utf-8") as f:
            base = [json.loads(line) for line in f]
        mixed = []
        for e in base:
            mixed.append(e)
            if e["type"] == "session_start":
                # session_start gains its v0.7 fields; every new event follows it.
                e.update(appVersion="0.7", appCommit="abc1234", arcoreVersion="1.48.0", displayPx="1080x2400", traceOn=True)
                mixed += v07_events(e["session"], e["t"] + 1)
            elif e["type"] == "query_start":
                mixed += [dict(t=e["t"] + 1, session=e["session"], type="capture_rejected", stage="verify", reason="behind",
                               value=None, query=e["query"])]
        with open(os.path.join(b, "events.jsonl"), "w", encoding="utf-8") as f:
            for e in mixed:
                f.write(json.dumps(e) + "\n")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            da, db = L.Dataset([a]), L.Dataset([b])
            qa = L.build_queries(da.tagged())
            qb = L.build_queries(db.tagged())
        assert db.folders[0].bad_lines == 0 and db.duplicates == 0
        assert len(db.events) > len(da.events)
        assert json.dumps(_scrub(qa), sort_keys=True, default=str) == json.dumps(_scrub(qb), sort_keys=True, default=str)
        assert _scrub(L.registrations_from(da.tagged())) == _scrub(L.registrations_from(db.tagged()))
        outs = []
        for d in (a, b):
            r = subprocess.run([sys.executable, os.path.join(HERE, "evaluate.py"), d, "--exclude-session", "u0"],
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            assert r.stderr.strip() == "", r.stderr
            outs.append(r.stdout.replace(d, "<dir>"))
        assert outs[0] == outs[1]
        # And the trace summary reads the same log (no trace folders: nothing is reported by default).
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            assert T.main([b, "--no-plots", "--out", os.path.join(b, "out")]) == 0
        assert "No session with a trace" in buf.getvalue()


# ----------------------------------------------------------------------------- runner

def _tests():
    return [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]


if __name__ == "__main__":
    only = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else None
    failed = 0
    run = 0
    for name, fn in _tests():
        if only and only not in name:
            continue
        run += 1
        try:
            fn()
            print("ok  ", name)
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print("FAIL", name)
            traceback.print_exc()
    print(f"\n{run} tests, {failed} failed")
    sys.exit(1 if failed else 0)
