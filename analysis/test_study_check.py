#!/usr/bin/env python3
"""Tests of the post-run check of the v0.8 smoke test (analysis/study_check.py). The tests run on
synthetic pulls built here: one that passes, and several that are each broken in one way. No phone
data and no real image.

    python3 analysis/test_study_check.py
    analysis/.venv-bench/bin/python analysis/test_study_check.py

The passing pull follows FindBack/docs/PLAN_v0.8_study_logging.md. It is made by running the
replay's own rules (scan_replay.Replay) and the check's view-retention replay over hand-placed
detections. Decisions, observation lines, keyframe JSON, events, views, history, `confirmed` and
the inventory log therefore agree by construction. The retention rule itself is tested separately,
on hand-worked cases. Each broken pull changes one thing and must fail the check named in its test.
"""
import base64
import io
import json
import math
import os
import shutil
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scan_replay as sr  # noqa: E402
import study_check as sc  # noqa: E402

W, H, ROT = 1920, 1080, 90
DIM = 8
MODEL = "embedder_a"

# Objects (map frame, metres) and their embedding axes. The cup is moved 1 m between the passes;
# the two bottles are a same-model pair 0.4 m apart (same embedding axis).
OBJECTS = {
    "cup": {"label": "cup", "p1": (0.0, 0.0, 0.0), "p2": (0.0, 0.0, 1.0), "axis": 0, "box": (800, 400, 900, 520), "score": 0.6},
    "bowl": {"label": "bowl", "p1": (0.6, 0.0, 0.0), "p2": (0.6, 0.0, 0.0), "axis": 1, "box": (300, 300, 460, 400), "score": 0.5},
    "bottleA": {"label": "bottle", "p1": (1.2, 0.0, 0.0), "p2": (1.2, 0.0, 0.0), "axis": 2, "box": (1200, 200, 1260, 420), "score": 0.55},
    "bottleB": {"label": "bottle", "p1": (1.6, 0.0, 0.0), "p2": (1.6, 0.0, 0.0), "axis": 2, "box": (1500, 200, 1560, 420), "score": 0.45},
}
JITTER = [(0.0, 0.0, 0.0), (0.01, 0.0, -0.01), (-0.01, 0.005, 0.01), (0.005, -0.005, 0.0), (0.0, 0.01, 0.005)]
# One world frame per app session: T_WM differs (map +X turned by yawDeg, then translated).
M2W = {"s1": {"x": 0.5, "y": -1.0, "z": 2.0, "yawDeg": 30.0}, "s2": {"x": -0.3, "y": -1.1, "z": 1.0, "yawDeg": -75.0}}


def emb(axis, k):
    v = [0.02 * ((i * 7 + k * 3) % 5) for i in range(DIM)]
    v[axis] = 1.0
    return v


def b64(v):
    return base64.b64encode(struct.pack("<%df" % len(v), *v)).decode()


def jpeg_bytes(w, h):
    """A minimal JPEG header (SOI, SOF0, EOI): enough for study_check.jpeg_size."""
    sof = struct.pack(">BBHBHHB", 0xFF, 0xC0, 17, 8, h, w, 3) + bytes([1, 0x11, 0, 2, 0x11, 1, 3, 0x11, 1])
    return b"\xff\xd8" + sof + b"\xff\xd9"


def camera_map(pass_no, k):
    """Camera position in the map frame: on a circle of 1.5 m around the scene, 25 degrees apart."""
    a = math.radians(25.0 * k + (10.0 if pass_no == 2 else 0.0))
    return (0.8 + 1.5 * math.sin(a), 1.4, 1.5 * math.cos(a))


def q_yaw(deg):
    h = math.radians(deg) / 2
    return [0.0, math.sin(h), 0.0, math.cos(h)]


def scene(pass_no, stray):
    """[(kf, t, [(name, label, map point, score, box, embedding)])]: pass 1 has 3 keyframes that see
    every object (and with `stray` a 4th that sees only a cup-like detection 0.3 m from the cup);
    pass 2 has 5, so every view group gets a fifth offer."""
    t0 = 1_000_000 * pass_no
    out = []
    for k in range(3 if pass_no == 1 else 5):
        dets = []
        for name, o in OBJECTS.items():
            p, j = o["p%d" % pass_no], JITTER[k]
            dets.append((name, o["label"], (p[0] + j[0], p[1] + j[1], p[2] + j[2]), o["score"],
                         tuple(b + 4 * k for b in o["box"]), emb(o["axis"], k)))
        out.append((k, t0 + 500 * k, dets))
    if stray and pass_no == 1:
        p = OBJECTS["cup"]["p1"]
        out.append((3, t0 + 1500, [("stray", "cup", (p[0] + 0.3, p[1], p[2]), 0.4, (100, 100, 160, 180), emb(0, 9))]))
    return out


def app_entry(e, map_id, created_scan, confirmed=None, history=None):
    d = {"id": e.id, "name": e.label, "label": e.label, "labelVotes": dict(e.votes), "mapId": map_id,
         "mapPosition": list(e.position), "viewPosition": None, "roomId": None, "container": None, "sizeM": e.size_m,
         "bestQuality": e.best_quality, "embedding": b64(e.embedding),
         "sightings": [{"t": s[0], "p": list(s[1]), "sizeM": s[2], "label": s[3], "score": s[4], "kf": s[5]} for s in e.sightings],
         "sightingCount": e.count, "firstSeenMs": e.first_ms, "lastSeenMs": e.last_ms, "createdScan": created_scan}
    if confirmed is not None:
        d["confirmed"] = json.loads(json.dumps(confirmed))
    if history:
        d["history"] = json.loads(json.dumps(history))
    return d


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build(folder, rules=sr.APP_RULES, stray=True, same_session=False, study_off=False, no_end=False,
          not_located=False, other_map=False):
    """Write a consistent two-pass pull into `folder`. Returns what was written."""
    os.makedirs(folder, exist_ok=True)
    events, inv_log = [], []
    scans = ["scanA", "scanB"]
    sessions = ["s1", "s1" if same_session else "s2"]
    maps = ["m1", "m2" if other_map else "m1"]
    rp = sr.Replay(sr.sim_embedding, rules=rules)
    hist, confirmed, created_scan = {}, {}, {}
    meta = {"scans": scans, "views": {}, "obs": {}, "evicted": [], "not_complementary": []}
    n_new = 0
    for pi, scan in enumerate(scans):
        pass_no, sess = pi + 1, sessions[pi]
        m2w = M2W[sess]
        t_start = 1_000_000 * pass_no - 100
        events.append({"t": t_start - 50, "session": sess, "type": "session_start", "appVersion": "0.8"})
        events.append({"t": t_start, "session": sess, "type": "scan_start", "scan": scan, "mapId": maps[pi],
                       "studyLogging": not (study_off and pi == 1), "fixPoint": None if (not_located and pi == 1) else "fp1",
                       "saveFrames": True, "embeddingModel": MODEL, "entries": len(rp.real),
                       "study": {"schema": 1, "groupRadiusM": 0.30, "viewsPerGroup": 4, "viewsPerScan": 400,
                                 "spreadBinDeg": 10.0, "obsFile": "obs/%s.jsonl" % scan}})
        lines, linked, created = [], {}, []
        retention = sc.RetentionReplay()
        for kf, t, raw in scene(pass_no, stray):
            cam_m = camera_map(pass_no, kf)
            cam_w = sr.world_from_map(cam_m, m2w)
            dets = [sr.Det("%s-%d-%d" % (scan, kf, j), kf, t, label, score, box, p, 0.1, embedding=e, order=j)
                    for j, (_, label, p, score, box, e) in enumerate(raw)]
            used, results, cands = set(), {}, {}
            for d in sr.processing_order(dets, rules.reverse_ties):
                dec = rp.decide(d, t, used)
                cands[d.key] = sr.nearest_candidates(rp, d, used)
                if dec.kind == sr.KIND_NEW:
                    n_new += 1
                    eid = "e%02d" % n_new
                else:
                    eid = dec.entry
                rp.apply(d, t, dec.kind, eid, eid)
                used.add(eid)
                results[d.key] = (dec, eid)
                linked.setdefault(eid, []).append(d.key)        # processing order
                if dec.kind == sr.KIND_PROMOTED:
                    inv_log.append({"t": t + 1, "session": sess, "op": "add", "entry": app_entry(rp.entries[eid], maps[pi], scan)})
                    created_scan[eid] = scan
                    created.append(eid)
            kfjson = {"scan": scan, "index": kf, "t": t, "session": sess, "image": None, "imageSkipped": None,
                      "personDetections": 0, "rotation": ROT, "cameraPose": {"t": list(cam_w), "q": q_yaw(m2w["yawDeg"])},
                      "mapToWorld": m2w, "detections": [{"box": [0, 0, 50, 50], "label": "tv", "score": 0.9, "kept": False,
                                                         "reason": "stop_list"}]}
            stats = {"obs": 0, "viewsWritten": 0, "viewsEvicted": 0, "viewsSkipped": 0, "lineErrors": 0}
            for d in dets:
                dec, eid = results[d.key]
                kfjson["detections"].append({"box": list(d.box), "label": d.label, "score": d.score, "kept": True,
                                             "reason": None, "depthM": 1.0, "sizeM": d.size_m, "mapPoint": list(d.map_point),
                                             "kind": dec.kind, "entry": eid})
                rect = sc.view_rect_of(d.box, W, H)
                line = {"v": 1, "type": "obs", "obs": d.key, "kind": "scan", "scan": scan, "session": sess,
                        "mapId": maps[pi], "room": None, "kf": kf, "det": d.order, "rawDet": d.order + 1, "t": t,
                        "studyTag": "exp=smoke", "label": d.label, "score": d.score, "box": list(d.box), "depthM": 1.0,
                        "sizeM": d.size_m, "distanceM": sr.distance(d.map_point, cam_m), "mapPoint": list(d.map_point),
                        "worldPoint": list(sr.world_from_map(d.map_point, m2w)), "quality": d.quality,
                        "image": {"w": W, "h": H}, "rotation": ROT,
                        "camera": {"world": {"t": list(cam_w), "q": q_yaw(m2w["yawDeg"])},
                                   "map": {"t": list(cam_m), "q": [0.0, 0.0, 0.0, 1.0], "forward": [0.0, 0.0, -1.0],
                                           "yawDeg": 0.0, "pitchDeg": 0.0}},
                        "mapToWorld": m2w, "intrinsics": {"fx": 1400.0, "fy": 1400.0, "cx": 960.0, "cy": 540.0, "width": W, "height": H},
                        "emb": b64(d.embedding), "embDim": DIM, "embModel": MODEL, "personInFrame": False, "personOverlap": 0.0,
                        "viewRect": list(rect), "viewReason": None, "view": None, "viewPx": None, "viewReplaces": None,
                        "decision": {"matcher": "baseline", "kind": dec.kind, "entry": eid, "sim": dec.sim, "distM": dec.dist,
                                     "byLabel": dec.by_label, "candidates": cands[d.key]}}
                group, reason, evict = retention.feed(dict(line, viewReason="pending"))
                line["viewGroup"], line["viewReason"], line["viewReplaces"] = group, reason, evict
                stats["obs"] += 1
                if reason == "kept":
                    line["view"] = "views/%s/%s.jpg" % (scan, d.key)
                    line["viewPx"] = list(sc.upright_crop_size(rect, W, H, ROT))
                    os.makedirs(os.path.join(folder, "views", scan), exist_ok=True)
                    with open(os.path.join(folder, line["view"]), "wb") as f:
                        f.write(jpeg_bytes(*line["viewPx"]))
                    meta["views"][d.key] = line["view"]
                    stats["viewsWritten"] += 1
                else:
                    stats["viewsSkipped"] += 1
                    meta["not_complementary"].append(d.key)
                lines.append(line)
                meta["obs"][d.key] = line
                if evict:
                    old = meta["obs"][evict]["view"]
                    os.remove(os.path.join(folder, old))
                    lines.append({"v": 1, "type": "view_evicted", "scan": scan, "obs": evict, "view": old, "by": d.key,
                                  "viewGroup": group, "t": t + 1})
                    meta["evicted"].append(evict)
                    stats["viewsEvicted"] += 1
            os.makedirs(os.path.join(folder, "scan", scan), exist_ok=True)
            with open(os.path.join(folder, "scan", scan, "kf%04d.json" % kf), "w") as f:
                json.dump(kfjson, f)
            order = sr.processing_order(dets, rules.reverse_ties)
            kinds = [results[d.key][0].kind for d in dets]
            events.append({"t": t + 2, "session": sess, "type": "scan_keyframe", "scan": scan, "index": kf, "kept": len(dets),
                           "matched": kinds.count("known"), "new": kinds.count("new"), "tentative": kinds.count("tentative"),
                           "promoted": kinds.count("promoted"), "study": stats,
                           "results": [{"label": d.label, "score": d.score, "kind": results[d.key][0].kind,
                                        "entry": results[d.key][1], "sim": results[d.key][0].sim,
                                        "distM": results[d.key][0].dist} for d in order]})
        dropped = rp.finish()
        t_end = 1_000_000 * pass_no + 5000
        retained = sorted(retention.retained_ids())
        for eid, e in rp.entries.items():
            ls = [meta["obs"][k] for k in linked.get(eid, [])]
            n = len(ls)
            item = {"scan": scan, "tMs": t_end, "outcome": "seen" if n else "missed", "eligible": True,
                    "position": [sum(l["mapPoint"][i] for l in ls) / n for i in range(3)] if n else None,
                    "sightings": n, "lastObsMs": max(l["t"] for l in ls) if n else None, "obs": [l["obs"] for l in ls],
                    "link": "created" if created_scan.get(eid) == scan else "baseline_unconfirmed"}
            hist.setdefault(eid, []).append(item)
            events.append({"t": t_end - 20, "session": sess, "type": "inv_evidence", "scan": scan, "inventoryId": eid,
                           "outcome": item["outcome"]})
            if created_scan.get(eid) == scan:
                confirmed[eid] = {"mapPosition": list(e.position), "viewPosition": None, "roomId": None,
                                  "tMs": item["lastObsMs"], "scan": scan, "source": "created",
                                  "refs": [x for x in item["obs"] if x in retained], "query": None, "similarity": None}
            full = app_entry(e, maps[pi], created_scan.get(eid), confirmed.get(eid), hist[eid])
            if n:
                inv_log.append({"t": t_end - 10, "session": sess, "op": "update", "entry": full})
            else:
                inv_log.append({"t": t_end - 10, "session": sess, "op": "evidence", "id": eid, "historyItem": item})
        views = {"kept": len(retained), "written": sum(1 for l in lines if l["type"] == "obs" and l.get("view")),
                 "evicted": sum(1 for l in lines if l["type"] == "view_evicted"), "skipped": {}, "groups": len(retention.groups),
                 "bytes": 0}
        lines.append({"v": 1, "type": "scan_end", "scan": scan, "session": sess, "t": t_end, "reason": "stop",
                      "keyframes": len(scene(pass_no, stray)), "observations": sum(1 for l in lines if l["type"] == "obs"),
                      "views": views, "retained": retained, "created": created,
                      "tentativeDropped": [{"entry": i, "obs": linked.get(i, [])} for i in dropped], "error": None})
        os.makedirs(os.path.join(folder, "obs"), exist_ok=True)
        write_jsonl(os.path.join(folder, "obs", "%s.jsonl" % scan), lines)
        if not (no_end and pi == 1):
            events.append({"t": t_end + 1, "session": sess, "type": "scan_end", "scan": scan, "studyLogging": True,
                           "study": {"views": views}})
        meta["dropped_%d" % pass_no] = dropped
        meta["retained_%d" % pass_no] = retained
    inventory = [app_entry(e, "m1", created_scan.get(eid), confirmed.get(eid), hist.get(eid)) for eid, e in rp.entries.items()]
    with open(os.path.join(folder, "inventory.json"), "w") as f:
        json.dump({"version": 1, "entries": inventory}, f)
    write_jsonl(os.path.join(folder, "events.jsonl"), sorted(events, key=lambda e: e["t"]))
    write_jsonl(os.path.join(folder, "inventory_log.jsonl"), sorted(inv_log, key=lambda l: l["t"]))
    meta["entries"] = {e["id"]: e for e in inventory}
    meta["created"] = created_scan
    return meta


# ----------------------------------------------------------------------------- tools

def rewrite_jsonl(path, fn):
    rows = sr.read_jsonl(path)
    write_jsonl(path, [r for r in (fn(r) for r in rows) if r is not None])


def edit_json(path, fn):
    with open(path) as f:
        d = json.load(f)
    fn(d)
    with open(path, "w") as f:
        json.dump(d, f)


def run_check(folder, scans=None):
    buf = io.StringIO()
    code = sc.run(folder, scans, out=buf)
    status = {}
    for line in buf.getvalue().splitlines():
        if line.startswith(("PASS  ", "FAIL  ")):
            status[line[6:19].strip()] = line[:4]
    return code, status, buf.getvalue()


def failed(status):
    return {k for k, v in status.items() if v == "FAIL"}


class Pull:
    """A temporary passing pull, changed by each test."""

    def __init__(self, **opts):
        self.dir = tempfile.mkdtemp(prefix="studycheck_")
        self.meta = build(self.dir, **opts)

    def path(self, *p):
        return os.path.join(self.dir, *p)

    def pass1_entry(self):
        return next(i for i, s in sorted(self.meta["created"].items()) if s == "scanA" and i in self.meta["entries"])

    def matched_entry(self):
        """A pass-1 entry the baseline also linked observations to in pass 2."""
        return next(i for i, e in sorted(self.meta["entries"].items())
                    if len(e.get("history") or []) == 2 and all(h["sightings"] for h in e["history"]))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        shutil.rmtree(self.dir)


# ----------------------------------------------------------------------------- helpers

def test_jpeg_size_and_crop_size():
    d = tempfile.mkdtemp(prefix="studycheck_")
    try:
        p = os.path.join(d, "x.jpg")
        with open(p, "wb") as f:
            f.write(jpeg_bytes(123, 45))
        assert sc.jpeg_size(p) == (123, 45)
        with open(p, "wb") as f:
            f.write(b"not a jpeg")
        assert sc.jpeg_size(p) is None
        assert sc.jpeg_size(os.path.join(d, "missing.jpg")) is None
    finally:
        shutil.rmtree(d)
    # Box 100 x 120 at (800, 400): padded by 8 and 9.6 px -> 792..908 x 390.4..529.6; truncated
    # 116 x 139 (529.6 - 390.4 = 139.2); turned upright by 90 degrees -> 139 x 116.
    r = sc.view_rect_of((800, 400, 900, 520), W, H)
    assert sc.upright_crop_size(r, W, H, 90) == (139, 116)
    assert sc.upright_crop_size(r, W, H, 0) == (116, 139)
    # Clamped at the image corner: box (0, 0, 100, 50) -> (-8, -4, 108, 54) -> (0, 0, 108, 54).
    assert sc.view_rect_of((0, 0, 100, 50), W, H) == (0.0, 0.0, 108.0, 54.0)


def test_frames():
    # T_WM with yaw 90: map +X becomes world -Z, then the translation; the inverse brings it back.
    m2w = {"x": 1.0, "y": 2.0, "z": 3.0, "yawDeg": 90.0}
    w = sr.world_from_map((1.0, 0.0, 0.0), m2w)
    assert sc.near(w, (1.0, 2.0, 2.0), 1e-12)
    assert sc.near(sc.map_from_world(w, m2w), (1.0, 0.0, 0.0), 1e-12)
    # -Z turned by +90 degrees about +Y (q = (0, sin 45, 0, cos 45)) points along -X.
    assert sc.near(sc.rotate_by_quaternion(q_yaw(90.0), (0.0, 0.0, -1.0)), (-1.0, 0.0, 0.0), 1e-12)


def unit_at(deg):
    a = math.radians(deg)
    return (math.cos(a), 0.0, math.sin(a))


def view(name, deg, q=1.0):
    return {"obs": name, "dir": unit_at(deg), "quality": q}


def test_leave_out_by_spread():
    # Kept at 0, 5, 40 and 80 degrees; new at 120. Leaving out new: min angle 5 (bin 0); v0: 35
    # (5-40, bin 3); v5: 40 (bin 4); v40: 5; v80: 5. v5 leaves for the best spread.
    kept = [view("v0", 0), view("v5", 5), view("v40", 40), view("v80", 80)]
    assert sc.leave_out(kept, view("new", 120))["obs"] == "v5"
    # Kept at 0, 10, 20, 30; new at 90: every set of four has a 10-degree pair (bin 1) and equal
    # quality, so the first of the pool, the new view, is left out (not complementary).
    kept = [view("a", 0), view("b", 10), view("c", 20), view("d", 30)]
    assert sc.leave_out(kept, view("new", 90))["obs"] == "new"


def test_leave_out_quality_breaks_ties_strictly():
    # Kept at 0, 30, 60, 90 (quality 1); new at 120 with quality 5. Every set has spread 30 (bin 3).
    # Without new the quality is 4; without any kept view 8. v0 is the first with 8, and the later
    # ones (also 8) are not strictly better, so v0 is evicted.
    kept = [view("v0", 0), view("v30", 30), view("v60", 60), view("v90", 90)]
    assert sc.leave_out(kept, view("new", 120, q=5.0))["obs"] == "v0"


def test_retention_groups_and_cap():
    # Anchors never move; a point exactly 0.30 m from an anchor joins it; a farther one starts a group.
    rr = sc.RetentionReplay(per_scan=2)
    line = lambda oid, x: {"obs": oid, "mapPoint": [x, 0.0, 0.0], "quality": 1.0, "viewReason": "pending",  # noqa: E731
                           "camera": {"map": {"t": [x, 1.0, 1.0]}}}
    assert rr.feed(line("a", 0.0)) == (0, "kept", None)
    assert rr.feed(line("b", 0.3)) == (0, "kept", None)
    # The scan already retains 2 (its cap here): a new group's first view is not kept.
    assert rr.feed(line("c", 0.31)) == (1, "scan_cap", None)
    # A person-blocked line is grouped but not offered.
    assert rr.feed(dict(line("d", 0.0), viewReason="person")) == (0, None, None)


# ----------------------------------------------------------------------------- passing pull

def test_passing_pull():
    with Pull() as p:
        code, status, text = run_check(p.dir)
        assert code == 0, text
        assert set(status) == set(sc.CHECKS) and not failed(status), text
        # Pass 1: 4 objects in 3 keyframes (4 new, 4 promoted, 4 known) and the stray cup (new,
        # dropped): 13. Pass 2: 4 objects in 5 keyframes: 20 (the moved cup makes one new entry).
        assert "pass 1: n = 13: 13 reproduced, 0 need a similarity, 0 mismatch" in text, text
        assert "pass 2: n = 20: 20 reproduced, 0 need a similarity, 0 mismatch" in text, text
        # The cup moved 1 m: 3 entries are matched in both passes, each about 1 cm from itself.
        assert "n = 3 entries matched in both passes" in text, text
        assert p.meta["dropped_1"] and p.meta["evicted"] + p.meta["not_complementary"]


def test_explicit_scans_argument():
    with Pull() as p:
        code, status, _ = run_check(p.dir, "scanA,scanB")
        assert code == 0 and not failed(status)


def test_pre_v08_pull_is_named():
    d = tempfile.mkdtemp(prefix="studycheck_")
    try:
        write_jsonl(os.path.join(d, "events.jsonl"), [{"t": 1, "type": "scan_start", "scan": "old1", "mapId": "m"}])
        code, status, text = run_check(d)
        assert code == 1 and "pre-v0.8 pull" in text and failed(status) == set(sc.CHECKS), text
    finally:
        shutil.rmtree(d)


# ----------------------------------------------------------------------------- broken pulls

def test_same_session_fails_scans():
    with Pull(same_session=True) as p:
        code, status, text = run_check(p.dir)
        assert code == 1 and failed(status) == {"scans"} and "not restarted" in text, text


def test_study_logging_off_fails():
    # Pass 2 with study logging off (but obs/ written anyway): named explicitly, scans and files fail.
    with Pull(study_off=True) as p:
        code, status, text = run_check(p.dir, "scanA,scanB")
        assert code == 1 and {"scans", "files"} <= failed(status) and "study logging off" in text, text


def test_missing_scan_end_fails_scans():
    with Pull(no_end=True) as p:
        code, status, text = run_check(p.dir)
        assert code == 1 and "scans" in failed(status) and "no scan_end" in text, text


def test_not_located_fails_scans():
    with Pull(not_located=True) as p:
        code, status, text = run_check(p.dir)
        assert code == 1 and failed(status) == {"scans"} and "not located" in text, text


def test_other_map_fails():
    with Pull(other_map=True) as p:
        code, status, text = run_check(p.dir)
        assert code == 1 and "scans" in failed(status) and "different maps" in text, text


def test_truncated_line_in_the_middle_fails_files():
    with Pull() as p:
        path = p.path("obs", "scanA.jsonl")
        with open(path) as f:
            rows = f.read().splitlines()
        rows[2] = rows[2][:40]
        with open(path, "w") as f:
            f.write("\n".join(rows) + "\n")
        code, status, text = run_check(p.dir)
        assert "files" in failed(status) and "line 3 of obs/scanA.jsonl does not parse" in text, text


def test_missing_observation_line_fails():
    with Pull() as p:
        rewrite_jsonl(p.path("obs", "scanA.jsonl"), lambda r: None if r.get("obs") == "scanA-2-3" else r)
        code, status, text = run_check(p.dir)
        assert "observations" in failed(status) and "kf 2: 4 kept detections, 3 observation lines" in text, text


def test_missing_embedding_fails():
    with Pull() as p:
        rewrite_jsonl(p.path("obs", "scanB.jsonl"), lambda r: dict(r, emb=None) if r.get("obs") == "scanB-0-1" else r)
        code, status, text = run_check(p.dir)
        assert "observations" in failed(status) and "no usable embedding" in text, text


def test_bad_camera_frame_fails():
    with Pull() as p:
        def move(r):
            if r.get("obs") == "scanB-1-0":
                r["camera"]["map"]["t"] = [9.0, 9.0, 9.0]
            return r
        rewrite_jsonl(p.path("obs", "scanB.jsonl"), move)
        code, status, text = run_check(p.dir)
        assert "observations" in failed(status) and "camera.map.t is not" in text, text


def test_decision_differs_from_keyframe_json():
    # The keyframe JSON says the bowl of pass 2, keyframe 1, made a new entry; the line says matched.
    with Pull() as p:
        def change(d):
            for det in d["detections"]:
                if det.get("label") == "bowl":
                    det["kind"], det["entry"] = "new", "eXX"
        edit_json(p.path("scan", "scanB", "kf0001.json"), change)
        code, status, text = run_check(p.dir)
        assert failed(status) == {"observations"} and "keyframe JSON new eXX" in text, text


def test_missing_view_file_fails():
    with Pull() as p:
        os.remove(p.path(p.meta["views"][p.meta["retained_1"][0]]))
        code, status, text = run_check(p.dir)
        assert "views" in failed(status) and "missing or not a JPEG" in text, text


def test_wrong_view_size_fails():
    with Pull() as p:
        with open(p.path(p.meta["views"][p.meta["retained_1"][1]]), "wb") as f:
            f.write(jpeg_bytes(10, 10))
        code, status, text = run_check(p.dir)
        assert failed(status) == {"views"} and "is 10x10 px" in text, text


def test_view_overlapping_a_person_fails():
    with Pull() as p:
        oid = p.meta["retained_1"][2]
        rewrite_jsonl(p.path("obs", "scanA.jsonl"),
                      lambda r: dict(r, personInFrame=True, personOverlap=0.25) if r.get("obs") == oid else r)
        code, status, text = run_check(p.dir)
        assert "views" in failed(status) and "has a view but personOverlap 0.25" in text, text


def test_fifth_view_in_a_group_fails():
    # Undo one leave-out decision of pass 2: a view that should have been evicted (or not kept)
    # stays, so its group holds five.
    with Pull() as p:
        if p.meta["evicted"]:
            oid = p.meta["evicted"][0]
            rewrite_jsonl(p.path("obs", "scanB.jsonl"), lambda r: None if r.get("type") == "view_evicted" and r.get("obs") == oid else r)
            line = p.meta["obs"][oid]
            with open(p.path(line["view"]), "wb") as f:
                f.write(jpeg_bytes(*line["viewPx"]))
        else:
            oid = p.meta["not_complementary"][-1]
            line = p.meta["obs"][oid]
            line["view"], line["viewReason"] = "views/scanB/%s.jpg" % oid, "kept"
            line["viewPx"] = list(sc.upright_crop_size(line["viewRect"], W, H, ROT))
            with open(p.path(line["view"]), "wb") as f:
                f.write(jpeg_bytes(*line["viewPx"]))
            rewrite_jsonl(p.path("obs", "scanB.jsonl"), lambda r: line if r.get("obs") == oid else r)
        code, status, text = run_check(p.dir)
        assert "views" in failed(status) and "keeps 5 views (limit 4)" in text, text


def test_confirmed_changed_without_user_fails():
    with Pull() as p:
        eid = p.pass1_entry()
        edit_json(p.path("inventory.json"), lambda d: [e["confirmed"].update(mapPosition=[9, 9, 9])
                                                         for e in d["entries"] if e["id"] == eid])
        code, status, text = run_check(p.dir)
        assert failed(status) == {"entries"} and "without a user confirmation" in text, text


def test_confirmed_changed_on_a_later_log_line_fails():
    # Plan item 18: the pass-2 update line of a pass-1 entry carries another confirmed.
    with Pull() as p:
        eid = p.matched_entry()

        def change(r):
            if r.get("op") == "update" and r["entry"]["id"] == eid and r["t"] > 2_000_000:
                r["entry"]["confirmed"]["tMs"] = 1
            return r
        rewrite_jsonl(p.path("inventory_log.jsonl"), change)
        code, status, text = run_check(p.dir)
        assert "entries" in failed(status) and "confirmed changed on a update line" in text, text


def test_confirmed_changed_by_the_user_passes():
    with Pull() as p:
        eid = p.pass1_entry()
        new = {"mapPosition": [9, 9, 9], "viewPosition": None, "roomId": None, "tMs": 3_000_000, "scan": None,
               "source": "user_verify", "refs": ["q1234567-x1"], "query": "q1234567", "similarity": 0.8}
        entry = None

        def change(d):
            nonlocal entry
            for e in d["entries"]:
                if e["id"] == eid:
                    e["confirmed"] = new
                    entry = e
        edit_json(p.path("inventory.json"), change)
        with open(p.path("inventory_log.jsonl"), "a") as f:
            f.write(json.dumps({"t": 3_000_010, "op": "confirm", "entry": entry, "source": "user_verify"}) + "\n")
        with open(p.path("events.jsonl"), "a") as f:
            f.write(json.dumps({"t": 3_000_011, "type": "inv_confirm", "inventoryId": eid, "source": "user_verify",
                                "confirmed": new}) + "\n")
        write_jsonl(p.path("obs", "extra_views.jsonl"),
                    [{"v": 1, "type": "obs", "obs": "q1234567-x1", "kind": "extra_view", "entry": eid, "view": None,
                      "viewReason": "person", "personInFrame": True, "personOverlap": 0.4}])
        code, status, text = run_check(p.dir)
        assert code == 0 and "changed by a logged user confirmation 1" in text and "1 extra views" in text, text


def test_bad_confirmed_ref_fails():
    with Pull() as p:
        eid = p.pass1_entry()
        edit_json(p.path("inventory.json"), lambda d: [e["confirmed"]["refs"].append("nope-1-1")
                                                         for e in d["entries"] if e["id"] == eid])
        code, status, text = run_check(p.dir)
        assert "entries" in failed(status) and "nope-1-1 is no observation" in text, text


def test_missing_history_item_fails():
    with Pull() as p:
        eid = p.matched_entry()
        edit_json(p.path("inventory.json"), lambda d: [e.__setitem__("history", e["history"][:1])
                                                         for e in d["entries"] if e["id"] == eid])
        code, status, text = run_check(p.dir)
        assert {"entries", "history"} <= failed(status) and "0 history items for scan scanB" in text, text


def test_wrong_history_position_fails():
    with Pull() as p:
        eid = p.matched_entry()
        edit_json(p.path("inventory.json"), lambda d: [e["history"][-1].update(position=[5, 5, 5])
                                                         for e in d["entries"] if e["id"] == eid])
        code, status, text = run_check(p.dir)
        assert failed(status) == {"history"} and "history position" in text, text


def test_replay_mismatch_is_reported():
    # Pass 1 made with a fixed 0.5 m radius: the stray cup 0.3 m from the cup is then matched to it,
    # which the app's rule (0.20 m for 0.1 m objects) would not do. Every file agrees with that
    # decision, so only the replay fails, and it names the detection.
    with Pull(rules=sr.Rules(radius_fn=lambda a, b: 0.5)) as p:
        code, status, text = run_check(p.dir)
        assert code == 1 and failed(status) == {"replay"}, text
        assert "scanA-3-0 cup logged known" in text and "replay new" in text, text


def test_logged_similarity_must_agree():
    # The same wrong sim in the line and in scan_keyframe.results: only the replay can see it.
    with Pull() as p:
        def bump(r):
            if r.get("obs") == "scanB-1-0":
                r["decision"]["sim"] = 0.1
            if r.get("type") == "scan_keyframe" and r.get("scan") == "scanB" and r.get("index") == 1:
                for x in r["results"]:
                    if x["label"] == "cup":
                        x["sim"] = 0.1
            return r
        rewrite_jsonl(p.path("obs", "scanB.jsonl"), bump)
        rewrite_jsonl(p.path("events.jsonl"), bump)
        code, status, text = run_check(p.dir)
        assert failed(status) == {"replay"} and "sim logged 0.10000" in text, text


def test_logged_candidates_must_agree():
    with Pull() as p:
        def move(r):
            if r.get("obs") == "scanB-2-1" and r["decision"]["candidates"]:
                r["decision"]["candidates"][0]["distM"] += 0.01
            return r
        rewrite_jsonl(p.path("obs", "scanB.jsonl"), move)
        code, status, text = run_check(p.dir)
        assert failed(status) == {"replay"} and "scanB-2-1 candidate" in text, text


if __name__ == "__main__":
    n_failed = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        try:
            fn()
            print("ok   ", name)
        except Exception as e:   # noqa: BLE001
            n_failed += 1
            print("FAIL ", name, type(e).__name__, str(e)[:3000])
    print("%d failed" % n_failed)
    sys.exit(1 if n_failed else 0)
