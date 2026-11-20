#!/usr/bin/env python3
"""Post-run check of a two-pass smoke test of the v0.8 study build (paper/IDENTITY_PILOT_PROTOCOL.md).

    python3 analysis/study_check.py findback_data_smoke
    python3 analysis/study_check.py findback_data_smoke --scans <first scan>,<second scan>

The contract is FindBack/docs/PLAN_v0.8_study_logging.md (section 3 for the files, section 4 for view
retention, section 6 for `confirmed` and `history`, section 12 for what a post-run check can verify).
The check prints one PASS or FAIL line per check with counts, then the alignment between the two
launches (reported, not judged). It exits with 1 when any check fails, and with 2 when the folder
does not exist.

Without --scans, it takes the last two scans with study logging on. A pull with no such scan (made
before v0.8) is named as one, and every check fails for that reason.

Standard library only: JPEG sizes come from the file header, and the replay is
analysis/scan_replay.py, run with the logged embeddings.
"""
import argparse
import collections
import glob
import json
import math
import os
import statistics
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scan_replay as sr  # noqa: E402

# StudyConfig defaults, used when scan_start.study lacks a value.
GROUP_RADIUS_M = 0.30
VIEWS_PER_GROUP = 4
VIEWS_PER_SCAN = 400
SPREAD_BIN_DEG = 10.0
VERIFY_THRESHOLD = 0.75

POS_TOL_M = 1e-9              # section 12: positions 1e-9 m
PX_TOL = 1e-3                 # float32-derived pixel values
SIM_TOL = 1e-4                # a logged float32 cosine against the replay's double one
LINE_TYPES = {"obs", "view_evicted", "scan_end"}
VIEW_NOT_OFFERED = {"person", "no_crop", "write_error"}


# ----------------------------------------------------------------------------- reading

def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_jsonl_strict(path):
    """(rows, line numbers that do not parse, number of lines); rows only of parsed lines."""
    rows, bad, n = [], [], 0
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                n = i
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    bad.append(i)
    except OSError:
        return None, [], 0
    return rows, bad, n


def jpeg_size(path):
    """(width, height) from the SOF marker of a JPEG file, or None."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01, 0xFF) or 0xD0 <= marker <= 0xD7:
            i += 1 if marker == 0xFF else 2
            continue
        if marker == 0xD9:
            return None
        seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 9 > len(data):
                return None
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        i += 2 + seg_len
    return None


def view_rect_of(box, w, h):
    """viewRect = clamp(pad(box, 0.08)) in float32 (StudyImage.viewRect)."""
    f = sr.f32
    left, top, right, bottom = (f(v) for v in box)
    dx, dy = f(f(right - left) * f(0.08)), f(f(bottom - top) * f(0.08))
    left, top, right, bottom = f(left - dx), f(top - dy), f(right + dx), f(bottom + dy)
    return (min(max(left, 0.0), f(w - 1.0)), min(max(top, 0.0), f(h - 1.0)),
            min(max(right, 1.0), float(w)), min(max(bottom, 1.0), float(h)))


def upright_crop_size(rect, w, h, rotation):
    """StudyImage.uprightCropSize: clamp, truncate to whole pixels (at least 1), swap for 90/270."""
    f = sr.f32
    left = min(max(f(rect[0]), 0.0), f(w - 1.0))
    top = min(max(f(rect[1]), 0.0), f(h - 1.0))
    right = min(max(f(rect[2]), 1.0), float(w))
    bottom = min(max(f(rect[3]), 1.0), float(h))
    x, y = int(left), int(top)
    cw = max(1, min(int(f(right - left)), w - x))
    ch = max(1, min(int(f(bottom - top)), h - y))
    return (cw, ch) if int(rotation) % 180 == 0 else (ch, cw)


def map_from_world(p, m2w):
    """T_WM^-1 applied to a world point: R(-yaw)(p - t) (section 3 conventions)."""
    yaw = math.radians(m2w["yawDeg"])
    c, s = math.cos(yaw), math.sin(yaw)
    x, y, z = p[0] - m2w["x"], p[1] - m2w["y"], p[2] - m2w["z"]
    return (c * x - s * z, y, s * x + c * z)


def rotate_by_quaternion(q, v):
    """v turned by the unit quaternion q = (x, y, z, w)."""
    x, y, z, w = q
    # t = 2 q_vec x v; v' = v + w t + q_vec x t
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return (v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx))


# ----------------------------------------------------------------------------- view retention (section 4)

def _unit(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (0.0, 0.0, 0.0) if n < 1e-12 else (v[0] / n, v[1] / n, v[2] / n)


def angle_deg(a, b):
    return math.degrees(math.acos(min(1.0, max(-1.0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))))


def spread_deg(views):
    """Smallest angle between the item-to-camera directions of any two views (180 for one)."""
    best = 180.0
    for i in range(len(views)):
        for k in range(i + 1, len(views)):
            best = min(best, angle_deg(views[i]["dir"], views[k]["dir"]))
    return best


def leave_out(kept, new, bin_deg=SPREAD_BIN_DEG):
    """ViewRetention.leaveOut for a full group: the element of [new] + kept to leave out."""
    pool = [new] + list(kept)
    best, best_bin, best_q = None, None, None
    for out in pool:
        s = [v for v in pool if v is not out]
        b = int(math.floor(spread_deg(s) / bin_deg + 1e-9))
        q = 0.0
        for v in s:
            q += v["quality"]
        if best is None or b > best_bin or (b == best_bin and q > best_q):
            best, best_bin, best_q = out, b, q
    return best


class RetentionReplay:
    """Section 4 over one scan's obs lines in file order. feed() returns the expected
    (viewGroup, viewReason or None when not offered, evicted obs id or None)."""

    def __init__(self, radius=GROUP_RADIUS_M, per_group=VIEWS_PER_GROUP, per_scan=VIEWS_PER_SCAN, bin_deg=SPREAD_BIN_DEG):
        self.radius, self.per_group, self.per_scan, self.bin_deg = radius, per_group, per_scan, bin_deg
        self.anchors = []
        self.groups = []
        self.retained = 0
        self.max_group = 0
        self.max_scan = 0

    def group_of(self, p):
        best, best_d = -1, None
        for i, a in enumerate(self.anchors):
            d = sr.distance(a, p)
            if d <= self.radius and (best_d is None or d < best_d):
                best, best_d = i, d
        if best >= 0:
            return best
        self.anchors.append(tuple(p))
        self.groups.append([])
        return len(self.anchors) - 1

    def feed(self, line):
        g = self.group_of(line["mapPoint"])
        if line.get("viewReason") in VIEW_NOT_OFFERED:
            return g, None, None
        cam = ((line.get("camera") or {}).get("map") or {}).get("t") or line["mapPoint"]
        v = {"obs": line["obs"], "dir": _unit([cam[i] - line["mapPoint"][i] for i in range(3)]),
             "quality": float(line.get("quality") or 0.0)}
        views = self.groups[g]
        if len(views) < self.per_group:
            if self.retained >= self.per_scan:
                return g, "scan_cap", None
            views.append(v)
            self.retained += 1
            self._track(g)
            return g, "kept", None
        out = leave_out(views, v, self.bin_deg)
        if out is v:
            return g, "not_complementary", None
        views.remove(out)
        views.append(v)
        self._track(g)
        return g, "kept", out["obs"]

    def _track(self, g):
        self.max_group = max(self.max_group, len(self.groups[g]))
        self.max_scan = max(self.max_scan, self.retained)

    def retained_ids(self):
        return {v["obs"] for vs in self.groups for v in vs}


# ----------------------------------------------------------------------------- the pull

class Pull:
    """Everything the check reads from one pulled folder."""

    def __init__(self, folder):
        self.folder = folder
        self.events = sr.read_jsonl(os.path.join(folder, "events.jsonl"))
        inv = read_json(os.path.join(folder, "inventory.json"))
        self.inventory = ((inv or {}).get("entries") or []) if isinstance(inv, dict) else []
        self.inv_log = sr.read_jsonl(os.path.join(folder, "inventory_log.jsonl"))
        self.scan_start, self.scan_end = {}, {}
        self.keyframe_events = collections.defaultdict(dict)
        self.evidence = {}
        for e in self.events:
            t = e.get("type")
            if t == "scan_start":
                self.scan_start[e.get("scan")] = e
            elif t == "scan_end":
                self.scan_end[e.get("scan")] = e
            elif t == "scan_keyframe":
                self.keyframe_events[e.get("scan")][e.get("index")] = e
            elif t == "inv_evidence" and e.get("scan"):
                self.evidence[(e.get("scan"), e.get("inventoryId"))] = e
        self._lines = {}

    def path(self, rel):
        return os.path.join(self.folder, rel)

    def study_scans(self):
        s = [e for e in self.scan_start.values() if e.get("studyLogging") is True]
        return [e["scan"] for e in sorted(s, key=lambda e: e.get("t") or 0)]

    def raw_lines(self, scan):
        if scan not in self._lines:
            self._lines[scan] = read_jsonl_strict(self.path(os.path.join("obs", "%s.jsonl" % scan)))
        return self._lines[scan]

    def obs_lines(self, scan):
        rows = self.raw_lines(scan)[0]
        return rows or []

    def observations(self, scan):
        return [o for o in self.obs_lines(scan) if o.get("type") == "obs"]

    def end_line(self, scan):
        return next((o for o in self.obs_lines(scan) if o.get("type") == "scan_end"), None)

    def extra_lines(self):
        rows = read_jsonl_strict(self.path(os.path.join("obs", "extra_views.jsonl")))[0]
        return rows or []

    def keyframe_json(self, scan):
        out = {}
        for p in sorted(glob.glob(self.path(os.path.join("scan", scan, "kf*.json")))):
            d = read_json(p)
            if isinstance(d, dict):
                out[d.get("index")] = d
        return out

    def inventory_state(self, t_ms):
        """Entries (id -> full entry, store order) from the full-entry lines of inventory_log.jsonl
        written before t_ms (section 6 replay; `evidence` lines do not change what is used here)."""
        state = collections.OrderedDict()
        for line in self.inv_log:
            if (line.get("t") or 0) >= t_ms:
                break
            op, ent = line.get("op"), line.get("entry")
            if op == "delete" and isinstance(ent, dict):
                state.pop(ent.get("id"), None)
            elif isinstance(ent, dict) and ent.get("id"):
                if op == "add":
                    state.pop(ent["id"], None)
                state[ent["id"]] = ent
        return state


# ----------------------------------------------------------------------------- report

class Report:
    def __init__(self):
        self.lines, self.info = [], []
        self.failed = self.total = 0

    def check(self, ok, name, detail, problems=()):
        problems = list(problems)
        self.total += 1
        self.failed += 0 if ok else 1
        self.lines.append("%s  %-13s %s" % ("PASS" if ok else "FAIL", name, detail))
        for p in problems[:12]:
            self.lines.append("        - %s" % p)
        if len(problems) > 12:
            self.lines.append("        - ... and %d more" % (len(problems) - 12))
        return ok


def processing_key(o):
    """The matcher's order: keyframe, then descending score, then det (stable sort)."""
    return (o.get("kf") or 0, -sr.f32(o.get("score") or 0.0), o.get("det") or 0)


def near(a, b, tol):
    return a is not None and b is not None and len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


# ----------------------------------------------------------------------------- 1. scans

def check_scans(p, scans, rep):
    problems = []
    starts = [p.scan_start.get(s) for s in scans]
    for s, a in zip(scans, starts):
        if a is None:
            problems.append("%s: no scan_start" % s)
            continue
        if p.scan_end.get(s) is None:
            problems.append("%s: no scan_end" % s)
        if a.get("studyLogging") is not True:
            problems.append("%s: study logging off" % s)
        # "Located": a scan only starts when located; scan_start carries the localizer's fix fields.
        if not a.get("fixPoint"):
            problems.append("%s: not located at scan start (scan_start.fixPoint is empty)" % s)
    if all(starts):
        sessions = [a.get("session") for a in starts]
        maps = [a.get("mapId") for a in starts]
        if len(set(sessions)) != len(sessions):
            problems.append("both scans in one app session (%s): the app was not restarted" % sessions[0])
        if len(set(maps)) != 1:
            problems.append("different maps: %s" % ", ".join(str(m) for m in maps))
        if (starts[0].get("t") or 0) > (starts[1].get("t") or 0):
            problems.append("the scans are not in time order")
    detail = "%d scans; sessions %s; map %s; located at %s" % (
        len(scans), ", ".join(str((a or {}).get("session")) for a in starts),
        ", ".join(sorted({str((a or {}).get("mapId")) for a in starts})),
        ", ".join(str((a or {}).get("fixPoint")) for a in starts))
    return rep.check(not problems, "scans", detail, problems)


# ----------------------------------------------------------------------------- 2. files and lines

def check_files(p, scans, rep):
    """Section 12, items 1, 2, 8 and 21."""
    problems = []
    n_lines = 0
    study = set(p.study_scans())
    for s in sorted(set(p.scan_start) - study):
        if os.path.exists(p.path(os.path.join("obs", "%s.jsonl" % s))):
            problems.append("%s: study logging off, but obs/%s.jsonl exists" % (s, s))
        if os.path.isdir(p.path(os.path.join("views", s))):
            problems.append("%s: study logging off, but views/%s/ exists" % (s, s))
    for s in scans:
        rows, bad, n = p.raw_lines(s)
        if rows is None:
            problems.append("%s: obs/%s.jsonl missing" % (s, s))
            continue
        n_lines += len(rows)
        has_end_event = s in p.scan_end
        for b in bad:
            if b != n or has_end_event:
                problems.append("%s: line %d of obs/%s.jsonl does not parse" % (s, b, s))
        model = (p.scan_start.get(s) or {}).get("embeddingModel")
        ids = collections.Counter()
        ends = [i for i, o in enumerate(rows) if o.get("type") == "scan_end"]
        if len(ends) > 1 or (ends and ends[0] != len(rows) - 1):
            problems.append("%s: the scan_end line is not the one last line" % s)
        if has_end_event and not ends:
            problems.append("%s: no scan_end line" % s)
        for o in rows:
            if o.get("v") != 1 or o.get("type") not in LINE_TYPES:
                problems.append("%s: line with v %s, type %s" % (s, o.get("v"), o.get("type")))
            if o.get("type") != "obs":
                continue
            ids[o.get("obs")] += 1
            if o.get("obs") != "%s-%s-%s" % (s, o.get("kf"), o.get("det")) or o.get("kind") != "scan":
                problems.append("%s: obs id %s for kf %s det %s" % (s, o.get("obs"), o.get("kf"), o.get("det")))
            if model and o.get("embModel") != model:
                problems.append("%s: %s embModel %s, scan_start %s" % (s, o.get("obs"), o.get("embModel"), model))
        for oid, c in ids.items():
            if c > 1:
                problems.append("%s: observation %s appears %d times" % (s, oid, c))
    return rep.check(not problems, "files", "%d lines in %d observation files" % (n_lines, len(scans)), problems)


# ----------------------------------------------------------------------------- 3. observations

def check_observations(p, scans, rep):
    """Every kept detection has one line (item 3) with embedding, camera pose and scan id (8, 7);
    its decision equals scan_keyframe.results (4) and the saved keyframe JSON (6)."""
    problems = []
    n_obs = n_kept = 0
    for s in scans:
        kfe = p.keyframe_events.get(s, {})
        kfj = p.keyframe_json(s)
        by_kf = collections.defaultdict(list)
        for o in p.observations(s):
            by_kf[o.get("kf")].append(o)
            n_obs += 1
            oid = o.get("obs")
            if o.get("scan") != s:
                problems.append("%s: %s has scan %s" % (s, oid, o.get("scan")))
            if o.get("label") == "person":
                problems.append("%s: %s is a person detection" % (s, oid))
            try:
                v = sr.decode_b64_f32(o["emb"]) if isinstance(o.get("emb"), str) else None
            except (ValueError, TypeError, struct.error):
                v = None
            if not v or len(v) != o.get("embDim"):
                problems.append("%s: %s has no usable embedding" % (s, oid))
            cam = o.get("camera") or {}
            world, cmap, m2w = cam.get("world") or {}, cam.get("map") or {}, o.get("mapToWorld")
            if not world.get("t") or not world.get("q"):
                problems.append("%s: %s has no camera pose in the world frame" % (s, oid))
            elif not cmap.get("t") or not cmap.get("q") or not m2w:
                problems.append("%s: %s has no camera pose in the map frame" % (s, oid))
            else:
                if not near(cmap["t"], map_from_world(world["t"], m2w), POS_TOL_M):
                    problems.append("%s: %s camera.map.t is not T_WM^-1 camera.world.t" % (s, oid))
                if o.get("worldPoint") and not near(o.get("mapPoint"), map_from_world(o["worldPoint"], m2w), POS_TOL_M):
                    problems.append("%s: %s mapPoint is not T_WM^-1 worldPoint" % (s, oid))
                fw = cmap.get("forward")
                if fw is None or not near(fw, rotate_by_quaternion(cmap["q"], (0.0, 0.0, -1.0)), 1e-9):
                    problems.append("%s: %s camera.map.forward is not -Z turned by camera.map.q" % (s, oid))
            dec = o.get("decision")
            if not isinstance(dec, dict) or not dec.get("kind") or not dec.get("entry"):
                problems.append("%s: %s has no baseline decision" % (s, oid))
        for idx, ev in sorted(kfe.items(), key=lambda kv: kv[0]):
            kept = (ev.get("kept") or 0) - ((ev.get("study") or {}).get("lineErrors") or 0)
            n_kept += ev.get("kept") or 0
            lines = sorted(by_kf.get(idx, []), key=lambda o: o.get("det") if o.get("det") is not None else -1)
            if len(lines) != kept:
                problems.append("%s kf %s: %d kept detections, %d observation lines" % (s, idx, kept, len(lines)))
            if [o.get("det") for o in lines] != list(range(len(lines))):
                problems.append("%s kf %s: det numbers %s" % (s, idx, [o.get("det") for o in lines]))
            st = ev.get("study") or {}
            if st and st.get("obs") is not None and st["obs"] != len(lines):
                problems.append("%s kf %s: scan_keyframe.study.obs %s, lines %d" % (s, idx, st["obs"], len(lines)))
            # item 4: aligned with scan_keyframe.results in processing order
            res = ev.get("results") or []
            ordered = sorted(lines, key=processing_key)
            if len(res) == len(ordered):
                for o, r in zip(ordered, res):
                    d = o.get("decision") or {}
                    if (o.get("label"), d.get("kind"), d.get("entry")) != (r.get("label"), r.get("kind"), r.get("entry")) \
                            or d.get("sim") != r.get("sim") or d.get("distM") != r.get("distM"):
                        problems.append("%s: %s decision %s %s, scan_keyframe.results %s %s"
                                        % (s, o.get("obs"), d.get("kind"), d.get("entry"), r.get("kind"), r.get("entry")))
            elif lines:
                problems.append("%s kf %s: %d results, %d lines" % (s, idx, len(res), len(ordered)))
            kinds = collections.Counter((o.get("decision") or {}).get("kind") for o in lines)
            for kind, key in (("known", "matched"), ("new", "new"), ("tentative", "tentative"), ("promoted", "promoted")):
                if ev.get(key) is not None and ev[key] != kinds.get(kind, 0):
                    problems.append("%s kf %s: scan_keyframe.%s %s, lines %d" % (s, idx, key, ev[key], kinds.get(kind, 0)))
            # item 6: the saved keyframe JSON, when frames were saved
            kj = kfj.get(idx)
            if kj is None:
                if (p.scan_start.get(s) or {}).get("saveFrames", True):
                    problems.append("%s kf %s: no keyframe JSON" % (s, idx))
                continue
            kept_json = [d for d in kj.get("detections") or [] if d.get("kept")]
            for o in lines:
                j = o.get("det")
                if j is None or j >= len(kept_json):
                    problems.append("%s: %s has no kept detection %s in the keyframe JSON" % (s, o.get("obs"), j))
                    continue
                d, dec = kept_json[j], o.get("decision") or {}
                if (d.get("kind"), d.get("entry")) != (dec.get("kind"), dec.get("entry")):
                    problems.append("%s: %s decision %s %s, keyframe JSON %s %s"
                                    % (s, o.get("obs"), dec.get("kind"), dec.get("entry"), d.get("kind"), d.get("entry")))
                if d.get("label") != o.get("label") or sr.f32(d.get("score", -1)) != sr.f32(o.get("score", -2)) \
                        or list(d.get("box") or []) != list(o.get("box") or []) or d.get("mapPoint") != o.get("mapPoint"):
                    problems.append("%s: %s is not the keyframe JSON's kept detection %d" % (s, o.get("obs"), j))
                if kj.get("t") is not None and kj["t"] != o.get("t"):
                    problems.append("%s: %s t %s, keyframe JSON t %s" % (s, o.get("obs"), o.get("t"), kj["t"]))
                pose = kj.get("cameraPose") or {}
                if pose.get("t") and pose["t"] != ((o.get("camera") or {}).get("world") or {}).get("t"):
                    problems.append("%s: %s camera.world differs from the keyframe JSON cameraPose" % (s, o.get("obs")))
                if o.get("personInFrame") is not None and "imageSkipped" in kj and \
                        bool(o["personInFrame"]) != (kj.get("imageSkipped") == "person"):
                    problems.append("%s: %s personInFrame %s, keyframe imageSkipped %s"
                                    % (s, o.get("obs"), o["personInFrame"], kj.get("imageSkipped")))
        for idx in sorted(set(by_kf) - set(kfe), key=lambda x: (x is None, x)):
            problems.append("%s: observation lines for keyframe %s, which has no scan_keyframe event" % (s, idx))
    if n_kept == 0:
        problems.append("no kept detection")
    return rep.check(not problems, "observations", "%d observation lines for %d kept detections" % (n_obs, n_kept), problems)


# ----------------------------------------------------------------------------- 4. views

def retained_views(p, scan):
    """(obs lines by id, retained ids, evicted ids) by section 3.2's rule."""
    lines = p.obs_lines(scan)
    by_id = {o.get("obs"): o for o in lines if o.get("type") == "obs"}
    evicted = {o.get("obs") for o in lines if o.get("type") == "view_evicted"}
    with_view = {oid for oid, o in by_id.items() if o.get("view")}
    return by_id, with_view - evicted, evicted


def person_rule_problems(o, label):
    out = []
    ov, pif = o.get("personOverlap"), o.get("personInFrame")
    if o.get("view") and ((ov or 0) > 0 or pif is None):
        out.append("%s has a view but personOverlap %s, personInFrame %s" % (label, ov, pif))
    if o.get("viewReason") == "person" and not ((ov or 0) > 0 and pif is True):
        out.append("%s viewReason person with personOverlap %s, personInFrame %s" % (label, ov, pif))
    if pif is False and (ov or 0) > 0:
        out.append("%s personInFrame false with personOverlap %s" % (label, ov))
    return out


def check_views(p, scans, rep):
    """Section 12, items 9 to 14; extra views: items 12 and 20 in part."""
    problems = []
    n_views = n_groups = n_left_out = 0
    for s in scans:
        cfg = (p.scan_start.get(s) or {}).get("study") or {}
        rr = RetentionReplay(cfg.get("groupRadiusM") or GROUP_RADIUS_M, cfg.get("viewsPerGroup") or VIEWS_PER_GROUP,
                             cfg.get("viewsPerScan") or VIEWS_PER_SCAN, cfg.get("spreadBinDeg") or SPREAD_BIN_DEG)
        lines = p.obs_lines(s)
        by_id, retained, evicted = retained_views(p, s)
        has_end = p.end_line(s) is not None
        evicted_by = {}
        for o in lines:
            if o.get("type") == "view_evicted":
                evicted_by[o.get("obs")] = o.get("by")
                continue
            if o.get("type") != "obs":
                continue
            oid = o.get("obs")
            problems += ["%s: %s" % (s, x) for x in person_rule_problems(o, oid)]
            img = o.get("image") or {}
            w, h = img.get("w"), img.get("h")
            if o.get("box") and w and h and o.get("viewRect") is not None and \
                    not near(o["viewRect"], view_rect_of(o["box"], w, h), PX_TOL):
                problems.append("%s: %s viewRect is not the padded, clamped box" % (s, oid))
            if o.get("viewPx") is not None and o.get("viewRect") is not None and w and h:
                want = upright_crop_size(o["viewRect"], w, h, o.get("rotation") or 0)
                if any(abs(a - b) > 1 for a, b in zip(o["viewPx"], want)):
                    problems.append("%s: %s viewPx %s, its padded box gives %dx%d" % (s, oid, o["viewPx"], want[0], want[1]))
            if o.get("view") and o.get("viewReason") != "kept":
                problems.append("%s: %s has a view with viewReason %s" % (s, oid, o.get("viewReason")))
            g, reason, ev = rr.feed(o)
            if g != o.get("viewGroup"):
                problems.append("%s: %s viewGroup %s, retention replay %d" % (s, oid, o.get("viewGroup"), g))
            if reason is not None and (reason != o.get("viewReason") or ev != o.get("viewReplaces")):
                if not (reason == "kept" and o.get("viewReason") == "write_error"):
                    problems.append("%s: %s viewReason %s replaces %s, retention replay %s replaces %s"
                                    % (s, oid, o.get("viewReason"), o.get("viewReplaces"), reason, ev))
            n_left_out += reason == "not_complementary" or ev is not None
        for oid in sorted(retained):
            o = by_id[oid]
            n_views += 1
            size = jpeg_size(p.path(o["view"]))
            if size is None:
                problems.append("%s: view %s missing or not a JPEG" % (s, o["view"]))
            elif o.get("viewPx") is None or tuple(size) != tuple(o["viewPx"]):
                problems.append("%s: %s is %dx%d px, viewPx %s" % (s, o["view"], size[0], size[1], o.get("viewPx")))
        for oid in sorted(evicted):
            o = by_id.get(oid)
            if o is None or not o.get("view"):
                problems.append("%s: view_evicted for %s, which had no view" % (s, oid))
            elif os.path.exists(p.path(o["view"])) and has_end:
                problems.append("%s: evicted view %s still on disk" % (s, o["view"]))
            if by_id.get(evicted_by.get(oid), {}).get("viewReplaces") != oid:
                problems.append("%s: view_evicted %s by %s, whose viewReplaces differs" % (s, oid, evicted_by.get(oid)))
        named = {os.path.basename(o["view"]) for o in by_id.values() if o.get("view")}
        for f in sorted(glob.glob(p.path(os.path.join("views", s, "*")))):
            if os.path.basename(f) not in named:
                problems.append("%s: file views/%s/%s has no observation line" % (s, s, os.path.basename(f)))
        if rr.retained_ids() != retained:
            problems.append("%s: %d views retained by the lines, %d by the retention replay" % (s, len(retained), len(rr.retained_ids())))
        if rr.max_group > rr.per_group or rr.max_scan > rr.per_scan:
            problems.append("%s: up to %d views in a group, %d in the scan" % (s, rr.max_group, rr.max_scan))
        groups = collections.Counter(by_id[i].get("viewGroup") for i in retained)
        for g, c in groups.items():
            if c > rr.per_group:
                problems.append("%s: view group %s keeps %d views (limit %d)" % (s, g, c, rr.per_group))
        end = p.end_line(s)
        if end is not None:
            vw = end.get("views") or {}
            obs = [o for o in lines if o.get("type") == "obs"]
            want = {"kept": len(retained), "written": sum(1 for o in obs if o.get("view")), "evicted": len(evicted)}
            if set(end.get("retained") or []) != retained:
                problems.append("%s: scan_end lists %d retained views, the lines give %d" % (s, len(end.get("retained") or []), len(retained)))
            for k, v in want.items():
                if vw.get(k) is not None and vw[k] != v:
                    problems.append("%s: scan_end views.%s %s, lines %d" % (s, k, vw[k], v))
            if end.get("observations") is not None and end["observations"] != len(obs):
                problems.append("%s: scan_end observations %s, lines %d" % (s, end["observations"], len(obs)))
            ev = (p.scan_end.get(s) or {}).get("study") or {}
            if ev.get("views") and vw and ev["views"] != vw:
                problems.append("%s: scan_end event study.views differs from the scan_end line" % s)
        n_groups += len(rr.groups)
    # extra views
    n_extra = 0
    extra = {o.get("obs"): o for o in p.extra_lines() if o.get("type") == "obs"}
    for oid, o in extra.items():
        n_extra += 1
        problems += person_rule_problems(o, "extra view " + str(oid))
        if o.get("personInFrame") is None and (o.get("view") or o.get("viewReason") != "person_check_failed"):
            problems.append("extra view %s without a person check, viewReason %s" % (oid, o.get("viewReason")))
        if o.get("view") and jpeg_size(p.path(o["view"])) != tuple(o.get("viewPx") or ()):
            problems.append("extra view %s: file missing or not of viewPx" % oid)
    for e in p.events:
        if e.get("type") == "verify" and e.get("extraViewObs") and e["extraViewObs"] not in extra:
            problems.append("verify %s attempt %s: extra view %s has no line" % (e.get("query"), e.get("attempt"), e["extraViewObs"]))
    return rep.check(not problems, "views", "%d views kept in %d groups (%d leave-out decisions); %d extra views"
                     % (n_views, n_groups, n_left_out, n_extra), problems)


# ----------------------------------------------------------------------------- 5. entries

def check_entries(p, scans, rep):
    """`confirmed` and `history` exist; refs (item 17); invariance (18, and pass 1 against the final
    file); user confirmations (19, in part); history items per scan, outcome and link (16)."""
    problems = []
    map_id = (p.scan_start.get(scans[0]) or {}).get("mapId")
    entries = {e["id"]: e for e in p.inventory if e.get("mapId") == map_id}
    obs_of = {}
    retained_of = {}
    study = set(p.study_scans())
    # Every study scan of the pull, not only the two being checked: an entry created by an earlier
    # scan has `confirmed.refs` into that scan's observation file (seen on the second device run,
    # 2026-11-19, where the map already held entries of two earlier scans).
    for s in list(scans) + sorted(study - set(scans)):
        try:
            by_id, retained, _ = retained_views(p, s)
        except Exception:  # noqa: BLE001  (an older scan's file that cannot be read is reported by `files`)
            continue
        obs_of.update(by_id)
        retained_of[s] = retained
    extra = {o.get("obs"): o for o in p.extra_lines() if o.get("type") == "obs"}
    confirms = collections.defaultdict(list)
    for e in p.events:
        if e.get("type") == "inv_confirm":
            confirms[e.get("inventoryId")].append(e)
    # created (item 15): the add lines during each scan
    for s in scans:
        a, b = (p.scan_start.get(s) or {}).get("t") or 0, (p.scan_end.get(s) or {}).get("t") or 0
        added = [l["entry"]["id"] for l in p.inv_log if l.get("op") == "add" and a <= (l.get("t") or 0) <= b]
        end = p.end_line(s)
        # Compared as sets: on the first device run (2026-11-19) the two lists held the same ids in a
        # different order (the `created` list is not in the order of the add lines), which loses nothing.
        if end is not None and end.get("created") is not None and sorted(end["created"]) != sorted(added):
            problems.append("%s: scan_end created %s, add lines %s" % (s, sorted(end["created"]), sorted(added)))
    # invariance over the log (item 18)
    last = {}
    for line in p.inv_log:
        ent, op = line.get("entry"), line.get("op")
        if not isinstance(ent, dict) or op == "delete":
            continue
        eid, conf = ent.get("id"), ent.get("confirmed")
        if eid in last and conf != last[eid] and op != "confirm":
            if not (last[eid] is None and op == "update" and (conf or {}).get("source") == "created"):
                problems.append("%s: confirmed changed on a %s line at t %s" % (eid, op, line.get("t")))
        if op == "confirm" and not confirms.get(eid):
            problems.append("%s: confirm line without an inv_confirm event" % eid)
        last[eid] = conf
    t_end1 = (p.scan_end.get(scans[0]) or {}).get("t") or 0
    after1 = p.inventory_state(t_end1 + 1)
    n1 = n_same = n_user = 0
    for eid, e in entries.items():
        hist = e.get("history") or []
        conf = e.get("confirmed")
        if not hist:
            problems.append("%s: no history" % eid)
        if e.get("createdScan") in study and not conf:
            problems.append("%s: no confirmed" % eid)
        if last.get(eid, conf) != conf:
            problems.append("%s: confirmed in inventory.json differs from the last log line" % eid)
        conf = conf or {}
        for ref in conf.get("refs") or []:
            if conf.get("source") == "created":
                o = obs_of.get(ref)
                if o is None:
                    problems.append("%s: confirmed ref %s is no observation" % (eid, ref))
                elif ref not in retained_of.get(conf.get("scan"), set()) or not os.path.exists(p.path(o.get("view") or "")):
                    problems.append("%s: confirmed ref %s has no retained view file" % (eid, ref))
            else:
                o = extra.get(ref)
                if o is None:
                    problems.append("%s: confirmed ref %s is no extra-view line" % (eid, ref))
                elif o.get("view") and not os.path.exists(p.path(o["view"])):
                    problems.append("%s: confirmed ref %s names a missing view" % (eid, ref))
        if conf.get("source") == "created":
            item = next((h for h in hist if h.get("scan") == conf.get("scan")), None)
            if conf.get("scan") != e.get("createdScan"):
                problems.append("%s: confirmed.scan %s, createdScan %s" % (eid, conf.get("scan"), e.get("createdScan")))
            if item is not None:
                want = [x for x in item.get("obs") or [] if x in retained_of.get(conf.get("scan"), set())]
                if list(conf.get("refs") or []) != want:
                    problems.append("%s: confirmed refs %s, history obs with a retained view %s" % (eid, conf.get("refs"), want))
                if conf.get("tMs") != item.get("lastObsMs"):
                    problems.append("%s: confirmed tMs %s, history lastObsMs %s" % (eid, conf.get("tMs"), item.get("lastObsMs")))
        elif conf.get("source") == "user_verify" and not confirms.get(eid):
            problems.append("%s: confirmed by the user without an inv_confirm event" % eid)
        # one history item per study scan the entry was alive at the end of (item 16)
        counts = collections.Counter(h.get("scan") for h in hist)
        for s in scans:
            alive = eid in p.inventory_state(((p.scan_end.get(s) or {}).get("t") or 0) + 1)
            if alive and counts.get(s, 0) != 1:
                problems.append("%s: %d history items for scan %s" % (eid, counts.get(s, 0), s))
            item = next((h for h in hist if h.get("scan") == s), None)
            if item is not None:
                ev = p.evidence.get((s, eid))
                if ev is not None and ev.get("outcome") != item.get("outcome"):
                    problems.append("%s: history outcome %s, inv_evidence %s in %s" % (eid, item.get("outcome"), ev.get("outcome"), s))
                if (item.get("link") == "created") != (e.get("createdScan") == s):
                    problems.append("%s: history link %s in %s, createdScan %s" % (eid, item.get("link"), s, e.get("createdScan")))
        # pass-1 entries: confirmed after pass 2 identical unless the user confirmed it
        if e.get("createdScan") == scans[0]:
            n1 += 1
            before = (after1.get(eid) or {}).get("confirmed")
            user = [c for c in confirms.get(eid, []) if (c.get("t") or 0) > t_end1]
            if before is None:
                problems.append("%s: no confirmed at the end of pass 1 in inventory_log.jsonl" % eid)
            elif user:
                n_user += 1
            elif before != e.get("confirmed"):
                problems.append("%s: confirmed changed after pass 1 without a user confirmation" % eid)
            else:
                n_same += 1
    detail = ("%d entries in map %s; created in pass 1: %d (confirmed unchanged %d, changed by a logged user "
              "confirmation %d)" % (len(entries), map_id, n1, n_same, n_user))
    if not entries:
        problems.append("no entry in the map")
    return rep.check(not problems, "entries", detail, problems)


# ----------------------------------------------------------------------------- 6. history from obs/

def rebuild_history(p, scan):
    """entry id -> (obs ids in the matcher's order, count, mean map point, last t)."""
    acc = collections.OrderedDict()
    for o in sorted(p.observations(scan), key=processing_key):
        eid = (o.get("decision") or {}).get("entry")
        if not eid:
            continue
        a = acc.setdefault(eid, {"obs": [], "sum": [0.0, 0.0, 0.0], "n": 0, "last": None})
        a["obs"].append(o.get("obs"))
        mp = o.get("mapPoint") or [0.0, 0.0, 0.0]
        for i in range(3):
            a["sum"][i] += mp[i]
        a["n"] += 1
        a["last"] = o.get("t") if a["last"] is None else max(a["last"], o.get("t"))
    return {eid: (a["obs"], a["n"], [x / a["n"] for x in a["sum"]], a["last"]) for eid, a in acc.items()}


def check_history(p, scans, rep):
    problems = []
    n_items = 0
    by_entry = {e["id"]: e for e in p.inventory}
    for s in scans:
        rebuilt = rebuild_history(p, s)
        end = p.end_line(s) or {}
        dropped = {d.get("entry"): d.get("obs") for d in end.get("tentativeDropped") or []}
        for eid, e in by_entry.items():
            item = next((h for h in e.get("history") or [] if h.get("scan") == s), None)
            r = rebuilt.get(eid)
            if item is None:
                if r is not None:
                    problems.append("%s: %s took %d observations but has no history item" % (s, eid, r[1]))
                continue
            n_items += 1
            if r is None:
                if item.get("sightings") or item.get("obs") or item.get("position") is not None:
                    problems.append("%s: %s history has %s sightings, the observations none" % (s, eid, item.get("sightings")))
                continue
            obs, n, pos, last = r
            if list(item.get("obs") or []) != obs:
                problems.append("%s: %s history obs %s, rebuilt %s" % (s, eid, item.get("obs"), obs))
            if item.get("sightings") != n:
                problems.append("%s: %s history sightings %s, rebuilt %d" % (s, eid, item.get("sightings"), n))
            if item.get("position") is None or sr.distance(item["position"], pos) > POS_TOL_M:
                problems.append("%s: %s history position %s, rebuilt %s" % (s, eid, item.get("position"), [round(x, 4) for x in pos]))
            if item.get("lastObsMs") != last:
                problems.append("%s: %s history lastObsMs %s, rebuilt %s" % (s, eid, item.get("lastObsMs"), last))
        for eid, r in rebuilt.items():
            if eid in by_entry:
                continue
            if eid not in dropped:
                problems.append("%s: observations went to %s, neither an entry nor a dropped tentative one" % (s, eid))
            elif list(dropped[eid] or []) != r[0]:
                problems.append("%s: dropped %s lists obs %s, rebuilt %s" % (s, eid, dropped[eid], r[0]))
    if n_items == 0:
        problems.append("no history item")
    return rep.check(not problems, "history", "%d history items rebuilt from obs/" % n_items, problems)


# ----------------------------------------------------------------------------- 7. replay

def obs_keyframes(p, scan):
    """The replay's input from obs/<scan>.jsonl: [(kf, t, [Det])] with logged embeddings and decisions."""
    by_kf = collections.defaultdict(list)
    for o in p.observations(scan):
        dec = o.get("decision") or {}
        try:
            v = sr.decode_b64_f32(o["emb"]) if isinstance(o.get("emb"), str) else None
        except (ValueError, TypeError, struct.error):
            v = None
        by_kf[o.get("kf")].append(sr.Det(
            o.get("obs"), o.get("kf"), o.get("t"), o.get("label"), o.get("score") or 0.0, o.get("box") or (0, 0, 0, 0),
            o.get("mapPoint") or (0, 0, 0), o.get("sizeM") or 0.0, embedding=v,
            logged={"kind": dec.get("kind"), "entry": dec.get("entry"), "sim": dec.get("sim"), "dist": dec.get("distM"),
                    "by_label": dec.get("byLabel"), "candidates": dec.get("candidates")},
            order=o.get("det") or 0))
    return [(kf, by_kf[kf][0].t, by_kf[kf]) for kf in sorted(by_kf, key=lambda x: (x is None, x))]


def initial_entries(p, scan):
    """The store at scan start, from inventory_log.jsonl: real entries of the scan's map, open area."""
    st = p.scan_start.get(scan) or {}
    state = p.inventory_state(st.get("t") or 0)
    return [sr.Entry.from_app(e) for e in state.values() if e.get("mapId") == st.get("mapId") and not e.get("container")]


def check_replay(p, scans, rep):
    """Replay of each pass with the logged embeddings: decisions, byLabel, sim, distM, the logged
    candidates (item 5) and the dropped tentative entries."""
    problems = []
    totals = []
    for k, s in enumerate(scans):
        kfs = obs_keyframes(p, s)
        if not kfs:
            problems.append("%s: nothing to replay" % s)
            continue
        cand_problems = []

        def observe(det, dec, rp, used, s=s, cand_problems=cand_problems):
            logged = det.logged.get("candidates")
            if logged is None:
                return
            mine = sr.nearest_candidates(rp, det, used, max(3, len(logged)))[:len(logged)]
            for a, b in zip(logged, mine):
                same = (a.get("entry") == b["entry"] and abs((a.get("distM") or 0) - b["distM"]) <= POS_TOL_M and
                        a.get("sameLabel") == b["sameLabel"] and a.get("withinRadius") == b["withinRadius"] and
                        bool(a.get("tentative")) == b["tentative"] and
                        (b["sim"] is None or a.get("sim") is None or abs(a["sim"] - b["sim"]) <= SIM_TOL))
                if not same:
                    show = lambda c: "%s d %.4f sim %s sameLabel %s withinRadius %s tentative %s" % (  # noqa: E731
                        c.get("entry"), c.get("distM") or -1, "-" if c.get("sim") is None else "%.4f" % c["sim"],
                        c.get("sameLabel"), c.get("withinRadius"), bool(c.get("tentative")))
                    cand_problems.append("%s: %s candidate logged %s; replay %s" % (s, det.key, show(a), show(b)))
                    break
            if len(logged) > 3 or len(logged) != len(mine):
                cand_problems.append("%s: %s lists %d candidates, the replay %d" % (s, det.key, len(logged), len(mine)))

        try:
            decs, rp, dropped = sr.run_forced(kfs, sr.sim_embedding, initial_entries(p, s), observer=observe)
        except (ValueError, KeyError, TypeError) as ex:
            problems.append("%s: replay failed: %s" % (s, ex))
            continue
        rows = sr.compare(decs)
        totals.append((k + 1, sr.tally(rows)))
        for r, d in zip(rows, decs):
            lg = d.det.logged
            if r["status"] != sr.OK:
                problems.append("%s: %s %s logged %s %s, replay %s %s%s"
                                % (s, r["key"], r["label"], r["logged_kind"], r["logged_entry"], r["replay_kind"],
                                   r["replay_entry"], " (missing similarity)" if r["missing"] else ""))
                continue
            if lg.get("by_label") is not None and d.by_label is not None and lg["by_label"] != d.by_label:
                problems.append("%s: %s byLabel logged %s, replay %s" % (s, r["key"], lg["by_label"], d.by_label))
            if lg.get("sim") is not None and d.sim is not None and abs(lg["sim"] - d.sim) > SIM_TOL:
                problems.append("%s: %s sim logged %.5f, replay %.5f" % (s, r["key"], lg["sim"], d.sim))
            if lg.get("dist") is not None and d.dist is not None and abs(lg["dist"] - d.dist) > POS_TOL_M:
                problems.append("%s: %s distM logged %.6f, replay %.6f" % (s, r["key"], lg["dist"], d.dist))
        problems += cand_problems
        end = p.end_line(s) or {}
        logged_dropped = sorted(d.get("entry") for d in end.get("tentativeDropped") or [])
        if end and sorted(dropped) != logged_dropped:
            problems.append("%s: the replay drops %d tentative entries, the log %d" % (s, len(dropped), len(logged_dropped)))
    detail = "; ".join("pass %d: %s" % (k, sr.fmt_tally(t)) for k, t in totals) or "nothing replayed"
    ok = not problems and len(totals) == len(scans)
    return rep.check(ok, "replay", detail, problems)


def alignment(p, scans):
    """Distances between the pass-1 and pass-2 history positions of entries the baseline linked to
    observations in both passes."""
    d = []
    for e in p.inventory:
        h = {x.get("scan"): x for x in e.get("history") or []}
        a, b = h.get(scans[0]), h.get(scans[1])
        if a and b and a.get("position") is not None and b.get("position") is not None:
            d.append((e["id"], sr.distance(a["position"], b["position"])))
    return d


# ----------------------------------------------------------------------------- main

CHECKS = ("scans", "files", "observations", "views", "entries", "history", "replay")


def run(folder, scans_arg=None, out=sys.stdout):
    p = Pull(folder)
    rep = Report()
    study = p.study_scans()
    scans = [s for s in scans_arg.split(",") if s] if scans_arg else study[-2:]
    print("Study check of %s" % folder, file=out)
    if not study:
        print("This is a pre-v0.8 pull: %d scan(s), none with study logging on (scan_start.studyLogging), %s."
              % (len(p.scan_start), "an obs/ folder exists" if os.path.isdir(p.path("obs")) else "no obs/ folder"), file=out)
        print("It cannot pass: a v0.8 two-pass smoke test is needed.", file=out)
    if len(scans) < 2:
        rep.check(False, "scans", "need two scans with study logging, found %d" % len(study),
                  ["scans with study logging: %s" % (", ".join(study) or "none"),
                   "all scans: %s" % (", ".join(sorted(str(s) for s in p.scan_start)) or "none")])
        for name in CHECKS[1:]:
            rep.check(False, name, "not checked: no two-pass study data")
    else:
        scans = scans[:2]
        print("Scans: pass 1 %s, pass 2 %s" % tuple(scans), file=out)
        check_scans(p, scans, rep)
        check_files(p, scans, rep)
        check_observations(p, scans, rep)
        check_views(p, scans, rep)
        check_entries(p, scans, rep)
        check_history(p, scans, rep)
        check_replay(p, scans, rep)
        al = alignment(p, scans)
        if al:
            ds = [x for _, x in al]
            rep.info.append("alignment between the launches (reported, not judged): n = %d entries matched in both "
                            "passes, median %.3f m, max %.3f m" % (len(ds), statistics.median(ds), max(ds)))
        else:
            rep.info.append("alignment between the launches: no entry matched in both passes")
    for line in rep.lines + rep.info:
        print(line, file=out)
    print(("FAIL: %d of %d checks" % (rep.failed, rep.total)) if rep.failed else ("PASS: all %d checks" % rep.total), file=out)
    return 1 if rep.failed else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", help="pulled data folder")
    ap.add_argument("--scans", default=None, help="pass-1 and pass-2 scan ids, comma-separated")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.folder):
        print("No folder %s" % args.folder, file=sys.stderr)
        return 2
    return run(args.folder, args.scans)


if __name__ == "__main__":
    sys.exit(main())
