#!/usr/bin/env python3
"""Offline identity replay: several ways of deciding which known item a test-scan sighting is, run
over the same observations and the same history (paper/IDENTITY_PILOT_PROTOCOL.md, Part B).

    python3 analysis/identity_replay.py <pull> --history <scan>[,<scan>...] --test <scan>[,<scan>...]
    python3 analysis/identity_replay.py <pull> --history H1,H2 --test T1,T2 \
        --labels <obs,item csv> [--entry-labels <entry,item csv>] [--sheet <filled ground-truth csv>]
    python3 analysis/identity_replay.py --freeze analysis/out/identity_replay/frozen_params.json
    python3 analysis/identity_replay.py <pull> ... --frozen analysis/out/identity_replay/frozen_params.json

Input: one pulled data folder of the v0.8 study build (FindBack/docs/PLAN_v0.8_study_logging.md):
`obs/<scan>.jsonl`, `obs/extra_views.jsonl`, `scan/<scan>/kf*.json` (and the saved depth),
`events.jsonl`, `inventory_log.jsonl`. History scans are read for the known items; every test scan
is decided against the state at the end of the last history scan (test scans do not feed each other).

Methods (design and parameters: analysis/notes/2026-11-19_identity_replay_design.md):

* ``current``: the app's own matcher, from the logged `decision` of each observation
  (re-checked by analysis/scan_replay.py through analysis/study_check.py).
* ``generic``: rules that need no knowledge of this home: an alignment correction estimated from the
  test scan, match in place, exclusivity, moved only to items whose place was looked at and found
  empty (with a similarity threshold and a margin), items whose place was not looked at only
  unresolved, otherwise add.
* ``household``: the same rules with a pack built from the history scans only (several reference
  views per item, look-alike groups, a threshold per item and group, a stability prior). The method
  reads only the pack file and the test scan.
* ``extra_view``: household plus the extra views logged after the test scan.

Every decision is per sighting: `update <item>`, `add`, or `unresolved <candidates>` (the current
matcher works per observation and also has `drop`: the observation went to a tentative entry that
the scan dropped). With label files the outcomes are scored per physical item and condition.

Standard library only. No image is read: every similarity comes from the logged embeddings. Output
with item codes or positions goes to analysis/out/ (git-ignored).
"""
import argparse
import array
import base64
import collections
import csv
import hashlib
import json
import math
import operator
import os
import statistics
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import scan_replay as sr  # noqa: E402
import study_check as sc  # noqa: E402

# ============================================================================ parameters, one place
#
# Every threshold and rule switch. Each entry: name -> (value, where it came from). The full set is
# printed with every run and written into the result file with the git commit, so it can be frozen
# (--freeze) before the pilot's test scans are read, and checked (--frozen) when they are.
# "pass 1" = scan e4144f49 of the smoke pull (backups/phone_files_2026-11-19_smoke_1752), the only
# data used to choose values; nothing was chosen from its pass 2 or from the audit labels.

PARAMETERS = collections.OrderedDict([
    # -- sightings
    ("SIGHTING_SIM", (0.40, "an observation joins a sighting of its view group only at this cosine to one of its "
                            "members: the same value as SIM_PLACE")),
    # -- in place
    ("PLACE_RADIUS_M", (0.30, "between launches the same entries moved by a median of 0.12 m, at most 0.29 m "
                              "(notes/2026-11-19_smoke_test.md); in pass 1, pairs of observations with cosine "
                              ">= 0.75 and the same label lay a median 0.04 m, 95th percentile 0.24 m apart (n = 53)")),
    ("SIM_PLACE", (0.40, "pass 1: 95th percentile (0.402) of the cosine between observations of the same label "
                         "more than FAR_M apart, which are different objects (n = 1209 pairs); 94 % of pairs of "
                         "the same label within 0.10 m in different keyframes lie above it (n = 83)")),
    # -- moved, unresolved, add
    ("SIM_MOVED", (0.60, "pass 1: the highest cosine (0.598) between observations of the same label more than "
                         "FAR_M apart (n = 1209); the app's SEEN_SIMILARITY is also 0.60")),
    ("MOVED_MARGIN", (0.10, "chosen, not derived: the best moved candidate must beat every other candidate by "
                            "this much")),
    ("SIM_UNRESOLVED", (0.40, "a candidate at or above this (SIM_PLACE) and not decided makes the sighting "
                              "unresolved; below it the sighting is added")),
    ("UNRESOLVED_MAX_CANDIDATES", (3, "candidates listed (and sent to a server) per unresolved sighting")),
    ("ELSEWHERE_SAME_LABEL", (True, "moved and unresolved candidates need the sighting to carry the item's "
                                    "detector label (the app's moved rule also asks for the same label); "
                                    "in place, appearance alone decides")),
    # -- place looked at (the app's evidence rule, ScanConfig v0.8)
    ("MISSED_MIN_KEYFRAMES", (3, "app ScanConfig.MISSED_MIN_KEYFRAMES")),
    ("VIEW_MARGIN", (0.08, "app ScanConfig.VIEW_MARGIN")),
    ("VIEW_MIN_M", (0.3, "app ScanConfig.VIEW_MIN_M")),
    ("VIEW_MAX_M", (3.0, "app ScanConfig.VIEW_MAX_M")),
    ("DEPTH_PATCH", (5, "app ScanConfig.DEPTH_PATCH")),
    ("HIDDEN_MIN_M", (0.15, "app ScanConfig.HIDDEN_MIN_M")),
    ("HIDDEN_FRACTION", (0.15, "app ScanConfig.HIDDEN_FRACTION")),
    ("USE_PHONE_APPEARANCE_CHECK", (True, "an unmatched item whose expected window the phone checked "
                                          "(scan_keyframe.checks[].sim) at or above SEEN_SIMILARITY counts as "
                                          "seen in place, not missed")),
    ("SEEN_SIMILARITY", (0.60, "app ScanConfig.SEEN_SIMILARITY")),
    # -- alignment between launches
    ("ALIGN", (True, "estimate the correction (switch off with --align off; --align both runs both)")),
    ("ALIGN_SEARCH_M", (0.50, "pairs for the correction are searched this far: above the 0.29 m largest "
                              "launch-to-launch difference of the smoke note; the app's largest merge radius")),
    ("ALIGN_MIN_SIM", (0.60, "a pair for the correction needs a confident appearance match: SIM_MOVED")),
    ("ALIGN_MIN_PAIRS", (3, "fewest mutual-best pairs for a translation")),
    ("ALIGN_INLIER_M", (0.20, "a pair agrees with a correction within this residual: about the 90th percentile "
                              "(0.18 m) of the pass-1 spread of very similar observations")),
    ("ALIGN_MIN_PAIRS_YAW", (4, "fewest agreeing pairs to fit a yaw as well")),
    ("ALIGN_MIN_SPAN_M", (0.50, "and their horizontal spread around their centre must reach this")),
    ("ALIGN_MAX_SHIFT_M", (0.50, "a correction that moves the pairs' centre more than this is refused")),
    ("ALIGN_MAX_YAW_DEG", (10.0, "a yaw larger than this is refused (falls back to the translation)")),
    # -- household pack (history only)
    ("FAR_M", (1.0, "observations of one scan this far apart are different objects (the impostor set)")),
    ("REF_CONSISTENCY_SIM", (0.40, "a history view becomes a reference of an entry only at this cosine to the "
                                   "entry's own stored embedding (SIM_PLACE): drops views of a swapped neighbour")),
    ("SELF_FALLBACK", (0.75, "self-similarity of an item with one reference: pass-1 median (0.758) of pairs of "
                             "the same label within 0.10 m in different keyframes (n = 83)")),
    ("HH_IMPOSTOR_MARGIN", (0.05, "chosen: an item's moved threshold is its highest impostor cosine plus this")),
    ("HH_THR_MIN", (0.40, "an item's moved threshold is never below SIM_PLACE")),
    ("HH_THR_MAX", (0.90, "an item whose threshold would exceed this can only be unresolved, never moved")),
    ("STABLE_SPREAD_M", (0.10, "an item whose history positions (aligned) spread at most this is stable")),
    ("STABLE_MOVED_BONUS", (0.05, "a stable item needs this much more similarity to be moved")),
    ("STABILITY_PAD_M", (0.10, "an item's place radius is its history spread plus this ...")),
    ("PLACE_RADIUS_MAX_M", (0.50, "... between PLACE_RADIUS_M and this")),
    ("PACK_EMBEDDING", ("float16", "references are stored as little-endian float16 in base64")),
    # -- extra views
    ("EXTRA_VIEW_LOOKS", (3, "an extra view aimed within an item's place radius counts as this many looks at "
                             "the place (MISSED_MIN_KEYFRAMES: the user was guided there)")),
])


class Params:
    """The parameter set; tests may override single values: Params(SIM_MOVED=0.7)."""

    def __init__(self, **over):
        for k, (v, _) in PARAMETERS.items():
            setattr(self, k, v)
        for k, v in over.items():
            if k not in PARAMETERS:
                raise KeyError("unknown parameter %s" % k)
            setattr(self, k, v)

    def as_dict(self):
        return collections.OrderedDict((k, getattr(self, k)) for k in PARAMETERS)


DEFAULT = Params()

METHODS = ("current", "generic", "household", "extra_view")
UPDATE, ADD, UNRESOLVED, DROP, NONE = "update", "add", "unresolved", "drop", "none"

# ============================================================================ vectors

def _dot(a, b):
    return sum(map(operator.mul, a, b))


def unit(v):
    n = math.sqrt(_dot(v, v))
    return [x / n for x in v] if n > 0 else list(v)


def encode_f16(v):
    return base64.b64encode(struct.pack("<%de" % len(v), *v)).decode("ascii")


def decode_f16(s):
    raw = base64.b64decode(s)
    return list(struct.unpack("<%de" % (len(raw) // 2), raw))


def dist(p, q):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


def median(xs):
    return statistics.median(xs) if xs else None


# ============================================================================ data

class Obs:
    """One observation (a kept detection of a scan, or an extra view)."""

    __slots__ = ("id", "scan", "kf", "det", "t", "label", "score", "map_point", "size_m", "emb", "view_group",
                 "retained", "decision", "extra", "entry", "query", "ms_to_capture", "request_ms")

    def __init__(self, oid, scan, kf, det, t, label, score, map_point, emb, size_m=0.1, view_group=None,
                 retained=False, decision=None, extra=False, entry=None):
        self.id, self.scan, self.kf, self.det, self.t = oid, scan, kf, det, t
        self.label, self.score, self.map_point, self.size_m = label, score, tuple(map_point), size_m
        self.emb, self.view_group, self.retained, self.decision = emb, view_group, retained, decision or {}
        self.extra, self.entry = extra, entry
        self.query = self.ms_to_capture = self.request_ms = None


class DepthMap:
    """A saved depth image (uint16 millimetres, row-major) and its image-to-depth affine map."""

    def __init__(self, width, height, mm, from_image):
        self.width, self.height, self.mm, self.a = width, height, mm, from_image

    @classmethod
    def load(cls, path, width, height, from_image):
        a = array.array("H")
        with open(path, "rb") as f:
            a.frombytes(f.read())
        if sys.byteorder != "little":
            a.byteswap()
        return cls(width, height, a, from_image)

    def median_patch(self, u, v, size):
        """ScanGeometry.DepthMap.medianPatch: metres, or None."""
        a, b, c, d, e, f = self.a
        x, y = int(math.floor(a * u + b * v + c)), int(math.floor(d * u + e * v + f))
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return None
        r = size // 2
        vals = []
        for yy in range(max(0, y - r), min(self.height - 1, y + r) + 1):
            for xx in range(max(0, x - r), min(self.width - 1, x + r) + 1):
                m = self.mm[yy * self.width + xx]
                if m:
                    vals.append(m)
        if not vals:
            return None
        vals.sort()
        return vals[len(vals) // 2] / 1000.0


def quat_rotate(q, v):
    """The app's CameraPose.rotate: v turned by the unit quaternion (x, y, z, w)."""
    x, y, z, w = q
    cx, cy, cz = y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0]
    dx, dy, dz = y * cz - z * cy, z * cx - x * cz, x * cy - y * cx
    return (v[0] + 2 * (w * cx + dx), v[1] + 2 * (w * cy + dy), v[2] + 2 * (w * cz + dz))


class KeyGeom:
    """One keyframe's camera: intrinsics, world pose, map transform and (when saved) depth."""

    def __init__(self, intrinsics, cam_t, cam_q, map_to_world, depth=None):
        self.k, self.t, self.q, self.m2w = intrinsics, tuple(cam_t), tuple(cam_q), map_to_world
        self.depth = depth          # a DepthMap, a zero-argument loader, or None

    def depth_map(self):
        if callable(self.depth):
            try:
                self.depth = self.depth()
            except (OSError, ValueError):
                self.depth = None
        return self.depth

    def project_map(self, p):
        """(u, v, z) of a map point in the CPU image, or None behind the camera (KeyframeGeometry.projectMap)."""
        w = sr.world_from_map(p, self.m2w)
        c = quat_rotate((-self.q[0], -self.q[1], -self.q[2], self.q[3]), (w[0] - self.t[0], w[1] - self.t[1], w[2] - self.t[2]))
        d = -c[2]
        if d <= 1e-6:
            return None
        return self.k["cx"] + self.k["fx"] * c[0] / d, self.k["cy"] - self.k["fy"] * c[1] / d, d


class ScanData:
    """What one scan logged, as the methods read it."""

    def __init__(self, scan, observations, keyframes=None, kf_list=None, checks=None, evidence=None,
                 created=(), dropped=(), session=None, map_id=None, t_start=0, t_end=0):
        self.scan, self.observations = scan, list(observations)
        self.keyframes = keyframes or {}
        self.kf_list = sorted(kf_list if kf_list is not None else set(self.keyframes) | {o.kf for o in self.observations})
        self.checks = checks or {}           # (kf, entry) -> {"vis", "sim"}: the phone's evidence checks
        self.evidence = evidence or {}       # entry -> the phone's outcome of this scan
        self.created, self.dropped = set(created), set(dropped)
        self.session, self.map_id, self.t_start, self.t_end = session, map_id, t_start, t_end


class History:
    """The history scans and the entries alive at the end of the last one (dicts as in inventory.json)."""

    def __init__(self, scans, entries, map_id=None):
        self.scans, self.entries, self.map_id = list(scans), collections.OrderedDict(entries), map_id

    def obs_by_id(self):
        return {o.id: o for s in self.scans for o in s.observations}


# ---------------------------------------------------------------------------- reading a pull

def inventory_at(inv_log, t_ms):
    """Entries (id -> entry) from inventory_log.jsonl lines before t_ms, with `evidence` lines applied
    (plan section 6: they append their historyItem)."""
    state = collections.OrderedDict()
    for line in inv_log:
        if (line.get("t") or 0) >= t_ms:
            break
        op, ent = line.get("op"), line.get("entry")
        if op == "delete" and isinstance(ent, dict):
            state.pop(ent.get("id"), None)
        elif op == "evidence":
            e = state.get(line.get("id"))
            if e is not None:
                e = dict(e)
                e["evidence"] = line.get("evidence")
                if line.get("historyItem"):
                    e["history"] = list(e.get("history") or []) + [line["historyItem"]]
                state[e["id"]] = e
        elif isinstance(ent, dict) and ent.get("id"):
            if op == "add":
                state.pop(ent["id"], None)
            state[ent["id"]] = ent
    return state


def obs_from_line(o, retained=False):
    emb = unit(sr.decode_b64_f32(o["emb"])) if isinstance(o.get("emb"), str) else None
    return Obs(o.get("obs"), o.get("scan"), o.get("kf"), o.get("det"), o.get("t"), o.get("label"),
               o.get("score") or 0.0, o.get("mapPoint") or (0.0, 0.0, 0.0), emb, o.get("sizeM") or 0.0,
               o.get("viewGroup"), retained, o.get("decision"))


def load_scan(pull, scan):
    lines = pull.obs_lines(scan)
    end = pull.end_line(scan) or {}
    _, retained, _ = sc.retained_views(pull, scan)
    observations = [obs_from_line(o, o.get("obs") in retained) for o in lines if o.get("type") == "obs"]
    start, end_ev = pull.scan_start.get(scan) or {}, pull.scan_end.get(scan) or {}
    keyframes = {}
    folder = pull.path(os.path.join("scan", scan))
    for idx, kj in pull.keyframe_json(scan).items():
        if not kj.get("cameraPose") or not kj.get("intrinsics") or not kj.get("mapToWorld"):
            continue
        depth = None
        dj = kj.get("depth") or {}
        if dj.get("file") and kj.get("depthFromImage"):
            path = os.path.join(folder, dj["file"])
            if os.path.exists(path):
                depth = (lambda path=path, dj=dj, a=tuple(kj["depthFromImage"]):
                         DepthMap.load(path, dj["width"], dj["height"], a))
        keyframes[idx] = KeyGeom(kj["intrinsics"], kj["cameraPose"]["t"], kj["cameraPose"]["q"], kj["mapToWorld"], depth)
    checks = {}
    for idx, ev in pull.keyframe_events.get(scan, {}).items():
        for c in ev.get("checks") or []:
            checks[(idx, c.get("entry"))] = {"vis": c.get("vis"), "sim": c.get("sim"), "u": c.get("u"),
                                             "v": c.get("v"), "zM": c.get("zM"), "measuredM": c.get("measuredM")}
    evidence = {eid: e.get("outcome") for (s, eid), e in pull.evidence.items() if s == scan}
    kf_list = set(pull.keyframe_events.get(scan, {})) | set(keyframes) | {o.kf for o in observations}
    return ScanData(scan, observations, keyframes, kf_list, checks, evidence, end.get("created") or (),
                    [d.get("entry") for d in end.get("tentativeDropped") or []], start.get("session"),
                    start.get("mapId"), start.get("t") or 0, end_ev.get("t") or end.get("t") or 0)


def load_history(pull, scans):
    data = [load_scan(pull, s) for s in scans]
    map_id = data[0].map_id
    state = inventory_at(pull.inv_log, (data[-1].t_end or 0) + 1)
    entries = [(eid, e) for eid, e in state.items() if e.get("mapId") == map_id and not e.get("container")]
    return History(data, entries, map_id)


def load_extra_views(pull, scan):
    """Extra-view lines logged after the test scan and before the next scan (any session)."""
    nxt = min([e.get("t") for e in pull.scan_start.values() if (e.get("t") or 0) > scan.t_end] or [float("inf")])
    ends = {e.get("query"): e for e in pull.events if e.get("type") == "query_end"}
    out = []
    for o in pull.extra_lines():
        if o.get("type") != "obs" or o.get("kind") != "extra_view" or not isinstance(o.get("emb"), str):
            continue
        if not (scan.t_end < (o.get("t") or 0) < nxt) or (o.get("mapId") and o.get("mapId") != scan.map_id):
            continue
        tgt = (o.get("target") or {}).get("mapPoint")
        if not tgt:
            continue
        x = Obs(o["obs"], scan.scan, "x:" + o["obs"], 0, o.get("t"), None, 0.0, tgt, unit(sr.decode_b64_f32(o["emb"])),
                extra=True, entry=o.get("entry"))
        x.query, x.ms_to_capture, x.request_ms = o.get("query"), o.get("msSinceQueryStart"), o.get("requestStartMs")
        qe = ends.get(o.get("query"))
        x.decision = {"wholeMs": (qe.get("t") - o["requestStartMs"]) if qe and o.get("requestStartMs") else None,
                      "groundTruth": (qe or {}).get("groundTruth")}
        out.append(x)
    return out


def all_extra_views(pull):
    """Every extra view of the pull with the scan it follows (for the report)."""
    starts = sorted((e.get("t") or 0, s) for s, e in pull.scan_start.items())
    ends = {e.get("query"): e for e in pull.events if e.get("type") == "query_end"}
    out = []
    for o in pull.extra_lines():
        if o.get("type") != "obs":
            continue
        before = [s for t, s in starts if t < (o.get("t") or 0)]
        qe = ends.get(o.get("query"))
        out.append({"obs": o.get("obs"), "after_scan": before[-1] if before else None,
                    "ms_to_capture": o.get("msSinceQueryStart"),
                    "whole_ms": (qe.get("t") - o["requestStartMs"]) if qe and o.get("requestStartMs") else None})
    return out


# ============================================================================ sightings

class Sighting:
    __slots__ = ("id", "order", "obs", "kfs", "pos", "labels", "extra")

    def __init__(self, sid, order, first):
        self.id, self.order, self.obs, self.kfs = sid, order, [first], {first.kf}
        self.extra = first.extra
        self.pos = first.map_point
        self.labels = collections.Counter()
        if first.label:
            self.labels[first.label] += first.score or 1e-6

    def add(self, o):
        self.obs.append(o)
        self.kfs.add(o.kf)
        n = len(self.obs)
        self.pos = tuple(sum(x.map_point[i] for x in self.obs) / n for i in range(3))
        if o.label:
            self.labels[o.label] += o.score or 1e-6

    @property
    def label(self):
        return self.labels.most_common(1)[0][0] if self.labels else None


def build_sightings(scan, p=DEFAULT, extra=()):
    """Sightings by place: the phone's view groups (anchors at least 0.30 m apart, every observation in
    the nearest anchor's group within 0.30 m; plan section 4), split within a group so that no sighting
    holds two observations of one keyframe and an observation joins the most similar sighting only at
    SIGHTING_SIM. File order. No sighting is dropped, however few observations it has. Each extra view
    is a sighting of its own."""
    groups = collections.defaultdict(list)
    rr = None
    out = []
    for o in sorted(scan.observations, key=lambda o: (o.kf, o.det)):
        g = o.view_group
        if g is None:
            rr = rr or sc.RetentionReplay()
            g = rr.group_of(o.map_point)
        best = None
        for s in groups[g]:
            if o.kf in s.kfs:
                continue
            sim = max(_dot(o.emb, m.emb) for m in s.obs)
            if sim >= p.SIGHTING_SIM and (best is None or sim > best[0]):
                best = (sim, s)
        if best is not None:
            best[1].add(o)
        else:
            s = Sighting("%s-s%d" % (scan.scan, len(out)), len(out), o)
            groups[g].append(s)
            out.append(s)
    for x in extra:
        out.append(Sighting("%s-x%d" % (scan.scan, len(out)), len(out), x))
    return out


# ============================================================================ items

class Item:
    """A known item as a method sees it."""

    __slots__ = ("id", "order", "label", "place", "refs", "thr_place", "thr_moved", "radius", "aliases", "group")

    def __init__(self, iid, order, label, place, refs, thr_place, thr_moved, radius, aliases=(), group=None):
        self.id, self.order, self.label, self.place, self.refs = iid, order, label, tuple(place), refs
        self.thr_place, self.thr_moved, self.radius = thr_place, thr_moved, radius
        self.aliases, self.group = set(aliases), group


def entry_place(e):
    return (e.get("confirmed") or {}).get("mapPosition") or e.get("mapPosition")


def entry_embedding(e):
    emb = e.get("embedding")
    return unit(sr.decode_b64_f32(emb)) if isinstance(emb, str) else None


def generic_items(history, p=DEFAULT):
    """Generic items: each entry's last confirmed place and its one stored embedding, fixed thresholds."""
    out = []
    for k, (eid, e) in enumerate(history.entries.items()):
        v = entry_embedding(e)
        out.append(Item(eid, k, e.get("label"), entry_place(e), [v] if v else [], p.SIM_PLACE, p.SIM_MOVED,
                        p.PLACE_RADIUS_M))
    return out


# ============================================================================ alignment

class Align:
    """p_history = R(yaw) p_test + t (rotation about +Y as Pose4.rotate)."""

    def __init__(self, yaw_deg=0.0, t=(0.0, 0.0, 0.0), kind="none", info=None):
        self.yaw_deg, self.t, self.kind, self.info = yaw_deg, tuple(t), kind, info or {}

    def apply(self, p):
        a = math.radians(self.yaw_deg)
        c, s = math.cos(a), math.sin(a)
        return (c * p[0] + s * p[2] + self.t[0], p[1] + self.t[1], -s * p[0] + c * p[2] + self.t[2])

    def inverse(self, q):
        a = math.radians(-self.yaw_deg)
        c, s = math.cos(a), math.sin(a)
        x, y, z = q[0] - self.t[0], q[1] - self.t[1], q[2] - self.t[2]
        return (c * x + s * z, y, -s * x + c * z)

    def summary(self):
        d = {"kind": self.kind, "yawDeg": round(self.yaw_deg, 3), "t": [round(x, 4) for x in self.t]}
        d.update(self.info)
        return d


def fit_translation(pairs):
    return Align(0.0, [statistics.median(q[i] - p[i] for p, q in pairs) for i in range(3)], "translation")


def fit_yaw_translation(pairs):
    n = len(pairs)
    cp = [sum(p[i] for p, _ in pairs) / n for i in range(3)]
    cq = [sum(q[i] for _, q in pairs) / n for i in range(3)]
    sxx = sxz = 0.0
    for p, q in pairs:
        ax, az, bx, bz = p[0] - cp[0], p[2] - cp[2], q[0] - cq[0], q[2] - cq[2]
        sxx += bx * ax + bz * az
        sxz += bx * az - bz * ax
    yaw = math.degrees(math.atan2(sxz, sxx))
    r = Align(yaw, (0.0, 0.0, 0.0)).apply(cp)
    return Align(yaw, (cq[0] - r[0], cq[1] - r[1], cq[2] - r[2]), "yaw+translation")


def span_xz(points):
    cx, cz = sum(p[0] for p in points) / len(points), sum(p[2] for p in points) / len(points)
    return max(math.hypot(p[0] - cx, p[2] - cz) for p in points)


def robust_alignment(pairs, p=DEFAULT):
    """A correction from (test point, history point) pairs: translation by medians, then a yaw when
    enough agreeing pairs spread far enough; refused when outside the limits or no better."""
    ident = Align()
    info = {"pairs": len(pairs)}
    res0 = [dist(a, b) for a, b in pairs]
    info["medianResidualBeforeM"] = round(median(res0), 4) if res0 else None
    if len(pairs) < p.ALIGN_MIN_PAIRS:
        info["reason"] = "fewer than %d pairs" % p.ALIGN_MIN_PAIRS
        return Align(info=info)
    tr = fit_translation(pairs)
    inl = [(a, b) for a, b in pairs if dist(tr.apply(a), b) <= p.ALIGN_INLIER_M]
    if len(inl) < p.ALIGN_MIN_PAIRS:
        info["reason"] = "no %d pairs agree on a translation" % p.ALIGN_MIN_PAIRS
        return Align(info=info)
    best = fit_translation(inl)
    if len(inl) >= p.ALIGN_MIN_PAIRS_YAW and span_xz([a for a, _ in inl]) >= p.ALIGN_MIN_SPAN_M:
        yt = fit_yaw_translation(inl)
        inl2 = [(a, b) for a, b in pairs if dist(yt.apply(a), b) <= p.ALIGN_INLIER_M]
        if len(inl2) >= p.ALIGN_MIN_PAIRS_YAW:
            yt = fit_yaw_translation(inl2)
        med_t = median([dist(best.apply(a), b) for a, b in pairs])
        med_y = median([dist(yt.apply(a), b) for a, b in pairs])
        info["yawFitDeg"], info["medianResidualTranslationM"] = round(yt.yaw_deg, 3), round(med_t, 4)
        info["medianResidualYawM"] = round(med_y, 4)
        if abs(yt.yaw_deg) > p.ALIGN_MAX_YAW_DEG:
            info["yawRefusedDeg"] = round(yt.yaw_deg, 2)
        elif med_y < med_t:
            # the yaw is kept only when it fits the pairs better than the translation alone
            best, inl = yt, [(a, b) for a, b in pairs if dist(yt.apply(a), b) <= p.ALIGN_INLIER_M]
    centre = [sum(a[i] for a, _ in inl) / len(inl) for i in range(3)]
    shift = dist(best.apply(centre), centre)
    res1 = [dist(best.apply(a), b) for a, b in pairs]
    info.update(inliers=len(inl), shiftM=round(shift, 4), medianResidualAfterM=round(median(res1), 4))
    if shift > p.ALIGN_MAX_SHIFT_M:
        info["reason"] = "shift %.3f m over the limit" % shift
        return Align(info=info)
    if median(res1) >= median(res0):
        info["reason"] = "no smaller residual"
        return Align(info=info)
    best.info = info
    return best if best.kind != "none" else ident


def estimate_alignment(sightings, items, S, p=DEFAULT):
    """Mutual-best pairs (same label, within ALIGN_SEARCH_M, cosine >= ALIGN_MIN_SIM) -> a correction."""
    best_s, best_i = {}, {}
    for s in sightings:
        if s.extra:
            continue
        for it in items:
            sim = S[s.id][it.id]
            if sim is None or sim < p.ALIGN_MIN_SIM or it.label not in s.labels:
                continue
            if dist(s.pos, it.place) > p.ALIGN_SEARCH_M:
                continue
            if sim > best_s.get(s.id, (-2, None))[0]:
                best_s[s.id] = (sim, it.id)
            if sim > best_i.get(it.id, (-2, None))[0]:
                best_i[it.id] = (sim, s.id)
    by_s = {s.id: s for s in sightings}
    by_i = {it.id: it for it in items}
    pairs, names = [], []
    for sid, (_, iid) in sorted(best_s.items()):
        if best_i.get(iid, (None, None))[1] == sid:
            pairs.append((by_s[sid].pos, by_i[iid].place))
            names.append((sid, iid))
    a = robust_alignment(pairs, p)
    a.info["pairNames"] = names
    return a


# ============================================================================ place looked at

def classify_place(place_test, geom, check, p=DEFAULT):
    """(visibility, source) of a map point (test frame) in one keyframe; the app's EvidenceRules.classify.
    Depth: the saved depth image; without one, the phone's own check of that entry in that keyframe."""
    if geom is None:
        return (check["vis"], "phone") if check else ("unknown", "none")
    pr = geom.project_map(place_test)
    if pr is None:
        return "out", "geometry"
    u, v, z = pr
    w, h = geom.k["width"], geom.k["height"]
    m = p.VIEW_MARGIN
    if not (m * w <= u <= (1 - m) * w and m * h <= v <= (1 - m) * h and p.VIEW_MIN_M <= z <= p.VIEW_MAX_M):
        return "out", "geometry"
    dm = geom.depth_map()
    if dm is not None:
        meas = dm.median_patch(u, v, p.DEPTH_PATCH)
        if meas is None:
            return "no_depth", "saved_depth"
        hidden = meas < z - max(p.HIDDEN_MIN_M, p.HIDDEN_FRACTION * z)
        return ("hidden" if hidden else "observable"), "saved_depth"
    if check is not None and check.get("vis") in ("observable", "hidden", "no_depth"):
        return check["vis"], "phone"
    return "no_depth", "none"


def looks(item, place_test, scan, p=DEFAULT):
    counts, sources = collections.Counter(), collections.Counter()
    for kf in scan.kf_list:
        vis, src = classify_place(place_test, scan.keyframes.get(kf), scan.checks.get((kf, item.id)), p)
        counts[vis] += 1
        sources[src] += 1
    return counts, sources


def geometry_self_check(scan, entries):
    """Recompute the phone's logged checks (u, v, zM; measured depth where a depth file was saved) for
    entries the baseline never linked in this scan (their position did not change during it)."""
    linked = {(o.decision or {}).get("entry") for o in scan.observations}
    du, dz, dm, n, n_depth = 0.0, 0.0, 0.0, 0, 0
    for (kf, eid), c in scan.checks.items():
        e = entries.get(eid)
        g = scan.keyframes.get(kf)
        if e is None or g is None or eid in linked or c.get("u") is None:
            continue
        pr = g.project_map(e.get("mapPosition"))
        if pr is None:
            continue
        n += 1
        du = max(du, abs(pr[0] - c["u"]), abs(pr[1] - c["v"]))
        dz = max(dz, abs(pr[2] - c["zM"]))
        d = g.depth_map()
        if d is not None and c.get("measuredM") is not None:
            mm = d.median_patch(pr[0], pr[1], DEFAULT.DEPTH_PATCH)
            if mm is not None:
                n_depth += 1
                dm = max(dm, abs(mm - c["measuredM"]))
    return {"checks": n, "maxPixelDiff": round(du, 6), "maxDepthAlongAxisDiffM": round(dz, 9),
            "withSavedDepth": n_depth, "maxMeasuredDepthDiffM": round(dm, 6)}


# ============================================================================ the decision engine

class Decision:
    __slots__ = ("kind", "item", "how", "score", "dist", "candidates")

    def __init__(self, kind, item=None, how=None, score=None, distance=None, candidates=()):
        self.kind, self.item, self.how, self.score, self.dist = kind, item, how, score, distance
        self.candidates = list(candidates)

    def text(self):
        if self.kind == UPDATE:
            return "update %s" % self.item
        if self.kind == UNRESOLVED:
            return "unresolved %s" % ",".join(self.candidates)
        if self.kind in (DROP, NONE):
            return self.kind
        return "add"

    def as_dict(self):
        return {"decision": self.kind, "item": self.item, "how": self.how,
                "score": None if self.score is None else round(self.score, 4),
                "distM": None if self.dist is None else round(self.dist, 4), "candidates": self.candidates}


def score_matrix(sightings, items):
    S = {}
    for s in sightings:
        row = {}
        for it in items:
            row[it.id] = max((_dot(o.emb, r) for o in s.obs for r in it.refs if o.emb is not None), default=None)
        S[s.id] = row
    return S


def run_engine(items, sightings, scan, p=DEFAULT, align=True):
    """The generic rules (a) to (f) over given items (generic or from a pack). Returns a dict with the
    decision per sighting, the belief per item, the alignment and the compute time."""
    t0 = time.perf_counter()
    by_i = {it.id: it for it in items}
    S = score_matrix(sightings, items)
    C = estimate_alignment(sightings, items, S, p) if align else Align(kind="off")
    pos = {s.id: C.apply(s.pos) for s in sightings}

    def sim(s, it):
        v = S[s.id][it.id]
        return -2.0 if v is None else v

    # (b, c) match in place, best pairs first; no item takes two sightings of one keyframe
    pairs = []
    for s in sightings:
        for it in items:
            d = dist(pos[s.id], it.place)
            if d <= it.radius and sim(s, it) >= it.thr_place:
                pairs.append((-sim(s, it), d, s.order, it.order, s, it))
    pairs.sort(key=lambda x: x[:4])
    dec, item_s, item_kf = {}, collections.defaultdict(list), collections.defaultdict(set)
    for negsim, d, _, _, s, it in pairs:
        if s.id in dec or (s.kfs & item_kf[it.id]):
            continue
        dec[s.id] = Decision(UPDATE, it.id, "in_place", -negsim, d)
        item_s[it.id].append(s.id)
        item_kf[it.id] |= s.kfs
    belief = {}
    for it in items:
        if item_s[it.id]:
            belief[it.id] = {"state": "seen_in_place", "sightings": list(item_s[it.id])}
    for it in items:
        if it.id not in belief:
            al = sorted(a for a in it.aliases if a in item_s)
            if al:
                belief[it.id] = {"state": "seen_in_place", "via": al}
    # place looked at? (the app's evidence rule, at the corrected place)
    extra_at = collections.defaultdict(list)
    for s in sightings:
        if s.extra:
            for it in items:
                if dist(pos[s.id], it.place) <= it.radius:
                    extra_at[it.id].append(s.id)
    for it in items:
        if it.id in belief:
            continue
        counts, sources = looks(it, C.inverse(it.place), scan, p)
        n_look = counts.get("observable", 0) + p.EXTRA_VIEW_LOOKS * len(extra_at[it.id])
        b = {"observableKeyframes": counts.get("observable", 0), "visibility": dict(counts),
             "depthSources": dict(sources), "extraViewLooks": len(extra_at[it.id])}
        app = [c.get("sim") for (kf, eid), c in scan.checks.items() if eid == it.id and c.get("sim") is not None]
        b["phoneAppearanceBest"] = max(app) if app else None
        if p.USE_PHONE_APPEARANCE_CHECK and app and max(app) >= p.SEEN_SIMILARITY:
            b["state"] = "seen_appearance"
        elif n_look >= p.MISSED_MIN_KEYFRAMES:
            b["state"] = "missed"
        else:
            b["state"] = "not_observed"
        belief[it.id] = b

    # (d, e, f) the remaining sightings
    def label_ok(s, it):
        return (not p.ELSEWHERE_SAME_LABEL) or s.extra or not s.labels or it.label in s.labels

    def pool(s, states, exclude=()):
        return sorted(((sim(s, it), it) for it in items if belief[it.id]["state"] in states and it.id not in exclude
                       and label_ok(s, it)), key=lambda x: (-x[0], x[1].order))

    rest = [s for s in sightings if s.id not in dec]
    rest.sort(key=lambda s: (-max([x for x, _ in pool(s, ("missed",))] or [-2.0]), s.order))
    moved = collections.OrderedDict()      # item -> sighting ids at its new place
    moved_kf = collections.defaultdict(set)
    for s in rest:
        if s.extra:
            # an extra view is a look at a place, not a detection: it never adds or moves an item
            dec[s.id] = Decision(NONE, None, "extra_view_no_match")
            continue
        # another sighting of an item already moved here
        join = None
        for iid, sids in moved.items():
            it = by_i[iid]
            if not (s.kfs & moved_kf[iid]) and dist(pos[s.id], pos[sids[0]]) <= it.radius and sim(s, it) >= it.thr_place:
                if join is None or sim(s, it) > join[0]:
                    join = (sim(s, it), it)
        if join is not None:
            dec[s.id] = Decision(UPDATE, join[1].id, "moved_fragment", join[0], dist(pos[s.id], join[1].place))
            moved[join[1].id].append(s.id)
            moved_kf[join[1].id] |= s.kfs
            continue
        M = pool(s, ("missed",), exclude=moved)
        cands = pool(s, ("missed", "not_observed"), exclude=moved)
        if M:
            top, it = M[0]
            others = [x for x, j in cands if j is not it and j.id not in it.aliases]
            runner = others[0] if others else None
            if it.thr_moved is not None and top >= it.thr_moved and (runner is None or top - runner >= p.MOVED_MARGIN):
                dec[s.id] = Decision(UPDATE, it.id, "moved", top, dist(pos[s.id], it.place))
                moved[it.id] = [s.id]
                moved_kf[it.id] |= s.kfs
                continue
        listed = [j.id for x, j in cands if x >= p.SIM_UNRESOLVED][:p.UNRESOLVED_MAX_CANDIDATES]
        if listed:
            dec[s.id] = Decision(UNRESOLVED, None, "ambiguous", cands[0][0], None, listed)
        else:
            dec[s.id] = Decision(ADD, None, "new", cands[0][0] if cands else None)
    for iid, sids in moved.items():
        belief[iid]["state"] = "moved"
        belief[iid]["to"] = sids
        n = len(sids)
        belief[iid]["toPosition"] = [round(sum(pos[x][i] for x in sids) / n, 4) for i in range(3)]
    for s in sightings:
        if dec[s.id].kind == UNRESOLVED:
            for iid in dec[s.id].candidates:
                if belief[iid]["state"] in ("missed", "not_observed"):
                    belief[iid].setdefault("candidateOf", []).append(s.id)
    for b in belief.values():
        if b["state"] in ("missed", "not_observed") and b.get("candidateOf"):
            b["stateBeforeUnresolved"] = b["state"]
            b["state"] = "unresolved"
    return {"decisions": dec, "beliefs": belief, "alignment": C,
            "positions": pos, "ms": (time.perf_counter() - t0) * 1000.0}


# ============================================================================ the household pack

def build_pack(history, p=DEFAULT):
    """Everything the household method needs, from the history scans only. Returns the pack (dict)."""
    t0 = time.perf_counter()
    obs = history.obs_by_id()
    scans = [s.scan for s in history.scans]
    retained = {s.scan: {o.id for o in s.observations if o.retained} for s in history.scans}
    entries = history.entries
    # history scans aligned to the first one, from the positions the entries had in each scan
    frames = {scans[0]: Align(kind="reference")}
    for k in range(1, len(scans)):
        pairs = []
        for e in entries.values():
            h = {x.get("scan"): x for x in e.get("history") or []}
            a = h.get(scans[k], {}).get("position")
            ref = next((frames[scans[j]].apply(h[scans[j]]["position"]) for j in range(k)
                        if h.get(scans[j], {}).get("position") is not None), None)
            if a is not None and ref is not None:
                pairs.append((a, ref))
        frames[scans[k]] = robust_alignment(pairs, p)

    def to_ref(scan, point):
        return frames[scan].apply(point) if scan in frames else tuple(point)

    items = []
    for eid, e in entries.items():
        own = entry_embedding(e)
        created = e.get("createdScan") if e.get("createdScan") in frames else scans[0]
        place = to_ref(created if (e.get("confirmed") or {}).get("source") != "user_verify" else scans[-1], entry_place(e))
        cand = []
        for ref in (e.get("confirmed") or {}).get("refs") or []:
            if ref in obs:
                cand.append(ref)
        positions = []
        for h in e.get("history") or []:
            sc_id = h.get("scan")
            if sc_id not in frames:
                continue
            if h.get("position") is not None:
                positions.append(to_ref(sc_id, h["position"]))
            for oid in h.get("obs") or []:
                if oid in retained.get(sc_id, ()) and oid not in cand:
                    cand.append(oid)
        refs, keys, dropped = [], set(), 0
        if own is not None:
            refs.append(own)
            keys.add(tuple(round(x, 6) for x in own[:8]))
        for oid in cand:
            v = obs[oid].emb
            if v is None:
                continue
            k = tuple(round(x, 6) for x in v[:8])
            if k in keys:
                continue
            if own is not None and _dot(v, own) < p.REF_CONSISTENCY_SIM:
                dropped += 1
                continue
            refs.append(v)
            keys.add(k)
        spread = None
        if len(positions) >= 2:
            c = [sum(q[i] for q in positions) / len(positions) for i in range(3)]
            spread = max(dist(q, c) for q in positions)
        pair_sims = [_dot(refs[a], refs[b]) for a in range(len(refs)) for b in range(a + 1, len(refs))]
        items.append({"id": eid, "label": e.get("label"), "place": list(place), "refs": refs,
                      "refsDropped": dropped, "self": median(pair_sims), "spread": spread,
                      "historyPositions": len(positions), "linked": {o for h in e.get("history") or [] for o in h.get("obs") or []}})
    # look-alike groups and aliases (same object, several entries)
    n = len(items)
    parent_g, parent_a = list(range(n)), list(range(n))

    def find(par, x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    lookalike_pairs = []
    for a in range(n):
        for b in range(a + 1, n):
            A, B = items[a], items[b]
            if not A["refs"] or not B["refs"]:
                continue
            cross = median([_dot(x, y) for x in A["refs"] for y in B["refs"]])
            sa = A["self"] if A["self"] is not None else p.SELF_FALLBACK
            sb = B["self"] if B["self"] is not None else p.SELF_FALLBACK
            if cross >= min(sa, sb):
                same_place = dist(A["place"], B["place"]) <= p.PLACE_RADIUS_M
                lookalike_pairs.append((A["id"], B["id"], round(cross, 4), same_place))
                parent_g[find(parent_g, a)] = find(parent_g, b)
                if same_place:
                    parent_a[find(parent_a, a)] = find(parent_a, b)
    groups = collections.defaultdict(list)
    alias_sets = collections.defaultdict(list)
    for k in range(n):
        groups[find(parent_g, k)].append(k)
        alias_sets[find(parent_a, k)].append(k)
    group_of = {}
    for gi, (root, members) in enumerate(sorted(groups.items(), key=lambda kv: min(kv[1]))):
        for k in members:
            group_of[k] = gi
    # impostors: history observations FAR_M or more from the item's place, not linked to its group
    hist_obs = [(o, to_ref(o.scan, o.map_point)) for s in history.scans for o in s.observations if o.emb is not None]
    for k, it in enumerate(items):
        members = {items[j]["id"] for j in groups[find(parent_g, k)]}
        linked = set().union(*[items[j]["linked"] for j in groups[find(parent_g, k)]])
        imp = None
        for o, q in hist_obs:
            if dist(q, it["place"]) < p.FAR_M or o.id in linked or (o.decision or {}).get("entry") in members:
                continue
            for r in it["refs"]:
                v = _dot(o.emb, r)
                if imp is None or v > imp:
                    imp = v
        it["impostor"] = imp
        stable = it["spread"] is not None and it["spread"] <= p.STABLE_SPREAD_M
        if imp is None:
            thr = p.SIM_MOVED
        elif imp + p.HH_IMPOSTOR_MARGIN > p.HH_THR_MAX:
            thr = None
        else:
            thr = max(p.HH_THR_MIN, imp + p.HH_IMPOSTOR_MARGIN)
        if thr is not None and stable:
            thr = thr + p.STABLE_MOVED_BONUS
            thr = None if thr > p.HH_THR_MAX else thr
        it["thrItem"], it["stable"] = thr, stable
        it["radius"] = p.PLACE_RADIUS_M if it["spread"] is None else \
            min(p.PLACE_RADIUS_MAX_M, max(p.PLACE_RADIUS_M, it["spread"] + p.STABILITY_PAD_M))
    for k, it in enumerate(items):
        thrs = [items[j]["thrItem"] for j in groups[find(parent_g, k)]]
        it["thrMoved"] = None if any(t is None for t in thrs) else max(thrs)
        it["aliases"] = sorted(items[j]["id"] for j in alias_sets[find(parent_a, k)] if j != k)
        it["group"] = group_of[k]
    pack = collections.OrderedDict([
        ("pack", 1), ("map", history.map_id), ("historyScans", scans),
        ("historyFrames", {s: frames[s].summary() for s in scans}),
        ("params", {k: getattr(p, k) for k in ("SIM_PLACE", "PLACE_RADIUS_M", "SIM_MOVED", "MOVED_MARGIN",
                                                "SIM_UNRESOLVED", "UNRESOLVED_MAX_CANDIDATES", "ELSEWHERE_SAME_LABEL",
                                                "MISSED_MIN_KEYFRAMES", "PACK_EMBEDDING")}),
        ("stabilityPrior", any(it["spread"] is not None for it in items)),
        ("lookalikePairs", lookalike_pairs),
        ("items", [collections.OrderedDict([
            ("id", it["id"]), ("label", it["label"]), ("place", [round(x, 5) for x in it["place"]]),
            ("radius", round(it["radius"], 4)), ("thrPlace", p.SIM_PLACE),
            ("thrMoved", None if it["thrMoved"] is None else round(it["thrMoved"], 4)),
            ("thrItem", None if it["thrItem"] is None else round(it["thrItem"], 4)),
            ("impostorMax", None if it["impostor"] is None else round(it["impostor"], 4)),
            ("self", None if it["self"] is None else round(it["self"], 4)),
            ("spreadM", None if it["spread"] is None else round(it["spread"], 4)), ("stable", it["stable"]),
            ("group", it["group"]), ("aliases", it["aliases"]), ("refsDropped", it["refsDropped"]),
            ("refs", [encode_f16(r) for r in it["refs"]])]) for it in items]),
    ])
    pack["buildMs"] = round((time.perf_counter() - t0) * 1000.0, 2)
    return pack


def write_pack(pack, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pack, f, separators=(",", ":"))
    return os.path.getsize(path)


def pack_items(pack):
    """Items as the household method sees them: only from the pack."""
    out = []
    for k, d in enumerate(pack["items"]):
        out.append(Item(d["id"], k, d["label"], d["place"], [unit(decode_f16(r)) for r in d["refs"]],
                        d["thrPlace"], d["thrMoved"], d["radius"], d.get("aliases") or (), d.get("group")))
    return out


# ============================================================================ the current matcher

def current_method(scan, known_ids):
    """Decisions per observation from the logged `decision` fields; beliefs from inv_evidence."""
    dec = {}
    for o in scan.observations:
        d = o.decision or {}
        e = d.get("entry")
        if e in scan.dropped:
            dec[o.id] = Decision(DROP, e, d.get("kind"))
        elif e in scan.created or e not in known_ids:
            dec[o.id] = Decision(ADD, e, d.get("kind"))
        else:
            dec[o.id] = Decision(UPDATE, e, d.get("kind"), d.get("sim"), d.get("distM"))
    beliefs = {}
    for eid in known_ids:
        out = scan.evidence.get(eid)
        beliefs[eid] = {"state": {"seen": "seen", "missed": "missed", "not_observed": "not_observed"}.get(out, "unknown")}
    return {"obs": dec, "beliefs": beliefs}


# ============================================================================ per observation, disagreement

def obs_decisions(result, sightings):
    """Every observation inherits its sighting's decision; adds and unresolved get the sighting as entity."""
    out = {}
    for s in sightings:
        d = result["decisions"][s.id]
        for o in s.obs:
            out[o.id] = (d, s.id)
    return out


def decision_key(d):
    if d.kind == UPDATE:
        return "update " + str(d.item)
    if d.kind == UNRESOLVED:
        return "unresolved"
    return d.kind


def sighting_view(method_obs, s):
    keys = collections.Counter(decision_key(method_obs[o.id][0]) for o in s.obs if o.id in method_obs)
    return keys.most_common(1)[0][0] if len(keys) == 1 else "mixed(" + ",".join(sorted(keys)) + ")"


# ============================================================================ labels and scoring

def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_labels(path, key):
    out = {}
    for r in read_csv(path):
        k, v = (r.get(key) or "").strip(), (r.get("item") or "").strip()
        if k:
            out[k] = v or "unknown"
    return out


def entry_item_map(entries, obs_labels, entry_labels=None, linked_extra=None):
    """entry -> physical item: the entry label file first; else the majority item label of the
    observations the matcher linked to the entry (history items and confirmed refs); None on a tie."""
    out, how = {}, collections.Counter()
    for eid, e in entries.items():
        if entry_labels and entry_labels.get(eid) and entry_labels[eid] != "unknown":
            out[eid] = entry_labels[eid]
            how["entry label"] += 1
            continue
        linked = set((e.get("confirmed") or {}).get("refs") or [])
        for h in e.get("history") or []:
            linked |= set(h.get("obs") or [])
        linked |= set((linked_extra or {}).get(eid, ()))
        votes = collections.Counter(obs_labels[o] for o in linked if obs_labels.get(o) not in (None, "", "unknown"))
        if not votes:
            out[eid] = None
            how["no labelled observation"] += 1
            continue
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            out[eid] = None
            how["tie"] += 1
        else:
            out[eid] = top[0][0]
            how["majority" if len(top) > 1 else "unanimous"] += 1
    return out, how


def yes(v):
    return str(v or "").strip().lower() in ("yes", "y", "true", "1")


def no(v):
    return str(v or "").strip().lower() in ("no", "n", "false", "0")


def conditions(items, known, sheet_rows, n_hist, t_index):
    """Condition of every physical item in test scan t (1-based) from the sheet: stays, moved,
    not_visited, absent, new_same_model, new_to_inventory; without a sheet row: known / new (no sheet)."""
    rows = {(r.get("item_code") or "").strip(): r for r in sheet_rows or []}
    out = {}
    last_h = "h%d" % n_hist
    t = "t%d" % t_index
    for x in items:
        r = rows.get(x)
        if r is None:
            out[x] = "known (no sheet)" if x in known else "new (no sheet)"
            continue
        if x not in known:
            hist_absent = all(no(r.get("h%d_present" % k)) for k in range(1, n_hist + 1))
            out[x] = "new_same_model" if (r.get("same_model_group") or "").strip() and hist_absent else "new_to_inventory"
            continue
        if no(r.get(t + "_visited")):
            out[x] = "not_visited"
        elif no(r.get(t + "_present")):
            out[x] = "absent"
        else:
            a, b = (r.get(last_h + "_place_code") or "").strip(), (r.get(t + "_place_code") or "").strip()
            moved = (a != b) if (a and b) else yes(r.get(t + "_moved"))
            out[x] = "moved" if moved else "stays"
    return out


OUTCOMES = ("correct_update", "wrong_merge", "duplicate", "unresolved", "correct_add", "kept_not_observed",
            "dropped", "unscorable")


def score_method(obs_dec, beliefs, test_obs_ids, obs_labels, emap, item_universe):
    """One outcome per physical item. obs_dec: obs id -> (Decision, entity id for add/unresolved).
    Wrong merge: an entity (an existing entry, or a new entry/sighting) holds observations of two items,
    or an existing entry of item X takes observations of another item."""
    lab = {o: obs_labels.get(o) for o in test_obs_ids}
    excluded = collections.Counter("unlabelled" if v is None else "unknown" for v in lab.values()
                                   if v in (None, "", "unknown"))
    ent_items = collections.defaultdict(set)
    ent_kind = {}
    by_item = collections.defaultdict(list)
    for o in test_obs_ids:
        x = lab[o]
        if x in (None, "", "unknown") or o not in obs_dec:
            continue
        d, entity = obs_dec[o]
        by_item[x].append(d)
        if d.kind == UPDATE:
            key = ("entry", d.item)
            ent_items[key].add(x)
            if emap.get(d.item):
                ent_items[key].add(emap[d.item])
        elif d.kind == ADD:
            key = ("new", entity)
            ent_items[key].add(x)
        else:
            continue
        ent_kind[key] = d.kind
    wrong = set()
    for key, xs in ent_items.items():
        if len(xs) > 1:
            wrong |= xs
    out = {}
    known_items = {v for v in emap.values() if v}
    for x in item_universe:
        known = x in known_items
        ds = by_item.get(x, [])
        if x in wrong:
            out[x] = "wrong_merge"
        elif not ds:
            out[x] = "kept_not_observed" if known else None
        elif known and any(d.kind == ADD for d in ds):
            out[x] = "duplicate"
        elif any(d.kind == UNRESOLVED for d in ds):
            out[x] = "unresolved"
        elif known and any(d.kind == UPDATE and emap.get(d.item) == x for d in ds):
            out[x] = "correct_update"
        elif not known and any(d.kind == ADD for d in ds) and not any(d.kind == UPDATE for d in ds):
            out[x] = "correct_add"
        elif all(d.kind == DROP for d in ds):
            out[x] = "dropped"
        else:
            out[x] = "unscorable"
    return out, excluded


def item_belief(x, emap, beliefs):
    """The most positive belief over the item's history entries."""
    rank = ["seen_in_place", "seen", "seen_appearance", "moved", "unresolved", "missed", "not_observed", "unknown"]
    states = [beliefs[e]["state"] for e, v in emap.items() if v == x and e in beliefs]
    return min(states, key=lambda s: rank.index(s) if s in rank else len(rank)) if states else None


# ============================================================================ provenance

def git_info():
    def run(*args):
        try:
            return subprocess.run(["git"] + list(args), cwd=REPO, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    path = os.path.abspath(__file__)
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    status = run("status", "--porcelain", "--", os.path.relpath(path, REPO))
    return {"commit": run("rev-parse", "HEAD"), "scriptSha256": sha,
            "scriptUncommitted": bool(status) if status is not None else None}


def param_lines(p):
    return ["  %-28s %s" % (k, json.dumps(getattr(p, k))) for k in PARAMETERS]


# ============================================================================ one test scan

def run_test_scan(pull, history, test_id, p, align_modes, out_dir, labels=None, entry_labels=None, sheet=None,
                  t_index=1, hist_pack=None, pack_info=None):
    scan = load_scan(pull, test_id)
    extra = load_extra_views(pull, scan)
    sightings = build_sightings(scan, p)
    sightings_x = build_sightings(scan, p, extra)
    known_ids = list(history.entries)
    res = {"scan": test_id, "observations": len(scan.observations), "sightings": len(sightings),
           "extraViews": [x.id for x in extra]}
    res["geometrySelfCheck"] = geometry_self_check(scan, history.entries)
    methods = collections.OrderedDict()
    cur = current_method(scan, set(known_ids))
    methods["current"] = {"obs": {o: (d, d.item) for o, d in cur["obs"].items()}, "beliefs": cur["beliefs"],
                          "ms": None, "decisions": None}
    g_items = generic_items(history, p)
    for mode in align_modes:
        suffix = "" if mode == align_modes[0] else "_" + ("noalign" if not mode else "align")
        pp = Params(**dict(p.as_dict(), ALIGN=mode))
        r = run_engine(g_items, sightings, scan, pp, align=mode)
        r["obs"] = obs_decisions(r, sightings)
        methods["generic" + suffix] = r
        items_h = pack_items(hist_pack)
        r = run_engine(items_h, sightings, scan, pp, align=mode)
        r["obs"] = obs_decisions(r, sightings)
        methods["household" + suffix] = r
        if mode == align_modes[0]:
            r = run_engine(items_h, sightings_x, scan, pp, align=mode)
            r["obs"] = obs_decisions(r, sightings_x)
            methods["extra_view"] = r
    res["methods"] = methods
    res["sightingObjs"] = sightings
    res["sightingObjsX"] = sightings_x
    res["extra"] = extra
    res["scanData"] = scan
    res["pack"] = pack_info
    return res


# ============================================================================ report

def fmt_counts(c, keys=None):
    keys = keys or sorted(c)
    return ", ".join("%s %d" % (k, c.get(k, 0)) for k in keys)


def summarise(res, p, labels, entry_labels, sheet_rows, history, n_hist, t_index, out, per_item=False):
    """Print the report of one test scan; return the JSON-ready part."""
    pr = lambda *a: print(*a, file=out)  # noqa: E731
    scan = res["scanData"]
    sightings = res["sightingObjs"]
    methods = res["methods"]
    J = collections.OrderedDict()
    pr("\n== test scan %s (t%d): %d observations, %d sightings (%d of one observation), %d extra views after it"
       % (scan.scan, t_index, len(scan.observations), len(sightings), sum(1 for s in sightings if len(s.obs) == 1),
          len(res["extra"])))
    sizes = collections.Counter(len(s.obs) for s in sightings)
    pr("  sighting sizes (observations: sightings): %s" % ", ".join("%d: %d" % kv for kv in sorted(sizes.items())))
    g = res["geometrySelfCheck"]
    pr("  geometry self-check against the phone's logged checks (n = %d, entries the matcher never linked): max "
       "pixel difference %.3g, max depth-along-axis difference %.3g m; with a saved depth image n = %d, max "
       "measured-depth difference %.3g m" % (g["checks"], g["maxPixelDiff"], g["maxDepthAlongAxisDiffM"],
                                               g["withSavedDepth"], g["maxMeasuredDepthDiffM"]))
    J["geometrySelfCheck"] = g
    J["sightings"] = [{"id": s.id, "labels": dict(s.labels), "obs": [o.id for o in s.obs], "extra": s.extra,
                       "pos": [round(x, 4) for x in s.pos]} for s in res["sightingObjsX"]]
    # alignment
    for name, m in methods.items():
        if name == "current":
            continue
        a = m["alignment"]
        i = a.info
        pr("  alignment, %-18s %s: yaw %.2f deg, t (%.3f, %.3f, %.3f) m; %s pairs, %s agree; median residual %s -> %s m%s"
           % (name, a.kind, a.yaw_deg, a.t[0], a.t[1], a.t[2], i.get("pairs", "-"), i.get("inliers", "-"),
              i.get("medianResidualBeforeM", "-"), i.get("medianResidualAfterM", "-"),
              ("; " + i["reason"]) if i.get("reason") else ""))
    J["alignment"] = {n: m["alignment"].summary() for n, m in methods.items() if n != "current"}
    # decisions
    J["decisions"] = {}
    pr("  decisions per sighting (n = %d):" % len(sightings))
    for name, m in methods.items():
        if name == "current":
            c = collections.Counter(d.kind for d, _ in m["obs"].values())
            pr("    %-18s per observation (n = %d): %s" % (name, len(m["obs"]), fmt_counts(c, (UPDATE, ADD, DROP))))
            J["decisions"][name] = {o: d.as_dict() for o, (d, _) in m["obs"].items()}
            continue
        sl = res["sightingObjsX"] if name == "extra_view" else sightings
        c = collections.Counter((m["decisions"][s.id].kind + (" " + m["decisions"][s.id].how if
                                                              m["decisions"][s.id].kind == UPDATE else "")) for s in sl)
        co = collections.Counter(d.kind for d, _ in m["obs"].values())
        pr("    %-18s %s | per observation: %s" % (name, fmt_counts(c), fmt_counts(co, (UPDATE, ADD, UNRESOLVED))))
        J["decisions"][name] = {s.id: m["decisions"][s.id].as_dict() for s in sl}
    # single-observation sightings
    singles = [s for s in sightings if len(s.obs) == 1]
    pr("  sightings of one observation (n = %d), by method:" % len(singles))
    for name, m in methods.items():
        c = collections.Counter(decision_key(m["obs"][s.obs[0].id][0]).split(" ")[0] for s in singles)
        pr("    %-18s %s" % (name, fmt_counts(c)))
    # beliefs
    J["beliefs"] = {}
    pr("  beliefs about the %d known items:" % len(history.entries))
    for name, m in methods.items():
        c = collections.Counter(b["state"] for b in m["beliefs"].values())
        pr("    %-18s %s" % (name, fmt_counts(c)))
        J["beliefs"][name] = m["beliefs"]
    # disagreement
    names = [n for n in methods]
    pr("  agreement per observation (same decision and same updated item; n = %d observations):" % len(scan.observations))
    J["agreement"] = {}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            A, B = methods[names[a]]["obs"], methods[names[b]]["obs"]
            ids = [o.id for o in scan.observations if o.id in A and o.id in B]
            same = sum(1 for o in ids if decision_key(A[o][0]) == decision_key(B[o][0]))
            pr("    %-18s vs %-18s agree %d, disagree %d" % (names[a], names[b], same, len(ids) - same))
            J["agreement"]["%s|%s" % (names[a], names[b])] = {"agree": same, "disagree": len(ids) - same}
    rows = []
    for s in sightings:
        views = {n: sighting_view(methods[n]["obs"], s) for n in names}
        if len(set(views.values())) > 1:
            rows.append((s, views))
    pr("  sightings on which the methods disagree: %d of %d" % (len(rows), len(sightings)))
    for s, views in rows:
        pr("    %-14s %-10s n=%d  %s" % (s.id.split("-", 1)[1], s.label, len(s.obs),
                                         " | ".join("%s: %s" % (n, v) for n, v in views.items())))
    J["disagreements"] = [{"sighting": s.id, "label": s.label, "n": len(s.obs), "methods": v} for s, v in rows]
    # extra views
    if "extra_view" in methods:
        hh, xv = methods["household"], methods["extra_view"]
        ch = collections.Counter()
        for s in sightings:
            a, b = hh["decisions"][s.id], xv["decisions"][s.id]
            if a.text() != b.text():
                ch["%s -> %s" % (a.kind, b.kind)] += 1
        xs = [s for s in res["sightingObjsX"] if s.extra]
        pr("  extra view: %d extra views after this scan; household -> extra_view changed %d of %d sighting decisions%s"
           % (len(xs), sum(ch.values()), len(sightings), (": " + fmt_counts(ch)) if ch else ""))
        for s in xs:
            o = s.obs[0]
            pr("    extra view %s (query %s, queried entry %s): its decision %s; request to capture %s ms, whole query %s ms"
               % (o.id, o.query, o.entry, xv["decisions"][s.id].text(), o.ms_to_capture, o.decision.get("wholeMs")))
        caps = [s.obs[0].ms_to_capture for s in xs if s.obs[0].ms_to_capture is not None]
        whole = [s.obs[0].decision.get("wholeMs") for s in xs if s.obs[0].decision.get("wholeMs") is not None]
        if caps:
            pr("    request to capture: median %.0f ms, range %d-%d (n = %d); whole query: median %s ms, range %s (n = %d)"
               % (median(caps), min(caps), max(caps), len(caps), "%.0f" % median(whole) if whole else "-",
                  "%d-%d" % (min(whole), max(whole)) if whole else "-", len(whole)))
        J["extraView"] = {"n": len(xs), "changes": dict(ch)}
    # cost
    pr("  cost:")
    J["cost"] = {}
    for name, m in methods.items():
        if name == "current":
            pr("    %-18s 0 sent to a server; on the phone during the scan" % name)
            continue
        un = [d for d in m["decisions"].values() if d.kind == UNRESOLVED]
        cands = {c for d in un for c in d.candidates}
        pk = res["pack"] if name.startswith("household") or name == "extra_view" else None
        pr("    %-18s unresolved sightings %d (would be sent to a server), distinct candidate items %d; decide %.1f ms%s"
           % (name, len(un), len(cands), m["ms"], ("; pack %d bytes, built in %.1f ms" % (pk["bytes"], pk["buildMs"])) if pk else ""))
        J["cost"][name] = {"unresolved": len(un), "candidateItems": len(cands), "decideMs": round(m["ms"], 2),
                           "packBytes": pk["bytes"] if pk else None, "packBuildMs": pk["buildMs"] if pk else None}
    # scoring
    if labels is not None:
        J["scoring"] = score_report(res, labels, entry_labels, sheet_rows, history, n_hist, t_index, out, per_item)
    return J


def score_report(res, labels, entry_labels, sheet_rows, history, n_hist, t_index, out, per_item=False):
    pr = lambda *a: print(*a, file=out)  # noqa: E731
    scan = res["scanData"]
    methods = res["methods"]
    emap, how = entry_item_map(history.entries, labels, entry_labels)
    test_ids = [o.id for o in scan.observations]
    known = {v for v in emap.values() if v}
    hist_ids = [o.id for s in history.scans for o in s.observations]
    universe = sorted({labels[o] for o in test_ids + hist_ids if labels.get(o) not in (None, "", "unknown")}
                      | known | {(r.get("item_code") or "").strip() for r in sheet_rows or [] if r.get("item_code")})
    cond = conditions(universe, known, sheet_rows, n_hist, t_index)
    n_lab = sum(1 for o in test_ids if labels.get(o) not in (None, "", "unknown"))
    n_unk = sum(1 for o in test_ids if labels.get(o) == "unknown")
    n_none = sum(1 for o in test_ids if o not in labels)
    pr("  scoring: test observations labelled with an item %d, labelled unknown %d, without a label %d (the last two "
       "are excluded); history entries mapped to an item %d of %d (%s)"
       % (n_lab, n_unk, n_none, len([e for e in emap.values() if e]), len(emap), fmt_counts(how)))
    J = {"labelled": n_lab, "unknown": n_unk, "unlabelled": n_none, "entryMap": dict(how), "byMethod": {}}
    # rule (c) assumes one detection per object and keyframe: count where the labels say otherwise
    per_kf = collections.Counter((o.kf, labels[o.id]) for o in scan.observations
                                 if labels.get(o.id) not in (None, "", "unknown"))
    double = sorted((kf, x) for (kf, x), c in per_kf.items() if c > 1)
    pr("  keyframes in which two labelled observations show the same item (breaks 'two detections in one keyframe "
       "are two objects'): %d (items: %d)" % (len(double), len({x for _, x in double})))
    J["sameKeyframeSameItem"] = len(double)
    cond_n = collections.Counter(c for x, c in cond.items() if x in known or any(labels.get(o) == x for o in test_ids))
    order = ["stays", "moved", "not_visited", "absent", "new_same_model", "new_to_inventory", "known (no sheet)",
             "new (no sheet)"]
    present = [c for c in order if cond_n.get(c)]
    pr("  items per condition: %s" % ", ".join("%s n = %d" % (c, cond_n[c]) for c in present))
    J["conditions"] = dict(cond_n)
    per = {}
    for name, m in methods.items():
        outcome, excluded = score_method(m["obs"], m["beliefs"], test_ids, labels, emap, universe)
        per[name] = outcome
        table = collections.defaultdict(collections.Counter)
        for x, o in outcome.items():
            if o is not None:
                table[cond[x]][o] += 1
        pr("    %s:" % name)
        for c in present:
            pr("      %-18s n = %-3d %s" % (c, sum(table[c].values()),
                                            ", ".join("%s %d" % (k, v) for k, v in sorted(table[c].items()))))
        bel = collections.defaultdict(collections.Counter)
        for x in universe:
            if x in known:
                bel[cond[x]][item_belief(x, emap, m["beliefs"])] += 1
        pr("      beliefs about the known items' entries: %s" % "; ".join(
            "%s: %s" % (c, ", ".join("%s %d" % kv for kv in sorted(bel[c].items()))) for c in present if bel.get(c)))
        J["byMethod"][name] = {c: dict(table[c]) for c in present}
        J["byMethod"][name + "_beliefs"] = {c: dict(bel[c]) for c in present if bel.get(c)}
    bel_item = {n: {x: item_belief(x, emap, methods[n]["beliefs"]) for x in universe} for n in methods}
    J["perItem"] = {x: {"condition": cond[x], **{n: [per[n][x], bel_item[n][x]] for n in per}} for x in universe}
    if per_item:
        pr("    per item (private; outcome / belief about its entries):")
        for x in universe:
            pr("      %-28s %-18s %s" % (x, cond[x], " | ".join("%s %s/%s" % (n, per[n][x], bel_item[n][x]) for n in per)))
    return J


# ============================================================================ main

def to_jsonable(x):
    if isinstance(x, Align):
        return x.summary()
    if isinstance(x, Decision):
        return x.as_dict()
    if isinstance(x, set):
        return sorted(x)
    if isinstance(x, tuple):
        return list(x)
    return str(x)


def main(argv=None, out=sys.stdout):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", nargs="?", help="pulled data folder")
    ap.add_argument("--history", default=None, help="history scan ids in order, comma-separated")
    ap.add_argument("--test", default=None, help="test scan ids in order, comma-separated")
    ap.add_argument("--labels", default=None, help="CSV obs,item (default: <folder>/labels.private.csv if present)")
    ap.add_argument("--entry-labels", default=None, help="CSV entry,item (default: <folder>/entry_labels.private.csv)")
    ap.add_argument("--no-labels", action="store_true", help="ignore label files: the label-free report only")
    ap.add_argument("--sheet", default=None, help="filled ground-truth sheet (template paper/identity_pilot_ground_truth_template.csv)")
    ap.add_argument("--align", choices=("on", "off", "both"), default="on")
    ap.add_argument("--out", default=None, help="result folder (default analysis/out/identity_replay/<folder name>)")
    ap.add_argument("--per-item", action="store_true", help="print the per-item outcomes (item codes)")
    ap.add_argument("--freeze", default=None, help="write the parameter set with the commit to this file and stop")
    ap.add_argument("--frozen", default=None, help="refuse to run unless the parameters equal this frozen file")
    args = ap.parse_args(argv)
    p = DEFAULT
    gi = git_info()
    if args.freeze:
        os.makedirs(os.path.dirname(os.path.abspath(args.freeze)), exist_ok=True)
        with open(args.freeze, "w", encoding="utf-8") as f:
            json.dump({"params": p.as_dict(), "sources": {k: v[1] for k, v in PARAMETERS.items()}, "git": gi,
                       "frozenAt": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=1)
        print("Parameters written to %s (commit %s%s)" % (args.freeze, gi["commit"],
                                                          ", script has uncommitted changes" if gi["scriptUncommitted"] else ""), file=out)
        return 0
    if args.frozen:
        with open(args.frozen, encoding="utf-8") as f:
            fz = json.load(f)
        diff = {k: (fz["params"].get(k), v) for k, v in p.as_dict().items() if fz["params"].get(k) != v}
        if diff:
            print("Parameters differ from the frozen set %s: %s" % (args.frozen, diff), file=out)
            return 3
        frozen_sha = (fz.get("git") or {}).get("scriptSha256")
        if frozen_sha and frozen_sha != gi["scriptSha256"]:
            print("NOTE: the parameters equal the frozen set, but identity_replay.py changed since it was frozen "
                  "(commit then %s): report any change of rules as a change" % (fz.get("git") or {}).get("commit"), file=out)
        gi["frozen"] = {"file": args.frozen, "sameParams": True, "sameScript": frozen_sha == gi["scriptSha256"]}
    if not args.folder or not os.path.isdir(args.folder):
        ap.error("no data folder")
    pull = sc.Pull(args.folder)
    study = pull.study_scans()
    hist = [s for s in (args.history or "").split(",") if s] or study[:-1]
    tests = [s for s in (args.test or "").split(",") if s] or study[-1:]
    if not hist or not tests:
        ap.error("need at least one history scan and one test scan (study scans here: %s)" % ", ".join(study))
    labels_path = args.labels or (os.path.join(args.folder, "labels.private.csv")
                                  if os.path.exists(os.path.join(args.folder, "labels.private.csv")) else None)
    elab_path = args.entry_labels or (os.path.join(args.folder, "entry_labels.private.csv")
                                      if os.path.exists(os.path.join(args.folder, "entry_labels.private.csv")) else None)
    if args.no_labels:
        labels_path = elab_path = None
    labels = read_labels(labels_path, "obs") if labels_path else None
    entry_labels = read_labels(elab_path, "entry") if elab_path else None
    sheet_rows = read_csv(args.sheet) if args.sheet else None
    out_dir = args.out or os.path.join(HERE, "out", "identity_replay", os.path.basename(os.path.normpath(args.folder)))
    os.makedirs(out_dir, exist_ok=True)
    align_modes = {"on": [True], "off": [False], "both": [True, False]}[args.align]

    print("Identity replay of %s" % args.folder, file=out)
    print("History scans %s; test scans %s; commit %s%s" % (",".join(hist), ",".join(tests), gi["commit"],
          " (identity_replay.py has uncommitted changes)" if gi["scriptUncommitted"] else ""), file=out)
    print("Labels: %s; entry labels: %s; ground-truth sheet: %s" % (
        "%d observations" % len(labels) if labels is not None else "none",
        "%d entries" % len(entry_labels) if entry_labels is not None else "none",
        "%d rows" % len(sheet_rows) if sheet_rows is not None else "none"), file=out)
    print("Parameters:", file=out)
    for line in param_lines(p):
        print(line, file=out)
    history = load_history(pull, hist)
    pack = build_pack(history, p)
    result = collections.OrderedDict([("folder", os.path.abspath(args.folder)), ("history", hist), ("tests", tests),
                                      ("git", gi), ("params", p.as_dict()), ("argv", argv if argv is not None else sys.argv[1:]),
                                      ("labels", labels_path), ("entryLabels", elab_path), ("sheet", args.sheet),
                                      ("historyEntries", len(history.entries)), ("testScans", [])])
    print("History: %d known items (entries alive at the end of %s); stability prior: %s" % (
        len(history.entries), hist[-1], "yes" if pack["stabilityPrior"] else
        "none (one history scan, or no item placed in two history scans)"), file=out)
    groups = collections.Counter(it["group"] for it in pack["items"])
    print("Household pack: %d look-alike pairs (%d at the same place: one object, several entries), %d groups of "
          "two or more; %d items can only be unresolved (threshold over HH_THR_MAX); references per item median %s"
          % (len(pack["lookalikePairs"]), sum(1 for x in pack["lookalikePairs"] if x[3]),
             sum(1 for v in groups.values() if v > 1), sum(1 for it in pack["items"] if it["thrMoved"] is None),
             median([len(it["refs"]) for it in pack["items"]])), file=out)
    print("Extra views in the pull: %s" % ("; ".join("%s after scan %s (request to capture %s ms, whole query %s ms)"
                                                       % (x["obs"], x["after_scan"], x["ms_to_capture"], x["whole_ms"])
                                                       for x in all_extra_views(pull)) or "none"), file=out)
    for k, test_id in enumerate(tests):
        path = os.path.join(out_dir, "pack_%s.json" % test_id)
        t_w = time.perf_counter()
        size = write_pack(pack, path)
        write_ms = (time.perf_counter() - t_w) * 1000.0
        t_r = time.perf_counter()
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        info = {"path": path, "bytes": size, "buildMs": pack["buildMs"], "writeMs": round(write_ms, 2),
                "readMs": round((time.perf_counter() - t_r) * 1000.0, 2)}
        res = run_test_scan(pull, history, test_id, p, align_modes, out_dir, labels, entry_labels, sheet_rows,
                            k + 1, loaded, info)
        J = summarise(res, p, labels, entry_labels, sheet_rows, history, len(hist), k + 1, out, args.per_item)
        J["scan"], J["pack"] = test_id, info
        result["testScans"].append(J)
    rpath = os.path.join(out_dir, "result.json")
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, default=to_jsonable)
    print("\nResult file (private: item codes and positions): %s" % rpath, file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
