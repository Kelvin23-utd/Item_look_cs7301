#!/usr/bin/env python3
"""Replay of the app's scan merging, driven only by the recorded inputs, against the app's own log.

    python3 analysis/scan_replay.py backups/phone_files_2026-11-17_0008
    analysis/.venv-bench/bin/python analysis/scan_replay.py backups/phone_files_2026-11-17_0008 --offline-embeddings

What is re-implemented (FindBack `scan/ScanSession.kt` `merge`, `scan/Merger.kt`, constants of
`scan/ScanConfig.kt`; v0.5 Block A, unchanged up to v0.7):

* the kept detections of a keyframe are merged one by one in descending score order; the sort is
  stable, so equal scores keep the order of the detector's output (the order of the saved keyframe
  JSON, which only leaves out `person` reports);
* candidates are the real entries of the map (store order: order of promotion) followed by the
  tentative entries (order of creation), less the entries already used in this keyframe;
* a candidate qualifies when its distance from the detection's map point is at most
  clamp(0.5 x max(entry size, detection size) + 0.15, 0.20, 0.50) m, and its winning label equals the
  detection's label or the cosine similarity of the two embeddings is at least 0.55 (float32);
* the nearest qualifying candidate wins (strict "<", so on equal distances the first in candidate
  order wins); no winner: a new tentative entry;
* a tentative entry matched from a keyframe at least 300 ms after its first sighting is promoted
  (kind `promoted`), else it stays tentative (kind `tentative`); a real entry matched is `known`;
* every match and every new entry is "used": one detection per entry per keyframe;
* a sighting appends (time, map point, size, label, score), keeps the latest 10, sets the position to
  their mean, adds the score to the label's vote (the current label stays on a tie), and replaces the
  best crop (size, embedding, quality = box area x score, both float32) when the new quality is higher;
* tentative entries left when the scan ends are dropped.

Similarity is the one input the v0.7 log lacks per detection. The modes say what is done about it:

* ``geometry``: no similarity at all. A decision that depends on a similarity is "needs similarity".
* ``logged``: the `sim` of `scan_keyframe.results` (similarity to the entry the app matched, before
  the sighting). Similarities to the other candidates are still unknown.
* ``offline``: embeddings recomputed on this Mac with the app's embedder from the saved half-size
  keyframe JPEG (only keyframes with an image); an entry's embedding is the one of its best crop.
* ``embedding``: embeddings given with the input (the v0.8 observation log); used by study_check.py.

Every decision is checked from the app's own state (each step starts from the state the app's logged
decisions produce: a "forced" replay), so one disagreement does not spill into the next ones. A
second, "free" run applies the replay's own decisions (the logged one only where it cannot decide)
and its end state is compared with inventory.json. A rule-sensitivity run replays with one rule
changed at a time, to show which rules this data can confirm at all.

The core (Replay, Det, Rules) uses the standard library only; the offline embeddings need numpy,
Pillow and mediapipe (analysis/.venv-bench). No image is read unless --offline-embeddings is given,
and no image is written.
"""
import argparse
import base64
import collections
import glob
import json
import math
import os
import statistics
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- ScanConfig (v0.5-v0.7)

MERGE_RADIUS_FACTOR = 0.5
MERGE_RADIUS_OFFSET_M = 0.15
MERGE_RADIUS_MIN_M = 0.20
MERGE_RADIUS_MAX_M = 0.50
POSITION_SIGHTINGS = 10
TENTATIVE_CONFIRM_MS = 300
CROP_PAD = 0.08

KIND_NEW = "new"
KIND_PROMOTED = "promoted"
KIND_KNOWN = "known"
KIND_TENTATIVE = "tentative"

OK = "reproduced"
NEEDS_SIM = "needs similarity"
MISMATCH = "mismatch"

DEFAULT_WATCH = ("19b9bc7b", "ff153f1e", "11f9b575")


def f32(x):
    """Round a Python float to the nearest float32 (Kotlin Float arithmetic: each operation on two
    floats is exact in double and rounded once, so rounding the double result is faithful)."""
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


MERGE_SIMILARITY = f32(0.55)        # 0.55f


def radius(size_a, size_b):
    """Merger.radius: clamp(0.5 x max(sizes) + 0.15, 0.20, 0.50) metres."""
    r = MERGE_RADIUS_FACTOR * max(size_a, size_b) + MERGE_RADIUS_OFFSET_M
    return min(max(r, MERGE_RADIUS_MIN_M), MERGE_RADIUS_MAX_M)


def distance(p, q):
    """Vec3 distance: sqrt(x*x + y*y + z*z) of the difference, as the app computes it."""
    x, y, z = p[0] - q[0], p[1] - q[1], p[2] - q[2]
    return math.sqrt(x * x + y * y + z * z)


def box_area(box):
    """Box.area in float32: max(0, right - left) * max(0, bottom - top)."""
    left, top, right, bottom = (f32(v) for v in box)
    w, h = f32(right - left), f32(bottom - top)
    return f32(max(0.0, w) * max(0.0, h))


def quality(box, score):
    """Observation.quality: box area x score, float32."""
    return f32(box_area(box) * f32(score))


def cosine(a, b):
    """Merger.cosine (computed in double here; the app sums in float32, a difference near 1e-6)."""
    if a is None or b is None or len(a) != len(b) or len(a) == 0:
        return None
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    den = math.sqrt(na) * math.sqrt(nb)
    return 0.0 if den == 0 else dot / den


def decode_b64_f32(s):
    """The app's embedding encoding (base64 of little-endian float32)."""
    raw = base64.b64decode(s)
    return list(struct.unpack("<%df" % (len(raw) // 4), raw))


# ----------------------------------------------------------------------------- rules (and variants)

class Rules:
    """The app's rules; each keyword changes one of them (rule-sensitivity runs only)."""

    def __init__(self, radius_fn=radius, one_per_entry=True, position="mean10", replace_best=True,
                 label_votes=True, reverse_ties=False):
        self.radius_fn = radius_fn
        self.one_per_entry = one_per_entry
        self.position = position          # "mean10" (app), "last", "first"
        self.replace_best = replace_best
        self.label_votes = label_votes
        self.reverse_ties = reverse_ties


APP_RULES = Rules()


# ----------------------------------------------------------------------------- inputs

class Det:
    """One kept detection as the matcher sees it, plus what the app logged about it.

    key: stable name ("kf0012/3": keyframe index, position in the keyframe's detection list).
    order: position in the detector's output (ties of the score sort keep it).
    logged: dict(kind, entry, sim, dist, by_label) or None. Only the comparison reads it, except in
    the `logged` similarity mode, which reads `sim` for the logged entry.
    """

    __slots__ = ("key", "kf", "t", "label", "score", "box", "map_point", "size_m", "embedding",
                 "logged", "has_image", "order")

    def __init__(self, key, kf, t, label, score, box, map_point, size_m, embedding=None, logged=None,
                 has_image=False, order=0):
        self.key, self.kf, self.t = key, kf, t
        self.label, self.score, self.box = label, f32(score), tuple(box)
        self.map_point, self.size_m = tuple(map_point), float(size_m)
        self.embedding, self.logged, self.has_image, self.order = embedding, logged, has_image, order

    @property
    def quality(self):
        return quality(self.box, self.score)


def processing_order(dets, reverse_ties=False):
    """ScanSession.merge: `observations.sortedByDescending { it.score }`, a stable sort."""
    sign = -1 if reverse_ties else 1
    return sorted(dets, key=lambda d: (-d.score, sign * d.order))


# ----------------------------------------------------------------------------- state

class Entry:
    __slots__ = ("id", "label", "votes", "position", "size_m", "best_quality", "best_key", "best_kf",
                 "embedding", "sightings", "count", "first_ms", "last_ms", "tentative", "first_kf",
                 "dets", "rules")

    def __init__(self, eid, det, rules=APP_RULES):
        """Merger.newEntry: a first sighting, tentative."""
        self.rules = rules
        self.id = eid
        self.label = det.label
        self.votes = collections.OrderedDict([(det.label, float(det.score))])
        self.position = det.map_point
        self.size_m = det.size_m
        self.best_quality = det.quality
        self.best_key, self.best_kf = det.key, det.kf
        self.embedding = det.embedding
        self.sightings = [(det.t, det.map_point, det.size_m, det.label, det.score, det.kf)]
        self.count = 1
        self.first_ms = self.last_ms = det.t
        self.tentative = True
        self.first_kf = det.kf
        self.dets = [det.key]

    @classmethod
    def from_app(cls, d, rules=APP_RULES):
        """A real entry as the app stored it (inventory.json or an inventory_log line), for a scan
        that starts with entries from an earlier scan."""
        e = cls.__new__(cls)
        e.rules = rules
        e.id = d["id"]
        e.label = d.get("label") or ""
        e.votes = collections.OrderedDict((k, float(v)) for k, v in (d.get("labelVotes") or {}).items())
        e.position = tuple(d["mapPosition"])
        e.size_m = float(d.get("sizeM") or 0.0)
        e.best_quality = f32(d.get("bestQuality") or 0.0)
        e.best_key, e.best_kf = d.get("bestObs"), None
        emb = d.get("embedding")
        e.embedding = decode_b64_f32(emb) if isinstance(emb, str) else emb
        e.sightings = [(s.get("t"), tuple(s.get("p")), s.get("sizeM", 0.0), s.get("label", ""),
                        f32(s.get("score", 0.0)), s.get("kf")) for s in d.get("sightings") or []]
        e.count = int(d.get("sightingCount") or len(e.sightings))
        e.first_ms = d.get("firstSeenMs")
        e.last_ms = d.get("lastSeenMs")
        e.tentative = False
        e.first_kf = None
        e.dets = []
        return e

    def add(self, det):
        """Merger.addSighting. Returns whether the best crop was replaced."""
        r = self.rules
        self.sightings = (self.sightings + [(det.t, det.map_point, det.size_m, det.label, det.score,
                                             det.kf)])[-POSITION_SIGHTINGS:]
        if r.position == "mean10":
            sx = sy = sz = 0.0
            for s in self.sightings:
                sx += s[1][0]
                sy += s[1][1]
                sz += s[1][2]
            n = len(self.sightings)
            self.position = (sx / n, sy / n, sz / n)
        elif r.position == "last":
            self.position = det.map_point
        # "first": the position stays
        if r.label_votes:
            self.votes[det.label] = self.votes.get(det.label, 0.0) + float(det.score)
            current = self.votes.get(self.label, 0.0)
            top_label, top_value = None, None
            for k, v in self.votes.items():          # maxByOrNull: the first of equal maxima
                if top_value is None or v > top_value:
                    top_label, top_value = k, v
            if top_value is not None and top_value > current:
                self.label = top_label
        self.count += 1
        self.last_ms = max(self.last_ms, det.t)
        self.dets.append(det.key)
        better = det.quality > self.best_quality
        if better and r.replace_best:
            self.size_m = det.size_m
            self.best_quality = det.quality
            self.best_key, self.best_kf = det.key, det.kf
            self.embedding = det.embedding
        return better

    def summary(self):
        return {"id": self.id, "label": self.label, "votes": dict(self.votes), "position": list(self.position),
                "sizeM": self.size_m, "bestQuality": self.best_quality, "bestKey": self.best_key,
                "bestKf": self.best_kf, "sightingCount": self.count, "firstSeenMs": self.first_ms,
                "lastSeenMs": self.last_ms, "sightingKfs": [s[5] for s in self.sightings],
                "tentative": self.tentative, "dets": list(self.dets)}


class Decision:
    """What the replay says about one detection.

    For NEEDS_SIM, `if_similar` is what it would decide if every missing similarity were at least
    0.55, `if_not` if none were (one missing candidate: exactly the two possible outcomes)."""

    __slots__ = ("det", "status", "kind", "entry", "dist", "by_label", "sim", "missing", "candidates",
                 "if_similar", "if_not")

    def __init__(self, det, status, kind=None, entry=None, dist=None, by_label=None, sim=None, missing=(),
                 candidates=(), if_similar=None, if_not=None):
        self.det, self.status, self.kind, self.entry = det, status, kind, entry
        self.dist, self.by_label, self.sim = dist, by_label, sim
        self.missing = list(missing)          # ids of candidates whose similarity was needed and unknown
        self.candidates = list(candidates)    # (id, label, distance, radius, sim or None) within radius
        self.if_similar, self.if_not = if_similar, if_not


class Replay:
    """The state of one scan: real entries (store order) and tentative ones (creation order).

    sim_fn(det, entry) -> similarity or None (unknown). `initial` entries (real, from an earlier
    scan of the same map) are put in first, in their store order.
    """

    def __init__(self, sim_fn, initial=(), rules=APP_RULES):
        self.sim_fn = sim_fn
        self.rules = rules
        self.real = []              # ids, store order
        self.tent = []              # ids, creation order
        self.entries = {}
        self.created_in_scan = []
        for e in initial:
            e.tentative = False
            e.rules = rules
            self.entries[e.id] = e
            self.real.append(e.id)

    def candidates(self, used):
        ids = self.real + self.tent
        if self.rules.one_per_entry:
            ids = [i for i in ids if i not in used]
        return [self.entries[i] for i in ids]

    def _kind(self, e, t_ms):
        if e.tentative:
            return KIND_PROMOTED if t_ms - e.first_ms >= TENTATIVE_CONFIRM_MS else KIND_TENTATIVE
        return KIND_KNOWN

    def decide(self, det, t_ms, used):
        """Merger.match plus the kind ScanSession.merge gives it, without changing the state."""
        within = []
        for order, e in enumerate(self.candidates(used)):
            d = distance(e.position, det.map_point)
            r = self.rules.radius_fn(e.size_m, det.size_m)
            if d > r:
                continue
            by_label = e.label == det.label
            sim = self.sim_fn(det, e)
            if by_label:
                ok = True
            elif sim is None:
                ok = None                    # unknown
            else:
                ok = f32(sim) >= MERGE_SIMILARITY
            within.append((order, e, d, r, by_label, sim, ok))
        cands = [(e.id, e.label, d, r, sim) for _, e, d, r, _, sim, _ in within]

        def nearest(items):
            best = None
            for item in items:               # iteration order, strict "<"
                if best is None or item[2] < best[2]:
                    best = item
            return best

        best = nearest([it for it in within if it[6] is True])
        # An unknown candidate matters when it would have beaten the best known one.
        missing = [it for it in within if it[6] is None and
                   (best is None or it[2] < best[2] or (it[2] == best[2] and it[0] < best[0]))]

        def outcome(item):
            if item is None:
                return (KIND_NEW, None)
            return (self._kind(item[1], t_ms), item[1].id)

        if missing:
            alt = nearest([it for it in within if it[6] is True or it in missing])
            return Decision(det, NEEDS_SIM, missing=[it[1].id for it in missing], candidates=cands,
                            if_similar=outcome(alt), if_not=outcome(best))
        if best is None:
            return Decision(det, OK, KIND_NEW, None, candidates=cands)
        _, e, d, _, by_label, sim, _ = best
        return Decision(det, OK, self._kind(e, t_ms), e.id, d, by_label, sim, candidates=cands)

    def apply(self, det, t_ms, kind, entry_id, new_id):
        """Carry out a decision (the replay's or the app's). Returns the entry id used."""
        if kind == KIND_NEW:
            e = Entry(new_id, det, self.rules)
            self.entries[new_id] = e
            self.tent.append(new_id)
            self.created_in_scan.append(new_id)
            return new_id
        e = self.entries[entry_id]
        e.add(det)
        if e.tentative and t_ms - e.first_ms >= TENTATIVE_CONFIRM_MS:
            e.tentative = False
            self.tent.remove(e.id)
            self.real.append(e.id)
        return e.id

    def finish(self):
        """Drop the tentative entries. Returns their ids."""
        dropped = list(self.tent)
        for i in dropped:
            del self.entries[i]
        self.tent = []
        return dropped


# ----------------------------------------------------------------------------- similarity modes

def sim_none(det, entry):
    return None


def sim_logged(det, entry):
    """The logged similarity, for the entry the app matched only (ids are the app's)."""
    lg = det.logged or {}
    if lg.get("sim") is not None and lg.get("entry") == entry.id:
        return float(lg["sim"])
    return None


def sim_embedding(det, entry):
    """Cosine of the detection's embedding and the entry's best-crop embedding, when both are known."""
    return cosine(det.embedding, entry.embedding)


SIM_MODES = {"geometry": sim_none, "logged": sim_logged, "offline": sim_embedding,
             "embedding": sim_embedding}


# ----------------------------------------------------------------------------- runs

def nearest_candidates(rp, det, used, n=3):
    """The v0.8 `decision.candidates` as the replay sees them: the n candidates nearest to the
    detection (any distance), nearest first, as dicts {entry, distM, sim, sameLabel, withinRadius,
    tentative}; sim is None when an embedding is missing."""
    rows = []
    for e in rp.candidates(used):
        d = distance(e.position, det.map_point)
        rows.append({"entry": e.id, "distM": d, "sim": rp.sim_fn(det, e), "sameLabel": e.label == det.label,
                     "withinRadius": d <= rp.rules.radius_fn(e.size_m, det.size_m), "tentative": e.tentative})
    rows.sort(key=lambda r: r["distM"])
    return rows[:n]


def run_forced(keyframes, sim_fn, initial=(), rules=APP_RULES, observer=None):
    """Every detection decided from the state the app's logged decisions produce.

    keyframes: list of (kf index, t ms, [Det]) in capture order; every Det needs `logged`.
    observer(det, decision, replay, used), when given, is called before the logged decision is applied.
    Returns (decisions in processing order, final Replay, ids of dropped tentative entries)."""
    rp = Replay(sim_fn, initial, rules)
    out = []
    for kf, t, dets in keyframes:
        used = set()
        for det in processing_order(dets, rules.reverse_ties):
            dec = rp.decide(det, t, used)
            if observer is not None:
                observer(det, dec, rp, used)
            lg = det.logged
            if lg["kind"] != KIND_NEW and lg["entry"] not in rp.entries:
                raise ValueError("%s: the app's entry %s is unknown to the replay" % (det.key, lg["entry"]))
            rp.apply(det, t, lg["kind"], lg["entry"], lg["entry"])
            used.add(lg["entry"])
            out.append(dec)
    dropped = rp.finish()
    return out, rp, dropped


def run_free(keyframes, sim_fn, initial=(), rules=APP_RULES):
    """The replay's own decisions; the logged one only where it cannot decide.

    A replay entry created by the same detection as an app entry takes the app's id (first
    appearance); any other gets "R<n>". Returns (decisions, Replay, dropped ids)."""
    rp = Replay(sim_fn, initial, rules)
    out = []
    n_new = 0
    for kf, t, dets in keyframes:
        used = set()
        for det in processing_order(dets, rules.reverse_ties):
            dec = rp.decide(det, t, used)
            lg = det.logged or {}
            if dec.status == NEEDS_SIM:
                kind, eid = lg.get("kind"), lg.get("entry")
                if kind != KIND_NEW and (eid not in rp.entries or eid in used):
                    kind = KIND_NEW          # the app's entry does not exist (or is used) in this run
            else:
                kind, eid = dec.kind, dec.entry
            if kind == KIND_NEW:
                if lg.get("kind") == KIND_NEW and lg.get("entry") not in rp.entries:
                    eid = lg["entry"]
                else:
                    n_new += 1
                    eid = "R%d" % n_new
            used.add(rp.apply(det, t, kind, eid, eid))
            out.append(dec)
    dropped = rp.finish()
    return out, rp, dropped


def compare(decisions):
    """One row per decision: status against the logged kind and entry."""
    rows = []
    for dec in decisions:
        lg = dec.det.logged or {}
        row = {"key": dec.det.key, "kf": dec.det.kf, "label": dec.det.label, "score": dec.det.score,
               "logged_kind": lg.get("kind"), "logged_entry": lg.get("entry"), "logged_sim": lg.get("sim"),
               "logged_dist": lg.get("dist"), "replay_kind": dec.kind, "replay_entry": dec.entry,
               "replay_dist": dec.dist, "by_label": dec.by_label, "missing": dec.missing,
               "candidates": dec.candidates, "has_image": dec.det.has_image,
               "if_similar": dec.if_similar, "if_not": dec.if_not}
        if dec.status == NEEDS_SIM:
            row["status"] = NEEDS_SIM
        elif dec.kind == lg.get("kind") and (dec.kind == KIND_NEW or dec.entry == lg.get("entry")):
            row["status"] = OK
        else:
            row["status"] = MISMATCH
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- v0.7 pull

def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_v07(folder, scan_id):
    """Keyframes of a pre-v0.8 pull: (keyframes, info).

    keyframes: list of (index, t, [Det]) from scan/<scan>/kf*.json, logged decisions from the
    keyframe JSON (`kind`, `entry`) joined with `scan_keyframe.results` (`sim`, `distM`), which are
    in processing order; the join is on (label, score, entry) in that order."""
    kfs = []
    for p in sorted(glob.glob(os.path.join(folder, "scan", scan_id, "kf*.json"))):
        with open(p, encoding="utf-8") as f:
            kfs.append(json.load(f))
    kfs.sort(key=lambda k: k["index"])
    events = read_jsonl(os.path.join(folder, "events.jsonl"))
    results = {e.get("index"): e for e in events if e.get("type") == "scan_keyframe" and e.get("scan") == scan_id}
    keyframes = []
    join_problems = []
    order_equal = 0
    ties_reversed_differs = 0
    for k in kfs:
        dets = []
        for j, d in enumerate(k.get("detections") or []):
            if not d.get("kept"):
                continue
            dets.append(Det("kf%04d/%d" % (k["index"], j), k["index"], k["t"], d["label"], d["score"], d["box"],
                            d["mapPoint"], d["sizeM"], logged={"kind": d.get("kind"), "entry": d.get("entry")},
                            has_image=bool(k.get("image")), order=j))
        res = list((results.get(k["index"]) or {}).get("results") or [])
        logged_order = [(r.get("label"), f32(r.get("score", -1)), r.get("entry")) for r in res]
        replay_order = [(d.label, d.score, d.logged["entry"]) for d in processing_order(dets)]
        order_equal += logged_order == replay_order
        if len({d.score for d in dets}) < len(dets):
            rev = [(d.label, d.score, d.logged["entry"]) for d in processing_order(dets, reverse_ties=True)]
            ties_reversed_differs += rev != logged_order
        for det in processing_order(dets):
            hit = next((i for i, r in enumerate(res) if r.get("label") == det.label and
                        f32(r.get("score", -1)) == det.score and r.get("entry") == det.logged["entry"]), None)
            if hit is None:
                join_problems.append(det.key)
                det.logged.update(sim=None, dist=None)
                continue
            r = res.pop(hit)
            det.logged.update(sim=r.get("sim"), dist=r.get("distM"), kind_event=r.get("kind"))
        keyframes.append((k["index"], k["t"], dets))
    ties = sum(1 for _, _, ds in keyframes if len({d.score for d in ds}) < len(ds))
    info = {"keyframe_files": len(kfs), "with_image": sum(1 for k in kfs if k.get("image")),
            "events_keyframes": len(results), "join_problems": join_problems, "order_equal": order_equal,
            "keyframes_with_ties": ties, "ties_reversed_differs": ties_reversed_differs, "kf_json": {k["index"]: k for k in kfs}, "events": events}
    return keyframes, info


def load_inventory(folder):
    p = os.path.join(folder, "inventory.json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return (json.load(f) or {}).get("entries") or []


# ----------------------------------------------------------------------------- offline embeddings

def offline_embeddings(folder, scan_id, keyframes, kf_json, cache=None):
    """Embed every kept detection of a keyframe with an image, as the app crops it (box padded by
    8 %, clamped, truncated, turned upright) but from the half-size JPEG. Needs analysis/.venv-bench.

    Returns {det key: list of floats}. `cache`: a JSON file read when present and written after."""
    if cache and os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    sys.path.insert(0, HERE)
    import bench_common as bc
    from bench_match import APP_MODELS, MediaPipeEmbedder
    model = MediaPipeEmbedder(os.path.join(APP_MODELS, "mobilenet_v3_small.tflite"))
    out = {}
    for kf, _t, dets in keyframes:
        k = kf_json[kf]
        img = k.get("image")
        if not img:
            continue
        sensor = bc.load_rgb(os.path.join(folder, "scan", scan_id, img["file"]))
        src_w = img.get("sourceWidth") or sensor.shape[1]
        src_h = img.get("sourceHeight") or sensor.shape[0]
        scale = sensor.shape[1] / float(src_w)
        for det in dets:
            rect = bc.clamp_rect(bc.pad_rect(det.box, CROP_PAD), src_w, src_h)
            crop = bc.crop_from_sensor(sensor, rect, k.get("rotation") or 0, scale)
            out[det.key] = [float(x) for x in model.features(crop)]
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(out, f)
    return out


def offline_fidelity(keyframes, rp_forced, inventory, inv_log):
    """How well offline half-size embeddings stand in for the phone's.

    stored: offline embedding of the detection that supplied an entry's best crop against the
    embedding the phone stored for it (inventory.json at the end; the `add` line of
    inventory_log.jsonl at promotion), when that keyframe has an image.
    sims: offline similarity against the logged `sim` of every matched detection whose own keyframe
    and whose entry's best crop (at that moment) both have an image."""
    dets = {d.key: d for _, _, ds in keyframes for d in ds}
    stored = []
    for a in inventory:
        e = rp_forced.entries.get(a["id"])
        d = dets.get(e.best_key) if e else None
        if d is not None and d.embedding is not None and a.get("embedding"):
            stored.append(("end", a["id"], cosine(d.embedding, decode_b64_f32(a["embedding"]))))
    # at promotion: the best crop of the first two sightings (the higher quality; ties: the first)
    for line in inv_log:
        if line.get("op") != "add":
            continue
        a = line["entry"]
        e = rp_forced.entries.get(a["id"])
        if e is None or len(e.dets) < 2:
            continue
        d1, d2 = dets[e.dets[0]], dets[e.dets[1]]
        d = d2 if d2.quality > d1.quality else d1
        if d.embedding is not None and a.get("embedding"):
            stored.append(("add", a["id"], cosine(d.embedding, decode_b64_f32(a["embedding"]))))
    return stored


def sim_agreement(rows):
    """(logged, offline) similarity of the matched entry, where both exist."""
    out = []
    for r in rows:
        if r["logged_sim"] is None:
            continue
        off = next((c[4] for c in r["candidates"] if c[0] == r["logged_entry"]), None)
        if off is not None:
            out.append((r["key"], float(r["logged_sim"]), off))
    return out


# ----------------------------------------------------------------------------- end state

def app_best_kf(entry, kf_camera):
    """Keyframe whose camera position (scan_keyframe.cameraMap) equals the entry's viewPosition:
    the app sets viewPosition from the keyframe of the best crop."""
    vp = entry.get("viewPosition")
    hits = [i for i, c in kf_camera.items() if c is not None and vp is not None and
            all(abs(a - b) < 1e-9 for a, b in zip(c, vp))]
    return hits[0] if len(hits) == 1 else None


def crop_size(box, width, height):
    """(cropWidthPx, cropHeightPx): CameraFrame.pad by 8 % then clamp, in float32."""
    left, top, right, bottom = (f32(v) for v in box)
    dx, dy = f32(f32(right - left) * f32(CROP_PAD)), f32(f32(bottom - top) * f32(CROP_PAD))
    left, top, right, bottom = f32(left - dx), f32(top - dy), f32(right + dx), f32(bottom + dy)
    left = min(max(left, 0.0), f32(width - 1.0))
    top = min(max(top, 0.0), f32(height - 1.0))
    right = min(max(right, 1.0), float(width))
    bottom = min(max(bottom, 1.0), float(height))
    return f32(right - left), f32(bottom - top)


def world_from_map(p, m2w):
    """Pose4.transformPoint with the keyframe JSON's mapToWorld (x, y, z, yawDeg)."""
    yaw = math.radians(m2w["yawDeg"])
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * p[0] + s * p[2] + m2w["x"], p[1] + m2w["y"], -s * p[0] + c * p[2] + m2w["z"])


def end_state(rp, inventory, kf_camera, kf_json=None):
    """Compare the replay's entries with inventory.json (same ids). Returns (rows, app ids with no
    replay entry). With kf_json, also the fields derived from the best crop's keyframe: crop size,
    world position (from the map point and the keyframe's mapToWorld) and capture distance."""
    rows = []
    by_app = {e["id"]: e for e in inventory}
    seen = set()
    for rid, e in rp.entries.items():
        a = by_app.get(rid)
        if a is None:
            rows.append({"id": rid, "replay_only": True, "label": e.label, "count": e.count})
            continue
        seen.add(rid)
        app_kfs = [s.get("kf") for s in a.get("sightings") or []]
        row = {
            "id": rid, "label": e.label, "app_label": a.get("label"), "label_equal": e.label == a.get("label"),
            "dpos_m": distance(e.position, a["mapPosition"]),
            "count": e.count, "app_count": a.get("sightingCount"),
            "sighting_kfs_equal": [s[5] for s in e.sightings] == app_kfs,
            "best_kf": e.best_kf, "app_best_kf": app_best_kf(a, kf_camera),
            "best_quality": e.best_quality, "app_best_quality": a.get("bestQuality"),
            "size_equal": e.size_m == a.get("sizeM"),
            "votes_equal": set(e.votes) == set(a.get("labelVotes") or {}) and
            all(abs(e.votes.get(k, 0) - v) < 1e-12 for k, v in (a.get("labelVotes") or {}).items()),
            "first_equal": e.first_ms == a.get("firstSeenMs"),
            "last_equal": e.last_ms == a.get("lastSeenMs"),
        }
        k = (kf_json or {}).get(e.best_kf)
        if k is not None and e.best_key:
            j = int(e.best_key.split("/")[1])
            det = k["detections"][j]
            w, h = k["intrinsics"]["width"], k["intrinsics"]["height"]
            row["crop_equal"] = crop_size(det["box"], w, h) == (f32(a.get("cropWidthPx", -1)),
                                                                f32(a.get("cropHeightPx", -1)))
            world = world_from_map(det["mapPoint"], k["mapToWorld"])
            row["dworld_m"] = distance(world, a["worldPosition"]) if a.get("worldPosition") else None
            row["dcapture_m"] = abs(distance(world, k["cameraPose"]["t"]) - a.get("captureDistance", 0.0))
        rows.append(row)
    missing = sorted(set(by_app) - seen)
    return rows, missing


def promotion_check(keyframes, sim_fn, inv_log):
    """Forced replay; each entry's state right after its promotion against the `add` line of
    inventory_log.jsonl (written at that moment). Returns rows."""
    adds = {l["entry"]["id"]: l["entry"] for l in inv_log if l.get("op") == "add"}
    rp = Replay(sim_fn)
    rows = []
    for kf, t, dets in keyframes:
        for det in processing_order(dets):
            lg = det.logged
            rp.apply(det, t, lg["kind"], lg["entry"], lg["entry"])
            if lg["kind"] == KIND_PROMOTED and lg["entry"] in adds:
                e, a = rp.entries[lg["entry"]], adds[lg["entry"]]
                rows.append({"id": e.id, "dpos_m": distance(e.position, a["mapPosition"]),
                             "count_equal": e.count == a.get("sightingCount"),
                             "quality_equal": e.best_quality == f32(a.get("bestQuality", -1)),
                             "label_equal": e.label == a.get("label")})
    return rows


# ----------------------------------------------------------------------------- report

def tally(rows, keys=None):
    rows = [r for r in rows if keys is None or r["key"] in keys]
    c = collections.Counter(r["status"] for r in rows)
    return {"n": len(rows), OK: c.get(OK, 0), NEEDS_SIM: c.get(NEEDS_SIM, 0), MISMATCH: c.get(MISMATCH, 0)}


def fmt_tally(t):
    return "n = %d: %d reproduced, %d need a similarity, %d mismatch" % (t["n"], t[OK], t[NEEDS_SIM], t[MISMATCH])


def fmt_outcome(o):
    if o is None:
        return "-"
    return o[0] if o[1] is None else "%s %s" % o


SENSITIVITY = [
    ("app rules", {}),
    ("fixed radius 0.20 m", {"radius_fn": lambda a, b: MERGE_RADIUS_MIN_M}),
    ("fixed radius 0.50 m", {"radius_fn": lambda a, b: MERGE_RADIUS_MAX_M}),
    ("radius from the detection's size only", {"radius_fn": lambda a, b: radius(0.0, b)}),
    ("no one-detection-per-entry limit", {"one_per_entry": False}),
    ("position = latest sighting", {"position": "last"}),
    ("position = first sighting", {"position": "first"}),
    ("best crop (size) never replaced", {"replace_best": False}),
    ("label = first label (no votes)", {"label_votes": False}),
    ("equal scores in reverse detector order", {"reverse_ties": True}),
]


def sensitivity(keyframes, sim_fn, keys):
    """Forced replays with one rule changed. Returns [(name, tally)]."""
    out = []
    for name, kw in SENSITIVITY:
        decs, _, _ = run_forced(keyframes, sim_fn, rules=Rules(**kw))
        out.append((name, tally(compare(decs), keys)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", help="pulled data folder (scan/<scan>/kf*.json, events.jsonl, inventory.json)")
    ap.add_argument("--scan", default=None, help="scan id (default: the one with the most saved keyframes)")
    ap.add_argument("--offline-embeddings", action="store_true",
                    help="recompute embeddings from the saved keyframe JPEGs (needs analysis/.venv-bench)")
    ap.add_argument("--embedding-cache", default=None,
                    help="JSON cache of offline embeddings (default analysis/out/scan_replay/<scan>_emb.json, "
                         "git-ignored)")
    ap.add_argument("--watch", default=",".join(DEFAULT_WATCH),
                    help="entry ids also reported on their own (comma-separated)")
    ap.add_argument("--json", default=None, help="write every row to this JSON file")
    args = ap.parse_args(argv)

    scan = args.scan
    if scan is None:
        scans = [(len(glob.glob(os.path.join(p, "kf*.json"))), os.path.basename(p))
                 for p in glob.glob(os.path.join(args.folder, "scan", "*"))]
        scans = [s for s in scans if s[0] > 0]
        if not scans:
            ap.error("no scan/<scan>/kf*.json in %s" % args.folder)
        scan = max(scans)[1]
    keyframes, info = load_v07(args.folder, scan)
    inventory = [e for e in load_inventory(args.folder) if e.get("createdScan") == scan]
    inv_log = read_jsonl(os.path.join(args.folder, "inventory_log.jsonl"))
    saved = {e["id"] for e in inventory}
    watch = set(x for x in args.watch.split(",") if x)
    all_dets = [d for _, _, ds in keyframes for d in ds]
    k_saved = {d.key for d in all_dets if d.logged["entry"] in saved}
    k_dropped = {d.key for d in all_dets if d.logged["entry"] not in saved}
    k_watch = {d.key for d in all_dets if d.logged["entry"] in watch}
    k_rest = k_saved - k_watch

    emb = None
    if args.offline_embeddings:
        cache = args.embedding_cache or os.path.join(HERE, "out", "scan_replay", "%s_emb.json" % scan)
        emb = offline_embeddings(args.folder, scan, keyframes, info["kf_json"], cache)
        for d in all_dets:
            d.embedding = emb.get(d.key)

    kf_camera = {e.get("index"): e.get("cameraMap") for e in info["events"]
                 if e.get("type") == "scan_keyframe" and e.get("scan") == scan}
    print("Scan %s: %d keyframe files (%d with an image), %d kept detections: %d to the %d saved entries "
          "(%d of them to the %d watched entries), %d to tentative entries dropped at the end."
          % (scan, info["keyframe_files"], info["with_image"], len(all_dets), len(k_saved), len(saved),
             len(k_watch), len(watch & saved), len(k_dropped)))
    print("Join of keyframe JSON and scan_keyframe.results: %d problems. Processing order (stable sort by "
          "score) equals the order of scan_keyframe.results in %d of %d keyframes; %d keyframes have equal "
          "scores among their kept detections, and in %d of them the reverse order of equal scores would "
          "differ from the logged order." % (len(info["join_problems"]), info["order_equal"], len(keyframes),
                                             info["keyframes_with_ties"], info["ties_reversed_differs"]))
    n_emb = sum(1 for d in all_dets if d.embedding is not None)
    if emb is not None:
        print("Offline embeddings: %d of %d kept detections (the others are in keyframes without an image)."
              % (n_emb, len(all_dets)))
    report = {"scan": scan, "modes": {}, "groups": {"saved": sorted(k_saved), "dropped": sorted(k_dropped),
                                                    "watched": sorted(k_watch)}}
    modes = ["geometry", "logged"] + (["offline"] if emb is not None else [])
    for mode in modes:
        decs, rp, _ = run_forced(keyframes, SIM_MODES[mode])
        rows = compare(decs)
        print("\n== mode %s (forced: each step from the app's state)" % mode)
        print("  saved entries          ", fmt_tally(tally(rows, k_saved)))
        print("    watched entries      ", fmt_tally(tally(rows, k_watch)))
        print("    other saved entries  ", fmt_tally(tally(rows, k_rest)))
        print("  dropped entries        ", fmt_tally(tally(rows, k_dropped)))
        for r in rows:
            if r["status"] != OK:
                grp = "watched" if r["key"] in k_watch else "saved" if r["key"] in k_saved else "dropped"
                print("   %-16s kf %3d %-8s %-8s logged %-8s %-9s replay %-8s %-9s | if sim(%s) >= 0.55: %s;"
                      " else: %s" % (r["status"], r["kf"], r["label"], grp, r["logged_kind"], r["logged_entry"],
                                     r["replay_kind"], r["replay_entry"], ",".join(r["missing"]),
                                     fmt_outcome(r["if_similar"]), fmt_outcome(r["if_not"])))
                for c in r["candidates"]:
                    print("        candidate %-9s %-9s d %.3f m, radius %.3f m, sim %s"
                          % (c[0], c[1], c[2], c[3], "-" if c[4] is None else "%.3f" % c[4]))
        matched = [r for r in rows if r["replay_kind"] in (KIND_KNOWN, KIND_PROMOTED)]
        dd = [abs(r["replay_dist"] - r["logged_dist"]) for r in matched if r["logged_dist"] is not None]
        print("  matches: %d, by label %d, by similarity %d; |replay distance - logged distM| max %s (n = %d)"
              % (len(matched), sum(1 for r in matched if r["by_label"]),
                 sum(1 for r in matched if not r["by_label"]), "%.3g" % max(dd) if dd else "-", len(dd)))
        fr_decs, frp, dropped = run_free(keyframes, SIM_MODES[mode])
        fr_rows = compare(fr_decs)
        es_free, missing = end_state(frp, inventory, kf_camera)
        print("  free run: %s; entries kept %d, dropped %d; app entries without a replay entry: %s"
              % (fmt_tally(tally(fr_rows)), len(frp.entries), len(dropped), ",".join(missing) or "none"))
        es_forced, _ = end_state(rp, inventory, kf_camera)
        report["modes"][mode] = {"forced": rows, "free": fr_rows, "end_free": es_free, "end_forced": es_forced,
                                 "missing_in_free": missing}
        if mode == "offline":
            agree = sim_agreement(rows)
            if agree:
                diffs = [abs(a - b) for _, a, b in agree]
                side = sum(1 for _, a, b in agree if (f32(a) >= MERGE_SIMILARITY) == (f32(b) >= MERGE_SIMILARITY))
                print("  offline vs logged similarity of the matched entry: mean |diff| %.3f, median %.3f, max %.3f"
                      " (n = %d of %d logged); same side of 0.55 in %d of %d"
                      % (statistics.mean(diffs), statistics.median(diffs), max(diffs), len(agree),
                         sum(1 for r in rows if r["logged_sim"] is not None), side, len(agree)))
                for k, a, b in agree:
                    if (f32(a) >= MERGE_SIMILARITY) != (f32(b) >= MERGE_SIMILARITY):
                        print("    other side of 0.55: %s logged %.3f offline %.3f" % (k, a, b))
                report["modes"][mode]["sim_agreement"] = agree
            stored = offline_fidelity(keyframes, rp, inventory, inv_log)
            for which in ("end", "add"):
                cs = [c for w, _, c in stored if w == which]
                if cs:
                    print("  offline embedding vs phone-stored embedding of the same crop (%s): cosine median %.4f,"
                          " min %.4f (n = %d)" % ("inventory.json" if which == "end" else "inventory_log add",
                                                  statistics.median(cs), min(cs), len(cs)))
            report["modes"][mode]["stored_fidelity"] = stored
            # decisions that rested on an offline similarity to a candidate of another label
            for r in rows:
                for c in r["candidates"]:
                    if c[1] != r["label"] and c[4] is not None:
                        print("    offline sim to another label: kf %d %s -> %s (%s) %.3f, margin to 0.55 %.3f"
                              % (r["kf"], r["label"], c[0], c[1], c[4], c[4] - MERGE_SIMILARITY))

    _, rp_geo, dropped_geo = run_forced(keyframes, sim_none)
    es, _ = end_state(rp_geo, inventory, kf_camera, info["kf_json"])
    end_ev = next((e for e in info["events"] if e.get("type") == "scan_end" and e.get("scan") == scan), {})
    print("\n== end state of the forced run against inventory.json (%d entries; the app's scan_end: %s new, "
          "%s tentative dropped; replay: %d entries, %d dropped)"
          % (len(es), (end_ev.get("entries") or {}).get("new"), end_ev.get("tentativeDropped"), len(rp_geo.entries),
             len(dropped_geo)))
    dpos = [r["dpos_m"] for r in es if "dpos_m" in r]
    if dpos:
        print("  position difference: max %.3g m, median %.3g m (n = %d)" % (max(dpos), statistics.median(dpos), len(dpos)))
    for name in ("count", "best_kf", "best_quality"):
        ok = sum(1 for r in es if r.get(name) is not None and r.get(name) == r.get("app_" + name))
        print("  %s equal: %d of %d" % (name, ok, len(es)))
    for name in ("label_equal", "sighting_kfs_equal", "size_equal", "votes_equal", "first_equal", "last_equal",
                 "crop_equal"):
        print("  %s: %d of %d" % (name, sum(1 for r in es if r.get(name)), len(es)))
    for name in ("dworld_m", "dcapture_m"):
        xs = [r[name] for r in es if r.get(name) is not None]
        if xs:
            print("  %s: max %.3g (n = %d)" % (name, max(xs), len(xs)))
    pc = promotion_check(keyframes, sim_none, inv_log)
    if pc:
        print("  at promotion (inventory_log add lines, n = %d): position difference max %.3g m; sighting count "
              "equal %d, best quality equal %d, label equal %d"
              % (len(pc), max(r["dpos_m"] for r in pc), sum(r["count_equal"] for r in pc),
                 sum(r["quality_equal"] for r in pc), sum(r["label_equal"] for r in pc)))
    report["end_state"] = es
    report["promotion"] = pc
    wes = [r for r in es if r["id"] in watch]
    for r in wes:
        print("  watched %s: position difference %.3g m, sightings %d (app %s), best crop kf %s (app %s)"
              % (r["id"], r["dpos_m"], r["count"], r["app_count"], r["best_kf"], r["app_best_kf"]))

    print("\n== rule sensitivity (logged mode, forced; all %d kept detections)" % len(all_dets))
    sens = sensitivity(keyframes, sim_logged, None)
    for name, t in sens:
        print("  %-42s %s" % (name, fmt_tally(t)))
    report["sensitivity"] = sens
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, default=str)
    return report


if __name__ == "__main__":
    main()
