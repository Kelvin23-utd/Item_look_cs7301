#!/usr/bin/env python3
"""Shared code of the offline baseline benchmarks (analysis/BASELINES_PLAN.md, Tracks A and B).

Pure functions and small classes on numpy and Pillow only, so the pair construction and the
metrics can be tested without any model (analysis/test_bench.py). The models live in
bench_match.py (Track A) and bench_detect.py (Track B). The design these functions implement is
fixed in analysis/notes/2026-11-17_baselines_design.md.

Crops are cut exactly as the app cuts them (FindBack ml/CameraFrame.kt): a rectangle in *sensor*
image pixels is clamped to the image, truncated to integers, cut, and turned upright by the
display rotation (clockwise degrees).
"""
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

import fb_load

# The developer's own device checks, marked "Exclude" in the session log of analysis/README.md.
DEFAULT_EXCLUDED_SESSIONS = (
    "cab39b87", "1debdfe5", "2b883ce9", "5a4b83e9", "51e9efc9", "007cc164",
    "7a7718f6", "c5b9a6ae", "735b0f23", "1925c109", "acbda081", "c8990ea7",
)

POSITIVE_TRUTHS = ("present",)
NEGATIVE_TRUTHS = ("absent", "replaced")
CROP_PAD = 0.08          # ScanConfig.CROP_PAD and the verify detector-box padding
DUP_RADIUS_M = 0.5       # "same label within 0.5 m" may be one object (plan, Track A item 2)

BOOT_N = 2000
BOOT_SEED = 20261117
BOOT_ALPHA = 0.05
BOOT_MIN_VALID = 0.9
MIN_BOOT_ITEMS = 4
MIN_PER_CLASS = 10
MIN_NEG_FOR_TAR = 20
FAR_TARGET = 0.05
APP_THRESHOLD = 0.75     # VerificationPolicy.MATCH_THRESHOLD


# ----------------------------------------------------------------------------- crops

def clamp_rect(r, w, h):
    """CameraFrame.clamp: left/top in [0, w-1]/[0, h-1], right/bottom in [1, w]/[1, h]."""
    l, t, rr, b = (float(x) for x in r)
    return (min(max(l, 0.0), w - 1.0), min(max(t, 0.0), h - 1.0),
            min(max(rr, 1.0), float(w)), min(max(b, 1.0), float(h)))


def pad_rect(r, fraction):
    """CameraFrame.pad / Box.padded: grown by `fraction` of its size on every side."""
    l, t, rr, b = (float(x) for x in r)
    dx, dy = (rr - l) * fraction, (b - t) * fraction
    return (l - dx, t - dy, rr + dx, b + dy)


def crop_box_int(rect, w, h):
    """The integer box (x, y, cw, ch) that CameraFrame.cropUpright cuts for `rect` (sensor px)."""
    l, t, rr, b = clamp_rect(rect, w, h)
    x, y = int(l), int(t)                       # Kotlin toInt() truncates; values are >= 0
    cw = max(1, min(int(rr - l), w - x))
    ch = max(1, min(int(b - t), h - y))
    return x, y, cw, ch


def upright_to_sensor(img, rotation):
    """Undo the app's clockwise rotation of an upright image (H x W x C numpy array)."""
    k = (int(rotation) // 90) % 4
    return np.rot90(img, k=k) if k else img


def sensor_to_upright(img, rotation):
    """Rotate a sensor-orientation image clockwise by `rotation` degrees (CameraFrame.rotate)."""
    k = (int(rotation) // 90) % 4
    return np.rot90(img, k=-k) if k else img


def crop_from_sensor(sensor_img, rect, rotation, scale=1.0):
    """Upright crop of `rect` (full-resolution sensor pixels) from a sensor-orientation image that
    was stored at `scale` times the full resolution (scan keyframes: 0.5)."""
    h, w = sensor_img.shape[:2]
    r = tuple(float(v) * scale for v in rect)
    x, y, cw, ch = crop_box_int(r, w, h)
    return np.ascontiguousarray(sensor_to_upright(sensor_img[y:y + ch, x:x + cw], rotation))


def crop_from_upright(upright_img, rect, rotation):
    """Upright crop of `rect` (sensor pixels) from a full-size upright frame (verify frames)."""
    return crop_from_sensor(upright_to_sensor(upright_img, rotation), rect, rotation, 1.0)


def load_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def iou(a, b):
    """Intersection over union of two (l, t, r, b) boxes; 0 for empty boxes (Box.iou)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = w * h if (w > 0 and h > 0) else 0.0
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])  # noqa: E731
    u = area(a) + area(b) - inter
    return inter / u if u > 0 else 0.0


# ----------------------------------------------------------------------------- data

def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


class BenchData:
    """Events, records, inventory entries and saved scan keyframes of one or more pulled folders.

    `paths`: data folders. When several folders hold the same file, the first folder given wins.
    `excluded`: session ids to leave out entirely.
    """

    def __init__(self, paths, excluded=DEFAULT_EXCLUDED_SESSIONS):
        self.paths = list(paths)
        self.excluded = set(excluded)
        self.ds = fb_load.Dataset(self.paths)
        self.events = [e for e in self.ds.events if e.get("session") not in self.excluded]
        self.queries, self.captures, _ = fb_load.build_queries(self.events)
        self.query_meta = {}
        for e in self.events:
            if e.get("type") == "query_start" and e.get("query") is not None:
                self.query_meta.setdefault((e.get("session"), e.get("query")), e)
        self._load_inventory()
        self._load_keyframes()

    # -- files
    def find(self, relpath, session=None):
        """Absolute path of `relpath` in a folder of `session` (any folder when None), else None."""
        if not relpath:
            return None
        if session is not None:
            p = self.ds.find_file(session, relpath)
            if p:
                return p
        for folder in self.paths:
            p = os.path.join(folder, relpath)
            if os.path.exists(p):
                return p
        return None

    # -- inventory
    def _load_inventory(self):
        """self.entries: id -> entry dict (inventory.json, else the inv_add event's copy);
        self.entry_deleted_ms: id -> time of its first inv_delete."""
        self.entries = {}
        for folder in self.paths:
            d = _read_json(os.path.join(folder, "inventory.json"))
            for e in (d or {}).get("entries", []) if isinstance(d, dict) else []:
                if isinstance(e, dict) and e.get("id") and e.get("sessionId") not in self.excluded:
                    old = self.entries.get(e["id"])
                    if old is None or (fb_load.num(e.get("lastSeenMs")) or 0) > (fb_load.num(old.get("lastSeenMs")) or 0):
                        self.entries[e["id"]] = e
        self.entry_deleted_ms = {}
        for e in self.events:
            if e.get("type") == "inv_add" and isinstance(e.get("entry"), dict):
                ent = e["entry"]
                eid = e.get("inventoryId") or ent.get("id")
                if eid and eid not in self.entries:
                    self.entries[eid] = dict(ent, id=eid)
            elif e.get("type") == "inv_delete":
                eid = e.get("inventoryId") or (e.get("entry") or {}).get("id")
                if eid and eid not in self.entry_deleted_ms:
                    self.entry_deleted_ms[eid] = fb_load.num(e.get("t"))

    def entry_crop_path(self, eid):
        e = self.entries.get(eid) or {}
        rel = e.get("cropFile") or "inventory/%s.jpg" % eid
        return self.find(rel, e.get("sessionId"))

    # -- records
    def record_by_id(self, rid, t_ms=None, folder_index=None):
        """A record dict with this id: as active at t_ms in that folder's history when possible,
        else any copy in records.json or records_log.jsonl of any folder."""
        if rid is None:
            return None
        if folder_index is not None:
            act = self.ds.history(folder_index).active_at(t_ms)
            if rid in act:
                return act[rid]
        for f in self.ds.folders:
            for r in f.records:
                if r.get("id") == rid:
                    return r
            for row in f.records_log or []:
                rec = row.get("record")
                if isinstance(rec, dict) and rec.get("id") == rid:
                    return rec
        return None

    def record_crop_path(self, rec):
        if not isinstance(rec, dict):
            return None
        rel = rec.get("cropFile") or "crops/%s.jpg" % rec.get("id")
        return self.find(rel, rec.get("sessionId"))

    # -- scan keyframes
    def _load_keyframes(self):
        """self.keyframes: list of (scan_dir, kf dict), one per (scan, index), sessions not excluded."""
        self.keyframes = []
        seen = set()
        for folder in self.paths:
            for p in sorted(glob.glob(os.path.join(folder, "scan", "*", "kf*.json"))):
                d = _read_json(p)
                if not isinstance(d, dict) or d.get("session") in self.excluded:
                    continue
                key = (d.get("scan"), d.get("index"))
                if key in seen:
                    continue
                seen.add(key)
                self.keyframes.append((os.path.dirname(p), d))
        self.keyframes.sort(key=lambda x: (str(x[1].get("scan")), x[1].get("index") or 0))

    def keyframe_image_path(self, scan_dir, kf):
        img = kf.get("image")
        if not isinstance(img, dict) or not img.get("file"):
            return None
        p = os.path.join(scan_dir, img["file"])
        return p if os.path.exists(p) else None


def dup_entries(e1, e2, radius=DUP_RADIUS_M):
    """Two inventory entries may be one object: same winning label and map positions within
    `radius` metres in the same map."""
    if not e1 or not e2 or e1.get("id") == e2.get("id"):
        return False
    if (e1.get("label") or "") != (e2.get("label") or "") or e1.get("mapId") != e2.get("mapId"):
        return False
    p, q = e1.get("mapPosition"), e2.get("mapPosition")
    if not (isinstance(p, list) and isinstance(q, list) and len(p) == 3 and len(q) == 3):
        return False
    return math.dist(p, q) <= radius


# ----------------------------------------------------------------------------- Track A pairs

class PairSet:
    """Crops (upright RGB arrays) and labelled pairs.

    pair: dict(set, a=(crop keys: the method takes the best of them), b=crop key, label 1 same /
    0 different, item_a, item_b, dup (the two items may be one object), capture (key or None)).
    """

    def __init__(self):
        self.crops = {}
        self.crop_meta = {}
        self.pairs = []
        self.notes = []
        self.captures = []   # one dict per usable capture, for the reproduction check

    def add_crop(self, key, img, **meta):
        if key not in self.crops:
            self.crops[key] = img
            self.crop_meta[key] = meta
        return key

    def select(self, name):
        """Pair indices of a reported set (see REPORTED_SETS)."""
        f = REPORTED_SETS[name][1]
        return [i for i, p in enumerate(self.pairs) if f(p)]


REPORTED_SETS = {
    # name: (description, filter)
    "verify": ("Verify pairs: stored crop vs fresh capture, labelled by ground truth",
               lambda p: p["set"] == "verify"),
    "verify+imp": ("Verify pairs plus impostors (fresh capture vs other items' stored crops)",
                   lambda p: p["set"] in ("verify", "verify_imp") and not p["dup"]),
    "scan": ("Scan pairs, all", lambda p: p["set"] == "scan"),
    "scan-nodup": ("Scan pairs, without possible duplicates", lambda p: p["set"] == "scan" and not p["dup"]),
    "cross": ("Cross pairs: scan crop vs fresh capture, all", lambda p: p["set"] == "cross"),
    "cross-nodup": ("Cross pairs, without possible duplicates", lambda p: p["set"] == "cross" and not p["dup"]),
}


def capture_item(bd, cap):
    """(item id, kind, target dict) of a capture: 'inv:<entry>' for a scanned entry, else
    'rec:<lineage>' for a tapped record."""
    qs = bd.query_meta.get((cap["session"], cap["query"])) or {}
    inv_id = qs.get("inventoryId")
    if qs.get("source") == "scan" and inv_id:
        return "inv:" + inv_id, "inv", bd.entries.get(inv_id) or {"id": inv_id}
    rec = bd.record_by_id(cap.get("record"), cap.get("t"), cap.get("folder"))
    lineage = fb_load.lineage_of(rec) if rec else cap.get("record")
    return "rec:%s" % lineage, "rec", rec or {"id": cap.get("record")}


def build_pairs(bd, verbose=False):
    """All Track A pair sets of the design note. Returns a PairSet."""
    ps = PairSet()
    truth_of = {(q["session"], q["query"]): q["truth"] for q in bd.queries}
    image_cache = {}

    def image(path):
        if path not in image_cache:
            image_cache[path] = load_rgb(path)
        return image_cache[path]

    # ---- scan detection crops
    det_keys = []          # (key, entry id)
    n_tentative = n_unsaved = 0
    for scan_dir, kf in bd.keyframes:
        img_path = bd.keyframe_image_path(scan_dir, kf)
        for j, det in enumerate(kf.get("detections") or []):
            if not det.get("kept") or not det.get("entry"):
                continue
            eid = det["entry"]
            if eid not in bd.entries:
                n_tentative += 1
                continue
            if img_path is None:
                n_unsaved += 1
                continue
            sensor = image(img_path)
            src_w = (kf.get("image") or {}).get("sourceWidth") or sensor.shape[1]
            scale = sensor.shape[1] / float(src_w)
            src_h = (kf.get("image") or {}).get("sourceHeight") or round(sensor.shape[0] / scale)
            rect = clamp_rect(pad_rect(det["box"], CROP_PAD), src_w, src_h)
            crop = crop_from_sensor(sensor, rect, kf.get("rotation") or 0, scale)
            key = "det:%s:%04d:%d" % (kf.get("scan"), kf.get("index") or 0, j)
            ps.add_crop(key, crop, kind="scan", item="inv:" + eid, entry=eid, label=det.get("label"),
                        scan=kf.get("scan"), index=kf.get("index"), session=kf.get("session"))
            det_keys.append((key, eid))
    ps.notes.append("scan: %d kept detections of real entries in saved keyframes; %d of tentative "
                    "entries dropped at the end of a scan and %d in keyframes not saved (person in view) "
                    "left out" % (len(det_keys), n_tentative, n_unsaved))

    for i in range(len(det_keys)):
        for j in range(i + 1, len(det_keys)):
            (ka, ea), (kb, eb) = det_keys[i], det_keys[j]
            same = ea == eb
            ps.pairs.append({"set": "scan", "a": (ka,), "b": kb, "label": int(same),
                             "item_a": "inv:" + ea, "item_b": "inv:" + eb,
                             "dup": (not same) and dup_entries(bd.entries[ea], bd.entries[eb]),
                             "capture": None})

    # ---- captures
    skipped = defaultdict(int)
    for cap in bd.captures:
        truth = truth_of.get((cap["session"], cap["query"]))
        label = 1 if truth in POSITIVE_TRUTHS else 0 if truth in NEGATIVE_TRUTHS else None
        if label is None:
            skipped["ground truth %s" % (truth or "missing")] += 1
            continue
        ev = next((e for e in bd.events if e.get("type") == "verify" and e.get("session") == cap["session"]
                   and e.get("query") == cap["query"] and fb_load.num(e.get("attempt")) == cap["attempt"]), None)
        frame_path = bd.find((ev or {}).get("frameFile"), cap["session"])
        if ev is None or frame_path is None:
            skipped["no saved frame"] += 1
            continue
        scores = [s for s in ev.get("scores") or [] if isinstance(s, dict) and isinstance(s.get("rect"), list)]
        if not scores:
            skipped["no candidate rectangles"] += 1
            continue
        item, kind, target = capture_item(bd, cap)
        stored_path = bd.entry_crop_path(target.get("id")) if kind == "inv" else bd.record_crop_path(target)
        if stored_path is None:
            skipped["no stored crop"] += 1
            continue
        frame = image(frame_path)
        rotation = ev.get("rotation") or 0
        ckey = "%s:%s:%d" % (cap["session"], cap["query"], int(cap["attempt"] or 0))
        cand_keys = []
        for k, s in enumerate(scores):
            key = "cand:%s:%d" % (ckey, k)
            ps.add_crop(key, crop_from_upright(frame, s["rect"], rotation), kind="fresh", item=item,
                        capture=ckey, cand_kind=s.get("kind"), logged_sim=fb_load.num(s.get("sim")))
            cand_keys.append(key)
        skey = "stored:" + item
        ps.add_crop(skey, image(stored_path), kind="stored", item=item, path=stored_path)
        ps.pairs.append({"set": "verify", "a": tuple(cand_keys), "b": skey, "label": label,
                         "item_a": item, "item_b": item, "dup": False, "capture": ckey})
        ps.captures.append({"key": ckey, "session": cap["session"], "query": cap["query"],
                            "attempt": cap["attempt"], "item": item, "kind": kind, "truth": truth,
                            "label": label, "cand_keys": cand_keys, "stored_key": skey,
                            "logged_best": cap["sim"], "emb_file": cap.get("emb_file"),
                            "target_embedding": (target or {}).get("embedding")})

        # impostors: other items known at the capture time
        t = cap.get("t")
        others = {}
        for rid, rec in bd.ds.history(cap["folder"]).active_at(t).items():
            oitem = "rec:%s" % fb_load.lineage_of(rec)
            if oitem == item or oitem in others or rec.get("sessionId") in bd.excluded:
                continue
            p = bd.record_crop_path(rec)
            if p:
                others[oitem] = (p, False)
        for eid, ent in bd.entries.items():
            oitem = "inv:" + eid
            first = fb_load.num(ent.get("firstSeenMs"))
            gone = bd.entry_deleted_ms.get(eid)
            if oitem == item or (first is not None and t is not None and first > t) or \
                    (gone is not None and t is not None and gone <= t):
                continue
            p = bd.entry_crop_path(eid)
            if p:
                others[oitem] = (p, kind == "inv" and dup_entries(target, ent))
        for oitem, (p, dup) in sorted(others.items()):
            okey = "stored:" + oitem
            ps.add_crop(okey, image(p), kind="stored", item=oitem, path=p)
            ps.pairs.append({"set": "verify_imp", "a": tuple(cand_keys), "b": okey, "label": 0,
                             "item_a": item, "item_b": oitem, "dup": dup, "capture": ckey})

        # cross pairs: scanned entries only
        if kind == "inv":
            for dkey, eid in det_keys:
                same = "inv:" + eid == item
                ps.pairs.append({"set": "cross", "a": tuple(cand_keys), "b": dkey,
                                 "label": label if same else 0, "item_a": item, "item_b": "inv:" + eid,
                                 "dup": (not same) and dup_entries(target, bd.entries[eid]), "capture": ckey})
    if skipped:
        ps.notes.append("captures left out: " + ", ".join("%s %d" % kv for kv in sorted(skipped.items())))
    ps.notes.append("captures used: %d (%d same, %d different)" % (
        len(ps.captures), sum(c["label"] == 1 for c in ps.captures), sum(c["label"] == 0 for c in ps.captures)))
    return ps


# ----------------------------------------------------------------------------- metrics

def _prep(scores, labels, weights):
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    w = np.ones(s.size) if weights is None else np.asarray(weights, float)
    keep = w > 0
    return s[keep], y[keep], w[keep]


def auc_w(scores, labels, weights=None):
    """Weighted ROC AUC = P(score of a same pair > score of a different pair), ties 1/2."""
    s, y, w = _prep(scores, labels, weights)
    P, N = w[y].sum(), w[~y].sum()
    if P <= 0 or N <= 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s, y, w = s[order], y[order], w[order]
    _, start = np.unique(s, return_index=True)
    wp = np.add.reduceat(np.where(y, w, 0.0), start)
    wn = np.add.reduceat(np.where(~y, w, 0.0), start)
    below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (P * N))


def roc_w(scores, labels, weights=None):
    """(fpr, tpr, thresholds) for 'same when score >= t', from (0, 0) at +inf to (1, 1)."""
    s, y, w = _prep(scores, labels, weights)
    P, N = w[y].sum(), w[~y].sum()
    if s.size == 0 or P <= 0 or N <= 0:
        return np.array([0.0]), np.array([0.0]), np.array([np.inf])
    order = np.argsort(-s, kind="mergesort")
    s, y, w = s[order], y[order], w[order]
    ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
    tp = np.cumsum(np.where(y, w, 0.0))[ends]
    fp = np.cumsum(np.where(~y, w, 0.0))[ends]
    return np.r_[0.0, fp / N], np.r_[0.0, tp / P], np.r_[np.inf, s[ends]]


def eer_w(scores, labels, weights=None):
    """Equal error rate: where the false-accept and false-reject rates cross on the ROC
    polyline (linear interpolation between operating points)."""
    fpr, tpr, _ = roc_w(scores, labels, weights)
    if fpr.size < 2:
        return float("nan")
    d = fpr - (1.0 - tpr)              # rises from -1 to +1
    i = int(np.argmax(d >= 0))
    if d[i] == 0 or i == 0:
        return float(fpr[i])
    f = -d[i - 1] / (d[i] - d[i - 1])
    return float(fpr[i - 1] + f * (fpr[i] - fpr[i - 1]))


def tar_at_far_w(scores, labels, weights=None, far=FAR_TARGET):
    """Largest true-accept rate among thresholds whose false-accept rate is at most `far`."""
    fpr, tpr, _ = roc_w(scores, labels, weights)
    if fpr.size < 2:
        return float("nan")
    ok = fpr <= far + 1e-12
    return float(tpr[ok].max())


def rates_at_w(scores, labels, threshold, weights=None):
    """(true-accept rate, false-accept rate) of 'same when score >= threshold'."""
    s, y, w = _prep(scores, labels, weights)
    P, N = w[y].sum(), w[~y].sum()
    acc = s >= threshold
    tar = float(w[y & acc].sum() / P) if P > 0 else float("nan")
    far = float(w[~y & acc].sum() / N) if N > 0 else float("nan")
    return tar, far


METRICS = ("auc", "eer", "tar5")


def metric_vector(scores, labels, weights=None):
    return np.array([auc_w(scores, labels, weights), eer_w(scores, labels, weights),
                     tar_at_far_w(scores, labels, weights)], float)


def item_weights(items_a, items_b, counts, index):
    """Pair weights for one resample: c_a when both items are the same, else c_a * c_b."""
    ia = np.array([index[a] for a in items_a])
    ib = np.array([index[b] for b in items_b])
    ca, cb = counts[ia], counts[ib]
    return np.where(ia == ib, ca, ca * cb).astype(float)


def item_bootstrap(items_a, items_b, stat, n_boot=BOOT_N, seed=BOOT_SEED, alpha=BOOT_ALPHA,
                   min_items=MIN_BOOT_ITEMS):
    """Percentile interval of stat(weights) -> 1-D array, with items resampled with replacement.

    Every pair counts c_a times when its two items are one item, c_a * c_b times otherwise, where
    c is how often the item was drawn (0 when not drawn). Returns dict(lo, hi, valid, n_items, ok).
    """
    items = sorted(set(items_a) | set(items_b))
    index = {it: i for i, it in enumerate(items)}
    k = len(items)
    full = np.atleast_1d(np.asarray(stat(np.ones(len(items_a))), float))
    if k < min_items:
        nan = np.full(full.shape, np.nan)
        return {"lo": nan, "hi": nan, "valid": np.zeros(full.shape), "n_items": k, "ok": False}
    ia = np.array([index[a] for a in items_a])
    ib = np.array([index[b] for b in items_b])
    same = ia == ib
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot,) + full.shape)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, k, size=k), minlength=k).astype(float)
        w = np.where(same, counts[ia], counts[ia] * counts[ib])
        draws[b] = np.atleast_1d(np.asarray(stat(w), float))
    valid = np.mean(~np.isnan(draws), axis=0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = np.nanpercentile(draws, 100 * alpha / 2, axis=0)
        hi = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0)
    bad = valid < BOOT_MIN_VALID
    return {"lo": np.where(bad, np.nan, lo), "hi": np.where(bad, np.nan, hi), "valid": valid,
            "n_items": k, "ok": True}


def pair_counts(labels, items_a, items_b):
    """n same, n different, distinct items, and distinct items with a same pair."""
    y = np.asarray(labels, int)
    items = set(items_a) | set(items_b)
    pos_items = {a for a, l in zip(items_a, y) if l == 1}
    return {"n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()), "n_items": len(items),
            "n_pos_items": len(pos_items)}


# ----------------------------------------------------------------------------- formatting

def fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else ("%.*f" % (nd, x))


def fmt_ci(v, lo, hi, nd=3):
    if v is None or not math.isfinite(v):
        return "–"
    if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi)):
        return fmt(v, nd)
    return "%s [%s, %s]" % (fmt(v, nd), fmt(lo, nd), fmt(hi, nd))
