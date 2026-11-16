#!/usr/bin/env python3
"""Loading and joining FindBack logs for the paper pipeline (stdlib + numpy).

Reads one or more data folders pulled from phones and turns them into flat rows:
queries (query_start + verify + verify_label + query_end), captures, map fixes and
registrations. Nothing in this file computes a paper number; see fb_metrics.py.

Files read in a data folder (only events.jsonl is required):
  events.jsonl            the event log
  records.json            active object records
  records_log.jsonl       v0.4 record history: {t, session, op, record}
  map.json, archive/*/map.json   home maps (fix points, rooms, walk-graph nodes)
  verify/<vid>_emb.json   v0.4 candidate embeddings of one verification
  SYNTHETIC_DATA.txt      marker written by synth_generate.py

Selection rule (STUDY_SPEC_v0.4 section 4): an event counts for the paper only when
the study tag in effect when it was logged is non-empty. The tag in effect is the
event's own `studyTag` when it has one, otherwise the last `studyTag` seen earlier in
the same session (from session_start, study_settings or any tagged event).

Field names of the v0.4 events come from the spec, not from a real log, so every v0.4
field is read with .get() and a missing value becomes None. The exact list of fields
is in analysis/README.md ("Event fields the pipeline reads").
"""
import base64
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

SYNTHETIC_MARKER_FILE = "SYNTHETIC_DATA.txt"
SYNTHETIC_SESSION_PREFIX = "SYNTH"
SYNTHETIC_TAG_KEY = "synthetic"
SYNTHETIC_EVENT_KEY = "synthetic"

GROUND_TRUTHS = ("present", "displaced", "absent", "replaced", "skip")
LABELS = ("target", "empty", "other", "lookalike")
CONDITIONS = ("default", "near", "far", "angled", "dim", "bright", "occluded")
CAPTURE_MODES = ("FREE", "PAIRED", "BENCHMARK")

# Walk graph constant mirrored from FindBack map/WalkGraph.kt (ROOM_LOOKUP_M): the room of a
# point is the room of the nearest labelled node within this horizontal distance.
ROOM_LOOKUP_M = 1.5


# ----------------------------------------------------------------------------- small helpers

def num(x):
    """A finite float for numbers and numeric strings, else None. Booleans are not numbers."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def as_bool(x):
    """True/False for booleans and the usual spellings, else None."""
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and x in (0, 1):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    return None


def first_key(d, *keys):
    """Value of the first of `keys` present and not None in dict `d` (tolerates alias names)."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def parse_tag(raw):
    """'home=A;exp=E3;round=2' -> {'home': 'A', 'exp': 'E3', 'round': '2'}.

    Keys are lower-cased; values are stripped. Parts without '=' are collected under the key
    '_malformed' (a list) so the caller can warn about them.
    """
    out = {}
    if not isinstance(raw, str):
        return out
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            out.setdefault("_malformed", []).append(part)
            continue
        k, v = part.split("=", 1)
        k = k.strip().lower()
        if k:
            out[k] = v.strip()
    return out


def tag_exp(tag):
    v = tag.get("exp")
    return v.upper() if isinstance(v, str) and v else None


def tag_split(tag):
    v = tag.get("split")
    v = v.lower() if isinstance(v, str) else None
    return v if v in ("dev", "test") else None


def tag_home(tag):
    v = tag.get("home")
    return v if isinstance(v, str) and v else None


def read_jsonl(path):
    """Returns (rows, number_of_unreadable_lines). Blank lines are skipped."""
    rows, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                bad += 1
    return rows, bad


def decode_embedding(x, dim=None):
    """A float32 vector from a JSON list of numbers or a base64 string of little-endian float32.

    Returns None when the value is unusable or its length disagrees with `dim`.
    """
    try:
        if isinstance(x, str):
            raw = base64.b64decode(x, validate=False)
            if len(raw) == 0 or len(raw) % 4:
                return None
            v = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        elif isinstance(x, (list, tuple)):
            v = np.asarray(x, dtype=np.float32)
        else:
            return None
    except (ValueError, TypeError):
        return None
    if v.ndim != 1 or v.size == 0 or not np.all(np.isfinite(v)):
        return None
    if dim is not None and int(dim) != v.size:
        return None
    return v


def cosine(a, b):
    """Cosine similarity as in the app's Embedder.cosine (0 when a norm is 0 or sizes differ)."""
    if a is None or b is None or a.shape != b.shape or a.size == 0:
        return 0.0
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = math.sqrt(float(a @ a)) * math.sqrt(float(b @ b))
    return float(a @ b) / denom if denom else 0.0


# ----------------------------------------------------------------------------- folders

class Folder:
    """One data folder as pulled from a phone."""

    def __init__(self, path):
        self.path = path
        self.events, self.bad_lines = [], 0
        ev_path = os.path.join(path, "events.jsonl")
        self.has_events = os.path.exists(ev_path)
        if self.has_events:
            self.events, self.bad_lines = read_jsonl(ev_path)

        self.records = []
        rec_path = os.path.join(path, "records.json")
        self.has_records = os.path.exists(rec_path)
        if self.has_records:
            try:
                with open(rec_path, encoding="utf-8") as f:
                    data = json.load(f)
                self.records = [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
            except (json.JSONDecodeError, OSError):
                self.records = []

        self.records_log = None  # None = no file (fall back to records.json)
        log_path = os.path.join(path, "records_log.jsonl")
        if os.path.exists(log_path):
            self.records_log, bad = read_jsonl(log_path)
            self.bad_lines += bad

        # mapId -> (map dict, file size in bytes, is_current)
        self.maps = {}
        self.current_map_bytes = None
        candidates = [os.path.join(path, "map.json")] + sorted(glob.glob(os.path.join(path, "archive", "*", "map.json")))
        for i, p in enumerate(candidates):
            if not os.path.exists(p):
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    m = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(m, dict):
                continue
            size = os.path.getsize(p)
            if i == 0:
                self.current_map_bytes = size
            self.maps.setdefault(m.get("mapId"), (m, size, i == 0))

        self.synthetic_marker = os.path.exists(os.path.join(path, SYNTHETIC_MARKER_FILE))
        self.synthetic_records = any(r.get(SYNTHETIC_EVENT_KEY) for r in self.records) or any(
            isinstance(e.get("record"), dict) and e["record"].get(SYNTHETIC_EVENT_KEY) for e in (self.records_log or [])
        )


class Dataset:
    """Events of all folders, de-duplicated, each annotated with the study tag in effect.

    Added keys on every event: _folder (index into .folders), _tag_raw (str), _tag (dict),
    _mode (capture mode in effect or None), _seq (position in the merged list).
    """

    def __init__(self, paths):
        self.folders = [Folder(p) for p in paths]
        self.events = []
        self.duplicates = 0
        self.session_folders = defaultdict(list)   # session -> [folder index] in the order seen
        self.synthetic_sessions = defaultdict(set)  # session -> {reason}
        seen = set()
        for fi, folder in enumerate(self.folders):
            state = {}  # session -> [tag_raw, mode], carried forward in file order
            for e in folder.events:
                session = e.get("session")
                st = state.setdefault(session, ["", None])
                if isinstance(e.get("studyTag"), str):
                    st[0] = e["studyTag"].strip()
                if isinstance(e.get("captureMode"), str) and e["captureMode"].strip():
                    st[1] = e["captureMode"].strip().upper()
                key = json.dumps(e, sort_keys=True)
                if key in seen:
                    self.duplicates += 1
                    continue
                seen.add(key)
                e = dict(e)
                e["_folder"] = fi
                e["_tag_raw"] = st[0]
                e["_tag"] = parse_tag(st[0])
                e["_mode"] = st[1]
                e["_seq"] = len(self.events)
                self.events.append(e)
                if fi not in self.session_folders[session]:
                    self.session_folders[session].append(fi)
                for reason in self._synthetic_reasons(e, folder):
                    self.synthetic_sessions[session].add(reason)
        self._histories = {}

    @staticmethod
    def _synthetic_reasons(e, folder):
        reasons = []
        if e.get(SYNTHETIC_EVENT_KEY):
            reasons.append("event field synthetic=true")
        if isinstance(e.get("session"), str) and e["session"].upper().startswith(SYNTHETIC_SESSION_PREFIX):
            reasons.append("session id starts with " + SYNTHETIC_SESSION_PREFIX)
        if SYNTHETIC_TAG_KEY in e["_tag"]:
            reasons.append("study tag has key " + SYNTHETIC_TAG_KEY)
        if folder.synthetic_marker:
            reasons.append("folder has " + SYNTHETIC_MARKER_FILE)
        if folder.synthetic_records:
            reasons.append("folder records are marked synthetic")
        return reasons

    @property
    def any_synthetic(self):
        return bool(self.synthetic_sessions) or any(f.synthetic_marker or f.synthetic_records for f in self.folders)

    def sessions(self):
        return sorted({e.get("session") for e in self.events}, key=lambda s: str(s))

    def session_tags(self):
        """session -> sorted list of the distinct raw tags in effect during the session."""
        tags = defaultdict(set)
        for e in self.events:
            tags[e.get("session")].add(e["_tag_raw"])
        return {s: sorted(t) for s, t in tags.items()}

    def tagged(self, excluded_sessions=()):
        """Events that count for the paper: non-empty tag in effect, session not excluded."""
        ex = set(excluded_sessions)
        return [e for e in self.events if e["_tag_raw"] and e.get("session") not in ex]

    def find_file(self, session, relpath):
        """Absolute path of `relpath` in a folder that holds this session's events, or None."""
        if not isinstance(relpath, str) or not relpath:
            return None
        for fi in self.session_folders.get(session, []):
            p = os.path.join(self.folders[fi].path, relpath)
            if os.path.exists(p):
                return p
        return None

    def history(self, folder_index):
        if folder_index not in self._histories:
            self._histories[folder_index] = RecordHistory(self.folders[folder_index])
        return self._histories[folder_index]


# ----------------------------------------------------------------------------- record history

class RecordHistory:
    """Which records were active, and with which embedding, at a given time.

    Source 'records_log.jsonl' replays add / rename / delete / supersede up to the capture
    time. Records of records.json that the log never mentions (saved before the log existed)
    are treated as active from their lastSeenMs on. Source 'records.json' is the fallback when
    there is no log: active records with lastSeenMs <= t, so records deleted or superseded
    since then are missing. Source 'none' means neither file exists.
    """

    def __init__(self, folder):
        self.unknown_ops = 0
        self._log = []
        if folder.records_log is not None:
            self.source = "records_log.jsonl"
            for i, e in enumerate(folder.records_log):
                rec = e.get("record")
                if isinstance(rec, dict) and rec.get("id") is not None:
                    self._log.append((num(e.get("t")), i, str(e.get("op") or "").lower(), rec))
            # File order is append order; sort by time only to make the cut-off well defined.
            self._log.sort(key=lambda r: (r[0] if r[0] is not None else float("-inf"), r[1]))
        elif folder.has_records:
            self.source = "records.json"
        else:
            self.source = "none"
        logged = {r[3]["id"] for r in self._log}
        self._static = [r for r in folder.records if r.get("id") is not None and r["id"] not in logged]
        self._emb = {}

    def active_at(self, t_ms):
        """{record id: record dict} for records active at phone time `t_ms`."""
        active = {}
        for r in self._static:
            seen = num(r.get("lastSeenMs"))
            if seen is None or t_ms is None or seen <= t_ms:
                active[r["id"]] = r
        for t, _, op, rec in self._log:
            if t is not None and t_ms is not None and t > t_ms:
                break
            rid = rec["id"]
            old = rec.get("supersedes")
            if op == "add":
                active[rid] = rec
                if old is not None:
                    active.pop(old, None)
            elif op == "rename":
                if rid in active:
                    active[rid] = rec
            elif op == "delete":
                active.pop(rid, None)
            elif op == "supersede":
                # Either the new record (it names the one it replaces) or the old one leaving.
                if old is not None:
                    active.pop(old, None)
                    active[rid] = rec
                else:
                    active.pop(rid, None)
            else:
                self.unknown_ops += 1
        return active

    def embedding(self, rec):
        key = id(rec)
        if key not in self._emb:
            self._emb[key] = decode_embedding(rec.get("embedding"))
        return self._emb[key]


def lineage_of(rec):
    """lineageId of a record; old records without the field are their own lineage (spec S7)."""
    if not isinstance(rec, dict):
        return None
    return rec.get("lineageId") if rec.get("lineageId") is not None else rec.get("id")


# ----------------------------------------------------------------------------- embedding files

def load_emb_file(path):
    """Parse verify/<vid>_emb.json. Returns {'model', 'dim', 'record', 'candidates': [...]} or None.

    Accepted key aliases (the spec names the content, not the keys):
      model | modelName;  dim | dimension | embeddingDim;  record | target | targetRecord | recordId;
      candidates | scores;  per candidate: kind;  rect | rectangle;  sim | similarity;
      embedding | emb | vector (base64 little-endian float32, or a JSON list).
    """
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(d, dict):
        return None
    dim = num(first_key(d, "dim", "dimension", "embeddingDim"))
    cands = first_key(d, "candidates", "scores")
    if not isinstance(cands, list):
        return None
    out = []
    for c in cands:
        if not isinstance(c, dict):
            continue
        emb = decode_embedding(first_key(c, "embedding", "emb", "vector"), dim)
        out.append({
            "kind": c.get("kind"),
            "rect": first_key(c, "rect", "rectangle"),
            "sim": num(first_key(c, "sim", "similarity")),
            "embedding": emb,
        })
    return {
        "model": first_key(d, "model", "modelName"),
        "dim": int(dim) if dim is not None else None,
        "record": first_key(d, "record", "target", "targetRecord", "recordId"),
        "candidates": out,
    }


def margin_inputs(ds, capture):
    """Similarity of the best candidate to the best-matching *other* record at capture time.

    Returns a dict:
      ok            True when the margin can be evaluated
      reason        why not, when ok is False
      best_sim      similarity of the best candidate to the target, recomputed from embeddings
                    when the target record is known, else the value stored in the file
      other_sim     highest cosine of that candidate to any other active record (None when the
                    target is the only active record: then there is nothing to confuse it with)
      other_record  id of that record
      n_others      number of other active records compared
      source        'records_log.jsonl' or 'records.json'
      logged_diff   |recomputed - stored| similarity of the best candidate (None if unknown)
    "Other" excludes the target record and every record of the target's lineage.
    """
    path = ds.find_file(capture["session"], capture.get("emb_file")) if capture.get("emb_file") else None
    if path is None:
        guess = "verify/%s_%s_emb.json" % (capture["query"], capture.get("attempt"))
        path = ds.find_file(capture["session"], guess)
    if path is None:
        return {"ok": False, "reason": "no embedding file"}
    emb = load_emb_file(path)
    if emb is None:
        return {"ok": False, "reason": "unreadable embedding file"}
    cands = [c for c in emb["candidates"] if c["embedding"] is not None]
    if not cands:
        return {"ok": False, "reason": "no candidate embedding"}
    hist = ds.history(capture["folder"])
    if hist.source == "none":
        return {"ok": False, "reason": "no records file"}
    active = hist.active_at(capture.get("t"))
    target_id = capture.get("record") if capture.get("record") is not None else emb["record"]
    # The target may have been deleted or superseded after the capture; look it up as of the capture.
    target = active.get(target_id)
    target_emb = hist.embedding(target) if target is not None else None

    def sim_to_target(c):
        if target_emb is not None and target_emb.shape == c["embedding"].shape:
            return cosine(c["embedding"], target_emb)
        return c["sim"] if c["sim"] is not None else float("-inf")

    best = max(cands, key=sim_to_target)
    best_sim = sim_to_target(best)
    if best_sim == float("-inf"):
        return {"ok": False, "reason": "no similarity for the best candidate"}
    lineage = lineage_of(target) if target is not None else target_id
    other_sim, other_id, n_others = None, None, 0
    for rid, rec in active.items():
        if rid == target_id or lineage_of(rec) == lineage:
            continue
        v = hist.embedding(rec)
        if v is None or v.shape != best["embedding"].shape:
            continue
        n_others += 1
        s = cosine(best["embedding"], v)
        if other_sim is None or s > other_sim:
            other_sim, other_id = s, rid
    stored = best["sim"]
    return {
        "ok": True, "reason": None, "best_sim": best_sim, "other_sim": other_sim, "other_record": other_id,
        "n_others": n_others, "source": hist.source,
        "logged_diff": abs(best_sim - stored) if stored is not None and target_emb is not None else None,
    }


# ----------------------------------------------------------------------------- maps

def fix_point_room(map_dict, fix_id):
    """Room id of a fix point: the room of the nearest labelled walk-graph node within
    ROOM_LOOKUP_M (horizontal distance in the map frame, x and z), as WalkGraph.roomAt does.
    Returns None when the map, the fix point or a labelled node nearby is missing."""
    if not isinstance(map_dict, dict) or fix_id is None:
        return None
    fix = next((f for f in map_dict.get("fixPoints") or [] if isinstance(f, dict) and f.get("id") == fix_id), None)
    if fix is None:
        return None
    fx, fz = num(fix.get("x")), num(fix.get("z"))
    if fx is None or fz is None:
        return None
    best, best_d = None, ROOM_LOOKUP_M
    for n in map_dict.get("nodes") or []:
        if not isinstance(n, dict) or n.get("roomId") is None:
            continue
        x, z = num(n.get("x")), num(n.get("z"))
        if x is None or z is None:
            continue
        d = math.hypot(x - fx, z - fz)
        if d <= best_d:
            best, best_d = n["roomId"], d
    return best


# ----------------------------------------------------------------------------- queries

def _capture_from(e):
    lat = e.get("latency") if isinstance(e.get("latency"), dict) else {}
    sim = num(e.get("bestSimilarity"))
    if sim is None and isinstance(e.get("scores"), list):
        sims = [num(s.get("sim")) for s in e["scores"] if isinstance(s, dict)]
        sims = [s for s in sims if s is not None]
        sim = max(sims) if sims else None
    return {
        "session": e.get("session"), "query": e.get("query"), "record": e.get("record"),
        "folder": e["_folder"], "t": num(e.get("t")), "seq": e["_seq"],
        "attempt": num(e.get("attempt")),
        "sim": sim,
        "threshold": num(e.get("threshold")),
        "inference_calls": num(e.get("inferenceCalls")),
        "latency": {k: num(lat.get(k)) for k in ("convertMs", "detectMs", "embedMs", "totalMs")},
        "emb_file": e.get("embFile") if isinstance(e.get("embFile"), str) else None,
        "ms_since_near": num(e.get("msSinceNear")),
        "label": None, "condition": None, "label_events": 0,
        "tag": e["_tag"], "tag_raw": e["_tag_raw"], "capture_mode": e["_mode"],
        "exp": tag_exp(e["_tag"]), "split": tag_split(e["_tag"]), "home": tag_home(e["_tag"]),
    }


def build_queries(events):
    """Join the events of each (session, query) into one row. `events` is a list from
    Dataset.tagged() (or Dataset.events). Returns (queries, captures, stats).

    A query needs a query_start. Its study tag, experiment, split and site are those in effect
    at query_start. `truth` is None when the query has no query_end or no usable groundTruth.
    `captures` holds every verify event (also those whose query has no query_start).
    `stats` counts what could not be joined: 'unmatched_labels' (a verify_label with no verify
    of that attempt), 'queries_without_start', 'relabelled' (captures labelled more than once;
    the last label wins).
    """
    groups = defaultdict(lambda: {"start": None, "end": None, "captures": []})
    order = []
    stats = {"unmatched_labels": 0, "queries_without_start": 0, "relabelled": 0}
    for e in events:
        t = e.get("type")
        if t not in ("query_start", "verify", "verify_label", "query_end") or e.get("query") is None:
            continue
        key = (e.get("session"), e.get("query"))
        if key not in groups:
            order.append(key)
        g = groups[key]
        if t == "query_start":
            if g["start"] is None:
                g["start"] = e
        elif t == "query_end":
            g["end"] = e  # the last one wins
        elif t == "verify":
            g["captures"].append(_capture_from(e))
        else:  # verify_label: by attempt when given, else the latest capture so far
            att = num(e.get("attempt"))
            target = None
            if att is not None:
                target = next((c for c in g["captures"] if c["attempt"] == att), None)
            elif g["captures"]:
                target = g["captures"][-1]
            if target is None:
                stats["unmatched_labels"] += 1
            else:
                lab = e.get("label")
                target["label"] = lab.strip().lower() if isinstance(lab, str) and lab.strip() else None
                cond = e.get("condition")
                target["condition"] = cond.strip().lower() if isinstance(cond, str) and cond.strip() else None
                target["label_events"] += 1

    queries, captures = [], []
    for key in order:
        g = groups[key]
        caps = sorted(g["captures"], key=lambda c: (c["attempt"] if c["attempt"] is not None else float("inf"), c["seq"]))
        captures.extend(caps)
        stats["relabelled"] += sum(c["label_events"] > 1 for c in caps)
        s, end = g["start"], g["end"]
        if s is None:
            stats["queries_without_start"] += 1
            continue
        tag = s["_tag"]
        truth = None
        marker = {}
        if end is not None:
            gt = end.get("groundTruth")
            truth = gt.strip().lower() if isinstance(gt, str) else None
            if truth not in GROUND_TRUTHS:
                truth = None
            marker = end.get("markerError") if isinstance(end.get("markerError"), dict) else {}
        measured = num(marker.get("errorM"))
        mob = s.get("mobility")
        mob = mob.strip().lower() if isinstance(mob, str) else None
        usable_caps = [c for c in caps if c["sim"] is not None]
        q = {
            "session": key[0], "query": key[1], "folder": s["_folder"],
            "record": s.get("record"),
            "t_start": num(s.get("t")), "t_end": num(end.get("t")) if end else None,
            "tag_raw": s["_tag_raw"], "tag": tag,
            "exp": tag_exp(tag), "split": tag_split(tag), "home": tag_home(tag),
            "capture_mode": s["_mode"],
            "ended": end is not None,
            "truth": truth,
            # Measurement at Done. For `present` it is the marker error; for `displaced` it is
            # the displacement of the item from the marker.
            "marker_error_m": measured if truth == "present" else None,
            "displacement_m": measured if truth == "displaced" else None,
            # markerError.displaced, kept only to report a disagreement with the ground truth.
            "marker_displaced_flag": as_bool(marker.get("displaced")),
            "marker_source": marker.get("positionSource") or s.get("positionSource"),
            "marker_meters_since_fix": num(marker.get("metersSinceFix")),
            "marker_fix_point": marker.get("fixPoint"),
            # Pre-capture signals (spec section 2), all from query_start.
            "age_min": num(s.get("ageMin")),
            "mobility": mob if mob in ("rare", "some", "often") else None,
            "move_count": num(s.get("moveCount")),
            "position_source": s.get("positionSource"),
            "meters_since_fix": num(s.get("metersSinceFix")),
            "losses_since_fix": num(s.get("trackingLossesSinceFix")),
            "ms_since_fix": num(s.get("msSinceFix")),
            "fix_type": s.get("fixType"),
            "fix_point": s.get("fixPoint"),
            "lineage": s.get("lineageId") if s.get("lineageId") is not None else s.get("record"),
            "policy_verify": as_bool(s.get("policyVerify")),
            "item_room": s.get("itemRoom"), "item_room_name": s.get("itemRoomName"),
            "user_room": s.get("userRoom"),
            "map_id": s.get("mapId"),
            "no_capture_reason": end.get("noCaptureReason") if end else None,
            # Guidance: did the query reach NEAR (the ring on the item), and after how long.
            "near_reached": as_bool(end.get("nearReached")) if end else None,
            "time_to_near_ms": num(end.get("timeToNearMs")) if end else None,
            "captures": caps,
            "first_capture": usable_caps[0] if usable_caps else None,
        }
        queries.append(q)
    return queries, captures, stats


def fixes_from(events):
    """Rows for map_fix events."""
    rows = []
    for e in events:
        if e.get("type") != "map_fix":
            continue
        tag = e["_tag"]
        rows.append({
            "session": e.get("session"), "t": num(e.get("t")),
            "kind": e.get("kind"), "fix_type": e.get("fixType"),
            "applied": True if e.get("applied") is None else bool(e.get("applied")),
            "jump_m": num(e.get("jumpM")), "jump_deg": num(e.get("jumpDeg")),
            "disagreement": bool(e.get("disagreement")),
            "accepted": as_bool(e.get("accepted")),
            "exp": tag_exp(tag), "split": tag_split(tag), "home": tag_home(tag),
        })
    return rows


def registrations_from(events):
    rows = []
    for e in events:
        if e.get("type") != "register":
            continue
        tag = e["_tag"]
        lat = e.get("latency") if isinstance(e.get("latency"), dict) else {}
        rows.append({
            "session": e.get("session"), "record": e.get("record"), "t": num(e.get("t")),
            "latency": {k: num(lat.get(k)) for k in ("convertMs", "detectMs", "embedMs")},
            "record_bytes": num(e.get("recordBytes")),
            "room": e.get("room"),
            "lineage": e.get("lineageId") if e.get("lineageId") is not None else e.get("record"),
            "exp": tag_exp(tag), "split": tag_split(tag), "home": tag_home(tag),
        })
    return rows
