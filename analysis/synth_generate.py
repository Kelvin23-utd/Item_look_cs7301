#!/usr/bin/env python3
"""Synthetic FindBack logs. FOR TESTING THE ANALYSIS CODE ONLY (stdlib + numpy).

Nothing this script writes is a measurement. Every relationship in its output (which items
move, how similar a capture is, how large a marker error is) is an arbitrary rule chosen to
exercise code paths, and says nothing about the real system.

It writes two data folders (two sites) shaped like folders pulled from the phone under spec
v0.4, plus fake power-run files:

  <out>/site_A/, <out>/site_B/   events.jsonl, records.json, records_log.jsonl, map.json,
                                 verify/<vid>_emb.json, SYNTHETIC_DATA.txt
  <out>/power_runs/              files in the format of power_measure.py

How synthetic data is marked (paper_figures.py checks every one of these):
  * every event, record and power run carries "synthetic": true;
  * every session id starts with SYNTH, every record id with SYN, every item name with SYNTHETIC;
  * every study tag starts with synthetic=1;
  * each folder holds SYNTHETIC_DATA.txt.

What it covers: all five ground-truth values, the three capture modes, benchmark labels and
conditions (with a dismissed label and a relabel), embedding files, record history with add /
rename / delete / supersede (both plausible shapes of the supersede entry), first / refresh /
manual map fixes with a declined and an accepted disagreement, dev and test splits (E2, E3),
an experiment without splits (E5), two sites, a session whose tag is set half-way, queries
without a capture, and an unfinished query.

Usage:
  python3 analysis/synth_generate.py                     # writes under the system temp directory
  python3 analysis/synth_generate.py --out DIR --seed 3
  python3 analysis/synth_generate.py --out DIR --variant minimal   # v0.3-shaped events with only a tag
  python3 analysis/synth_generate.py --out DIR --items 4 --rounds 1  # a tiny set ("n too small" paths)
"""
import argparse
import base64
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

MARKER_FILE = "SYNTHETIC_DATA.txt"
MARKER_TEXT = (
    "SYNTHETIC DATA. Written by analysis/synth_generate.py to test the analysis code.\n"
    "Nothing in this folder was measured. Do not use any number derived from it in the paper.\n"
)
DIM = 32
T0_MS = 1_900_000_000_000          # 2030-03-17, far from any real session
MATCH_THRESHOLD = 0.75             # the app's VerificationPolicy.MATCH_THRESHOLD
CONDITIONS = ["default", "near", "far", "angled", "dim", "bright", "occluded"]
CONDITION_SIM = {"default": 0.84, "near": 0.82, "far": 0.74, "angled": 0.72, "dim": 0.64, "bright": 0.76, "occluded": 0.58}
MOBILITY = ["rare", "some", "often"]
MOVE_PROB = {"rare": 0.03, "some": 0.12, "often": 0.30}

# v0.4 fields removed by --variant minimal (the result has v0.3-shaped events plus studyTag).
V04_FIELDS = {"captureMode", "mobility", "lineageId", "moveCount", "lineageAgeMin", "embFile", "msSinceQueryStart",
              "msSinceNear", "noCaptureReason", "msToFirstVerify"}


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def vec_with_sim(u, s, rng):
    """A unit vector whose cosine to unit vector u is s (up to float32 rounding)."""
    w = rng.normal(size=u.size)
    w -= (w @ u) * u
    w /= np.linalg.norm(w)
    s = float(np.clip(s, -0.99, 0.99))
    return unit(s * u.astype(np.float64) + math.sqrt(1 - s * s) * w)


def cos(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def b64(v):
    return base64.b64encode(v.astype("<f4").tobytes()).decode("ascii")


class Site:
    def __init__(self, name, out_dir, rng, opts):
        self.name, self.dir, self.rng, self.o = name, out_dir, rng, opts
        self.t = T0_MS + (0 if name == "A" else 3_600_000)
        self.events, self.records_log = [], []
        self.records = {}          # active records by id
        self.items = []
        self.n_session = self.n_query = self.n_record = 0
        self.session = None
        self.map_id = "SYNmap" + name
        self.rooms = [("SYNroom%s1" % name, "Living room", 0.0, 4.0), ("SYNroom%s2" % name, "Bedroom", 4.0, 8.0),
                      ("SYNroom%s3" % name, "Kitchen", 8.0, 11.0)]
        self.fix_points = {
            "IMAGE": {"id": "SYNfix%sI" % name, "name": "Home picture", "type": "IMAGE", "x": 0.5, "y": 1.2, "z": 0.5,
                      "room": self.rooms[0][0]},
            "SPOT": {"id": "SYNfix%sS" % name, "name": "Bedroom door", "type": "SPOT", "x": 5.0, "y": 1.3, "z": 1.0,
                     "room": self.rooms[1][0]},
        }
        os.makedirs(os.path.join(out_dir, "verify"), exist_ok=True)

    # ------------------------------------------------------------------ log plumbing
    def tick(self, lo=500, hi=4000):
        self.t += int(self.rng.integers(lo, hi))
        return self.t

    def log(self, etype, fields, tagged=False):
        e = {"t": self.tick(50, 400), "session": self.session["id"], "type": etype, "synthetic": True}
        if tagged:  # spec S1: these event types carry the study settings
            e["studyTag"] = self.session["tag"]
            e["captureMode"] = self.session["mode"]
        e.update(fields)
        self.events.append(e)
        return e

    def start_session(self, tag, mode, gap_ms=1_800_000):
        self.t += gap_ms
        self.n_session += 1
        self.session = {"id": "SYNTH%s%02d" % (self.name, self.n_session), "tag": tag, "mode": mode,
                        "fix": None, "meters": 0.0, "losses": 0, "fix_t": None}
        self.log("session_start", {"records": len(self.records), "device": "SYNTHETIC", "android": "0",
                                   "refWidthM": 0.116, "mapId": self.map_id, "mapNodes": 0, "mapFixPoints": 2,
                                   "mapRooms": 3}, tagged=True)
        self.log("model_loaded", {"model": "SYNTHETIC", "loadMs": 100.0})

    def tag(self, **kv):
        """synthetic=1 first, so nobody reading a tag can miss it."""
        parts = ["synthetic=1", "home=" + self.name] + ["%s=%s" % (k, v) for k, v in kv.items() if v is not None]
        return ";".join(parts)

    def proxies(self):
        s = self.session
        if s["fix"] is None:
            return {"fixPoint": None, "fixType": None, "metersSinceFix": None, "trackingLossesSinceFix": None,
                    "msSinceFix": None}
        return {"fixPoint": s["fix"]["id"], "fixType": s["fix"]["type"], "metersSinceFix": round(s["meters"], 2),
                "trackingLossesSinceFix": s["losses"], "msSinceFix": self.t - s["fix_t"]}

    def walk(self, metres):
        if self.session["fix"] is not None:
            self.session["meters"] += metres
        self.t += int(metres * 900)

    def fix(self, ftype, kind, jump=None, disagreement=False, accepted=None, applied=True):
        s = self.session
        f = self.fix_points[ftype]
        prev = self.proxies() if s["fix"] is not None else None
        first = s["fix"] is None
        e = {"mapId": self.map_id, "fixPoint": f["id"], "fixName": f["name"], "fixType": ftype,
             "kind": "first" if first else kind, "samples": 20, "posSpreadM": 0.01, "yawSpreadDeg": 0.5,
             "cameraDistanceM": 0.7 if ftype == "IMAGE" else None, "quality": "good",
             "jumpM": None if first else jump, "jumpHorizM": None if first else jump,
             "jumpDeg": None if first else round(float(jump * 20), 3),
             "metersSincePrevFix": prev["metersSinceFix"] if prev else None,
             "lossesSincePrevFix": prev["trackingLossesSinceFix"] if prev else None,
             "msSincePrevFix": prev["msSinceFix"] if prev else None,
             "prevFixPoint": prev["fixPoint"] if prev else None,
             "disagreement": disagreement, "accepted": accepted, "applied": applied, "anchored": True,
             "worldPose": {"x": 0, "y": 0, "z": 0, "yawDeg": 0}}
        self.log("map_fix", e)
        if applied:
            s["fix"], s["meters"], s["losses"], s["fix_t"] = f, 0.0, 0, self.t

    # ------------------------------------------------------------------ records
    def new_record(self, item, supersedes=None):
        self.n_record += 1
        rid = "SYNr%s%03d" % (self.name, self.n_record)
        old = self.records.get(supersedes) if supersedes else None
        located = self.session["fix"] is not None
        rec = {
            "id": rid, "name": item["name"], "rawLabel": "", "rawConfidence": 0.0, "boxSource": "detector",
            "cropFile": "crops/%s.jpg" % rid, "embedding": [float(x) for x in item["u"]], "refPosition": None,
            "worldPosition": [0.0, 0.0, 0.0], "sessionId": self.session["id"], "captureDistance": 0.8,
            "cropWidthPx": 300.0, "cropHeightPx": 300.0, "imageWidth": 1920, "imageHeight": 1080,
            "hitType": "DepthPoint", "lastSeenMs": self.t,
            "mapId": self.map_id if located else None,
            "mapPosition": item["pos"] if located else None, "viewPosition": item["pos"] if located else None,
            "roomId": item["room"],
            "mobility": item["mobility"],
            "lineageId": (old.get("lineageId") or old["id"]) if old else rid,
            "supersedes": supersedes,
            "moveCount": (old.get("moveCount", 0) + 1) if old else 0,
            "synthetic": True,
        }
        if self.o.variant == "minimal":
            for k in ("mobility", "lineageId", "supersedes", "moveCount"):
                rec.pop(k)
        return rec

    def log_record(self, op, rec):
        self.records_log.append({"t": self.t, "session": self.session["id"], "op": op, "record": dict(rec),
                                 "synthetic": True})

    def register(self, item):
        rec = self.new_record(item)
        self.records[rec["id"]] = rec
        item.update(record=rec["id"], lineage=rec["id"], lineage_t=self.t, state="home")
        self.log_record("add", rec)
        lat = {"convertMs": float(self.rng.normal(58, 6)), "detectMs": float(self.rng.normal(75, 12)),
               "embedMs": float(self.rng.normal(18, 4))}
        self.log("register", {
            "record": rec["id"], "userName": item["name"], "nameSource": "typed", "rawLabel": "", "boxSource": "detector",
            "hitType": "DepthPoint", "distance": 0.8, "refAvailable": False, "refPosition": None,
            "cropBytes": int(self.rng.integers(20000, 60000)), "recordBytes": int(self.rng.integers(40000, 80000)),
            "embeddingDim": DIM, "latency": lat, "mapId": rec["mapId"], "mapPosition": rec["mapPosition"],
            "viewPosition": rec["viewPosition"], "room": item["room"], "roomName": self.room_name(item["room"]),
            "located": rec["mapId"] is not None, "mobility": item["mobility"], "lineageId": rec["id"], "moveCount": 0,
        } | self.proxies(), tagged=True)
        return rec

    def update_location(self, item):
        """Spec S7: a new record supersedes the old one."""
        old = self.records[item["record"]]
        rec = self.new_record(item, supersedes=old["id"])
        del self.records[old["id"]]
        self.records[rec["id"]] = rec
        if self.name == "A":      # shape 1: one entry, the new record naming the one it replaces
            self.log_record("supersede", rec)
        else:                      # shape 2: the old record leaves, then the new one is added
            self.log_record("supersede", old)
            self.log_record("add", rec)
        self.log("supersede", {"old": old["id"], "new": rec["id"], "lineageId": rec.get("lineageId"),
                               "moveCount": rec.get("moveCount"), "oldAgeMin": (self.t - old["lastSeenMs"]) / 60000.0,
                               "distanceM": 1.0}, tagged=True)
        item.update(record=rec["id"], state="home")

    def room_name(self, room_id):
        return next((r[1] for r in self.rooms if r[0] == room_id), None)

    def max_sim_others(self, rec):
        u = np.asarray(rec["embedding"], np.float32)
        sims = [cos(u, np.asarray(r["embedding"], np.float32)) for r in self.records.values() if r["id"] != rec["id"]]
        return max(sims) if sims else 0.0

    # ------------------------------------------------------------------ captures
    def candidates(self, item, at_marker, condition="default", weak=False):
        """Embeddings of the crops of one capture. `at_marker` is what really is at the marker:
        'target', 'empty', 'other', 'lookalike'."""
        rng, u = self.rng, item["u"]
        if at_marker == "target":
            mean = CONDITION_SIM[condition] - (0.25 if weak else 0.0)
            best = vec_with_sim(u, rng.normal(mean, 0.07), rng)
        elif at_marker == "empty":
            best = vec_with_sim(u, rng.normal(0.16, 0.08), rng)
        elif at_marker == "other":
            others = [i for i in self.items if i is not item and i.get("record") in self.records]
            if others and rng.random() < 0.5:  # another registered item sits at the marker
                best = vec_with_sim(others[int(rng.integers(len(others)))]["u"], rng.normal(0.82, 0.06), rng)
            else:
                best = vec_with_sim(u, rng.normal(0.22, 0.10), rng)
        else:  # lookalike
            p = item.get("partner")
            if p is not None and p.get("record") in self.records:  # the registered look-alike
                best = vec_with_sim(p["u"], rng.normal(0.84, 0.05), rng)
            else:
                best = vec_with_sim(u, rng.normal(0.62, 0.10), rng)
        out = [("projected", best)]
        for _ in range(int(rng.integers(0, 3))):
            out.append(("detector", vec_with_sim(u, cos(best, u) - rng.uniform(0.05, 0.3), rng)))
        return out

    def verify(self, q, item, at_marker, condition="default", weak=False, near_ms=None):
        rec = self.records[item["record"]]
        u = np.asarray(rec["embedding"], np.float32)
        q["attempt"] += 1
        cands = self.candidates(item, at_marker, condition, weak)
        sims = [cos(v, u) for _, v in cands]
        best = int(np.argmax(sims))
        vid = "%s_%d" % (q["id"], q["attempt"])
        emb_file = None
        if self.o.variant != "minimal":
            emb_file = "verify/%s_emb.json" % vid
            rects = [[100.0, 100.0, 400.0, 400.0]] * len(cands)
            if self.name == "A":
                doc = {"model": "SYNTHETIC", "dim": DIM, "record": rec["id"], "synthetic": True,
                       "candidates": [{"kind": k, "rect": r, "sim": s, "embedding": b64(v)}
                                      for (k, v), s, r in zip(cands, sims, rects)]}
            else:  # the other plausible spelling of the same content
                doc = {"modelName": "SYNTHETIC", "dimension": DIM, "target": rec["id"], "synthetic": True,
                       "candidates": [{"kind": k, "rectangle": r, "similarity": s, "embedding": b64(v)}
                                      for (k, v), s, r in zip(cands, sims, rects)]}
            with open(os.path.join(self.dir, emb_file), "w") as f:
                json.dump(doc, f)
        self.t += int(self.rng.integers(2000, 9000))  # aiming
        lat = {"convertMs": float(self.rng.normal(55, 6)), "detectMs": float(self.rng.normal(80, 12)),
               "embedMs": float(self.rng.normal(25, 5) * len(cands))}
        lat["totalMs"] = lat["convertMs"] + lat["detectMs"] + lat["embedMs"] + 15.0
        sim = sims[best]
        e = self.log("verify", {
            "query": q["id"], "record": rec["id"], "attempt": q["attempt"], "positionSource": q["source"],
            "refFullTracking": False, "distance": 0.9, "detectorBoxes": len(cands) - 1,
            "scores": [{"kind": k, "sim": s, "rect": [100.0, 100.0, 400.0, 400.0]} for (k, _), s in zip(cands, sims)],
            "bestSimilarity": sim, "bestKind": cands[best][0], "threshold": MATCH_THRESHOLD,
            "outcome": "CONFIRMED" if sim >= MATCH_THRESHOLD else "INSUFFICIENT_EVIDENCE",
            "inferenceCalls": 1 + len(cands), "latency": lat,
            "frameFile": "verify/%s_frame.jpg" % vid, "cropFile": "verify/%s_best.jpg" % vid,
            "embFile": emb_file, "guidanceMode": "NEAR",
            "msSinceQueryStart": self.t - q["t0"],
            "msSinceNear": (self.t - near_ms) if near_ms is not None else None,
        } | self.proxies(), tagged=True)
        q["last"] = e
        return e

    # ------------------------------------------------------------------ queries
    def start_query(self, item, force_source=None):
        s, rec = self.session, self.records[item["record"]]
        self.n_query += 1
        if force_source:
            source = force_source
        elif rec["sessionId"] == s["id"]:
            source = "SESSION"
        elif s["fix"] is not None and rec.get("mapId") == self.map_id:
            source = "MAP"
        else:
            source = "NONE"
        age = (self.t - rec["lastSeenMs"]) / 60000.0
        max_sim = self.max_sim_others(rec)
        p = self.proxies()
        reasons = []
        if source != "NONE":  # the app's VerificationPolicy.decide()
            if age > 30.0:
                reasons.append("age")
            if max_sim > 0.80:
                reasons.append("similar record")
            if source == "MAP" and (p["metersSinceFix"] or 0) > 20.0:
                reasons.append("walked")
            if source == "MAP" and (p["trackingLossesSinceFix"] or 0) > 0:
                reasons.append("tracking loss")
        q = {"id": "SYNq%s%04d" % (self.name, self.n_query), "attempt": 0, "source": source, "t0": self.t, "last": None,
             "item": item}
        self.log("query_start", {
            "query": q["id"], "record": rec["id"], "name": rec["name"], "ageMin": age, "maxSimOthers": max_sim,
            "positionSource": source, "refFullTracking": False, "refLocked": False, "cameraTracking": "TRACKING",
            "policyVerify": bool(reasons), "policyReasons": reasons or ["recent, distinct, tracking ok"],
            "mapId": self.map_id, "recordMapId": rec.get("mapId"), "located": s["fix"] is not None,
            "itemRoom": item["room"], "itemRoomName": self.room_name(item["room"]),
            "userRoom": self.rooms[0][0], "userRoomName": self.rooms[0][1],
            "guidanceMode": "NO_FIX" if source == "NONE" else "FAR_ROUTE", "nearRule": None, "routeM": 6.0, "straightM": 4.0,
            "lineageId": rec.get("lineageId", rec["id"]), "moveCount": rec.get("moveCount", 0),
            "lineageAgeMin": (self.t - item["lineage_t"]) / 60000.0, "mobility": item["mobility"],
        } | p, tagged=True)
        q["t0"] = self.t
        return q

    def marker_error(self, item, source):
        rng, s = self.rng, self.session
        if source == "SESSION":
            return float(rng.lognormal(math.log(0.03), 0.4))
        f = s["fix"]
        d = math.hypot(item["pos"][0] - f["x"], item["pos"][2] - f["z"])
        err = rng.lognormal(math.log(0.05), 0.5) + 0.0012 * s["meters"] + 0.012 * d
        if s["losses"] > 0:
            err += 0.3 * rng.random()
        return float(err)

    def end_query(self, q, truth, measured=None, no_capture=None, near_ms=None):
        marker = None
        if measured is not None:
            marker = {"errorM": measured, "horizontalErrorM": measured * 0.9, "verticalErrorM": measured * 0.3,
                      "positionSource": q["source"], "hitType": "DepthPoint", "displaced": truth == "displaced"}
            marker.update(self.proxies())
        last = q["last"]
        self.log("query_end", {
            "query": q["id"], "record": self.records[q["item"]["record"]]["id"], "groundTruth": truth,
            "verifications": q["attempt"], "lastOutcome": last["outcome"] if last else None,
            "lastSimilarity": last["bestSimilarity"] if last else None, "markerError": marker,
            "durationMs": self.t - q["t0"], "nearReached": near_ms is not None,
            "timeToNearMs": (near_ms - q["t0"]) if near_ms is not None else None,
            "modesSeen": ["FAR_ROUTE", "NEAR"], "finalMode": "NEAR",
            "noCaptureReason": no_capture,
            "msToFirstVerify": None if last is None else last["msSinceQueryStart"],
        }, tagged=True)

    def paired_query(self, item, state):
        """One PAIRED-mode trial. `state` is where the item really is: home / absent / displaced / replaced / swap."""
        rng = self.rng
        self.walk(float(rng.uniform(3, 8)))
        if rng.random() < 0.06:
            self.session["losses"] += 1
            self.log("tracking_lost", {"reason": "SYNTHETIC", "afterMs": 1000, "located": True})
        roll = rng.random()
        if roll < 0.03:   # cannot be placed: no marker, no capture
            q = self.start_query(item, force_source="NONE")
            self.end_query(q, "absent" if state != "home" else "present", no_capture="not_located")
            return
        q = self.start_query(item, force_source="SESSION" if roll > 0.9 else None)  # some same-session anchors
        self.t += int(rng.integers(5000, 20000))
        near = self.t
        if roll < 0.06:   # the experimenter finished without capturing
            self.end_query(q, "skip" if roll < 0.045 else ("present" if state == "home" else "absent"),
                           no_capture="user_skipped", near_ms=near)
            return
        if state == "home":
            err = self.marker_error(item, q["source"])
            off = err > 0.30
            self.verify(q, item, "empty" if off and rng.random() < 0.7 else "target", weak=off, near_ms=near)
            self.end_query(q, "present", measured=None if rng.random() < 0.08 else err, near_ms=near)
        elif state == "absent":
            self.verify(q, item, "empty", near_ms=near)
            self.end_query(q, "absent", near_ms=near)
        elif state == "displaced":
            d = float(rng.uniform(0.2, 1.5))
            self.verify(q, item, "target" if d < 0.35 and rng.random() < 0.5 else "empty", weak=True, near_ms=near)
            self.end_query(q, "displaced", measured=d, near_ms=near)
        else:
            self.verify(q, item, "lookalike" if state == "swap" else "other", near_ms=near)
            self.end_query(q, "replaced", near_ms=near)

    # ------------------------------------------------------------------ files
    def write(self):
        events = self.events
        if self.o.variant == "minimal":
            events = [{k: v for k, v in e.items() if k not in V04_FIELDS} for e in events
                      if e["type"] not in ("study_settings",)]
            for e in events:
                if isinstance(e.get("studyTag"), str):  # no dev/test split in the minimal variant
                    e["studyTag"] = ";".join(p for p in e["studyTag"].split(";") if not p.startswith("split="))
        with open(os.path.join(self.dir, "events.jsonl"), "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        with open(os.path.join(self.dir, "records.json"), "w") as f:
            json.dump(list(self.records.values()), f)
        if self.o.variant != "minimal":
            with open(os.path.join(self.dir, "records_log.jsonl"), "w") as f:
                for e in self.records_log:
                    f.write(json.dumps(e) + "\n")
        nodes, nid = [], 0
        for room_id, _, x0, x1 in self.rooms:
            for x in np.arange(x0 + 0.25, x1, 0.5):
                for z in np.arange(0.25, 3.0, 0.5):
                    nodes.append({"id": nid, "x": float(x), "y": 1.3, "z": float(z), "roomId": room_id,
                                  "sessionId": "SYNTH", "tMs": T0_MS})
                    nid += 1
        fixes = [{"id": f["id"], "name": f["name"], "type": f["type"], "x": f["x"], "y": f["y"], "z": f["z"], "yaw": 0.0,
                  "yawAxis": "X", "widthM": 0.116 if f["type"] == "IMAGE" else None, "ghostFile": None,
                  "createdMs": T0_MS, "sessionId": "SYNTH"} for f in self.fix_points.values()]
        with open(os.path.join(self.dir, "map.json"), "w") as f:
            json.dump({"version": 1, "mapId": self.map_id, "createdMs": T0_MS, "synthetic": True, "fixPoints": fixes,
                       "rooms": [{"id": r[0], "name": r[1], "createdMs": T0_MS} for r in self.rooms],
                       "nodes": nodes, "edges": []}, f)
        with open(os.path.join(self.dir, MARKER_FILE), "w") as f:
            f.write(MARKER_TEXT)


def build_site(name, out_dir, rng, o):
    s = Site(name, out_dir, rng, o)
    n = o.items
    # Items: every fourth item has a registered look-alike (the next item).
    for i in range(n):
        room = s.rooms[i % 3]
        s.items.append({
            "name": "SYNTHETIC item %s%02d" % (name, i + 1), "room": room[0],
            "pos": [float(rng.uniform(room[2] + 0.3, room[3] - 0.3)), 0.8, float(rng.uniform(0.3, 2.7))],
            "mobility": MOBILITY[int(rng.integers(3))], "u": unit(rng.normal(size=DIM)), "partner": None,
        })
    for i in range(0, n - 1, 4):
        a, b = s.items[i], s.items[i + 1]
        b["u"] = vec_with_sim(a["u"], 0.80, rng)
        a["partner"], b["partner"] = b, a

    # ---- E1, session 0: set up the map, register, measure with same-session anchors
    # E1 carries split=test, as the experiment protocol tags it.
    s.start_session(s.tag(exp="E1", round=0, split="test"), "FREE")
    s.log("map_created", {"mapId": s.map_id, "fixPoint": s.fix_points["IMAGE"]["id"], "fixType": "IMAGE",
                          "pose": {"x": 0, "y": 0, "z": 0, "yawDeg": 0}})
    s.fix("IMAGE", "first")
    for item in s.items:
        s.walk(float(rng.uniform(1, 3)))
        s.register(item)
    # One extra record that is renamed and then deleted (record history).
    ghost = {"name": "SYNTHETIC deleted item", "room": s.rooms[0][0], "pos": [1.0, 0.8, 1.0], "mobility": "some",
             "u": unit(rng.normal(size=DIM)), "partner": None}
    rec = s.register(ghost)
    renamed = dict(rec, name="SYNTHETIC renamed item")
    s.records[rec["id"]] = renamed
    s.log_record("rename", renamed)
    s.log("rename", {"record": rec["id"], "from": rec["name"], "to": renamed["name"]})
    for item in s.items[: max(2, n * 2 // 5)]:
        q = s.start_query(item)
        s.verify(q, item, "target")
        s.end_query(q, "present", measured=s.marker_error(item, "SESSION"), near_ms=s.t)
    del s.records[rec["id"]]
    s.log_record("delete", renamed)
    s.log("delete", {"record": rec["id"], "name": renamed["name"]})

    # ---- E1, three restarts: map-frame marker error, refresh and manual fixes
    for r in range(1, 4):
        ftype = "SPOT" if r == 2 else "IMAGE"
        s.start_session(s.tag(exp="E1", round=r, split="test"), "FREE")
        s.fix(ftype, "first")
        for j, item in enumerate(s.items[: max(3, n * 3 // 5)]):
            s.walk(float(rng.uniform(2, 7)))
            if j % 4 == 1:
                s.fix("IMAGE", "refresh", jump=float(rng.lognormal(math.log(0.015), 0.6)))
            if j % 4 == 3:
                s.fix("SPOT", "manual", jump=float(rng.lognormal(math.log(0.04), 0.6)))
            q = s.start_query(item)
            s.t += int(rng.integers(4000, 25000))    # the walk to the item
            near = None if j == 9 else s.t           # one query never reaches NEAR
            if j == 5:
                s.end_query(q, "skip", near_ms=near)
                continue
            s.end_query(q, "present", measured=None if j == 7 else s.marker_error(item, q["source"]), near_ms=near)
        if r == 1:  # the picture was moved: one fix declined, one accepted
            s.fix("IMAGE", "refresh", jump=1.1, disagreement=True, accepted=False, applied=False)
            s.fix("IMAGE", "refresh", jump=0.9, disagreement=True, accepted=True)

    # ---- E2: labelled benchmark, one dev and one test session
    for split in ("dev", "test"):
        s.start_session(s.tag(exp="E2", split=split), "BENCHMARK")
        s.fix("IMAGE", "first")
        k = 0
        for item in s.items:
            s.walk(float(rng.uniform(1, 4)))
            q = s.start_query(item)
            for label in ("target", "target", "target", "empty", "other", "lookalike"):
                cond = CONDITIONS[k % len(CONDITIONS)]
                k += 1
                e = s.verify(q, item, label, cond, near_ms=q["t0"])
                if k % 41 == 0:  # dismissed without a label
                    s.log("verify_label", {"query": q["id"], "record": e["record"], "attempt": e["attempt"],
                                           "label": None, "condition": cond})
                    continue
                if k % 53 == 0:  # a wrong label, corrected by a second one
                    s.log("verify_label", {"query": q["id"], "record": e["record"], "attempt": e["attempt"],
                                           "label": "empty", "condition": cond})
                s.log("verify_label", {"query": q["id"], "record": e["record"], "attempt": e["attempt"],
                                       "label": label, "condition": cond})
            s.end_query(q, "skip")

    # ---- E3: controlled paired trials, rounds 1-2 dev, the rest test
    for r in range(1, o.rounds + 1):
        s.start_session(s.tag(exp="E3", round=r, split="dev" if r <= max(1, o.rounds // 2) else "test"), "PAIRED")
        s.fix("SPOT" if r % 2 == 0 else "IMAGE", "first")
        if r % 2 == 0:
            s.walk(45.0)  # the long loop
        states = ["home"] * (n // 2) + ["absent"] * (n // 5) + ["displaced"] * (n * 3 // 20) + ["replaced", "swap"] * n
        states = states[:n]
        rng.shuffle(states)
        order = rng.permutation(n)
        for i in order:
            s.paired_query(s.items[i], states[i])

    # ---- E5: deployment, no split, natural movement, Update location
    for item in s.items:
        item["state"] = "home"
    for day in range(1, o.days + 1):
        for rnd in (1, 2):
            s.start_session(s.tag(exp="E5", day=day, round=rnd), "PAIRED", gap_ms=6 * 3_600_000)
            s.fix("IMAGE" if (day + rnd) % 3 else "SPOT", "first")
            for i in rng.permutation(n):
                item = s.items[i]
                if item["state"] == "home" and rng.random() < MOVE_PROB[item["mobility"]]:
                    item["state"] = ["absent", "displaced", "replaced"][int(rng.choice(3, p=[0.55, 0.30, 0.15]))]
                elif item["state"] != "home" and rng.random() < 0.3:
                    item["state"] = "home"  # somebody put it back
                s.paired_query(item, item["state"])
                if item["state"] != "home" and rng.random() < 0.6:
                    s.update_location(item)

    # ---- one session whose tag is set half-way (FREE mode, several captures per query)
    s.start_session("", "FREE")
    s.fix("IMAGE", "first")
    for j, item in enumerate(s.items[:6]):
        if j == 3:
            s.session["tag"] = s.tag(exp="E0")
            s.log("study_settings", {"studyTag": s.session["tag"], "captureMode": "FREE"})
            s.fix("SPOT", "manual", jump=0.03)  # an untagged event type after the tag was set
        q = s.start_query(item)
        for _ in range(j % 3):
            s.verify(q, item, "target", near_ms=q["t0"])
        if j == 5:
            break  # unfinished query: no query_end
        s.end_query(q, "present", near_ms=q["t0"])
    return s


def power_run(mode, label, start_ms, seconds, mw, verifications):
    rails = [("S1S_VDD_CAM", "Multimedia", "camera", 0.05), ("S2S_VDD_G3D", "GPU", "gpu", 0.05),
             ("S4M_VDD_CPUCL0", "CPU(LITTLE)", "cpu", 0.40), ("S7M_VDD_TPU", "TPU", "tpu", 0.05),
             ("VSYS_PWR_DISPLAY", "Display", "display", 0.20), ("S4S_VDD2H_MEM", "DDR", "memory", 0.10),
             ("S5M_VDD_INT", "INT", "other", 0.15)]
    out_rails, groups = [], {}
    for i, (name, sub, group, share) in enumerate(rails):
        p = mw * share
        out_rails.append({"index": i, "name": name, "subsystem": sub, "group": group, "samples": int(seconds) + 1,
                          "t_first_ms": 1000, "t_last_ms": 1000 + int(seconds * 1000), "duration_s": float(seconds),
                          "energy_j": p * seconds / 1000.0, "mean_mw": p})
        groups[group] = {"energy_j": p * seconds / 1000.0, "mean_mw": p, "rails": [name]}
    return {
        "tool": "power_measure.py", "format_version": 1, "synthetic": True, "mode": mode, "label": "SYNTHETIC-" + label,
        "note": "SYNTHETIC: not a measurement", "requested_seconds": seconds, "duration_s": float(seconds),
        "rails": out_rails, "groups": groups,
        "all_rails": {"energy_j": mw * seconds / 1000.0, "mean_mw": mw, "n_rails": len(rails), "note": "SYNTHETIC"},
        "phone_start_ms": start_ms, "phone_end_ms": start_ms + int(seconds * 1000), "problems": [],
        "verifications": verifications, "device": {"model": "SYNTHETIC", "build": "SYNTHETIC", "build_type": "SYNTHETIC"},
        "state_before": {"foreground": "edu.cs6301.findback/.MainActivity"},
        "state_after": {"foreground": "edu.cs6301.findback/.MainActivity"},
    }


def add_power(site, out, rng):
    """Fake power runs, and an E4 session whose verify events fall inside the tracking_verify runs."""
    pdir = os.path.join(out, "power_runs")
    os.makedirs(pdir, exist_ok=True)
    seconds = 300
    site.start_session(site.tag(exp="E4"), "FREE")
    site.fix("IMAGE", "first")
    item = site.items[0]
    for run in range(3):
        start = site.t + 60_000
        doc = power_run("tracking_only", "run%d" % run, start, seconds, float(rng.normal(1900, 30)), None)
        with open(os.path.join(pdir, "tracking_only__SYNTHETIC-run%d.json" % run), "w") as f:
            json.dump(doc, f, indent=1)
        site.t = start + seconds * 1000 + 60_000
    for run in range(3):
        start = site.t + 60_000
        site.t = start
        q = site.start_query(item)
        n_ver = 0
        while site.t + 20_000 < start + seconds * 1000:
            before = site.t
            site.verify(q, item, "target", near_ms=q["t0"])
            n_ver += 1
            site.t = before + 20_000
        site.end_query(q, "skip")
        # Run 0 leaves the count to the events; the others also carry the experimenter's count.
        doc = power_run("tracking_verify", "run%d" % run, start, seconds, float(rng.normal(2050, 30)),
                        None if run == 0 else n_ver)
        with open(os.path.join(pdir, "tracking_verify__SYNTHETIC-run%d.json" % run), "w") as f:
            json.dump(doc, f, indent=1)
        site.t = start + seconds * 1000 + 60_000


def is_real_data(path):
    """True when `path` looks like a data folder that this script did not write."""
    for root, _, files in os.walk(path):
        if "events.jsonl" in files and MARKER_FILE not in files:
            return True
    return False


def main(argv=None):
    default_out = os.path.join(tempfile.gettempdir(), "findback_synthetic")
    ap = argparse.ArgumentParser(description="Synthetic FindBack logs, for testing the analysis code only.")
    ap.add_argument("--out", default=default_out, help="scratch directory to write (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--variant", choices=("full", "minimal"), default="full",
                    help="minimal: no v0.4 fields except studyTag, no split, no embedding files, no records_log")
    ap.add_argument("--items", type=int, default=20, help="items per site")
    ap.add_argument("--rounds", type=int, default=4, help="E3 rounds per site")
    ap.add_argument("--days", type=int, default=5, help="E5 days per site (two rounds a day)")
    ap.add_argument("--sites", type=int, default=2, choices=(1, 2))
    a = ap.parse_args(argv)

    out = os.path.abspath(a.out)
    parts = [p.lower() for p in out.split(os.sep)]
    if any(p in ("paper", "findback_data", "backups", "findback") for p in parts):
        print("ERROR: refusing to write synthetic data under %s (a paper, data or app directory)." % out, file=sys.stderr)
        return 2
    if os.path.isdir(out) and is_real_data(out):
        print("ERROR: %s holds an events.jsonl without %s; refusing to overwrite what may be real data."
              % (out, MARKER_FILE), file=sys.stderr)
        return 2
    for sub in ["site_A", "site_B", "power_runs"]:
        shutil.rmtree(os.path.join(out, sub), ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, MARKER_FILE), "w") as f:
        f.write(MARKER_TEXT)

    rng = np.random.default_rng(a.seed)
    sites = []
    for name in ["A", "B"][: a.sites]:
        site = build_site(name, os.path.join(out, "site_" + name), rng, a)
        sites.append(site)
    add_power(sites[0], out, rng)
    for site in sites:
        site.write()
        print("SYNTHETIC %s: %d events, %d active records, %d record-log entries" % (
            site.dir, len(site.events), len(site.records), len(site.records_log)))
    print("SYNTHETIC power runs:", os.path.join(out, "power_runs"))
    print("Test data only. paper_figures.py refuses to write it under paper/ without --allow-synthetic.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
