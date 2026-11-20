#!/usr/bin/env python3
"""Tests of the offline identity replay (analysis/identity_replay.py) on synthetic scenes. No phone
data and no image.

    python3 analysis/test_identity_replay.py
    analysis/.venv-bench/bin/python analysis/test_identity_replay.py

Every scene is built in memory: objects on a line (map frame, metres) with an appearance (an
embedding axis: one axis per physical object, the same axis for a look-alike pair), keyframes from a
camera 1.5 m in front of the line looking straight at it (identity orientation, so a map point x is
in view when |x - camera x| <= about 1.2 m), a uniform depth image at 1.5 m, and the detections each
keyframe makes. The "current" decisions are made by the app's own matcher as re-implemented in
analysis/scan_replay.py (radius 0.20 m for these 0.1 m objects, same label merges), so they are what
the phone would have logged. A test scan is shifted by a launch offset (all its map points and its
camera). Each physical object labels its own observations, so the scoring runs on the outcomes.
"""
import base64
import copy
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
import identity_replay as ir  # noqa: E402
import scan_replay as sr  # noqa: E402

DIM = 16
INTR = {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0, "width": 1920, "height": 1080}
M2W = {"x": 0.0, "y": 0.0, "z": 0.0, "yawDeg": 0.0}
CAM_Z = 1.5
OFF_SMALL = (0.08, 0.0, -0.05)      # 0.094 m: inside the app's 0.20 m radius
OFF_LARGE = (0.30, 0.0, 0.15)       # 0.335 m: outside it, and outside PLACE_RADIUS_M (0.30 m)
JITTER = [(0.0, 0.0, 0.0), (0.01, 0.0, -0.01), (-0.01, 0.005, 0.01), (0.005, -0.005, 0.0), (0.0, 0.01, 0.005)]


def emb(axis, k):
    """Unit vector near axis `axis`; k varies the small noise (cosine about 0.98 within an axis,
    about 0.03 across axes)."""
    v = [0.06 * math.sin(1.7 * i + 2.3 * k + 0.9 * axis) for i in range(DIM)]
    v[axis] += 1.0
    return ir.unit(v)


def b64(v):
    return base64.b64encode(struct.pack("<%df" % len(v), *v)).decode()


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def uniform_depth():
    """160 x 90 depth image, every pixel 1.5 m; image pixels map to depth pixels by 1/12."""
    import array
    return ir.DepthMap(160, 90, array.array("H", [1500] * (160 * 90)), (1 / 12.0, 0.0, 0.0, 0.0, 1 / 12.0, 0.0))


# ----------------------------------------------------------------------------- scenes

def build_scan(spec, objects, initial_entries, t0):
    """One scan: observations with the app matcher's decisions, keyframes, and the matcher's state.
    spec: {"id", "offset", "pos": {code: point}, "kfs": [(camera x, [codes detected])]}."""
    off = spec.get("offset", (0.0, 0.0, 0.0))
    rp = sr.Replay(sr.sim_embedding, initial_entries)
    known = set(rp.real)
    obs, keyframes, labels, linked = [], {}, {}, {}
    n_new = [0]
    for k, (cam_x, codes) in enumerate(spec["kfs"]):
        t = t0 + 500 * k
        keyframes[k] = ir.KeyGeom(INTR, add((cam_x, 0.0, CAM_Z), off), (0.0, 0.0, 0.0, 1.0), M2W, uniform_depth())
        dets = []
        for j, code in enumerate(codes):
            label, axis = objects[code]
            p = add(add(spec["pos"][code], off), JITTER[(k + j) % len(JITTER)])
            oid = "%s-%d-%d" % (spec["id"], k, j)
            e = emb(axis, 10 * k + j + 100 * (t0 // 1_000_000))
            dets.append(sr.Det(oid, k, t, label, 0.6 - 0.01 * j, (100 * j, 100, 100 * j + 80, 200), p, 0.1,
                               embedding=e, order=j))
            labels[oid] = code
        used = set()
        for d in sr.processing_order(dets):
            dec = rp.decide(d, t, used)
            kind, eid = dec.kind, dec.entry
            if kind == sr.KIND_NEW:
                n_new[0] += 1
                eid = "%s_e%d" % (spec["id"], n_new[0])
            rp.apply(d, t, kind, eid, eid)
            used.add(eid)
            linked.setdefault(eid, []).append(d.key)
            o = ir.Obs(d.key, spec["id"], k, d.order, t, d.label, d.score, d.map_point, d.embedding, 0.1,
                       None, True, {"kind": kind, "entry": eid})
            obs.append(o)
    dropped = rp.finish()
    created = [i for i in rp.real if i not in known]
    return obs, keyframes, labels, linked, rp, set(dropped), created


def make_world(objects, scans):
    """History (ir.History) and test scans (ir.ScanData), with the observation labels."""
    labels = {}
    entries = {}            # id -> sr.Entry, the app's real entries after the history
    meta = {}               # id -> dict pieces of the inventory entry
    hist_data = []
    t0 = 1_000_000
    tests = []
    for spec in scans:
        if spec["role"] == "history":
            obs, kfs, lab, linked, rp, dropped, created = build_scan(spec, objects, list(entries.values()), t0)
            labels.update(lab)
            for eid in rp.real:
                e = rp.entries[eid]
                entries[eid] = e
                m = meta.setdefault(eid, {"createdScan": spec["id"], "history": []})
                if eid in created:
                    m["confirmed"] = {"mapPosition": list(e.position), "scan": spec["id"], "source": "created",
                                      "refs": list(linked.get(eid, []))}
                ks = linked.get(eid, [])
                pts = [o.map_point for o in obs if o.id in ks]
                m["history"].append({"scan": spec["id"], "obs": list(ks), "outcome": "seen" if ks else "missed",
                                     "position": [sum(p[i] for p in pts) / len(pts) for i in range(3)] if pts else None})
            hist_data.append(ir.ScanData(spec["id"], obs, kfs, sorted(kfs), created=created, dropped=dropped,
                                         map_id="m1", t_start=t0, t_end=t0 + 100_000))
        t0 += 1_000_000
    hist_entries = []
    for eid, e in entries.items():
        m = meta[eid]
        hist_entries.append((eid, {"id": eid, "label": e.label, "mapId": "m1", "mapPosition": list(e.position),
                                   "embedding": b64(e.embedding), "createdScan": m["createdScan"],
                                   "confirmed": m.get("confirmed"), "history": m["history"], "sizeM": e.size_m}))
    history = ir.History(hist_data, hist_entries, "m1")
    t0 = 1_000_000 * (1 + len(scans))
    for spec in scans:
        if spec["role"] != "test":
            continue
        start = [copy.deepcopy(e) for e in entries.values()]
        obs, kfs, lab, linked, rp, dropped, created = build_scan(spec, objects, start, t0)
        labels.update(lab)
        scan = ir.ScanData(spec["id"], obs, kfs, sorted(kfs), created=created, dropped=dropped, map_id="m1",
                           t_start=t0, t_end=t0 + 100_000)
        # the app's evidence outcome (inv_evidence): seen when linked; missed when its stored place was
        # observable in MISSED_MIN_KEYFRAMES keyframes; else not observed (the app's appearance check is
        # not simulated: here an empty place never looks like the item)
        for eid, e in entries.items():
            if linked.get(eid):
                scan.evidence[eid] = "seen"
                continue
            n = sum(1 for k in kfs if ir.classify_place(tuple(e.position), kfs[k], None)[0] == "observable")
            scan.evidence[eid] = "missed" if n >= ir.DEFAULT.MISSED_MIN_KEYFRAMES else "not_observed"
        tests.append((spec, scan))
        t0 += 1_000_000
    return history, tests, labels


def run_all(history, scan, labels, extra=(), p=ir.DEFAULT, align_off=False):
    """Every method on one test scan. Returns {method: result} and {method: {item code: outcome}}."""
    sightings = ir.build_sightings(scan, p)
    out = {}
    cur = ir.current_method(scan, set(history.entries))
    out["current"] = {"obs": {o: (d, d.item) for o, d in cur["obs"].items()}, "beliefs": cur["beliefs"]}
    gi = ir.generic_items(history, p)
    r = ir.run_engine(gi, sightings, scan, p, align=True)
    r["obs"] = ir.obs_decisions(r, sightings)
    out["generic"] = r
    if align_off:
        r = ir.run_engine(gi, sightings, scan, p, align=False)
        r["obs"] = ir.obs_decisions(r, sightings)
        out["generic_noalign"] = r
    folder = tempfile.mkdtemp(prefix="idr_test_")
    try:
        path = os.path.join(folder, "pack.json")
        pack = ir.build_pack(history, p)
        size = ir.write_pack(pack, path)
        assert size == os.path.getsize(path) and size > 0
        with open(path) as f:
            loaded = json.load(f)
    finally:
        shutil.rmtree(folder)
    hi = ir.pack_items(loaded)
    r = ir.run_engine(hi, sightings, scan, p, align=True)
    r["obs"] = ir.obs_decisions(r, sightings)
    out["household"] = r
    sx = ir.build_sightings(scan, p, extra)
    r = ir.run_engine(hi, sx, scan, p, align=True)
    r["obs"] = ir.obs_decisions(r, sx)
    r["sightings"] = sx
    out["extra_view"] = r
    emap, _ = ir.entry_item_map(history.entries, labels)
    universe = sorted(set(labels.values()))
    scores = {}
    for name, m in out.items():
        scores[name], _ = ir.score_method(m["obs"], m["beliefs"], [o.id for o in scan.observations], labels, emap, universe)
    out["_pack"], out["_emap"], out["_sightings"] = loaded, emap, sightings
    return out, scores


def entry_of(emap, code):
    return [e for e, x in emap.items() if x == code]


def belief(out, method, emap, code):
    return ir.item_belief(code, emap, out[method]["beliefs"])


def expect(scores, table):
    """table: {method: {code: outcome}}."""
    for method, want in table.items():
        for code, outcome in want.items():
            got = scores[method].get(code)
            assert got == outcome, "%s: %s is %s, expected %s (all: %s)" % (method, code, got, outcome, scores[method])


# Three distinct anchors that stay in every scene: they give the alignment its pairs.
ANCHORS = {"A1": ("cup", 0), "A2": ("bowl", 1), "A3": ("book", 2)}
ANCHOR_POS = {"A1": (0.0, 0.0, 0.0), "A2": (0.6, 0.0, 0.0), "A3": (1.2, 0.0, 0.0)}


def scene(objs, hist_pos, test_pos, hist_kfs, test_kfs, offset):
    objects = dict(ANCHORS, **objs)
    hp = dict(ANCHOR_POS, **hist_pos)
    tp = dict(ANCHOR_POS, **test_pos)
    return make_world(objects, [{"id": "h1", "role": "history", "pos": hp, "kfs": hist_kfs},
                                {"id": "t1", "role": "test", "pos": tp, "kfs": test_kfs, "offset": offset}])


# ----------------------------------------------------------------------------- 1. stays under an offset

def test_item_that_stays_under_an_alignment_offset():
    # S (bottle) and the anchors stay; the test launch places everything 0.335 m off (OFF_LARGE).
    # current: 0.335 m > the app's 0.20 m radius, so every object gets a new entry: 4 duplicates.
    # generic: the four mutual-best pairs give a translation of -OFF_LARGE; then all four are in place.
    # generic without the correction: 0.335 m > PLACE_RADIUS_M, so nothing is in place; every stored
    #   place is looked at and empty (missed), and each sighting's only similar missed item is its own
    #   (cosine 0.98 against 0.03): an update "moved" by 0.335 m. Same updates, other beliefs.
    # household and extra_view (no extra view): as generic.
    kfs = [(0.9, ["A1", "A2", "A3", "S"])] * 3
    history, tests, labels = scene({"S": ("bottle", 3)}, {"S": (1.8, 0, 0)}, {"S": (1.8, 0, 0)}, kfs, kfs, OFF_LARGE)
    out, scores = run_all(history, tests[0][1], labels, align_off=True)
    four = {c: "correct_update" for c in ("A1", "A2", "A3", "S")}
    expect(scores, {"current": {c: "duplicate" for c in four}, "generic": four, "generic_noalign": four,
                    "household": four, "extra_view": four})
    a = out["generic"]["alignment"]
    assert a.kind in ("translation", "yaw+translation"), a.summary()
    assert all(abs(a.t[i] + OFF_LARGE[i]) < 0.02 for i in range(3)) and abs(a.yaw_deg) < 1.0, a.summary()
    emap = out["_emap"]
    assert belief(out, "generic", emap, "S") == "seen_in_place"
    assert belief(out, "generic_noalign", emap, "S") == "moved"
    assert all(d.how == "in_place" for d in out["generic"]["decisions"].values())
    assert belief(out, "current", emap, "S") == "missed"


# ----------------------------------------------------------------------------- 2. moved, old place empty

def test_moved_item_whose_old_place_is_seen_empty():
    # M (bottle) moves from x = 1.8 to x = 2.6. The test walk looks at 1.8 in every keyframe (no
    # detection there) and sees M at 2.6 in two keyframes. Small launch offset (0.094 m).
    # current: no entry within 0.20 m of 2.6: a new entry (promoted at the second sighting): duplicate;
    #   its evidence for M: missed.
    # generic: M not in place; its place observable in 4 keyframes -> missed; the sighting at 2.6 has
    #   one candidate, M (0.98 >= SIM_MOVED, no runner-up): update M, moved. Correct update.
    # household: M's threshold from the pack is max(HH_THR_MIN, impostor + 0.05) = 0.40: the same.
    hist_kfs = [(0.9, ["A1", "A2", "A3", "M"])] * 3
    test_kfs = [(0.9, ["A1", "A2", "A3"]), (0.9, ["A1", "A2", "A3"]), (2.0, ["A3", "M"]), (2.0, ["A3", "M"])]
    history, tests, labels = scene({"M": ("bottle", 3)}, {"M": (1.8, 0, 0)}, {"M": (2.6, 0, 0)}, hist_kfs, test_kfs, OFF_SMALL)
    out, scores = run_all(history, tests[0][1], labels)
    anchors = {c: "correct_update" for c in ANCHORS}
    expect(scores, {"current": dict(anchors, M="duplicate"), "generic": dict(anchors, M="correct_update"),
                    "household": dict(anchors, M="correct_update"), "extra_view": dict(anchors, M="correct_update")})
    emap = out["_emap"]
    assert belief(out, "current", emap, "M") == "missed"
    for m in ("generic", "household"):
        assert belief(out, m, emap, "M") == "moved", (m, out[m]["beliefs"])
        mv = [d for d in out[m]["decisions"].values() if d.how == "moved"]
        assert len(mv) == 1 and mv[0].item == entry_of(emap, "M")[0]
    it = next(x for x in out["_pack"]["items"] if x["id"] == entry_of(emap, "M")[0])
    assert abs(it["thrMoved"] - 0.40) < 1e-9 and it["impostorMax"] < 0.35, it


# ----------------------------------------------------------------------------- 3. look-alikes, one moved

def test_lookalike_pair_one_in_place_one_moved():
    # L1 and L2 are two cups of one model (same embedding axis), at 1.8 and 2.4. L1 stays; L2 moves
    # to 3.2. The walk sees L1, the empty 2.4 and L2 at 3.2.
    # current: L1 matched (same label within 0.20 m); L2 at 3.2 has no entry within 0.20 m: new entry,
    #   duplicate.
    # generic: L1 in place (L2's place is 0.6 m off). Exclusivity: L1, matched in place, is no
    #   candidate elsewhere, so the sighting at 3.2 has one candidate, L2 (missed): update L2, moved.
    #   Without exclusivity L1 would tie with L2 (margin 0) and the sighting would be unresolved.
    # household: the pack groups L1 and L2 as look-alikes (not aliases: 0.6 m apart); the same decisions.
    hist_kfs = [(1.6, ["A2", "A3", "L1", "L2"])] * 3
    test_kfs = [(1.6, ["A2", "A3", "L1"]), (1.6, ["A2", "A3", "L1"]), (2.6, ["L1", "L2"]), (2.6, ["L1", "L2"])]
    objs = {"L1": ("cup", 4), "L2": ("cup", 4)}
    history, tests, labels = scene(objs, {"L1": (1.8, 0, 0), "L2": (2.4, 0, 0)}, {"L1": (1.8, 0, 0), "L2": (3.2, 0, 0)},
                                   hist_kfs, test_kfs, OFF_SMALL)
    out, scores = run_all(history, tests[0][1], labels)
    good = {"A2": "correct_update", "A3": "correct_update", "L1": "correct_update", "L2": "correct_update"}
    expect(scores, {"current": dict(good, L2="duplicate"), "generic": good, "household": good, "extra_view": good})
    emap = out["_emap"]
    pk = {x["id"]: x for x in out["_pack"]["items"]}
    l1, l2 = entry_of(emap, "L1")[0], entry_of(emap, "L2")[0]
    assert pk[l1]["group"] == pk[l2]["group"] and pk[l1]["aliases"] == [] and pk[l2]["aliases"] == []
    assert belief(out, "generic", emap, "L1") == "seen_in_place" and belief(out, "generic", emap, "L2") == "moved"
    # the same scene with exclusivity taken away (L1 not matched in place: its sighting is too far
    # from its place) shows what exclusivity buys: then the L2 sighting is unresolved
    p_far = ir.Params(PLACE_RADIUS_M=0.0)
    out2, scores2 = run_all(history, tests[0][1], labels, p=p_far)
    assert scores2["generic"]["L2"] == "unresolved", scores2["generic"]


# ----------------------------------------------------------------------------- 4. new same-model item

def test_new_same_model_item_while_both_known_are_in_place():
    # L1 and L2 (one model) stay at 1.8 and 2.4; a third copy N is put out at 3.0.
    # current: L1, L2 matched; N is 0.6 m from L2: a new entry, correct add.
    # generic: L1 and L2 in place, so neither can be N (exclusivity); no other cup is missed or
    #   unvisited: add. household: the same.
    hist_kfs = [(1.6, ["A2", "A3", "L1", "L2"])] * 3
    test_kfs = [(1.6, ["A2", "A3", "L1", "L2"])] * 2 + [(2.4, ["L1", "L2", "N"])] * 2
    objs = {"L1": ("cup", 4), "L2": ("cup", 4), "N": ("cup", 4)}
    history, tests, labels = scene(objs, {"L1": (1.8, 0, 0), "L2": (2.4, 0, 0)},
                                   {"L1": (1.8, 0, 0), "L2": (2.4, 0, 0), "N": (3.0, 0, 0)}, hist_kfs, test_kfs, OFF_SMALL)
    out, scores = run_all(history, tests[0][1], labels)
    good = {"A2": "correct_update", "A3": "correct_update", "L1": "correct_update", "L2": "correct_update",
            "N": "correct_add"}
    expect(scores, {"current": good, "generic": good, "household": good, "extra_view": good})


# ----------------------------------------------------------------------------- 5. four scans, old place not visited

def test_four_scans_old_place_not_visited_and_stability_prior():
    # Two history scans and two test scans. U (book... label "remote") sits at x = -1.5 in the history.
    # W ("mouse", 0.1 m) moves 0.16 m between H1 and H2 (the app links it: 0.16 + 0.02 < 0.20 m), so
    # the pack sees it move; the anchors do not move. H2 has its own launch offset (0.02 m).
    # T1: U has been moved to 2.8; the walk sees 0.3 to 3.2 only, never -1.5.
    #   current: U at 2.8: a new entry: duplicate. Its evidence: not observed.
    #   generic: U's place was not looked at, so U can only produce `unresolved` (rule e): the
    #     sighting at 2.8 is unresolved [U]. household: the same.
    # T2 (another launch, offset 0.06 m): U still at 2.8, but the walk covers 0.2 to 1.6 only: no
    #   sighting of U. Every method keeps U, not observed.
    objects = dict(ANCHORS, U=("remote", 5), W=("mouse", 6))
    h_pos = dict(ANCHOR_POS, U=(-1.5, 0, 0), W=(1.8, 0, 0))
    h2_pos = dict(h_pos, W=(1.96, 0, 0))
    t_pos = dict(ANCHOR_POS, U=(2.8, 0, 0), W=(1.8, 0, 0))
    hk = [(0.0, ["U", "A1", "A2", "A3", "W"])] * 3
    t1k = [(1.6, ["A1", "A2", "A3", "W"])] * 2 + [(2.0, ["A2", "A3", "W", "U"])] * 2
    t2k = [(0.9, ["A1", "A2", "A3"])] * 3
    history, tests, labels = make_world(objects, [
        {"id": "h1", "role": "history", "pos": h_pos, "kfs": hk},
        {"id": "h2", "role": "history", "pos": h2_pos, "kfs": hk, "offset": (0.02, 0.0, 0.0)},
        {"id": "t1", "role": "test", "pos": t_pos, "kfs": t1k, "offset": OFF_SMALL},
        {"id": "t2", "role": "test", "pos": t_pos, "kfs": t2k, "offset": (0.0, 0.0, 0.06)}])
    assert len(history.entries) == 5, list(history.entries)        # one entry per object: no duplicates in history
    out1, s1 = run_all(history, tests[0][1], labels)
    anchors = {c: "correct_update" for c in ANCHORS}
    expect(s1, {"current": dict(anchors, U="duplicate", W="correct_update"),
                "generic": dict(anchors, U="unresolved", W="correct_update"),
                "household": dict(anchors, U="unresolved", W="correct_update"),
                "extra_view": dict(anchors, U="unresolved", W="correct_update")})
    emap = out1["_emap"]
    u = entry_of(emap, "U")[0]
    for m in ("generic", "household"):
        assert out1[m]["beliefs"][u]["state"] == "unresolved" and out1[m]["beliefs"][u]["stateBeforeUnresolved"] == "not_observed"
        assert not any(d.kind == ir.UPDATE and d.item == u for d in out1[m]["decisions"].values())
    assert out1["current"]["beliefs"][u]["state"] == "not_observed"
    out2, s2 = run_all(history, tests[1][1], labels)
    for m in ("current", "generic", "household", "extra_view"):
        assert s2[m]["U"] == "kept_not_observed", (m, s2[m])
    assert out2["generic"]["beliefs"][u]["state"] == "not_observed"
    # the stability prior: W moved 0.16 m between the history scans (spread = half of it after the
    # H2 -> H1 correction); the anchors did not (spread near 0): stable, threshold + STABLE_MOVED_BONUS
    pk = {x["id"]: x for x in out1["_pack"]["items"]}
    w, a1 = pk[entry_of(emap, "W")[0]], pk[entry_of(emap, "A1")[0]]
    assert out1["_pack"]["stabilityPrior"] is True
    assert 0.06 < w["spreadM"] < 0.10 + 0.02 and a1["spreadM"] < 0.03, (w, a1)
    assert a1["stable"] and abs(a1["thrItem"] - (0.40 + ir.DEFAULT.STABLE_MOVED_BONUS)) < 1e-9, a1
    assert pk[u]["spreadM"] is not None and pk[u]["stable"]
    fr = out1["_pack"]["historyFrames"]["h2"]
    assert fr["kind"] != "none" and abs(fr["t"][0] + 0.02) < 0.01, fr
    # the same pack with one history scan has no stability prior
    one = ir.History(history.scans[:1], [(e, d) for e, d in history.entries.items()], "m1")
    assert ir.build_pack(one)["stabilityPrior"] is False


# ----------------------------------------------------------------------------- 6. extra view resolves

def test_extra_view_resolves_an_unresolved_case():
    # Look-alike cups L1 (1.8) and L2 (2.4). In the test L1 stays but the walk (camera at 3.3, seeing
    # 2.1 to 4.5) never looks at 1.8; L2 has moved to 3.2 and is detected once; its old place 2.4 is
    # looked at in 3 keyframes and empty. No anchor is in view: no alignment (fewer than 3 pairs).
    # current: the single detection at 3.2 makes a tentative entry, dropped at the end: dropped.
    # generic and household: the sighting at 3.2 has L2 (missed) and L1 (not observed) as candidates,
    #   both at cosine 0.98: margin 0, unresolved [L2, L1].
    # extra_view: an extra view after the scan, aimed at L1's stored place, shows L1: in place, so L1
    #   is excluded (exclusivity) and the sighting at 3.2 resolves to L2, moved: one decision changes,
    #   unresolved -> update, and it is correct.
    hist_kfs = [(1.6, ["A2", "A3", "L1", "L2"])] * 3
    test_kfs = [(3.3, ["L2"]), (3.3, []), (3.3, [])]
    objs = {"L1": ("cup", 4), "L2": ("cup", 4)}
    history, tests, labels = scene(objs, {"L1": (1.8, 0, 0), "L2": (2.4, 0, 0)},
                                   {"L1": (1.8, 0, 0), "L2": (3.2, 0, 0)}, hist_kfs, test_kfs, OFF_SMALL)
    scan = tests[0][1]
    emap, _ = ir.entry_item_map(history.entries, labels)
    l1 = entry_of(emap, "L1")[0]
    x = ir.Obs("q0000001-x1", "t1", "x:q0000001-x1", 0, scan.t_end + 5000, None, 0.0,
               tuple(history.entries[l1]["mapPosition"]), emb(4, 77), extra=True, entry=l1)
    x.ms_to_capture, x.decision = 15000, {"wholeMs": 21000}
    out, scores = run_all(history, scan, labels, extra=[x])
    expect(scores, {"current": {"L2": "dropped", "L1": "kept_not_observed"},
                    "generic": {"L2": "unresolved", "L1": "kept_not_observed"},
                    "household": {"L2": "unresolved", "L1": "kept_not_observed"},
                    "extra_view": {"L2": "correct_update", "L1": "kept_not_observed"}})
    s = [s for s in out["_sightings"] if not s.extra]
    assert len(s) == 1 and len(s[0].obs) == 1          # the single detection is a sighting, not dropped
    d_h, d_x = out["household"]["decisions"][s[0].id], out["extra_view"]["decisions"][s[0].id]
    assert d_h.kind == ir.UNRESOLVED and set(d_h.candidates) == {l1, entry_of(emap, "L2")[0]}, d_h.as_dict()
    assert d_x.kind == ir.UPDATE and d_x.item == entry_of(emap, "L2")[0] and d_x.how == "moved", d_x.as_dict()
    xs = [t for t in out["extra_view"]["sightings"] if t.extra][0]
    assert out["extra_view"]["decisions"][xs.id].text() == "update %s" % l1
    assert out["generic"]["alignment"].kind == "none"
    # an extra view of L2's old (empty) place instead: it matches nothing in place, adds nothing, and
    # makes no difference here (that place was already looked at in 3 keyframes)
    l2 = entry_of(emap, "L2")[0]
    y = ir.Obs("q0000002-x1", "t1", "x:q0000002-x1", 0, scan.t_end + 9000, None, 0.0,
               tuple(history.entries[l2]["mapPosition"]), emb(9, 5), extra=True, entry=l2)
    out3, scores3 = run_all(history, scan, labels, extra=[y])
    ys = [t for t in out3["extra_view"]["sightings"] if t.extra][0]
    assert out3["extra_view"]["decisions"][ys.id].kind == ir.NONE and scores3["extra_view"]["L2"] == "unresolved"


# ----------------------------------------------------------------------------- the command on a pull

def write_pull(folder, history, tests, labels, extra_line=None, sheet_rows=None):
    """The world of make_world as v0.8 files (plan sections 3 and 6): events, obs/, scan/ with depth,
    inventory_log.jsonl; labels and sheet next to it. Returns their paths."""
    os.makedirs(os.path.join(folder, "obs"), exist_ok=True)
    events, inv = [], []
    sessions = {}

    def obs_line(o, scan):
        return {"v": 1, "type": "obs", "obs": o.id, "kind": "scan", "scan": scan, "kf": o.kf, "det": o.det, "t": o.t,
                "label": o.label, "score": o.score, "mapPoint": list(o.map_point), "sizeM": o.size_m,
                "emb": b64(o.emb), "embDim": DIM, "viewGroup": None,
                "view": "views/%s/%s.jpg" % (scan, o.id) if o.retained else None, "decision": o.decision}

    for k, s in enumerate(history.scans + [sc for _, sc in tests]):
        sess = "sess%04d" % k
        sessions[s.scan] = sess
        events.append({"t": s.t_start, "session": sess, "type": "scan_start", "scan": s.scan, "mapId": "m1",
                       "studyLogging": True})
        rows = [obs_line(o, s.scan) for o in s.observations]
        rows.append({"v": 1, "type": "scan_end", "scan": s.scan, "t": s.t_end, "created": sorted(s.created),
                     "tentativeDropped": [{"entry": e, "obs": []} for e in sorted(s.dropped)]})
        with open(os.path.join(folder, "obs", "%s.jsonl" % s.scan), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        kdir = os.path.join(folder, "scan", s.scan)
        os.makedirs(kdir, exist_ok=True)
        for kf, g in s.keyframes.items():
            events.append({"t": s.t_start + 1 + kf, "session": sess, "type": "scan_keyframe", "scan": s.scan, "index": kf})
            d = g.depth_map()
            with open(os.path.join(kdir, "kf%04d_depth.u16" % kf), "wb") as f:
                f.write(d.mm.tobytes())
            with open(os.path.join(kdir, "kf%04d.json" % kf), "w") as f:
                json.dump({"index": kf, "t": s.t_start + kf, "intrinsics": g.k, "mapToWorld": g.m2w,
                           "cameraPose": {"t": list(g.t), "q": list(g.q)}, "depthFromImage": list(d.a),
                           "depth": {"file": "kf%04d_depth.u16" % kf, "width": d.width, "height": d.height}}, f)
        for eid, out in s.evidence.items():
            events.append({"t": s.t_end - 1, "session": sess, "type": "inv_evidence", "scan": s.scan,
                           "inventoryId": eid, "outcome": out})
        events.append({"t": s.t_end, "session": sess, "type": "scan_end", "scan": s.scan})
    for eid, e in history.entries.items():
        made = next(s for s in history.scans if s.scan == e["createdScan"])
        inv.append({"t": made.t_end - 5, "op": "add", "entry": e})
    if extra_line:
        with open(os.path.join(folder, "obs", "extra_views.jsonl"), "w") as f:
            f.write(json.dumps(extra_line) + "\n")
        events.append({"t": extra_line["t"] + 6000, "session": sessions[tests[0][1].scan], "type": "query_end",
                       "query": extra_line["query"], "groundTruth": "present"})
    events.sort(key=lambda e: e["t"])
    inv.sort(key=lambda r: r["t"])
    for name, rows in (("events.jsonl", events), ("inventory_log.jsonl", inv)):
        with open(os.path.join(folder, name), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    lab = os.path.join(folder, "labels.private.csv")
    with open(lab, "w") as f:
        f.write("obs,item\n" + "".join("%s,%s\n" % kv for kv in sorted(labels.items())))
    sheet = None
    if sheet_rows:
        sheet = os.path.join(folder, "sheet.private.csv")
        cols = ["item_code", "same_model_group", "h1_present", "t1_present", "t1_moved", "t1_visited"]
        with open(sheet, "w") as f:
            f.write(",".join(cols) + "\n" + "".join(",".join(r.get(c, "") for c in cols) + "\n" for r in sheet_rows))
    return lab, sheet


def test_command_on_a_written_pull():
    # The extra-view scene of test 6, written to files and run through main(): the loaders, the pack
    # file, the extra-view window (the line comes after the test scan), the label and sheet readers.
    # Expected per condition, as worked out in test 6: L2 (moved) is dropped by current, unresolved by
    # generic and household, a correct update with the extra view; L1 (stays, not looked at) kept.
    hist_kfs = [(1.6, ["A2", "A3", "L1", "L2"])] * 3
    test_kfs = [(3.3, ["L2"]), (3.3, []), (3.3, [])]
    objs = {"L1": ("cup", 4), "L2": ("cup", 4)}
    history, tests, labels = scene(objs, {"L1": (1.8, 0, 0), "L2": (2.4, 0, 0)},
                                   {"L1": (1.8, 0, 0), "L2": (3.2, 0, 0)}, hist_kfs, test_kfs, OFF_SMALL)
    scan = tests[0][1]
    emap, _ = ir.entry_item_map(history.entries, labels)
    l1 = entry_of(emap, "L1")[0]
    t_x = scan.t_end + 5000
    extra = {"v": 1, "type": "obs", "obs": "q0000001-x1", "kind": "extra_view", "entry": l1, "mapId": "m1",
             "t": t_x, "emb": b64(emb(4, 77)), "query": "q0000001", "requestStartMs": t_x - 15000,
             "msSinceQueryStart": 15000, "target": {"mapPoint": history.entries[l1]["mapPosition"]}}
    rows = [{"item_code": c, "h1_present": "yes", "t1_present": "yes", "t1_moved": "yes" if c == "L2" else "no"}
            for c in ("A2", "A3", "L1", "L2")]
    folder = tempfile.mkdtemp(prefix="idr_pull_")
    try:
        lab, sheet = write_pull(folder, history, tests, labels, extra, rows)
        out_dir = os.path.join(folder, "out")
        buf = io.StringIO()
        code = ir.main([folder, "--history", "h1", "--test", "t1", "--sheet", sheet, "--out", out_dir], out=buf)
        text = buf.getvalue()
        assert code == 0, text
        assert "Labels: %d observations" % len(labels) in text, text
        assert "extra view: 1 extra views after this scan; household -> extra_view changed 1 of 1 sighting decisions: " \
               "unresolved -> update 1" in text, text
        assert "request to capture: median 15000 ms, range 15000-15000 (n = 1); whole query: median 21000 ms" in text, text
        for name, want in (("current", "dropped 1"), ("generic", "unresolved 1"), ("household", "unresolved 1"),
                           ("extra_view", "correct_update 1")):
            block = text.split("    %s:\n" % name, 1)[1]
            assert block.split("\n")[1].strip().startswith("moved") and want in block.split("\n")[1], (name, block[:300])
        assert os.path.exists(os.path.join(out_dir, "pack_t1.json")) and os.path.exists(os.path.join(out_dir, "result.json"))
        with open(os.path.join(out_dir, "result.json")) as f:
            res = json.load(f)
        assert res["params"]["SIM_MOVED"] == ir.DEFAULT.SIM_MOVED and "commit" in res["git"]
        assert res["testScans"][0]["pack"]["bytes"] == os.path.getsize(os.path.join(out_dir, "pack_t1.json"))
        # without labels: the label-free report only
        buf = io.StringIO()
        assert ir.main([folder, "--history", "h1", "--test", "t1", "--no-labels", "--out", out_dir], out=buf) == 0
        assert "scoring:" not in buf.getvalue() and "agreement per observation" in buf.getvalue()
    finally:
        shutil.rmtree(folder)


# ----------------------------------------------------------------------------- pieces

def test_alignment_fit_with_yaw_and_an_outlier():
    # Ten points, turned by 3 degrees and shifted; one pair is wrong by 1 m. The fit finds the yaw and
    # the shift and leaves the wrong pair out; a 20-degree yaw is refused and a translation remains.
    truth = ir.Align(3.0, (0.12, -0.03, -0.08))
    src = [(0.4 * i - 2.0, 0.1 * (i % 3), 0.3 * ((i * 7) % 5) - 0.6) for i in range(10)]
    pairs = [(s, truth.apply(s)) for s in src]
    pairs[4] = (pairs[4][0], tuple(c + d for c, d in zip(pairs[4][1], (1.0, 0.0, 0.0))))
    a = ir.robust_alignment(pairs)
    assert a.kind == "yaw+translation" and abs(a.yaw_deg - 3.0) < 1e-6 and a.info["inliers"] == 9, a.summary()
    assert all(abs(a.t[i] - truth.t[i]) < 1e-6 for i in range(3))
    for s in src:
        q = a.apply(s)
        assert ir.dist(a.inverse(q), s) < 1e-9
    # 20 degrees about the centre of six points 0.52 m around (2, 0, 1) (span over ALIGN_MIN_SPAN_M): a
    # translation still agrees with every pair (residuals 0.18 m), the yaw is refused (over ALIGN_MAX_YAW_DEG)
    near = [(2.0 + 0.52 * math.cos(k), 0.0, 1.0 + 0.52 * math.sin(k)) for k in range(6)]
    turn = ir.Align(20.0, (0.0, 0.0, 0.0))
    c = turn.apply((2.0, 0.0, 1.0))
    big = ir.Align(20.0, (2.0 - c[0] + 0.05, 0.0, 1.0 - c[2]))
    a2 = ir.robust_alignment([(s, big.apply(s)) for s in near])
    assert a2.kind == "translation" and abs(a2.info["yawRefusedDeg"] - 20.0) < 1e-6, a2.summary()
    assert ir.robust_alignment(pairs[:2]).kind == "none"
    # a pair that moved 0.16 m (inside the inlier band) must not make a yaw that is worse than the
    # translation: five pairs offset by 0.02 m, one of them moved by another 0.16 m
    five = [((x, 0.0, 0.0), (x - 0.02 + (0.16 if x == 1.8 else 0.0), 0.0, 0.0)) for x in (-1.5, 0.0, 0.6, 1.2, 1.8)]
    a3 = ir.robust_alignment(five)
    assert a3.kind == "translation" and abs(a3.t[0] + 0.02) < 1e-9, a3.summary()


def test_place_visibility_rules():
    # The app's evidence rule: in view (8 % margin, 0.3 to 3.0 m along the axis), then depth: hidden
    # when the measured depth is nearer than z - max(0.15, 0.15 z).
    g = ir.KeyGeom(INTR, (0.0, 0.0, CAM_Z), (0.0, 0.0, 0.0, 1.0), M2W, uniform_depth())
    assert ir.classify_place((0.0, 0.0, 0.0), g, None) == ("observable", "saved_depth")
    assert ir.classify_place((1.5, 0.0, 0.0), g, None) == ("out", "geometry")          # u = 1960 > 1766
    assert ir.classify_place((0.0, 0.0, -0.5), g, None) == ("hidden", "saved_depth")   # z 2.0, depth 1.5
    assert ir.classify_place((0.0, 0.0, 2.0), g, None) == ("out", "geometry")          # behind the camera
    g2 = ir.KeyGeom(INTR, (0.0, 0.0, CAM_Z), (0.0, 0.0, 0.0, 1.0), M2W, None)
    assert ir.classify_place((0.0, 0.0, 0.0), g2, {"vis": "observable"}) == ("observable", "phone")
    assert ir.classify_place((0.0, 0.0, 0.0), g2, None) == ("no_depth", "none")
    assert ir.classify_place((0.0, 0.0, 0.0), None, {"vis": "hidden"}) == ("hidden", "phone")


def test_sightings_split_by_keyframe_and_appearance():
    # Four observations in one view group: two of one keyframe are two sightings; a later one joins
    # the most similar; one of another appearance starts its own; none is dropped.
    mk = lambda oid, kf, axis, x: ir.Obs(oid, "s", kf, 0, kf * 500, "cup", 0.5, (x, 0, 0), emb(axis, kf), view_group=0)  # noqa: E731
    scan = ir.ScanData("s", [mk("a", 0, 1, 0.0), mk("b", 0, 2, 0.1), mk("c", 1, 2, 0.1), mk("d", 2, 3, 0.05)])
    ss = ir.build_sightings(scan)
    assert [[o.id for o in s.obs] for s in ss] == [["a"], ["b", "c"], ["d"]]


def test_scoring_rules_by_hand():
    # Items X, Y known (entries ex, ey); Z new. Observations: x1 x2 of X, y1 of Y, z1 z2 of Z, u1 unknown.
    # x1 -> update ex, x2 -> add (sighting s9): X duplicate. y1 -> update ex: X and Y wrong merge
    # (entry ex holds two items); wrong merge comes first. z1, z2 -> add s5: Z correct add.
    D = ir.Decision
    emap = {"ex": "X", "ey": "Y"}
    labels = {"x1": "X", "x2": "X", "y1": "Y", "z1": "Z", "z2": "Z", "u1": "unknown"}
    dec = {"x1": (D(ir.UPDATE, "ex"), None), "x2": (D(ir.ADD), "s9"), "y1": (D(ir.UPDATE, "ex"), None),
           "z1": (D(ir.ADD), "s5"), "z2": (D(ir.ADD), "s5"), "u1": (D(ir.ADD), "s7")}
    out, excl = ir.score_method(dec, {}, list(labels), labels, emap, ["X", "Y", "Z"])
    assert out == {"X": "wrong_merge", "Y": "wrong_merge", "Z": "correct_add"} and excl["unknown"] == 1, out
    dec["y1"] = (D(ir.UPDATE, "ey"), None)
    out, _ = ir.score_method(dec, {}, list(labels), labels, emap, ["X", "Y", "Z"])
    assert out == {"X": "duplicate", "Y": "correct_update", "Z": "correct_add"}, out
    dec["z2"] = (D(ir.ADD), "s9")           # Z shares a new entity with X: both wrong merge
    out, _ = ir.score_method(dec, {}, list(labels), labels, emap, ["X", "Y", "Z"])
    assert out["X"] == out["Z"] == "wrong_merge", out
    dec = {"y1": (D(ir.UNRESOLVED, None, candidates=["ey"]), "s1")}
    out, _ = ir.score_method(dec, {}, ["y1"], labels, emap, ["X", "Y"])
    assert out == {"X": "kept_not_observed", "Y": "unresolved"}, out


def test_conditions_from_the_sheet():
    rows = [{"item_code": "I01", "h1_present": "yes", "t1_present": "yes", "t1_moved": "no", "t1_visited": "yes"},
            {"item_code": "I02", "h1_present": "yes", "h1_place_code": "P01", "t1_place_code": "P05", "t1_moved": "no"},
            {"item_code": "I03", "h1_present": "yes", "t1_visited": "no"},
            {"item_code": "I04", "same_model_group": "G1", "h1_present": "no", "t1_present": "yes"},
            {"item_code": "I05", "h1_present": "yes", "t1_present": "yes"}]
    c = ir.conditions(["I01", "I02", "I03", "I04", "I05", "I06"], {"I01", "I02", "I03"}, rows, 1, 1)
    assert c == {"I01": "stays", "I02": "moved", "I03": "not_visited", "I04": "new_same_model",
                 "I05": "new_to_inventory", "I06": "new (no sheet)"}, c


def test_parameters_are_one_set_and_freeze_round_trips():
    p = ir.Params(SIM_MOVED=0.7)
    assert p.SIM_MOVED == 0.7 and ir.DEFAULT.SIM_MOVED == ir.PARAMETERS["SIM_MOVED"][0]
    try:
        ir.Params(NOPE=1)
        raise AssertionError("an unknown parameter must be refused")
    except KeyError:
        pass
    folder = tempfile.mkdtemp(prefix="idr_freeze_")
    try:
        path = os.path.join(folder, "frozen.json")
        buf = io.StringIO()
        assert ir.main(["--freeze", path], out=buf) == 0
        with open(path) as f:
            fz = json.load(f)
        assert fz["params"] == dict(ir.DEFAULT.as_dict()) and "commit" in fz["git"]
        fz["params"]["SIM_MOVED"] = 0.99
        with open(path, "w") as f:
            json.dump(fz, f)
        buf = io.StringIO()
        assert ir.main(["--frozen", path, folder], out=buf) == 3 and "SIM_MOVED" in buf.getvalue()
    finally:
        shutil.rmtree(folder)


def test_float16_pack_keeps_cosines():
    v = [emb(3, k) for k in range(5)]
    for a in v:
        b = ir.unit(ir.decode_f16(ir.encode_f16(a)))
        assert abs(ir._dot(a, b) - 1.0) < 1e-5
    assert abs(ir._dot(v[0], v[1]) - ir._dot(ir.unit(ir.decode_f16(ir.encode_f16(v[0]))), v[1])) < 1e-3


if __name__ == "__main__":
    n_failed = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        try:
            fn()
            print("ok   ", name)
        except Exception as e:   # noqa: BLE001
            n_failed += 1
            import traceback
            print("FAIL ", name, type(e).__name__, str(e)[:3000])
            traceback.print_exc(limit=3)
    print("%d failed" % n_failed)
    sys.exit(1 if n_failed else 0)
