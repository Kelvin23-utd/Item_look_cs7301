#!/usr/bin/env python3
"""Tests of the scan-merging replay (analysis/scan_replay.py). Small hand-built cases; every expected
value is worked out in the comments. No phone data, no image, no model.

    python3 analysis/test_scan_replay.py
    analysis/.venv-bench/bin/python analysis/test_scan_replay.py
"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scan_replay as sr  # noqa: E402


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


def det(key, label, score, x, size=0.1, t=1000, kf=0, box=(0, 0, 10, 10), order=0, emb=None, logged=None):
    """A detection on the x axis of the map frame."""
    return sr.Det(key, kf, t, label, score, box, (x, 0.0, 0.0), size, embedding=emb, logged=logged, order=order)


def real_entry(eid, label, x, size=0.1, t=0, emb=None):
    """A real entry built from one sighting (as if promoted in an earlier scan)."""
    e = sr.Entry(eid, det("seed-" + eid, label, 0.5, x, size=size, t=t, emb=emb))
    e.tentative = False
    return e


# ----------------------------------------------------------------------------- small rules

def test_radius():
    # clamp(0.5 x max(sizes) + 0.15, 0.20, 0.50)
    assert sr.radius(0.02, 0.02) == 0.20          # 0.16 -> raised to the minimum
    assert close(sr.radius(0.3, 0.1), 0.30)       # 0.5 x 0.3 + 0.15; the larger size counts
    assert close(sr.radius(0.1, 0.3), 0.30)       # symmetric
    assert sr.radius(0.8, 0.1) == 0.50            # 0.55 -> capped at the maximum


def test_float32_quantities():
    # 0.55f is 0.550000011920929 in double; 0.5499999 rounds to 0.54999989... in float32, below it.
    assert close(sr.MERGE_SIMILARITY, 0.550000011920929, 1e-15)
    assert sr.f32(0.5499999) < sr.MERGE_SIMILARITY
    # quality = box area x score in float32: 10 x 10 x 0.5 = 50 exactly; with 0.3 (0.30000001192f)
    # the product 30.0000011920929 rounds to the float32 30.000001907348633.
    assert sr.quality((0, 0, 10, 10), 0.5) == 50.0
    assert sr.quality((0, 0, 10, 10), 0.3) == 30.000001907348633
    assert sr.box_area((5, 5, 3, 9)) == 0.0         # negative width -> empty box


def test_crop_size():
    # Box 100 x 50 at the corner: padded by 8 and 4 px, clamped at 0 -> (0, 0, 108, 54).
    assert sr.crop_size((0, 0, 100, 50), 1920, 1080) == (108.0, 54.0)
    # Inside the image: 100 + 2 x 8 = 116, 50 + 2 x 4 = 58.
    assert sr.crop_size((1800, 1000, 1900, 1050), 1920, 1080) == (116.0, 58.0)
    # Bottom clamped: 40 px high, padded by 3.2 -> top 1036.8, bottom 1083.2 -> 1080; height 43.2
    # (float32 43.199951...); width 60 + 2 x 4.8 = 69.6 (float32 69.6000977...).
    w, h = sr.crop_size((1850, 1040, 1910, 1080), 1920, 1080)
    assert abs(w - 69.6) < 1e-3 and abs(h - 43.2) < 1e-3


def test_world_from_map():
    # yaw +90 degrees turns local +X into parent -Z (Pose4); then the translation is added.
    p = sr.world_from_map((1.0, 0.0, 0.0), {"x": 1.0, "y": 2.0, "z": 3.0, "yawDeg": 90.0})
    assert close(p[0], 1.0, 1e-12) and p[1] == 2.0 and close(p[2], 2.0, 1e-12)


def test_processing_order_is_a_stable_sort():
    # Scores 0.4, 0.6, 0.4, 0.6 in detector order a, b, c, d -> b, d (0.6, detector order), a, c.
    ds = [det(k, "cup", s, 0.0, order=i) for i, (k, s) in enumerate([("a", 0.4), ("b", 0.6), ("c", 0.4), ("d", 0.6)])]
    assert [d.key for d in sr.processing_order(ds)] == ["b", "d", "a", "c"]
    assert [d.key for d in sr.processing_order(ds, reverse_ties=True)] == ["d", "b", "c", "a"]


# ----------------------------------------------------------------------------- matching

def test_distance_gate_includes_the_radius():
    # Entry and detection both 0.02 m: radius 0.20 m. At exactly 0.2 m the app's test "d > r" is
    # false, so the entry qualifies; at 0.3 m it does not and a new entry is made.
    rp = sr.Replay(sr.sim_none, [real_entry("E", "cup", 0.0, size=0.02)])
    assert sr.distance((0, 0, 0), (0.2, 0, 0)) == 0.2
    d = rp.decide(det("x", "cup", 0.5, 0.2, size=0.02), 5000, set())
    assert d.status == sr.OK and d.kind == sr.KIND_KNOWN and d.entry == "E" and d.by_label
    d = rp.decide(det("y", "cup", 0.5, 0.3, size=0.02), 5000, set())
    assert d.status == sr.OK and d.kind == sr.KIND_NEW and d.candidates == []


def test_radius_grows_with_the_larger_size():
    # Entry 0.3 m long at x = 0, detection 0.1 m at x = 0.25: radius 0.30 m covers 0.25 m.
    rp = sr.Replay(sr.sim_none, [real_entry("E", "laptop", 0.0, size=0.3)])
    d = rp.decide(det("x", "laptop", 0.5, 0.25, size=0.1), 5000, set())
    assert d.kind == sr.KIND_KNOWN and close(d.candidates[0][3], 0.30)


def test_other_label_needs_similarity():
    # Entry "bowl" 0.05 m away, detection "cup": qualifies only with similarity >= 0.55.
    e = real_entry("E", "bowl", 0.0)

    def run(sim):
        rp = sr.Replay(lambda d, en: sim, [e])
        return rp.decide(det("x", "cup", 0.5, 0.05), 5000, set())

    d = run(None)            # unknown: the decision cannot be made
    assert d.status == sr.NEEDS_SIM and d.missing == ["E"]
    assert d.if_similar == (sr.KIND_KNOWN, "E") and d.if_not == (sr.KIND_NEW, None)
    d = run(0.6)
    assert d.status == sr.OK and d.kind == sr.KIND_KNOWN and d.entry == "E" and d.by_label is False
    d = run(0.5)
    assert d.status == sr.OK and d.kind == sr.KIND_NEW
    d = run(0.550000011920929)          # exactly 0.55f: "sim < 0.55f" is false, so it qualifies
    assert d.kind == sr.KIND_KNOWN
    d = run(0.5499999)                  # float32 0.54999989 < 0.55f
    assert d.kind == sr.KIND_NEW


def test_nearest_qualifying_wins_and_unknowns_only_matter_when_nearer():
    # Same-label entries A at 0.10 m and B at 0.05 m: B, the nearer, wins.
    rp = sr.Replay(sr.sim_none, [real_entry("A", "cup", 0.10), real_entry("B", "cup", 0.05)])
    d = rp.decide(det("x", "cup", 0.5, 0.0), 5000, set())
    assert d.entry == "B" and close(d.dist, 0.05)
    # A "bowl" C at 0.12 m with unknown similarity is farther than B: B still wins, decidable.
    rp = sr.Replay(sr.sim_none, [real_entry("B", "cup", 0.05), real_entry("C", "bowl", 0.12)])
    d = rp.decide(det("x", "cup", 0.5, 0.0), 5000, set())
    assert d.status == sr.OK and d.entry == "B"
    # A "bowl" D at 0.02 m with unknown similarity is nearer than B: undecidable (D or B).
    rp = sr.Replay(sr.sim_none, [real_entry("B", "cup", 0.05), real_entry("D", "bowl", 0.02)])
    d = rp.decide(det("x", "cup", 0.5, 0.0), 5000, set())
    assert d.status == sr.NEEDS_SIM and d.missing == ["D"]
    assert d.if_similar == (sr.KIND_KNOWN, "D") and d.if_not == (sr.KIND_KNOWN, "B")


def test_equal_distances_first_candidate_wins():
    # A at x = -0.1 and B at x = +0.1 are both 0.1 m from the detection at 0: strict "<" keeps the
    # first in candidate order (store order), A.
    rp = sr.Replay(sr.sim_none, [real_entry("A", "cup", -0.1), real_entry("B", "cup", 0.1)])
    assert rp.decide(det("x", "cup", 0.5, 0.0), 5000, set()).entry == "A"


def test_tentative_then_promoted_after_300_ms():
    # t = 1000: new tentative entry. t = 1200 (200 ms later): matched, but too soon -> "tentative".
    # t = 1300 (300 ms after the first sighting): promoted (300 >= 300).
    rp = sr.Replay(sr.sim_none)
    d = rp.decide(det("a", "cup", 0.5, 0.0, t=1000), 1000, set())
    assert d.kind == sr.KIND_NEW
    rp.apply(d.det, 1000, d.kind, None, "T")
    assert rp.tent == ["T"] and rp.real == []
    d = rp.decide(det("b", "cup", 0.5, 0.01, t=1200), 1200, set())
    assert d.kind == sr.KIND_TENTATIVE and d.entry == "T"
    rp.apply(d.det, 1200, d.kind, d.entry, None)
    assert rp.entries["T"].tentative
    d = rp.decide(det("c", "cup", 0.5, 0.02, t=1300), 1300, set())
    assert d.kind == sr.KIND_PROMOTED
    rp.apply(d.det, 1300, d.kind, d.entry, None)
    assert rp.real == ["T"] and rp.tent == [] and rp.entries["T"].count == 3


def test_one_detection_per_entry_per_keyframe():
    # Two cups 0.02 m apart near one real entry, in one keyframe: the higher score (0.7) takes the
    # entry, the other (0.5) can no longer use it and becomes a new tentative entry.
    kfs = [(1, 5000, [det("lo", "cup", 0.5, 0.02, t=5000, order=0, logged={"kind": "new", "entry": "N"}),
                      det("hi", "cup", 0.7, 0.0, t=5000, order=1, logged={"kind": "known", "entry": "E"})])]
    decs, rp, dropped = sr.run_forced(kfs, sr.sim_none, [real_entry("E", "cup", 0.0)])
    assert [d.det.key for d in decs] == ["hi", "lo"]
    assert decs[0].entry == "E" and decs[1].kind == sr.KIND_NEW
    assert [r["status"] for r in sr.compare(decs)] == [sr.OK, sr.OK]
    assert dropped == ["N"]                 # seen once: dropped at the end of the scan
    # Without the limit, "lo" would also go to E: the variant shows a mismatch.
    decs, _, _ = sr.run_forced(kfs, sr.sim_none, [real_entry("E", "cup", 0.0)], sr.Rules(one_per_entry=False))
    assert [r["status"] for r in sr.compare(decs)] == [sr.OK, sr.MISMATCH]


def test_position_is_the_mean_of_the_latest_ten():
    # Sightings at x = 0, 1, ..., 11: the latest ten are 2..11, mean 6.5; the count is 12.
    e = sr.Entry("E", det("s0", "cup", 0.5, 0.0, t=0))
    for i in range(1, 12):
        e.add(det("s%d" % i, "cup", 0.5, float(i), t=i))
    assert e.position == (6.5, 0.0, 0.0) and e.count == 12 and len(e.sightings) == 10
    assert e.first_ms == 0 and e.last_ms == 11


def test_best_crop_replaced_only_when_quality_is_higher():
    # First crop 10 x 10 at score 0.5: quality 50. A 20 x 10 box at 0.3: 200 x 0.3 = 60.0000... > 50,
    # so size and embedding are replaced. A 10 x 10 box at 0.5 (quality 50 < 60) is not taken, nor is
    # an equal quality (strict ">").
    e = sr.Entry("E", det("a", "cup", 0.5, 0.0, size=0.10, emb=[1.0, 0.0]))
    assert e.add(det("b", "cup", 0.3, 0.0, size=0.20, box=(0, 0, 20, 10), emb=[0.0, 1.0]))
    assert e.size_m == 0.20 and e.embedding == [0.0, 1.0] and e.best_key == "b"
    assert not e.add(det("c", "cup", 0.5, 0.0, size=0.30, emb=[1.0, 1.0]))
    q = e.best_quality
    assert not e.add(det("d", "cup", 0.3, 0.0, size=0.40, box=(0, 0, 20, 10)))   # equal quality q
    assert e.best_quality == q and e.best_key == "b" and e.size_m == 0.20


def test_label_votes_keep_the_current_label_on_a_tie():
    # cup 0.5; + bowl 0.5 -> tie 0.5 : 0.5, the label stays cup; + bowl 0.25 -> bowl 0.75 > 0.5.
    e = sr.Entry("E", det("a", "cup", 0.5, 0.0))
    e.add(det("b", "bowl", 0.5, 0.0))
    assert e.label == "cup"
    e.add(det("c", "bowl", 0.25, 0.0))
    assert e.label == "bowl" and e.votes == {"cup": 0.5, "bowl": 0.75}


def test_similarity_follows_the_best_crop_embedding():
    # The entry's embedding is the best crop's: after a better crop with embedding (0, 1), a "cup"
    # detection with (0, 1) is similar (1.0) to the "bowl" entry; with (1, 0) it is not (0.0).
    e = sr.Entry("E", det("a", "bowl", 0.5, 0.0, emb=[1.0, 0.0]))
    e.add(det("b", "bowl", 0.9, 0.0, emb=[0.0, 1.0]))      # quality 90 > 50
    e.tentative = False
    rp = sr.Replay(sr.sim_embedding, [e])
    assert rp.decide(det("x", "cup", 0.5, 0.01, emb=[0.0, 1.0]), 9000, set()).entry == "E"
    assert rp.decide(det("y", "cup", 0.5, 0.01, emb=[1.0, 0.0]), 9000, set()).kind == sr.KIND_NEW
    assert rp.decide(det("z", "cup", 0.5, 0.01, emb=None), 9000, set()).status == sr.NEEDS_SIM


def test_logged_similarity_only_for_the_logged_entry():
    d = det("x", "cup", 0.5, 0.0, logged={"kind": "known", "entry": "E", "sim": 0.7})
    assert sr.sim_logged(d, real_entry("E", "bowl", 0.0)) == 0.7
    assert sr.sim_logged(d, real_entry("F", "bowl", 0.0)) is None


def test_forced_and_free_runs():
    # Keyframe 1 (t = 1000): a cup (new, app id A1). Keyframe 2 (t = 1500): the cup again 0.03 m
    # away (promoted A1) and a bowl 0.04 m from it (new, app id B1). The bowl's similarity to the cup
    # entry is unknown, which would make it undecidable if A1 were still a candidate.
    kfs = [(1, 1000, [det("k1/0", "cup", 0.6, 0.0, t=1000, kf=1, logged={"kind": "new", "entry": "A1"})]),
           (2, 1500, [det("k2/0", "cup", 0.6, 0.03, t=1500, kf=2, logged={"kind": "promoted", "entry": "A1"}),
                      det("k2/1", "bowl", 0.4, 0.07, t=1500, kf=2, order=1,
                          logged={"kind": "new", "entry": "B1"})])]
    decs, rp, dropped = sr.run_forced(kfs, sr.sim_none)
    st = [r["status"] for r in sr.compare(decs)]
    # The bowl comes after the cup in keyframe 2, so A1 is already used: no candidate is left and the
    # bowl is decidable after all (new). All three reproduce.
    assert st == [sr.OK, sr.OK, sr.OK]
    assert list(rp.entries) == ["A1"] and dropped == ["B1"]
    assert close(rp.entries["A1"].position[0], 0.015)          # mean of 0 and 0.03
    # Free run: the same ids by first appearance.
    decs, rp, dropped = sr.run_free(kfs, sr.sim_none)
    assert [r["status"] for r in sr.compare(decs)] == [sr.OK, sr.OK, sr.OK] and list(rp.entries) == ["A1"]
    # If the app had logged the second cup as a new entry, the replay disagrees (mismatch), and in the
    # free run the replay's own decision is kept.
    kfs[1][2][0].logged = {"kind": "new", "entry": "A2"}
    decs, rp, _ = sr.run_free(kfs, sr.sim_none)
    rows = sr.compare(decs)
    assert rows[1]["status"] == sr.MISMATCH and rows[1]["replay_kind"] == sr.KIND_PROMOTED


def test_free_run_names_replay_only_entries():
    # The app matched the second cup to A1, but it lies 0.5 m away: the replay makes a new entry,
    # which no app entry was created by, so it is named R1.
    kfs = [(1, 1000, [det("k1/0", "cup", 0.6, 0.0, t=1000, kf=1, logged={"kind": "new", "entry": "A1"})]),
           (2, 1500, [det("k2/0", "cup", 0.6, 0.5, t=1500, kf=2, logged={"kind": "promoted", "entry": "A1"})])]
    decs, rp, dropped = sr.run_free(kfs, sr.sim_none)
    rows = sr.compare(decs)
    assert rows[1]["status"] == sr.MISMATCH and rows[1]["replay_kind"] == sr.KIND_NEW
    assert sorted(dropped) == ["A1", "R1"]


def test_initial_entries_from_the_app():
    # A pass-2 scan starts with a pass-1 entry as stored by the app: its sightings carry on.
    stored = {"id": "P", "label": "cup", "labelVotes": {"cup": 1.0}, "mapPosition": [0.0, 0.0, 0.0], "sizeM": 0.1,
              "bestQuality": 50.0, "embedding": None, "sightings": [{"t": 1, "p": [-0.01, 0, 0], "sizeM": 0.1,
                                                                   "label": "cup", "score": 0.5, "kf": 0},
                                                                  {"t": 2, "p": [0.01, 0, 0], "sizeM": 0.1,
                                                                   "label": "cup", "score": 0.5, "kf": 1}],
              "sightingCount": 2, "firstSeenMs": 1, "lastSeenMs": 2}
    rp = sr.Replay(sr.sim_none, [sr.Entry.from_app(stored)])
    d = rp.decide(det("x", "cup", 0.5, 0.03, t=10_000), 10_000, set())
    assert d.kind == sr.KIND_KNOWN and d.entry == "P"
    rp.apply(d.det, 10_000, d.kind, d.entry, None)
    assert close(rp.entries["P"].position[0], 0.01) and rp.entries["P"].count == 3   # (-0.01+0.01+0.03)/3


def test_load_v07_folder():
    # A two-keyframe pull in the v0.7 layout, with the events in processing order.
    d = tempfile.mkdtemp(prefix="scanreplay_")
    try:
        os.makedirs(os.path.join(d, "scan", "s1"))
        k0 = {"scan": "s1", "index": 0, "t": 1000, "image": None, "detections": [
            {"box": [0, 0, 10, 10], "label": "cup", "score": 0.5, "kept": True, "reason": None, "depthM": 1.0,
             "sizeM": 0.1, "mapPoint": [0, 0, 0], "kind": "new", "entry": "A"},
            {"box": [0, 0, 10, 10], "label": "tv", "score": 0.9, "kept": False, "reason": "stop_list"}]}
        k1 = {"scan": "s1", "index": 1, "t": 1500, "image": {"file": "kf0001.jpg"}, "detections": [
            {"box": [0, 0, 10, 10], "label": "bowl", "score": 0.4, "kept": True, "reason": None, "depthM": 1.0,
             "sizeM": 0.1, "mapPoint": [1, 0, 0], "kind": "new", "entry": "B"},
            {"box": [0, 0, 20, 10], "label": "cup", "score": 0.5, "kept": True, "reason": None, "depthM": 1.0,
             "sizeM": 0.1, "mapPoint": [0.02, 0, 0], "kind": "promoted", "entry": "A"}]}
        for k in (k0, k1):
            with open(os.path.join(d, "scan", "s1", "kf%04d.json" % k["index"]), "w") as f:
                json.dump(k, f)
        ev = [{"type": "scan_keyframe", "scan": "s1", "index": 0,
               "results": [{"label": "cup", "score": 0.5, "kind": "new", "entry": "A", "sim": None, "distM": None}]},
              {"type": "scan_keyframe", "scan": "s1", "index": 1,
               "results": [{"label": "cup", "score": 0.5, "kind": "promoted", "entry": "A", "sim": 0.8, "distM": 0.02},
                           {"label": "bowl", "score": 0.4, "kind": "new", "entry": "B", "sim": None, "distM": None}]}]
        with open(os.path.join(d, "events.jsonl"), "w") as f:
            for e in ev:
                f.write(json.dumps(e) + "\n")
        kfs, info = sr.load_v07(d, "s1")
        assert info["order_equal"] == 2 and info["join_problems"] == [] and info["with_image"] == 1
        assert [x.key for x in kfs[1][2]] == ["kf0001/0", "kf0001/1"]        # file order; the stop-list tv is out
        cup = kfs[1][2][1]
        assert cup.logged["sim"] == 0.8 and cup.logged["dist"] == 0.02 and cup.has_image
        decs, rp, dropped = sr.run_forced(kfs, sr.sim_none)
        rows = sr.compare(decs)
        assert [r["status"] for r in rows] == [sr.OK] * 3
        assert close(rows[1]["replay_dist"], 0.02) and dropped == ["B"]
        # The promoted cup's best crop is the 20 x 10 box (quality 100 > 50) of keyframe 1.
        assert rp.entries["A"].best_kf == 1
    finally:
        shutil.rmtree(d)


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        try:
            fn()
            print("ok   ", name)
        except Exception as e:   # noqa: BLE001
            failed += 1
            print("FAIL ", name, type(e).__name__, e)
    print("%d failed" % failed)
    sys.exit(1 if failed else 0)
