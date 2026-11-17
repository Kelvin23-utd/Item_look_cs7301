#!/usr/bin/env python3
"""Tests for the offline baseline benchmarks (bench_common, bench_match, bench_detect, bench_judge_read).

Run:  analysis/.venv-bench/bin/python analysis/test_bench.py      (one line per test)
      analysis/.venv-bench/bin/python -m pytest analysis/test_bench.py -q

Needs numpy and Pillow (and OpenCV for one test); no model weights and no real data. Every data
folder here is built by the test in a temporary directory that is removed afterwards. Images are
written as PNG bytes (even under a .jpg name, which Pillow reads by content), so crops can be
checked pixel for pixel.
"""
import itertools
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench_common as bc  # noqa: E402
import bench_detect as bd_  # noqa: E402
import bench_judge_read as jr  # noqa: E402
import bench_match as bm  # noqa: E402


class tmp:
    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="fb_bench_test_")
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def coded_sensor(h, w):
    """A sensor image whose pixel (y, x) is (x, y, 7): every crop says where it came from."""
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = np.arange(w)[None, :]
    img[..., 1] = np.arange(h)[:, None]
    img[..., 2] = 7
    return img


def save_png(path, arr):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path, format="PNG")


# ----------------------------------------------------------------------------- crops

def test_crop_box_matches_camera_frame_crop_upright():
    # CameraFrame.cropUpright: clamp, truncate left/top, width = trunc(width) limited to the image.
    assert bc.crop_box_int((10.7, 5.2, 30.9, 25.0), 60, 40) == (10, 5, 20, 19)
    assert bc.crop_box_int((-5, -5, 100, 100), 60, 40) == (0, 0, 60, 40)
    # A rectangle entirely right of the image is clamped to the last column, at least 1 px wide.
    assert bc.crop_box_int((70, 10, 80, 20), 60, 40) == (59, 10, 1, 10)
    assert bc.pad_rect((10, 10, 30, 20), 0.08) == (10 - 1.6, 10 - 0.8, 30 + 1.6, 20 + 0.8)


def test_rotation_round_trip_and_crop_orientation():
    s = coded_sensor(40, 60)
    for rot in (0, 90, 180, 270):
        up = bc.sensor_to_upright(s, rot)
        assert np.array_equal(bc.upright_to_sensor(up, rot), s)
    up = bc.sensor_to_upright(s, 90)
    assert up.shape == (60, 40, 3)
    # Clockwise: the sensor's top-left pixel ends up at the upright top-right.
    assert tuple(up[0, -1, :2]) == (0, 0)
    rect = (10, 5, 30, 25)
    a = bc.crop_from_upright(up, rect, 90)
    b = np.rot90(s[5:25, 10:30], k=-1)
    assert np.array_equal(a, b)
    # Half-size sensor image (scan keyframes): the rectangle is scaled before cutting.
    small = s[::2, ::2]
    c = bc.crop_from_sensor(small, (20, 10, 60, 50), 0, scale=0.5)
    assert np.array_equal(c, small[5:25, 10:30])


def test_iou_and_duplicates():
    assert close(bc.iou((0, 0, 2, 2), (1, 1, 3, 3)), 1 / 7)
    assert bc.iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    e1 = {"id": "a", "label": "bottle", "mapId": "m", "mapPosition": [0, 0, 0]}
    e2 = {"id": "b", "label": "bottle", "mapId": "m", "mapPosition": [0.3, 0.4, 0]}  # 0.5 m
    e3 = {"id": "c", "label": "cup", "mapId": "m", "mapPosition": [0, 0, 0]}
    e4 = {"id": "d", "label": "bottle", "mapId": "other", "mapPosition": [0, 0, 0]}
    assert bc.dup_entries(e1, e2)
    assert not bc.dup_entries(e1, dict(e2, mapPosition=[0.3, 0.41, 0]))
    assert not bc.dup_entries(e1, e3) and not bc.dup_entries(e1, e4) and not bc.dup_entries(e1, e1)


# ----------------------------------------------------------------------------- pair construction

T_SCAN, T_Q1, T_Q2, T_Q3, T_Q4 = 1000, 2000, 2100, 2200, 6000
SENSOR_W, SENSOR_H = 60, 40


def build_folder(root):
    """A pulled data folder with four queries, a scan of two saved keyframes and one person frame.

    Records (records.json, no log): R1 and R3 share lineage R1; R2; R4 saved at t=5000.
    Inventory: E1 bottle at the origin, E2 bottle 0.1 m away (a possible duplicate of E1),
    E3 cup 2 m away, all first seen at the scan (t=1000); E4 first seen at t=3000.
    Queries: q1 on R1 present, q2 on R2 absent, q3 on R1 displaced (left out), q4 on scanned E1
    present (t=6000); plus one query in an excluded developer session.
    """
    ev = []

    def add(**e):
        ev.append(e)

    rect_a, rect_b = [10, 5, 30, 25], [30, 10, 50, 30]
    for s in ("S1", "cab39b87"):
        add(session=s, type="session_start", t=0)
    queries = [("q1", "R1", "tap", None, T_Q1, "present"), ("q2", "R2", "tap", None, T_Q2, "absent"),
               ("q3", "R1", "tap", None, T_Q3, "displaced"), ("q4", "E1", "scan", "E1", T_Q4, "present")]
    for q, rec, src, inv, t, truth in queries:
        add(session="S1", type="query_start", t=t, query=q, record=rec, source=src, inventoryId=inv)
        add(session="S1", type="verify", t=t + 1, query=q, record=rec, attempt=1, rotation=90,
            bestSimilarity=0.8, frameFile="verify/%s_1_frame.jpg" % q,
            scores=[{"kind": "projected", "sim": 0.8, "rect": rect_a}, {"kind": "detector", "sim": 0.5, "rect": rect_b}])
        add(session="S1", type="query_end", t=t + 2, query=q, groundTruth=truth)
        save_png(os.path.join(root, "verify", "%s_1_frame.jpg" % q), bc.sensor_to_upright(coded_sensor(SENSOR_H, SENSOR_W), 90))
    add(session="cab39b87", type="query_start", t=T_Q1, query="qx", record="R1")
    add(session="cab39b87", type="verify", t=T_Q1 + 1, query="qx", record="R1", attempt=1, rotation=90,
        bestSimilarity=0.9, frameFile="verify/qx_1_frame.jpg", scores=[{"kind": "projected", "sim": 0.9, "rect": rect_a}])
    add(session="cab39b87", type="query_end", t=T_Q1 + 2, query="qx", groundTruth="present")
    save_png(os.path.join(root, "verify", "qx_1_frame.jpg"), bc.sensor_to_upright(coded_sensor(SENSOR_H, SENSOR_W), 90))
    with open(os.path.join(root, "events.jsonl"), "w") as f:
        for e in ev:
            f.write(json.dumps(e) + "\n")

    recs = [{"id": "R1", "lineageId": "R1", "lastSeenMs": 500, "sessionId": "S1", "cropFile": "crops/R1.jpg"},
            {"id": "R2", "lineageId": "R2", "lastSeenMs": 500, "sessionId": "S1", "cropFile": "crops/R2.jpg"},
            {"id": "R3", "lineageId": "R1", "lastSeenMs": 500, "sessionId": "S1", "cropFile": "crops/R3.jpg"},
            {"id": "R4", "lineageId": "R4", "lastSeenMs": 5000, "sessionId": "S1", "cropFile": "crops/R4.jpg"}]
    with open(os.path.join(root, "records.json"), "w") as f:
        json.dump(recs, f)
    for r in recs:
        save_png(os.path.join(root, r["cropFile"]), np.full((8, 8, 3), 50, np.uint8))

    def entry(eid, label, pos, first):
        return {"id": eid, "label": label, "mapId": "m", "mapPosition": pos, "firstSeenMs": first,
                "lastSeenMs": first, "sessionId": "S1", "cropFile": "inventory/%s.jpg" % eid}
    entries = [entry("E1", "bottle", [0, 0, 0], T_SCAN), entry("E2", "bottle", [0.1, 0, 0], T_SCAN),
               entry("E3", "cup", [2, 0, 0], T_SCAN), entry("E4", "book", [3, 0, 0], 3000)]
    with open(os.path.join(root, "inventory.json"), "w") as f:
        json.dump({"version": 1, "entries": entries}, f)
    for e in entries:
        save_png(os.path.join(root, e["cropFile"]), np.full((8, 8, 3), 90, np.uint8))

    scan = os.path.join(root, "scan", "SC1")
    full = coded_sensor(SENSOR_H * 2, SENSOR_W * 2)      # full-resolution sensor image
    half = full[::2, ::2]                                # the saved half-size keyframe

    def det(box, label, entry_id, kept=True):
        return {"box": box, "label": label, "score": 0.6, "kept": kept, "entry": entry_id}

    kfs = [
        (0, [det([20, 10, 60, 50], "bottle", "E1"), det([70, 20, 110, 60], "cup", "E3"),
             det([0, 0, 10, 10], "cup", None, kept=False)], True),
        (1, [det([22, 12, 62, 52], "bottle", "E1"), det([60, 30, 100, 70], "bottle", "E2"),
             det([5, 5, 25, 25], "bowl", "T9")], True),           # T9: tentative, never real
        (2, [det([20, 10, 60, 50], "bottle", "E1")], False),       # person in view: not saved
    ]
    for idx, dets, saved in kfs:
        stem = "kf%04d" % idx
        d = {"scan": "SC1", "index": idx, "t": T_SCAN + idx, "session": "S1", "rotation": 0,
             "image": {"file": stem + ".jpg", "width": SENSOR_W, "height": SENSOR_H,
                       "sourceWidth": SENSOR_W * 2, "sourceHeight": SENSOR_H * 2} if saved else None,
             "imageSkipped": None if saved else "person", "detections": dets}
        os.makedirs(scan, exist_ok=True)
        with open(os.path.join(scan, stem + ".json"), "w") as f:
            json.dump(d, f)
        if saved:
            save_png(os.path.join(scan, stem + ".jpg"), half)
    return full


def test_build_pairs_counts_labels_and_variants():
    with tmp() as root:
        full = build_folder(root)
        data = bc.BenchData([root])
        ps = bc.build_pairs(data)
        by = lambda name: [ps.pairs[i] for i in ps.select(name)]  # noqa: E731

        # Verify pairs: q1 same, q2 different, q3 (displaced) and the excluded session left out.
        v = by("verify")
        assert sorted((p["capture"].split(":")[1], p["label"]) for p in v) == [("q1", 1), ("q2", 0), ("q4", 1)]
        assert all(p["item_a"] == p["item_b"] and len(p["a"]) == 2 for p in v)
        assert {c["item"] for c in ps.captures} == {"rec:R1", "rec:R2", "inv:E1"}

        # Impostors: other items known at the capture time, same lineage excluded.
        imp = [p for p in ps.pairs if p["set"] == "verify_imp"]
        others = {}
        for p in imp:
            others.setdefault(p["capture"].split(":")[1], set()).add((p["item_b"], p["dup"]))
        assert others["q1"] == {("rec:R2", False), ("inv:E1", False), ("inv:E2", False), ("inv:E3", False)}
        assert others["q2"] == {("rec:R1", False), ("inv:E1", False), ("inv:E2", False), ("inv:E3", False)}
        assert others["q4"] == {("rec:R1", False), ("rec:R2", False), ("rec:R4", False), ("inv:E2", True),
                                ("inv:E3", False), ("inv:E4", False)}
        vi = by("verify+imp")
        assert sum(p["label"] == 1 for p in vi) == 2 and sum(p["label"] == 0 for p in vi) == 1 + 14 - 1

        # Scan pairs: four saved detections of real entries (E1, E3 | E1, E2).
        s = by("scan")
        assert len(s) == 6 and sum(p["label"] for p in s) == 1
        assert sum(p["dup"] for p in s) == 2
        nd = by("scan-nodup")
        assert len(nd) == 4 and sum(p["label"] for p in nd) == 1
        assert any("2 of tentative" not in n and "1 of tentative" in n and "1 in keyframes not saved" in n for n in ps.notes)

        # Cross pairs: only the capture of the scanned entry E1.
        c = by("cross")
        assert len(c) == 4 and sum(p["label"] for p in c) == 2 and {p["capture"].split(":")[1] for p in c} == {"q4"}
        assert len(by("cross-nodup")) == 3

        # Crops are cut where the logged rectangles say, turned upright.
        cand = ps.crops["cand:S1:q1:1:0"]
        assert np.array_equal(cand, np.rot90(coded_sensor(SENSOR_H, SENSOR_W)[5:25, 10:30], k=-1))
        # Scan crop: box grown by 8% per side, cut from the half-size image.
        r = bc.clamp_rect(bc.pad_rect((20, 10, 60, 50), 0.08), SENSOR_W * 2, SENSOR_H * 2)
        x, y, w, h = bc.crop_box_int(tuple(v * 0.5 for v in r), SENSOR_W, SENSOR_H)
        assert np.array_equal(ps.crops["det:SC1:0000:0"], full[::2, ::2][y:y + h, x:x + w])


def test_build_pairs_without_excluding_developer_sessions():
    with tmp() as root:
        build_folder(root)
        ps = bc.build_pairs(bc.BenchData([root], excluded=()))
        assert len([p for p in ps.pairs if p["set"] == "verify"]) == 4


# ----------------------------------------------------------------------------- metrics

def brute_auc(pos, neg):
    return sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg) / (len(pos) * len(neg))


def test_auc_against_brute_force_with_ties_and_weights():
    rng = np.random.default_rng(1)
    for _ in range(20):
        s = rng.integers(0, 6, size=30).astype(float)
        y = rng.integers(0, 2, size=30)
        if y.min() == y.max():
            continue
        assert close(bc.auc_w(s, y), brute_auc(s[y == 1], s[y == 0]))
        w = rng.integers(0, 4, size=30)
        if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
            continue
        rep = np.repeat(np.arange(30), w)
        assert close(bc.auc_w(s, y, w), brute_auc(s[rep][y[rep] == 1], s[rep][y[rep] == 0]))
    assert math.isnan(bc.auc_w([1, 2], [1, 1]))


def test_eer_tar_and_fixed_threshold_by_hand():
    s = [0.9, 0.6, 0.4, 0.7, 0.3]
    y = [1, 1, 1, 0, 0]
    assert close(bc.auc_w(s, y), 4 / 6)
    # At 0.7: FAR 1/2, FRR 2/3; at 0.6: FAR 1/2, FRR 1/3. The two cross half-way, at 0.5.
    assert close(bc.eer_w(s, y), 0.5)
    assert close(bc.tar_at_far_w(s, y, far=0.05), 1 / 3)
    assert close(bc.tar_at_far_w(s, y, far=0.5), 1.0)
    assert bc.rates_at_w(s, y, 0.6) == (2 / 3, 0.5)
    assert bc.eer_w([3, 4, 1, 2], [1, 1, 0, 0]) == 0.0
    assert close(bc.eer_w([1, 2, 3, 4], [1, 1, 0, 0]), 1.0)
    fpr, tpr, thr = bc.roc_w(s, y)
    assert fpr[0] == 0 and tpr[0] == 0 and fpr[-1] == 1 and tpr[-1] == 1 and thr[0] == np.inf


def test_item_weights_and_bootstrap_resample_whole_items():
    index = {"a": 0, "b": 1, "c": 2}
    w = bc.item_weights(["a", "a", "b"], ["a", "b", "c"], np.array([2.0, 3.0, 0.0]), index)
    assert list(w) == [2.0, 6.0, 0.0]
    items = list("abcdef")
    # One same-item pair per item: the weights of every resample sum to the number of items.
    res = bc.item_bootstrap(items, items, lambda wt: np.array([wt.sum()]), n_boot=200)
    assert res["ok"] and res["lo"][0] == 6 and res["hi"][0] == 6
    few = bc.item_bootstrap(list("abc"), list("abc"), lambda wt: np.array([wt.sum()]), n_boot=50)
    assert not few["ok"] and math.isnan(few["lo"][0])
    # Same seed, same interval; and the point estimate lies inside it for a smooth statistic.
    rng = np.random.default_rng(3)
    a = [str(i) for i in rng.integers(0, 8, 60)]
    b = [str(i) for i in rng.integers(0, 8, 60)]
    sc = rng.random(60)
    lab = np.array([int(x == y_) for x, y_ in zip(a, b)])
    lab[:3] = 1
    stat = lambda wt: np.array([bc.auc_w(sc, lab, wt)])  # noqa: E731
    r1, r2 = bc.item_bootstrap(a, b, stat, n_boot=300), bc.item_bootstrap(a, b, stat, n_boot=300)
    assert np.array_equal(r1["lo"], r2["lo"]) and np.array_equal(r1["hi"], r2["hi"])
    full = stat(np.ones(60))[0]
    assert r1["lo"][0] <= full <= r1["hi"][0]


class _Fake(bm.Method):
    """Features are the crop's mean value; similarity is minus the absolute difference."""

    def features(self, rgb):
        return float(rgb.mean())

    def sim(self, fa, fb):
        return -abs(fa - fb)


def test_score_pairs_takes_the_best_candidate_and_evaluate_reports_delta_zero_for_ours():
    ps = bc.PairSet()
    for k, v in {"c1": 10, "c2": 50, "s": 52, "t": 200, "u": 49}.items():
        ps.add_crop(k, np.full((4, 4, 3), v, np.uint8))
    ps.pairs = [
        {"set": "scan", "a": ("c1", "c2"), "b": "s", "label": 1, "item_a": "i1", "item_b": "i1", "dup": False, "capture": None},
        {"set": "scan", "a": ("c1",), "b": "t", "label": 0, "item_a": "i1", "item_b": "i2", "dup": False, "capture": None},
        {"set": "scan", "a": ("c2",), "b": "u", "label": 0, "item_a": "i3", "item_b": "i4", "dup": True, "capture": None},
    ]
    scores, _, cost = bm.score_pairs(_Fake(), ps, lambda m: None)
    assert list(scores) == [-2.0, -190.0, -1.0]
    assert cost["n_crops"] == 5 and cost["n_comparisons"] == 4
    res = bm.evaluate(ps, {"ours": scores, "hist": -scores}, ["ours", "hist"], "scan", n_boot=50)
    assert res["counts"] == {"n_pos": 1, "n_neg": 2, "n_items": 4, "n_pos_items": 1}
    assert close(res["methods"]["ours"]["auc"][0], 0.5) and res["methods"]["ours"]["dauc"][0] == 0
    assert close(res["methods"]["hist"]["dauc"][0], 0.0)
    nd = bm.evaluate(ps, {"ours": scores}, ["ours"], "scan-nodup", n_boot=50)
    assert nd["counts"]["n_neg"] == 1 and nd["methods"]["ours"]["auc"][0] == 1.0


def test_hsv_histogram_intersection():
    h = bm.HsvHistogram()
    a = np.zeros((10, 10, 3), np.uint8)
    a[:, :5] = (255, 0, 0)
    b = a.copy()
    b[:, 5:] = (0, 0, 255)
    fa, fb = h.features(a), h.features(b)
    assert close(h.sim(fa, fa), 1.0) and close(fa.sum(), 1.0)
    assert close(h.sim(fa, fb), 0.5)          # the red half is shared, the rest differs


# ----------------------------------------------------------------------------- Track B

def uniform_depth(mm, w=12, h=8, scale=0.1):
    return bd_.DepthMap(np.full((h, w), mm, np.uint16), (scale, 0, 0, 0, scale, 0))


def test_keyframe_filter_rules_in_the_apps_order():
    depth = uniform_depth(1000)
    fx = 100.0
    d = lambda box, label, score: {"box": box, "label": label, "score": score}  # noqa: E731
    dets = [
        d((0, 0, 40, 40), "person", 0.9),
        d((5, 5, 15, 15), "cup", 0.5),          # inside the person box: on_person
        d((50, 0, 60, 10), "cup", 0.3),         # below the operating point
        d((50, 20, 60, 30), "sofa", 0.8),       # stop list
        d((70, 0, 80, 10), "bottle", 0.9),      # kept
        d((70, 0, 80, 11), "bottle", 0.7),      # same name, IoU > 0.6: duplicate
        d((50, 40, 120, 80), "book", 0.6),      # 70 px * 1 m / 100 px = 0.7 m: too large
    ]
    out = bd_.keyframe_filter(dets, 0.35, {"sofa"}, depth, fx)
    assert [o["reason"] for o in out] == ["person", "on_person", "low_score", "stop_list", None, "duplicate", "too_large"]
    assert close(out[4]["sizeM"], 0.1) and out[4]["depthM"] == 1.0
    far = bd_.keyframe_filter([d((0, 0, 10, 10), "cup", 0.9)], 0.35, set(), uniform_depth(5000), fx)
    assert far[0]["reason"] == "depth_range"
    none = bd_.keyframe_filter([d((0, 0, 10, 10), "cup", 0.9)], 0.35, set(), uniform_depth(0), fx)
    assert none[0]["reason"] == "no_depth"
    many = [d((i * 7.0, 0, i * 7 + 5.0, 5), "item%d" % i, 0.9 - i * 0.01) for i in range(11)]
    out = bd_.keyframe_filter(many, 0.35, set(), uniform_depth(1000, w=20), fx)
    assert [o["reason"] for o in out] == [None] * 10 + ["over_cap"]


def test_depth_median_in_box_matches_the_app():
    mm = np.arange(1, 13, dtype=np.uint16).reshape(3, 4) * 100   # 100 .. 1200 mm
    dm = bd_.DepthMap(mm, (1, 0, 0, 0, 1, 0))
    # Pixel centres inside [0.4, 2.6] x [0.4, 1.6]: columns 0..2, rows 0..1 -> values 1,2,3,5,6,7 (x100)
    assert dm.median_in_box((0.4, 0.4, 2.6, 1.6)) == 0.5      # values[n // 2] of 6 sorted values
    # Narrower than a pixel: the pixel under the middle.
    assert dm.median_in_box((1.6, 1.6, 1.7, 1.7)) == 0.6
    assert dm.median_in_box((10, 10, 12, 12)) is None
    dm0 = bd_.DepthMap(np.zeros((3, 4), np.uint16), (1, 0, 0, 0, 1, 0))
    assert dm0.median_in_box((0, 0, 4, 3)) is None


def test_box_mapping_between_upright_half_size_and_sensor():
    box = (10.0, 20.0, 110.0, 70.0)
    for rot in (0, 90, 180, 270):
        s = bd_.upright_box_to_sensor(box, 0.5, rot, 1920, 1080)
        back = bd_.sensor_box_to_upright(s, 0.5, rot, 1920, 1080)
        assert all(close(a, b, 1e-6) for a, b in zip(back, box))
    # Rotation 90: upright x runs along sensor -y.
    assert bd_.upright_box_to_sensor((0, 0, 10, 20), 0.5, 90, 1920, 1080) == (0, 1060, 40, 1080)


def test_vocabulary_file_and_parser():
    v = bd_.read_vocab()
    assert len(v["items"]) > 100 and "person" in v["stop"]
    assert not {n.lower() for n in v["items"]} & {n.lower() for n in v["stop"]}
    with tmp() as root:
        p = os.path.join(root, "v.txt")
        with open(p, "w") as f:
            f.write("# c\n[items]\ncup\nMug\n[stop]\nsofa\n")
        assert bd_.read_vocab(p) == {"items": ["cup", "Mug"], "stop": ["sofa"]}
        with open(p, "w") as f:
            f.write("[items]\ncup\n[stop]\nCup\n")
        try:
            bd_.read_vocab(p)
            raise AssertionError("a duplicate name must be refused")
        except ValueError:
            pass


def _fake_frames_and_results():
    frames = [{"id": "SC/%04d" % i, "scan": "SC", "index": i, "upright": np.zeros((20, 10, 3), np.uint8),
               "scale": 0.5, "rotation": 90, "sensor_w": 40, "sensor_h": 20} for i in range(3)]
    k = lambda box, label, score: {"box": box, "label": label, "score": score, "reason": None}  # noqa: E731
    results = {
        "effdet0": {"SC/0000": [k((0, 0, 10, 10), "cup", 0.9)], "SC/0001": [], "SC/0002": [k((0, 0, 4, 4), "book", 0.5)]},
        "owlv2": {"SC/0000": [k((1, 1, 10, 10), "mug", 0.4), k((20, 0, 30, 10), "cup", 0.3)], "SC/0001": [],
                  "SC/0002": [dict(k((0, 0, 4, 4), "book", 0.9), reason="too_large")]},
    }
    return frames, results


def test_judging_sheet_pools_boxes_and_hides_detectors():
    frames, results = _fake_frames_and_results()
    objs = bd_.cluster_frame({k: results[k]["SC/0000"] for k in results})
    assert len(objs) == 2 and sorted(m[0] for m in objs[0]["members"]) == ["effdet0", "owlv2"]
    sheet, key = bd_.build_judging(frames, results, ["effdet0", "owlv2"], seed=5)
    text = json.dumps([sf for sf, _ in sheet])
    assert "effdet0" not in text and "owlv2" not in text
    assert sorted(f["frame"] for f in key["frames"]) == ["SC/0000", "SC/0001", "SC/0002"]
    f0 = next(f for f in key["frames"] if f["frame"] == "SC/0000")
    assert sorted(sorted(o["names"]) for o in f0["objects"]) == [["cup"], ["cup", "mug"]]
    f2 = next(f for f in key["frames"] if f["frame"] == "SC/0002")
    assert [m["detector"] for o in f2["objects"] for m in o["members"]] == ["effdet0"]   # dropped boxes are not judged
    sheet2, key2 = bd_.build_judging(frames, results, ["effdet0", "owlv2"], seed=5)
    assert key2 == key


def test_judgments_read_back_by_hand():
    key = {"sheet_id": "x", "detectors": ["A", "B"], "frames": [
        {"id": "F1", "objects": [
            {"id": "F1-1", "names": ["cup", "mug"], "members": [{"detector": "A", "label": "cup"}, {"detector": "B", "label": "mug"}]},
            {"id": "F1-2", "names": ["bowl"], "members": [{"detector": "B", "label": "bowl"}]},
            {"id": "F1-3", "names": ["book"], "members": [{"detector": "A", "label": "book"}]}]},
        {"id": "F2", "objects": [
            {"id": "F2-1", "names": ["cup"], "members": [{"detector": "A", "label": "cup"}]}]}]}
    J = {"F1-1": {"q1": "item", "names": {"cup": "right", "mug": "close"}},
         "F1-2": {"q1": "item", "names": {"bowl": "wrong"}},
         "F1-3": {"q1": "not_item", "names": {}},
         "F2-1": {"q1": "item", "names": {}}}                      # F2 incomplete: name not judged
    done = [f for f in key["frames"] if jr.complete(f, J)]
    assert [f["id"] for f in done] == ["F1"]
    s, pool = jr.summarise([jr.frame_counts(f, J, key["detectors"]) for f in done], key["detectors"])
    assert pool == {"items": 2, "items_ok": 1}
    # A: boxes cup (item, right) and book (not an item). B: mug (item, close) and bowl (item, wrong).
    assert s["A"]["item_prec"] == 0.5 and s["A"]["name_strict"] == 1.0 and s["A"]["name_lenient"] == 1.0
    assert s["B"]["item_prec"] == 1.0 and s["B"]["name_strict"] == 0.0 and s["B"]["name_lenient"] == 0.5
    assert s["A"]["recall"] == 1.0 and s["B"]["recall"] == 1.0
    assert s["A"]["found"] == 0.5 and s["B"]["found"] == 1.0


def test_judgment_reader_cli_refuses_another_sheet():
    with tmp() as root:
        kp, jp = os.path.join(root, "k.json"), os.path.join(root, "j.json")
        with open(kp, "w") as f:
            json.dump({"sheet_id": "a", "detectors": [], "frames": []}, f)
        with open(jp, "w") as f:
            json.dump({"sheet_id": "b", "judgments": {}}, f)
        try:
            jr.main([jp, "--key", kp])
            raise AssertionError("must refuse")
        except SystemExit as e:
            assert "sheet" in str(e)


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
