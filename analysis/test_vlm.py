#!/usr/bin/env python3
"""Tests of the scoring code of the VLM naming experiment (vlm_common, vlm_report, vlm_baselines,
vlm_prepare, vlm_ws, vlm_cluster). No model, no network, no workstation, no cluster, and no private
model file (a fake one with made-up identifiers is used).

    python3 analysis/test_vlm.py                      # standard library parts
    analysis/.venv-bench/bin/python analysis/test_vlm.py   # also the parts that need numpy / Pillow

Every expected value is worked out by hand in the comments or checked against brute force.
"""
import argparse
import contextlib
import datetime
import io
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# The tests never read the private model file: a fake one with made-up identifiers (one alias has a former
# key, as the first workstation runs do).
FAKE_MODELS = os.path.join(tempfile.mkdtemp(prefix="vlmtest_models_"), "models.json")
with open(FAKE_MODELS, "w") as _f:
    json.dump({"models": {k: {"old_key": "oldkey_a" if k == "model_a" else None, "repo": "pub/fake-" + k,
                              "rev": "0" * 40, "gb": 1.0, "limit": {"image": 2}, "ignore": None}
                          for k in ("model_a", "model_b", "model_c", "model_d", "judge_a",
                                    "judge_b")}}, _f)
os.environ["FB_VLM_MODELS"] = FAKE_MODELS
import vlm_common as vc  # noqa: E402

try:
    import numpy  # noqa: F401
    import PIL  # noqa: F401
    HAVE_NP = True
except ImportError:
    HAVE_NP = False


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ----------------------------------------------------------------------------- names

def test_singular():
    cases = {"chips": "chip", "batteries": "battery", "boxes": "box", "dishes": "dish", "glasses": "glasses",
             "scissors": "scissors", "knives": "knife", "cups": "cup", "bus": "bus", "glass": "glass",
             "keys": "key", "toys": "toy", "cases": "case", "potatoes": "potato", "mug": "mug", "crisps": "crisp",
             "headphones": "headphones", "lens": "lens", "lenses": "lens", "matches": "match"}
    for w, s in cases.items():
        assert vc.singular(w) == s, (w, vc.singular(w), s)


def test_normalize_and_words():
    assert vc.name_words("My Phone") == ["phone"]
    assert vc.normalize("  The  Water-Bottles ") == "water bottle"
    assert vc.normalize("AA batteries!") == "aa battery"
    assert vc.normalize("") == "" and vc.name_words(None) == []
    assert vc.normalize("a") == ""          # only a stop word


def test_head_word():
    assert vc.head_word("bag of chips") == "bag"
    assert vc.head_word("chip bag") == "bag"
    assert vc.head_word("paper towel roll") == "roll"
    assert vc.head_word("holder for pens") == "holder"
    assert vc.head_word("bottles") == "bottle"
    assert vc.head_word(None) is None
    # a cut that would leave nothing falls back to the whole name
    assert vc.head_word("of mice") == "mouse"


def test_same_name():
    assert vc.same_name("water bottle", "Water Bottles")
    assert vc.same_name("plastic bottle", "water bottle")       # same head word
    assert vc.same_name("bag of chips", "snack bag")
    assert not vc.same_name("potato chips", "snack bag")         # chip vs bag
    assert not vc.same_name("mug", "cup")                        # no synonyms in the label-free rule
    assert not vc.same_name(None, "cup") and not vc.same_name("cup", "")


def test_reference_name():
    assert vc.reference_name("suger") == "sugar"
    assert vc.reference_name("my phone") == "my phone"
    assert vc.reference_name("item 14") is None and vc.reference_name("item1") is None
    assert vc.reference_name("object") is None and vc.reference_name(None) is None
    assert vc.reference_name("Fan") == "fan"


def test_human_match():
    m = vc.human_match
    assert m("smartphone", "my phone") and m("cell phone", "my phone") and m("black mobile", "phone")
    assert m("iPhone", "phone") and not m("tablet", "my phone")
    assert m("coffee mug", "cup") and m("teacup", "cup") and m("paper cups", "cup") and not m("glass", "cup")
    assert m("bag of potato chips", "chip") and m("crisps packet", "chip") and not m("snack bag", "chip")
    assert m("sugar jar", "sugar") and not m("salt shaker", "sugar")
    assert m("electric fan", "fan") and m("desk fans", "fan") and not m("fanny pack", "fan")
    assert m("phone holder", "holder") and not m("phone stand", "holder")
    assert m("AA batteries", "battery") and not m("power bank", "battery")
    assert not m(None, "fan") and not m("fan", None)
    # multi-word references need consecutive words
    assert m("blue water bottle", "water bottle") and not m("bottle of water", "water bottle")


# ----------------------------------------------------------------------------- answers and pool

def test_parse_answer():
    good = '{"name": "red cup", "category": "kitchenware", "attributes": [], "visible_text": "", "kind": "one item"}'
    assert vc.parse_answer(good)["name"] == "red cup"
    assert vc.parse_answer("```json\n" + good + "\n```")["name"] == "red cup"
    assert vc.parse_answer("Sure! " + good + " Hope this helps.")["name"] == "red cup"
    assert vc.parse_answer('{"name": "x", "category": "kitchenware"') is None       # truncated
    assert vc.parse_answer('{"category": "toy"}') is None                            # no name
    assert vc.parse_answer("") is None and vc.parse_answer(None) is None
    assert vc.parse_answer("[1, 2]") is None
    assert vc.answer_name({"name": "  water   bottle "}) == "water bottle"
    assert vc.answer_name({"name": ""}) is None and vc.answer_name({"name": 3}) is None and vc.answer_name(None) is None


def test_build_pool_dedup_order_and_blindness():
    names = ["Water bottle", "water bottles", "cup", None, "", "CUP", "a cup", "bottle"]
    p = vc.build_pool("inv:x", names)
    keys = [k for k, _ in p]
    assert sorted(keys) == ["bottle", "cup", "water bottle"]                 # deduplicated after normalising
    assert dict(p)["water bottle"] == "water bottle"                         # first in sort order of spellings
    assert dict(p)["cup"] == "a cup"                                          # 'a cup' < 'cup' as strings
    # the order does not depend on the order of the input (so not on which method came first)
    shuffled = list(names)
    random.Random(3).shuffle(shuffled)
    assert vc.build_pool("inv:x", shuffled) == p
    # it is a per-item shuffle, reproducible, and differs between items for some item ids
    orders = {tuple(k for k, _ in vc.build_pool("inv:%d" % i, ["a1 x", "b2 y", "c3 z", "d4 w", "e5 v"])) for i in range(20)}
    assert len(orders) > 1
    assert vc.build_pool("inv:x", names) == vc.build_pool("inv:x", names)


def test_parse_verdicts_and_schema():
    s = vc.judge_schema(3)
    assert s["required"] == ["n1", "n2", "n3"] and s["properties"]["n2"]["enum"] == vc.VERDICTS
    v = vc.parse_verdicts('{"n1": "right", "n2": "Close", "n3": "maybe"}', 3)
    assert v == {0: "right", 1: "close"}                                       # invalid verdict left out
    assert vc.parse_verdicts("nonsense", 2) == {}
    t = vc.judge_text(["mug", "cup"], with_frame=True)
    assert "n1: mug\nn2: cup" in t and vc.JUDGE_VIEW_CTX in t and vc.JUDGE_VIEW_CROP not in t


def test_chat_body():
    b = vc.chat_body("m", "hello", ["AAA", "BBB"], vc.NAME_SCHEMA, 200)
    c = b["messages"][0]["content"]
    assert [x["type"] for x in c] == ["image_url", "image_url", "text"]          # images first
    assert c[1]["image_url"]["url"] == "data:image/jpeg;base64,BBB"
    assert b["temperature"] == 0.0 and b["seed"] == 0 and b["stream"] is True
    assert b["chat_template_kwargs"] == {"enable_thinking": False}
    assert b["response_format"]["json_schema"]["schema"]["required"] == ["name", "category", "attributes",
                                                                          "visible_text", "kind"]
    assert "stream" not in vc.chat_body("m", "x", [], None, 5, stream=False)
    # the prompts quoted in the design note
    assert vc.NAME_PROMPT_CROP.startswith("This photo comes from a phone app")
    assert "Name the object in the red box the way its owner" in vc.NAME_PROMPT_CTX
    assert "kind (one of: one item, part of an item, several items, furniture or fixture, background or unclear)." \
        in vc.NAME_PROMPT_CROP


def test_prometheus():
    txt = "\n".join([
        "# HELP vllm:e2e_request_latency_seconds x",
        'vllm:e2e_request_latency_seconds_bucket{le="0.5",model_name="m"} 3.0',
        'vllm:e2e_request_latency_seconds_sum{model_name="m"} 1.25',
        'vllm:e2e_request_latency_seconds_count{model_name="m"} 4.0',
        'vllm:request_prefill_time_seconds_sum{engine="0",model_name="m"} 0.5',
        'vllm:request_prefill_time_seconds_sum{engine="1",model_name="m"} 0.25',
        "process_cpu_seconds_total 12.0",
        "vllm:num_requests_running{model_name=\"m\"} 1.0"])
    p = vc.parse_prometheus(txt)
    assert p["vllm:e2e_request_latency_seconds_sum"] == 1.25
    assert p["vllm:request_prefill_time_seconds_sum"] == 0.75                 # two engines summed
    assert "vllm:e2e_request_latency_seconds_bucket" not in p and "process_cpu_seconds_total" not in p
    d = vc.metric_deltas(p, dict(p, **{"vllm:e2e_request_latency_seconds_sum": 2.0}))
    assert close(d["vllm:e2e_request_latency_seconds_sum"], 0.75) and "vllm:num_requests_running" not in d


# ----------------------------------------------------------------------------- statistics

def test_wilson():
    lo, hi = vc.wilson(5, 10)          # textbook: 0.2366, 0.7634
    assert close(lo, 0.236593, 1e-5) and close(hi, 0.763407, 1e-5)
    lo, hi = vc.wilson(0, 8)           # 0 and 0.3244
    assert lo == 0.0 and close(hi, 0.324405, 1e-5)
    lo, hi = vc.wilson(8, 8)
    assert hi == 1.0 and close(lo, 0.675595, 1e-5)
    assert all(math.isnan(x) for x in vc.wilson(0, 0))


def test_mcnemar_exact():
    assert vc.mcnemar_exact(0, 0) == 1.0
    assert close(vc.mcnemar_exact(0, 5), 2 / 32)          # 2 * 0.5^5
    assert close(vc.mcnemar_exact(1, 4), 2 * 6 / 32)      # 2 * (1 + 5) / 32
    assert vc.mcnemar_exact(3, 3) == 1.0
    assert close(vc.mcnemar_exact(2, 9), vc.mcnemar_exact(9, 2))


def test_cohen_kappa():
    a = ["y"] * 20 + ["n"] * 20
    assert close(vc.cohen_kappa(a, a), 1.0)
    # 2x2 table [[20, 5], [10, 15]]: po = 0.7, pe = 0.5*0.6 + 0.5*0.4 = 0.5, kappa = 0.4
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert close(vc.cohen_kappa(a, b), 0.4)
    assert math.isnan(vc.cohen_kappa(["y", "y"], ["y", "y"]))
    assert math.isnan(vc.cohen_kappa([], []))


def test_quantile_matches_numpy_linear():
    xs = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6]
    s = sorted(xs)
    for q in (0.0, 0.1, 0.5, 0.9, 1.0):
        pos = (len(s) - 1) * q
        lo = int(pos)
        ref = s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (pos - lo)
        assert close(vc.quantile(xs, q), ref)
    assert vc.median([1, 3]) == 2 and math.isnan(vc.quantile([], 0.5))


def test_energy_trapezoid():
    ts = [0.0, 1.0, 2.0, 3.0]
    assert close(vc.energy_trapezoid(ts, [100, 100, 100, 100]), 300.0)
    assert close(vc.energy_trapezoid(ts, [0, 100, 200, 300]), 450.0)               # ramp: 300 * 3 / 2
    # sub-window with interpolated ends: power 100 t on [0.5, 2.5] -> 50 * (2.5^2 - 0.5^2) = 300
    assert close(vc.energy_trapezoid(ts, [0, 100, 200, 300], 0.5, 2.5), 300.0)
    assert vc.energy_trapezoid(ts, [1, 1, 1, 1], 2, 2) == 0.0
    assert math.isnan(vc.energy_trapezoid([0.0], [5.0]))


def test_share():
    assert vc.share([True, False, None, True]) == (2, 3)


# ----------------------------------------------------------------------------- report

def _status(d, state="done", t=100.0, gpu=None, **kw):
    st = dict({"state": state, "t": t, "notes": ["started", "finished"]}, **kw)
    if gpu:
        st["gpus"] = [0]
        st["gpus_before"] = [{"index": 0, "name": gpu, "mem_total_mib": 143771}]
    json.dump(st, open(os.path.join(d, "status.json"), "w"))


def _fake_out(judge_pool=True):
    """A tiny out folder: 3 scan entries (2 with a frame), 2 records (1 typed), one model (its primary run in
    the workstation mirror under its former key), one judge that judged the primary run's pool."""
    d = tempfile.mkdtemp(prefix="vlmtest_")
    os.makedirs(os.path.join(d, "local"))
    os.makedirs(os.path.join(d, "package"))
    items = {
        "inv:a": {"set": "S", "frame": {"kf": "s/0001"}, "phone_label": "cup"},
        "inv:b": {"set": "S", "frame": {"kf": "s/0002"}, "phone_label": "book"},
        "inv:c": {"set": "S", "frame": None, "phone_label": "bowl"},
        "rec:x": {"set": "R", "reference": "fan", "typed": "fan"},
        "rec:y": {"set": "R", "reference": None, "typed": None},
        "det:s:0001:0": {"set": "V", "entry": "a"},
    }
    json.dump({"items": items, "data": []}, open(os.path.join(d, "local", "items_local.json"), "w"))
    man = {"items": [{"id": "inv:a", "set": "S", "variants": {"crop-full": ["a.jpg"], "ctx-full": ["a.jpg", "af.jpg"]}},
                     {"id": "inv:b", "set": "S", "variants": {"crop-full": ["b.jpg"], "ctx-full": ["b.jpg", "bf.jpg"]}},
                     {"id": "inv:c", "set": "S", "variants": {"crop-full": ["c.jpg"]}},
                     {"id": "rec:x", "set": "R", "variants": {"crop-full": ["x.jpg"]}},
                     {"id": "rec:y", "set": "R", "variants": {"crop-full": ["y.jpg"]}}]}
    json.dump(man, open(os.path.join(d, "package", "items.json"), "w"))
    json.dump({"names": {"phone": {"inv:a": "cup", "inv:b": "book", "inv:c": "bowl", "rec:x": "chair", "rec:y": None},
                         "clip": {"inv:a": "mug", "inv:b": "paper", "inv:c": "bowl", "rec:x": "electric fan",
                                  "rec:y": "box"}}, "cost": {}, "notes": []},
              open(os.path.join(d, "baselines.json"), "w"))
    run = os.path.join(d, "ws_runs", "2026-11-17_oldkey_a")
    os.makedirs(run)
    _status(run, t=100.0)
    names = {"inv:a": "paper towel roll", "inv:b": "book", "inv:c": "Bowls", "rec:x": "desk fan", "rec:y": None}
    with open(os.path.join(run, "requests.jsonl"), "w") as f:
        for rep in (1, 2, 3):
            for i, n in names.items():
                f.write(json.dumps({"phase": "seq", "item": i, "variant": "crop-full", "repeat": rep,
                                    "name": n if not (rep == 3 and i == "inv:b") else "notebook",
                                    "latency_s": 1.0, "energy_j": 50.0}) + "\n")
    jrun = os.path.join(d, "cluster_runs", "2026-11-18_judge_a")
    os.makedirs(jrun)
    _status(jrun, t=300.0, gpu="NVIDIA H200 NVL")
    verdicts = {"inv:a": {"paper towel roll": "right", "cup": "wrong", "mug": "wrong"},
                "inv:b": {"book": "close", "paper": "wrong", "notebook": "right"},
                "inv:c": {"bowl": "right"},
                "rec:x": {"desk fan": "right", "chair": "wrong", "electric fan": "right", "fan": "right"},
                "rec:y": {"box": "unclear"}}
    with open(os.path.join(jrun, "judgments.jsonl"), "w") as f:
        for i, v in verdicts.items():
            f.write(json.dumps({"item": i, "order": "A", "verdicts": v}) + "\n")
            f.write(json.dumps({"item": i, "order": "B", "verdicts": dict(v, **({"book": "right"} if i == "inv:b" else {}))}) + "\n")
    if judge_pool:
        import vlm_report as vr
        json.dump({"version": 2, "items": vr.Results(d).expected_pool}, open(os.path.join(jrun, "judge_pool.json"), "w"))
    return d


def test_report_counts_by_hand():
    import vlm_report as vr
    d = _fake_out()
    try:
        R = vr.Results(d)
        assert R.models == ["model_a"] and R.judges == ["judge_a"]
        assert R.runs["model_a"]["id"] == "workstation/2026-11-17_model_a"     # former key mapped
        assert R.ids("S") == ["inv:a", "inv:b", "inv:c"] and R.ids("Sf") == ["inv:a", "inv:b"]
        assert R.ids("Rh") == ["rec:x"] and R.ids("RS") == ["inv:a", "inv:b", "inv:c", "rec:x", "rec:y"]
        nm = R.names("model_a", "crop-full")
        # S: a right, b close, c right ('Bowls' normalises to 'bowl') -> strict 2, lenient 3
        assert vr.judged(R, "judge_a", nm, R.ids("S"))[:2] == (2, 3)
        # phone on S: cup wrong, book close, bowl right -> strict 1, lenient 2
        assert vr.judged(R, "judge_a", R.names("phone", "crop-full"), R.ids("S"))[:2] == (1, 2)
        # R: x right, y no name -> strict 1 of 2
        r, l_, u, miss, n = vr.judged(R, "judge_a", nm, R.ids("R"))
        assert (r, l_, n) == (1, 1, 2)
        # clip on R: x right, y unclear (counted, not right)
        assert vr.judged(R, "judge_a", R.names("clip", "crop-full"), R.ids("R"))[:3] == (1, 1, 1)
        # McNemar vs phone on S, strict: model right on a and c, phone right on c -> 1 better, 0 worse
        p, b, c = vr.mcnemar_vs(vr.correct_vector(R, "judge_a", nm, R.ids("S")),
                                vr.correct_vector(R, "judge_a", R.names("phone", "crop-full"), R.ids("S")))
        assert (b, c) == (1, 0) and p == 1.0
        # baselines only on their own input
        assert R.names("phone", "crop-320") == {} and R.names("yoloe11s", "crop-full") == {}
        # the human match: desk fan matches 'fan', phone's chair does not
        assert vc.human_match(nm["rec:x"], R.items["rec:x"]["reference"])
        assert not vc.human_match(R.names("phone", "crop-full")["rec:x"], "fan")
        # order B differs on one pair
        A, B = R.verdicts("judge_a", "A"), R.verdicts("judge_a", "B")
        assert A["inv:b"]["book"] == "close" and B["inv:b"]["book"] == "right"
    finally:
        shutil.rmtree(d)


def test_pool_command_is_blind():
    import vlm_report as vr
    d = _fake_out()
    try:
        class A:
            out = d
        vr.cmd_pool(A)
        pool = json.load(open(os.path.join(d, "package", "judge_pool.json")))
        by = {p["item"]: p for p in pool["items"]}
        # rec:x pools the model's, phone's, clip's and the typed name, nothing says which is which
        assert sorted(by["rec:x"]["names"]) == ["chair", "desk fan", "electric fan", "fan"]
        assert set(by["rec:x"]) == {"item", "with_frame", "images", "names"}
        # inv:b: book (repeats 1-2), notebook (repeat 3), phone book (dedup), clip paper
        assert sorted(by["inv:b"]["names"]) == ["book", "notebook", "paper"]
        assert by["inv:a"]["with_frame"] and by["inv:a"]["images"] == ["a.jpg", "af.jpg"]
        assert not by["inv:c"]["with_frame"] and by["inv:c"]["images"] == ["c.jpg"]
        assert "typed" not in json.dumps(pool) and "phone" not in json.dumps(pool["items"])
    finally:
        shutil.rmtree(d)


def test_parse_log():
    import vlm_report as vr
    t = ("INFO Loading weights took 3.21 seconds\nINFO Model loading took 8.79 GiB memory and 4.5 seconds\n"
         "INFO GPU KV cache size: 123,456 tokens\nINFO init engine (profile, create kv cache, warmup model) took 61.2 seconds")
    p = vr.parse_log(t)
    assert p == {"weights_gib": 8.79, "load_s": 4.5, "weights_s": 3.21, "kv_tokens": 123456, "init_s": 61.2}


def test_mean_power_sums_gpus():
    import vlm_report as vr
    rows = [(0.0, 0, 100.0), (1.0, 0, 200.0), (0.0, 1, 50.0), (1.0, 1, 50.0), (5.0, 0, 999.0)]
    assert close(vr.mean_power(rows, 0.0, 1.0), 150.0 + 50.0)


# ----------------------------------------------------------------------------- Track B, packaging, pulling

def test_trackb_names():
    if not HAVE_NP:
        return
    import vlm_baselines as vb
    local = {"items": {
        "inv:a": {"set": "S", "sightings": [{"kf": "s/0001", "box": [0, 0, 100, 100]},
                                            {"kf": "s/0002", "box": [0, 0, 100, 100]}]},
        "inv:b": {"set": "S", "sightings": [{"kf": "s/0001", "box": [500, 500, 600, 600]}]},
        "inv:c": {"set": "S", "sightings": []},
        "rec:x": {"set": "R"}}}
    dets = {"yoloe11s": {
        "s/0001": [{"box": [0, 0, 100, 90], "label": "cup", "score": 0.4, "reason": None},     # IoU 0.9
                   {"box": [0, 0, 100, 100], "label": "mug", "score": 0.9, "reason": "stop_list"},  # not kept
                   {"box": [520, 520, 600, 600], "label": "book", "score": 0.3, "reason": None}],   # IoU 0.64
        "s/0002": [{"box": [0, 0, 100, 100], "label": "bowl", "score": 0.3, "reason": None},
                   {"box": [0, 0, 50, 50], "label": "plate", "score": 0.99, "reason": None}]}}     # IoU 0.25
    out = vb.trackb_names(local, dets, "yoloe11s")
    assert out == {"inv:a": "cup", "inv:b": "book"}   # cup 0.4 > bowl 0.3; c (no saved sighting) is left out


def test_prepare_geometry():
    if not HAVE_NP:
        return
    import vlm_prepare as vp
    import bench_detect as bd_
    assert vp.resized_size(640, 480, 320) == (320, 240) and vp.resized_size(200, 100, 320) == (200, 100)
    assert vp.resized_size(1202, 1049, 960) == (960, 838)
    for rot in (0, 90, 180, 270):
        b = (100.0, 200.0, 400.0, 260.0)
        assert vp.sensor_box_to_upright(b, 0.5, rot, 1920, 1080) == bd_.sensor_box_to_upright(b, 0.5, rot, 1920, 1080)
    # the drawn box: a red outline lands where the box is, on a grey frame resized from 960 to 320
    import numpy as np
    frame = np.full((960, 540, 3), 128, np.uint8)
    data, wh = vp.marked_frame(frame, (90, 300, 270, 600), 320)
    from PIL import Image
    import io
    im = np.asarray(Image.open(io.BytesIO(data)).convert("RGB")).astype(int)
    assert wh == (180, 320)
    assert im[150, 30, 0] > 200 and im[150, 30, 1] < 80          # left edge at x = 90/3 = 30, y = 450/3 = 150
    assert abs(im[150, 60, 0] - 128) < 20                          # inside stays grey


def test_queue_status_and_yield_check():
    d = tempfile.mkdtemp(prefix="vlmtest_")
    old = os.environ.get("FBV")
    try:
        os.environ["FBV"] = d
        sys.modules.pop("vlm_remote", None)
        import vlm_remote as vr_
        os.makedirs(vr_.RUNS_DIR)
        st = vr_.QueueStatus()
        st.finished.append("model_a (18:20)")
        st.set("waiting for 1 free GPU(s) for model_b", "0 of 2 GPUs free")
        lines = open(os.path.join(d, "runs", "STATUS")).read().splitlines()
        assert len(lines) == 6 and lines[0].startswith("state: waiting") and "model_a (18:20)" in lines[3]
        assert "24 h limit" in lines[2] and "queue-stop" in lines[5]
        assert lines[3] == "sessions finished: model_a (18:20)"           # an unrestricted queue says nothing more
        # check_yield: silent without a watchdog or a finding, raises once a foreign process was seen
        vr_.WATCH = None
        vr_.check_yield()
        w = vr_.Watchdog()
        vr_.WATCH = w
        vr_.check_yield()
        w.found = [{"gpu": 0, "pid": 1}]
        try:
            vr_.check_yield()
            raise AssertionError("no Yield")
        except vr_.Yield:
            pass
        vr_.WATCH = None
        # latest finished run: attempts and other suffixes are not taken
        # the first run under the former key, later ones under the alias
        for name, state in (("2026-11-17_oldkey_a", "done"), ("2026-11-17_oldkey_a_contaminated", "done"),
                            ("2026-11-18_model_a", "yielded"), ("2026-11-17_oldkey_a_smoke", "done")):
            os.makedirs(os.path.join(vr_.RUNS_DIR, name))
            json.dump({"state": state}, open(os.path.join(vr_.RUNS_DIR, name, "status.json"), "w"))
        assert os.path.basename(vr_.latest_done("model_a")) == "2026-11-17_oldkey_a"
        assert os.path.basename(vr_.latest_done("oldkey_a")) == "2026-11-17_oldkey_a"
        assert vr_.latest_done("model_c") is None
    finally:
        if old is None:
            os.environ.pop("FBV", None)
        else:
            os.environ["FBV"] = old
        sys.modules.pop("vlm_remote", None)
        shutil.rmtree(d)


def test_queue_model_filter_and_until():
    import vlm_remote as vr_
    import vlm_ws as vw
    # without --models: every contestant in the queue's order, then both judges (unchanged)
    assert vr_.queue_order() == vr_.queue_order(None) == vr_.CONTESTANTS + vr_.JUDGES
    assert vr_.queue_order(None)[-2:] == ["judge_a", "judge_b"]
    # with --models: those contestants only, in the queue's order (not the order given), and no judge
    assert vr_.queue_order(["model_c", "model_b", "model_d"]) == ["model_b", "model_d", "model_c"]
    assert vr_.queue_order(["model_b", "model_b"]) == ["model_b"]
    for bad in (["judge_a"], ["model_b", "judge_b"], ["unknown_key"]):
        try:
            vr_.queue_order(bad)
            raise AssertionError("no SystemExit for %s" % bad)
        except SystemExit as e:
            assert "contestants only" in str(e)
    # the command line refuses a judge or a former key before anything starts (argparse choices)
    for argv in (["queue", "--models", "judge_a"], ["queue", "--models", "oldkey_a"]):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                vr_.main(argv)
                raise AssertionError("no SystemExit for %s" % argv)
            except SystemExit as e:
                assert e.code == 2
    # --until: a local time in the future; anything else stops before the queue starts
    soon = (datetime.datetime.now() + datetime.timedelta(days=2)).replace(second=0, microsecond=0)
    assert vr_.parse_until(soon.isoformat(timespec="minutes")) == soon.timestamp()
    for bad in ("2020-01-01T00:00", "next week"):
        try:
            vr_.parse_until(bad)
            raise AssertionError("no SystemExit for %r" % bad)
        except SystemExit:
            pass
    # the Mac side passes the aliases and the time through, nothing else
    ns = argparse.Namespace(shared=False, models=["model_b", "model_c", "model_d"], until="2026-11-23T23:59")
    assert vw.queue_flags(ns) == "--models model_b model_c model_d --until 2026-11-23T23:59"
    assert vw.queue_flags(argparse.Namespace(shared=False, models=None, until=None)) == ""
    assert vw.queue_flags(argparse.Namespace(shared=True, models=None, until=None)) == "--shared"
    try:
        vw.queue_flags(argparse.Namespace(shared=False, models=None, until="23 Nov"))
        raise AssertionError("no SystemExit")
    except SystemExit:
        pass


def test_restricted_queue_runs_no_judge():
    """queue --models model_c model_b model_d with every GPU free and every weight present: the three run in
    the queue's order as sessions, no judge starts, no judging pool is written, the STATUS says so; a
    second start skips them all, and skips Model A by its finished run under the former key."""
    d = tempfile.mkdtemp(prefix="vlmtest_")
    old = os.environ.get("FBV")
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        os.environ["FBV"] = d
        sys.modules.pop("vlm_remote", None)
        import vlm_remote as vr_
        os.makedirs(os.path.join(d, "data"))
        os.makedirs(vr_.RUNS_DIR)
        first = os.path.join(vr_.RUNS_DIR, "2026-11-17_oldkey_a")
        os.makedirs(first)
        json.dump({"state": "done"}, open(os.path.join(first, "status.json"), "w"))
        started = []

        class FakeChild:
            """Stands in for the session process: records its command line and finishes at once."""

            def __init__(self, cmd, **kw):
                started.append(cmd)
                json.dump({"state": "done"}, open(os.path.join(cmd[cmd.index("--run") + 1], "status.json"), "w"))
                self.pid = -1

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

        vr_.subprocess = types.SimpleNamespace(Popen=FakeChild, STDOUT=subprocess.STDOUT)
        vr_.gpu_occupancy = lambda: [(0, []), (1, [])]          # both GPUs without any process
        vr_.snapshot_path = lambda key: d                       # weights present: nothing is fetched
        vr_.POLL_S = 0
        until = (datetime.datetime.now() + datetime.timedelta(days=5)).replace(second=0, microsecond=0)
        args = argparse.Namespace(shared=False, models=["model_c", "model_b", "model_d"],
                                  until=until.isoformat(timespec="minutes"))
        assert vr_.cmd_queue(args) == 0
        assert [c[2:4] for c in started] == [["session", "model_b"], ["session", "model_d"], ["session", "model_c"]]
        assert not any("judge" in " ".join(c) for c in started)
        assert not os.path.exists(os.path.join(d, "data", "judge_pool.json"))
        today = datetime.date.today().isoformat()
        for m in ("model_b", "model_c", "model_d"):                # run folders carry the alias
            assert vr_.run_state(os.path.join(vr_.RUNS_DIR, "%s_%s" % (today, m))) == "done"
        lines = open(os.path.join(vr_.RUNS_DIR, "STATUS")).read().splitlines()
        assert len(lines) == 6 and lines[0] == "state: done: every session finished"
        assert "gives up at %s (limit given at start)" % until.strftime("%m-%d %H:%M:%S") in lines[2]
        assert lines[3].startswith("sessions finished (this queue runs only model_b, model_d, model_c; no judges): "
                                   "model_b (")
        # started again with Model A in the list: nothing runs (Model A's run has the former key)
        del started[:]
        args = argparse.Namespace(shared=False, models=["model_a", "model_b", "model_c", "model_d"], until=None)
        assert vr_.cmd_queue(args) == 0 and started == []
        assert "24 h limit" in open(os.path.join(vr_.RUNS_DIR, "STATUS")).read()
    finally:
        for s, h in handlers.items():
            signal.signal(s, h)
        if old is None:
            os.environ.pop("FBV", None)
        else:
            os.environ["FBV"] = old
        sys.modules.pop("vlm_remote", None)
        shutil.rmtree(d)


def test_shared_mode_stop_signs():
    import vlm_remote as vr_
    base = {(0, 100): 6780, (1, 100): 8782}
    free = {0: 6000, 1: 15000}
    assert vr_.shared_signs(base, dict(base), free, 0, 0) == []                         # idle, enough memory
    assert vr_.shared_signs(base, {(0, 100): 6780}, free, 0, 0) == []                   # their process left a GPU
    why = vr_.shared_signs(base, {(0, 100): 7000, (1, 100): 8782}, free, 0, 0)
    assert len(why) == 1 and "6780 -> 7000" in why[0]                                   # their memory changed
    why = vr_.shared_signs(base, {**base, (0, 200): 500}, free, 0, 0)
    assert len(why) == 1 and "new process" in why[0] and "pid 200" in why[0]
    assert "utilisation 3%" in vr_.shared_signs(base, dict(base), free, 0, 3)[0]
    why = vr_.shared_signs(base, dict(base), {0: vr_.SHARED_FLOOR_STOP_MIB - 1}, 0, 0)
    assert len(why) == 1 and why[0].startswith("unreserved memory")                     # our own overshoot
    # the cap leaves the target free with the planned peak, and the stop floor sits below the target
    assert vr_.SHARED_FLOOR_STOP_MIB < vr_.SHARED_FLOOR_TARGET_MIB
    assert all(p["est_peak_mib"] > 0 and p["kv_bytes"] > 0 for p in vr_.SHARED.values())


def test_judge_pool_images():
    man = {"items": [{"id": "inv:b", "set": "S", "variants": {"crop-full": ["b.jpg"], "ctx-full": ["b.jpg", "bf.jpg"]}},
                     {"id": "inv:a", "set": "S", "variants": {"crop-full": ["a.jpg"]}},
                     {"id": "det:x", "set": "V", "variants": {"crop-full": ["x.jpg"]}}]}
    p = vc.judge_pool(man, {"inv:a": ["cup", "Cups", "mug"], "inv:b": []})
    assert [x["item"] for x in p] == ["inv:a", "inv:b"]                       # sorted, V left out
    assert sorted(p[0]["names"]) == ["cup", "mug"] and p[0]["images"] == ["a.jpg"] and not p[0]["with_frame"]
    assert p[1]["names"] == [] and p[1]["images"] == ["b.jpg", "bf.jpg"] and p[1]["with_frame"]


def test_sanitize_tree():
    import vlm_ws as vw
    d = tempfile.mkdtemp(prefix="vlmtest_")
    try:
        with open(os.path.join(d, "server.log"), "w") as f:
            f.write("loading /data/someone/fb_vlm/hf/x on host-a by someone\n")
        with open(os.path.join(d, "image.jpg"), "w") as f:
            f.write("/data/someone/fb_vlm")
        vw.sanitize_tree(d, [("/data/someone/fb_vlm", "<WS_DIR>"), ("someone", "<WS>"), ("host-a", "<WS>")])
        assert open(os.path.join(d, "server.log")).read() == "loading <WS_DIR>/hf/x on <WS> by <WS>\n"
        assert open(os.path.join(d, "image.jpg")).read() == "/data/someone/fb_vlm"     # only text files
    finally:
        shutil.rmtree(d)


# ----------------------------------------------------------------------------- aliases, sites, primary runs

def test_aliases_and_private_file():
    assert vc.canonical_key("oldkey_a") == "model_a" and vc.canonical_key("model_c") == "model_c"
    assert vc.canonical_key("unknown_key") == "unknown_key"
    assert vc.run_keys("model_a") == ["model_a", "oldkey_a"] and vc.run_keys("judge_b") == ["judge_b"]
    assert set(vc.PUBLIC_MODELS) == set(vc.CONTESTANTS) | set(vc.JUDGES) and vc.PRIMARY_JUDGE in vc.JUDGES
    assert vc.neutral_name("2026-11-17_oldkey_a_roundtrip") == "2026-11-17_model_a_roundtrip"
    assert vc.alias_label("model_d") == "Model D" and vc.alias_label("judge_a") == "Judge A"
    import vlm_remote as vr_
    assert vr_.model("oldkey_a")["repo"] == "pub/fake-model_a" and vr_.model("judge_b")["gpus"] == 2
    assert "oldkey_a" in vr_.model_choices() and "model_a" in vr_.model_choices()
    old = os.environ["FB_VLM_MODELS"]
    try:
        os.environ["FB_VLM_MODELS"] = os.path.join(tempfile.gettempdir(), "no_such_vlm_models.json")
        assert vc.private_models(required=False) == {} and vc.canonical_key("oldkey_a") == "oldkey_a"
        try:
            vc.private_models()
            raise AssertionError("no SystemExit")
        except SystemExit as e:
            assert "vlm_models.example.json" in str(e)
    finally:
        os.environ["FB_VLM_MODELS"] = old


def _add_run(d, sub, name, t, model, names, gpu=None, state="done", latency=1.0):
    run = os.path.join(d, sub, name)
    os.makedirs(run)
    _status(run, state=state, t=t, gpu=gpu, model=model)
    with open(os.path.join(run, "requests.jsonl"), "w") as f:
        for v in vc.VARIANTS:
            for i, n in names.items():
                f.write(json.dumps({"phase": "seq", "item": i, "variant": v, "repeat": 1, "name": n,
                                    "latency_s": latency, "energy_j": 10.0 * latency}) + "\n")
    return run


def test_primary_and_additional_runs_by_site():
    import vlm_report as vr
    d = _fake_out()
    try:
        # model_a: the workstation run (t = 100, former key) is primary; a later cluster run is additional
        cl = {"inv:a": "kitchen roll", "inv:b": "book", "inv:c": "bowl", "rec:x": "fan", "rec:y": None}
        _add_run(d, "cluster_runs", "2026-11-18_model_a", 200.0, "model_a", cl, gpu="NVIDIA H200 NVL", latency=0.5)
        # model_b: two cluster runs; the first completed (t = 150) is primary even though its folder sorts later
        _add_run(d, "cluster_runs", "2026-11-19_model_b", 150.0, "model_b", {"inv:a": "coffee mug"}, gpu="NVIDIA H200 NVL")
        _add_run(d, "cluster_runs", "2026-11-18_model_b", 250.0, "model_b", {"inv:a": "teacup"}, gpu="NVIDIA H200 NVL")
        # never taken: a set-aside attempt, an unfinished run
        _add_run(d, "cluster_runs", "2026-11-18_model_c_failed_001200", 120.0, "model_c", {"inv:a": "x"})
        _add_run(d, "cluster_runs", "2026-11-18_model_d", 130.0, "model_d", {"inv:a": "x"}, state="error")
        assert [r["id"] for r in vr.runs_of(d, "model_a")] == ["workstation/2026-11-17_model_a",
                                                                  "cluster/2026-11-18_model_a"]
        assert vr.primary_run(d, "model_b")["id"] == "cluster/2026-11-19_model_b"
        assert vr.primary_run(d, "model_c") is None and vr.primary_run(d, "model_d") is None
        R = vr.Results(d)
        assert R.models == ["model_a", "model_b"]
        assert R.runs["model_a"]["site"] == "workstation" and R.runs["model_a"]["hw"] == "RTX 4090, 24 GB"
        x = R.extra_runs["model_a"][0]
        assert x["site"] == "cluster" and x["label"] == "university cluster (H200 NVL, 140 GiB)"
        # quality uses the primary run only
        assert R.names("model_a", "crop-full")["inv:a"] == "paper towel roll"
        assert R.names("model_b", "crop-full") == {"inv:a": "coffee mug"}
        # cost: one row per run, primary first, sites never merged
        assert [(m, r["id"]) for m, r in R.cost_runs()] == [
            ("model_a", "workstation/2026-11-17_model_a"), ("model_a", "cluster/2026-11-18_model_a"),
            ("model_b", "cluster/2026-11-19_model_b"), ("model_b", "cluster/2026-11-18_model_b")]
        # agreement of the additional run with the primary, crop-full repeat 1 only exists in the primary:
        # inv:a paper towel roll / kitchen roll (same head word 'roll', not identical), inv:b book / book,
        # inv:c Bowls / bowl (identical after normalising), rec:x desk fan / fan (head word), rec:y none / none
        assert vr.run_agreement(R, "model_a", x, ["crop-full"]) == (2, 4, 5)
        # the pool comes from the primary runs: the additional runs' 'kitchen roll' and 'teacup' are not in it
        pool = {p["item"]: p["names"] for p in R.expected_pool}
        assert "coffee mug" in pool["inv:a"] and "kitchen roll" not in pool["inv:a"] and "teacup" not in pool["inv:a"]
        # the judge judged the pool of the earlier primary set, so it no longer matches: not used
        assert R.judges == [] and "none judged the pool" in R.judge_notes["judge_a"]
        # cost rows of the plot: one per site, quality from the primary run's names
        cost = {}
        for m, r in R.cost_runs():
            cost[(m, r["id"])] = {"run": r, "crop-full": {"lat": 1.0, "energy": 2.0}}
        assert vr.plot_rows(R, cost) == []                       # no primary judge: nothing plotted
    finally:
        shutil.rmtree(d)


def test_judge_run_used_only_with_the_primary_pool():
    import vlm_report as vr
    d = _fake_out(judge_pool=False)
    try:
        R = vr.Results(d)
        assert R.judges == [] and "judge_a" in R.judge_notes     # no pool recorded: cannot be checked
        jrun = os.path.join(d, "cluster_runs", "2026-11-18_judge_a")
        json.dump({"items": R.expected_pool}, open(os.path.join(jrun, "judge_pool.json"), "w"))
        R = vr.Results(d)
        assert R.judges == ["judge_a"] and R.runs["judge_a"]["site"] == "cluster"
        rows = vr.plot_rows(R, {("model_a", R.runs["model_a"]["id"]): {
            "run": R.runs["model_a"], "crop-full": {"lat": 0.7, "energy": 205.0}}})
        # S-frame = inv:a (paper towel roll: right), inv:b (book: close) -> strict 1 of 2
        assert rows == [("model_a", "crop-full", R.runs["model_a"]["label"],
                         "workstation/2026-11-17_model_a", 1, 2, 0.7, 205.0)]
    finally:
        shutil.rmtree(d)


def test_cluster_pool_equals_mac_pool():
    """The judges' pool built on the cluster (vlm_remote.build_pool_file: names of the workstation run from a
    file, the cluster run from its folder) is the pool the report rebuilds from the primary runs."""
    import vlm_report as vr
    d = _fake_out()
    fbv = tempfile.mkdtemp(prefix="vlmtest_fbv_")
    old = os.environ.get("FBV")
    try:
        _add_run(d, "cluster_runs", "2026-11-18_model_b", 150.0, "model_b",
                 {"inv:a": "Mug", "inv:c": "salad bowl", "rec:x": "fan"}, gpu="NVIDIA H200 NVL")
        R = vr.Results(d)
        os.makedirs(os.path.join(fbv, "data"))
        os.makedirs(os.path.join(fbv, "runs"))
        shutil.copy(os.path.join(d, "package", "items.json"), os.path.join(fbv, "data", "items.json"))
        json.dump({"extra_names": vr.extra_names(R)}, open(os.path.join(fbv, "data", "pool_inputs.json"), "w"))
        names = {}
        for r in R.runs["model_a"]["rows"]:
            if r.get("phase") == "seq" and r.get("name"):
                names.setdefault(r["item"], []).append(r["name"])
        json.dump({"runs": {"workstation/2026-11-17_model_a": {"model": "oldkey_a", "names": names}}},
                  open(os.path.join(fbv, "data", "pool_runs.json"), "w"))
        shutil.copytree(os.path.join(d, "cluster_runs", "2026-11-18_model_b"),
                        os.path.join(fbv, "runs", "2026-11-18_model_b"))
        os.environ["FBV"] = fbv
        sys.modules.pop("vlm_remote", None)
        import vlm_remote as vr_
        rec = vr_.build_pool_file(os.path.join(fbv, "pool.json"), [os.path.join(fbv, "runs", "2026-11-18_model_b")],
                                  os.path.join(fbv, "data", "pool_runs.json"), site="cluster")
        assert rec["items"] == R.expected_pool
        assert rec["runs"] == ["cluster/2026-11-18_model_b", "workstation/2026-11-17_model_a"]
        assert rec["models"]["workstation/2026-11-17_model_a"] == "model_a"
        # a second run of the same model may not enter the pool
        shutil.copytree(os.path.join(fbv, "runs", "2026-11-18_model_b"), os.path.join(fbv, "runs", "2026-11-19_model_b"))
        try:
            vr_.build_pool_file(os.path.join(fbv, "pool2.json"), [os.path.join(fbv, "runs", "2026-11-18_model_b"),
                                                                  os.path.join(fbv, "runs", "2026-11-19_model_b")])
            raise AssertionError("no SystemExit")
        except SystemExit as e:
            assert "only the primary run" in str(e)
        # a run that did not finish may not either; a dropped model is left out and named
        _status(os.path.join(fbv, "runs", "2026-11-19_model_b"), state="error")
        try:
            vr_.build_pool_file(os.path.join(fbv, "pool3.json"), [os.path.join(fbv, "runs", "2026-11-19_model_b")])
            raise AssertionError("no SystemExit")
        except SystemExit as e:
            assert "not a finished run" in str(e)
        open(os.path.join(fbv, "runs", "2026-11-19_model_c.dropped"), "w").write("x")
        rec = vr_.build_pool_file(os.path.join(fbv, "pool4.json"), [os.path.join(fbv, "runs", "2026-11-19_model_c")])
        assert rec["dropped"] == ["2026-11-19_model_c"] and rec["runs"] == []
    finally:
        if old is None:
            os.environ.pop("FBV", None)
        else:
            os.environ["FBV"] = old
        sys.modules.pop("vlm_remote", None)
        shutil.rmtree(d)
        shutil.rmtree(fbv)


def test_cluster_slurm_names_and_sanitize():
    import vlm_cluster as vcl
    old = {k: os.environ.get(k) for k in ("FB_SLURM_OPTS", "FB_SLURM_CPU_OPTS")}
    d = tempfile.mkdtemp(prefix="vlmtest_")
    try:
        os.environ["FB_SLURM_OPTS"] = "-A acct-x -q normal -p part-gpu --gres=gpu:typ:1"
        os.environ["FB_SLURM_CPU_OPTS"] = "--account=acct-x --partition=part-cpu,dev"
        assert vcl.slurm_names() == {"acct-x", "part-gpu", "part-cpu"}          # generic words are left alone
        with open(os.path.join(d, "server.log"), "w") as f:
            f.write("/work/u123/fb_vlm/models on node-7 by u123 (acct-x); u1234 stays; nvidia_typ_nvl; "
                    "login.example.edu, login-2\n")
        with open(os.path.join(d, "power.png"), "w") as f:
            f.write("u123")
        pats = vcl.replacements("/work/u123/fb_vlm", "/home/u123", "u123", {"login-2"}, {"node-7"},
                                "login.example.edu", vcl.slurm_names())
        vcl.sanitize(d, pats)
        assert open(os.path.join(d, "server.log")).read() == (
            "<CL_DIR>/models on <CL_NODE> by <CL_USER> (<CL_SLURM>); u1234 stays; nvidia_typ_nvl; <CL>, <CL_NODE>\n")
        assert open(os.path.join(d, "power.png")).read() == "u123"                  # only text files
        os.environ["FB_SLURM_OPTS"] = "-p x; rm -rf ~"
        try:
            vcl.safe_opts("FB_SLURM_OPTS")
            raise AssertionError("no SystemExit")
        except SystemExit:
            pass
        assert "unset PYTHONHOME PYTHONPATH" in vcl.env_sh("true") and 'HF_HOME="$D/models"' in vcl.env_sh("")
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(d)


def test_failed_judge_request_is_missing_data():
    import vlm_report as vr
    d = _fake_out()
    try:
        jrun = os.path.join(d, "cluster_runs", "2026-11-18_judge_a")
        rows = [json.loads(l) for l in open(os.path.join(jrun, "judgments.jsonl"))]
        with open(os.path.join(jrun, "judgments.jsonl"), "w") as f:
            for r in rows:
                if r["item"] == "inv:b":                       # its request was rejected: no verdict at all
                    r = dict(r, verdicts={}, error="400 Failed to apply processor")
                f.write(json.dumps(r) + "\n")
        R = vr.Results(d)
        nm = R.names("model_a", "crop-full")
        # S without inv:b: a right, c right -> 2 of 2 (not 2 of 3, and inv:b is not counted as wrong)
        assert vr.judged(R, "judge_a", nm, R.ids("S"))[::4] == (2, 2)
        assert vr.judged_ids(R, "judge_a", R.ids("Sf")) == ["inv:a"]
        assert len(vr.correct_vector(R, "judge_a", nm, R.ids("S"))) == 2
        # no judge covers S-frame -> no plot; a second judge that covers it is used instead
        assert vr.plot_judge(R) is None
        j2 = os.path.join(d, "cluster_runs", "2026-11-18_judge_b")
        shutil.copytree(jrun, j2)
        with open(os.path.join(j2, "judgments.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        R = vr.Results(d)
        assert R.judges == ["judge_a", "judge_b"] and vr.plot_judge(R) == "judge_b"
    finally:
        shutil.rmtree(d)


def test_environment_fault_is_not_a_model_failure():
    import vlm_remote as vr_
    d = tempfile.mkdtemp(prefix="vlmtest_")
    try:
        p = os.path.join(d, "server.log")
        open(p, "w").write("ERROR ... RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist\n")
        assert vr_.environment_fault(p) == "Could not find nvcc"
        open(p, "w").write("RuntimeError: Check failed: Failed to load dynamic shared library x.so: "
                           "version `GLIBCXX_3.4.32' not found\n")
        assert vr_.environment_fault(p) == "Failed to load dynamic shared library"
        open(p, "w").write("nvrtc: error: failed to open libnvrtc-builtins.so.13.0.\n")
        assert vr_.environment_fault(p) == "libnvrtc"
        open(p, "w").write("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n")
        assert vr_.environment_fault(p) is None                       # the listed settings are for this
        assert vr_.environment_fault(os.path.join(d, "missing.log")) is None
        assert vr_.parse_slurm_time("1:14:05") == 4445 and vr_.parse_slurm_time("2-00:00:10") == 172810
        assert vr_.parse_slurm_time("59:30") == 3570 and vr_.parse_slurm_time("INVALID") is None
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
