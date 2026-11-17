#!/usr/bin/env python3
"""Reads a person's judgments from the Track B judging sheet back into per-detector numbers.

    python3 analysis/bench_judge_read.py analysis/out/judge/judgments_<sheet>.json \
        [--key analysis/out/judge/judge_key.json] [--md analysis/notes/<date>_track_b_judged.md]

Standard library only. Definitions (analysis/notes/2026-11-17_baselines_design.md):

  complete frame  every object has Q1 answered and, when Q1 is "item", every proposed name judged.
                  Only complete frames are used.
  item precision  a detector's kept boxes in objects judged "item", over its kept boxes in objects
                  with a Q1 other than "can't tell".
  naming precision  its boxes in item objects whose name is "right" (strict) or "right" or "close"
                  (lenient), over its boxes in item objects.
  pooled relative recall  item objects in which it has a box with a lenient-correct name, over item
                  objects in which any detector has one. Objects are counted per keyframe, so a
                  physical object seen in five keyframes counts five times.
  found         item objects in which it has any box, over all item objects.

Intervals: 95% percentile bootstrap over complete frames (2000 resamples, seed 20261117).
Consecutive keyframes overlap, so these intervals are too narrow.
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
N_BOOT = 2000
SEED = 20261117
Q1_VALUES = ("item", "part", "not_item", "unclear")
NAME_VALUES = ("right", "close", "wrong")


def complete(frame, J):
    for o in frame["objects"]:
        a = J.get(o["id"]) or {}
        if a.get("q1") not in Q1_VALUES:
            return False
        if a["q1"] == "item" and any((a.get("names") or {}).get(n) not in NAME_VALUES for n in o["names"]):
            return False
    return True


def frame_counts(frame, J, detectors):
    """Per-detector counts of one complete frame (all integers, so frames can be summed)."""
    c = {k: {"boxes": 0, "decided": 0, "in_item": 0, "right": 0, "lenient": 0, "found": 0, "found_ok": 0}
         for k in detectors}
    pool = {"items": 0, "items_ok": 0}
    for o in frame["objects"]:
        a = J.get(o["id"]) or {}
        q1 = a.get("q1")
        names = a.get("names") or {}
        is_item = q1 == "item"
        ok_names = {n for n in o["names"] if names.get(n) in ("right", "close")}
        pool["items"] += is_item
        pool["items_ok"] += is_item and any(m["label"] in ok_names for m in o["members"])
        seen, seen_ok = set(), set()
        for m in o["members"]:
            k = m["detector"]
            if k not in c:
                continue
            c[k]["boxes"] += 1
            if q1 != "unclear":
                c[k]["decided"] += 1
            if is_item:
                c[k]["in_item"] += 1
                v = names.get(m["label"])
                c[k]["right"] += v == "right"
                c[k]["lenient"] += v in ("right", "close")
                seen.add(k)
                if v in ("right", "close"):
                    seen_ok.add(k)
        for k in seen:
            c[k]["found"] += 1
        for k in seen_ok:
            c[k]["found_ok"] += 1
    return c, pool


def summarise(per_frame, detectors):
    tot = {k: {f: sum(fc[0][k][f] for fc in per_frame) for f in per_frame[0][0][k]} for k in detectors} if per_frame else {}
    pool = {f: sum(fc[1][f] for fc in per_frame) for f in ("items", "items_ok")} if per_frame else {"items": 0, "items_ok": 0}

    def ratio(a, b):
        return a / b if b else float("nan")

    out = {}
    for k in detectors:
        t = tot[k]
        out[k] = {"boxes": t["boxes"], "item_prec": ratio(t["in_item"], t["decided"]),
                  "name_strict": ratio(t["right"], t["in_item"]), "name_lenient": ratio(t["lenient"], t["in_item"]),
                  "recall": ratio(t["found_ok"], pool["items_ok"]), "found": ratio(t["found"], pool["items"]),
                  "n_item_boxes": t["in_item"]}
    return out, pool


METRIC_KEYS = ("item_prec", "name_strict", "name_lenient", "recall", "found")


def bootstrap(per_frame, detectors, n_boot=N_BOOT, seed=SEED):
    rng = random.Random(seed)
    n = len(per_frame)
    draws = {k: {m: [] for m in METRIC_KEYS} for k in detectors}
    if n < 2:
        return None
    for _ in range(n_boot):
        sample = [per_frame[rng.randrange(n)] for _ in range(n)]
        s, _ = summarise(sample, detectors)
        for k in detectors:
            for m in METRIC_KEYS:
                if s[k][m] == s[k][m]:   # not nan
                    draws[k][m].append(s[k][m])
    out = {}
    for k in detectors:
        out[k] = {}
        for m in METRIC_KEYS:
            v = sorted(draws[k][m])
            if len(v) < 0.9 * n_boot:
                out[k][m] = (float("nan"), float("nan"))
            else:
                out[k][m] = (v[int(0.025 * (len(v) - 1))], v[int(0.975 * (len(v) - 1))])
    return out


def fmt(x, ci=None):
    if x != x:
        return "–"
    s = "%.2f" % x
    if ci and ci[0] == ci[0]:
        s += " [%.2f, %.2f]" % ci
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("judgments", help="JSON file downloaded from the judging sheet")
    ap.add_argument("--key", default=os.path.join(HERE, "out", "judge", "judge_key.json"))
    ap.add_argument("--md", help="also write the Markdown table here")
    args = ap.parse_args(argv)
    with open(args.key, encoding="utf-8") as f:
        key = json.load(f)
    with open(args.judgments, encoding="utf-8") as f:
        jd = json.load(f)
    if jd.get("sheet_id") != key.get("sheet_id"):
        sys.exit("judgments are for sheet %s, the key is for sheet %s" % (jd.get("sheet_id"), key.get("sheet_id")))
    J = jd.get("judgments") or {}
    detectors = key["detectors"]
    done = [fr for fr in key["frames"] if complete(fr, J)]
    per_frame = [frame_counts(fr, J, detectors) for fr in done]
    L = ["# Track B results: judged naming precision and pooled relative recall", ""]
    L.append("From `%s` (sheet `%s`, saved %s). %d of %d frames complete and used. Definitions in "
             "`analysis/bench_judge_read.py`. Intervals: bootstrap over frames, too narrow because "
             "consecutive keyframes overlap." % (os.path.basename(args.judgments), key["sheet_id"], jd.get("saved", "?"),
                                                 len(done), len(key["frames"])))
    L.append("")
    if not per_frame:
        L.append("No complete frame yet.")
    else:
        s, pool = summarise(per_frame, detectors)
        ci = bootstrap(per_frame, detectors) or {k: {m: None for m in METRIC_KEYS} for k in detectors}
        L.append("Item objects: %d; with at least one acceptable name from some detector: %d." % (pool["items"], pool["items_ok"]))
        L.append("")
        L.append("| Detector | Kept boxes judged | Item precision | Naming precision, strict | Naming precision, lenient | Pooled relative recall | Found (any name) |")
        L.append("| --- | --- | --- | --- | --- | --- | --- |")
        labels = {"effdet0": "Ours: EfficientDet-Lite0 (app)", "effdet2": "EfficientDet-Lite2", "yolo26n": "YOLO26n",
                  "yoloe11s": "YOLOE-11S, household vocabulary", "owlv2": "OWLv2 base, household vocabulary"}
        for k in detectors:
            r = s[k]
            L.append("| %s | %d | %s | %s | %s | %s | %s |" % (
                labels.get(k, k), r["boxes"], fmt(r["item_prec"], ci[k]["item_prec"]), fmt(r["name_strict"], ci[k]["name_strict"]),
                fmt(r["name_lenient"], ci[k]["name_lenient"]), fmt(r["recall"], ci[k]["recall"]), fmt(r["found"], ci[k]["found"])))
    md = "\n".join(L) + "\n"
    if args.md:
        with open(args.md, "w") as f:
            f.write(md)
    print(md)


if __name__ == "__main__":
    main()
