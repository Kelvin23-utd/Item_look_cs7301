#!/usr/bin/env python3
"""Baseline names for the VLM naming experiment, on the Mac (no workstation).

    analysis/.venv-bench/bin/python analysis/vlm_baselines.py

Needs analysis/out/vlm/local/items_local.json and the package from vlm_prepare.py. For Track B it
reads analysis/out/bench_detect/detections.json (from bench_detect.py on the same pull); without it
the two open-vocabulary detectors are left out. Writes analysis/out/vlm/baselines.json:
{"names": {method: {item: name or null}}, "cost": {...}, "notes": [...]}.

Methods (analysis/notes/2026-11-17_vlm_naming_design.md):
  phone     scan entries: the label the app stored; records: the app's EfficientDet-Lite0 run on the
            stored crop, highest-scoring non-person detection at the raw threshold 0.20
  clip      MobileCLIP2-S2 zero-shot over the household vocabulary (items and stop names),
            "a photo of a {name}.", top-1 on the stored crop
  yoloe11s, owlv2   Track B detectors: the entry's kept phone detections in saved keyframes matched to
            the detector's kept boxes at IoU >= 0.5, the name with the largest summed score
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".bench-cache")
os.environ.setdefault("HF_HOME", os.path.join(CACHE, "hf"))
os.environ.setdefault("TORCH_HOME", os.path.join(CACHE, "torch"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import bench_common as bc  # noqa: E402

OUT = os.path.join(HERE, "out", "vlm")
TRACKB_IOU = 0.5
TRACKB = ("yoloe11s", "owlv2")


def trackb_names(local, detections, detector):
    """item -> name for scan entries with saved sightings (None when no box matches)."""
    kept = {}
    for kf, dets in (detections.get(detector) or {}).items():
        kept[kf] = [d for d in dets if d.get("reason") is None]
    out = {}
    for iid, info in local["items"].items():
        if info["set"] != "S" or not info.get("sightings"):
            continue
        votes = {}
        for s in info["sightings"]:
            for d in kept.get(s["kf"], []):
                if bc.iou(s["box"], d["box"]) >= TRACKB_IOU:
                    votes[d["label"]] = votes.get(d["label"], 0.0) + float(d["score"])
        out[iid] = max(sorted(votes), key=lambda k: votes[k]) if votes else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--trackb", default=os.path.join(HERE, "out", "bench_detect", "detections.json"))
    args = ap.parse_args(argv)
    with open(os.path.join(args.out, "local", "items_local.json")) as f:
        local = json.load(f)
    with open(os.path.join(args.out, "package", "items.json")) as f:
        man = json.load(f)
    files = {it["id"]: os.path.join(args.out, "package", "images", it["variants"]["crop-full"][0])
             for it in man["items"] if it["set"] in ("R", "S")}
    # the stored crop as stored (crop-full is the original bytes except two records capped at 960)
    names, cost, notes = {}, {}, []

    # ---- phone
    import bench_detect as bd_mod
    det = bd_mod.MediaPipeDetector(os.path.join(bd_mod.APP_MODELS, "efficientdet_lite0.tflite"))
    names["phone"], times = {}, []
    for iid, info in sorted(local["items"].items()):
        if info["set"] == "S":
            names["phone"][iid] = info.get("phone_label")
        elif info["set"] == "R":
            rgb = bc.load_rgb(files[iid])
            t0 = time.perf_counter()
            dets = [d for d in det.detect(rgb) if d["label"] != "person"]
            times.append(time.perf_counter() - t0)
            names["phone"][iid] = max(dets, key=lambda d: d["score"])["label"] if dets else None
    cost["phone_offline_ms_per_crop"] = 1000 * statistics.median(times)
    notes.append("phone on records: EfficientDet-Lite0 found a non-person box in %d of %d crops" % (
        sum(1 for k, v in names["phone"].items() if v and local["items"][k]["set"] == "R"),
        sum(1 for v in local["items"].values() if v["set"] == "R")))

    # ---- clip
    import torch
    import open_clip
    from bench_match import MobileClip2S2
    vocab = bd_mod.read_vocab()
    labels = vocab["items"] + vocab["stop"]
    m = MobileClip2S2()
    tok = open_clip.get_tokenizer("MobileCLIP2-S2")
    with torch.no_grad():
        te = m.model.encode_text(tok(["a photo of a %s." % n for n in labels])).numpy()
    te /= np.linalg.norm(te, axis=1, keepdims=True)
    names["clip"], times = {}, []
    for iid in sorted(files):
        rgb = bc.load_rgb(files[iid])
        t0 = time.perf_counter()
        v = m.features(rgb)
        k = int(np.argmax(te @ v))
        times.append(time.perf_counter() - t0)
        names["clip"][iid] = labels[k]
    cost["clip_mac_ms_per_crop"] = 1000 * statistics.median(times[3:] or times)
    cost["clip_params_image_encoder"] = m.params

    # ---- Track B
    if os.path.exists(args.trackb):
        with open(args.trackb) as f:
            detections = json.load(f)
        for k in TRACKB:
            if k in detections:
                names[k] = trackb_names(local, detections, k)
            else:
                notes.append("%s missing from %s" % (k, os.path.basename(args.trackb)))
    else:
        notes.append("Track B detections not found; yoloe11s and owlv2 left out")

    with open(os.path.join(args.out, "baselines.json"), "w") as f:
        json.dump({"names": names, "cost": cost, "notes": notes}, f, indent=1)
    print("baselines written: %s; %s" % ({k: sum(1 for v in d.values() if v) for k, d in names.items()}, notes))


if __name__ == "__main__":
    main()
