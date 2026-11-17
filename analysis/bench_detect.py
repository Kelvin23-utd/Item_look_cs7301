#!/usr/bin/env python3
"""Track B of analysis/BASELINES_PLAN.md: detection and naming on the saved scan keyframes.

    analysis/.venv-bench/bin/python analysis/bench_detect.py --data backups/phone_files_2026-11-17_0008

Runs every detector on every saved keyframe (the half-size image the phone's detector saw), passes
each detector's output through the app's own keyframe filter (person, stop list, duplicates, depth
range, physical size, at most 10 per frame), and reports label-free numbers only: detections per
frame, distinct names, score quartiles, time per frame on this Mac. For EfficientDet-Lite0 it also
checks agreement with the detections the phone logged for the same keyframes.

It also writes a judging sheet for a person (analysis/out/judge/judge_sheet.html: boxes of all
detectors pooled per keyframe, names shuffled, detectors hidden) and the key that maps the sheet
back to detectors (judge_key.json). bench_judge_read.py turns the exported judgments into naming
precision and pooled relative recall. The design is fixed in
analysis/notes/2026-11-17_baselines_design.md.

Everything runs locally on the CPU; weights come from analysis/.bench-cache/ (see the methods note).
"""
import argparse
import base64
import datetime
import hashlib
import io
import json
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".bench-cache")
os.environ.setdefault("HF_HOME", os.path.join(CACHE, "hf"))
os.environ.setdefault("TORCH_HOME", os.path.join(CACHE, "torch"))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(CACHE, "ultralytics_cfg"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["YOLO_OFFLINE"] = "1"
os.environ["YOLO_AUTOINSTALL"] = "False"

import numpy as np  # noqa: E402

sys.path.insert(0, HERE)
import bench_common as bc  # noqa: E402

APP_MODELS = os.path.join(HERE, "..", "FindBack", "app", "src", "main", "assets", "models")
MODELS = os.path.join(CACHE, "models")
VOCAB_FILE = os.path.join(HERE, "bench_vocab_household.txt")
DETECTOR_ORDER = ["effdet0", "effdet2", "yolo26n", "yoloe11s", "owlv2"]
DETECTOR_LABEL = {
    "effdet0": "Ours: EfficientDet-Lite0 (app)",
    "effdet2": "EfficientDet-Lite2",
    "yolo26n": "YOLO26n",
    "yoloe11s": "YOLOE-11S, household vocabulary",
    "owlv2": "OWLv2 base, household vocabulary",
}
OPERATING_POINT = {"effdet0": 0.35, "effdet2": 0.35, "yolo26n": 0.25, "yoloe11s": 0.25, "owlv2": 0.10}
RAW_MIN = {"effdet0": 0.20, "effdet2": 0.20}     # the app runs the detector at 0.20, keeps >= 0.35
RAW_MAX = 25
TORCH_THREADS = 4
OWL_NMS_IOU = 0.5
JUDGE_SEED = 20261117
CLUSTER_IOU = 0.5

# ScanConfig (FindBack scan/ScanConfig.kt)
PERSON = "person"
COCO_STOP = {"couch", "bed", "dining table", "toilet", "tv", "refrigerator", "oven", "sink", "chair", "bench",
             "potted plant"}
MAX_OBJECT_M = 0.6
ON_PERSON_OVERLAP = 0.5
DUPLICATE_IOU = 0.6
BOX_CENTRAL_FRACTION = 0.40
DEPTH_MIN_M, DEPTH_MAX_M = 0.2, 4.0
MAX_PER_KEYFRAME = 10


# ----------------------------------------------------------------------------- vocabulary

def read_vocab(path=VOCAB_FILE):
    """{'items': [...], 'stop': [...]} from the vocabulary file; raises on a duplicate name."""
    out, role, seen = {"items": [], "stop": []}, None, set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s in ("[items]", "[stop]"):
                role = s[1:-1]
                continue
            if role is None:
                raise ValueError("name before a section header: %r" % s)
            if s.lower() in seen:
                raise ValueError("duplicate name in vocabulary: %r" % s)
            seen.add(s.lower())
            out[role].append(s)
    return out


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ----------------------------------------------------------------------------- geometry

def upright_box_to_sensor(box, scale, rotation, sensor_w, sensor_h):
    """Upright.boxToSensor: a box found in an upright image scaled by `scale`, in sensor pixels."""
    up_w = sensor_w if rotation % 180 == 0 else sensor_h
    up_h = sensor_h if rotation % 180 == 0 else sensor_w

    def to_sensor(x, y):
        if rotation == 0:
            return x, y
        if rotation == 90:
            return y, up_w - x
        if rotation == 180:
            return up_w - x, up_h - y
        if rotation == 270:
            return up_h - y, x
        raise ValueError(rotation)

    a = to_sensor(box[0] / scale, box[1] / scale)
    b = to_sensor(box[2] / scale, box[3] / scale)
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))


def sensor_box_to_upright(box, scale, rotation, sensor_w, sensor_h):
    """Inverse of upright_box_to_sensor (for drawing boxes on the upright keyframe)."""
    def from_sensor(x, y):
        if rotation == 0:
            return x, y
        if rotation == 90:
            return sensor_h - y, x
        if rotation == 180:
            return sensor_w - x, sensor_h - y
        if rotation == 270:
            return y, sensor_w - x
        raise ValueError(rotation)

    a = from_sensor(box[0], box[1])
    b = from_sensor(box[2], box[3])
    return (min(a[0], b[0]) * scale, min(a[1], b[1]) * scale, max(a[0], b[0]) * scale, max(a[1], b[1]) * scale)


class DepthMap:
    """FindBack scan/ScanGeometry.kt DepthMap: uint16 millimetres, 0 = no estimate, with the
    affine map from CPU image pixels to depth pixels."""

    def __init__(self, mm, from_image):
        self.mm = np.asarray(mm, np.uint16)
        self.h, self.w = self.mm.shape
        self.a, self.b, self.c, self.d, self.e, self.f = (float(v) for v in from_image)

    @classmethod
    def load(cls, path, w, h, from_image):
        return cls(np.fromfile(path, dtype="<u2", count=w * h).reshape(h, w), from_image)

    def _range(self, lo, hi, n):
        import math
        i0, i1 = math.ceil(lo - 0.5), math.floor(hi - 0.5)
        if i0 > i1:
            i0 = i1 = math.floor((lo + hi) / 2)
        if i1 < 0 or i0 > n - 1:
            return None
        return max(0, i0), min(n - 1, i1)

    def median_in_box(self, box):
        xs, ys = [], []
        for cx in (box[0], box[2]):
            for cy in (box[1], box[3]):
                xs.append(self.a * cx + self.b * cy + self.c)
                ys.append(self.d * cx + self.e * cy + self.f)
        xr, yr = self._range(min(xs), max(xs), self.w), self._range(min(ys), max(ys), self.h)
        if xr is None or yr is None:
            return None
        v = self.mm[yr[0]:yr[1] + 1, xr[0]:xr[1] + 1].ravel()
        v = np.sort(v[v != 0])
        return None if v.size == 0 else int(v[v.size // 2]) / 1000.0


def central(box, fraction):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = (box[2] - box[0]) * fraction / 2, (box[3] - box[1]) * fraction / 2
    return (cx - w, cy - h, cx + w, cy + h)


def inter_area(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if (w > 0 and h > 0) else 0.0


def keyframe_filter(dets, op, stop, depth, fx):
    """FindBack DetectionFilter.apply with the detector's operating point `op` and stop list.

    dets: list of dict(box (sensor px), label, score). Returns the same dicts with 'reason'
    (None when kept), 'depthM' and 'sizeM' added, in the given order.
    """
    out = [dict(d, label=d["label"].strip().lower(), reason=None, depthM=None, sizeM=None) for d in dets]
    persons = [d["box"] for d in out if d["label"] == PERSON]
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])  # noqa: E731
    for d in out:
        b = d["box"]
        if d["label"] == PERSON:
            d["reason"] = "person"
        elif d["score"] < op:
            d["reason"] = "low_score"
        elif not d["label"] or d["label"] == "???":
            d["reason"] = "no_label"
        elif d["label"] in stop:
            d["reason"] = "stop_list"
        elif area(b) <= 0:
            d["reason"] = "empty_box"
        elif any(inter_area(b, p) / area(b) > ON_PERSON_OVERLAP for p in persons):
            d["reason"] = "on_person"
    winners = []
    for d in sorted([d for d in out if d["reason"] is None], key=lambda d: -d["score"]):
        if any(w["label"] == d["label"] and bc.iou(w["box"], d["box"]) > DUPLICATE_IOU for w in winners):
            d["reason"] = "duplicate"
        else:
            winners.append(d)
    for d in winners:
        b = d["box"]
        z = depth.median_in_box(central(b, BOX_CENTRAL_FRACTION)) if depth is not None else None
        if z is None:
            d["reason"] = "no_depth"
            continue
        d["depthM"] = z
        if z < DEPTH_MIN_M or z > DEPTH_MAX_M:
            d["reason"] = "depth_range"
            continue
        d["sizeM"] = max(b[2] - b[0], b[3] - b[1]) * z / fx
        if d["sizeM"] > MAX_OBJECT_M:
            d["reason"] = "too_large"
    for d in [w for w in winners if w["reason"] is None][MAX_PER_KEYFRAME:]:
        d["reason"] = "over_cap"
    return out


# ----------------------------------------------------------------------------- detectors

class Detector:
    params = None
    file_bytes = None
    stop = COCO_STOP

    def detect(self, upright):
        """Raw detections on the upright half-size image: list of dict(box (upright px), label, score)."""
        raise NotImplementedError


class MediaPipeDetector(Detector):
    def __init__(self, path):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        self._mp = mp
        self.det = vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
            base_options=mpt.BaseOptions(model_asset_path=path), score_threshold=0.20, max_results=RAW_MAX))
        self.file_bytes = os.path.getsize(path)
        from bench_match import tflite_constants
        self.params = tflite_constants(path)

    def detect(self, upright):
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(upright))
        out = []
        for d in self.det.detect(img).detections:
            best = max(d.categories, key=lambda c: c.score)
            bb = d.bounding_box
            out.append({"box": (bb.origin_x, bb.origin_y, bb.origin_x + bb.width, bb.origin_y + bb.height),
                        "label": best.category_name or "", "score": float(best.score)})
        return out


class UltralyticsDetector(Detector):
    def __init__(self, weights, conf, names=None, stop=None):
        import torch
        torch.set_num_threads(TORCH_THREADS)
        path = os.path.join(MODELS, weights)
        if names is not None:
            from ultralytics import YOLOE
            self.model = YOLOE(path)
            self.model.set_classes(names, self.model.get_text_pe(names))
        else:
            from ultralytics import YOLO
            self.model = YOLO(path)
        self.conf = conf
        if stop is not None:
            self.stop = stop
        self.file_bytes = os.path.getsize(path)
        self.params = sum(p.numel() for p in self.model.model.parameters())

    def detect(self, upright):
        r = self.model.predict(np.ascontiguousarray(upright[..., ::-1]), conf=self.conf, imgsz=640, device="cpu",
                               verbose=False, max_det=300)[0]
        names = r.names
        out = []
        for xyxy, c, k in zip(r.boxes.xyxy.numpy(), r.boxes.conf.numpy(), r.boxes.cls.numpy().astype(int)):
            out.append({"box": tuple(float(v) for v in xyxy), "label": names[int(k)], "score": float(c)})
        return out


class Owlv2Detector(Detector):
    def __init__(self, names, stop, threshold):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        from bench_match import hf_snapshot
        torch.set_num_threads(TORCH_THREADS)
        self.torch = torch
        path = hf_snapshot("google/owlv2-base-patch16-ensemble")
        self.processor = Owlv2Processor.from_pretrained(path)
        self.model = Owlv2ForObjectDetection.from_pretrained(path).eval()
        self.names = list(names)
        self.queries = ["a photo of a %s" % n for n in self.names]
        self.threshold = threshold
        self.stop = stop
        self.params = sum(p.numel() for p in self.model.parameters())
        self.file_bytes = os.path.getsize(os.path.join(path, "model.safetensors"))

    def detect(self, upright):
        from PIL import Image
        from torchvision.ops import batched_nms
        img = Image.fromarray(upright)
        inputs = self.processor(text=[self.queries], images=img, return_tensors="pt")
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        s = max(img.size)   # OWLv2 pads to a square at the bottom and right
        res = self.processor.image_processor.post_process_object_detection(outputs, threshold=self.threshold,
                                                            target_sizes=self.torch.tensor([[s, s]]))[0]
        boxes, scores, labels = res["boxes"], res["scores"], res["labels"]
        if boxes.numel():
            keep = batched_nms(boxes, scores, labels, OWL_NMS_IOU)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        h, w = upright.shape[:2]
        out = []
        for b, sc, lb in zip(boxes.numpy(), scores.numpy(), labels.numpy()):
            box = (max(0.0, float(b[0])), max(0.0, float(b[1])), min(float(w), float(b[2])), min(float(h), float(b[3])))
            if box[2] > box[0] and box[3] > box[1]:
                out.append({"box": box, "label": self.names[int(lb)], "score": float(sc)})
        return out


def make_detector(key, vocab):
    all_names = vocab["items"] + vocab["stop"]
    vstop = {n.lower() for n in vocab["stop"]} | {PERSON}
    if key == "effdet0":
        return MediaPipeDetector(os.path.join(APP_MODELS, "efficientdet_lite0.tflite"))
    if key == "effdet2":
        return MediaPipeDetector(os.path.join(MODELS, "efficientdet_lite2_int8_1.tflite"))
    if key == "yolo26n":
        return UltralyticsDetector("yolo26n.pt", OPERATING_POINT[key])
    if key == "yoloe11s":
        return UltralyticsDetector("yoloe-11s-seg.pt", OPERATING_POINT[key], names=all_names, stop=vstop)
    if key == "owlv2":
        return Owlv2Detector(all_names, vstop, OPERATING_POINT[key])
    raise ValueError(key)


# ----------------------------------------------------------------------------- frames

def load_frames(bd):
    """Saved keyframes with an image: dict(id, scan, index, session, upright (H x W x 3), scale,
    rotation, sensor_w, sensor_h, depth (DepthMap or None), fx, logged (raw detections))."""
    frames = []
    for scan_dir, kf in bd.keyframes:
        p = bd.keyframe_image_path(scan_dir, kf)
        if p is None:
            continue
        sensor = bc.load_rgb(p)
        img = kf["image"]
        sw, sh = img.get("sourceWidth") or sensor.shape[1], img.get("sourceHeight") or sensor.shape[0]
        rot = int(kf.get("rotation") or 0)
        depth = None
        dmeta = kf.get("depth")
        if isinstance(dmeta, dict) and dmeta.get("file") and isinstance(kf.get("depthFromImage"), list):
            dp = os.path.join(scan_dir, dmeta["file"])
            if os.path.exists(dp):
                depth = DepthMap.load(dp, dmeta["width"], dmeta["height"], kf["depthFromImage"])
        frames.append({
            "id": "%s/%04d" % (kf.get("scan"), kf.get("index") or 0), "scan": kf.get("scan"), "index": kf.get("index"),
            "session": kf.get("session"), "upright": np.ascontiguousarray(bc.sensor_to_upright(sensor, rot)),
            "scale": sensor.shape[1] / float(sw), "rotation": rot, "sensor_w": int(sw), "sensor_h": int(sh),
            "depth": depth, "fx": float((kf.get("intrinsics") or {}).get("fx") or 0.0),
            "logged": [{"box": tuple(float(v) for v in d["box"]), "label": (d.get("label") or "").strip().lower(),
                        "score": float(d.get("score") or 0.0), "kept": bool(d.get("kept"))}
                       for d in kf.get("detections") or []],
            "detector_logged": kf.get("detector"),
        })
    return frames


def run_detector(key, det, frames, log):
    """Per frame: filtered reports (sensor boxes) and the time of the detect call."""
    per_frame, times = {}, []
    for i, fr in enumerate(frames):
        t0 = time.perf_counter()
        raw = det.detect(fr["upright"])
        dt = time.perf_counter() - t0
        if i >= 2:
            times.append(dt)
        dets = []
        for d in raw:
            if d["score"] < RAW_MIN.get(key, 0.0):
                continue
            box = upright_box_to_sensor(d["box"], fr["scale"], fr["rotation"], fr["sensor_w"], fr["sensor_h"])
            dets.append({"box": box, "label": d["label"], "score": d["score"]})
        dets.sort(key=lambda d: -d["score"])
        per_frame[fr["id"]] = keyframe_filter(dets, OPERATING_POINT[key], det.stop, fr["depth"], fr["fx"] or 1.0)
        if (i + 1) % 20 == 0:
            log("    %s: %d / %d frames" % (key, i + 1, len(frames)))
    return per_frame, times


# ----------------------------------------------------------------------------- label-free numbers

def label_free(key, per_frame, times):
    raw_at_op = [sum(1 for d in ds if d["score"] >= OPERATING_POINT[key] and d["reason"] != "person"
                     and d["label"] != PERSON) for ds in per_frame.values()]
    kept = [[d for d in ds if d["reason"] is None] for ds in per_frame.values()]
    n_kept = [len(k) for k in kept]
    scores = sorted(d["score"] for k in kept for d in k)
    names = {}
    for k in kept:
        for d in k:
            names[d["label"]] = names.get(d["label"], 0) + 1
    reasons = {}
    for ds in per_frame.values():
        for d in ds:
            if d["score"] >= OPERATING_POINT[key] or d["reason"] == "person":
                reasons[d["reason"] or "kept"] = reasons.get(d["reason"] or "kept", 0) + 1
    person_frames = sum(any(d["label"] == PERSON and d["score"] >= OPERATING_POINT[key] for d in ds)
                        for ds in per_frame.values())

    def q(p):
        return float(np.quantile(scores, p)) if scores else float("nan")

    return {
        "frames": len(per_frame), "raw_mean": float(np.mean(raw_at_op)) if raw_at_op else float("nan"),
        "kept_total": sum(n_kept), "kept_mean": float(np.mean(n_kept)) if n_kept else float("nan"),
        "kept_median": float(np.median(n_kept)) if n_kept else float("nan"),
        "frames_none": sum(n == 0 for n in n_kept), "distinct": len(names),
        "top_names": sorted(names.items(), key=lambda kv: (-kv[1], kv[0]))[:8],
        "score_q": (q(0.25), q(0.5), q(0.75)), "ms_frame": 1000 * statistics.median(times) if times else float("nan"),
        "reasons": reasons, "person_frames": person_frames,
    }


def phone_agreement(frames, per_frame):
    """Offline EfficientDet-Lite0 against the raw detections the phone logged for the same frames."""
    n_log = n_off = n_match = 0
    ious, dscore = [], []
    kept_log = kept_off = kept_both = 0
    for fr in frames:
        logged = list(fr["logged"])
        off = per_frame.get(fr["id"], [])
        n_log += len(logged)
        n_off += len(off)
        used = set()
        for d in sorted(off, key=lambda d: -d["score"]):
            best, bi = 0.0, None
            for j, l in enumerate(logged):
                if j in used or l["label"] != d["label"]:
                    continue
                v = bc.iou(d["box"], l["box"])
                if v > best:
                    best, bi = v, j
            if bi is not None and best >= 0.5:
                used.add(bi)
                n_match += 1
                ious.append(best)
                dscore.append(abs(d["score"] - logged[bi]["score"]))
                if d["reason"] is None and logged[bi]["kept"]:
                    kept_both += 1
        kept_log += sum(l["kept"] for l in logged)
        kept_off += sum(d["reason"] is None for d in off)
    return {"frames": len(frames), "n_logged": n_log, "n_offline": n_off, "n_matched": n_match,
            "iou_median": float(np.median(ious)) if ious else float("nan"),
            "dscore_median": float(np.median(dscore)) if dscore else float("nan"),
            "dscore_max": float(np.max(dscore)) if dscore else float("nan"),
            "kept_logged": kept_log, "kept_offline": kept_off, "kept_both": kept_both}


# ----------------------------------------------------------------------------- judging sheet

def cluster_frame(dets_by_detector):
    """Greedy grouping of the kept boxes of all detectors in one keyframe into objects: boxes in
    descending score order join the first object whose first box overlaps them at IoU >= 0.5."""
    boxes = [(d["score"], k, d) for k, ds in dets_by_detector.items() for d in ds]
    boxes.sort(key=lambda x: -x[0])
    objects = []
    for _, k, d in boxes:
        for o in objects:
            if bc.iou(o["rep"]["box"], d["box"]) >= CLUSTER_IOU:
                o["members"].append((k, d))
                break
        else:
            objects.append({"rep": d, "members": [(k, d)]})
    return objects


def build_judging(frames, results, detectors, seed=JUDGE_SEED):
    """(sheet data for the HTML, key mapping objects and names back to detectors)."""
    rng = random.Random(seed)
    order = list(range(len(frames)))
    rng.shuffle(order)
    sheet, key = [], {"seed": seed, "detectors": detectors, "frames": []}
    for n, fi in enumerate(order):
        fr = frames[fi]
        kept = {k: [d for d in results[k][fr["id"]] if d["reason"] is None] for k in detectors}
        objs = cluster_frame(kept)
        fid = "F%03d" % (n + 1)
        sf = {"id": fid, "objects": []}
        kf = {"id": fid, "frame": fr["id"], "objects": []}
        for j, o in enumerate(objs):
            oid = "%s-%d" % (fid, j + 1)
            names = sorted({d["label"] for _, d in o["members"]})
            rng.shuffle(names)
            ub = sensor_box_to_upright(o["rep"]["box"], fr["scale"], fr["rotation"], fr["sensor_w"], fr["sensor_h"])
            sf["objects"].append({"id": oid, "box": [round(v, 1) for v in ub], "names": names})
            kf["objects"].append({"id": oid, "names": names, "members": [
                {"detector": k, "label": d["label"], "score": round(d["score"], 4),
                 "box": [round(v, 1) for v in d["box"]]} for k, d in o["members"]]})
        sheet.append((sf, fr))
        key["frames"].append(kf)
    return sheet, key


SHEET_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Naming judgments</title>
<style>
:root{--bg:#fafafa;--fg:#1d1d1f;--muted:#6e6e73;--card:#fff;--line:#d2d2d7;--accent:#0a64d6;--ok:#1b7f3b;--warn:#a05a00}
@media (prefers-color-scheme: dark){:root{--bg:#151516;--fg:#f2f2f2;--muted:#a1a1a6;--card:#1f1f21;--line:#3a3a3c;--accent:#5aa2ff;--ok:#4cc38a;--warn:#e0a040}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;z-index:2;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
header b{margin-right:auto}
main{max-width:980px;margin:0 auto;padding:16px}
.frame{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:0 0 22px;padding:12px}
.frame h2{font-size:16px;margin:0 0 8px}
.obj{display:grid;grid-template-columns:150px 180px 1fr;gap:12px;border-top:1px solid var(--line);padding:10px 0}
@media (max-width:640px){.obj{grid-template-columns:1fr 1fr}.obj .q{grid-column:1/3}}
canvas{width:100%;height:auto;border-radius:6px;background:#000}
.q fieldset{border:0;margin:0 0 8px;padding:0}
.q legend{font-weight:600;padding:0;margin-bottom:2px}
.q label{display:inline-block;margin:2px 10px 2px 0;white-space:nowrap}
.name{font-family:ui-monospace,Menlo,monospace;background:rgba(127,127,127,.15);padding:1px 6px;border-radius:4px}
.done{color:var(--ok)} .todo{color:var(--warn)}
button,.btn{font:inherit;padding:5px 10px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
.help{color:var(--muted);font-size:14px}
</style></head><body>
<header><b>Naming judgments</b><span id="progress"></span>
<button id="export">Download judgments</button><label class="btn">Load judgments<input id="import" type="file" accept=".json" hidden></label></header>
<main>
<p class="help">For each outlined object: first say what is in the box (Q1). If it is an item, judge every proposed name (Q2):
<b>right</b> = a correct name for the main thing in the box; <b>close</b> = acceptable but loose or a near synonym (for example "mug" for a cup);
<b>wrong</b> = not what it is. Which detector proposed which name is not shown. Work frame by frame; a frame counts only when it is complete.
Answers are saved in this browser; press <i>Download judgments</i> to save them to a file.</p>
<div id="frames"></div></main>
<script>
const SHEET = __SHEET__;
const KEY = "findback-judge-" + SHEET.sheet_id;
let J = {};
try { J = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { J = {}; }
function save(){ try { localStorage.setItem(KEY, JSON.stringify(J)); } catch (e) {} progress(); }
function el(t, a, kids){ const e=document.createElement(t); for (const k in (a||{})) { if (k==="text") e.textContent=a[k]; else e.setAttribute(k,a[k]); } (kids||[]).forEach(c=>e.appendChild(c)); return e; }
function radio(name, value, label, checked, onch){ const i=el("input",{type:"radio",name:name,value:value}); if(checked) i.checked=true; i.addEventListener("change",onch); return el("label",{},[i,document.createTextNode(" "+label)]); }
function frameDone(f){ return f.objects.every(o => { const a=J[o.id]; if(!a||!a.q1) return false; if(a.q1!=="item") return true; return o.names.every(n => a.names && a.names[n]); }); }
function progress(){ const done=SHEET.frames.filter(frameDone).length; document.getElementById("progress").textContent = done+" of "+SHEET.frames.length+" frames complete";
  SHEET.frames.forEach(f=>{ const s=document.getElementById("st-"+f.id); if(s){ const d=frameDone(f); s.textContent=d?"complete":"incomplete"; s.className=d?"done":"todo"; } }); }
function draw(img, cv, crop, box){
  const ctx=cv.getContext("2d");
  if(!crop){ cv.width=img.width/2; cv.height=img.height/2; ctx.drawImage(img,0,0,cv.width,cv.height); ctx.lineWidth=3; ctx.strokeStyle="#ff2d55";
    ctx.strokeRect(box[0]/2,box[1]/2,(box[2]-box[0])/2,(box[3]-box[1])/2); return; }
  const w=box[2]-box[0], h=box[3]-box[1], px=w*0.08, py=h*0.08;
  const x0=Math.max(0,box[0]-px), y0=Math.max(0,box[1]-py), x1=Math.min(img.width,box[2]+px), y1=Math.min(img.height,box[3]+py);
  const s=Math.min(360/(x1-x0), 360/(y1-y0)); cv.width=Math.max(1,Math.round((x1-x0)*s)); cv.height=Math.max(1,Math.round((y1-y0)*s));
  ctx.drawImage(img,x0,y0,x1-x0,y1-y0,0,0,cv.width,cv.height); }
const root=document.getElementById("frames");
SHEET.frames.forEach(f=>{
  const box=el("section",{class:"frame"});
  box.appendChild(el("h2",{},[document.createTextNode("Frame "+f.id+" · "+f.objects.length+" object(s) · "), el("span",{id:"st-"+f.id})]));
  if(!f.objects.length) box.appendChild(el("p",{class:"help",text:"No detector kept anything in this frame."}));
  const img=new Image(); img.src=f.image;
  f.objects.forEach(o=>{
    const a=J[o.id]||(J[o.id]={names:{}}); if(!a.names) a.names={};
    const c1=el("canvas"), c2=el("canvas"); img.addEventListener("load",()=>{ draw(img,c1,false,o.box); draw(img,c2,true,o.box); });
    const q=el("div",{class:"q"});
    const fs1=el("fieldset",{},[el("legend",{text:"Q1 · object "+o.id+": what is in the box?"})]);
    [["item","one item a person could look for"],["part","part of an item or several items"],["not_item","not an item (furniture, wall, background)"],["unclear","can't tell"]].forEach(([v,l])=>
      fs1.appendChild(radio(o.id+"-q1",v,l,a.q1===v,()=>{ a.q1=v; save(); })));
    q.appendChild(fs1);
    const fs2=el("fieldset",{},[el("legend",{text:"Q2 · if it is an item, is each name right?"})]);
    o.names.forEach(n=>{ const row=el("div",{},[el("span",{class:"name",text:n}),document.createTextNode("  ")]);
      [["right","right"],["close","close"],["wrong","wrong"]].forEach(([v,l])=> row.appendChild(radio(o.id+"-"+n,v,l,a.names[n]===v,()=>{ a.names[n]=v; save(); })));
      fs2.appendChild(row); });
    q.appendChild(fs2);
    box.appendChild(el("div",{class:"obj"},[c1,c2,q]));
  });
  root.appendChild(box);
});
document.getElementById("export").addEventListener("click",()=>{
  const blob=new Blob([JSON.stringify({sheet_id:SHEET.sheet_id, saved:new Date().toISOString(), judgments:J},null,1)],{type:"application/json"});
  const a=el("a",{href:URL.createObjectURL(blob),download:"judgments_"+SHEET.sheet_id+".json"}); document.body.appendChild(a); a.click(); a.remove(); });
document.getElementById("import").addEventListener("change",ev=>{ const f=ev.target.files[0]; if(!f) return; f.text().then(t=>{ const d=JSON.parse(t);
  if(d.sheet_id!==SHEET.sheet_id && !confirm("These judgments are for another sheet. Load anyway?")) return; J=d.judgments||{}; save(); location.reload(); }); });
progress();
</script></body></html>
"""


def write_sheet(sheet, key, out_dir):
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for sf, fr in sheet:
        buf = io.BytesIO()
        Image.fromarray(fr["upright"]).save(buf, format="JPEG", quality=85)
        frames.append(dict(sf, image="data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()))
    body = json.dumps({"frames": [{"id": f["id"], "objects": f["objects"]} for f, _ in sheet]}, sort_keys=True)
    sheet_id = hashlib.sha256(body.encode()).hexdigest()[:12]
    key["sheet_id"] = sheet_id
    html = SHEET_HTML.replace("__SHEET__", json.dumps({"sheet_id": sheet_id, "frames": frames}))
    with open(os.path.join(out_dir, "judge_sheet.html"), "w", encoding="utf-8") as f:
        f.write(html)
    with open(os.path.join(out_dir, "judge_key.json"), "w", encoding="utf-8") as f:
        json.dump(key, f, indent=1)
    return sheet_id


# ----------------------------------------------------------------------------- report

def markdown(args, frames, stats, agree, costs, vocab, sheet_info, detectors, dropped=None):
    L = []
    w = L.append
    w("# Track B results: detection, label-free numbers")
    w("")
    w("Generated by `analysis/bench_detect.py` on %s from %s. Design: "
      "`analysis/notes/2026-11-17_baselines_design.md`; methods and citations: "
      "`analysis/notes/2026-11-17_methods.md`. Vocabulary: `analysis/bench_vocab_household.txt` "
      "(%d item names, %d stop names, sha256 %s…)." % (
          datetime.date.today().isoformat(),
          ", ".join("`%s`" % os.path.relpath(p, os.path.join(HERE, "..")) for p in args.data),
          len(vocab["items"]), len(vocab["stop"]), file_sha256(VOCAB_FILE)[:12]))
    w("")
    w("**Small sample.** %d saved keyframes from %d scan(s) of one room (keyframes with a person "
      "detected by the phone are never saved and are missing). Consecutive keyframes overlap heavily, "
      "so the frames are not independent. No labels: nothing below says whether a detection is right. "
      "Naming precision and relative recall need the judging sheet." % (
          len(frames), len({f["scan"] for f in frames})))
    w("")
    w("Every detector's raw output goes through the app's keyframe filter (person, stop list, same-name "
      "boxes over IoU 0.6, depth 0.2–4.0 m, longer side at most 0.6 m, at most 10 per frame). *Raw* counts "
      "detections at the operating point that are not a person; *kept* counts what passes the filter.")
    w("")
    w("| Detector | Operating point | Raw per frame | Kept per frame (mean / median) | Frames with none kept | Kept in all | Distinct names kept | Kept score quartiles | Frames with a person | ms per frame |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for k in detectors:
        s = stats[k]
        w("| %s | %.2f | %.2f | %.2f / %.0f | %d of %d | %d | %d | %s / %s / %s | %d | %.0f |" % (
            DETECTOR_LABEL[k], OPERATING_POINT[k], s["raw_mean"], s["kept_mean"], s["kept_median"], s["frames_none"],
            s["frames"], s["kept_total"], s["distinct"], bc.fmt(s["score_q"][0], 2), bc.fmt(s["score_q"][1], 2),
            bc.fmt(s["score_q"][2], 2), s["person_frames"], s["ms_frame"]))
    w("")
    w("Scores are not comparable across detectors (each has its own calibration and operating point).")
    w("")
    w("### Why detections were dropped (at or above the operating point)")
    w("")
    reasons = sorted({r for k in detectors for r in stats[k]["reasons"]})
    w("| Detector | " + " | ".join(reasons) + " |")
    w("| --- | " + " | ".join("---" for _ in reasons) + " |")
    for k in detectors:
        w("| %s | %s |" % (DETECTOR_LABEL[k], " | ".join(str(stats[k]["reasons"].get(r, 0)) for r in reasons)))
    w("")
    w("### Most frequent kept names")
    w("")
    for k in detectors:
        w("- %s: %s" % (DETECTOR_LABEL[k], ", ".join("%s %d" % kv for kv in stats[k]["top_names"]) or "none"))
    w("")
    if agree:
        a = agree
        w("## Ours reproduces the phone (EfficientDet-Lite0)")
        w("")
        w("Raw detections at score 0.20 or more on the same %d keyframes: the phone logged %d, the offline run "
          "found %d; %d pairs match (same label, IoU at least 0.5), median IoU %s, median |score difference| %s "
          "(max %s). Kept after the filter: phone %d, offline %d, kept by both %d." % (
              a["frames"], a["n_logged"], a["n_offline"], a["n_matched"], bc.fmt(a["iou_median"]),
              bc.fmt(a["dscore_median"]), bc.fmt(a["dscore_max"]), a["kept_logged"], a["kept_offline"], a["kept_both"]))
        w("")
    w("## Cost on this Mac (not phone time)")
    w("")
    w("CPU only (%s), %d torch threads, one frame at a time, median after two warm-up frames, model "
      "call only (includes each library's own pre- and post-processing)." % (args.cpu_name, TORCH_THREADS))
    w("")
    w("| Detector | Parameters | File | ms per frame |")
    w("| --- | --- | --- | --- |")
    for k in detectors:
        c = costs[k]
        w("| %s | %s | %s | %.0f |" % (DETECTOR_LABEL[k], "–" if not c["params"] else "%.2f M" % (c["params"] / 1e6),
                                       "–" if not c["file_bytes"] else "%.1f MB" % (c["file_bytes"] / 1e6), stats[k]["ms_frame"]))
    w("")
    if sheet_info:
        w("## Judging sheet")
        w("")
        w("`analysis/out/judge/judge_sheet.html` (sheet id `%s`): %d frames, %d objects, %d (object, name) "
          "judgments to make. The key is `analysis/out/judge/judge_key.json`. Read the exported judgments with "
          "`analysis/bench_judge_read.py`." % (sheet_info["sheet_id"], sheet_info["frames"], sheet_info["objects"],
                                               sheet_info["names"]))
        w("")
    if dropped:
        w("## Detectors that could not be run")
        w("")
        for k, why in dropped.items():
            w("- %s: %s" % (DETECTOR_LABEL[k], why))
        w("")
    w("## Changes after the design was fixed")
    w("")
    w(args.changes)
    w("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", required=True, help="pulled data folder (repeatable)")
    ap.add_argument("--exclude-session", action="append", default=[])
    ap.add_argument("--no-default-exclude", action="store_true")
    ap.add_argument("--detectors", default=",".join(DETECTOR_ORDER))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "bench_detect"))
    ap.add_argument("--judge-out", default=os.path.join(HERE, "out", "judge"))
    ap.add_argument("--no-sheet", action="store_true")
    ap.add_argument("--md", help="also write the Markdown report here (numbers only)")
    ap.add_argument("--changes", default="None.")
    args = ap.parse_args(argv)
    import platform
    import subprocess
    try:
        args.cpu_name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or platform.processor()
    except OSError:
        args.cpu_name = platform.processor()
    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]
    for d in detectors:
        if d not in DETECTOR_ORDER:
            ap.error("unknown detector %s" % d)

    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    vocab = read_vocab()
    excluded = set(args.exclude_session) | (set() if args.no_default_exclude else set(bc.DEFAULT_EXCLUDED_SESSIONS))
    bd = bc.BenchData(args.data, excluded)
    frames = load_frames(bd)
    log("%d saved keyframes" % len(frames))
    results, stats, costs, agree, dropped = {}, {}, {}, None, {}
    for k in list(detectors):
        log("running %s ..." % k)
        try:
            det = make_detector(k, vocab)
            per_frame, times = run_detector(k, det, frames, log)
        except Exception as e:   # a detector that cannot run is dropped and named in the report
            log("  %s dropped: %s: %s" % (k, type(e).__name__, e))
            dropped[k] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:200] if str(e) else "")
            detectors.remove(k)
            continue
        results[k] = per_frame
        stats[k] = label_free(k, per_frame, times)
        costs[k] = {"params": det.params, "file_bytes": det.file_bytes}
        if k == "effdet0":
            agree = phone_agreement(frames, per_frame)
        del det
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "detections.json"), "w") as f:
        json.dump({k: {fid: [{kk: (list(v) if kk == "box" else v) for kk, v in d.items()} for d in ds]
                       for fid, ds in res.items()} for k, res in results.items()}, f)
    sheet_info = None
    if not args.no_sheet and frames:
        sheet, key = build_judging(frames, results, detectors)
        sid = write_sheet(sheet, key, args.judge_out)
        sheet_info = {"sheet_id": sid, "frames": len(sheet), "objects": sum(len(sf["objects"]) for sf, _ in sheet),
                      "names": sum(len(o["names"]) for sf, _ in sheet for o in sf["objects"])}
    md = markdown(args, frames, stats, agree, costs, vocab, sheet_info, detectors, dropped)
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(md)
    if args.md:
        with open(args.md, "w") as f:
            f.write(md)
    print(md)


if __name__ == "__main__":
    main()
