#!/usr/bin/env python3
"""Track A of analysis/BASELINES_PLAN.md: appearance matching, offline, on the crops the phone saved.

    analysis/.venv-bench/bin/python analysis/bench_match.py --data backups/phone_files_2026-11-17_0008

Builds the pair sets of analysis/notes/2026-11-17_baselines_design.md (bench_common.build_pairs),
first checks that the offline "ours" reproduces the similarities the phone logged, then scores
every pair with every method and writes a number-only Markdown table (ROC AUC, equal error rate,
true-accept rate at 5% false accepts, each with an item bootstrap interval, and each method's AUC
difference from ours). Per-pair scores go to --out (default analysis/out/bench_match/, git-ignored).

Everything runs on this Mac, on the CPU. Model weights are read from analysis/.bench-cache/
(downloaded once from their official sources; see analysis/notes/2026-11-17_methods.md). The
script sets the Hugging Face libraries to offline mode, so nothing is fetched while it runs.
"""
import argparse
import csv
import datetime
import math
import os
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

import numpy as np  # noqa: E402

sys.path.insert(0, HERE)
import bench_common as bc  # noqa: E402
import fb_load  # noqa: E402

APP_MODELS = os.path.join(HERE, "..", "FindBack", "app", "src", "main", "assets", "models")
MODELS = os.path.join(CACHE, "models")
METHOD_ORDER = ["ours", "hist", "orb", "mnv3l", "dinov2s", "mobileclip2s2", "splg"]
METHOD_LABEL = {
    "ours": "Ours: MobileNetV3-small (app)",
    "hist": "HSV colour histogram",
    "orb": "ORB + RANSAC inliers",
    "mnv3l": "MobileNetV3-large",
    "dinov2s": "DINOv2 ViT-S/14",
    "mobileclip2s2": "MobileCLIP2-S2",
    "splg": "SuperPoint + LightGlue inliers",
}
MATCH_SIDE = 480        # longer side of the grey crop for the keypoint matchers
TORCH_THREADS = 4
# Hugging Face snapshots used (repository, commit), downloaded once into analysis/.bench-cache/hf.
HF_REVISIONS = {
    "facebook/dinov2-small": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
    "timm/MobileCLIP2-S2-OpenCLIP": "ac6b37c8fc40b62b623d09ed228a6f0c9bc29fe6",
    "google/owlv2-base-patch16-ensemble": "cfd3195ba4ea9592eec887ded089f4c08eff231d",
}


def hf_snapshot(repo):
    """Local folder of the pinned snapshot of `repo` (never downloads)."""
    path = os.path.join(os.environ["HF_HOME"], "hub", "models--" + repo.replace("/", "--"), "snapshots",
                        HF_REVISIONS[repo])
    if not os.path.isdir(path):
        raise FileNotFoundError("%s is not in %s; download it once (see analysis/README.md)" % (repo, path))
    return path


# ----------------------------------------------------------------------------- methods

def tflite_constants(path):
    """Number of stored constant elements (weights, biases, anchors) of a .tflite file."""
    try:
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path=path)
        inputs = {d["index"] for d in it.get_input_details()}
        n = 0
        # Before allocate_tensors() only constant tensors hold data; the others raise.
        for d in it.get_tensor_details():
            if d["index"] in inputs:
                continue
            try:
                n += int(it.get_tensor(d["index"]).size)
            except ValueError:
                continue
        return n
    except Exception:
        return None


class Method:
    kind = "embed"          # "embed": features compared by sim(); "match": pairwise match()
    params = None
    file_bytes = None
    note = ""

    def features(self, rgb):
        raise NotImplementedError

    def sim(self, fa, fb):
        return float(np.dot(fa, fb))


class MediaPipeEmbedder(Method):
    def __init__(self, path):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        self._mp = mp
        self.embedder = vision.ImageEmbedder.create_from_options(vision.ImageEmbedderOptions(
            base_options=mpt.BaseOptions(model_asset_path=path), l2_normalize=True, quantize=False))
        self.file_bytes = os.path.getsize(path)
        self.params = tflite_constants(path)

    def features(self, rgb):
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        v = np.asarray(self.embedder.embed(img).embeddings[0].embedding, np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n else v


class HsvHistogram(Method):
    params = 0
    note = "8x8x8 HSV bins, histogram intersection"

    def features(self, rgb):
        import cv2
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).ravel().astype(np.float64)
        s = h.sum()
        return h / s if s else h

    def sim(self, fa, fb):
        return float(np.minimum(fa, fb).sum())


def grey_resized(rgb, side=MATCH_SIDE):
    import cv2
    g = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    h, w = g.shape
    s = side / float(max(h, w))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.resize(g, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)


def ransac_inliers(pa, pb):
    """Inliers of a RANSAC homography (5 px) between matched points; 0 with fewer than 4."""
    import cv2
    if len(pa) < 4:
        return 0
    _, mask = cv2.findHomography(np.float32(pa), np.float32(pb), cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


class OrbRansac(Method):
    kind = "match"
    params = 0
    note = "1000 ORB features, ratio 0.8, RANSAC homography 5 px"

    def __init__(self):
        import cv2
        self.cv2 = cv2
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def features(self, rgb):
        kp, des = self.orb.detectAndCompute(grey_resized(rgb), None)
        return (np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32), des)

    def match(self, fa, fb):
        (pa, da), (pb, db) = fa, fb
        if da is None or db is None or len(da) < 2 or len(db) < 2:
            return 0.0
        good = [m for m, n in (x for x in self.bf.knnMatch(da, db, k=2) if len(x) == 2) if m.distance < 0.8 * n.distance]
        return float(ransac_inliers([pa[m.queryIdx] for m in good], [pb[m.trainIdx] for m in good]))


class Dinov2Small(Method):
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)
    note = "CLS token after the final layer norm, 224 x 224"

    def __init__(self):
        import torch
        from transformers import AutoModel
        torch.set_num_threads(TORCH_THREADS)
        self.torch = torch
        path = hf_snapshot("facebook/dinov2-small")
        self.model = AutoModel.from_pretrained(path).eval()
        self.params = sum(p.numel() for p in self.model.parameters())
        self.file_bytes = os.path.getsize(os.path.join(path, "model.safetensors"))
        self.revision = os.path.basename(path)

    def features(self, rgb):
        from PIL import Image
        x = np.asarray(Image.fromarray(rgb).resize((224, 224), Image.BILINEAR), np.float32) / 255.0
        x = (x - self.MEAN) / self.STD
        t = self.torch.from_numpy(x.transpose(2, 0, 1)[None].copy())
        with self.torch.no_grad():
            v = self.model(pixel_values=t).pooler_output[0].numpy()
        return v / np.linalg.norm(v)


class MobileClip2S2(Method):
    note = "image encoder only, 256 x 256"

    def __init__(self):
        import torch
        import open_clip
        torch.set_num_threads(TORCH_THREADS)
        self.torch = torch
        weights = os.path.join(hf_snapshot("timm/MobileCLIP2-S2-OpenCLIP"), "open_clip_model.safetensors")
        model, _, _ = open_clip.create_model_and_transforms("MobileCLIP2-S2", pretrained=weights)
        self.model = model.eval()
        # Fold the train-time branches for inference, as Apple's MobileCLIP README does. The
        # embeddings are unchanged (cosine 1.000 on test crops); only the time per crop drops.
        from timm.utils import reparameterize_model
        self.model.visual = reparameterize_model(self.model.visual)
        cfg = open_clip.get_pretrained_cfg("MobileCLIP2-S2", "dfndr2b")
        self.mean = np.array(cfg.get("mean", (0, 0, 0)), np.float32)
        self.std = np.array(cfg.get("std", (1, 1, 1)), np.float32)
        size = getattr(self.model.visual, "image_size", 256)
        self.size = size[0] if isinstance(size, (tuple, list)) else int(size)
        self.params = sum(p.numel() for p in self.model.visual.parameters())
        path = hf_snapshot("timm/MobileCLIP2-S2-OpenCLIP")
        self.file_bytes = os.path.getsize(os.path.join(path, "open_clip_model.safetensors"))
        self.revision = os.path.basename(path)

    def features(self, rgb):
        from PIL import Image
        x = np.asarray(Image.fromarray(rgb).resize((self.size, self.size), Image.BILINEAR), np.float32) / 255.0
        x = (x - self.mean) / self.std
        t = self.torch.from_numpy(x.transpose(2, 0, 1)[None].copy())
        with self.torch.no_grad():
            v = self.model.encode_image(t)[0].numpy()
        return v / np.linalg.norm(v)


class SuperPointLightGlue(Method):
    kind = "match"
    note = "SuperPoint (<= 1024 keypoints) + LightGlue, RANSAC homography 5 px"

    def __init__(self):
        import torch
        from lightglue import LightGlue, SuperPoint
        torch.set_num_threads(TORCH_THREADS)
        self.torch = torch
        self.extractor = SuperPoint(max_num_keypoints=1024).eval()
        self.matcher = LightGlue(features="superpoint").eval()
        self.params = sum(p.numel() for p in self.extractor.parameters()) + sum(p.numel() for p in self.matcher.parameters())
        ck = os.path.join(os.environ["TORCH_HOME"], "hub", "checkpoints")
        self.file_bytes = sum(os.path.getsize(os.path.join(ck, f)) for f in os.listdir(ck)
                              if f.startswith("superpoint")) if os.path.isdir(ck) else None

    def features(self, rgb):
        g = grey_resized(rgb).astype(np.float32) / 255.0
        t = self.torch.from_numpy(g)[None]
        with self.torch.no_grad():
            return self.extractor.extract(t, resize=None)

    def match(self, fa, fb):
        from lightglue.utils import rbd
        if fa["keypoints"].shape[1] < 2 or fb["keypoints"].shape[1] < 2:
            return 0.0
        with self.torch.no_grad():
            out = rbd(self.matcher({"image0": fa, "image1": fb}))
        m = out["matches"].numpy()
        if len(m) < 4:
            return 0.0
        ka, kb = rbd(fa)["keypoints"].numpy(), rbd(fb)["keypoints"].numpy()
        return float(ransac_inliers(ka[m[:, 0]], kb[m[:, 1]]))


def make_method(key):
    if key == "ours":
        return MediaPipeEmbedder(os.path.join(APP_MODELS, "mobilenet_v3_small.tflite"))
    if key == "mnv3l":
        return MediaPipeEmbedder(os.path.join(MODELS, "mobilenet_v3_large_float32_1.tflite"))
    if key == "hist":
        return HsvHistogram()
    if key == "orb":
        return OrbRansac()
    if key == "dinov2s":
        return Dinov2Small()
    if key == "mobileclip2s2":
        return MobileClip2S2()
    if key == "splg":
        return SuperPointLightGlue()
    raise ValueError(key)


# ----------------------------------------------------------------------------- scoring

def score_pairs(method, ps, log):
    """Scores of every pair (best over the pair's 'a' crops) and the cost measurements."""
    needed = sorted({k for p in ps.pairs for k in p["a"]} | {p["b"] for p in ps.pairs})
    feats, t_feat = {}, []
    for i, k in enumerate(needed):
        t0 = time.perf_counter()
        feats[k] = method.features(ps.crops[k])
        if i >= 3:
            t_feat.append(time.perf_counter() - t0)
    scores, t_pair, cache = np.empty(len(ps.pairs)), [], {}
    for i, p in enumerate(ps.pairs):
        best = -math.inf
        for a in p["a"]:
            key = (a, p["b"])
            if key not in cache:
                t0 = time.perf_counter()
                cache[key] = method.match(feats[a], feats[p["b"]]) if method.kind == "match" else method.sim(feats[a], feats[p["b"]])
                t_pair.append(time.perf_counter() - t0)
            best = max(best, cache[key])
        scores[i] = best
        if method.kind == "match" and i and i % 500 == 0:
            log("    %d / %d pairs" % (i, len(ps.pairs)))
    cost = {"ms_crop": 1000 * statistics.median(t_feat) if t_feat else float("nan"),
            "ms_pair": 1000 * statistics.median(t_pair) if t_pair else float("nan"),
            "n_crops": len(needed), "n_comparisons": len(cache)}
    return scores, feats, cost


# ----------------------------------------------------------------------------- reproduction check

def _agree(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if xs.size == 0:
        return {"n": 0}
    d = np.abs(xs - ys)
    r = float(np.corrcoef(xs, ys)[0, 1]) if xs.size > 2 and np.std(xs) > 0 and np.std(ys) > 0 else float("nan")
    return {"n": int(xs.size), "mean_abs": float(d.mean()), "max_abs": float(d.max()), "r": r}


def reproduction(bd, ps, ours, feats):
    """Offline ours against what the phone stored and logged."""
    out = {}
    # (a) stored crop JPEG -> embedding, against the stored embedding
    cos_a = []
    seen = set()
    for f in bd.ds.folders:
        for rec in list(f.records) + [r.get("record") for r in (f.records_log or []) if isinstance(r.get("record"), dict)]:
            if not isinstance(rec, dict) or rec.get("id") in seen or rec.get("sessionId") in bd.excluded:
                continue
            seen.add(rec.get("id"))
            v, p = fb_load.decode_embedding(rec.get("embedding")), bd.record_crop_path(rec)
            if v is not None and p:
                cos_a.append(("rec", float(np.dot(ours.features(bc.load_rgb(p)), v / np.linalg.norm(v)))))
    for eid, ent in bd.entries.items():
        v, p = fb_load.decode_embedding(ent.get("embedding")), bd.entry_crop_path(eid)
        if v is not None and p:
            cos_a.append(("inv", float(np.dot(ours.features(bc.load_rgb(p)), v / np.linalg.norm(v)))))
    out["stored"] = cos_a

    # (b) every logged candidate; (c) the best one and the 0.75 decision
    cand_off, cand_log, cand_phone_ref, cand_log2, emb_cos = [], [], [], [], []
    best_off, best_log = [], []
    for c in ps.captures:
        offs = []
        target = fb_load.decode_embedding(c["target_embedding"])
        emb = None
        if c.get("emb_file"):
            p = bd.find(c["emb_file"], c["session"])
            emb = fb_load.load_emb_file(p) if p else None
        for i, k in enumerate(c["cand_keys"]):
            s_off = float(np.dot(feats[k], feats[c["stored_key"]]))
            offs.append(s_off)
            logged = ps.crop_meta[k].get("logged_sim")
            if logged is not None:
                cand_off.append(s_off)
                cand_log.append(logged)
                if target is not None and target.size == feats[k].size:
                    cand_phone_ref.append(float(np.dot(feats[k], target / np.linalg.norm(target))))
                    cand_log2.append(logged)
            if emb and i < len(emb["candidates"]) and emb["candidates"][i]["embedding"] is not None:
                v = emb["candidates"][i]["embedding"]
                emb_cos.append(float(np.dot(feats[k], v / np.linalg.norm(v))))
        if c["logged_best"] is not None and offs:
            best_off.append(max(offs))
            best_log.append(c["logged_best"])
    out["cand"] = _agree(cand_off, cand_log)
    out["cand_phone_ref"] = _agree(cand_phone_ref, cand_log2)
    out["emb_cos"] = emb_cos
    out["best"] = _agree(best_off, best_log)
    out["decision"] = (sum((a >= bc.APP_THRESHOLD) == (b >= bc.APP_THRESHOLD) for a, b in zip(best_off, best_log)),
                       len(best_off))
    out["best_pairs"] = list(zip(best_log, best_off))
    return out


# ----------------------------------------------------------------------------- report

def evaluate(ps, scores, methods, set_name, n_boot):
    idx = ps.select(set_name)
    if not idx:
        return None
    labels = np.array([ps.pairs[i]["label"] for i in idx], int)
    ia = [ps.pairs[i]["item_a"] for i in idx]
    ib = [ps.pairs[i]["item_b"] for i in idx]
    counts = bc.pair_counts(labels, ia, ib)
    S = {m: scores[m][idx] for m in methods}
    ref = "ours" if "ours" in methods else None

    def stat(w):
        rows = [bc.metric_vector(S[m], labels, w) for m in methods]
        out = np.concatenate(rows)
        if ref:
            a_ref = rows[methods.index(ref)][0]
            out = np.r_[out, [r[0] - a_ref for r in rows]]
        return out

    full = stat(np.ones(len(idx)))
    boot = bc.item_bootstrap(ia, ib, stat, n_boot=n_boot)
    k = len(bc.METRICS)
    res = {"counts": counts, "methods": {}}
    for j, m in enumerate(methods):
        res["methods"][m] = {met: (full[j * k + q], boot["lo"][j * k + q], boot["hi"][j * k + q])
                             for q, met in enumerate(bc.METRICS)}
        if ref:
            o = len(methods) * k + j
            res["methods"][m]["dauc"] = (full[o], boot["lo"][o], boot["hi"][o])
    res["boot_ok"] = boot["ok"]
    res["n_items_boot"] = boot["n_items"]
    if "ours" in methods:
        res["ours_at_075"] = bc.rates_at_w(S["ours"], labels, bc.APP_THRESHOLD)
    return res


def markdown(bd, ps, results, costs, repro, methods, args, notes_changes):
    L = []
    w = L.append
    w("# Track A results: appearance matching")
    w("")
    w("Generated by `analysis/bench_match.py` on %s from %s. Sessions left out: %s. Design and pair "
      "definitions: `analysis/notes/2026-11-17_baselines_design.md`; methods and citations: "
      "`analysis/notes/2026-11-17_methods.md`." % (
          datetime.date.today().isoformat(), ", ".join("`%s`" % os.path.relpath(p, os.path.join(HERE, "..")) for p in args.data),
          ", ".join(sorted(bd.excluded)) or "none"))
    w("")
    w("**Small samples.** Every number below comes from one home and a handful of sessions, none "
      "of them tagged for the study; the item count n is the unit of the bootstrap. Intervals are "
      "95%% percentile intervals over %d resamples of items (seed %d) and are too narrow when there "
      "are few items. Treat differences whose interval includes 0 as not shown." % (args.boot, bc.BOOT_SEED))
    w("")
    for note in ps.notes:
        w("- " + note)
    w("")
    w("## Ours reproduces the phone")
    w("")
    sa = [c for _, c in repro["stored"]]
    if sa:
        w("- Stored crops: offline embedding of the saved JPEG against the embedding the phone stored, "
          "cosine median %.4f, minimum %.4f (n = %d: %d tapped records, %d scan entries)." % (
              statistics.median(sa), min(sa), len(sa), sum(k == "rec" for k, _ in repro["stored"]),
              sum(k == "inv" for k, _ in repro["stored"])))
    if repro["emb_cos"]:
        w("- Fresh candidates with a logged embedding (`verify/<vid>_emb.json`): offline embedding of the "
          "rectangle cut from the saved frame against the logged one, cosine median %.4f, minimum %.4f (n = %d)." % (
              statistics.median(repro["emb_cos"]), min(repro["emb_cos"]), len(repro["emb_cos"])))
    a = repro["cand_phone_ref"]
    if a.get("n"):
        w("- Candidate similarity, offline fresh crop against the phone's stored embedding, versus the "
          "logged `scores[].sim`: mean |difference| %.3f, max %.3f, Pearson r %.3f (n = %d candidates)." % (
              a["mean_abs"], a["max_abs"], a["r"], a["n"]))
    a = repro["cand"]
    if a.get("n"):
        w("- Candidate similarity, fully offline (both crops from JPEG, as every method below sees them), "
          "versus logged: mean |difference| %.3f, max %.3f, Pearson r %.3f (n = %d candidates)." % (
              a["mean_abs"], a["max_abs"], a["r"], a["n"]))
    a = repro["best"]
    if a.get("n"):
        k, n = repro["decision"]
        w("- Best candidate per capture, fully offline versus logged `bestSimilarity`: mean |difference| "
          "%.3f, max %.3f, r %.3f (n = %d captures); the 0.75 decision agrees in %d of %d." % (
              a["mean_abs"], a["max_abs"], a["r"], a["n"], k, n))
        w("- Logged → offline best similarity per capture: " + ", ".join(
            "%.3f → %.3f" % (x, y) for x, y in sorted(repro["best_pairs"])) + ".")
    w("")
    w("## Results")
    w("")
    w("AUC: area under the ROC curve (1 = perfect, 0.5 = chance). EER: equal error rate (lower is "
      "better). TAR@5%%: share of same pairs accepted at a threshold that accepts at most 5%% of "
      "different pairs. ΔAUC: the method's AUC minus ours, with a paired interval. A value marked * "
      "rests on fewer than %d pairs of one class; TAR@5%% needs %d different pairs; an interval needs "
      "%d items." % (bc.MIN_PER_CLASS, bc.MIN_NEG_FOR_TAR, bc.MIN_BOOT_ITEMS))
    w("")
    for name, res in results.items():
        desc = bc.REPORTED_SETS[name][0]
        if res is None:
            w("### %s\n\nNo pairs.\n" % desc)
            continue
        c = res["counts"]
        w("### %s" % desc)
        w("")
        w("n: %d same pairs (from %d items), %d different pairs, %d items in all." % (
            c["n_pos"], c["n_pos_items"], c["n_neg"], c["n_items"]))
        if not res["boot_ok"]:
            w(" No intervals: fewer than %d items." % bc.MIN_BOOT_ITEMS)
        w("")
        small = min(c["n_pos"], c["n_neg"]) < bc.MIN_PER_CLASS
        star = "*" if small else ""
        w("| Method | AUC | ΔAUC vs ours | EER | TAR@5% |")
        w("| --- | --- | --- | --- | --- |")
        for m in methods:
            r = res["methods"][m]
            tar = bc.fmt_ci(*r["tar5"]) + star if c["n_neg"] >= bc.MIN_NEG_FOR_TAR else "n too small"
            d = "–" if m == "ours" else bc.fmt_ci(*r["dauc"])
            w("| %s | %s%s | %s | %s%s | %s |" % (METHOD_LABEL[m], bc.fmt_ci(*r["auc"]), star, d,
                                                 bc.fmt_ci(*r["eer"]), star, tar))
        if "ours_at_075" in res:
            tar, far = res["ours_at_075"]
            w("")
            w("Ours at the app's threshold 0.75: true-accept rate %s, false-accept rate %s." % (bc.fmt(tar), bc.fmt(far)))
        w("")
    w("## Cost on this Mac (not phone time)")
    w("")
    w("CPU only (%s), %d torch threads, batch size 1, median over all crops or comparisons of the run "
      "after three warm-up crops. Parameters: stored constants for .tflite files, trainable tensors "
      "for PyTorch; MobileCLIP2-S2 counts the image encoder only (its file holds the text encoder too)." % (
          args.cpu_name, TORCH_THREADS))
    w("")
    w("| Method | Parameters | File | ms per crop | ms per comparison |")
    w("| --- | --- | --- | --- | --- |")
    for m in methods:
        c = costs[m]
        par = "–" if c["params"] in (None, 0) else ("%.2f M" % (c["params"] / 1e6))
        fb = "–" if not c["file_bytes"] else ("%.1f MB" % (c["file_bytes"] / 1e6))
        w("| %s | %s | %s | %s | %s |" % (METHOD_LABEL[m], par, fb, bc.fmt(c["ms_crop"], 1), bc.fmt(c["ms_pair"], 2)))
    w("")
    w("## Changes after the design was fixed")
    w("")
    w(notes_changes)
    w("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", required=True, help="pulled data folder (repeatable)")
    ap.add_argument("--exclude-session", action="append", default=[], help="also leave out this session")
    ap.add_argument("--no-default-exclude", action="store_true", help="keep the developer's device-check sessions")
    ap.add_argument("--methods", default=",".join(METHOD_ORDER), help="comma-separated subset of %s" % METHOD_ORDER)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "bench_match"))
    ap.add_argument("--md", help="also write the Markdown table here (numbers only)")
    ap.add_argument("--boot", type=int, default=bc.BOOT_N)
    ap.add_argument("--changes", default="None.", help="text for the 'changes after the design' section")
    args = ap.parse_args(argv)
    import platform
    import subprocess
    try:
        args.cpu_name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or platform.processor()
    except OSError:
        args.cpu_name = platform.processor()
    excluded = set(args.exclude_session) | (set() if args.no_default_exclude else set(bc.DEFAULT_EXCLUDED_SESSIONS))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHOD_ORDER:
            ap.error("unknown method %s" % m)
    if "ours" in methods:
        methods = ["ours"] + [m for m in methods if m != "ours"]

    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    bd = bc.BenchData(args.data, excluded)
    ps = bc.build_pairs(bd)
    for n in ps.notes:
        log(n)
    log("%d crops, %d pairs" % (len(ps.crops), len(ps.pairs)))

    scores, costs, repro = {}, {}, None
    for m in methods:
        log("scoring %s ..." % m)
        meth = make_method(m)
        s, feats, cost = score_pairs(meth, ps, log)
        scores[m] = s
        cost.update(params=meth.params, file_bytes=meth.file_bytes)
        costs[m] = cost
        if m == "ours":
            repro = reproduction(bd, ps, meth, feats)
            log("reproduction: best %s, decision %s" % (repro["best"], repro["decision"]))
        del meth, feats

    results = {name: evaluate(ps, scores, methods, name, args.boot) for name in bc.REPORTED_SETS}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "pair_scores.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["set", "label", "item_a", "item_b", "dup", "capture", "a", "b"] + methods)
        for i, p in enumerate(ps.pairs):
            wr.writerow([p["set"], p["label"], p["item_a"], p["item_b"], int(p["dup"]), p["capture"] or "",
                         "|".join(p["a"]), p["b"]] + ["%.6f" % scores[m][i] for m in methods])
    md = markdown(bd, ps, results, costs, repro or {"stored": [], "emb_cos": [], "cand": {}, "cand_phone_ref": {},
                                                    "best": {}, "decision": (0, 0), "best_pairs": []},
                  methods, args, args.changes)
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(md)
    if args.md:
        with open(args.md, "w") as f:
            f.write(md)
    print(md)


if __name__ == "__main__":
    main()
