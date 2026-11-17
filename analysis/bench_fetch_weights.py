#!/usr/bin/env python3
"""Downloads the model weights of the offline baselines into analysis/.bench-cache/ (once).

    analysis/.venv-bench/bin/python analysis/bench_fetch_weights.py

Only official sources are contacted (Google's MediaPipe model storage, Ultralytics' GitHub
releases, Hugging Face, the LightGlue GitHub releases); no data of the project is sent. Files with
a known checksum are verified. Hugging Face snapshots are pinned to the commits in
bench_match.HF_REVISIONS. The Ultralytics settings in analysis/.bench-cache/ultralytics_cfg are
written with usage analytics ("sync") off, so the global Ultralytics settings are not touched.
"""
import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".bench-cache")
MODELS = os.path.join(CACHE, "models")
os.environ.setdefault("HF_HOME", os.path.join(CACHE, "hf"))
os.environ.setdefault("TORCH_HOME", os.path.join(CACHE, "torch"))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(CACHE, "ultralytics_cfg"))
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

MP = "https://storage.googleapis.com/mediapipe-models/"
UL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"
FILES = [  # (local name, URL, sha256)
    ("mobilenet_v3_large_float32_1.tflite", MP + "image_embedder/mobilenet_v3_large/float32/1/mobilenet_v3_large.tflite",
     "11af3c560dfeed7737cb4c03c23bf52a8403020784192d4dea0b74862a12828d"),
    ("efficientdet_lite2_int8_1.tflite", MP + "object_detector/efficientdet_lite2/int8/1/efficientdet_lite2.tflite",
     "b3f50554cb0ea559e90328845f7d9ba4d13c8bff372914d24e06bc8bb72fa896"),
    ("yolo26n.pt", UL + "yolo26n.pt", "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"),
    ("yoloe-11s-seg.pt", UL + "yoloe-11s-seg.pt", "8e439445c87338b79d9ce21dec109f4621e26df67e94d26ea1a98c1e64dce3e3"),
    ("mobileclip_blt.ts", UL + "mobileclip_blt.ts", "a67804d1b0f07b8b9a20c1761ec0847f34660f5fa338ec70e8f3fce68ed95e54"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(MODELS, exist_ok=True)
    for name, url, digest in FILES:
        path = os.path.join(MODELS, name)
        if not os.path.exists(path):
            print("downloading", url)
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(req) as r, open(path + ".part", "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            os.replace(path + ".part", path)
        got = sha256(path)
        print("%-40s %s" % (name, "ok" if got == digest else "CHECKSUM MISMATCH " + got))
        if got != digest:
            sys.exit(1)

    from ultralytics.utils import SETTINGS
    SETTINGS.update({"sync": False, "clearml": False, "comet": False, "dvc": False, "mlflow": False,
                     "raytune": False, "weights_dir": MODELS, "runs_dir": os.path.join(CACHE, "runs"),
                     "datasets_dir": os.path.join(CACHE, "datasets")})

    sys.path.insert(0, HERE)
    from bench_match import HF_REVISIONS
    from huggingface_hub import snapshot_download
    patterns = {"timm/MobileCLIP2-S2-OpenCLIP": ["open_clip_model.safetensors", "*.json", "*.txt", "README.md"]}
    for repo, rev in HF_REVISIONS.items():
        p = snapshot_download(repo, revision=rev, allow_patterns=patterns.get(repo, ["*.json", "*.safetensors", "*.txt", "*.md"]))
        print("%-40s %s" % (repo, p))

    # LightGlue and its SuperPoint weights come from the LightGlue GitHub release (torch hub cache).
    from lightglue import LightGlue, SuperPoint
    SuperPoint(max_num_keypoints=1024)
    LightGlue(features="superpoint")
    print("LightGlue + SuperPoint weights in", os.path.join(os.environ["TORCH_HOME"], "hub", "checkpoints"))


if __name__ == "__main__":
    main()
