#!/usr/bin/env python3
"""Builds the image package of the VLM naming experiment (analysis/VLM_NAMING_PLAN.md) on the Mac.

    analysis/.venv-bench/bin/python analysis/vlm_prepare.py --data backups/phone_files_2026-11-17_0008

Writes, under --out (default analysis/out/vlm, git-ignored):

  package/items.json        what the workstation needs: item ids, set and image files per input
                            variant. No names, labels, positions or other log content.
  package/images/*.jpg      the stored crops (original bytes when no resizing is needed), their
                            320-px copies, one marked keyframe per scan entry at two sizes, and the
                            crops of every scan detection in saved keyframes (cross-view set)
  local/items_local.json    stays on the Mac: typed names (human reference), phone labels, the
                            chosen keyframe and boxes (for the Track B baselines)

Item sets and inputs are defined in analysis/notes/2026-11-17_vlm_naming_design.md.
"""
import argparse
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import bench_common as bc  # noqa: E402
import fb_load  # noqa: E402
import vlm_common as vc  # noqa: E402

JPEG_QUALITY = 90
BOX_RGB = (255, 0, 0)


def sensor_box_to_upright(box, scale, rotation, sensor_w, sensor_h):
    """Same as bench_detect.sensor_box_to_upright (copied to keep this script light)."""
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


def resized_size(w, h, cap):
    """(w, h) after capping the longer side at `cap` (downscale only)."""
    s = cap / float(max(w, h))
    if s >= 1:
        return w, h
    return max(1, round(w * s)), max(1, round(h * s))


def jpeg_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def crop_variant(path, cap):
    """JPEG bytes of a stored crop for one size: the original file when it is small enough."""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
        nw, nh = resized_size(w, h, cap)
        if (nw, nh) == (w, h):
            with open(path, "rb") as f:
                return f.read(), (w, h), False
        return jpeg_bytes(im.convert("RGB").resize((nw, nh), Image.LANCZOS)), (nw, nh), True


def marked_frame(upright_rgb, box_upright, cap):
    """JPEG bytes of the upright keyframe capped at `cap`, with the box drawn after resizing."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(upright_rgb)
    w, h = im.size
    nw, nh = resized_size(w, h, cap)
    if (nw, nh) != (w, h):
        im = im.resize((nw, nh), Image.LANCZOS)
    s = nw / float(w)
    l, t, r, b = (v * s for v in box_upright)
    l, t = max(0.0, l), max(0.0, t)
    r, b = min(nw - 1.0, r), min(nh - 1.0, b)
    lw = max(2, round(max(nw, nh) / 240.0))
    ImageDraw.Draw(im).rectangle([l, t, r, b], outline=BOX_RGB, width=lw)
    return jpeg_bytes(im), (nw, nh)


def safe(item_id):
    return item_id.replace(":", "_").replace("/", "_")


def typed_names(bd):
    """record id -> the name the user typed (register events with nameSource = typed)."""
    out = {}
    for e in bd.events:
        if e.get("type") == "register" and e.get("nameSource") == "typed" and e.get("record"):
            out[e["record"]] = e.get("userName")
    return out


def all_records(bd):
    """Every record (latest copy) of the non-excluded sessions, from records.json and the log."""
    recs = {}
    for f in bd.ds.folders:
        rows = list(f.records) + [r.get("record") for r in (f.records_log or []) if isinstance(r.get("record"), dict)]
        for r in rows:
            if isinstance(r, dict) and r.get("id") and r.get("sessionId") not in bd.excluded:
                recs.setdefault(r["id"], r)
    return recs


def build(bd, out_dir, log=print):
    pkg = os.path.join(out_dir, "package")
    img_dir = os.path.join(pkg, "images")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "local"), exist_ok=True)
    for f in os.listdir(img_dir):
        if f.endswith(".jpg"):
            os.remove(os.path.join(img_dir, f))
    items, local = [], {}
    deleted = {e.get("record") for e in bd.events if e.get("type") == "delete"}
    typed = typed_names(bd)

    def write(name, data):
        with open(os.path.join(img_dir, name), "wb") as f:
            f.write(data)
        return name

    def add_crops(item_id, path):
        files, meta = {}, {}
        for size in ("full", "320"):
            data, wh, resized = crop_variant(path, vc.SIZE_CAP[size])
            files[size] = write("%s__crop-%s.jpg" % (safe(item_id), size), data)
            meta[size] = {"w": wh[0], "h": wh[1], "bytes": len(data), "resized": resized}
        return files, meta

    # ---- R: tapped records
    for rid, rec in sorted(all_records(bd).items()):
        path = bd.record_crop_path(rec)
        if not path:
            continue
        iid = "rec:" + rid
        files, meta = add_crops(iid, path)
        ref = vc.reference_name(typed.get(rid)) if rid in typed else None
        items.append({"id": iid, "set": "R", "variants": {"crop-full": [files["full"]], "crop-320": [files["320"]]}})
        local[iid] = {"set": "R", "record": rid, "session": rec.get("sessionId"), "stored_name": rec.get("name"),
                      "typed": typed.get(rid), "reference": ref, "deleted": rid in deleted, "images": meta,
                      "crop_source": os.path.relpath(path, os.path.join(HERE, ".."))}

    # ---- S: scan entries, with the best saved keyframe
    image_cache = {}

    def image(path):
        if path not in image_cache:
            image_cache[path] = bc.load_rgb(path)
        return image_cache[path]

    sightings = {}        # entry -> list of (quality, scan_dir, kf, det index, det)
    for scan_dir, kf in bd.keyframes:
        if not bd.keyframe_image_path(scan_dir, kf):
            continue
        for j, det in enumerate(kf.get("detections") or []):
            eid = det.get("entry")
            if det.get("kept") and eid and eid in bd.entries:
                b = det["box"]
                q = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) * float(det.get("score") or 0)
                sightings.setdefault(eid, []).append((q, scan_dir, kf, j, det))

    for eid, ent in sorted(bd.entries.items()):
        path = bd.entry_crop_path(eid)
        if not path:
            continue
        iid = "inv:" + eid
        files, meta = add_crops(iid, path)
        variants = {"crop-full": [files["full"]], "crop-320": [files["320"]]}
        info = {"set": "S", "entry": eid, "session": ent.get("sessionId"), "phone_label": ent.get("label"),
                "phone_votes": ent.get("labelVotes"), "images": meta, "frame": None,
                "crop_source": os.path.relpath(path, os.path.join(HERE, "..")),
                "sightings": [{"kf": "%s/%04d" % (kf.get("scan"), kf.get("index") or 0), "box": det["box"],
                               "label": det.get("label"), "score": det.get("score")}
                              for _, _, kf, _, det in sorted(sightings.get(eid, []), key=lambda x: -x[0])]}
        if sightings.get(eid):
            q, scan_dir, kf, j, det = max(sightings[eid], key=lambda x: (x[0], -(x[2].get("index") or 0)))
            sensor = image(bd.keyframe_image_path(scan_dir, kf))
            imeta = kf.get("image") or {}
            src_w = imeta.get("sourceWidth") or sensor.shape[1] * 2
            src_h = imeta.get("sourceHeight") or sensor.shape[0] * 2
            scale = sensor.shape[1] / float(src_w)
            rot = kf.get("rotation") or 0
            upright = np.ascontiguousarray(bc.sensor_to_upright(sensor, rot))
            rect = bc.clamp_rect(bc.pad_rect(det["box"], bc.CROP_PAD), src_w, src_h)
            box_up = sensor_box_to_upright(rect, scale, rot, src_w, src_h)
            fmeta = {}
            for size in ("full", "320"):
                data, wh = marked_frame(upright, box_up, vc.SIZE_CAP[size])
                fname = write("%s__frame-%s.jpg" % (safe(iid), size), data)
                variants["ctx-" + size] = [files[size], fname]
                fmeta[size] = {"w": wh[0], "h": wh[1], "bytes": len(data)}
            info["frame"] = {"kf": "%s/%04d" % (kf.get("scan"), kf.get("index") or 0), "det": j,
                             "quality": q, "images": fmeta}
        items.append({"id": iid, "set": "S", "variants": variants})
        local[iid] = info

    # ---- V: every kept detection of an entry in a saved keyframe
    for eid in sorted(sightings):
        for q, scan_dir, kf, j, det in sorted(sightings[eid], key=lambda x: (x[2].get("index") or 0, x[3])):
            sensor = image(bd.keyframe_image_path(scan_dir, kf))
            imeta = kf.get("image") or {}
            src_w = imeta.get("sourceWidth") or sensor.shape[1] * 2
            src_h = imeta.get("sourceHeight") or sensor.shape[0] * 2
            scale = sensor.shape[1] / float(src_w)
            rect = bc.clamp_rect(bc.pad_rect(det["box"], bc.CROP_PAD), src_w, src_h)
            crop = bc.crop_from_sensor(sensor, rect, kf.get("rotation") or 0, scale)
            from PIL import Image
            data = jpeg_bytes(Image.fromarray(crop))
            iid = "det:%s:%04d:%d" % (kf.get("scan"), kf.get("index") or 0, j)
            fname = write("%s__crop-full.jpg" % safe(iid), data)
            items.append({"id": iid, "set": "V", "variants": {"crop-full": [fname]}})
            local[iid] = {"set": "V", "entry": eid, "kf": "%s/%04d" % (kf.get("scan"), kf.get("index") or 0),
                          "det": j, "label": det.get("label"), "w": int(crop.shape[1]), "h": int(crop.shape[0]),
                          "bytes": len(data)}

    manifest = {"version": 1, "n_items": len(items), "variants": vc.VARIANTS, "items": items}
    with open(os.path.join(pkg, "items.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(out_dir, "local", "items_local.json"), "w") as f:
        json.dump({"data": [os.path.relpath(p, os.path.join(HERE, "..")) for p in bd.paths],
                   "excluded": sorted(bd.excluded), "items": local}, f, indent=1)
    sets = {}
    for it in items:
        sets[it["set"]] = sets.get(it["set"], 0) + 1
    n_frame = sum(1 for it in items if it["set"] == "S" and "ctx-full" in it["variants"])
    n_ref = sum(1 for v in local.values() if v.get("reference"))
    total = sum(os.path.getsize(os.path.join(img_dir, f)) for f in os.listdir(img_dir))
    log("items: %s; S with a saved keyframe: %d; R with a typed name: %d; %d images, %.1f MB" % (
        sets, n_frame, n_ref, len(os.listdir(img_dir)), total / 1e6))
    return manifest, local


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", required=True, help="pulled data folder (repeatable)")
    ap.add_argument("--exclude-session", action="append", default=[])
    ap.add_argument("--no-default-exclude", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "vlm"))
    args = ap.parse_args(argv)
    excluded = set(args.exclude_session) | (set() if args.no_default_exclude else set(bc.DEFAULT_EXCLUDED_SESSIONS))
    bd = bc.BenchData(args.data, excluded)
    build(bd, args.out)


if __name__ == "__main__":
    main()
