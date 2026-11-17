# Baselines and state-of-the-art comparisons

Status: plan, 2026-11-17. High-level only; the implementer chooses the details and records them.

## Why and how

The first scan walk (`analysis/notes/2026-11-16_first_scan_walk.md`) showed where FindBack is weak:
appearance matching rejects items that are present, and the 80-class detector names about half of
the entries wrongly. To claim anything, each component needs a comparison against simple baselines
and against current strong models.

Whole published systems cannot be rebuilt in a week. The comparison is therefore **per component,
offline, on the frames the phone already saved**, so every method sees identical input. The phone
app is not changed by this work.

## Track A: appearance matching (first priority)

Question: given a remembered photo of an item and a fresh view, is it the same item?

| Role | Method |
| --- | --- |
| Ours | MobileNetV3-small embedding (the app's `.tflite`), cosine similarity |
| Simple baselines | Colour histogram; ORB keypoints with a geometric check |
| Same family, larger | One larger mobile classifier embedding (for example MobileNetV3-large or EfficientNet-Lite) |
| Strong models | DINOv2 (small); one CLIP-family image encoder with a mobile variant (for example MobileCLIP or SigLIP); optionally a learned local-feature matcher (SuperPoint + LightGlue) |

Pairs come from our own logs, defined before any model is scored:

1. **Verify pairs**: stored crop against the fresh crop at the marker, labelled by the query's
   ground truth (present = same; absent or replaced = different).
2. **Scan pairs**: crops of detections that the scan merged into one entry (same) and crops of
   different entries (different), from the saved keyframes. Report with and without pairs that may
   be duplicates of one object (same label within 0.5 m).
3. **Cross pairs**: scan crop against verify crop of the same entry, the hard case.

Report threshold-free quality (ROC AUC, equal error rate, true-accept rate at 5% false accepts)
with bootstrap intervals resampled by item, plus model size and per-crop time on the Mac (stated as
not phone time). First check that the offline "ours" reproduces the similarities the phone logged.

## Track B: detection and naming (second priority)

Question: on the saved scan keyframes, how many items does a detector find and are the names right?

| Role | Method |
| --- | --- |
| Ours | EfficientDet-Lite0, 80 classes |
| Baselines | EfficientDet-Lite2; one current small closed-set detector (YOLO nano class) |
| Strong models | One or two open-vocabulary detectors (for example YOLO-World or YOLOE, OWLv2 or Grounding DINO), prompted with a fixed household vocabulary written down before running |

Without labels, report only label-free numbers: detections per frame, distinct names used, score
distributions, time per frame. For naming precision, build a local judging sheet (crop plus proposed
name, methods shuffled and hidden) that a person fills in; a script reads the judgments back.
Pooled judgments across methods give relative recall. No model judges another model.

## Track C: system-level baselines (already specified elsewhere)

Memory-only, always-verify, age-only and selective verification (`FindBack/docs/STUDY_SPEC_v0.4.md`);
for scans, "not detected means missing" against the visibility-aware rule
(`FindBack/docs/PLAN_v0.5_scan_once.md`). These need the tagged study sessions and a second scan,
so they are not part of this work.

## Rules

- Everything runs locally. No image, crop or log leaves the Mac; no cloud API. Model weights may be
  downloaded from their official sources; record source, version, licence and size for each.
- A dedicated virtual environment (git-ignored); do not change the system Python.
- No invented numbers: every figure comes from a script run on real logs, and each table states
  its sample sizes. With today's data the samples are small; say so next to the numbers.
- Photos stay out of git (`analysis/out/` is ignored). Number-only tables may go in
  `analysis/notes/`.
- Methods, pair definitions, vocabulary and thresholds are fixed before looking at results; any
  later change is listed.
