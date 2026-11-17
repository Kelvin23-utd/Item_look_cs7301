# Baselines: fixed design (Tracks A and B)

Written 2026-11-17, before any baseline or strong model was scored on FindBack data. It fixes the
details that `analysis/BASELINES_PLAN.md` leaves to the implementer. Only one thing had been run on
the data when this was written: the app's own embedder on the stored crops, to check that the
offline "ours" reproduces the stored embeddings (cosine 0.983 to 0.999 over 20 record crops and 24
inventory crops). Any later change is listed at the end, with the reason and whether results had
been seen.

## Data and selection

- Input: one or more pulled data folders (`--data DIR`, repeatable). Events are merged and
  de-duplicated with `fb_load.Dataset`; a file is looked up in the folders that hold its session.
- Sessions left out by default: the developer's own device checks marked "Exclude" in the session
  log of `analysis/README.md` (cab39b87, 1debdfe5, 2b883ce9, 5a4b83e9, 51e9efc9, 007cc164,
  7a7718f6, c5b9a6ae, 735b0f23, 1925c109, acbda081, c8990ea7). `--exclude-session` adds more,
  `--no-default-exclude` keeps them. Untagged sessions are used (there are no tagged ones yet), so
  none of these numbers is a study number.
- Crops are rebuilt from the saved images exactly as the app cuts them: a rectangle in sensor
  pixels, clamped, cut, turned upright. Every method sees the same upright RGB crops decoded from
  the same JPEG files. Nothing uses the embeddings logged by the phone except the reproduction check.

## Track A: pair sets

An *item* is a tapped record's lineage or a scan inventory entry.

1. **Verify pairs (plan definition).** One pair per capture (`verify` event): the item's stored
   crop (`crops/<record>.jpg`, or `inventory/<entry>.jpg` for a scanned entry) against the fresh
   capture. Label from the query's `groundTruth`: `present` = same; `absent` or `replaced` =
   different; `displaced`, `skip` and unanswered queries are left out (the item is near the marker
   but not at it, so the label of the fresh crop is unclear).
2. **Verify pairs with impostors (added, not in the plan).** Set 1, plus for every capture the
   fresh capture against the stored crop of every *other* item known at the capture time (records
   active then by `RecordHistory`, other lineages only; inventory entries first seen before it),
   labelled different. Reason: set 1 has a single "different" pair today. An impostor that may be
   a duplicate of the target (an inventory entry with the same label within 0.5 m) is left out.
3. **Scan pairs (plan definition).** Every kept detection, in a saved keyframe, that belongs to a
   real inventory entry (in `inventory.json` or an `inv_add` event; tentative entries dropped at
   the end of a scan are left out). Crop: the detection box grown by 8% per side, as the scan does,
   cut from the half-size keyframe JPEG. Same = two detections of one entry; different = detections
   of two entries. Variants: **all**; **no possible duplicates** (different pairs whose two entries
   have the same winning label and map positions within 0.5 m are dropped). Detections in keyframes
   with a person are not saved and so are missing.
4. **Cross pairs (plan definition).** For every capture of a query on a scanned entry E: the fresh
   capture against each scan detection crop of E (same when the ground truth is `present`,
   different when `absent` or `replaced`), and against each scan detection crop of every other
   entry (different). Variants as in set 3 (all; without other entries that may be duplicates of E).

**The fresh capture** is not the phone's saved best crop, which the phone chose with its own
embedder. Each method scores every candidate rectangle the phone logged for that capture (the
projected window and the detector boxes, `verify.scores[].rect`), cut from the saved upright frame,
and takes the maximum, as the app does.

## Track A: methods

| Key | Method | Score of a pair |
| --- | --- | --- |
| `ours` | App embedder: MediaPipe Image Embedder, MobileNet-V3 small, the app's own `.tflite`, L2-normalised | cosine |
| `hist` | HSV colour histogram, 8 × 8 × 8 bins over the whole crop (OpenCV ranges), L1-normalised | histogram intersection |
| `orb` | ORB, 1000 features, on the grey crop resized to a longer side of 480 px; brute-force Hamming, 2-NN ratio test 0.8; RANSAC homography, 5 px | inlier count (0 with fewer than 4 matches) |
| `mnv3l` | MediaPipe Image Embedder, MobileNet-V3 large (same family, larger), float32 v1 | cosine |
| `dinov2s` | DINOv2 ViT-S/14 (`facebook/dinov2-small`), CLS token after the final layer norm | cosine |
| `mobileclip2s2` | MobileCLIP2-S2 image encoder (OpenCLIP `MobileCLIP2-S2`, `dfndr2b`) | cosine |
| `splg` | SuperPoint (at most 1024 keypoints) + LightGlue, same resize as `orb`; RANSAC homography, 5 px | inlier count (0 with fewer than 4 matches) |

Deep models get the whole crop resized to their square input without cropping (a centre crop
would cut elongated objects) and their own published normalisation. All run on the Mac CPU.

**Metrics** (no threshold to tune): ROC AUC (ties count one half), equal error rate (interpolated),
true-accept rate at a false-accept rate of at most 5% (the largest TAR among thresholds whose FAR
is at most 0.05). For `ours` also TAR and FAR at the app's fixed 0.75. Bootstrap: 2000 resamples of
items with replacement, percentile 95% interval, seed 20261117. A pair of items a and b is counted
c_a times when a = b and c_a × c_b times otherwise, where c is how often the item was drawn. The
difference of each method's AUC from `ours` gets its own paired interval (same resamples).
Minimum sizes: an interval needs 4 items; values resting on fewer than 10 pairs of a class are
flagged; TAR at 5% FAR needs 20 different pairs. Cost: parameters, file size, median time per crop
(embedders) or per pair (matchers) on the Mac CPU, never phone time.

**Ours reproduces the phone**: (a) cosine between the offline embedding of each stored crop JPEG
and the stored embedding; (b) for every logged candidate, offline cosine (fresh rectangle from the
saved frame against the offline embedding of the stored crop) against the logged `sim`; (c) the
same for the best candidate against `bestSimilarity`, and agreement of the 0.75 decision.

## Track B: detection and naming

Frames: every saved scan keyframe (frames with a person are not saved). Input: the saved
960 × 540 sensor-orientation JPEG turned upright (540 × 960), which is the half-size upright image
the phone's detector saw; boxes are mapped back to full sensor pixels.

| Key | Detector | Operating point |
| --- | --- | --- |
| `effdet0` (ours) | EfficientDet-Lite0 int8, the app's `.tflite`, MediaPipe Object Detector, raw threshold 0.20, at most 25 | score 0.35 (app) |
| `effdet2` | EfficientDet-Lite2 int8 v1 from MediaPipe model storage, same settings | 0.35 |
| `yolo26n` | Ultralytics YOLO26n (COCO 80), 640 px | 0.25 (Ultralytics default) |
| `yoloe11s` | YOLOE-11S-seg, text prompts = `analysis/bench_vocab_household.txt` (plain names), 640 px | 0.25 (Ultralytics default) |
| `owlv2` | OWLv2 base patch16 ensemble, queries "a photo of a <name>" from the same file, per-name NMS at IoU 0.5 | 0.10 (Hugging Face post-processing default) |

Every detector's output then goes through the app's own keyframe filter (`DetectionFilter`):
person and stop list dropped (COCO stop list; for the open vocabulary its `[stop]` section),
same-name boxes with IoU over 0.6 merged, depth from the saved depth image (median of the central
40% of the box) within 0.2–4.0 m, longer side at most 0.6 m, at most 10 per frame by score.

Label-free numbers per detector: frames, raw detections at the operating point per frame, kept
detections per frame, frames with none kept, distinct names kept, kept score quartiles, median time
per frame on the Mac CPU. For `effdet0`, agreement with the detections the phone logged in the same
keyframes (boxes matched at IoU 0.5 with the same label; score difference).

**Judging sheet** (`analysis/out/judge/judge_sheet.html`, local only). Per keyframe, the kept boxes
of all detectors are pooled and grouped greedily (IoU at least 0.5, highest score first) into
*objects*. The sheet shows each object once (the keyframe with the group's highest-score box drawn,
and that box enlarged) with the distinct names proposed for it in random order and no detector named
anywhere in the file. The judge answers: Q1 is it one item a person could look for (item / part or
several / not an item / can't tell); Q2 per name: right / close / wrong. Keyframes come in a fixed
random order (seed 20261117); judgments are kept in the browser and exported as a JSON file. The key
that maps objects and names back to detectors is a separate file.

`bench_judge_read.py` uses only keyframes in which every object has Q1 answered and, for item
objects, every name judged. Per detector: **item precision** = its kept boxes in objects judged
"item", over its kept boxes; **naming precision** = its boxes in item objects whose name is right
(strict) or right or close (lenient), over its boxes in item objects; **pooled relative recall** =
item objects where it has a box with an acceptable (lenient) name, over item objects where any
detector has one. Objects are counted per keyframe, so one physical object seen in five keyframes
counts five times; this is relative recall, not recall of the room.

## Changes after this was written

1. **MobileCLIP2-S2 is reparameterised before use** (`timm.utils.reparameterize_model`, the
   inference step of Apple's MobileCLIP README). Made after the first full Track A run, whose
   results had been seen, because that run timed the train-time multi-branch network (287 ms per
   crop against 233 ms folded). Embeddings are unchanged (cosine 1.000 on test crops), so every
   quality number is the same; only the cost row changes.
2. **Not a design change, listed for completeness.** The first Track B runs stopped at YOLOE (the
   Ultralytics CLIP tokenizer package was missing from the environment) and then dropped OWLv2 (a
   post-processing method moved in transformers 5). Both were fixed and Track B was rerun with
   identical settings. The label-free numbers of the first three detectors had been seen by then;
   nothing was changed in response.

## Caveats found while running (no change made)

- **Scan "same" labels come from the app's own merger**, which joins a detection to the nearest
  entry with the same label, or with app-embedder similarity of at least 0.55. One entry (a spoon)
  holds a detection labelled "bowl", so its pairs were merged by "ours" itself. Within the cluster
  of three bottle entries less than 0.11 m apart, the merger may also have mixed bottles: the
  lowest-scoring "same" pairs for every embedder are bottle pairs.
- **Easy pairs dominate.** Scan "same" pairs mostly come from keyframes seconds apart, and the
  impostors of set 2 are mostly objects of other kinds, so every AUC is high. The hard cases
  (cross pairs, verify pairs) have 2 and 9 items with a "same" pair.
- **Possible mislabelled impostors.** The highest-scoring "different" pairs of set 2 under all
  four embedders are the scanned laptop against tapped record `1b328d30` (a large crop from the
  first pilot) and two pairs of tapped records from the first pilot. They may be one object
  registered twice; a person should look before these pairs are trusted.
- **Missing scan detections.** 31 of the 105 kept detections of real entries are in keyframes not
  saved because a person was detected (37 of 110 keyframes); the queried laptop entry keeps only
  one saved detection.
- **Session 1debdfe5 is left out** (the session log marks it as a developer test with the phone
  lying still), so the verify set has 12 "present" captures, where the first-walk note counted 13.
