# FindBack v0.5: scan once, find later

Status: plan, 2026-11-16, git branch `scan-once`. `main` stays at v0.4 for the experiments already scheduled.

## 1. Why

v0.4 remembers an item only after the user taps it, names it and, to be sure later, aims the camera at it again. That is too much work per item, and a phone is not always looking: it suits things that are put away and wanted again much later, not things that move every day.

v0.5 removes the per-item work:

1. **Scan.** The user walks through the home once. The app detects and names many items per camera frame, places each in the home map, and merges repeated sightings into one inventory entry.
2. **Find.** The user searches the inventory by text and is guided with the existing route guidance.
3. **Stay honest for free.** Whenever the camera sees a place where an entry should be, the app records whether it was seen again or looked for and not seen. A place that was hidden counts as neither. No extra photos are asked of the user.

Point 3 is the research contribution (see `paper/research/pivot_prior_art.md`): evidence about each entry maintained from chance observations, where absence only counts if the place was actually visible.

## 2. Scope

In scope:

- Scan mode with a pluggable on-device detector, depth-based 3D placement, merging, and a persistent inventory.
- Text search and Find for inventory entries through the existing guidance and verification.
- Visibility-aware evidence, updated during scans and during low-rate passive keyframes.
- A named container as a scan target ("desk drawer"), so items inside containers have a usable location.
- Saved keyframes for offline comparison of detectors, and logging for evaluation.

Out of scope for this version:

- Open-vocabulary detectors that need a custom export (YOLOE, YOLO-World), text-image embedding search, and on-device language models. The detector interface must make them easy to add.
- Any network use. ARCore stays pinned at 1.48.0.
- Changes to tap registration, the v0.4 study modes, the verification pipeline's thresholds and candidate rules, or `VerificationPolicy.decide()`. With scan features unused, the app must behave as v0.4.

## 3. Detectors

`interface SceneDetector { val id: String; fun detect(frame: CameraFrame): List<SceneDetection> }`, where a detection has a rectangle in sensor image pixels, a label and a score.

| Id | Implementation | Notes |
| --- | --- | --- |
| `effdet0` (default) | MediaPipe Tasks `ObjectDetector` with EfficientDet-Lite0 | 80 COCO classes, many objects per frame. `tasks-vision` is already a dependency. Download the model from Google's MediaPipe model storage, check its licence (expected Apache-2.0) and record the URL and licence in the README. |
| `effdet2` | Same with EfficientDet-Lite2 | Slower, more accurate |
| `mlkit` | The existing ML Kit detector, plus ML Kit Image Labeling on each crop for a name | At most 5 objects per frame. Adds the bundled image-labeling dependency. |

Keep detections with score at least 0.35, at most 10 per keyframe. Drop a fixed stop list: `person` always (privacy), and furniture-scale classes (couch, bed, dining table, toilet, tv, refrigerator, oven, sink, chair, bench, potted plant). Drop anything physically wider than 0.6 m. The detector id is a setting and is logged with every scan.

## 4. Keyframes

Taken on the GL thread when all of these hold: tracking; the map is located; no keyframe is being processed; at least 400 ms since the last one; and the camera has moved 0.10 m or turned 8° since the last one (scan mode), or 2 s have passed (passive mode, see section 8). Skip when the camera is turning faster than 60° per second.

At that moment, capture: the CPU image, the 16-bit depth image copied out and closed immediately, `camera.pose`, the image intrinsics, the display rotation, the current map transform, the user's room, and an affine map from CPU image pixels to depth pixels (from `transformCoordinates2d` on three points, since the frame is not available later). If the depth image is not available, skip the keyframe.

Everything else runs off the GL thread.

## 5. From a detection to a map position

For each kept detection:

1. **Depth:** the median of valid depth pixels inside the central 40% of the box. Reject outside 0.2–4.0 m.
2. **Camera-frame point** from the box centre `(u, v)` and depth `d`, with image intrinsics `fx, fy, cx, cy`: `(d·(u − cx)/fx, d·(cy − v)/fy, −d)`. Verify this convention against the ARCore documentation for `Camera.getPose` and the CPU image; do not trust this line blindly.
3. **World point** through `camera.pose`, then **map point** through the inverse map transform.
4. **Size:** box width in pixels × `d / fx`.
5. **Embedding** of the crop with the existing embedder, for re-identification and for the existing verification.

**Self-check, required:** on the first 20 keyframes of every scan, also hit-test the screen point of the image centre on the GL thread and log the distance between the hit point and the point this pipeline computes for the image centre (`scan_depth_check`). The audit uses it to confirm the geometry on the device.

## 6. Inventory

Scan entries live in their own store, `inventory.json`, with an append-only `inventory_log.jsonl`. `records.json` is untouched, so the v0.4 experiments are unaffected.

An entry holds: id; display name (user-editable) and whether the user edited it; label votes (label → summed score) and the winning label; map id, map position, standing position of the best sighting, room, optional container; physical size; the best crop's file, embedding, capture distance, crop size and image size (everything needed to build an `ObjectRecord` for Find); sightings; first and last seen times; detector id; and the evidence fields of section 7.

**Merging** (pure Kotlin, unit-tested). For a new detection, among entries of the same map within `clamp(0.5 × max(sizes) + 0.15, 0.20, 0.50)` metres, take the nearest whose label equals the detection's or whose embedding has cosine similarity of at least 0.55.

- Match: add a sighting, update the position as the mean of the last 10 sightings, add the label vote, update last seen, and replace the best crop and its embedding when the new crop's quality (box area × score) is higher.
- No match: create a tentative entry. It becomes a real entry at its second sighting from a keyframe at least 0.3 s later. Tentative entries left at the end of a scan are dropped and counted.
- Within one keyframe, two detections with the same label and box overlap over 0.6 are one detection.

**Containers.** A scan has a target: "open area" (default) or a named container. Entries created in a container scan carry the container's name, and their location reads "in <container>, <room>".

## 7. Evidence

Pure Kotlin, unit-tested. For each keyframe and each real entry of the current map that no detection matched in this keyframe:

1. Project the entry's position into the image. It is **in view** when it lies inside the image with an 8% margin and 0.3–3.0 m in front of the camera.
2. Read the depth at that pixel (median of a 5×5 patch), `m`; the entry's distance along the optical axis is `z`.
   - `m < z − max(0.15, 0.15·z)`: something is in front. **Hidden**; this keyframe says nothing.
   - Otherwise the place is **observable**.
3. If observable, crop the entry's expected window as verification does and compare with its embedding. Similarity of at least `SEEN_SIMILARITY` (0.60, a named constant to calibrate) counts as an appearance match. Do this for at most 4 entries per keyframe, those checked least recently first.

Per scan (and per passive window of one minute), each entry gets exactly one outcome:

| Outcome | Rule |
| --- | --- |
| `seen` | Matched by a detection in any keyframe, or appearance-matched in any keyframe |
| `missed` | Observable in at least 3 keyframes, never seen |
| `not_observed` | Everything else |

Entries in a container are only eligible during a scan of that container; otherwise they are `not_observed`.

The entry keeps counts of each outcome, the last time of each, and the last outcome. A confirmed verification during Find also counts as `seen`. Nothing is ever deleted or moved automatically.

**Moved candidates** (log only): when a new entry's embedding matches, at 0.70 or more with the same label, an entry whose last outcome is `missed`, log `inv_moved_candidate` and show "may be <name>, last seen in <room>" on the new entry.

## 8. User interface

- **Scan** control, visible when idle and located. Before starting: choose the target (open area or a container name). During a scan: the status hint shows the count ("Scanning: 14 items · walk slowly"), the overlay draws this keyframe's boxes and labels (known entries and new ones in different colours), and the control reads Stop. Registration taps are ignored during a scan.
- **Scan summary** at Stop: new, seen again, missed, not observed, and tentative entries dropped.
- **Items list:** a search field filters by name and labels, case-insensitive. Scanned entries and tapped records appear in one list, each marked with its kind. A scanned entry's second line gives location and evidence, for example "Bedroom · seen 2 d ago", "Kitchen · looked for in the last scan, not seen", "in desk drawer, Office · not checked since Nov 17".
- **Entry actions:** Find, Rename, Delete. Find builds an `ObjectRecord` from the entry and uses the existing query flow unchanged; `query_start` gains `source` (`scan` or `tap`) and `inventoryId`.
- **Settings** (in the study settings dialog): detector, "Save scan frames", "Passive updates" (default on: one keyframe every 2 s outside scan mode while located, evidence only, no new entries).

## 9. Saved frames and logging

With "Save scan frames" on, each keyframe writes into `scan/<scanId>/`: a JPEG at 960 px width, the depth as raw little-endian uint16 with its size, and a JSON file with pose, intrinsics, map transform, the depth mapping, and every detection (box, label, score, depth, map point, kept or the reason it was dropped). This lets any detector be run offline on identical frames.

Events (keep every existing event unchanged):

| Event | Fields |
| --- | --- |
| `scan_start` | scan id, detector, target, map id, study tag |
| `scan_keyframe` | scan id, index, latency by stage, detections, kept, matched, new, evidence checks, share of valid depth |
| `scan_depth_check` | scan id, index, distance between hit test and pipeline, hit type |
| `inv_add`, `inv_rename`, `inv_delete` | entry id and the entry |
| `inv_evidence` | scan id or passive window, entry id, outcome, observable keyframes, best similarity, hidden keyframes |
| `inv_moved_candidate` | new entry, old entry, similarity |
| `scan_end` | duration, metres walked, keyframes, entry counts by outcome, tentative dropped, median latency by stage |

## 10. Code layout, threading and tests

- New package `scan/`: pure Kotlin for geometry (unprojection, projection, depth mapping), merging and evidence, with no Android or ARCore imports; plus `InventoryStore`, the detectors, and a `ScanController` that owns scan state.
- `FindBackRenderer` stays the glue: it takes keyframes and hands them over. One keyframe is in flight at a time. No shared mutable state without synchronization. No dialog may block the GL thread.
- User-typed text is never used as a format string.

Unit tests:

- Geometry: a point placed at a known map position projects and unprojects to itself for several camera poses; the depth mapping; rejection ranges.
- Merging: one object over ten keyframes gives one entry; two same-label objects 60 cm apart give two; a label flip with a matching embedding stays one entry; tentative entries are dropped; the within-keyframe duplicate rule.
- Evidence: hidden, observable, the three outcomes, the cap of 4 appearance checks, the container rule.
- Inventory JSON round trip; entry to `ObjectRecord`.

## 11. Acceptance

1. `./gradlew testDebugUnitTest assembleDebug` passes (78 tests pass before this work).
2. With no scan run and passive updates off, behaviour and logs match v0.4.
3. On the phone, with a person walking: a scan of a room lists its obvious items once each, with sensible names and rooms; `scan_depth_check` distances are a few centimetres; Find on a scanned entry leads to it; after moving one item away and re-scanning, that entry reads "looked for, not seen" while an entry hidden behind something does not.
4. The README describes scan mode, the settings, the files and the events.

## 12. Known risks

- **Depth.** ARCore's depth comes from motion, is weak on plain surfaces and at very short range, and is wrong when the phone is still. Small items and the inside of drawers will suffer; the container label is the fallback.
- **Vocabulary.** 80 classes will name many items wrongly or not at all. The saved frames are there so that better detectors can be compared offline before any is ported.
- **Detector misses look like absence.** The appearance match in section 7 is the guard; its threshold needs calibrating on real data.
- **Load and heat.** Detection, embedding and ARCore share the phone. Keyframe pacing is the control.
