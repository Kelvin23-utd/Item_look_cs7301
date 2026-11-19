# FindBack v0.8: study logging

Status: implemented 2026-11-19 on branch `scan-once`; unit-tested and built, **not yet run on the phone**. Everything below that depends on the device (file writes, timings, person detection on verification frames, the settings checkbox) is unverified until the first device run; see section 9.

## 1. Why

The pilot asks whether the same physical item can be recognised across scans made on different days. Until v0.7 the app kept, per inventory entry, one crop and one embedding that a later detection could overwrite, and it threw away the embedding of every other detection. A fair offline replay of other matching policies needs the raw observations, kept independently of what the app's own matcher decided.

v0.8 therefore adds, without changing the matcher:

1. an **observation log**: one line per kept detection of a scan, with its embedding, geometry, camera pose and, kept apart, what the baseline matcher did with it;
2. a few **full-resolution views** per place, chosen independently of the matcher for spread of viewing direction, never touching a person;
3. per entry, a **confirmed** state that only the creating scan or the user can set, and a **history** of what each scan said;
4. an **extra view** condition built on the existing Find → Verify flow, whose "present" answer after a capture at or above the threshold is the user's explicit confirmation.

The matcher, thresholds, detector, embedder and merge rules are unchanged. With study logging on or off the matcher's decisions, the entries' existing fields and the existing events are identical for the same inputs (section 8, tested).

## 2. The setting

Menu ⋮ → Settings → **Study logging**, default **on**, stored in preferences (`studyLogging`).

- Read when a **scan** starts (`ActiveScan.studyLogging`) and when a **query** starts (`Query.studyLogging`); a change applies from the next scan or search.
- Logged as `scan_start.studyLogging` and `verify.studyLogging`; a change in the dialog logs `ui_action` `dialog:settings` with detail `study_logging_on` or `study_logging_off` (v0.7 trace on).
- **Off** means: no `obs/` lines, no `views/` files, no `history` items, no `confirmed` set or changed, no `study` field on `scan_keyframe`, no `study_extra_view` or `inv_confirm` event, no `historyItem` on `evidence` lines. Entries then serialise exactly as in v0.7 (the new keys are absent). The only differences from v0.7 are the new keys `studyLogging`, `studySchema` and `study` (null) on `scan_start`/`scan_end` and the v0.8 keys on `verify` (section 6), which are logged in both states.

App version: `versionName` 0.8 (`session_start.appVersion`).

## 3. Files

All paths are relative to the files root (`/sdcard/Android/data/edu.cs6301.findback/files`). All files are append-only or write-once; nothing of an earlier scan is ever changed.

| Path | What |
| --- | --- |
| `obs/<scanId>.jsonl` | One scan's lines: `obs` (one per kept detection), `view_evicted`, and a final `scan_end` |
| `views/<scanId>/<obsId>.jpg` | Retained view crops of that scan |
| `obs/extra_views.jsonl` | Every extra-view line, all sessions, appended |
| `views/extra/<obsId>.jpg` | Extra-view crops |

Common conventions:

- One JSON object per line, UTF-8, `\n` terminated. A line is written in one append per keyframe; if the app is killed during a write the **last line can be truncated**: a reader must skip a final line that does not parse.
- Every line has `v` (schema version, int, **1**) and `type`.
- Times are epoch milliseconds (`System.currentTimeMillis()` on the phone), the same clock as `events.jsonl` `t`.
- Positions are metres. **World frame**: ARCore's world frame of that app session (changes every session). **Map frame**: the app's gravity-aligned home map (v0.3), stable across sessions; +Y up in both. `mapToWorld` (T_WM) is `{x, y, z, yawDeg}`: a rotation by `yawDeg` about +Y (right-handed, +90° turns map +X into world −Z) followed by the translation; world = R(yaw)·map + t.
- Image pixels are **CPU image (sensor) pixels**, the frame ARCore calls `IMAGE_PIXELS`: u right, v down in sensor readout order, width `image.w`, height `image.h` (1920×1080 on the Pixel 8). Boxes are `[left, top, right, bottom]`, float32 values written as JSON numbers.
- Embeddings: `emb` is the float32 vector as little-endian bytes in standard base64 with padding (as `verify/<vid>_emb.json`): `np.frombuffer(base64.b64decode(s), dtype="<f4")`, length `embDim` (1024 for `mobilenet_v3_small`).
- Ids: scan ids, session ids and query ids are 8 hex characters. Observation ids: `<scanId>-<kf>-<det>` for scan observations, `<queryId>-x<attempt>` for extra views.

### 3.1 `obs` line, `kind: "scan"`

One per detection that passed the scan filter (`DetectionReport.kept`), in keyframe order, and within a keyframe in `det` order. This includes detections whose entry stays tentative and is dropped at scan end.

| Field | Type | Meaning |
| --- | --- | --- |
| `v` | int | 1 |
| `type` | string | `"obs"` |
| `obs` | string | `<scanId>-<kf>-<det>` |
| `kind` | string | `"scan"` |
| `scan` | string | scan id (`scan_start.scan`) |
| `session` | string | app session id |
| `mapId` | string | map of the scan |
| `room` | string or null | user's room at capture (`KeyframeInfo.roomId`) |
| `kf` | int | keyframe index (`scan_keyframe.index`) |
| `det` | int | index among this keyframe's kept detections, 0 … `scan_keyframe.kept` − 1, in the detector's raw order |
| `rawDet` | int | index in the detector's raw output of this keyframe. Index in the saved keyframe JSON (`scan/<scanId>/kfNNNN.json` `detections`, which omits persons) = `rawDet` − number of raw `person` detections with a smaller raw index |
| `t` | int | capture time of the keyframe (same as the saved keyframe JSON `t`) |
| `studyTag` | string | study tag at scan start |
| `label` | string | detector label, trimmed and lower-cased |
| `score` | number | detector score (float32) |
| `box` | [4] numbers | detector box, sensor pixels, unpadded |
| `depthM` | number | median depth of the valid depth pixels in the central 40 % of the box, metres along the optical axis |
| `sizeM` | number | longer box side (px) × `depthM` / fx |
| `distanceM` | number | straight distance camera → `worldPoint` |
| `mapPoint` | [3] numbers | the detection's point (box centre at `depthM`) in the map frame |
| `worldPoint` | [3] numbers | the same point in the world frame |
| `quality` | number | box area (px²) × score, float32: what the baseline uses to pick the best crop |
| `image` | `{w, h}` | CPU image size, pixels |
| `rotation` | int | sensor-to-display rotation, degrees clockwise (0, 90, 180, 270); the view JPEG is turned by it |
| `camera` | object | see below |
| `mapToWorld` | `{x, y, z, yawDeg}` | T_WM at capture |
| `intrinsics` | `{fx, fy, cx, cy, width, height}` | `Camera.getImageIntrinsics()` scaled to `image` |
| `emb` | string | base64 float32 LE embedding of the padded crop (the very vector the matcher used) |
| `embDim` | int | number of floats |
| `embModel` | string | embedding model name |
| `personInFrame` | bool | the keyframe has at least one raw `person` detection with score ≥ 0.20 |
| `personOverlap` | number, 0–1 | largest share of `viewRect`'s area inside one person box padded by 10 %; 0 when none touches it |
| `viewGroup` | int | retention group (section 4), 0-based in order of creation within the scan |
| `viewRect` | [4] numbers | the crop rectangle: `box` padded by 8 % of its width and height on each side, clamped to the image |
| `view` | string or null | path of the view JPEG relative to the files root (`views/<scanId>/<obs>.jpg`), or null |
| `viewPx` | [2] ints or null | width and height of that JPEG (upright), or null |
| `viewReason` | string | `kept`, or why there is no view: `person`, `not_complementary`, `scan_cap`, `no_crop`, `write_error` |
| `viewReplaces` | string or null | obs id of the view this one evicted |
| `decision` | object | the baseline matcher, below |

`camera`:

```json
{ "world": { "t": [x, y, z], "q": [qx, qy, qz, qw] },
  "map":   { "t": [x, y, z], "q": [qx, qy, qz, qw], "forward": [x, y, z], "yawDeg": 12.3, "pitchDeg": -20.1 } }
```

- `world`: `Camera.getPose()` (the physical camera; camera frame +X right, +Y up, −Z the viewing direction, relative to the sensor image readout).
- `map`: the same pose in the map frame: `t` = T_WM⁻¹·t_world; `q` = q_yaw(−yawDeg) ⊗ q_world (Hamilton product, x y z w order), so `CameraPose(map.t, map.q).toWorld(p_cam)` gives map points; `forward` = the camera's −Z axis in the map frame (unit); `yawDeg` = degrees(atan2(−forward.x, −forward.z)) (0 = looking along map −Z, +90 = along map −X; the `pose.csv` `myaw` convention), null when looking straight up or down; `pitchDeg` = degrees(asin(forward.y)), + up. `map` is null only when T_WM is unknown (never for scan lines).

`decision` (what the baseline did; never used to decide retention):

| Field | Type | Meaning |
| --- | --- | --- |
| `matcher` | string | `"baseline"` |
| `kind` | string | `new` (no match; a tentative entry was created, `entry` is its id), `tentative` (a later sighting of a tentative entry less than 300 ms after its first), `promoted` (that sighting made the tentative entry real), `known` (matched an existing real entry) |
| `entry` | string | the entry id (tentative or real) |
| `sim` | number or null | cosine similarity to that entry's embedding before this sighting; null for `new` |
| `distM` | number or null | distance to that entry's position before this sighting; null for `new` |
| `byLabel` | bool or null | the match qualified by equal label (`Match.byLabel`); null for `new` |
| `candidates` | array | up to 3 entries the matcher looked at, nearest first: `{entry, distM, sim, sameLabel, withinRadius, tentative}` |

The matcher takes a keyframe's detections in **descending score order** (stable, so ties keep `det` order); an entry already taken by a higher-scoring detection of the same keyframe is not a candidate. `candidates` are computed from exactly the list `Merger.match` sees for that detection: real entries of the same map and target plus the scan's tentative entries, minus those already taken. `withinRadius` is `distM` ≤ clamp(0.5·max(entry size, detection size) + 0.15, 0.20, 0.50). The match is the nearest candidate (over all, not only these three) with `withinRadius` and (`sameLabel` or `sim` ≥ 0.55).

### 3.2 `view_evicted` line

Written right after the `obs` line whose view replaced it, in the same append.

| Field | Type | Meaning |
| --- | --- | --- |
| `v`, `type` | | 1, `"view_evicted"` |
| `scan` | string | scan id |
| `obs` | string | the obs id whose view was deleted |
| `view` | string | the deleted file's path |
| `by` | string | the obs id that replaced it (its `viewReplaces` is `obs`) |
| `viewGroup` | int | the group |
| `t` | int | when the line was built |

Final state: the retained views of a scan are the `obs` lines with `view` ≠ null, minus the `obs` ids of its `view_evicted` lines. Earlier lines are never rewritten.

### 3.3 `scan_end` line

The last line of `obs/<scanId>.jsonl`, written when the scan ends (missing when the app was killed mid-scan).

| Field | Type | Meaning |
| --- | --- | --- |
| `v`, `type` | | 1, `"scan_end"` |
| `scan`, `session`, `t`, `reason` | | as `scan_end` event (`reason`: `stop`, `paused`, `closed`, `model_unavailable`) |
| `keyframes` | int | keyframes processed |
| `observations` | int | `obs` lines written |
| `views` | object | `{kept, written, evicted, skipped: {<viewReason>: n}, groups, bytes}`: kept = retained at the end (≤ 400), written = view files written, evicted = `view_evicted` lines, skipped = `obs` lines without a view by reason, groups = retention groups, bytes = bytes of retained view files |
| `retained` | [string] | obs ids of the retained views |
| `created` | [string] | entry ids this scan created (promoted); the order is not that of the `add` lines (seen on the first device run), so compare as a set |
| `tentativeDropped` | [{entry, obs: [string]}] | tentative entries dropped at the end, with the obs ids linked to each |
| `error` | string or null | `finish_failed` when the session's end threw |

### 3.4 `obs` line, `kind: "extra_view"` (`obs/extra_views.jsonl`)

One per verification capture of a **scanned entry** (a query with `source: "scan"`) made with study logging on. The same fields as 3.1, with these differences:

| Field | Value |
| --- | --- |
| `obs` | `<queryId>-x<attempt>` |
| `kind` | `"extra_view"` |
| `scan`, `kf`, `det`, `rawDet`, `label`, `score`, `depthM`, `sizeM`, `mapPoint`, `worldPoint`, `quality`, `viewGroup`, `viewReplaces`, `decision` | null |
| `room` | the user's room at capture |
| `mapId` | the current map at capture |
| `t` | capture time (the GL frame of the Verify press), = `verify.captureTMs` |
| `box`, `viewRect` | the winning verification candidate's rectangle (sensor pixels, clamped): for `verify.bestKind` `detector` the ML Kit box padded by 8 %, for `projected` the projected window. It is exactly the crop that was embedded |
| `distanceM` | camera to the remembered world position (= `verify.distance`) |
| `emb` | the embedding of that crop (= the `candidates[i].embedding` of `verify/<vid>_emb.json` for the winning candidate) |
| `personInFrame`, `personOverlap` | from the scene detector (`effdet0`) run on the whole verification frame; null when it could not run |
| `viewReason` | `kept`, `person`, `person_check_failed` (detector unavailable: no crop is ever written then), `write_error` |
| `view` | `views/extra/<obs>.jpg` or null |

Added fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `entry` | string | the queried inventory entry (`query_start.inventoryId`) |
| `record` | string | the query's record id (same as `entry` for scanned entries) |
| `query`, `attempt`, `captureMode` | | as the `verify` event |
| `requestStartMs` | int | query start (`Query.startMs`, taken on the GL thread when Find was handled) |
| `pressMs` | int or null | the last Verify press before this capture (UI thread) |
| `msSinceQueryStart`, `msSinceNear` | int, int or null | as the `verify` event (monotonic clock) |
| `verify` | object | `{sim, outcome, threshold, bestKind, positionSource}` of this capture |
| `target` | object | `{mapPoint, worldPoint}`: the remembered position the capture was aimed at (map: the entry's `mapPosition` at query start; world: where it was projected from) |

## 4. View retention (rule 2)

Pure Kotlin in `scan/StudyLog.kt` (`ViewSelector`, `ViewRetention`), unit-tested in `StudyLogTest`. Independent of the matcher: it uses only each observation's `mapPoint`, `camera.map.t` and `quality`. Constants: radius 0.30 m, 4 views per group, 400 per scan, spread step 10°.

For each scan observation line, in file order:

1. **Group.** Among the scan's groups, in creation order, take the one whose **anchor** is nearest to `mapPoint` (3D Euclidean, map frame) with distance ≤ 0.30 m; a later group wins only when strictly nearer. If none, create a group whose anchor is this `mapPoint`. Anchors never move. Every built line gets a group, also those that cannot have a view.
2. **Person rule** (section 5). If `viewRect` touches a padded person box: no view (`person`). Not offered to the group.
3. **No crop** (`no_crop`, not expected on the device): not offered.
4. **Offer.** Direction d = unit(`camera.map.t` − `mapPoint`).
   - Group has fewer than 4 views: keep, unless the scan already retains 400 (`scan_cap`).
   - Group has 4 views: pool = [new] + the group's views in the order they were kept. For each element `out` of the pool, in that order, score the set S = pool without `out` (in pool order) as (floor(spread(S) / 10 + 1e-9), Σ quality over S in that order), where spread(S) = the smallest angle in degrees, acos(clamp(dᵢ·dⱼ, −1, 1)), over all pairs of S. The first `out` with the highest score (compare the step, then the quality sum; a later `out` must be strictly better) is left out. If it is the new one: no view (`not_complementary`). Otherwise the new view is kept and `out` is evicted (`viewReplaces`, a `view_evicted` line, the file deleted), and the new view is appended to the end of the group's list. A replacement is allowed at the 400 cap (the count does not grow).
5. **Write.** The JPEG is the crop the embedder saw (`CameraFrame.cropUpright(viewRect)`), at the CPU image's full resolution, upright, JPEG quality 90, the same padding (8 %) and quality as the entry photo. If the write fails: `write_error`, and the group is unchanged.

Pixel size of a view: with `c` = `viewRect` (already clamped), x = trunc(c.left), y = trunc(c.top), w = max(1, min(trunc(c.right − c.left), image.w − x)), h = max(1, min(trunc(c.bottom − c.top), image.h − y)), the subtractions in float32; `viewPx` = (w, h) for rotation 0 or 180, (h, w) for 90 or 270 (`StudyImage.uprightCropSize`).

Views of a scan live in `views/<scanId>/`; eviction deletes only files of the running scan. Views of earlier scans are never deleted or overwritten.

## 5. Person rule (rule 3)

- **Person boxes**: the keyframe's raw detections labelled `person` (after trim and lower-case) with score ≥ 0.20 (the detector's own floor, `ScanConfig.RAW_SCORE_MIN`), each grown by 10 % of its width and height on every side (`Box.padded(0.10)`).
- **A view is cut only when** `viewRect` has zero area in common with every padded person box (touching edges allowed). Otherwise `viewReason` is `person` and nothing is written.
- **Never written**: no line of `obs/` contains a person box (only `personInFrame` and `personOverlap`); a `person` detection is never kept by the filter, so it never has an `obs` line or a crop.
- **Extra views** use the same rule with the scene detector run on the whole verification frame. If the detector cannot run, no crop is written (`person_check_failed`).
- Unchanged: keyframe JPEGs and depth files are still skipped when the frame has any person detection (v0.5 rule); the verification flow's own files are unchanged (see below).

**What the existing code does for the entry photo with a person in the frame (unchanged, reported only).** The filter drops `person` detections and every detection whose box lies more than 50 % (by its own area) inside an **unpadded** person box (`on_person`). Every other kept detection is cropped (box padded by 8 %) and embedded whether or not a person is in the frame, and when it becomes an entry's best crop its photo `inventory/<id>.jpg` is written with no person check. So an entry photo can show part of a person: up to half of the detection's box may lie inside a person box, plus the 8 % padding. The keyframe JPEG/depth skip does not apply to entry photos. Likewise, verification (v0.4) always writes the full upright frame `verify/<vid>_frame.jpg` and the best crop `verify/<vid>_best.jpg`, people or not.

## 6. Entries: `confirmed` and `history` (rule 4)

Two new keys on each entry in `inventory.json` and in full-entry lines of `inventory_log.jsonl` (`add`, `update`, `rename`, `delete`, `confirm`). They are **absent** when not set, so entries without them are written exactly as in v0.7, and v0.7 files load unchanged (`confirmed` null, `history` empty). No existing field changes meaning, and the matcher never reads them.

`confirmed` (object, absent until set):

| Field | Type | Meaning |
| --- | --- | --- |
| `mapPosition` | [3] | where the item was confirmed, map frame |
| `viewPosition` | [3] or null | where the camera stood, map frame |
| `roomId` | string or null | |
| `tMs` | int | capture time of the latest observation behind it |
| `scan` | string or null | the creating scan (`created`), null for `user_verify` |
| `source` | string | `created` or `user_verify` |
| `refs` | [string] | observation ids of its reference views |
| `query` | string or null | `user_verify`: the query |
| `similarity` | number or null | `user_verify`: the capture's similarity |

**Confirmation rule.**

- `created`: set at the end of the scan that created the entry (only if not set yet): `mapPosition` = the entry's `mapPosition` at that scan's end (the baseline mean of its last ≤ 10 sightings), `viewPosition` = its `viewPosition` (camera at the best-quality sighting), `roomId` = its `roomId`, `tMs` = `t` of the last observation linked to it in that scan, `scan` = that scan, `refs` = the observations linked to it in that scan (in link order, tentative phase included) whose view is retained at the scan's end (possibly empty).
- `user_verify`: during a query of a scanned entry with study logging on, a capture whose similarity is ≥ the verification threshold (0.75, `outcome` `CONFIRMED`) followed by the ground truth `present` (study prompt "At the marker", or everyday "Yes, it's here"; after the optional measuring tap) sets `confirmed` from the **latest** such capture of the query: `mapPosition` = the remembered position the capture was aimed at (`target.mapPoint`), `viewPosition` = the camera in the map frame at capture, `roomId` = the item's room as the query resolved it (`query_start.itemRoom`), else the user's room, `tMs` = capture time, `scan` null, `refs` = [the extra view's obs id] (its `view` may be null under the person rule), `query`, `similarity`. Logged as an `inv_confirm` event and a `confirm` line. `displaced`, `absent`, `replaced` and `skip` never confirm. Not when Done was pressed before the capture's result arrived (FREE mode allows that; the capture then has no effect on `confirmed`).
- Nothing else changes `confirmed`. In particular a merge by the baseline matcher in a later scan never does (tested).

`history` (array, absent when empty), one item per scan that gave the entry an evidence outcome (every entry of the scan's map, in scan mode; passive windows add none), oldest first:

| Field | Type | Meaning |
| --- | --- | --- |
| `scan` | string | scan id |
| `tMs` | int | scan end |
| `outcome` | string | `seen`, `missed` or `not_observed`, as that scan's `inv_evidence` event |
| `eligible` | bool | the entry was in the scan's map and target |
| `position` | [3] or null | mean `mapPoint` of the observations the matcher linked to the entry in this scan (all of them, not only the last 10); null when none |
| `sightings` | int | number of those observations |
| `lastObsMs` | int or null | `t` of the latest of them |
| `obs` | [string] | their obs ids, in link order |
| `link` | string | `created` (this scan created the entry) or `baseline_unconfirmed` (a later scan) |

An appearance match (v0.5 evidence) can make the outcome `seen` with `obs` empty.

**`inventory_log.jsonl` additions.** An `evidence` line written at a study scan's end carries `"historyItem"` (the item appended; same object as above). A new op `confirm` carries the full entry and `"source": "user_verify"`. Replay: `add`, `update`, `rename`, `confirm` → the entry is the line's `entry`; `evidence` → set `evidence` and `lastSeenMs`, and append `historyItem` if present; `delete` → remove.

## 7. Events

| Event | Change |
| --- | --- |
| `scan_start` | adds `studyLogging` (bool), `studySchema` (1), `study` (null when off; else `{schema, groupRadiusM, viewsPerGroup, viewsPerScan, spreadBinDeg, personMinScore, personPad, cropPad, jpegQuality, decisionCandidates, obsFile}`) |
| `scan_keyframe` | with study logging on only: adds `study` `{obs, viewsWritten, viewsEvicted, viewsSkipped, obsBytes, viewBytes, personInFrame, lineErrors, ms}` (`ms`: study work of this keyframe, **not** included in `latency.totalMs`) |
| `scan_end` | adds `studyLogging`, `studySchema`, `study` (null when off; else `{schema, obsFile, observations, obsBytes, views: {kept, written, evicted, skipped, groups, bytes}, keyframesWithPerson, lineErrors, studyMsTotal, historyItems, confirmedSet}`) |
| `verify` | adds `captureTMs` (epoch ms of the captured frame), `pressMs` (last Verify press, or null), `camera` (as 3.1), `mapToWorld` (or null), `intrinsics` (scaled to the CPU image), `studyLogging`, `extraViewObs` (obs id, or null when no extra view is logged); `studyError` only if these could not be built. Logged in both states |
| `study_extra_view` (new) | after an extra-view line: `obs`, `query`, `attempt`, `inventoryId`, `view`, `viewReason`, `personInFrame`, `detectError`, `obsFile`, `obsBytes`, `viewBytes`, `ms` |
| `inv_confirm` (new) | a `user_verify` confirmation: `inventoryId`, `name`, `source`, `query`, `attempt`, `obs`, `similarity`, `confirmed` (the new object), `previous` (the old one or null) |
| `internal_error` | new sites `study_keyframe`, `study_scan_end`, `study_extra_view`, `inv_confirm` |

Every other event and field is unchanged.

## 8. Baseline untouched (rule 6)

- The matcher (`Merger.match`), thresholds, detector, embedder, filter and merge rules are unchanged. The study log only reads: candidates come from a separate read-only `Merger.nearest` called before `match`; history and `confirmed` are written after the scan's outcomes and moved-candidate search, and `Merger.addSighting`/`withOutcome` copy them untouched.
- `StudySessionTest.studyLoggingChangesNoDecisionNoExistingFieldAndNoExistingLine` drives two sessions with the same 14 synthetic keyframes (known, new, tentative, promoted and dropped detections, label flips, two detections of one entry in a keyframe, evidence checks, a missed entry), one with study logging: every merge result, evidence check and end-of-scan outcome, every entry field but the two new ones, `inventory.json` and every `inventory_log.jsonl` line (without `t`, `historyItem` and the two new keys) are equal; with logging off the new keys do not exist. The events of a scan are built from these results, so they are equal too (the controller itself has Android dependencies and is not unit-tested).
- `StudySessionTest.aLaterScanMergesIntoAnEarlierEntryAndConfirmedStaysAsItWas`: scan 1 creates the entry; in scan 2 a detection 6 cm away merges into it (`known`), the baseline moves its position, `confirmed` stays equal, `history` gains a `baseline_unconfirmed` item whose position is scan 2's own sightings.
- Views, groups and the person rule never feed back into the matcher.

## 9. Cost and storage

Per keyframe, on the scan worker (never the GL thread), with study logging on:

- One append to `obs/<scanId>.jsonl` with one line per kept detection (about 7.5 KB each: 5.5 KB is the base64 embedding of 1024 floats).
- One JPEG write per kept detection whose view is retained (first scan walk: entry photos of the same crop and quality were 4–27 KB, median about 8 KB), and one delete per eviction.
- CPU: building the lines (base64, a few quaternion products) and the JPEG encodes; no extra crop or embedding (the crop and embedding are the ones the pipeline already made). Logged per keyframe as `scan_keyframe.study.ms`.

From the first scan walk (110 keyframes, 105 kept detections, at most 8 in one keyframe, median padded crop about 175 px a side): about 105 lines (≈ 0.8 MB) and at most 105 view writes (≈ 1 MB) per scan, so roughly 2 MB per scan of that size. Upper bound: 400 retained views (≈ 10–20 MB) plus 7.5 KB per kept detection. Extra views: one line (≈ 7.5 KB) and one crop per verification of a scanned entry, plus one scene-detector pass on the scan worker.

## 10. Extra-view condition (rule 5)

It reuses Items → Find → Verify unchanged; no new guidance UI. The experimenter asks for one more close view of item X; the user opens Items, picks X, Find, follows the guidance, presses Verify, then Done → "present"/"Yes, it's here".

Reading the interruption from the logs (one query `Q`):

| Moment | Where |
| --- | --- |
| Items opened, row tapped (needs Detailed trace) | `ui_action` `items_open`, `items_row` |
| Request start (Find handled) | `query_start.t` ≈ extra-view `requestStartMs` (same instant, wall clock) |
| Guidance reached NEAR | `query_start.t` + `query_end.timeToNearMs` |
| Verify pressed | extra-view `pressMs` = `verify.pressMs` (refused presses: `capture_rejected` with `stage: "verify"`) |
| Frame captured | extra-view `t` = `verify.captureTMs`; `msSinceQueryStart` on the monotonic clock |
| Result logged | `verify.t` |
| Answer given | `query_end.t` (and `inv_confirm.t` when it confirmed) |

Interruption = `query_end.t` − `requestStartMs` (whole task), or `msSinceQueryStart` (request to capture). The time between the experimenter's spoken request and opening Items is not in the log.

## 11. What stays unverified until a device run

- That view JPEGs are written, their sizes, and that `viewPx` equals the decoded JPEG size on the phone.
- The per-keyframe cost on the Pixel 8 (`scan_keyframe.study.ms`, the change in keyframe rate) and storage per scan.
- The scene detector on verification frames (load time on first use, person boxes, `study_extra_view.ms`).
- `camera.map` against `pose.csv` (same `myaw` convention expected) and against the saved keyframe JSON's `cameraPose`.
- The settings checkbox, and that a change applies at the next scan or search.
- The confirmation path end to end: capture ≥ 0.75, "present", `inv_confirm`, `confirm` line, `inventory.json`.
- Crash behaviour (truncated last line, orphan files).

## 12. What a post-run check can verify

For a pulled folder. "Study scans" are those whose `scan_start.studyLogging` is true. Float comparisons: positions 1e-9 m unless noted; float32-derived pixel values 1e-3 px.

**Files and lines**

1. Every study scan has `obs/<scanId>.jsonl`; no other scan has one. Every line parses (only the very last line of a file may be truncated, and only if the scan has no `scan_end` event), has `v` = 1 and a known `type`.
2. Per study scan, `obs` ids are unique and equal `<scan>-<kf>-<det>`; at most one `scan_end` line, and it is the last line; `view_evicted` lines follow their `by` line within the same keyframe.
3. **One line per kept detection**: for every `scan_keyframe` event (scan S, index K) of a study scan, the number of `obs` lines with `scan` S and `kf` K equals `scan_keyframe.kept` − `scan_keyframe.study.lineErrors` (expected 0), their `det` are exactly 0 … n−1, and there is no `obs` line for a (S, K) without a `scan_keyframe` event (the lines are written just before the event, so only a crash in between can leave lines without an event). `scan_keyframe.study.obs`, `viewsWritten`, `viewsSkipped`, `viewsEvicted` equal the counts of those lines (with view, without view, `view_evicted` lines whose `by` is one of them).
4. **Decisions match the event**: the keyframe's `obs` lines sorted by (−`score`, `det`) align one-to-one with `scan_keyframe.results`, with equal `label`, `decision.kind` = `kind`, `decision.entry` = `entry`, `decision.sim` = `sim`, `decision.distM` = `distM`; the counts of each `decision.kind` equal `matched` (known), `new`, `tentative`, `promoted`.
5. **Matcher consistency**: `candidates` are sorted by `distM`, at most 3; for `known`/`tentative`/`promoted`, a candidate with the decision's `entry` (if listed) has the decision's `distM` and `sim`, and every listed candidate strictly nearer is not qualifying (`withinRadius` false, or `sameLabel` false and `sim` < 0.55); for `new`, no listed candidate is qualifying.
6. With the saved keyframe JSON present (`saveFrames` true): `obs.t` = the frame's `t`; the frame's `detections[rawDet − persons before it]` has the same `box`, `label`, `score`, `depthM`, `mapPoint`, `kept` = true; `camera.world` = `cameraPose`; `mapToWorld` equal; when `personInFrame` is true the frame has `imageSkipped` = `person` and `personDetections` > 0, and vice versa.
7. `mapPoint` = T_WM⁻¹ · `worldPoint`; `camera.map.t` = T_WM⁻¹ · `camera.world.t`; `camera.map.forward` = rotation of (0, 0, −1) by `camera.map.q`; |`forward`| = 1 (1e-9).
8. `emb` decodes to `embDim` float32 values; `embModel` = `scan_start.embeddingModel`.

**Views**

9. Every `obs` line with `view` ≠ null and no later `view_evicted` for it: the file exists, decodes as a JPEG of exactly `viewPx`; `viewReason` = `kept`. Every evicted one: the file does not exist (an existing file is a crash orphan, allowed only if the scan has no `scan_end` line).
10. `viewRect` = clamp(pad(`box`, 0.08)) (float32 arithmetic, 1e-3 px); `viewPx` = `uprightCropSize(viewRect, image, rotation)` (section 4; ±1 px tolerance).
11. No file in `views/<S>/` without an `obs` line of scan S naming it (orphans only after a crash); every file's modification time is before the `scan_end` event of S plus a few seconds (never written by a later scan).
12. **Person rule**: every line with a view (scan or extra) has `personOverlap` = 0 and `personInFrame` true or false (never null); `viewReason` = `person` ⇒ `personOverlap` > 0 and `personInFrame` true; `personInFrame` false ⇒ `personOverlap` = 0. No `obs` line has `label` `person`. Extra views with `personInFrame` null have `viewReason` `person_check_failed` and no file.
13. **Retention replay**: replaying section 4 over the scan's `obs` lines in file order (using `mapPoint`, `camera.map.t`, `quality`; lines with `viewReason` `person`, `no_crop`, `write_error` are grouped but not offered) reproduces every `viewGroup`, every `viewReason` among `kept`/`not_complementary`/`scan_cap`, every `viewReplaces` and every `view_evicted` line. At every point at most 4 retained views per group and 400 per scan.
14. `scan_end` line: `retained` = the replayed retained set; `views.kept` = |`retained`|, `views.written` = lines with a view, `views.evicted` = `view_evicted` lines, `views.skipped` = lines without a view by reason, `observations` = `obs` lines; the same numbers as `scan_end` event `study.views`.

**Entries**

15. Every `decision.entry` of a study scan is either an entry that exists in the inventory log after that scan (created by it or older) or listed in that scan's `tentativeDropped`; `created` = the entries whose `add` line (op `add`) appeared during the scan.
16. **History**: for every study scan S of map M that has a `scan_end` event, every entry of map M alive at S's end has exactly one `history` item with `scan` S, and: `outcome` = the `inv_evidence` event (scan S, that entry) `outcome`; `obs` = the obs ids of S whose `decision.entry` is the entry, in file order (an entry takes at most one detection per keyframe, so this is keyframe order); `sightings` = their count; `position` = the mean of their `mapPoint`; `lastObsMs` = the largest `t`; `link` = `created` iff `createdScan` = S. Scans with study logging off add no item.
17. **Confirmed refs**: every `confirmed.refs` id exists as an `obs` line (`obs/<scan>.jsonl` for `created`, `obs/extra_views.jsonl` for `user_verify`); for `created`: `confirmed.scan` = `createdScan`, every ref is in the `created` history item's `obs` and in that scan's `scan_end.retained`, and `refs` = (that item's `obs`) ∩ `retained` in order; `confirmed.tMs` = that item's `lastObsMs`; `confirmed.mapPosition`, `viewPosition`, `roomId` = the entry's `mapPosition`, `viewPosition`, `roomId` on the `update` line written at the end of that scan.
18. **Confirmed invariance**: replaying `inventory_log.jsonl`, an entry's `confirmed` changes only (a) from absent to `source: created` on the `update` line at the end of its creating study scan, and (b) on a `confirm` line. Every other full-entry line of the entry carries a `confirmed` identical to the previous one. In particular, after a later scan whose `decision.kind` `known` lines name the entry, `confirmed` is unchanged unless a `confirm` line came in between.
19. **User confirmations**: each `confirm` line has an `inv_confirm` event with the same `confirmed`; its query has an extra-view line with `obs` = `refs[0]` = `<query>-x<attempt>`, `verify.outcome` `CONFIRMED` (`verify.sim` ≥ `verify.threshold`), `entry` = the entry, and a `query_end` with `groundTruth` `present` logged before the `inv_confirm`; `confirmed.mapPosition` = that line's `target.mapPoint`, `tMs` = its `t`. No `confirm` exists for queries with any other ground truth.
20. **Extra views**: every `verify` event with `extraViewObs` ≠ null has exactly one line in `obs/extra_views.jsonl` with that `obs` and one `study_extra_view` event; `t` = `verify.captureTMs`, `attempt` and `query` equal, `verify.sim` = `verify.bestSimilarity`, `verify.bestKind` = `bestKind`, `entry` = `query_start.inventoryId`, `emb` = the winning candidate's embedding in `verify/<query>_<attempt>_emb.json`, `box` = that candidate's `rect`. `extraViewObs` is non-null exactly for queries with `source` `scan` and `verify.studyLogging` true.
21. **Off means off**: scans with `studyLogging` false have no `obs` file, no `views/<scan>/` folder, no `study` key on their `scan_keyframe` events, `scan_end.study` null, and add no `history` item; queries with `verify.studyLogging` false have no extra-view line and no `confirm`.
