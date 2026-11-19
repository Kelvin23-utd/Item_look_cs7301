# Scan replay: the app's merging re-run from the log of the one real scan

Generated on 2026-11-19 by `analysis/scan_replay.py` from `backups/phone_files_2026-11-17_0008`, scan `aaaf7da9` (session `f52b04d2`, the first scan walk). Tests of the replay's rules: `analysis/test_scan_replay.py` (21 tests, all pass under `python3` and `analysis/.venv-bench/bin/python`).

```bash
python3 analysis/scan_replay.py backups/phone_files_2026-11-17_0008                       # modes (a) and (b)
analysis/.venv-bench/bin/python analysis/scan_replay.py backups/phone_files_2026-11-17_0008 --offline-embeddings   # adds (c)
```

The newer pull `backups/phone_files_2026-11-19_1640` holds the same scan, and modes (a) and (b) give the same numbers on it. The offline embeddings are cached in `analysis/out/scan_replay/` (git-ignored). No image was looked at; the embedder read the saved keyframe JPEGs locally.

## What was replayed

The rules come from `scan/ScanSession.kt` (`merge`, `finish`), `scan/Merger.kt` and `scan/ScanConfig.kt`. That code did not change between v0.5 Block A (commit 33e0e20) and v0.7, so it is the code that ran during this scan. The log does not record the app build: this session's `session_start` has no `appVersion` or `appCommit`, which v0.7 added. The `config` object of `scan_start` confirms two of the constants (`mergeSimilarity` 0.55, `tentativeConfirmMs` 300). The radius constants and the 10-sighting window are not logged, so they come from the code.

The replay takes only the recorded inputs, in capture order: every kept detection of each keyframe JSON (label, score, box, `mapPoint`, `sizeM`, keyframe time). It then applies the app's rules:

- detections go in descending score order, using a stable sort;
- candidates are the real entries followed by the tentative ones, minus the entries already used in this keyframe;
- the distance gate is clamp(0.5 × max(sizes) + 0.15, 0.20, 0.50) m;
- a candidate qualifies by the same label or by a similarity of at least 0.55;
- the nearest qualifying candidate wins;
- promotion needs a second sighting at least 300 ms after the first;
- position is the mean of the latest 10 sightings;
- the best crop is replaced when box area × score is higher (float32);
- tentative entries are dropped at the end.

Every decision starts from the state that the app's logged decisions had produced up to that point (a "forced" replay), so one disagreement cannot spread to later decisions. A "free" run then applies the replay's own decisions, falling back to the logged one only where the replay cannot decide.

Input, as checked by the script:

- 110 keyframe files. 73 have an image. The other 37 have none because a `person` detection kept the image off the disk.
- 105 kept detections:
  - 85 ended in the 24 saved entries;
  - 20 created tentative entries that were dropped at the end (`scan_end.tentativeDropped` = 20).
- Logged kinds: 44 `new`, 24 `promoted`, 37 `known`, 0 `tentative`.
- The store was empty when the scan started (`scan_start.entries` = 0).
- Joining the keyframe JSON with `scan_keyframe.results` gave 0 problems.
- The stable sort reproduces the logged processing order in 110 of 110 keyframes. Nine keyframes have equal scores among their kept detections. In all 9, the reverse order of the tied detections would differ from the logged order, so the data confirm that ties keep the detector's order.
- Of the 85 detections, 15 went to the three bottle entries `19b9bc7b`, `ff153f1e` and `11f9b575`, which lie within about 10 cm of each other. Which physical bottle each of these entries is remains **unknown** in this note. Below, these 15 are reported on their own and also inside the totals. A separate visual audit, `analysis/notes/2026-11-19_bottle_audit.md`, assigns each of the 15 to a physical bottle; this note does not use it, and none of its numbers depend on it.

## Decisions reproduced, by mode

"Needs similarity" means the replay cannot decide because a field is missing. In every case here the missing field is the similarity to a candidate whose label differs from the detection's. No detection lacked any other field.

| Mode | Saved entries (n = 85): reproduced / needs similarity / mismatch | of which the three bottle entries (n = 15) | other saved entries (n = 70) | Dropped tentative entries (n = 20) |
| --- | --- | --- | --- | --- |
| (a) geometry and label only | 83 / 2 / 0 | 14 / 1 / 0 | 69 / 1 / 0 | 19 / 1 / 0 |
| (b) with the logged `sim` | 84 / 1 / 0 | 14 / 1 / 0 | 70 / 0 / 0 | 19 / 1 / 0 |
| (c) with offline embeddings (62 of 105 detections have one) | 85 / 0 / 0 | 15 / 0 / 0 | 70 / 0 / 0 | 19 / 1 / 0 |

The free run gives the same counts in each mode: over all 105 detections, 102 / 3 / 0 in (a), 103 / 2 / 0 in (b) and 104 / 1 / 0 in (c). It ends with the same 24 entries under the same ids and drops the same 20.

Of the 61 matches, 60 qualified by label and 1 by similarity. The replay's distance to the matched entry equals the logged `distM` exactly in all 61 (largest difference 0).

## Mismatches and undecidable cases

There are no mismatches in any mode. The undecidable cases:

| Keyframe | Label (score) | Group | Logged decision | Replayed decision | Cause | Undecidable in |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | bottle (0.477) | bottle entries | `new` → `ff153f1e` | if similarity to `27deebcc` (label bowl, 0.203 m away, radius 0.282 m) ≥ 0.55: `promoted 27deebcc`; else `new` | Missing field: the similarity to a candidate that the app did not match. The app logs `sim` only for the entry it matched, and here it matched none. The detection's embedding is not logged. The keyframe has an image: the offline similarity is 0.210, so the replay says `new` (reproduced in (c)). | (a), (b) |
| 44 | bowl (0.500) | other saved | `known` → `cc7b3487` | if similarity to `cc7b3487` (label spoon at that moment, 0.050 m, radius 0.250 m) ≥ 0.55: `known cc7b3487`; else `new` | Missing field in (a) only. The logged `sim` is 0.749, which decides it in (b); the offline value is 0.756 in (c). This is the only match made by similarity. | (a) |
| 57 | clock (0.430) | dropped | `new` → `6e52bf8f` | if similarity to `98e69aa3` (label book, 0.377 m, radius 0.444 m) ≥ 0.55: `known 98e69aa3`; else `new` | Missing field: no logged `sim` (the app matched no entry). No offline embedding either: the keyframe has no image because a person was detected in it. | (a), (b), (c) |

Other causes were checked and do not occur here:

- Ordering within a keyframe: reproduced in 110 of 110 keyframes, and reversing the tied scores changes no decision.
- Rounding: positions are bit-identical and the distances equal the logged `distM`.
- A rule not confirmed in the code: every rule above is in the code. Which of them the data actually exercise is shown in the rule-coverage table below.

### How far the offline similarity can be trusted

This affects mode (c) and the keyframe 11 decision. The offline embeddings are cut from the half-size keyframe JPEG, while the phone embedded the full-size frame.

- Offline embedding against the embedding the phone stored for the same crop:
  - entries whose final best crop is in a keyframe with an image (`inventory.json`): cosine median 0.911, minimum 0.813 (n = 12);
  - the best crop at promotion (`add` lines of `inventory_log.jsonl`): median 0.923, minimum 0.791 (n = 14).
  - For comparison, Track A found a median of 0.997 when embedding the full-size saved crops (`2026-11-17_track_a_results.md`). The half-size keyframes reproduce the phone's embeddings much less well.
- Offline similarity against the logged `sim` of the matched entry: mean absolute difference 0.031, median 0.024, maximum 0.106 (n = 36 of the 61 logged values; the others need a keyframe without an image). They are on the same side of 0.55 in 35 of 36. The exception is keyframe 12, bottle: logged 0.475, offline 0.559. That match qualified by label, so the threshold did not decide it.
- The keyframe 11 decision in (c) rests on an offline similarity of 0.210. That is 0.340 below the threshold, more than three times the largest difference seen in those 36 pairs. It is still an offline estimate, not the phone's value.

## End state

The forced replay was compared with `inventory.json` for all 24 entries. The free run ends in the same state.

| Check | Result |
| --- | --- |
| Entries kept / tentative dropped | 24 / 20 (the app's `scan_end`: 24 new, 20 dropped); same ids |
| Position (`mapPosition`) difference | 0 m for all 24 (bit-identical) |
| `sightingCount` | equal in 24 of 24 |
| Keyframe of the best crop | equal in 24 of 24. The app's keyframe was identified as the one whose `scan_keyframe.cameraMap` equals the entry's `viewPosition`. In 19 of 24 entries a later sighting replaced the first crop |
| `bestQuality`, `sizeM`, `cropWidthPx` and `cropHeightPx` | equal in 24 of 24 |
| `label`, `labelVotes`, keyframes of the stored sightings, `firstSeenMs`, `lastSeenMs` | equal in 24 of 24 |
| `worldPosition` (rebuilt from the best crop's `mapPoint` and the keyframe's `mapToWorld`) | largest difference 1.8e-15 m (n = 24) |
| `captureDistance` | largest difference 5.6e-8 m (n = 24; the app stores it as float32) |
| State at promotion against the 24 `add` lines of `inventory_log.jsonl` | position difference 0 m; sighting count, best quality and label equal in 24 of 24 |

The three bottle entries (identities unknown). For each one the position difference is 0 m and the replay's best-crop keyframe equals the app's.

| Entry | Sightings (replay / app) | Keyframes of its sightings | Best crop from keyframe (replay / app) |
| --- | --- | --- | --- |
| `11f9b575` | 8 / 8 | 6–13 | 12 / 12 |
| `19b9bc7b` | 5 / 5 | 9–13 | 12 / 12 |
| `ff153f1e` | 2 / 2 | 11–12 | 11 / 11 |

## Which rules this data can confirm

Each row is a forced replay in mode (b) with one rule changed, over all 105 kept detections. A rule is confirmed by the decisions only if changing it creates mismatches.

| Rule changed | Reproduced / needs similarity / mismatch |
| --- | --- |
| none (the app's rules) | 103 / 2 / 0 |
| fixed radius 0.20 m | 101 / 0 / 4 |
| fixed radius 0.50 m | 89 / 9 / 7 |
| radius from the detection's size only (not the larger of both) | 103 / 2 / 0 |
| no one-detection-per-entry limit | 87 / 8 / 10 |
| position = latest sighting | 101 / 2 / 2 |
| position = first sighting | 100 / 3 / 2 |
| best crop (and so the size in the radius) never replaced | 103 / 2 / 0 |
| label = first label (no votes) | 103 / 2 / 0 |
| equal scores in reverse detector order | 102 / 3 / 0 |

What the data confirm and what they leave open:

- **Confirmed by the decisions:** the size-dependent radius, the one-detection-per-entry limit, the position as a mean of sightings (2 decisions), and the tie order (through the logged order).
- **Confirmed by the end state, not by the decisions:** the best-crop replacement and the vote sums.
- **Not exercised at all:**
  - the 10-sighting window: no entry had more than 8 sightings, so "latest 10" and "all" give the same result;
  - the 300 ms promotion boundary: the shortest gap between a first and second sighting was 389 ms, so every second sighting promoted;
  - a label switch by votes: one entry got a second label (spoon 0.727, bowl 0.500) and kept its label;
  - whether the radius uses the larger of the two sizes or only the detection's size;
  - the value of the similarity threshold: the only match by similarity had 0.749. The rejections are not logged, so the data only bound the threshold from above at 0.749.

## Fields the log lacks for a full replay

1. **Per-detection embedding** of every kept detection. It should be written with each kept detection: in `detections[]` of `scan/<scan>/kfNNNN.json`, or in the planned observation log `obs/<scanId>.jsonl`. Without it, the similarity to any candidate other than the matched one is unknown. Recomputing it offline is impossible for the 37 keyframes without an image (43 of 105 kept detections), and only approximate for the others (see above).
2. **Similarity, distance and radius to every candidate within the radius**, with that candidate's label at that moment, not only for the winner. These belong in `scan_keyframe.results[]` (for example as `candidates`). They would let a replay without embeddings decide every case, and show near misses.
3. **`byLabel`** of each match (whether the label or the similarity qualified it), in `scan_keyframe.results[]`. The replay can derive it, but logging it makes the check independent of the replay.
4. **App build and the remaining merge constants.** v0.7 adds `appVersion` and `appCommit` to `session_start`, but this scan's session predates that, so here the rules are tied to the scan only through commit dates. Still missing from `scan_start.config` in v0.7: the radius factor, offset, minimum and maximum, and the sighting window.
5. **Explicit processing rank** of each kept detection, in the keyframe JSON. The replay relies on the array order of the keyframe JSON being the detector's order (which holds here: 110 of 110), and on the order of `results`.
6. **Per-detection `worldPoint` and camera distance**, in the keyframe JSON. These are not needed for the decisions. They were rebuilt here from `mapPoint`, `mapToWorld` (yaw logged in degrees) and `cameraPose` with differences of 1.8e-15 m and 5.6e-8 m, so for now they can be derived.
7. **Store state at scan start** (store order and each entry's sightings, size, label and embedding) for a scan that starts with entries. Here the store was empty. For a second pass it has to be rebuilt from the `update` lines that `inventory_log.jsonl` writes at the end of each scan, so those lines must be complete.

What the v0.8 study build adds (from `FindBack/docs/PLAN_v0.8_study_logging.md`, not yet seen in a pull): items 1, 3 and 5 directly (`emb`, `decision.byLabel`, `det` and `rawDet` in `obs/<scanId>.jsonl`), item 6 (`worldPoint`, `distanceM`), the build (`session_start.appVersion`), and item 2 in part: `decision.candidates` lists only the 3 nearest candidates, but with every embedding logged the replay can compute the rest. Still not logged: the merge constants of item 4.

Images of keyframes with a person in them cannot be saved, by design. That is why item 1 has to be logged on the phone and cannot be recovered offline.

## What the replay does and does not establish

Starting from the app's state at each step, the Python re-implementation of the v0.5–v0.7 merging rules reproduces every logged decision it can decide. That is 83 to 85 of the 85 detections of the saved entries and 19 of the 20 of the dropped ones, depending on how similarity is supplied, with no mismatch in any mode. The final 24 entries (positions, sighting counts, best crops, labels, votes) and their state at promotion are reproduced exactly from the recorded detections alone.

This establishes that the keyframe JSON and `scan_keyframe.results` of this scan are enough to rebuild the inventory. It also establishes that the app ran the rules as written for the rules this scan exercises. Those rules are: the size-dependent radius, one detection per entry per keyframe, the mean position, the tie order and best-crop replacement.

It does not establish:

- the rules this scan never exercises: the 10-sighting window, the 300 ms boundary, label switching, and the threshold value;
- the decision at keyframe 57, which cannot be decided from any recorded field;
- that the offline similarities equal the phone's: they differ by up to 0.106 on 36 pairs, and the keyframe 11 decision rests on one of them;
- anything about whether the decisions were right. It checks that the app did what its code says, not that the 24 entries are 24 distinct physical items. In particular it says nothing about which bottle each of `19b9bc7b`, `ff153f1e` and `11f9b575` shows.

It rests on one scan of one room: n = 105 decisions, 24 entries.
