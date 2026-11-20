# Second two-pass run: one moved drink can

Date: 2026-11-19, about 19:30. App build `a13520a` (v0.8), same map as the smoke test. Data: one private pull (not in the repository). Ground truth from the person who ran it, not derived from the app: **between the two scans one drink can was moved; nothing else was changed.**

## What was recorded

| Scan | Session | Duration, path | Keyframes | Kept detections | Result in the app |
| --- | --- | --- | --- | --- | --- |
| `696ce663` | `b7d71585` | 34 s, 9.4 m | 50 | 0 | nothing: every detection that passed the score filter had no depth (57 dropped as `no_depth`, valid depth 0 in every keyframe) |
| `8621fb08` | same | 7 s, 1.0 m | 9 | 0 | nothing, same cause |
| `0c3c261f` (scan A) | `63e90caa` | 62 s, 19.5 m | 97 | 82 | 8 new, 8 seen again, 6 missed, 24 not observed |
| `b6221dcb` (scan B) | `a785a537` | 22 s, 8.2 m | 40 | 35 | 4 new, 6 seen again, 3 missed, 37 not observed |

The first two scans are void: the depth image was empty for that whole app session, and it worked again after a restart. The cause is not known. No study tag was set. Scan B covered less than scan A.

`analysis/study_check.py` on scans A and B: all 7 checks pass (117 observation lines for 117 kept detections, 75 views, matcher replay 82 of 82 and 35 of 35). The first run failed the `entries` line because the check resolved an entry's reference views only within the two scans being checked, and this map holds entries from two earlier scans; the check now reads every study scan's observation file for that. No data was wrong. Alignment between the two launches: median 0.05 m (maximum 0.07 m, n = 4 entries matched in both).

## The moved can (visual audit of the retained views of the 26 entries the two scans touched, and of the dropped detections)

- Scan A: entry `e98ae991` (label "cup"), 17 observations.
- Scan B: its old place was checked in 19 keyframes and is empty (highest similarity there 0.19); the app marked the entry **missed**, which is right.
- Scan B: the can was detected in two keyframes 2.55 m away and the app created a **new entry** `f9a56c12` for it. So the app now holds the can twice: once as missed at the old place and once as a new item.
- By the phone's embedding the two new observations are closest to the can's own scan-A observations among all scan-A entries: best cosine 0.64 and 0.76 (median 0.56 and 0.61 over its 17 observations); the best other entry scores 0.43 and 0.38.

Nothing else that was seen in scan A and visited in scan B changed its outcome wrongly: the two other entries missed in scan B had already been missed in scan A.

## The four methods on scan B (parameters frozen before this data existed)

`analysis/identity_replay.py` with the parameter file frozen on the smoke data (history: the three earlier scans with observations; test: scan B). Scan B has 35 observations in 16 sightings.

| | Current app | Generic rules | Household |
| --- | --- | --- | --- |
| The moved can (1 sighting, 2 observations) | new item (duplicate) | linked to its entry as moved | linked to its entry as moved |
| Single detections (8 sightings) | 7 dropped, 1 added | 8 added | 6 added, 1 unresolved, 1 linked to an existing entry |
| A container first saved in scan B (4 observations) | added | unresolved | unresolved |

The household method linked one single detection (label "teddy bear") to an existing entry of another label; whether that is right is unknown (its view is unclear). No close-up was taken after scan B, so the extra-view method equals the household method. The script was edited after the freeze (two rule fixes made before its first commit; no parameter differs), which it reports itself.

## Across the two runs so far

| Moved object | Current app | Generic rules | Household |
| --- | --- | --- | --- |
| Cup, smoke test (seen once at the new place) | sighting discarded | unresolved | linked |
| Can, this run (seen twice at the new place) | duplicate item | linked | linked |

n = 2 moved objects in one home. This is not a rate. Neither run contains a same-model pair, a new same-model item or a deliberately unvisited old place.
