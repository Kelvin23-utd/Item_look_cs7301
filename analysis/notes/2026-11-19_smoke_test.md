# Two-pass smoke test of the v0.8 study build

Date: 2026-11-19. App build `a13520a` (v0.8), Pixel 8. Data: one private pull (`backups/phone_files_2026-11-19_smoke_1752`, not in the repository). Protocol: `paper/IDENTITY_PILOT_PROTOCOL.md`, Part A.

## What was run

| | Pass 1 (scan `e4144f49`) | Pass 2 (scan `03f9e780`) |
| --- | --- | --- |
| App session | `5f960b0a` | `12232b14` (after a restart) |
| Duration, path | 141.5 s, 41.5 m | 32.0 s, 9.0 m |
| Keyframes (with a person detection) | 225 (66) | 64 (30) |
| Kept detections = observation lines | 140 | 81 |
| Views kept (written, evicted, skipped for a person) | 113 (128, 15, 8) | 66 (72, 6, 9) |
| Entries | 28 new, 53 tentative dropped | 10 new, 17 seen again, 2 missed, 9 not observed, 20 tentative dropped |

Both passes are on one map and were located at the same home spot. Departures from the protocol: no study tag was set, and pass 2 was much shorter than pass 1, so 9 entries were not observed. Whether a look-alike pair was among the objects is not recorded.

Physical ground truth, as stated by the person who ran it and not derived from the app: **one cup was moved between the passes; nothing else changed.**

## Post-run check

`python3 analysis/study_check.py <pull>`: all 7 checks pass.

- observations: 221 lines for 221 kept detections, each with embedding, camera pose and scan id
- views: 179 kept in 98 place groups, none touching a person box; the retention rule replayed in Python gives the same choices
- entries: 38; the last-confirmed state of all 28 pass-1 entries is unchanged after pass 2
- history: 66 items rebuilt from the observation files
- replay of the matcher with the logged embeddings: 140 of 140 and 81 of 81 decisions reproduced, none undecidable, no mismatch

**One change to the check after its first run on this data.** The first run failed the `entries` line because the list of created entries at the end of an observation file was not in the same order as the `add` lines of the inventory log (same ids in both, for both passes). The plan had called that list "in promotion order". The check now compares the two as sets, and the plan's wording was corrected. No other check was changed.

## What the passes show (logs only; no image was looked at)

- **Alignment between the two launches:** for the 15 entries the matcher linked in both passes, the pass-1 and pass-2 positions differ by a median of 0.12 m (maximum 0.29 m). The merge radius is 0.20 to 0.50 m, so this error alone can put an unmoved item out of reach.
- **The current matcher against the ground truth.** One cup moved, yet pass 2 created 10 new entries (bowl 3, cup 1, book 1, laptop 1, suitcase 1, carrot 1, keyboard 1, cell phone 1). Six of them have a pass-1 entry of the same label 0.06 to 2.09 m away that was itself seen again in pass 2; three have a label that pass 1 never saved. Which of the 10 are duplicates of existing items and which are objects first detected in pass 2 is **unknown**: it needs a look at the views.
- **The moved cup.** Two pass-1 cup entries are marked missed and one new cup entry appeared. The new entry's stored embedding has a cosine of 0.22 and 0.16 to the two missed cups (0.46 at most to any pass-1 cup, and 0.46 to a bottle). Over all logged observations, the best cosine between the new cup's 2 observations and the missed cups' pass-1 observations is 0.25 and 0.28. So no pass-1 cup resembles it by the phone's embedding. Whether the new entry is the moved cup is **unknown** until the views are looked at.
- **Extra view.** One find-and-verify query (label "teddy bear"): the two captures came 15.7 s and 18.1 s after the request, and the whole interruption lasted 21.1 s. Their similarities were 0.72 and 0.73, under the 0.75 threshold, so the result was "insufficient evidence" although the answer was "present", and no confirmation was written. Both captures were saved as extra views.
- **Cost of the logging on the phone:** the added time per keyframe was a median of 0.1 ms in pass 1 and 5.3 ms in pass 2 (maximum 54 ms), against a median of about 200 to 250 ms for a whole keyframe. The two passes wrote 1.7 MB of observation lines and 2.9 MB of views.
- **Views:** the retained views have a median shorter side of 208 px (pass 1) and 179 px (pass 2), against 32 to 58 px for the other-angle crops of the one entry examined in the first scan walk (its saved keyframes were half size).

n is one room, two passes, one moved object. This run tests the logging, not a matching method.
