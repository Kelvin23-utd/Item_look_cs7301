# Visual audit of the three nearby "bottle" entries (scan `aaaf7da9`)

Date: 2026-11-19. One reading by the project's assistant, not confirmed by a person who knows the home.

**What was looked at.** The saved half-size keyframes 5 to 14 of the scan (960 × 540), cut to the region of the bottle boxes, with every kept "bottle" box drawn in a colour per entry (box and entry id from each keyframe's JSON). Three contact sheets were made in a scratch folder outside the repository and are not kept. No other image was opened.

**The physical objects.** The boxes fall on a row of bottles standing side by side, a few centimetres apart. Three of them are involved in the three entries; they are easy to tell apart by colour:

- bottle P: dark, with a white label
- bottle Q: amber, with a yellow cap
- bottle R: tall, dark green

## Which bottle each kept detection shows

| Keyframe | Entry `11f9b575` | Entry `19b9bc7b` | Entry `ff153f1e` |
| --- | --- | --- | --- |
| 6 | R | – | – |
| 7 | R | – | – |
| 8 | R | – | – |
| 9 | R | Q | – |
| 10 | R | Q (blurred frame) | – |
| 11 | **Q** | **P** | R |
| 12 | R | P | **Q** |
| 13 | R | P | – |

All 15 kept detections of the three entries could be assigned; none is recorded as unknown. Keyframe 10 is blurred, and its box for `19b9bc7b` is read as Q because the yellow cap sits in the middle of it. A fourth bottle (clear) was detected once in keyframe 11 as tentative entry `036a2456` and dropped at the end of the scan.

## What this means

- **No entry is one physical bottle.** `11f9b575` is bottle R in 7 of its 8 sightings and bottle Q in one. `19b9bc7b` starts on bottle Q (2 sightings) and ends on bottle P (3 sightings). `ff153f1e` is bottle R once and bottle Q once.
- **The saved inventory has R twice, P once and Q not at all.** The stored photo of `11f9b575` comes from keyframe 12 (R), that of `ff153f1e` from keyframe 11 (R), and that of `19b9bc7b` from keyframe 12 (P). Bottle Q was detected four times and has no entry whose photo shows it.
- **This explains the low similarity inside `19b9bc7b`.** Its embedding at creation (keyframe 10, bottle Q) and its final embedding (keyframe 12, bottle P) have cosine 0.30: they are two different bottles.
- **Cause.** The bottles stand closer together than the merge radius (0.20 to 0.50 m), and a detection merges into the nearest entry with the same label without an appearance test. Depth noise of a few centimetres is then enough to swap neighbours.

## For later analyses

- For a replay or a matching benchmark, the identity of each of these 15 detections is the bottle letter in the table, not the entry id. 4 of the 15 detections show a different bottle from the one in their entry's stored photo (keyframes 9 and 10 of `19b9bc7b`, keyframe 11 of `11f9b575`, keyframe 12 of `ff153f1e`), so "same entry" pairs that include them are not same-object pairs. In addition, `ff153f1e` and `11f9b575` are two entries whose photos show the same bottle R, so "different entry" pairs between them are not different-object pairs.
- Any bottle not listed here, and every other entry of the scan, was not audited and its identity stays unknown.
