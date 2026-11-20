# Three-scan run: two unchanged scans, then one moved cup

Date: 2026-11-19, about 19:56. App build `a13520a` (v0.8), same map as the earlier runs. Data: one private pull (not in the repository). Ground truth from the person who ran it, not derived from the app: **scans 1 and 2 show the same scene; before scan 3 one cup was moved; nothing else changed.**

## What was recorded

| Scan | Duration, path | Keyframes | Kept detections | Result in the app |
| --- | --- | --- | --- | --- |
| 1 `0dfd8938` | 15 s, 2.5 m | 22 | 76 | 10 new, 4 seen again, 1 missed |
| 2 `fb0bf2d9` | 13 s, 2.4 m | 19 | 68 | 10 new, 6 seen again, 6 missed |
| 3 `fb65023c` | 11 s, 2.7 m | 22 | 55 | 3 new, 8 seen again, 2 missed |

All three scans cover one small, dense area (many objects within about a metre). They were made in **one app session** (the app was not restarted), with a new "Locate" before each scan, so they test repeated locating, not a restart. No study tag was set. A fourth scan of 1.5 s right before them is void.

`analysis/study_check.py` on scans 1 and 2 and on scans 2 and 3: six of the seven checks pass (observation lines 144 of 144 and 123 of 123, views, entries, history, matcher replay with no mismatch); the `scans` check fails only because both scans of a pair are in the same session.

**The app's position of the same objects shifted between the scans:** for the entries the matcher linked in both scans the median distance is 0.21 m between scans 1 and 2 (maximum 0.26 m, n = 6) and 0.17 m between scans 2 and 3 (n = 4). The replay's own estimate before any correction is larger (median 0.31 to 0.36 m over 6 to 8 object pairs for scans 1 and 2). The merge radius is 0.20 to 0.50 m.

## Scan 2: nothing had changed (visual audit of the retained views)

With nothing moved, the app's matcher linked only 22 of scan 2's 68 observations to an existing entry and created 10 new entries. Audited examples:

- **The cup.** In scan 1 it is entry `a1be09aa` (19 observations). In scan 2 its 12 observations were merged into entry `30452d65`, which in scan 1 showed a different object (a stack of cans) 0.26 m away; that entry's photo was overwritten with the cup. The cup's own entry was marked missed, and the cans got a new entry. The scan-2 observations of the cup match its scan-1 observations with a median best cosine of 0.88, and the cans with 0.36: appearance was clear, but the matcher merges a same-label detection into the nearest entry without an appearance test.
- Three more unchanged objects got a second entry in scan 2 (a stack of mugs, a red cup, a tumbler).

The same observations replayed with the frozen rules of `analysis/identity_replay.py` (history = all earlier scans, test = scan 2; 68 observations in 27 sightings):

| | Current app | Generic rules | Household |
| --- | --- | --- | --- |
| Observations linked to an existing item | 22 | 55 | 58 |
| Observations given a new item | 40 | 11 | 10 |
| Dropped or unresolved | 6 dropped | 2 unresolved | 0 |
| Existing items marked missed | 6 | 1 | 2 |
| The cup (14 observations) | merged into the cans' entry | linked to its own entry | linked to its own entry |

The rules first correct the shift between the scans (median residual 0.31 m to 0.08 m for the generic rules, 0.36 m to 0.06 m with the household pack). Not every link of the rules was audited: at least one looks doubtful (two observations the app gave to a bottle entry are linked to a mug's entry), so "linked" is not the same as "linked correctly".

## Scan 3: the cup moved

- The cup now stands 0.70 m from its scan-2 position, 0.37 m higher. The app created a **new entry** `8c1233db` for it (10 observations). Its scan-1 entry is marked missed; the entry it was merged into in scan 2 is "not observed". The inventory now has three entries that show this one cup.
- Both old places were looked at in scan 3 and are empty (6 and 5 keyframes; highest similarity there 0.24 and 0.06).
- **Appearance is weak here.** The scan-3 observations match the cup's scan-2 observations with a median best cosine of 0.46 (minimum 0.37) and its scan-1 observations with 0.40 (minimum 0.24). The two groups that show the cup are still the best two of all history groups, but the next ones follow at 0.37 and 0.34.

Replay with the frozen rules (history = all earlier scans, test = scan 3):

| | Current app | Generic rules | Household |
| --- | --- | --- | --- |
| The moved cup (10 observations) | new item (duplicate) | unresolved | new item (duplicate) |

So neither rule set links the cup: the generic rules at least decline to decide; the household rules add a duplicate, like the app. The household threshold for this cup is high because the history holds the cup under two entries that look alike.

## The three moved objects so far

| Moved object | Seen at the new place | Best cosine to its own earlier views | Current app | Generic rules | Household |
| --- | --- | --- | --- | --- | --- |
| Cup, first run | once, other side | 0.57 | sighting discarded | unresolved | linked |
| Can, second run | twice | 0.64 and 0.76 | duplicate | linked | linked |
| Cup, this run | ten times, 0.37 m higher | 0.46 (median) | duplicate | unresolved | duplicate |

n = 3 moved objects in one home; these are cases, not a rate. No run yet has a same-model pair or a deliberately unvisited old place.

## What this run adds

1. **The larger problem is not the moved item but the unchanged scene.** With nothing moved, a shift of 0.2 to 0.3 m between two locates made the current matcher duplicate or swap most of a dense table. Correcting the shift from the scan itself and requiring appearance agreement removed most of that.
2. **History made by the current matcher is already wrong** (one cup under two entries, an entry whose photo changed to another object), and rules that build on that history inherit the errors. A fair comparison needs the history rebuilt from the observations by the method under test, not taken from the app's inventory.
3. **For the cup moved to a different height the phone's embedding is not enough** (0.40 to 0.46 against 0.88 for the same cup unmoved). This is the kind of case a stronger matcher off the phone, or one more close view, would have to decide.
