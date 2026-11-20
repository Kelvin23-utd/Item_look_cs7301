# Two-scan run with two identical glasses

Date: 2026-11-19, about 20:44. App build `a13520a` (v0.8), same map as the earlier runs, both scans in one app session with a new "Locate" before each. Data: one private pull (not in the repository).

**Ground truth is incomplete.** The person who ran it reported that the experiment was done; what was changed has not been stated yet. What follows is read from the views (visual audit of the retained views of the 22 entries the two scans touched) and is marked where it depends on the missing ground truth.

## What was recorded

| Scan | Duration, path | Keyframes | Kept detections | Result in the app |
| --- | --- | --- | --- | --- |
| 1 `f65eefb7` | 18 s, 5.5 m | 26 | 20 | 4 new, 3 seen again, 3 missed |
| 2 `a4aa9877` | 13 s, 5.6 m | 23 | 53 | 5 new, 11 seen again, 4 missed |

`analysis/study_check.py`: six of seven checks pass (73 observation lines for 73 kept detections, matcher replay 20 of 20 and 53 of 53); the `scans` check fails only because both scans are in one session. Scan 2 covered a wider area than scan 1, so only one entry was matched in both.

## The two glasses, as seen in the views

- **Scan 1:** two glasses of the same model stand on one table, 0.29 m apart (6 and 3 observations; never both detected in one keyframe).
- **Scan 2:** one glass of that model stands on the table (7 observations) and one stands on a counter 1.3 m away (5 observations). So one of the pair was moved. **Which one is not known from the data.**

The glass that stayed is 0.14 m from one scan-1 position and 0.18 m from the other, with a shift between the two locates of about 0.15 to 0.19 m, so position does not say which glass it is. Appearance does not either:

| Median best cosine (phone embedding) | Scan-1 glass, first position | Scan-1 glass, second position |
| --- | --- | --- |
| Scan-2 glass on the table | 0.77 | 0.61 |
| Scan-2 glass on the counter | 0.46 | 0.45 |

The two scan-1 glasses match each other at 0.72, the two scan-2 glasses at 0.60.

## What the app did

Four entries now stand for the two glasses, two of them older entries of other objects:

- In scan 1 one glass was merged into an old "bowl" entry that used to show a pot at that place (same label, within reach); the entry's photo became the glass. The other glass got a new "bowl" entry.
- In scan 2 the glass on the table was merged into an old "cup" entry that used to show a drink can there, and that entry's photo became the glass. The scan-1 entry of the other glass was marked missed.
- The glass on the counter got a new "cup" entry.

## The frozen rules on scan 2

`analysis/identity_replay.py`, history = all earlier scans, test = scan 2 (53 observations in 19 sightings; shift estimated from 5 to 6 object pairs, of which 3 to 4 agree).

| | Current app | Generic rules | Household |
| --- | --- | --- | --- |
| Glass on the table (7 observations) | merged into the old can entry | linked to one scan-1 glass entry | linked to the same glass entry |
| Glass on the counter (4 observations in one sighting) | new item | new item | **unresolved** |

- The household rules leave the moved glass unresolved, which is the intended answer for an identical pair: the pack marks the pair as look-alikes.
- Both rule sets link the staying glass to the one scan-1 glass entry whose reference views show a glass. The other scan-1 glass sits in an entry whose references still show the pot, so it could not be chosen. Whether the choice is right is unknown and may be unknowable; the honest answer for the staying glass is also "one of the pair".
- The generic rules add the moved glass as a new item, like the app.

## What this run adds

- It is the first identical-pair case: neither position (with a 0.15 m shift) nor the phone's appearance score separates the two glasses, for the glass that stayed as well as for the one that moved.
- "Unresolved between the two" was produced only by the method that knows the pair from history.
- The app's history is again the weak point: an entry's reference views can show a different object from the one it now holds.

## Ground truth supplied afterwards

After the sections above were written, the person who ran the scans stated: **the glass that was moved is the right-hand one of the two, in the central part of the table.**

Mapped to the data with the camera poses of scan 1: in all 9 keyframes that detected either glass, the glass at the first position (the one the app merged into the old "bowl" entry, 6 observations) lies to the right of the other as seen from the camera. So that glass was moved to the counter, and the glass at the second position (the new "bowl" entry of scan 1, 3 observations) stayed.

Scored against that:

| | Truth | Current app | Generic rules | Household |
| --- | --- | --- | --- | --- |
| Glass on the table in scan 2 | the glass that stayed (second scan-1 position) | wrong: merged into the old can entry, and the right entry marked missed | right entry | right entry |
| Glass on the counter in scan 2 | the moved glass (first scan-1 position) | new item (duplicate) | new item (duplicate) | unresolved (not wrong, not decided) |

Two cautions about the rules' right answer for the glass that stayed:

- The raw evidence pointed the other way. The glass that stayed is 0.18 m from its own scan-1 position and 0.14 m from the moved glass's old position, and the phone's appearance score also favours the moved glass's scan-1 views (0.77 against 0.61). Nearest position or best appearance alone would have named the wrong glass.
- The rules could not have picked the other entry, because its reference views still show the pot it used to hold. Whether the correction of the shift between the locates would have chosen correctly on its own was not tested.

n = one identical pair, one test scan. A case, not a rate.
