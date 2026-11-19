# Identity pilot protocol

Two people run this with one phone and one Mac. Part A, the smoke test, checks that the study build records what the analysis needs. Part B, the pilot, is prepared here but is frozen only after Part A passes.

Marks used in the steps:

- **[hands on phone]**: a person holds or touches the phone.
- **[Mac]**: done at the Mac, with the phone connected by USB or wireless adb. The phone must be unlocked for adb.

Button and setting names are the app's own words in the study build (v0.8), as written in its code and in `FindBack/docs/PLAN_v0.8_study_logging.md`. None of them has been seen on the phone yet: the build is installed, but no v0.8 scan has been run. Where a label comes only from the code, the step says "the box labelled ... in the code".

The data folders and the filled ground-truth sheet describe a home. They stay out of git: folders named `findback_data*` and files named `paper/identity_pilot_ground_truth_*.private.csv` are ignored. No photo, floor plan or description of the home goes into the repository, the notes or the paper.

## Part A: smoke test (about 15 minutes)

### What it tests

There are two passes over one small area, in two app launches, on one map. Both passes are scanned with study logging on. Between the passes one object is moved and the app is restarted. The post-run check (`analysis/study_check.py`) then confirms that the following are complete and consistent:

- the observation log, with one line per kept detection;
- the retained close-up views;
- each entry's `confirmed` and `history`;
- the replay of the app's matcher.

It also reports how far apart the two launches place the same entries.

### Objects

Use 6 to 8 objects of kinds the phone's detector names (COCO classes). In the first scan walk it found cup, bowl, bottle, book, laptop, keyboard, banana and spoon. Include one look-alike pair: two objects of the same model, for example two identical cups.

Put them in one small area, such as a table and the shelf next to it, at least 15 cm apart. Give each object a code (A1, A2, ...) on a paper note. Note which two codes form the pair and which object you will move. Keep people, and pictures of people, out of the area: a frame with a person in it keeps no image and no views.

### Steps

Before you start (the v0.8 study build is already installed on the phone):

1. **[hands on phone]** Unlock the phone and keep it unlocked while the Mac talks to it. If the Mac uses wireless adb, connect it: **[Mac]** `adb connect <phone address>:5555`, then check with `adb devices`. After a phone restart, wireless adb needs one USB connection first.
2. **[Mac]** Back up what is on the phone now, into a new folder:
   ```
   adb pull /sdcard/Android/data/edu.cs6301.findback/files findback_data_before_smoke
   ```
3. **[hands on phone]** Open FindBack. Menu ⋮ shows the Home sheet; the study build logs `appVersion` 0.8 at start. To start the smoke test with an empty item list:
   - open the menu (⋮, top right) → **Start a new map…** → **New map** (nothing is deleted; the old files move to `archive/`, and the app restarts);
   - stand at a spot you can return to exactly, for example with the phone against a door frame;
   - press **Set home here**, hold still, press **Lock**, and type a name when asked "Name the home spot".
4. **[hands on phone]** Tap the status pill (top left, "Which room?") → **New room…**, and type a name.
5. **[hands on phone]** Menu ⋮ → **Settings…**. Check that the box labelled **Study logging** in the code ("Study logging (every scan detection with its pose and embedding, …)") is ticked: it is on by default, and a change applies from the next scan. Tick **Save scan frames** too (the check compares the observation log with the saved keyframe files). Leave **Detailed trace** on. Press **Save**.
6. **[hands on phone]** Menu ⋮ → **Study settings…**. Set the study tag to `exp=smoke` and leave the capture mode at FREE. Press **Save**. An amber line "Study: FREE · exp=smoke" appears under the status pill.

Pass 1:

7. **[hands on phone]** Place the 6 to 8 objects. Write each object's place on the note.
8. **[hands on phone]** Locate. If the status pill says "Not located": go to the home spot, press **Locate**, line up the camera with the faint picture, hold still, press **Lock**. You are located when the pill shows the room name with a green dot and the bottom buttons read **Items** and **Scan**.
9. **[hands on phone]** Press **Scan**. Walk slowly past the area for 30 to 60 seconds:
   - keep every object in view, 0.5 to 1.5 m away;
   - look at each object from at least two directions;
   - the scan bar shows "Scanning · N items".
10. **[hands on phone]** Press **Stop scan**. The summary "Scan finished" shows "N new items". Write N on the note and press **OK**.

Between the passes:

11. **[hands on phone]** Move the chosen object to another place in the area, at least 0.6 m from where it was (the merge radius is at most 0.5 m). Do not touch the look-alike pair. Note the new place.
12. Restart the app. Either:
    - **[Mac]** `adb shell am force-stop edu.cs6301.findback`, then `adb shell am start -n edu.cs6301.findback/.MainActivity`; or
    - **[hands on phone]** open the recent apps, swipe FindBack away, and tap its icon.

Pass 2:

13. **[hands on phone]** Locate again as in step 8 (**Locate** → line up → **Lock**).
14. **[hands on phone]** Press **Scan** and walk the same way as in pass 1. Press **Stop scan**, write down the summary line ("N new items · M seen again"), and press **OK**.

Afterwards:

15. **[Mac]** Pull the data into a new folder:
    ```
    adb pull /sdcard/Android/data/edu.cs6301.findback/files findback_data_smoke
    ```
16. **[Mac]** Run the post-run check:
    ```
    python3 analysis/study_check.py findback_data_smoke
    ```
    It finds the last two scans with study logging on. To name them yourself, add `--scans <first scan id>,<second scan id>`. It follows section 12 of `FindBack/docs/PLAN_v0.8_study_logging.md`.

### What a pass and a fail look like

The check prints seven lines, each PASS or FAIL with a name and counts:

| Name | What it checks |
| --- | --- |
| `scans` | both scans have `scan_start` and `scan_end`, run in different app sessions on one map, with study logging on, located at scan start |
| `files` | every observation file parses, has one `scan_end` line at the end, and has well-formed ids; no observation file exists for a scan with study logging off |
| `observations` | one line per kept detection, with embedding, camera pose (world and map frame) and scan id; the baseline decision equals `scan_keyframe.results` and the saved keyframe file |
| `views` | every retained view file exists at its logged pixel size, which matches its padded box; no view touches a person; replaying the view-retention rule gives the same groups, the same views and at most 4 per group |
| `entries` | `confirmed` and `history` exist; `confirmed` refs point at observations with view files; `confirmed` of pass-1 entries is unchanged after pass 2 unless a user confirmation was logged; one history item per scan |
| `history` | each entry's per-scan positions and linked observations rebuilt from `obs/` equal its `history` |
| `replay` | `analysis/scan_replay.py`, run with the logged embeddings, reproduces every baseline decision, `byLabel`, similarity, distance and listed candidate |

Then comes one line with the alignment between the two launches (median and maximum distance between an entry's pass-1 and pass-2 positions), reported but not judged. The last line is `PASS: all 7 checks` with exit code 0, or `FAIL: k of 7 checks` with exit code 1. A pull from before v0.8 is named as one ("This is a pre-v0.8 pull"), and all seven lines fail.

Pass: all seven lines are PASS, the replay line reads "0 need a similarity, 0 mismatch" for both passes, and the alignment line has numbers.

Fail: any FAIL line, for example:

- a scan without `scan_end` (the app was killed during a scan);
- both scans in one session (the app was not restarted);
- study logging off;
- not located before a scan;
- a kept detection with no observation line;
- a view file missing or of the wrong size;
- a view from a frame region that touches a person;
- `confirmed` changed by pass 2 without a user confirmation;
- a replay mismatch.

On a FAIL:

- keep the folder;
- write down the failing lines;
- do not run Part B;
- if the cause is a step that was skipped, repeat Part A with a new map. Otherwise the app or the check needs fixing first.

## Part B: the pilot (prepared now; freeze after Part A passes)

### Design

- **Items:** about 12 items, codes I01 to I12, of detector-supported kinds, in two or three rooms. Two of them form a same-model group. A third copy of that model is kept out of sight until the test scans.
- **Places:** each place gets a code (P01, P02, ...). The meaning of each code is written only on the private sheet.
- **Scans:** four scans, each in its own app launch, all on one map, all with study logging on:
  - two history scans (H1, H2), with nothing deliberately changed between them;
  - two test scans (T1, T2), after the changes below.
- **Study tags:** `exp=identity;scan=H1`, `scan=H2`, `scan=T1` and `scan=T2`, set in **Study settings…** before each scan.

Conditions, set up between H2 and T1 (T2 repeats them on a different launch, with a second route):

| Condition | What is done | Items |
| --- | --- | --- |
| Stays | left where it was | about 6 |
| Moved | taken to another place, at least 1 m away, in the same or another room | 2 |
| New same-model item | the third copy of the same-model group is put out in a new place | 1 (plus the 2 already listed) |
| Old place not visited | left in place or moved, but the test walk does not go past its old place | 1 to 2 |
| Live extra view | after the test scan, one more close view of a listed item (below) | 2 to 3, chosen before T1 |

### Steps per scan

1. **[hands on phone]** Launch the app (restart it if it is open, as in Part A step 12). Set the study tag for this scan. Check that **Study logging** and **Save scan frames** are still ticked in **Settings…**.
2. **[hands on phone]** Locate at the home spot (**Locate** → line up → **Lock**).
3. **[hands on phone]** Press **Scan** and walk the route planned for this scan, at walking pace, with the same habits as in Part A. Press **Stop scan**, then **OK**.
4. The second person fills in this scan's columns of the private ground-truth sheet from what is physically there, never from the app:
   - present or absent;
   - place code;
   - moved since the previous scan or not;
   - whether the walk went past the item's place;
   - their initials and the time.

   Start the sheet from `paper/identity_pilot_ground_truth_template.csv` and save it as `paper/identity_pilot_ground_truth_<date>.private.csv`, or outside the repository.
5. **[Mac]** Pull the data into a new folder per scan (`findback_data_pilot_H1`, ...). After H2, T1 and T2, check the latest two scans: `python3 analysis/study_check.py <folder> --scans <previous scan id>,<this scan id>`. Each must pass before the next scan. Each scan's id is the `scan` of its `scan_start` line in `events.jsonl`.

Live extra view (after the T1 and T2 scans, for the items chosen before T1):

6. The second person names one item aloud. No stopwatch is needed: the app logs the times itself. The time between the spoken request and opening Items is not logged, so start at once.
7. **[hands on phone]** **Items** → tap the item → **Find**. Follow the arrows to the ring.
8. **[hands on phone]** Hold the phone 0.3 to 0.6 m from the item and press **Verify**. Wait for the result before going on.
9. **[hands on phone]** Press **Done** and answer the ground-truth question from what is physically there. With the study tag set, the question is the five-way one ("At the marker", "Nearby, but not at the marker", ...). After "At the marker" the app asks for a tap on the item's true centre; tap it, or tap **Skip**.

What the logs give for this condition (plan section 10):

- the extra-view line in `obs/extra_views.jsonl` holds the close view: its embedding, and a crop unless a person was in the frame;
- request to capture: `msSinceQueryStart` of that line. The request starts when Find is handled (`requestStartMs`, which is about `query_start.t`);
- the whole interruption: `query_end.t` − `requestStartMs`.

An answer "At the marker" after a capture at or above the verification threshold (0.75) is a user confirmation. It sets that entry's `confirmed` (`source: user_verify`, logged as `inv_confirm`), and T2 then starts from it. Record on the sheet which items were confirmed this way.

### Ground-truth sheet

`paper/identity_pilot_ground_truth_template.csv` has one row per item and these columns:

- `item_code`, `kind`, `same_model_group` (empty when none);
- for each scan `h1`, `h2`, `t1`, `t2`:
  - `_present` (yes or no);
  - `_place_code`;
  - `_moved` (yes or no, against the previous scan);
  - `_visited` (yes or no: the walk went past the item's place);
  - `_filled_by` (initials);
  - `_filled_at` (date and time);
- `notes`.

The filled sheet is private because it describes a home. It must stay out of git: `.gitignore` excludes `paper/identity_pilot_ground_truth_*.private.csv`. Never derive the sheet from the app's output.

### Planned comparison (not implemented yet)

The same images and the same history are replayed offline through:

1. **The app's current matcher**: the baseline decisions logged in the observation log, re-checked with `analysis/scan_replay.py`.
2. **Generic rules.** Each test observation is linked to an entry using:
   - place evidence (distance to the entry's last confirmed place);
   - exclusivity: an item seen in its place in this scan cannot also be the item seen elsewhere;
   - fixed similarity and distance thresholds.
3. **The same plus household-specific information**, built only from the history scans:
   - several reference views per item (the retained views of H1 and H2);
   - the same-model groups;
   - a threshold per group;
   - how stable each item's place was in the history.
4. **For the live condition only:** method 3 plus the one extra close view.

Each item in each test scan gets one outcome per method:

- **correct update:** the item's observations are linked to its own entry. For an item not seen, the entry is kept, not marked as seen elsewhere;
- **wrong merge:** observations of one item are linked to another item's entry;
- **duplicate:** a present item gets a new entry instead of its own;
- **unresolved:** the method declines to decide.

Outcomes are counted per method and per condition, with n, against the private sheet. For the live condition the report also gives:

- the interruption times (median and range, with n);
- how many decisions the extra view changed, and in which direction.

All rules and thresholds of methods 2 to 4 are fixed from the smoke test and the history scans alone, and written down with the commit hash before any test-scan folder is read. A change after that is reported as a change, with both results.
