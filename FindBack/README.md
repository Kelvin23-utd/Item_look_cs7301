# FindBack v0: evidence-guided smartphone object memory

Android (Kotlin, ARCore) prototype for the CS 6301 project. It builds on Google's ARCore `ml_kotlin` sample (Apache 2.0).

Current version: **v0.6 (a simpler screen)**, see [the v0.6 section](#v06-a-simpler-screen-nov-16) and `docs/PLAN_v0.6_ui.md`, on top of v0.5 Block A (scan mode, `docs/PLAN_v0.5_scan_once.md`), v0.4 (study instrumentation) and v0.3 (home map). See [the v0.4 section](#v04-study-instrumentation-nov-16) and `docs/STUDY_SPEC_v0.4.md`, and [the v0.3 section](#v03-home-map-nov-16) and `docs/PLAN_v0.3_home_map.md`. The table below describes v0; where a later section differs, the later section wins.

## What v0 does

| Step | Implementation |
| --- | --- |
| Room reference | ARCore Augmented Image (`assets/reference.jpg`, printed 200 mm wide). Stored positions are offsets in the image's coordinate frame, so they survive app restarts. |
| Register | Tap an object. The app hit-tests the tap (depth, plane or feature point) and runs ML Kit to get boxes. It picks the smallest box containing the tap, or uses a fixed crop around the tap if no box does. MediaPipe MobileNet-V3 then embeds the crop and you confirm a name. The detector's raw label and the user's name are stored separately. |
| Retrieve and guide | Items → Find shows a 3D marker at the last-seen spot, plus distance and direction, the photo, and whether the position comes from the same-session anchor or the reference image. Everything is labelled historical. |
| Verify | Projects the remembered point into the current camera frame and crops the expected region, plus any detector boxes that overlap it. Each crop is embedded and compared to the stored embedding. A score of 0.75 or higher counts as "seen here now"; anything lower is "insufficient evidence", never "moved". |
| Selective policy | Uses only signals available before any fresh capture: age, similarity to the nearest other record, position source, and reference tracking. It is logged at query start, so the three conditions can be simulated offline from the same data. |
| Ground truth | Done asks whether the item is really at the marked spot. The answer is logged with the query. |

## Build, test and install

Enable **Developer options → USB debugging** on the phone, connect it by USB and accept the prompt, then run:

```bash
cd FindBack
./gradlew testDebugUnitTest     # JVM unit tests (map math, walk graph, guidance, map.json, old records, v0.4 record fields, records log, embedding files)
./gradlew installDebug          # build and install
adb shell am start -n edu.cs6301.findback/.MainActivity
adb logcat -s FindBack FindBackEvent   # live log
```

## Session procedure (v0.3)

One-time setup of the home map:

1. Choose a home fix point. Either show the Earth picture (iPad full screen, or the print) in a fixed place, or pick a **spot**: a phone pose you can repeat, such as the phone pressed against a door frame.
2. Set the picture width first if you use the picture: menu button (⋮, top right) → **Settings…** → **Picture width** (iPad mini 6/7 ≈ 11.6 cm). The app restarts to apply it.
3. Set home:
   - Picture: hold the phone 0.5–1 m from it until the app asks **Set home here?**, then tap **Set home**.
   - Spot: press **Set home here** (bottom button) → hold the pose → **Lock** → name it. The phone can be upright or on its side, as long as you use the same way each time.
4. The status pill (top left) shows **Which room?**. Tap it and pick **New room…** to name the room you are in.
5. Walk slowly through every room. The mini-map (top right; tap to enlarge) shows the trail. When you enter a new room, set it with the status pill; walking into a room you already named switches the pill after two trail points.
6. Optional: in far rooms, add more spots (menu ⋮ → **Add a spot here**) so a later session can locate there too.
7. Tap each object and name it. The register dialog shows the room (tap to change) and warns when you are not located or more than 1.5 m away.

Each later session:

1. Launch, then locate: look at the picture (it locates automatically), or press **Locate** (bottom button; also in the menu ⋮) → line up the faint ghost image with the live view → **Lock**.
2. Query trial: Items → pick one → Find. The card names the room and the walk length; follow the white arrows along the walked path until the ring appears on the item (NEAR). Then Verify (aim at the ring) → Done → ground truth (five choices in study mode, see v0.4 and v0.6) → for "At the marker" or "Nearby", tap the object's true centre (marker error or displacement).

Restart test: kill the app, relaunch, locate, and check that the markers land on the objects.

To start over: menu ⋮ → **Start a new map…** moves `records.json` and `map.json` into `archive/<yyyyMMdd-HHmmss>/` and restarts. Nothing is deleted.

## Pull the study data

```bash
adb pull /sdcard/Android/data/edu.cs6301.findback/files ./findback_data
```

| File | Contents |
| --- | --- |
| `records.json` | The active object memories: names, raw detector output, embeddings, world positions, (v0.3) `mapId`, `mapPosition`, `viewPosition`, `roomId`, and (v0.4) `mobility`, `lineageId`, `supersedes`, `moveCount`, `lineageStartMs`. v0.2 `refPosition` is still read but no longer written. |
| `records_log.jsonl` | v0.4: every record change, one JSON object per line, with the full record (embedding included). Holds the records that left `records.json` too. |
| `map.json` | v0.3 home map: fix points (pose in the map frame, picture width and heading axis, ghost image), rooms, walk-graph nodes and edges |
| `fix/` | v0.3 ghost images of spots (upright camera frame at half size) |
| `archive/<time>/` | `records.json` and `map.json` moved aside by **New map** |
| `crops/` | Registration crops |
| `verify/` | Fresh full frames and best crops from each verification, for offline re-scoring. v0.4 adds `<vid>_emb.json`: the embedding of every candidate crop. |
| `events.jsonl` | `session_start`, `model_loaded`, `reference_locked` (time to relock), `register`, `query_start` (policy inputs and decision), `verify` (all candidate scores, latencies, inference calls), `query_end` (ground truth), plus the v0.3 and v0.4 events below |

## Settings to tune

- Picture width: menu ⋮ → **Settings…** → **Picture width** (stored in preferences; `MainActivity.DEFAULT_REF_WIDTH_M` is the default).
- `VerificationPolicy.MATCH_THRESHOLD`, `AGE_MINUTES`, `DISTRACTOR_SIMILARITY`, `DRIFT_METERS`: set these on the development session and freeze them before the test session.
- v0.3 map thresholds (starting values from the plan) are named constants at the top of each class: `ImageFixWindow` and `SpotCollector` (`map/FixEstimator.kt`), `Localizer`, `WalkGraph`, `GuidancePlanner`, `DriftTracker` and `TrackingLossDetector` (`map/Drift.kt`), `MapStore`.

## v0.2 changes (Nov 13, after the first pilot)

- **Guidance overlay:** a pulsing ring on the target when it's on screen, an edge arrow with "turn left/right" when it's off screen or behind you, and a haptic tick when the target is centred. The ring turns green when verified and amber when not confirmed.
- **Guide card:** shows the photo, a "last seen" time that keeps updating, distance and direction, the policy recommendation (Verify ★ when recommended) and the verification result. If no location is available, it tells you to find the room reference.
- **Status panel:** colored dots for tracking and the reference image, the item count, and a hint for the next step.
- **Items list:** thumbnails, Rename, and Delete with a confirmation step. Deleting removes the item from the list but keeps its photo and logs.
- **Registration:**
  - Scene-sized detector boxes are rejected; the crop falls back to a 0.22 m window around the tapped point, sized from the camera intrinsics.
  - Depth and plane hits are preferred over sparse feature points, and hits beyond 5 m are rejected.
  - Default names are "item N", and the log records whether you typed the name.
- **Marker error:** after "Yes, here", tap the object's true centre and the app logs the 3D, horizontal and vertical error in `query_end.markerError`.
- **Verification:** detector boxes must be within 0.25–4× the expected crop area, so scene boxes can't produce matches from the background alone.
- **Build:** pinned to ARCore SDK 1.48.0 to match the phone. Android 17 shows a "16 KB alignment" warning for debug builds; it is harmless on this phone and goes away once ARCore is updated and the SDK version is raised.

## v0.3: home map (Nov 16)

Positions no longer hang off one reference picture. The app keeps its own gravity-aligned **map frame** and re-attaches it each session from one **fix point**; ARCore's motion tracking carries it from there through the flat.

- **Map frame and fix points.** The map frame differs from ARCore's world by a yaw and a translation only (`map/Pose4.kt`). A fix point is the picture (pose averaged until 20 samples from the last 1.5 s agree within 2 cm and 2°, or the median after 3 s, logged as `loose`) or a spot (camera pose averaged over 10 frames; the heading comes from whichever display axis is more horizontal, so the phone can be upright or on its side, and the choice is stored with the spot). After a fix, an ARCore anchor carries the fix pose so ARCore's own map corrections move the map with the world.
- **Refresh and drift.** While the picture stays in view, the app re-fixes at most every 2 s and logs the jump (the drift measurement). A jump over 0.75 m or 15° asks before applying; the usual cause is that the iPad was moved.
- **Walk graph and rooms.** Breadcrumbs every 0.5 m of where the phone has been while located; an edge only joins consecutive points within 1.2 m, so routes follow walked paths. Rooms are named with the Room chip (v0.6: the status pill); unlabelled trail takes the current room, and an explicit pick relabels trail within 1 m.
- **Guidance modes.** `NO_FIX` (cannot be placed: room, photo, "go to a fix point"), `FAR_ROUTE` (room, walk length, white chevron 1.5 m ahead along the route), `FAR_NO_ROUTE` (room and straight distance, no arrow), `NEAR` (ring and edge arrow as in v0.2). NEAR holds within 3 m of remaining route, or 5 m straight in the same room, or 2.5 m when not known to be in different rooms, or 4 m for a same-session anchor; leaving needs 1 m more.
- **Mini-map** (top right, tap to enlarge): trail, room names, fix points (square = picture, diamond = spot), items, target (green ring), route (blue) and you (blue dot with heading).
- **Location sheet** (tap or long-press the status panel): picture width, add a spot, use a spot (ghost image), rename a room, new map. v0.6: the Home sheet, opened with the menu button; "Use a spot" is **Locate**.
- **Records.** New records store `mapId`, `mapPosition` (item in the map frame), `viewPosition` (where the camera stood, the route's end) and `roomId`. Records without a map position, or from another map, are listed as "photo only (saved before this map)"; the v0.2 pilot items load in this state. Position source is now `SESSION`, `MAP` or `NONE` (old logs contain `REFERENCE`; the evaluator accepts both).
- **Policy.** The v0.2 "reference not fully tracked" trigger is replaced by: position source `MAP` and (more than `DRIFT_METERS` = 20 m walked since the fix, or a tracking loss since the fix). These are known before any capture.
- **Unchanged:** crop logic, candidate rules, embedder, detector, `MATCH_THRESHOLD`.

### v0.3 events

All v0.2 keys are kept with their meaning (`register.refAvailable` is now always `false` and `refPosition` `null`, because no picture-relative position is stored). Drift proxies, where listed, are `fixPoint`, `fixType`, `metersSinceFix`, `trackingLossesSinceFix`, `msSinceFix` (all `null` when not located).

| Event | Fields |
| --- | --- |
| `session_start` | adds `mapId`, `mapNodes`, `mapFixPoints`, `mapRooms` |
| `map_created` | `mapId`, `fixPoint`, `fixType`, `pose` (`x`, `y`, `z`, `yawDeg` in the map frame) |
| `fix_point_added` | `mapId`, `fixPoint`, `name`, `fixType`, `pose`, `yawAxis`, `widthM`, `ghostFile`, `samples`, `posSpreadM`, `yawSpreadDeg` |
| `map_fix` | `mapId`, `fixPoint`, `fixName`, `fixType`, `kind` (`first`: located from not located; `refresh`: automatic picture re-fix; `manual`: spot lock while located), `samples`, `posSpreadM`, `yawSpreadDeg`, `cameraDistanceM` (picture), `quality` (`good`/`loose`), `jumpM`, `jumpHorizM`, `jumpDeg` (null for `first`), `metersSincePrevFix`, `lossesSincePrevFix`, `msSincePrevFix`, `prevFixPoint`, `disagreement`, `accepted`, `applied`, `anchored`, `worldPose` |
| `map_fix_lost` | `mapId`, `fixPoint`, `reason` (ARCore stopped the anchor) |
| `map_create_declined` | the user answered "Not now" to **Set home here?** |
| `map_saved` | `mapId`, `nodes`, `edges`, `rooms`, `fixPoints` (at most every 5 s while changed, and on pause) |
| `map_archived` | `mapId`, `dir`, `records`, `recordsMoved`, `mapMoved` |
| `tracking_lost`, `tracking_resumed` | `reason`, `afterMs` / `durationMs`, `located` (pauses of 1 s or longer only) |
| `room_created`, `room_renamed` | `mapId`, `room`, `roomName` / `from`, `to` |
| `room_set` | `mapId`, `room`, `roomName`, `previousRoom`, `relabelled`, `source` (`chip`/`register`), `located` |
| `room_auto` | `mapId`, `room`, `roomName`, `previousRoom`, `relabelled` (0), `node` |
| `spot_cancel` | `fixPoint` (a captured spot was not named) |
| `register` | adds `mapId`, `mapPosition`, `viewPosition`, `room`, `roomName`, `located`, drift proxies |
| `query_start` | adds `mapId`, `recordMapId`, `located`, `itemRoom`, `itemRoomName`, `userRoom`, `userRoomName`, `guidanceMode`, `nearRule`, `routeM`, `straightM`, drift proxies |
| `query_end` | adds `nearReached`, `timeToNearMs`, `modesSeen`, `finalMode`; `markerError` adds the drift proxies |
| `verify` | adds the drift proxies at capture and `guidanceMode` (`positionSource` was already logged) |
| `internal_error` | `site`, `error`: an exception in the v0.3 map code, caught so the AR loop keeps running (at most every 5 s per site) |

`analysis/evaluate.py` adds three sections at the end: marker error by position source, map-fix jumps (median and max), and queries by guidance mode.

## v0.4: study instrumentation (Nov 16)

Implements section 3 of `docs/STUDY_SPEC_v0.4.md`. Everything is additive: every earlier event key keeps its name and meaning, and the verification pipeline and `VerificationPolicy.decide()` are unchanged. With capture mode FREE and no study tag the app behaves as v0.3, except for the five-way ground truth, the mobility question, and the new **Update location** and **Study settings…** entries. (v0.6: in that everyday mode the study prompts are replaced by simpler ones; see the v0.6 section. With a tag or another mode everything below is unchanged.)

### Study settings (S1)

Menu ⋮ → **Study settings…**. Stored in preferences; they take effect at once and survive restarts. They cannot be changed during a search, so all events of one query carry the same settings.

- **Study tag:** free text, up to 60 characters, by convention `key=value` pairs separated by `;`, for example `home=A;exp=E3;round=2;light=evening;split=dev`. The app does not check the form.
- **Capture mode:** `FREE` (default), `PAIRED` or `BENCHMARK`.

When a tag is set or the mode is not FREE, a small amber line under the status pill shows `Study: PAIRED · home=A;exp=E3;round=2`.

### Capture modes (S2, S3)

- **FREE:** any number of captures per query, no extra prompts.
- **PAIRED:** one capture per query. Verify is hidden once a capture has produced its result (v0.4: disabled). Done without a capture asks **Capture once before finishing?** (Back / Finish without capture) when the item has a position right now; when it has none, Done goes straight to the ground truth. `query_end.noCaptureReason` records which.
- **BENCHMARK:** any number of captures. After each result is closed the app asks **What is at the marker right now?** (The item / Nothing / A different object / A look-alike) with a **Condition** choice (default, near, far, angled, dim, bright, occluded) that stays as last set until the app restarts. Closing that dialog without a choice logs the label as `null`.

In PAIRED and BENCHMARK, Done waits while a capture is still being processed.

### Ground truth at Done (S4)

Five choices: **At the marker** (`present`), **Nearby, but not at the marker** (`displaced`), **Not in this area** (`absent`), **Another object is at the marker** (`replaced`), **Skip (not usable)** (`skip`). `present` and `displaced` continue with a tap on the item's true centre (or Skip): the distance to the marker is the marker error for `present` and the displacement for `displaced`.

### Mobility (S6) and Update location (S7, "Moved it" since v0.6)

- The register dialog asks **How often does it move?** (Rarely / Sometimes / Often, default Sometimes), stored as `mobility` = `rare`, `some` or `often`.
- **Update location:** Items → the item → **Moved it** (v0.4: **Update location**). The card shows the item; tap it where it is now. The normal capture runs and the register dialog opens as **New place for "X"** (v0.4: **Update location of X**), with the name and mobility of the old record and the room of the new place (the old room when the new place has none). Save replaces the old record: the new one gets the same `lineageId`, `supersedes` = the old id and `moveCount` + 1, and the old record leaves `records.json`. Its crop, verification files and log lines stay. Cancel, in the card or in the dialog, changes nothing.
- A new ordinary record has `lineageId` = its own id, `moveCount` 0 and `lineageStartMs` = `lastSeenMs`; files from before v0.4 read as that. `lineageStartMs` is carried along a lineage, so `lineageAgeMin` counts from the item's first registration.
- Records are still never changed by a verification.

### Files (S5)

`verify/<vid>_emb.json`, where `<vid>` is `<query>_<attempt>` as for the frame and crop files. The embeddings are the ones the pipeline computed for its scores; nothing is embedded again.

```json
{ "model": "mobilenet_v3_small", "dim": 1024, "encoding": "base64-f32le",
  "record": "<target record id>", "query": "<query id>", "attempt": 1,
  "candidates": [ { "kind": "projected", "rect": [812.0, 402.5, 1100.0, 690.5], "sim": 0.81, "embedding": "<base64>" } ] }
```

`candidates` is in the order of the `verify` event's `scores`; `rect` is left, top, right, bottom in sensor image pixels, as there. `embedding` is the float32 vector as little-endian bytes in standard base64: `np.frombuffer(base64.b64decode(s), dtype="<f4")`.

`records_log.jsonl`, one line per change:

```json
{"t": 1794800000000, "session": "ab12cd34", "op": "add", "record": { ...as in records.json, with the embedding... }}
```

| `op` | `record` |
| --- | --- |
| `add` | The new record |
| `rename` | The record under its new name |
| `delete` | The record that was removed |
| `supersede` | The new record of **Moved it** (Update location); its `supersedes` is the record that left the active list |

Replaying the lines in order gives the active list at any time. Two kinds of line carry one more key: `"backfill": true` on the `add` lines reconstructed, once, for records that were already in `records.json` when the log was first created (their `t` and `session` are the record's own `lastSeenMs` and `sessionId`, as at registration), and `"reason": "map_archived"` on the `delete` lines written when **New map** moves `records.json` to the archive.

### v0.4 events

`studyTag` and `captureMode` are added to `session_start`, `register`, `query_start`, `verify`, `query_end`, `verify_label` and `supersede`. For the events of a query they are the settings at its start.

| Event | Fields |
| --- | --- |
| `study_settings` | `studyTag`, `captureMode`, `prevStudyTag`, `prevCaptureMode` (logged when either changes) |
| `register` | adds `mobility`, `lineageId`, `supersedes` (null for a new item), `moveCount`. `nameSource` is `kept` when Update location kept the old name. |
| `supersede` | `oldRecord`, `newRecord`, `name`, `lineageId`, `moveCount`, `oldAgeMin`, `distanceM` (old to new map position; null unless both are in the current map), `oldRoom`, `newRoom`. Logged after the new record's `register`. |
| `update_cancel` | `record`, `stage` (`waiting`: before a tap; `dialog`: the register dialog was cancelled). `register_cancel` adds `updateOf`. |
| `query_start` | adds `mobility`, `lineageId`, `moveCount`, `lineageAgeMin` |
| `verify` | adds `embFile`, `msSinceQueryStart`, `msSinceNear` (null if NEAR was never reached). Both are taken when the frame is captured, on the monotonic clock. |
| `verify_label` | `query`, `record`, `attempt`, `label` (`target`, `empty`, `other`, `lookalike`, or null when dismissed), `condition`. BENCHMARK only, one per `verify`. |
| `query_end` | `groundTruth` is now one of `present`, `displaced`, `absent`, `replaced`, `skip`. Adds `captures` (captures that produced a `verify` event; `verifications` still counts attempts), `noCaptureReason` (null when `captures` > 0, else `user_skipped` when the item had a position at Done, else `not_located`; logged in every mode), `msToFirstVerify` (the first capture's `msSinceQueryStart`, or null). `markerError` adds `displaced` (true or false). |

## v0.6: a simpler screen (Nov 16)

`docs/PLAN_v0.6_ui.md`. Only the user interface changed. Map, localizer, scan pipeline, verification, `VerificationPolicy`, stores, file formats and every event stay as they were, except one added field on `query_end`.

- **One screen state at a time,** decided by the pure `ui/UiModel.kt` (unit-tested): locating with the lock bar, measuring, finding, "Moved it", scanning, no map, not located, ready. The bottom bar shows only buttons that work now, primary on the right (filled), secondary on the left (outlined): Items / Locate when not located, Items / Scan when located, Done / Verify during a search, Stop scan during a scan, Set home here without a map, Cancel / Lock in the lock bar.
- **Top:** a status pill (room, "Which room?", "Not located", "No home yet", "Tracking lost"; tap to pick the room), the menu button (⋮, the Home sheet), the study line in study mode, one instruction line, and a message banner that replaces the Snackbar, so messages never move the buttons.
- **Study mode** is on when a study tag is set or the capture mode is not FREE, and is fixed for a query at its start. It keeps every v0.5 prompt: the five-way ground truth and the measuring tap, PAIRED's capture-first prompt, the verification result dialog with the BENCHMARK label after it, the policy line, and the registration debug line and mobility question.
- **Everyday mode** (no tag, FREE): Done asks **Did you find it?**: **Yes, it's here** (`present`), **It was somewhere else** (`displaced`; then **Moved it** for a tapped item, or a hint to scan again for a scanned one), **Not found** (`absent`), **Keep looking** or Back (nothing logged). There is no measuring tap. The verification result appears in the guide card ("Still here" or "Could not confirm"); a tap on it shows both photos. Registration asks no mobility question: an update keeps the record's value, a new item gets `some`.
- **Locate** is a bottom button when not located, and also in the guide card and the Home sheet. One spot opens its lock bar at once; several ask which, the home spot first.
- **Items** is a full-screen panel: search over names, labels and room names, room chips with two or more rooms, and an item sheet with Find, Rename, Moved it (tapped items) and Delete.
- **Home sheet** (menu): Locate, Set home here or Add a spot here, Rooms…, Settings… (picture width, Save scan frames), Study settings… (tag and mode), Start a new map….
- **Registration:** the name field starts with the default name (or the old name for Moved it); the first tap selects it, so typing replaces it.
- **Back** closes the Items panel, cancels the lock bar and Moved it, and does not leave the app during a search or a scan.

### v0.6 events

| Event | Change |
| --- | --- |
| `query_end` | adds `truthPrompt`: `study` (the five-way prompt) or `simple` ("Did you find it?"). Everyday queries carry no study tag, so the analysis already leaves them out. |
| `register` | unchanged fields. `nameSource` is `typed` only when the saved name differs from the default name (new item) or from the old name (Moved it); with the field now pre-filled, saving the default unchanged is `default`, as before. |

## Known limits

- v0.3: yaw error at the fix point grows with distance (2° is 17 cm at 5 m), so far rooms may fail to verify an object that is there. Measure the marker error first; a wider verification search is a separate change.
- v0.3: the picture is a movable landmark (mitigated by the disagreement prompt and by spots), and a spot is only as repeatable as the user's pose.

- v0.4: in FREE mode, Done pressed while a capture is still being processed ends the query first; that capture's `verify` event then follows `query_end`, which reports `captures` without it. PAIRED and BENCHMARK make Done wait.
- v0.4: `analysis/evaluate.py` still keeps only `present` and `absent` queries; `displaced` and `replaced` need the v0.4 analysis.

- v0.2 only: the reference pose was a single live estimate, not averaged over frames (v0.3 averages).
- Records are never updated by verification, so every condition queries the same memory. The only change to an item's memory is **Moved it** (v0.4 Update location), which replaces the record.
- There is no tap-to-measure for guidance error yet; that is planned for v0.2.
