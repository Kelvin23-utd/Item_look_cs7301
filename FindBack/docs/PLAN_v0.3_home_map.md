# FindBack v0.3: home map (one fix point locates the whole flat)

Status: plan, 2026-11-16. Implementation is delegated; Claude (main session) audits and runs the device tests.

## 1. Why

v0.2 stores each item's position relative to one printed reference image and guides with a straight arrow. Two problems came out of the pilot:

1. The user cannot print, and has one small iPad. A reference picture per room is not available.
2. When the item is in another room, a straight arrow points through walls, says nothing about which room, and is wrong as soon as tracking has drifted.

The user's proposal, which this plan adopts: scan the flat once, save a map, save items relative to that map, and re-attach the whole map each session from **one known point**.

ARCore cannot save its own scan to a file (only Cloud Anchors persist, and they need a Google Cloud project and a network). So the app keeps its **own lightweight map**: a home coordinate frame, the trail the user has walked, room labels on that trail, and item positions. ARCore's motion tracking carries the map from the fix point to wherever the user walks.

## 2. Scope

In scope:

- A persistent, gravity-aligned **map frame** and **fix points** that locate it in each session.
- Two fix-point types: the existing reference picture (iPad or print), and a marker-free **spot** (the phone held in a repeatable pose, with a ghost-image alignment aid).
- A **walk graph** (breadcrumbs of where the phone has been), **room labels**, a **mini-map**, and **route guidance** along walked paths.
- Guidance that names the room first and shows the ring only when the user is near the item.
- Logging of everything above, so drift and guidance quality can be reported in the paper.

Out of scope (do not build):

- Cloud Anchors, any network use, any new heavy dependency.
- Any change to the verification pipeline: crop logic, candidate rules, embedder, detector, `MATCH_THRESHOLD`. Only the *source* of the remembered point changes.
- Wall detection, real floor plans, multi-floor support. The flat is treated as one floor.
- Using natural pictures (posters, book covers) as landmarks. Possible later as a third fix-point type.

## 3. Frames and math

- **World frame W:** ARCore's session frame. +Y is up. Origin and heading are new on every launch.
- **Map frame M:** persistent. +Y is up. Origin and heading are those of the first fix point.
- Both are gravity-aligned, so W and M differ by a yaw about +Y and a translation: 4 degrees of freedom. Never use a full 6-DoF transform between them. Tilt error in a 6-DoF image pose would be multiplied by the distance to far rooms.

`Pose4(x, y, z, yaw)` (pure Kotlin, no Android imports): a rotation about +Y followed by a translation. It needs `transformPoint`, `inverse`, `compose`, and a constructor from a position plus a **reference axis**: a 3D direction that is projected onto the horizontal plane and defines the local +X heading.

Reference axis per fix type:

| Fix type | Position | Reference axis | Reject when |
| --- | --- | --- | --- |
| IMAGE | `AugmentedImage.centerPose` translation | The image's local +X axis (along its width). If its horizontal length is under 0.5, use local +Z instead. Store the choice in the fix point as `yawAxis` and always reuse it. | Stored axis has horizontal length under 0.3 |
| SPOT | `camera.displayOrientedPose` translation | The display's +X axis (screen right). It stays horizontal at any pitch as long as the phone is not rolled. | Horizontal length under 0.7 ("hold the phone without rolling it") |

Check the axis conventions of `AugmentedImage.centerPose` against the ARCore 1.48 javadoc; do not rely on this table for signs.

Locating: a fix point stores `mapPose` (its Pose4 in M; identity for the first one). Observing it gives `worldPose` (its Pose4 in W). Then `T_WM = worldPose ∘ inverse(mapPose)`.

Adding a fix point while located: `mapPose = inverse(T_WM) ∘ worldPose`.

## 4. Localizer (session state, GL thread)

- **IMAGE fix.** Collect samples only while the image is `FULL_TRACKING`, the camera is `TRACKING`, and the camera is within `clamp(6 × refWidthM, 0.5, 1.5)` metres of the image. Keep a sliding window of about 30 samples. Accept when at least 20 samples from the last 1.5 s agree within 2 cm and 2°. Use the mean position and the circular-mean yaw. If the window has not converged after 3 s of gated tracking, accept the median and log `quality: "loose"`. All of these are named constants.
- **SPOT fix.** On "Lock", average the gated camera pose over about 10 frames.
- **Anchor.** After a fix, create an ARCore `Anchor` at the fix's world pose. Each frame, re-read the anchor, re-extract a Pose4 from it the same way, and recompute `T_WM`. This lets ARCore's own map corrections move the map with the world. Detach replaced anchors on the GL thread through the existing `anchorsToDetach` queue.
- **Refresh.** While the picture stays in view, accept a new IMAGE fix at most every 2 s. Log the jump between the old and new `T_WM`, measured at the fix location, in metres and degrees. That jump is the drift measurement for the paper.
- **Disagreement.** If a new fix moves the map by more than 0.75 m or 15° while already located, do not apply it silently. Ask: "Position jumped 1.2 m. Use the new fix?" The likely cause is that the iPad was moved.
- **Drift proxies**, reset at every fix: `metersSinceFix` (camera path length), `trackingLossesSinceFix` (tracking paused for 1 s or longer), `msSinceFix`, plus the fix type and fix-point id.
- **First map.** If no map exists and the picture passes the gate, ask once: "Set home here?" Do not create a map silently. A spot can also be the first fix point.
- **Width change.** `refWidthM` is stored on the IMAGE fix point. If the setting later differs, ignore that fix point and say so in the location sheet.

## 5. Persisted map (`map.json` in the app's files root)

```json
{ "version": 1, "mapId": "8 chars", "createdMs": 0,
  "fixPoints": [ { "id": "", "name": "", "type": "IMAGE|SPOT", "x": 0, "y": 0, "z": 0, "yaw": 0,
                   "yawAxis": "X|Z", "widthM": 0.116, "ghostFile": "fix/<id>.jpg", "createdMs": 0, "sessionId": "" } ],
  "rooms": [ { "id": "", "name": "", "createdMs": 0 } ],
  "nodes": [ { "id": 0, "x": 0, "y": 0, "z": 0, "roomId": null, "sessionId": "", "tMs": 0 } ],
  "edges": [ [0, 1] ] }
```

`MapStore` follows `MemoryStore`'s pattern: synchronized, write to a temp file and rename. Save at most every 5 s when dirty, and on pause. A corrupt file is copied aside, never deleted.

**New map** (location sheet, with confirmation): move `records.json` and `map.json` into `archive/<yyyyMMdd-HHmmss>/`, then restart the activity. Crops, verification frames and `events.jsonl` stay where they are. Nothing is deleted.

## 6. Walk graph (pure Kotlin, unit-tested)

Fed with the camera position in M at about 5 Hz while located and tracking.

- Snap to the nearest node within 0.35 m horizontally. Otherwise add a node once the phone is 0.5 m from the current node.
- Add an edge between consecutively visited nodes only if they are within 1.2 m. This keeps every edge a path somebody actually walked.
- Break the chain (no edge across the gap) at session start, after a tracking loss, and after a fix that moves the map by more than 0.3 m.
- `route(from, to)`: Dijkstra between the nodes nearest each end (each within 1.5 m). Returns the polyline and its length, or null.
- `roomAt(p)`: the room of the nearest labelled node within 1.5 m, or null.

Room labels:

- The user picks the current room from a chip in the status panel, or in the register dialog. Options: existing rooms, "New room…", "No room".
- Unlabelled nodes the user visits take the current room.
- An existing different label is never overwritten by walking. After two consecutive nodes of another room, the current room follows the label and the chip updates.
- An explicit pick relabels nodes within 1.0 m of the user. That is the correction tool.

## 7. Records

Add to `ObjectRecord`, all nullable, all absent in old files:

- `mapId`, `mapPosition` (xyz in M),
- `viewPosition` (camera xyz in M at capture: a walkable point next to the item, used as the route's end),
- `roomId`.

`refPosition` stays readable but is no longer written or used. Records without `mapId`, or with another map's id, are listed as "photo only (saved before this map)". The 16 pilot items on the phone are in this state and must load without errors.

`PositionSource` becomes `SESSION`, `MAP`, `NONE`. Old logs contain `REFERENCE`; the evaluator must accept both.

`resolve(rec)`: a tracking same-session anchor, else `T_WM.transformPoint(mapPosition)` when located and the map id matches, else null.

Registration keeps working when not located, with a clear warning that the item is valid for this session only. The dialog shows the capture distance and suggests 0.5–1 m when it is over 1.5 m.

## 8. Guidance

Decided by a pure, unit-tested function, about 5 times a second:

| Mode | When | What the user sees |
| --- | --- | --- |
| NO_FIX | The item cannot be resolved | Room name, last-seen time, photo, and "Go to a fix point so I can locate you". Mini-map without the user dot. No arrow. |
| FAR_ROUTE | Located, not near, a walked route exists | "In the Bedroom · 7 m walk". The route on the mini-map. The arrow points to a waypoint 1.5 m ahead along the route, at camera height, drawn as a chevron and never as the ring. |
| FAR_NO_ROUTE | Located, not near, no route | "In the Bedroom · about 6 m away. No walked path yet: use the map." No arrow. |
| NEAR | See below | The ring and edge arrow on the item, as in v0.2. |

NEAR holds when any of these is true:

- the remaining route is 3 m or less;
- the user and the item are in the same room and within 5 m in a straight line;
- they are not known to be in different rooms and are within 2.5 m;
- the item has a same-session anchor, is within 4 m, and is not known to be in a different room.

Leaving NEAR needs the distance to exceed the entry limit by 1 m. Re-plan the route about once a second, and whenever the user is more than 1 m off it.

3D labels: outside a query, draw labels only for items within 6 m that are not known to be in another room. During a query, draw the target's label only in NEAR.

**Mini-map** (`MiniMapView`): top-down x–z sketch in a fixed map orientation. It shows edges, room names at the centroid of their nodes, fix points, items, the target, the route, and the user's position with heading. It fits the map's bounds automatically. Small at the top right; tap to enlarge and tap again to shrink. It is visible whenever the map has nodes, so the user can watch the trail grow while scanning. It must not pass touches to the camera view.

**Location sheet:** a single tap or a long-press on the status panel opens it. Entries: picture width, "Add a spot here", "Use a spot" (ghost image over the live view at about 45% opacity, then Lock or Cancel), rename a room, "New map". The ghost must coincide with the live view when the phone is in the saved pose, so derive its placement from the same view-to-image mapping the background renderer uses.

Status panel: second dot is "Located · Home picture · 14 m walked" or "Not located". Hints for a first-time user: no map yet → set a home point; map but not located → look at the picture or use a spot; located → name the room, walk the rooms, tap objects.

## 9. Policy and logging

`PolicyInputs` gains `metersSinceFix`, `trackingLossesSinceFix`, `msSinceFix`, `fixType`. The "reference not fully tracked" trigger is replaced by: position source is MAP and (`metersSinceFix` > `DRIFT_METERS`, initially 20, or `trackingLossesSinceFix` > 0). These are all known before any fresh capture, as the study design requires.

Keep every existing event key. Add:

| Event | New fields |
| --- | --- |
| `map_created`, `map_archived`, `fix_point_added` | ids, type, pose |
| `map_fix` | fix point, type, `kind` (first, refresh, manual), samples, position and yaw spread, camera distance, `quality`, `jumpM`, `jumpDeg`, metres and losses since the previous fix, whether the user accepted a disagreement |
| `tracking_lost`, `tracking_resumed` | reason, duration |
| `room_set`, `room_auto` | room, nodes relabelled |
| `map_saved` | node, edge, room and fix-point counts |
| `register` | `mapId`, `mapPosition`, `viewPosition`, room, all drift proxies |
| `query_start` | item room, user room, guidance mode, route length, straight distance, all drift proxies |
| `query_end` | whether NEAR was reached, time to NEAR, modes seen |
| `verify` | position source and drift proxies at capture |

`analysis/evaluate.py`: output for the existing data must not change (baseline is given in the hand-off). Add marker error grouped by position source, fix jumps (median and max), and queries by guidance mode. Keep it stdlib-only.

## 10. Code layout and threading

New code goes in new files; `FindBackRenderer` stays the glue.

- `map/Pose4.kt`, `map/WalkGraph.kt`, `map/GuidancePlanner.kt`: pure Kotlin, no Android or ARCore imports.
- `map/MapStore.kt`, `map/Localizer.kt`, `MiniMapView.kt`.

The graph and localizer change on the GL thread. The UI reads immutable snapshots published with the existing 5 Hz update. No shared mutable state without synchronization.

Tests, in `app/src/test`, run by `./gradlew testDebugUnitTest`:

- Pose4: inverse, compose and round trips; yaw from an axis; a simulated second session with a different world frame recovers every item position to within 1 mm.
- WalkGraph: an L-shaped walk through a doorway gives a route about as long as the walk and much longer than the straight line; revisits merge; no edge across a jump; label, follow and relabel rules.
- GuidancePlanner: every mode, each NEAR rule, and the hysteresis.
- MapStore: JSON round trip (add `org.json` as a test dependency; Android's is stubbed in unit tests).

## 11. Acceptance (checked by the audit)

1. `./gradlew testDebugUnitTest assembleDebug` passes.
2. `python3 analysis/evaluate.py findback_data --exclude-session cab39b87 --exclude-session 1debdfe5` prints the baseline text unchanged, followed only by the new sections.
3. The verification code path is unchanged except for the position source and the added log fields.
4. Old `records.json` loads; the pilot items show as photo-only.
5. On the phone, without walking: the app starts, the sheet opens, a spot can be created as home, the mini-map appears, an item registers with a map position and room, and after an app restart "Use a spot" brings its marker back.
6. Walk test by the user (two rooms): set home, name two rooms, register two or three items per room, restart, locate, then query across rooms. Pass means the room is named correctly, the route follows the walked path, and the marker error is logged. The error itself is a result to report, not a pass condition.
7. `FindBack/README.md` describes the new procedure and events.

## 12. Known risks

- **Yaw error grows with distance.** A 2° error at the fix point is 17 cm at 5 m. Verification crops a 22 cm window around the remembered point, so far rooms may fail to confirm an object that is present. Measure the marker error first; a wider verification search is a later, separate change because it alters the study's false-positive rate.
- **The iPad is a movable landmark.** Mitigated by the disagreement prompt and by spots.
- **Spot accuracy depends on the user.** The ghost image helps; pressing the phone against a fixed corner helps more.
- **Tracking loss in a featureless hallway** breaks the chain. Mitigated by adding a spot in the far room.

## 13. Changes made during the audit (2026-11-16)

Found on the Pixel 8 with the phone resting on its long edge:

- **Spots work at any roll.** The plan took a spot's heading from the display +X axis and rejected a rolled phone, which refused the most stable resting pose (phone on its side). A new spot now uses display +X or +Y, whichever is more horizontal (always 0.707 or more), stores the choice in `yawAxis`, and reuses it on later locks. The reuse gate is 0.5.
- **Mini-map heading** now comes from the viewing direction, or from the top of the screen when looking steeply up or down, so it is valid at any roll.
- **The lock bar moves up with toasts**, like the guide card, so the bottom buttons no longer cover Cancel and Lock.

Checked on the device with the phone still: old items load as photo-only; home spot; room naming; mini-map; the map survives a restart; "Use a spot" re-locates; the ghost coincides with the live view; a second lock from the same pose moved the map by 6 mm and 0.0002°.

Not yet checked on the device: saving an item with a map position (a still phone gives unusable depth), the picture as a fix point, routes and the chevron, the disagreement prompt, and "New map". These need the walk test.
