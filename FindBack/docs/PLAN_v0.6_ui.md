# FindBack v0.6: a simpler screen

Status: plan, 2026-11-16, git branch `scan-once`, on top of v0.5 Block A (commit `33e0e20`).
The user's feedback: "UI logic is kinda confusing, it is hard to use."

This version changes only the user interface. Map, localizer, scan pipeline, verification, policy,
stores and every existing log event stay as they are.

## 1. What is wrong now

1. **Locating is hidden.** Every session starts with locating at the home spot, but it is reached
   only by tapping the status panel, then "Use a spot". The panel is both a status display and a
   hidden menu.
2. **The bottom bar changes under the user.** Verify and Done are greyed out most of the time; Scan
   appears and disappears and resizes the other buttons; Done means Done, Skip or Cancel depending
   on the state; and every message (Snackbar) pushes the bar up, so taps land on the wrong button.
3. **Too much text, with internal words:** fix point, trail points, metres walked, similarity 0.84,
   "historical location", policy reasons, "Detector: cup (0.62) · crop: mlkit · plane".
4. **Study questions in everyday use.** After every search the five-way ground truth is asked, then
   a measuring tap; registration asks how often the item moves. These belong to the study only.
5. **Stacked dialogs.** Items list → item → Rename/Delete; a modal result after every Verify; the
   search field in the Items dialog needs a window-flag workaround to get a keyboard.

## 2. Principles

- **One screen state at a time**, decided by one pure function (section 4), unit-tested.
- **Only buttons that work now are shown.** No greyed-out buttons. The primary action is on the
  right and filled; the secondary is on the left and outlined. A button keeps one meaning.
- **One instruction line** says the next step in plain words. Short messages appear in their own
  banner at the top and never move the buttons.
- **Study controls appear only in study mode** and then behave exactly as now.

## 3. Study mode

`studyMode = StudySettings.statusLine(studyTag, captureMode) != null`, that is, a study tag is set
or the capture mode is not FREE. The analysis ignores untagged sessions and the experiment protocol
always sets a tag, so everyday use can be simplified without touching the study.

A query fixes `studyMode` at its start (the study settings cannot change during a query already).

| | Study mode (unchanged from v0.5) | Everyday mode (new) |
| --- | --- | --- |
| Done | Five-way ground truth, then the measuring tap for present/displaced; PAIRED capture-first prompt | "Did you find it?" (section 6.3); no measuring tap |
| Verify result | Modal with both photos; BENCHMARK label after it | Shown in the guide card; tap it to see both photos |
| Policy | Policy line with reasons | One plain line only when the policy recommends verifying |
| Registration | Debug line (detector, crop source, hit type), mobility choice | Photo, name, room; warnings in plain words; mobility = record's value on update, else `Mobility.DEFAULT` |
| Status | Amber study line with tag and mode under the status row | Nothing |

## 4. Screen states

A new pure-Kotlin file `ui/UiModel.kt` (no Android or ARCore imports) holds:

```kotlin
enum class Screen { LOCKING, MEASURING, FINDING, UPDATING, SCANNING, NO_MAP, NOT_LOCATED, READY }

data class UiInput(
  val lockActive: Boolean, val queryActive: Boolean, val measuring: Boolean, val updating: Boolean,
  val scanning: Boolean, val hasMap: Boolean, val located: Boolean, val tracking: Boolean,
  val busy: Boolean, val roomSet: Boolean, val itemCount: Int, val pictureUsable: Boolean,
  val hasSpots: Boolean, val guidanceMode: GuidanceMode?, val itemName: String?, /* … */
)

object UiModel {
  fun screen(i: UiInput): Screen            // first matching row of the table below
  fun instruction(i: UiInput): String?      // the one instruction line, or null
  fun buttons(s: Screen, i: UiInput): BottomButtons  // secondary and primary, each nullable
}
```

`GuidanceMode` lives in `map/`; if `UiModel` cannot import it without Android, pass a plain enum or
string. Keep `UiModel` testable on the JVM.

Priority, first match wins:

| Screen | When | Bottom area | Instruction line |
| --- | --- | --- | --- |
| LOCKING | spot lock bar open | Lock bar: its text, [Cancel] [Lock] | none (the bar has the text) |
| MEASURING | query and measuring (study mode only) | Guide card, [Skip] | "Tap the true centre of "X"" |
| FINDING | query | Guide card, [Done] [Verify]; when not located a [Locate] button inside the card | from the guidance mode, as now but in plain words (section 7) |
| UPDATING | Update location | Guide card, [Cancel] | "Tap "X" where it is now" |
| SCANNING | scan running | Scan bar: "Scanning · 14 items", hint, [Stop scan] | none (the bar has the hint) |
| NO_MAP | no map | Welcome card, [Set home here] | none |
| NOT_LOCATED | map, not located | [Items] [Locate] | "Go to your home spot and press Locate" (+ " or look at the picture" when usable) |
| READY | located | [Items] [Scan] | room not set: "Which room is this? Tap the room name above"; no items: "Press Scan and walk slowly, or tap an object to remember it"; else "Tap an object to remember it" |

Overrides that keep the screen and buttons: `busy` (a capture is being analysed) sets the
instruction to "Analysing…"; tracking lost turns the status pill red (section 5.1) and sets the
instruction to "Tracking lost: move the phone slowly".

Camera taps go to the renderer only in READY, UPDATING and MEASURING, as now. In other states the
banner explains why nothing happened ("Press Stop scan first", "Press Done to end this search",
"Line up the view, then press Lock").

## 5. Layout

No theme change: the activity stays `Theme.AppCompat.Light.NoActionBar`, and buttons are plain
`Button`s with two new drawables (`btn_primary.xml` filled, `btn_secondary.xml` outlined), at least
52 dp high, text not all caps. Remove `layout_dodgeInsetEdges` everywhere and stop using
`SnackbarHelper` in `MainView`.

### 5.1 Top

1. **Status row.** On the left, a status pill (wraps its content):
   - located: "● Living room ▾" (green dot), or "● Which room? ▾" (amber) when no room is set;
   - map, not located: "● Not located" (amber);
   - no map: "● No home yet" (grey);
   - tracking lost overrides: "● Tracking lost" (red).

   Tapping the pill opens the room picker when a map exists, otherwise the menu. On the right, a
   48 dp menu button (⋮, or the existing `ic_settings.png`) opens the Home sheet (section 6.4).
2. **Study line**, study mode only: small amber text with the tag and mode (existing
   `StudySettings.statusLine`).
3. **Instruction line**: one panel with the instruction from `UiModel.instruction`; hidden when null.
4. **Message banner**: replaces the Snackbar. `MainView.toast(message)` keeps its signature (the
   localizer, scan controller and renderer all call it) and shows the text in a banner below the
   instruction line for 3 s (5 s when longer than 60 characters); a tap dismisses it; a new message
   replaces the old one.
5. **Mini-map** at the right below, as now.

### 5.2 Bottom

One bottom bar with the two buttons of `UiModel.buttons`, plus one card above it depending on the
screen: lock bar, guide card, scan bar or welcome card. Positions never change with messages.

- **Guide card** (FINDING, UPDATING, MEASURING): thumbnail 64 dp; name (18 sp bold); "in Kitchen ·
  last seen 2 d ago" (13 sp); cue (17 sp, "2.3 m · turn left 40°", "Follow the arrow to Kitchen");
  one status line (section 6.2). Study mode adds the existing policy line.
- **Scan bar** (SCANNING): "Scanning · 14 items" (bold), the scan hint ("Walk slowly", "Move the
  phone sideways so depth can start", "Paused: not located"), and [Stop scan] as the primary button.
- **Welcome card** (NO_MAP): "Set up your home" and three short lines: "1. Stand where you can return
  exactly, for example with the phone against a door frame. 2. Press Set home here, hold still,
  press Lock. 3. Walk around and press Scan." Plus "Or show the reference picture to the camera."
  when the picture is usable. Primary button [Set home here] starts the existing add-spot flow with
  `home = true`.

## 6. Flows

### 6.1 Locate

[Locate] (bottom bar, guide card or Home sheet): one spot → its lock bar with the ghost at once;
several spots → "Locate at which spot?" list, home spot first; no spots → the banner says so.
Lock-bar texts in plain words: "Line up the camera with the faint picture, hold still, then press
Lock." On failure the bar stays, as now.

### 6.2 Find

Items → row → item sheet → [Find] starts the existing query. The guide card's status line, in
everyday mode:

- before any capture: when the policy recommends verifying, "May have moved since it was last seen:
  press Verify when you see the ring"; otherwise "Press Verify at the ring to check it is still
  there";
- not located: "Not located: press Locate" with the [Locate] button in the card;
- after a capture: "✓ Still here (just checked)" in green, or "? Could not confirm: it may still be
  here" in amber. Tapping this line opens the existing two-photo dialog (without the BENCHMARK
  label).

Study mode keeps the current lines, the policy line and the modal result.

### 6.3 Done in everyday mode

Done opens "Did you find it?" with:

| Answer | Ground truth passed to `endQuery` | Then |
| --- | --- | --- |
| "Yes, it's here" | `PRESENT` | query ends, no measuring tap |
| "It was somewhere else" | `DISPLACED` | query ends; for a tapped record, start Update location for it; for a scanned entry, the banner says "Scan that area again to update it" |
| "Not found" | `ABSENT` | query ends |
| Back or "Keep looking" | none | the search continues, nothing logged |

`query_end` gains `"truthPrompt": "simple"` or `"study"`. Nothing else in its fields changes. The
measuring tap is never entered in everyday mode.

### 6.4 Home sheet (menu button)

Title "Home". A plain status line ("Located at the spot "living room"", "Not located", "No home
yet"), then a small grey line with details (map id, rooms, spots, trail points, metres since the
fix). Entries, each shown only when it works now:

1. Locate (when spots exist)
2. Set home here (no map) or Add a spot here (located)
3. Rooms… (rename; when rooms exist)
4. Settings… (picture width; "Save scan frames", moved here from Study settings)
5. Study settings… (tag and capture mode only; unchanged otherwise)
6. Start a new map… (last; same confirmation as now)

### 6.5 Items screen

Replace the AlertDialog list with a full-screen panel inside `activity_main.xml` (opaque
background, hidden by default), so the keyboard works without window flags:

- Header: "Items" and a close button. Back closes the panel.
- Search field: matches names and labels as now (`InventoryText.matches`), and also room names.
- Room filter chips ("All", then each room) in a horizontal scroll row when there are two or more
  rooms.
- Rows: thumbnail, name in bold, second line "Kitchen · seen 2 d ago". Scanned entries use the
  existing `InventoryText` location and evidence text; tapped records use "Kitchen · saved 2 d ago",
  "this session only" or "photo only". Scanned rows end with a small grey "from a scan".
- Empty list: "No items yet. Press Scan, or tap an object in the camera view." Empty search: "Nothing
  matches "x"."
- A row opens the item sheet (a dialog is fine): large photo, name, location line, evidence line,
  primary [Find], [Close], and a row of text buttons "Rename", "Moved it" (tapped records only; the
  existing Update location), "Delete" (with the existing confirmation).
- The list keeps the current sort (newest first). Opening it during a scan shows the banner "Stop
  the scan first", as now.

### 6.6 Registration (tap an object)

The dialog shows the photo, then the name field pre-filled with the default name and fully
selected, then the room row. `typed` keeps its meaning: true only when the saved name differs from
the default (new item) or from the record's name (update). Plain-word warnings stay ("2.1 m away:
stand 0.5–1 m from the object for a reliable photo", "Not located: this item is saved for this
session only. Press Locate first to put it on the home map."). Study mode adds the current debug
line and the mobility choice.

### 6.7 Scan summary

Title "Scan finished". Lines: "12 new items · 5 seen again"; when missed > 0, "3 not seen where they
were before"; when tentative > 0, "4 glimpsed once and ignored"; a small grey details line with
seconds, metres walked, frames and "depth check 3.1 cm" when present. Buttons [See items] [OK].

## 7. Wording

Use these words in all user-visible text, including toasts from the localizer and renderer:

| Now | v0.6 |
| --- | --- |
| fix point, "tap here → Use a spot" | spot; "press Locate" |
| "Located via X" | "Located at "X"" |
| "Re-fixed at X: moved 4 cm, 2°" | "Located again at "X" (moved 4 cm)" |
| "Map position lost: look at the picture or use a spot again." | "Lost the map position: press Locate at your home spot." |
| "historical location" | "where it was last seen" |
| similarity numbers, thresholds | study mode only |
| "Follow the chevron along your walked path" | "Follow the arrows along the path you walked" |
| "No walked path yet: use the map (top right)" | "No path walked there yet: use the small map" |

Instruction texts for FINDING by guidance mode: NO_FIX "Not located: press Locate"; FAR_ROUTE "Follow
the arrows along the path you walked"; FAR_NO_ROUTE located "No path walked there yet: use the small
map", not located "Far away: press Locate for a route"; NEAR "Find "X": follow the arrow to the
ring".

## 8. Back button

Via `OnBackPressedDispatcher`: Items panel open → close it; lock bar open → Cancel; FINDING →
banner "Press Done to end the search"; SCANNING → banner "Press Stop scan first"; UPDATING →
Cancel; otherwise the default (leave the app).

## 9. Must not change

- Study-mode behaviour and every existing event name and field, except the added `truthPrompt` on
  `query_end`.
- `VerificationPolicy`, thresholds, the localizer's logic, the scan pipeline, stores and file
  formats. Only user-visible strings change in `Localizer` and `ScanController`.
- Which taps reach the renderer, and the GL-thread rules: no dialog blocks the GL thread; UI work
  stays on the UI thread.
- User-typed text is never used as a format string.

## 10. Tests and acceptance

1. `UiModelTest`: every row of the section 4 table and its priority; both overrides; the
   instruction for each guidance mode; buttons per screen; the study-mode gate; the "Did you find
   it?" answers map to `PRESENT`, `DISPLACED`, `ABSENT`.
2. `./gradlew testDebugUnitTest assembleDebug` passes (137 tests before this work).
3. In study mode, a query produces the same events and dialogs as v0.5 Block A.
4. On the phone: no greyed-out button is ever visible; a message never moves the bottom buttons;
   Locate is one tap from the bottom bar when not located; the Items search opens the keyboard.
5. `FindBack/README.md` and `paper/EXPERIMENT_PROTOCOL.md` use the new names (menu button instead of
   "status panel", "Locate" instead of "Use a spot", "Moved it" for Update location). The protocol's
   steps must stay the same otherwise.
