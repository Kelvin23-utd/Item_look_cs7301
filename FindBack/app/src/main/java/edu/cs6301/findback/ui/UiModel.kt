package edu.cs6301.findback.ui

import edu.cs6301.findback.map.GuidanceMode
import edu.cs6301.findback.study.CaptureMode
import edu.cs6301.findback.study.GroundTruth
import edu.cs6301.findback.study.StudySettings
import java.util.Locale
import kotlin.math.roundToInt

/**
 * v0.6 (docs/PLAN_v0.6_ui.md, section 4): which screen the app is on, its one instruction line and
 * its bottom buttons, decided from one snapshot of the app state. Pure Kotlin, no Android imports,
 * unit-tested in UiModelTest.
 */
enum class Screen { LOCKING, MEASURING, FINDING, UPDATING, SCANNING, NO_MAP, NOT_LOCATED, READY }

/** What a bottom-bar button does. Each button keeps one meaning. */
enum class Action { ITEMS, LOCATE, SCAN, STOP_SCAN, SET_HOME, LOCK, CANCEL_LOCK, DONE, VERIFY, SKIP, CANCEL_UPDATE }

data class UiButton(val action: Action, val label: String)

/** The secondary button (left, outlined) and the primary button (right, filled); null is not shown. */
data class BottomButtons(val secondary: UiButton?, val primary: UiButton?)

enum class Tone { NORMAL, GOOD, WARN, BAD, MUTED }

/** The status pill at the top left. [opensRoomPicker]: a tap picks the room (else it opens the menu). */
data class Pill(val text: String, val tone: Tone, val opensRoomPicker: Boolean)

data class UiInput(
  /** The spot lock bar is open (Locate, Set home here, Add a spot here). */
  val lockActive: Boolean = false,
  val queryActive: Boolean = false,
  /** Study mode only: after the ground truth, the next tap measures. */
  val measuring: Boolean = false,
  /** "Moved it" (Update location) is waiting for a tap. */
  val updating: Boolean = false,
  val scanning: Boolean = false,
  val hasMap: Boolean = false,
  val located: Boolean = false,
  val tracking: Boolean = true,
  /** A capture is being analysed. */
  val busy: Boolean = false,
  /** Name of the current room, or null when no room is set. */
  val roomName: String? = null,
  val itemCount: Int = 0,
  /** Looking at the reference picture locates (the map has it as a fix point at this width). */
  val pictureUsable: Boolean = false,
  /** The map has at least one spot, so Locate works. */
  val hasSpots: Boolean = false,
  /** FINDING: the guidance mode of the query, or null before the first plan. */
  val guidanceMode: GuidanceMode? = null,
  /** The item of the query or of "Moved it". */
  val itemName: String? = null,
  /** FINDING: the query's study mode (fixed at its start). */
  val studyMode: Boolean = false,
  /** FINDING: another capture is allowed (PAIRED allows one). */
  val verifyAllowed: Boolean = true,
  /** FINDING: the item has a position right now, so Verify can capture. */
  val resolvable: Boolean = false,
  /** FINDING, study mode: the policy recommends verifying (Verify ★). */
  val policyRecommends: Boolean = false,
) {
  val roomSet: Boolean
    get() = roomName != null
}

/** Everything the guide card's status line depends on. */
data class GuideInput(
  val studyMode: Boolean,
  /** Study mode: the measuring tap is next. */
  val measuring: Boolean = false,
  /** Study mode, measuring: the ground truth was "displaced". */
  val displaced: Boolean = false,
  /** The query's last capture outcome ("CONFIRMED", "INSUFFICIENT_EVIDENCE") or null. */
  val lastOutcome: String? = null,
  val lastSimilarity: Float? = null,
  val located: Boolean = false,
  val resolvable: Boolean = false,
  val near: Boolean = false,
  val hasSpots: Boolean = false,
  /** The policy recommended verifying, with a position at query start. */
  val recommend: Boolean = false,
)

/**
 * The guide card's status line. [showLocate]: the card shows a Locate button. [opensResult]: a tap
 * on the line shows both photos of the last capture (everyday mode).
 */
data class GuideStatus(val text: String, val tone: Tone, val showLocate: Boolean, val opensResult: Boolean)

/** Everyday-mode Done: "Did you find it?" and the ground truth each answer logs. */
enum class FoundAnswer(val label: String, val truth: String) {
  HERE("Yes, it's here", GroundTruth.PRESENT),
  ELSEWHERE("It was somewhere else", GroundTruth.DISPLACED),
  NOT_FOUND("Not found", GroundTruth.ABSENT),
}

object UiModel {
  /** `query_end.truthPrompt`. */
  const val TRUTH_PROMPT_SIMPLE = "simple"
  const val TRUTH_PROMPT_STUDY = "study"

  const val OUTCOME_CONFIRMED = "CONFIRMED"

  /** Section 3: study mode is on when a study tag is set or the capture mode is not FREE. */
  fun studyMode(studyTag: String, captureMode: CaptureMode): Boolean =
    StudySettings.statusLine(studyTag, captureMode) != null

  fun truthPrompt(studyMode: Boolean): String = if (studyMode) TRUTH_PROMPT_STUDY else TRUTH_PROMPT_SIMPLE

  /** First matching row of the section 4 table. */
  fun screen(i: UiInput): Screen =
    when {
      i.lockActive -> Screen.LOCKING
      i.queryActive && i.measuring -> Screen.MEASURING
      i.queryActive -> Screen.FINDING
      i.updating -> Screen.UPDATING
      i.scanning -> Screen.SCANNING
      !i.hasMap -> Screen.NO_MAP
      !i.located -> Screen.NOT_LOCATED
      else -> Screen.READY
    }

  /** The one instruction line, or null when the screen's card already says it. */
  fun instruction(i: UiInput): String? {
    if (i.busy) return "Analysing…"
    if (!i.tracking) return "Tracking lost: move the phone slowly"
    val name = i.itemName ?: "the item"
    return when (screen(i)) {
      Screen.LOCKING, Screen.SCANNING, Screen.NO_MAP -> null
      Screen.MEASURING -> "Tap the true centre of \"$name\""
      Screen.FINDING -> findingInstruction(i.guidanceMode, name, i.located, i.hasSpots, i.pictureUsable)
      Screen.UPDATING -> "Tap \"$name\" where it is now"
      Screen.NOT_LOCATED ->
        when {
          i.hasSpots -> "Go to your home spot and press Locate" + if (i.pictureUsable) " or look at the picture" else ""
          i.pictureUsable -> "Look at the reference picture to locate"
          else -> "Not located: open the menu (⋮) for details"
        }
      Screen.READY ->
        when {
          !i.roomSet -> "Which room is this? Tap the room name above"
          i.itemCount == 0 -> "Press Scan and walk slowly, or tap an object to remember it"
          else -> "Tap an object to remember it"
        }
    }
  }

  /** Section 7: the FINDING instruction for each guidance mode. */
  fun findingInstruction(mode: GuidanceMode?, name: String, located: Boolean, hasSpots: Boolean, pictureUsable: Boolean): String =
    when (mode ?: GuidanceMode.NO_FIX) {
      GuidanceMode.NO_FIX ->
        when {
          located -> "No position on the home map: look for it by its photo"
          hasSpots -> "Not located: press Locate"
          pictureUsable -> "Not located: look at the reference picture"
          else -> "Not located"
        }
      GuidanceMode.FAR_ROUTE -> "Follow the arrows along the path you walked"
      GuidanceMode.FAR_NO_ROUTE ->
        if (located) "No path walked there yet: use the small map" else "Far away: press Locate for a route"
      GuidanceMode.NEAR -> "Find \"$name\": follow the arrow to the ring"
    }

  /** Section 4: the bottom buttons. Only buttons that work now are returned. */
  fun buttons(s: Screen, i: UiInput): BottomButtons =
    when (s) {
      Screen.LOCKING -> BottomButtons(UiButton(Action.CANCEL_LOCK, "Cancel"), UiButton(Action.LOCK, "Lock"))
      Screen.MEASURING -> BottomButtons(UiButton(Action.SKIP, "Skip"), null)
      Screen.FINDING -> {
        val done = UiButton(Action.DONE, "Done")
        when {
          // PAIRED after its capture: Done is the next step.
          !i.verifyAllowed -> BottomButtons(null, done)
          // No position because not located: locating is the next step, so it is the primary button.
          !i.resolvable && !i.located && i.hasSpots -> BottomButtons(done, UiButton(Action.LOCATE, "Locate"))
          // No position now: a capture cannot be taken.
          !i.resolvable -> BottomButtons(done, null)
          else -> BottomButtons(done, UiButton(Action.VERIFY, if (i.studyMode && i.policyRecommends) "Verify ★" else "Verify"))
        }
      }
      Screen.UPDATING -> BottomButtons(UiButton(Action.CANCEL_UPDATE, "Cancel"), null)
      Screen.SCANNING -> BottomButtons(null, UiButton(Action.STOP_SCAN, "Stop scan"))
      Screen.NO_MAP -> BottomButtons(null, UiButton(Action.SET_HOME, "Set home here"))
      Screen.NOT_LOCATED ->
        BottomButtons(UiButton(Action.ITEMS, "Items"), if (i.hasSpots) UiButton(Action.LOCATE, "Locate") else null)
      Screen.READY -> BottomButtons(UiButton(Action.ITEMS, "Items"), UiButton(Action.SCAN, "Scan"))
    }

  /** Section 5.1: the status pill. Tracking lost overrides the location. */
  fun pill(i: UiInput): Pill =
    when {
      !i.tracking -> Pill("Tracking lost", Tone.BAD, i.hasMap)
      !i.hasMap -> Pill("No home yet", Tone.MUTED, false)
      // Not located: a tap locates (MainView), so no room-picker arrow.
      !i.located -> Pill("Not located", Tone.WARN, false)
      i.roomName == null -> Pill("Which room?", Tone.WARN, true)
      else -> Pill(i.roomName, Tone.GOOD, true)
    }

  /**
   * Section 6.2: the guide card's status line. Study mode keeps the v0.5 lines (with similarity);
   * everyday mode uses plain words and shows the capture result here instead of a modal.
   */
  fun guideStatus(g: GuideInput): GuideStatus {
    val locate = !g.located && g.hasSpots
    if (g.studyMode) {
      return when {
        g.measuring ->
          GuideStatus(
            if (g.displaced) "Tap the item's true centre where it is now to measure the displacement, or tap Skip"
            else "Tap the object's true centre to measure marker error, or tap Skip",
            Tone.NORMAL,
            false,
            false,
          )
        g.lastOutcome == OUTCOME_CONFIRMED ->
          GuideStatus("✓ Seen here now (similarity %.2f)".format(Locale.US, g.lastSimilarity ?: 0f), Tone.GOOD, locate, false)
        g.lastOutcome != null ->
          GuideStatus(
            "? Not confirmed (%.2f): insufficient evidence, may still be here".format(Locale.US, g.lastSimilarity ?: 0f),
            Tone.WARN,
            locate,
            false,
          )
        // Locate is then the bottom bar's primary button, not a card button.
        !g.resolvable -> GuideStatus("Verification needs a location: press Locate", Tone.MUTED, false, false)
        g.near -> GuideStatus("Not verified yet: aim at the ring and tap Verify", Tone.NORMAL, locate, false)
        else -> GuideStatus("Not verified yet: walk closer; Verify once the ring is on the item", Tone.NORMAL, locate, false)
      }
    }
    return when {
      g.lastOutcome == OUTCOME_CONFIRMED -> GuideStatus("✓ Still here (just checked)", Tone.GOOD, locate, true)
      g.lastOutcome != null -> GuideStatus("? Could not confirm: it may still be here", Tone.WARN, locate, true)
      // The instruction line already says it and Locate is the primary button: no status line.
      !g.resolvable && !g.located -> GuideStatus("", Tone.MUTED, false, false)
      !g.resolvable -> GuideStatus("No position on the home map, so it cannot be checked", Tone.MUTED, false, false)
      g.recommend -> GuideStatus("May have moved since it was last seen: press Verify when you see the ring", Tone.WARN, locate, false)
      else -> GuideStatus("Press Verify at the ring to check it is still there", Tone.NORMAL, locate, false)
    }
  }

  /**
   * The guide card's cue line: "2.3 m · turn left 40°" when near ([nearText], built from the camera
   * pose), else the walk or straight distance and the room.
   */
  fun cue(mode: GuidanceMode?, roomName: String?, routeM: Double?, straightM: Double?, located: Boolean, nearText: String?): String {
    val room = roomName?.takeIf { it.isNotBlank() }?.trim()
    return when (mode ?: GuidanceMode.NO_FIX) {
      GuidanceMode.NEAR -> nearText ?: (room?.let { "Follow the arrow to $it" } ?: "Follow the arrow to the ring")
      GuidanceMode.FAR_ROUTE -> "${routeM?.roundToInt() ?: "?"} m walk" + (room?.let { " to $it" } ?: "")
      GuidanceMode.FAR_NO_ROUTE -> "About ${straightM?.roundToInt() ?: "?"} m away" + (room?.let { " in $it" } ?: "")
      GuidanceMode.NO_FIX -> if (located) "No position on the map: use its photo" else "No direction until located"
    }
  }

  /** "in Kitchen · last seen 2 d ago", or "last seen 2 d ago" when the room is not known. */
  fun guideSubtitle(roomName: String?, ago: String): String =
    (roomName?.takeIf { it.isNotBlank() }?.let { "in ${it.trim()} · " } ?: "") + "last seen $ago"

  /** Section 5.2: the scan bar's first line. */
  fun scanTitle(items: Int): String = "Scanning · " + if (items == 1) "1 item" else "$items items"

  /** Section 5.2: the scan bar's hint. */
  fun scanHint(located: Boolean, lockActive: Boolean, waitingForDepth: Boolean): String =
    when {
      !located -> "Paused: not located. Locate from the menu (⋮)"
      lockActive -> "Paused for the spot lock"
      waitingForDepth -> "Move the phone sideways so depth can start"
      else -> "Walk slowly"
    }

  /** Section 6.7: the scan summary's main lines. */
  fun scanSummaryLines(newCount: Int, seenAgain: Int, missed: Int, tentativeDropped: Int): List<String> {
    val out = ArrayList<String>()
    out.add(count(newCount, "new item", "new items") + " · " + "$seenAgain seen again")
    if (missed > 0) out.add("$missed not seen where they were before")
    if (tentativeDropped > 0) out.add("$tentativeDropped glimpsed once and ignored")
    return out
  }

  /** Section 6.7: the scan summary's small details line. */
  fun scanSummaryDetails(durationMs: Long, metersWalked: Double, keyframes: Int, depthCheckMedianM: Double?): String {
    val parts =
      mutableListOf(
        "%.0f s".format(Locale.US, durationMs / 1000.0),
        "%.0f m walked".format(Locale.US, metersWalked),
        count(keyframes, "frame", "frames"),
      )
    depthCheckMedianM?.let { parts.add("depth check %.1f cm".format(Locale.US, it * 100)) }
    return parts.joinToString(" · ")
  }

  /** Section 6.5: the Items search also matches room names (case-insensitive, trimmed). */
  fun roomMatches(roomName: String?, query: String): Boolean {
    val q = query.trim().lowercase(Locale.ROOT)
    if (q.isEmpty()) return true
    return roomName?.lowercase(Locale.ROOT)?.contains(q) == true
  }

  /** Section 6.5: the Items list's message when no row is shown, or null when rows are shown. */
  fun itemsEmptyText(total: Int, shown: Int, query: String, roomName: String?): String? =
    when {
      total == 0 -> "No items yet. Press Scan, or tap an object in the camera view."
      shown > 0 -> null
      query.isNotBlank() -> "Nothing matches \"" + query.trim() + "\"" + (roomName?.let { " in $it" } ?: "") + "."
      roomName != null -> "No items in $roomName yet."
      else -> "No items."
    }

  private fun count(n: Int, one: String, many: String) = if (n == 1) "1 $one" else "$n $many"
}
