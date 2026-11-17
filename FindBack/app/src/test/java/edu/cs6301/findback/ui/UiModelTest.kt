package edu.cs6301.findback.ui

import edu.cs6301.findback.map.GuidanceMode
import edu.cs6301.findback.study.CaptureMode
import edu.cs6301.findback.study.GroundTruth
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class UiModelTest {
  /** Located, tracking, a room set, some items: READY. */
  private val ready = UiInput(hasMap = true, located = true, roomName = "Kitchen", itemCount = 3, hasSpots = true)

  private fun buttons(i: UiInput) = UiModel.buttons(UiModel.screen(i), i)

  private fun actions(i: UiInput): Pair<Action?, Action?> = buttons(i).let { it.secondary?.action to it.primary?.action }

  // ---------------------------------------------------------------- screen table and priority

  @Test
  fun everyRowOfTheTable() {
    assertEquals(Screen.READY, UiModel.screen(ready))
    assertEquals(Screen.NOT_LOCATED, UiModel.screen(ready.copy(located = false)))
    assertEquals(Screen.NO_MAP, UiModel.screen(UiInput()))
    assertEquals(Screen.SCANNING, UiModel.screen(ready.copy(scanning = true)))
    assertEquals(Screen.UPDATING, UiModel.screen(ready.copy(updating = true)))
    assertEquals(Screen.FINDING, UiModel.screen(ready.copy(queryActive = true)))
    assertEquals(Screen.MEASURING, UiModel.screen(ready.copy(queryActive = true, measuring = true)))
    assertEquals(Screen.LOCKING, UiModel.screen(ready.copy(lockActive = true)))
  }

  @Test
  fun firstMatchWins() {
    val all = UiInput(lockActive = true, queryActive = true, measuring = true, updating = true, scanning = true)
    assertEquals(Screen.LOCKING, UiModel.screen(all))
    assertEquals(Screen.MEASURING, UiModel.screen(all.copy(lockActive = false)))
    assertEquals(Screen.FINDING, UiModel.screen(all.copy(lockActive = false, measuring = false)))
    assertEquals(Screen.UPDATING, UiModel.screen(all.copy(lockActive = false, queryActive = false)))
    assertEquals(Screen.SCANNING, UiModel.screen(all.copy(lockActive = false, queryActive = false, updating = false)))
    // Measuring without a query is not a state of its own.
    assertEquals(Screen.READY, UiModel.screen(ready.copy(measuring = true)))
    // A search while not located, or without a map, is still FINDING.
    assertEquals(Screen.FINDING, UiModel.screen(UiInput(queryActive = true)))
    // A lock bar opened without a map (Set home here).
    assertEquals(Screen.LOCKING, UiModel.screen(UiInput(lockActive = true)))
  }

  // ---------------------------------------------------------------- instruction line

  @Test
  fun instructionPerScreen() {
    assertNull(UiModel.instruction(ready.copy(lockActive = true)))
    assertNull(UiModel.instruction(ready.copy(scanning = true)))
    assertNull(UiModel.instruction(UiInput()))
    assertEquals("Tap the true centre of \"mug\"", UiModel.instruction(ready.copy(queryActive = true, measuring = true, itemName = "mug")))
    assertEquals("Tap \"mug\" where it is now", UiModel.instruction(ready.copy(updating = true, itemName = "mug")))
    assertEquals("Tap an object to remember it", UiModel.instruction(ready))
    assertEquals("Which room is this? Tap the room name above", UiModel.instruction(ready.copy(roomName = null)))
    assertEquals(
      "Press Scan and walk slowly, or tap an object to remember it",
      UiModel.instruction(ready.copy(itemCount = 0)),
    )
    // The room question comes before the empty-list hint.
    assertEquals("Which room is this? Tap the room name above", UiModel.instruction(ready.copy(roomName = null, itemCount = 0)))
  }

  @Test
  fun notLocatedInstruction() {
    val nl = ready.copy(located = false)
    assertEquals("Go to your home spot and press Locate", UiModel.instruction(nl))
    assertEquals("Go to your home spot and press Locate or look at the picture", UiModel.instruction(nl.copy(pictureUsable = true)))
    assertEquals("Look at the reference picture to locate", UiModel.instruction(nl.copy(hasSpots = false, pictureUsable = true)))
    assertEquals("Not located: open the menu (⋮) for details", UiModel.instruction(nl.copy(hasSpots = false)))
  }

  @Test
  fun findingInstructionForEachGuidanceMode() {
    val q = ready.copy(queryActive = true, itemName = "keys")
    fun at(mode: GuidanceMode?, located: Boolean = true, spots: Boolean = true) =
      UiModel.instruction(q.copy(guidanceMode = mode, located = located, hasSpots = spots))
    assertEquals("Not located: press Locate", at(GuidanceMode.NO_FIX, located = false))
    assertEquals("Not located: press Locate", at(null, located = false))
    assertEquals("No position on the home map: look for it by its photo", at(GuidanceMode.NO_FIX))
    assertEquals("Not located", at(GuidanceMode.NO_FIX, located = false, spots = false))
    assertEquals(
      "Not located: look at the reference picture",
      UiModel.instruction(q.copy(guidanceMode = GuidanceMode.NO_FIX, located = false, hasSpots = false, pictureUsable = true)),
    )
    assertEquals("Follow the arrows along the path you walked", at(GuidanceMode.FAR_ROUTE))
    assertEquals("No path walked there yet: use the small map", at(GuidanceMode.FAR_NO_ROUTE))
    assertEquals("Far away: press Locate for a route", at(GuidanceMode.FAR_NO_ROUTE, located = false))
    assertEquals("Find \"keys\": follow the arrow to the ring", at(GuidanceMode.NEAR))
  }

  @Test
  fun overridesKeepScreenAndButtons() {
    val busy = ready.copy(busy = true)
    assertEquals("Analysing…", UiModel.instruction(busy))
    assertEquals(Screen.READY, UiModel.screen(busy))
    assertEquals(buttons(ready), buttons(busy))
    // Busy also fills screens that normally have no instruction.
    assertEquals("Analysing…", UiModel.instruction(ready.copy(busy = true, lockActive = true)))

    val lost = ready.copy(tracking = false)
    assertEquals("Tracking lost: move the phone slowly", UiModel.instruction(lost))
    assertEquals(Screen.READY, UiModel.screen(lost))
    assertEquals(buttons(ready), buttons(lost))
    val finding = ready.copy(queryActive = true, resolvable = true, guidanceMode = GuidanceMode.NEAR)
    assertEquals(buttons(finding), buttons(finding.copy(tracking = false)))
    assertEquals("Tracking lost: move the phone slowly", UiModel.instruction(finding.copy(tracking = false)))
    // Busy is shown first when both apply.
    assertEquals("Analysing…", UiModel.instruction(ready.copy(busy = true, tracking = false)))
  }

  // ---------------------------------------------------------------- buttons

  @Test
  fun buttonsPerScreen() {
    assertEquals(Action.ITEMS to Action.SCAN, actions(ready))
    assertEquals(Action.ITEMS to Action.LOCATE, actions(ready.copy(located = false)))
    assertEquals(Action.ITEMS to null, actions(ready.copy(located = false, hasSpots = false)))
    assertEquals(null to Action.SET_HOME, actions(UiInput()))
    assertEquals(null to Action.STOP_SCAN, actions(ready.copy(scanning = true)))
    assertEquals(Action.CANCEL_UPDATE to null, actions(ready.copy(updating = true)))
    assertEquals(Action.SKIP to null, actions(ready.copy(queryActive = true, measuring = true)))
    assertEquals(Action.CANCEL_LOCK to Action.LOCK, actions(ready.copy(lockActive = true)))
    assertEquals(Action.CANCEL_LOCK to Action.LOCK, actions(ready.copy(lockActive = true, queryActive = true, scanning = true)))
  }

  @Test
  fun buttonLabels() {
    val b = buttons(ready.copy(lockActive = true))
    assertEquals("Cancel", b.secondary?.label)
    assertEquals("Lock", b.primary?.label)
    assertEquals("Stop scan", buttons(ready.copy(scanning = true)).primary?.label)
    assertEquals("Set home here", buttons(UiInput()).primary?.label)
    assertEquals("Locate", buttons(ready.copy(located = false)).primary?.label)
    assertEquals("Scan", buttons(ready).primary?.label)
    assertEquals("Items", buttons(ready).secondary?.label)
  }

  @Test
  fun findingButtons() {
    val f = ready.copy(queryActive = true, resolvable = true)
    assertEquals(Action.DONE to Action.VERIFY, actions(f))
    assertEquals("Verify", buttons(f).primary?.label)
    // No position now: Verify is hidden, never greyed out.
    assertEquals(Action.DONE to null, actions(f.copy(resolvable = false)))
    // PAIRED after its capture: Done becomes the primary button.
    assertEquals(null to Action.DONE, actions(f.copy(verifyAllowed = false)))
    assertEquals(null to Action.DONE, actions(f.copy(verifyAllowed = false, resolvable = false)))
    // The star marks the policy's recommendation in study mode only.
    assertEquals("Verify ★", buttons(f.copy(studyMode = true, policyRecommends = true)).primary?.label)
    assertEquals("Verify", buttons(f.copy(studyMode = false, policyRecommends = true)).primary?.label)
    assertEquals("Verify", buttons(f.copy(studyMode = true, policyRecommends = false)).primary?.label)
  }

  // ---------------------------------------------------------------- pill

  @Test
  fun statusPill() {
    assertEquals(Pill("Kitchen", Tone.GOOD, true), UiModel.pill(ready))
    assertEquals(Pill("Which room?", Tone.WARN, true), UiModel.pill(ready.copy(roomName = null)))
    assertEquals(Pill("Not located", Tone.WARN, true), UiModel.pill(ready.copy(located = false)))
    assertEquals(Pill("No home yet", Tone.MUTED, false), UiModel.pill(UiInput()))
    assertEquals(Pill("Tracking lost", Tone.BAD, true), UiModel.pill(ready.copy(tracking = false)))
    assertEquals(Pill("Tracking lost", Tone.BAD, false), UiModel.pill(UiInput(tracking = false)))
  }

  // ---------------------------------------------------------------- study mode

  @Test
  fun studyModeGate() {
    assertFalse(UiModel.studyMode("", CaptureMode.FREE))
    assertTrue(UiModel.studyMode("home=A", CaptureMode.FREE))
    assertTrue(UiModel.studyMode("", CaptureMode.PAIRED))
    assertTrue(UiModel.studyMode("", CaptureMode.BENCHMARK))
    assertTrue(UiModel.studyMode("home=A;exp=E3", CaptureMode.PAIRED))
    assertEquals("simple", UiModel.truthPrompt(false))
    assertEquals("study", UiModel.truthPrompt(true))
  }

  @Test
  fun foundAnswersMapToGroundTruth() {
    assertEquals(GroundTruth.PRESENT, FoundAnswer.HERE.truth)
    assertEquals(GroundTruth.DISPLACED, FoundAnswer.ELSEWHERE.truth)
    assertEquals(GroundTruth.ABSENT, FoundAnswer.NOT_FOUND.truth)
    assertEquals(listOf("Yes, it's here", "It was somewhere else", "Not found"), FoundAnswer.values().map { it.label })
    // Every answer is one of the study's five values, so the analysis reads them unchanged.
    val values = GroundTruth.CHOICES.map { it.first }
    assertTrue(FoundAnswer.values().all { it.truth in values })
  }

  // ---------------------------------------------------------------- guide card

  @Test
  fun everydayGuideStatus() {
    val g = GuideInput(studyMode = false, located = true, resolvable = true, hasSpots = true)
    assertEquals(GuideStatus("Press Verify at the ring to check it is still there", Tone.NORMAL, false, false), UiModel.guideStatus(g))
    assertEquals(
      "May have moved since it was last seen: press Verify when you see the ring",
      UiModel.guideStatus(g.copy(recommend = true)).text,
    )
    val notLocated = UiModel.guideStatus(g.copy(located = false, resolvable = false))
    assertEquals("Not located: press Locate", notLocated.text)
    assertTrue(notLocated.showLocate)
    assertFalse(UiModel.guideStatus(g.copy(located = false, resolvable = false, hasSpots = false)).showLocate)
    // A same-session item can be checked while not located: the Locate button stays for the route.
    val anchorOnly = UiModel.guideStatus(g.copy(located = false, resolvable = true))
    assertEquals("Press Verify at the ring to check it is still there", anchorOnly.text)
    assertTrue(anchorOnly.showLocate)

    val ok = UiModel.guideStatus(g.copy(lastOutcome = "CONFIRMED", lastSimilarity = 0.9f))
    assertEquals(GuideStatus("✓ Still here (just checked)", Tone.GOOD, false, true), ok)
    val no = UiModel.guideStatus(g.copy(lastOutcome = "INSUFFICIENT_EVIDENCE", lastSimilarity = 0.5f, recommend = true))
    assertEquals(GuideStatus("? Could not confirm: it may still be here", Tone.WARN, false, true), no)
    // No numbers in everyday mode.
    assertFalse(ok.text.contains("0."))
  }

  @Test
  fun studyGuideStatusKeepsTheV05Lines() {
    val g = GuideInput(studyMode = true, located = true, resolvable = true, near = true, hasSpots = true)
    assertEquals("Not verified yet: aim at the ring and tap Verify", UiModel.guideStatus(g).text)
    assertEquals("Not verified yet: walk closer; Verify once the ring is on the item", UiModel.guideStatus(g.copy(near = false)).text)
    assertEquals("Verification needs a location: press Locate", UiModel.guideStatus(g.copy(resolvable = false)).text)
    assertEquals("✓ Seen here now (similarity 0.84)", UiModel.guideStatus(g.copy(lastOutcome = "CONFIRMED", lastSimilarity = 0.84f)).text)
    assertEquals(
      "? Not confirmed (0.61): insufficient evidence, may still be here",
      UiModel.guideStatus(g.copy(lastOutcome = "INSUFFICIENT_EVIDENCE", lastSimilarity = 0.61f)).text,
    )
    assertEquals(
      "Tap the object's true centre to measure marker error, or tap Skip",
      UiModel.guideStatus(g.copy(measuring = true, lastOutcome = "CONFIRMED")).text,
    )
    assertEquals(
      "Tap the item's true centre where it is now to measure the displacement, or tap Skip",
      UiModel.guideStatus(g.copy(measuring = true, displaced = true)).text,
    )
    // The study modal stays: the line never opens the photos.
    assertFalse(UiModel.guideStatus(g.copy(lastOutcome = "CONFIRMED")).opensResult)
  }

  @Test
  fun cueAndSubtitle() {
    assertEquals("2.3 m · turn left 40°", UiModel.cue(GuidanceMode.NEAR, "Kitchen", null, 2.3, true, "2.3 m · turn left 40°"))
    assertEquals("7 m walk to Kitchen", UiModel.cue(GuidanceMode.FAR_ROUTE, "Kitchen", 7.2, 4.0, true, null))
    assertEquals("7 m walk", UiModel.cue(GuidanceMode.FAR_ROUTE, null, 7.2, 4.0, true, null))
    assertEquals("About 6 m away in Hall", UiModel.cue(GuidanceMode.FAR_NO_ROUTE, "Hall", null, 5.8, true, null))
    assertEquals("No direction until located", UiModel.cue(GuidanceMode.NO_FIX, "Hall", null, null, false, null))
    assertEquals("No position on the map: use its photo", UiModel.cue(null, null, null, null, true, null))
    assertEquals("in Kitchen · last seen 2 d ago", UiModel.guideSubtitle("Kitchen", "2 d ago"))
    assertEquals("last seen 5 min ago", UiModel.guideSubtitle(null, "5 min ago"))
  }

  // ---------------------------------------------------------------- scan and items

  @Test
  fun scanTexts() {
    assertEquals("Scanning · 14 items", UiModel.scanTitle(14))
    assertEquals("Scanning · 1 item", UiModel.scanTitle(1))
    assertEquals("Walk slowly", UiModel.scanHint(located = true, lockActive = false, waitingForDepth = false))
    assertEquals("Move the phone sideways so depth can start", UiModel.scanHint(true, false, true))
    assertEquals("Paused for the spot lock", UiModel.scanHint(true, true, true))
    assertEquals("Paused: not located. Locate from the menu (⋮)", UiModel.scanHint(false, true, true))
    assertEquals(
      listOf("12 new items · 5 seen again", "3 not seen where they were before", "4 glimpsed once and ignored"),
      UiModel.scanSummaryLines(12, 5, 3, 4),
    )
    assertEquals(listOf("1 new item · 0 seen again"), UiModel.scanSummaryLines(1, 0, 0, 0))
    assertEquals("42 s · 18 m walked · 30 frames · depth check 3.1 cm", UiModel.scanSummaryDetails(42_000, 18.2, 30, 0.031))
    assertEquals("5 s · 0 m walked · 1 frame", UiModel.scanSummaryDetails(5_000, 0.2, 1, null))
  }

  @Test
  fun itemsSearchAndEmptyTexts() {
    assertTrue(UiModel.roomMatches("Living room", "living"))
    assertTrue(UiModel.roomMatches("Living room", "  ROOM "))
    assertFalse(UiModel.roomMatches(null, "kitchen"))
    assertFalse(UiModel.roomMatches("Bedroom", "kitchen"))
    assertTrue(UiModel.roomMatches(null, " "))
    assertEquals("No items yet. Press Scan, or tap an object in the camera view.", UiModel.itemsEmptyText(0, 0, "x", null))
    assertNull(UiModel.itemsEmptyText(4, 2, "mug", null))
    assertEquals("Nothing matches \"mug\".", UiModel.itemsEmptyText(4, 0, " mug ", null))
    assertEquals("Nothing matches \"mug\" in Kitchen.", UiModel.itemsEmptyText(4, 0, "mug", "Kitchen"))
    assertEquals("No items in Kitchen yet.", UiModel.itemsEmptyText(4, 0, "", "Kitchen"))
  }
}
