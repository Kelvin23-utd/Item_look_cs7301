package edu.cs6301.findback.map

import edu.cs6301.findback.map.GuidanceMode.FAR_NO_ROUTE
import edu.cs6301.findback.map.GuidanceMode.FAR_ROUTE
import edu.cs6301.findback.map.GuidanceMode.NEAR
import edu.cs6301.findback.map.GuidanceMode.NO_FIX
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GuidancePlannerTest {
  private fun input(
    resolved: Boolean = true,
    located: Boolean = true,
    anchor: Boolean = false,
    straight: Double? = 6.0,
    route: Double? = null,
    user: String? = null,
    item: String? = null,
  ) = GuidanceInput(resolved, located, anchor, straight, route, user, item)

  private fun mode(i: GuidanceInput, prev: GuidanceMode? = null) = GuidancePlanner.decide(i, prev).mode

  @Test
  fun noFixWhenTheItemCannotBeResolved() {
    assertEquals(NO_FIX, mode(input(resolved = false, located = false, straight = null)))
    assertEquals(NO_FIX, mode(input(resolved = false, located = true, straight = null, route = 1.0)))
  }

  @Test
  fun farModes() {
    // Through a wall: 2 m straight, different rooms, 8 m walked route.
    assertEquals(FAR_ROUTE, mode(input(straight = 2.0, route = 8.0, user = "hall", item = "bed")))
    // Located but nothing walked yet.
    assertEquals(FAR_NO_ROUTE, mode(input(straight = 6.0, route = null, user = "hall", item = "bed")))
    // Same-session anchor but not located and far: no route possible.
    assertEquals(FAR_NO_ROUTE, mode(input(located = false, anchor = true, straight = 7.0)))
  }

  @Test
  fun nearByRemainingRoute() {
    val d = GuidancePlanner.decide(input(straight = 2.9, route = 3.0, user = "a", item = "b"), null)
    assertEquals(NEAR, d.mode)
    assertEquals(GuidancePlanner.RULE_ROUTE, d.nearRule)
    assertEquals(FAR_ROUTE, mode(input(straight = 2.9, route = 3.1, user = "a", item = "b")))
  }

  @Test
  fun nearBySameRoom() {
    val d = GuidancePlanner.decide(input(straight = 5.0, route = 9.0, user = "bed", item = "bed"), null)
    assertEquals(NEAR, d.mode)
    assertEquals(GuidancePlanner.RULE_SAME_ROOM, d.nearRule)
    assertEquals(FAR_ROUTE, mode(input(straight = 5.1, route = 9.0, user = "bed", item = "bed")))
    // Same distance, but the rooms differ.
    assertEquals(FAR_ROUTE, mode(input(straight = 4.5, route = 9.0, user = "hall", item = "bed")))
  }

  @Test
  fun nearWhenNotKnownToBeInDifferentRooms() {
    // Either room unknown counts as "not known to be different".
    for ((u, i) in listOf(null to null, "hall" to null, null to "bed")) {
      val d = GuidancePlanner.decide(input(straight = 2.5, route = 9.0, user = u, item = i), null)
      assertEquals(NEAR, d.mode)
      assertEquals(GuidancePlanner.RULE_OPEN, d.nearRule)
    }
    assertEquals(FAR_ROUTE, mode(input(straight = 2.6, route = 9.0)))
    assertEquals(FAR_ROUTE, mode(input(straight = 1.0, route = 9.0, user = "hall", item = "bed")))
  }

  @Test
  fun nearBySessionAnchor() {
    val d = GuidancePlanner.decide(input(located = false, anchor = true, straight = 4.0), null)
    assertEquals(NEAR, d.mode)
    assertEquals(GuidancePlanner.RULE_SESSION_ANCHOR, d.nearRule)
    assertEquals(FAR_NO_ROUTE, mode(input(located = false, anchor = true, straight = 4.1)))
    // Not when known to be in another room.
    assertEquals(FAR_ROUTE, mode(input(anchor = true, straight = 3.0, route = 7.0, user = "hall", item = "bed")))
  }

  @Test
  fun hysteresisNeedsOneMetreMoreToLeaveNear() {
    // Route rule: enter at 3, leave above 4.
    assertEquals(FAR_ROUTE, mode(input(straight = 9.0, route = 3.5, user = "a", item = "b"), FAR_ROUTE))
    assertEquals(NEAR, mode(input(straight = 9.0, route = 3.5, user = "a", item = "b"), NEAR))
    assertEquals(NEAR, mode(input(straight = 9.0, route = 4.0, user = "a", item = "b"), NEAR))
    assertEquals(FAR_ROUTE, mode(input(straight = 9.0, route = 4.1, user = "a", item = "b"), NEAR))
    // Open rule: enter at 2.5, leave above 3.5.
    assertEquals(NEAR, mode(input(straight = 3.4, route = 20.0), NEAR))
    assertEquals(FAR_ROUTE, mode(input(straight = 3.6, route = 20.0), NEAR))
    // Same-room rule: enter at 5, leave above 6.
    assertEquals(NEAR, mode(input(straight = 5.9, route = 20.0, user = "a", item = "a"), NEAR))
    assertEquals(FAR_ROUTE, mode(input(straight = 6.1, route = 20.0, user = "a", item = "a"), NEAR))
    // Session-anchor rule: enter at 4, leave above 5.
    assertEquals(NEAR, mode(input(located = false, anchor = true, straight = 4.9), NEAR))
    assertEquals(FAR_NO_ROUTE, mode(input(located = false, anchor = true, straight = 5.1), NEAR))
    // Room changes are not hysteretic.
    assertEquals(FAR_ROUTE, mode(input(straight = 3.0, route = 20.0, user = "a", item = "b"), NEAR))
  }

  @Test
  fun replanOncePerSecondOrWhenOffRoute() {
    val r = Route(listOf(Vec3(0.0, 0.0, 0.0), Vec3(4.0, 0.0, 0.0)), listOf(0, 1))
    assertTrue(GuidancePlanner.shouldReplan(2000, 1000, r, Vec3(1.0, 0.0, 0.0)))
    assertFalse(GuidancePlanner.shouldReplan(1500, 1000, r, Vec3(1.0, 0.0, 0.5)))
    assertTrue(GuidancePlanner.shouldReplan(1500, 1000, r, Vec3(1.0, 0.0, 1.2)))
    assertFalse(GuidancePlanner.shouldReplan(1500, 1000, null, Vec3(1.0, 0.0, 1.2)))
  }

  @Test
  fun cueTexts() {
    assertEquals("In the Bedroom · 7 m walk", GuidancePlanner.cue(FAR_ROUTE, "Bedroom", 7.2, 4.0))
    assertEquals(
      "In the Bedroom · about 6 m away. No walked path yet: use the map.",
      GuidancePlanner.cue(FAR_NO_ROUTE, "Bedroom", null, 5.8),
    )
    assertEquals("Room not named. Go to a fix point so I can locate you.", GuidancePlanner.cue(NO_FIX, null, null, null))
    assertEquals("In Sam's room · 3 m walk", GuidancePlanner.cue(FAR_ROUTE, "Sam's room", 3.4, 1.0))
    assertEquals(
      "In the Hall · about 7 m away. Go to a fix point for a walked route.",
      GuidancePlanner.cue(FAR_NO_ROUTE, "Hall", null, 6.9, located = false),
    )
  }
}
