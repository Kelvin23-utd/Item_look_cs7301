package edu.cs6301.findback.map

import kotlin.math.roundToInt

enum class GuidanceMode {
  /** The item cannot be resolved in this session: photo and room only. */
  NO_FIX,
  /** Located, not near, and a walked route exists: chevron along the route. */
  FAR_ROUTE,
  /** Not near and no walked route: room and straight distance, no arrow. */
  FAR_NO_ROUTE,
  /** Close enough to show the ring on the item. */
  NEAR,
}

/** Everything the planner needs, computed by the renderer about 5 times a second. */
data class GuidanceInput(
  /** The item has a position in this session (same-session anchor, or map position while located). */
  val resolved: Boolean,
  /** The map frame is located, so the user has a map position. */
  val located: Boolean,
  /** The item has a same-session anchor that is tracking. */
  val hasSessionAnchor: Boolean,
  /** Horizontal straight-line distance user to item (metres), when resolved. */
  val straightM: Double?,
  /** Walking distance left along the current route, or null when there is no route. */
  val routeRemainingM: Double?,
  val userRoom: String?,
  val itemRoom: String?,
)

data class GuidanceDecision(val mode: GuidanceMode, val nearRule: String?)

/**
 * Pure guidance decision. NEAR holds when any rule below holds; once in NEAR, every distance limit
 * is raised by [NEAR_EXIT_MARGIN_M] so the mode does not flicker at the boundary.
 */
object GuidancePlanner {
  /** NEAR when the remaining walked route is at most this. */
  const val NEAR_ROUTE_M = 3.0
  /** NEAR when user and item are in the same room and at most this far apart in a straight line. */
  const val NEAR_SAME_ROOM_M = 5.0
  /** NEAR when they are not known to be in different rooms and at most this far apart. */
  const val NEAR_OPEN_M = 2.5
  /** NEAR when the item has a same-session anchor, is at most this far, and is not known elsewhere. */
  const val NEAR_SESSION_ANCHOR_M = 4.0
  /** Leaving NEAR needs the distance to exceed the entry limit by this much. */
  const val NEAR_EXIT_MARGIN_M = 1.0
  /** FAR_ROUTE: the chevron points at the route point this far ahead of the user. */
  const val WAYPOINT_AHEAD_M = 1.5
  /** Re-plan the route this often... */
  const val REPLAN_MS = 1000L
  /** ...and immediately when the user is more than this far off it. */
  const val OFF_ROUTE_M = 1.0

  const val RULE_ROUTE = "route"
  const val RULE_SAME_ROOM = "same_room"
  const val RULE_OPEN = "open"
  const val RULE_SESSION_ANCHOR = "session_anchor"

  fun decide(i: GuidanceInput, previous: GuidanceMode?): GuidanceDecision {
    if (!i.resolved) return GuidanceDecision(GuidanceMode.NO_FIX, null)
    val margin = if (previous == GuidanceMode.NEAR) NEAR_EXIT_MARGIN_M else 0.0
    val differentRooms = i.userRoom != null && i.itemRoom != null && i.userRoom != i.itemRoom
    val sameRoom = i.userRoom != null && i.userRoom == i.itemRoom
    val d = i.straightM
    val rule =
      when {
        i.routeRemainingM != null && i.routeRemainingM <= NEAR_ROUTE_M + margin -> RULE_ROUTE
        sameRoom && d != null && d <= NEAR_SAME_ROOM_M + margin -> RULE_SAME_ROOM
        !differentRooms && d != null && d <= NEAR_OPEN_M + margin -> RULE_OPEN
        i.hasSessionAnchor && !differentRooms && d != null && d <= NEAR_SESSION_ANCHOR_M + margin ->
          RULE_SESSION_ANCHOR
        else -> null
      }
    if (rule != null) return GuidanceDecision(GuidanceMode.NEAR, rule)
    return if (i.located && i.routeRemainingM != null) GuidanceDecision(GuidanceMode.FAR_ROUTE, null)
    else GuidanceDecision(GuidanceMode.FAR_NO_ROUTE, null)
  }

  /** Re-plan about once a second, and at once when the user has left the route. */
  fun shouldReplan(nowMs: Long, lastPlanMs: Long, route: Route?, user: Vec3?): Boolean {
    if (nowMs - lastPlanMs >= REPLAN_MS) return true
    if (route == null || user == null) return false
    return route.distanceTo(user) > OFF_ROUTE_M
  }

  /** "In the Bedroom", or a fallback when the item's room was never named. */
  fun roomPhrase(roomName: String?): String {
    if (roomName.isNullOrBlank()) return "Room not named"
    val n = roomName.trim()
    return if (n.startsWith("the ", ignoreCase = true) || n.contains('\'')) "In $n" else "In the $n"
  }

  /**
   * Card text for the modes that do not show the ring. NEAR text is built from the camera pose.
   * FAR_NO_ROUTE while not [located] (a far same-session item) cannot use the map yet.
   */
  fun cue(mode: GuidanceMode, roomName: String?, routeM: Double?, straightM: Double?, located: Boolean = true): String {
    val room = roomPhrase(roomName)
    val about = "about ${straightM?.roundToInt() ?: "?"} m away"
    return when (mode) {
      GuidanceMode.NO_FIX -> "$room. Go to a fix point so I can locate you."
      GuidanceMode.FAR_ROUTE -> "$room · ${routeM?.roundToInt() ?: "?"} m walk"
      GuidanceMode.FAR_NO_ROUTE ->
        if (located) "$room · $about. No walked path yet: use the map."
        else "$room · $about. Go to a fix point for a walked route."
      GuidanceMode.NEAR -> room
    }
  }
}
