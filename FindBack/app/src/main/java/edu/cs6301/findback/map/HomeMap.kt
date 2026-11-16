package edu.cs6301.findback.map

/** How a fix point is observed. */
enum class FixType {
  /** The reference picture (iPad or print), seen through ARCore Augmented Images. */
  IMAGE,
  /** A marker-free phone pose the user can repeat, aided by a ghost image. */
  SPOT,
}

/**
 * A known point that locates the map in a new session. [x], [y], [z], [yaw] are its Pose4 in the map
 * frame M (identity for the first fix point of a map).
 */
data class FixPoint(
  val id: String,
  val name: String,
  val type: FixType,
  val x: Double,
  val y: Double,
  val z: Double,
  val yaw: Double,
  /**
   * Axis the heading is taken from, always reused. IMAGE: the picture's local "X" or "Z". SPOT: the
   * display's "X" (phone upright) or "Y" (phone on its side); null means "X".
   */
  val yawAxis: String?,
  /** IMAGE only: picture width the pose was measured with. A different setting ignores this point. */
  val widthM: Double?,
  /** SPOT only: ghost image path relative to the files root (fix/<id>.jpg). */
  val ghostFile: String?,
  val createdMs: Long,
  val sessionId: String,
) {
  val mapPose: Pose4
    get() = Pose4(x, y, z, yaw)

  companion object {
    fun at(
      id: String,
      name: String,
      type: FixType,
      mapPose: Pose4,
      yawAxis: String?,
      widthM: Double?,
      ghostFile: String?,
      createdMs: Long,
      sessionId: String,
    ) = FixPoint(id, name, type, mapPose.x, mapPose.y, mapPose.z, mapPose.yaw, yawAxis, widthM, ghostFile, createdMs, sessionId)
  }
}

data class Room(val id: String, val name: String, val createdMs: Long)

/** One breadcrumb of the walk graph, in map coordinates (camera position, so y is camera height). */
data class MapNode(
  val id: Int,
  val x: Double,
  val y: Double,
  val z: Double,
  val roomId: String?,
  val sessionId: String,
  val tMs: Long,
) {
  val position: Vec3
    get() = Vec3(x, y, z)
}

/** Immutable snapshot of everything in map.json. */
data class HomeMap(
  val version: Int = VERSION,
  val mapId: String,
  val createdMs: Long,
  val fixPoints: List<FixPoint>,
  val rooms: List<Room>,
  val nodes: List<MapNode>,
  /** Undirected edges as node-id pairs. */
  val edges: List<Pair<Int, Int>>,
) {
  fun room(id: String?): Room? = if (id == null) null else rooms.firstOrNull { it.id == id }

  companion object {
    const val VERSION = 1
  }
}
