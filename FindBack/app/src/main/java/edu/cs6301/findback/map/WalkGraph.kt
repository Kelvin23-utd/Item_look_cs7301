package edu.cs6301.findback.map

import java.util.PriorityQueue
import kotlin.math.max

/**
 * Breadcrumbs of where the phone has been, in map coordinates, with room labels. Every edge joins
 * two nodes visited one after the other and no more than [MAX_EDGE_M] apart, so a route through the
 * graph only uses paths somebody actually walked. All distances are horizontal: the flat is one floor.
 *
 * Not thread-safe: owned by the GL thread. The UI reads copies ([nodes], [edges]).
 */
class WalkGraph(nodes: List<MapNode> = emptyList(), edges: List<Pair<Int, Int>> = emptyList()) {
  companion object {
    /** A position within this of an existing node visits that node instead of adding one. */
    const val SNAP_M = 0.35
    /** Add a new node once the phone is this far from the current node. */
    const val NEW_NODE_M = 0.5
    /** Consecutive nodes further apart than this are not joined (tracking jump, not a walk). */
    const val MAX_EDGE_M = 1.2
    /** A route's ends must each be within this of a node. */
    const val ROUTE_END_M = 1.5
    /** [roomAt] looks for the nearest labelled node within this. */
    const val ROOM_LOOKUP_M = 1.5
    /** Consecutive nodes of another room needed before the current room follows the labels. */
    const val FOLLOW_NODES = 2
    /** An explicit room pick relabels nodes within this of the user. */
    const val RELABEL_M = 1.0
  }

  class Node(
    val id: Int,
    val x: Double,
    val y: Double,
    val z: Double,
    var roomId: String?,
    val sessionId: String,
    val tMs: Long,
  ) {
    val position: Vec3
      get() = Vec3(x, y, z)

    fun toMapNode() = MapNode(id, x, y, z, roomId, sessionId, tMs)
  }

  /** Result of feeding one position. [followedRoom] is set when the current room changed by walking. */
  data class Visit(val nodeId: Int?, val created: Boolean, val changed: Boolean, val followedRoom: String?)

  private val nodeList = ArrayList<Node>()
  private val byId = HashMap<Int, Node>()
  private val adjacency = HashMap<Int, MutableSet<Int>>()
  private var nextId = 0
  private var edgeCount = 0

  /** Bumped on every change, so callers can tell when to snapshot or save. */
  var version = 0
    private set

  /** Node the chain continues from; null after [breakChain] (and at session start). */
  var currentNodeId: Int? = null
    private set

  /** The room the user is in, as picked or followed. Session state, not persisted. */
  var currentRoom: String? = null
    private set

  private var streakRoom: String? = null
  private var streakCount = 0

  init {
    for (n in nodes) addNodeInternal(Node(n.id, n.x, n.y, n.z, n.roomId, n.sessionId, n.tMs))
    for ((a, b) in edges) if (a in byId && b in byId) addEdgeInternal(a, b)
  }

  val nodeCount: Int
    get() = nodeList.size

  val edgeTotal: Int
    get() = edgeCount

  fun node(id: Int): Node? = byId[id]

  fun neighbours(id: Int): Set<Int> = adjacency[id] ?: emptySet()

  fun hasEdge(a: Int, b: Int) = adjacency[a]?.contains(b) == true

  fun nodes(): List<MapNode> = nodeList.map { it.toMapNode() }

  fun edges(): List<Pair<Int, Int>> =
    adjacency.flatMap { (a, ns) -> ns.filter { a < it }.map { a to it } }.sortedWith(compareBy({ it.first }, { it.second }))

  // ------------------------------------------------------------------ walking

  /** Feed the camera position (map frame). Called at about 5 Hz while located and tracking. */
  fun visit(p: Vec3, sessionId: String, tMs: Long): Visit {
    val snapped = nearestNode(p, SNAP_M)
    val cur = currentNodeId?.let { byId[it] }
    val target: Node
    var created = false
    if (snapped != null) {
      if (snapped.id == cur?.id) return Visit(snapped.id, created = false, changed = false, followedRoom = null)
      target = snapped
    } else {
      if (cur != null && horizontalDistance(cur.position, p) < NEW_NODE_M) {
        return Visit(cur.id, created = false, changed = false, followedRoom = null)
      }
      target = Node(nextId, p.x, p.y, p.z, null, sessionId, tMs)
      addNodeInternal(target)
      created = true
    }
    if (cur != null && horizontalDistance(cur.position, target.position) <= MAX_EDGE_M) {
      addEdgeInternal(cur.id, target.id)
    }
    currentNodeId = target.id
    version++
    return Visit(target.id, created, changed = true, followedRoom = applyRoomRules(target))
  }

  private fun applyRoomRules(n: Node): String? {
    val label = n.roomId
    when {
      label == null -> {
        // Unlabelled nodes the user visits take the current room.
        if (currentRoom != null) n.roomId = currentRoom
        resetStreak()
      }
      label == currentRoom -> resetStreak()
      else -> {
        // Never overwrite a different label by walking; follow it after FOLLOW_NODES in a row.
        if (streakRoom == label) streakCount++ else {
          streakRoom = label
          streakCount = 1
        }
        if (streakCount >= FOLLOW_NODES) {
          currentRoom = label
          resetStreak()
          return label
        }
      }
    }
    return null
  }

  private fun resetStreak() {
    streakRoom = null
    streakCount = 0
  }

  /** No edge across the next gap: session start, tracking loss, or a fix that moved the map. */
  fun breakChain() {
    currentNodeId = null
    resetStreak()
  }

  /**
   * Explicit room pick (chip or register dialog). Sets the current room and relabels nodes within
   * [RELABEL_M] of [p] (the correction tool). "No room" is [roomId] null. Returns how many nodes
   * changed label. [p] null (not located) only sets the current room.
   */
  fun pickRoom(roomId: String?, p: Vec3?): Int {
    currentRoom = roomId
    resetStreak()
    var changed = 0
    if (p != null) {
      for (n in nodeList) {
        if (horizontalDistance(n.position, p) <= RELABEL_M && n.roomId != roomId) {
          n.roomId = roomId
          changed++
        }
      }
    }
    version++
    return changed
  }

  // ------------------------------------------------------------------ queries

  fun nearestNode(p: Vec3, maxM: Double, filter: (Node) -> Boolean = { true }): Node? {
    var best: Node? = null
    var bestD = maxM
    for (n in nodeList) {
      if (!filter(n)) continue
      val d = horizontalDistance(n.position, p)
      if (d <= bestD) {
        bestD = d
        best = n
      }
    }
    return best
  }

  /** Room of the nearest labelled node within [ROOM_LOOKUP_M], or null. */
  fun roomAt(p: Vec3): String? = nearestNode(p, ROOM_LOOKUP_M) { it.roomId != null }?.roomId

  /** Centroid (map frame) of each labelled room's nodes, for the mini-map labels. */
  fun roomCentroids(): Map<String, Vec3> {
    val sums = HashMap<String, DoubleArray>()
    for (n in nodeList) {
      val r = n.roomId ?: continue
      val s = sums.getOrPut(r) { DoubleArray(4) }
      s[0] += n.x
      s[1] += n.y
      s[2] += n.z
      s[3] += 1.0
    }
    return sums.mapValues { (_, s) -> Vec3(s[0] / s[3], s[1] / s[3], s[2] / s[3]) }
  }

  /**
   * Shortest walked path between the nodes nearest [from] and [to] (each within [ROUTE_END_M]).
   * The polyline starts at [from], follows the nodes and ends at [to]. Null when either end is off
   * the graph or the two parts are not connected.
   */
  fun route(from: Vec3, to: Vec3): Route? {
    val start = nearestNode(from, ROUTE_END_M) ?: return null
    val goal = nearestNode(to, ROUTE_END_M) ?: return null
    val path = shortestPath(start.id, goal.id) ?: return null
    val pts = ArrayList<Vec3>(path.size + 2)
    pts.add(from)
    path.forEach { pts.add(byId.getValue(it).position) }
    pts.add(to)
    return Route(pts, path)
  }

  private fun shortestPath(a: Int, b: Int): List<Int>? {
    if (a == b) return listOf(a)
    val dist = HashMap<Int, Double>().apply { put(a, 0.0) }
    val prev = HashMap<Int, Int>()
    val pq = PriorityQueue<Pair<Double, Int>>(compareBy { it.first })
    pq.add(0.0 to a)
    while (pq.isNotEmpty()) {
      val (d, u) = pq.poll()!!
      if (d > (dist[u] ?: Double.MAX_VALUE)) continue
      if (u == b) break
      val pu = byId.getValue(u).position
      for (v in neighbours(u)) {
        val nd = d + horizontalDistance(pu, byId.getValue(v).position)
        if (nd < (dist[v] ?: Double.MAX_VALUE)) {
          dist[v] = nd
          prev[v] = u
          pq.add(nd to v)
        }
      }
    }
    if (b !in dist) return null
    val out = ArrayList<Int>()
    var c: Int? = b
    while (c != null) {
      out.add(c)
      c = prev[c]
    }
    return out.reversed()
  }

  // ------------------------------------------------------------------ internals

  private fun addNodeInternal(n: Node) {
    nodeList.add(n)
    byId[n.id] = n
    nextId = max(nextId, n.id + 1)
  }

  private fun addEdgeInternal(a: Int, b: Int) {
    if (a == b) return
    val added = adjacency.getOrPut(a) { HashSet() }.add(b)
    adjacency.getOrPut(b) { HashSet() }.add(a)
    if (added) edgeCount++
  }
}

/** A walked route as a horizontal polyline. Lengths and projections ignore height. */
class Route(val points: List<Vec3>, val nodeIds: List<Int>) {
  private val cumulative = DoubleArray(points.size)

  init {
    for (i in 1 until points.size) cumulative[i] = cumulative[i - 1] + horizontalDistance(points[i - 1], points[i])
  }

  val lengthM: Double
    get() = if (points.isEmpty()) 0.0 else cumulative.last()

  /** Closest point on the polyline to [p]; [alongM] is the route distance from the start to it. */
  data class Projection(val segment: Int, val t: Double, val point: Vec3, val distanceM: Double, val alongM: Double)

  fun project(p: Vec3): Projection {
    if (points.size < 2) {
      val q = points.firstOrNull() ?: p
      return Projection(0, 0.0, q, horizontalDistance(p, q), 0.0)
    }
    var best: Projection? = null
    for (i in 0 until points.size - 1) {
      val a = points[i]
      val b = points[i + 1]
      val dx = b.x - a.x
      val dz = b.z - a.z
      val len2 = dx * dx + dz * dz
      val t = if (len2 < 1e-12) 0.0 else (((p.x - a.x) * dx + (p.z - a.z) * dz) / len2).coerceIn(0.0, 1.0)
      val q = a + (b - a) * t
      val d = horizontalDistance(p, q)
      if (best == null || d < best.distanceM - 1e-9) {
        best = Projection(i, t, q, d, cumulative[i] + t * (cumulative[i + 1] - cumulative[i]))
      }
    }
    return best!!
  }

  fun distanceTo(p: Vec3) = project(p).distanceM

  /** Walking distance left: back onto the route, then along it to the end. */
  fun remainingFrom(p: Vec3): Double {
    val pr = project(p)
    return pr.distanceM + (lengthM - pr.alongM)
  }

  /** Point [aheadM] further along the route than [p]'s projection, clamped to the end. */
  fun pointAhead(p: Vec3, aheadM: Double): Vec3 = pointAt(project(p).alongM + aheadM)

  fun pointAt(alongM: Double): Vec3 {
    if (points.size < 2) return points.first()
    val s = alongM.coerceIn(0.0, lengthM)
    for (i in 0 until points.size - 1) {
      if (s <= cumulative[i + 1] || i == points.size - 2) {
        val segLen = cumulative[i + 1] - cumulative[i]
        val t = if (segLen < 1e-12) 1.0 else ((s - cumulative[i]) / segLen).coerceIn(0.0, 1.0)
        return points[i] + (points[i + 1] - points[i]) * t
      }
    }
    return points.last()
  }
}
