import math
import random
from abc import ABC, abstractmethod
from enum import Enum, auto

import click
import pygame

# ======================
# Basic configuration
# ======================

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800

BACKGROUND_COLOR = (30, 30, 30)
ROAD_COLOR = (60, 60, 60)
ROAD_EDGE_COLOR = (100, 100, 100)
ROUNDABOUT_ISLAND_COLOR = (80, 80, 80)
VEHICLE_COLOR = (200, 200, 50)  # Default (not used when color-coded by direction)
TEXT_COLOR = (230, 230, 230)
SIGNAL_GREEN = (0, 180, 0)
SIGNAL_RED = (200, 0, 0)

FPS = 60

# Geometry in "meters" (abstract units)
APPROACH_LENGTH = 80.0  # distance from spawn to stop line (in s-space)
EXIT_LENGTH = 60.0  # distance from stop line to disappearance

SPAWN_S = -APPROACH_LENGTH
STOP_LINE_S = 0.0  # intersection entry at s=0 (in logical coordinates)
STOP_BUFFER = 20.0  # vehicles stop 20 m before the intersection
STOP_LIMIT_S = STOP_LINE_S - STOP_BUFFER
EXIT_S = EXIT_LENGTH

# Intersection distance for turns (used for delay / conflicts)
BASE_INTERSECTION_S = 12.0  # base distance inside intersection for RIGHT turns

# Vehicle and behavior
VEHICLE_LENGTH = 4.0
MIN_GAP = 2.0  # minimal gap between vehicles (m)
DEFAULT_DESIRED_SPEED = 12.0  # m/s
DEFAULT_DESIRED_SPEED_STD = 2.0
DEFAULT_MAX_ACCEL = 2.0
DEFAULT_GAP_ACCEPT_MEAN = 2.0
DEFAULT_GAP_ACCEPT_STD = 0.5

INF = 1e9

# Queue management
MAX_QUEUE_PER_DIRECTION = (
    2000  # Maximum vehicles per direction to prevent memory issues
)

# Drawing scale and lane offset (shared for all straight roads)
SCALE = 2.0  # pixels per "meter"
LANE_OFFSET = 8.0  # lateral lane offset (pixels)


class Direction(Enum):
    NORTH = auto()  # flows N -> S
    EAST = auto()  # flows E -> W
    SOUTH = auto()  # flows S -> N
    WEST = auto()  # flows W -> E


class Turn(Enum):
    RIGHT = auto()
    STRAIGHT = auto()
    LEFT = auto()


DIRECTIONS = [
    Direction.NORTH,
    Direction.EAST,
    Direction.SOUTH,
    Direction.WEST,
]

# Vehicle colors by origin direction
VEHICLE_COLORS = {
    Direction.NORTH: (100, 150, 255),  # Light blue
    Direction.SOUTH: (255, 100, 100),  # Red/coral
    Direction.EAST: (100, 255, 150),  # Green
    Direction.WEST: (255, 200, 100),  # Yellow/orange
}


def opposite_direction(d):
    if d == Direction.NORTH:
        return Direction.SOUTH
    if d == Direction.SOUTH:
        return Direction.NORTH
    if d == Direction.EAST:
        return Direction.WEST
    return Direction.EAST


def vehicles_conflict_opposite_dirs(turn_a, turn_b):
    """
    Check if two vehicles from opposite directions conflict.
    Returns True if vehicle A (with turn_a) must yield to vehicle B (with turn_b).

    Rules:
    - Left turn must yield to oncoming straight or right
    - Straight has right-of-way over oncoming left
    - Right has right-of-way over oncoming left
    - Both left, both straight, both right, or straight+right combinations can go
      together
    """
    # Vehicle A turning left must yield to B going straight or right
    if turn_a == Turn.LEFT and turn_b in (Turn.STRAIGHT, Turn.RIGHT):
        return True

    # No conflict in all other cases:
    # - Both left: can go together
    # - Both straight: can go together
    # - Both right: can go together
    # - Straight + right: can go together
    # - Right vs left: right has priority (left yields, not right)
    # - Straight vs left: straight has priority (left yields, not straight)
    return False


# ======================
# Straight-road geometry
# ======================


def crossroad_position(center, direction, s_world):
    """
    Plus-shaped intersection with TWO lanes per road, expressed in "world s":

      - NORTH (N->S): lane slightly left of centerline
      - SOUTH (S->N): lane slightly right of centerline
      - EAST (E->W):  lane slightly above centerline
      - WEST (W->E):  lane slightly below centerline

    s_world = 0 is at the intersection center; s_world > 0 is "downstream".
    """
    cx, cy = center

    if direction == Direction.NORTH:
        x = cx - LANE_OFFSET
        y = cy + s_world * SCALE
    elif direction == Direction.SOUTH:
        x = cx + LANE_OFFSET
        y = cy - s_world * SCALE
    elif direction == Direction.EAST:
        x = cx - s_world * SCALE
        y = cy - LANE_OFFSET
    else:  # WEST
        x = cx + s_world * SCALE
        y = cy + LANE_OFFSET

    return x, y


def exit_side_for_cross(origin, turn):
    """
    For cross intersections: given origin side and turn,
    return which arm (Direction) the vehicle exits onto.
    """
    if turn == Turn.STRAIGHT:
        return opposite_direction(origin)

    order = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
    idx = order.index(origin)

    if turn == Turn.RIGHT:
        # 90° clockwise
        return order[(idx + 1) % 4]
    else:  # Turn.LEFT
        # 90° counter-clockwise
        return order[(idx - 1) % 4]


def crossroad_position_with_turn(center, origin, s_world, turn):
    """
    Turn-aware position for *non-roundabout* intersections.

    - STRAIGHT: same as crossroad_position (no change).
    - RIGHT / LEFT:
        * Vehicle follows the approach lane up to a fixed 'corner' s.
        * From that s onward, it instantly switches to the exit arm,
          with (x, y) continuous (no jump).
        * Right turns use an 'early' corner (negative s).
        * Left turns use a 'later' corner (positive s), exactly one
          lane-width further along the approach than the right turn.

    Note: this is purely geometric; it does not affect dynamics.
    """
    # Straight movement: just use the normal lane
    if turn == Turn.STRAIGHT:
        return crossroad_position(center, origin, s_world)

    lane_offset_m = LANE_OFFSET / SCALE  # convert lane offset into "s" units

    # Corner positions in 's' along the approach and exit lanes.
    # Derived so that:
    #   crossroad_position(origin, s_origin_corner)
    #   == crossroad_position(exit_side, s_exit_corner)
    if turn == Turn.RIGHT:
        s_origin_corner = -lane_offset_m  # before center
        s_exit_corner = +lane_offset_m  # after center
    else:  # Turn.LEFT (bigger turn)
        s_origin_corner = +lane_offset_m  # one lane-width further than right turn
        s_exit_corner = -lane_offset_m

    # Before the corner: still on the approach lane
    if s_world <= s_origin_corner:
        return crossroad_position(center, origin, s_world)

    # After the corner: move on the exit arm, shifted so there is no jump
    exit_dir = exit_side_for_cross(origin, turn)
    # Keep longitudinal speed by just shifting s so the corner matches:
    #   s_world == s_origin_corner  ->  s_on_exit == s_exit_corner
    s_on_exit = s_exit_corner + (s_world - s_origin_corner)
    return crossroad_position(center, exit_dir, s_on_exit)


# ======================
# Roundabout geometry
# ======================

ROUNDABOUT_RADIUS = 55.0  # pixels, radius of circulating lane

# Tangent distance calculation (used in vehicle position functions)
# This constant is kept for reference but mostly computed inline where needed
ROUNDABOUT_ENTRY_S_WORLD = math.sqrt(
    max(ROUNDABOUT_RADIUS**2 - LANE_OFFSET**2, 1.0)
)  # in pixels


def direction_entry_angle(direction):
    """
    Angle on the circle where each approach enters the roundabout.
    Screen coordinates: x right, y down, angle measured CCW from +x.
    """
    if direction == Direction.EAST:
        return 0.0
    if direction == Direction.SOUTH:
        return math.pi / 2.0
    if direction == Direction.WEST:
        return math.pi
    return -math.pi / 2.0  # NORTH


def turn_step_count(turn):
    if turn == Turn.RIGHT:
        return 1
    if turn == Turn.STRAIGHT:
        return 2
    return 3  # LEFT


def exit_heading_for_movement(origin, turn):
    """Given origin and turn, return the direction heading for the exit arm."""
    order = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
    idx = order.index(origin)
    steps = turn_step_count(turn)
    dest_side = order[(idx - steps) % len(order)]
    # Exit heading is from center to that side => opposite of "approach" direction
    return opposite_direction(dest_side)


# ======================
# Vehicle data
# ======================


class VehicleBlueprint:
    """
    Shared per-vehicle data across all intersection designs.
    Same blueprint is used so the car is "identical" in all four designs.
    """

    def __init__(
        self,
        vehicle_id,
        arrival_time,
        origin,
        desired_speed,
        max_accel,
        gap_acceptance,
        free_flow_time,
        turn,
        intersection_exit_s,
    ):
        self.id = vehicle_id
        self.arrival_time = arrival_time
        self.origin = origin
        self.desired_speed = desired_speed
        self.max_accel = max_accel
        self.gap_acceptance = gap_acceptance
        self.free_flow_time = free_flow_time
        self.turn = turn
        self.intersection_exit_s = intersection_exit_s


class VehicleState:
    """Per-intersection copy of a vehicle (position, speed, exit time)."""

    def __init__(self, blueprint):
        self.blueprint = blueprint
        self.s = SPAWN_S  # longitudinal coordinate in "logical" s
        self.v = blueprint.desired_speed  # Start at cruising speed
        self.entered = False
        self.exit_time = None


class IntersectionMetrics:
    def __init__(self):
        self.departed_count = 0
        self.total_delay = 0.0
        self.delays = []
        self.max_queue = 0

    @property
    def mean_delay(self):
        if self.departed_count == 0:
            return 0.0
        return self.total_delay / self.departed_count

    @property
    def p95_delay(self):
        if not self.delays:
            return 0.0
        sorted_delays = sorted(self.delays)
        idx = int(0.95 * (len(sorted_delays) - 1))
        return sorted_delays[idx]


def vehicle_in_intersection(v):
    bp = v.blueprint
    return v.s >= STOP_LINE_S and v.s <= bp.intersection_exit_s


def roundabout_vehicle_angle(v):
    """Return the angular position of a vehicle that is in the roundabout (s>0)."""
    bp = v.blueprint
    if v.s <= 0.0 or v.s > bp.intersection_exit_s:
        return None
    theta_entry = direction_entry_angle(bp.origin)
    steps = turn_step_count(bp.turn)
    delta_theta = steps * (math.pi / 2.0)
    u = max(0.0, min(1.0, v.s / bp.intersection_exit_s))
    return theta_entry - u * delta_theta


def roundabout_vehicle_position(center, direction, vehicle):
    """
    Map vehicle.s to (x, y) for a single-lane roundabout:

      - s <= 0: straight approach on proper lane, tangent to the circle.
      - 0 < s <= intersection_exit_s: motion along circular arc of radius
        ROUNDABOUT_RADIUS.
      - s > intersection_exit_s: motion on proper exit lane, tangent from the circle.

    Note: s=0 is at the entry tangent point (where vehicle enters the ring).
          This gives smooth transitions.
    """
    bp = vehicle.blueprint
    s = vehicle.s
    cx, cy = center

    # Calculate tangent distance from center to where lane meets circle (in pixels)
    tangent_dist = math.sqrt(ROUNDABOUT_RADIUS**2 - LANE_OFFSET**2)

    # Approach on straight lane (s <= 0, approaching tangent point)
    if s <= 0.0:
        # Direct calculation of approach position
        if direction == Direction.NORTH:
            x = cx - LANE_OFFSET
            y = cy - tangent_dist + s * SCALE
        elif direction == Direction.SOUTH:
            x = cx + LANE_OFFSET
            y = cy + tangent_dist - s * SCALE
        elif direction == Direction.EAST:
            x = cx + tangent_dist - s * SCALE
            y = cy - LANE_OFFSET
        else:  # WEST
            x = cx - tangent_dist + s * SCALE
            y = cy + LANE_OFFSET
        return x, y

    steps = turn_step_count(bp.turn)
    theta_entry = direction_entry_angle(direction)

    # Inside the roundabout: move along the circular arc
    if s <= bp.intersection_exit_s:
        u = max(0.0, min(1.0, s / bp.intersection_exit_s))
        theta = theta_entry - u * steps * (math.pi / 2.0)
        x = cx + ROUNDABOUT_RADIUS * math.cos(theta)
        y = cy + ROUNDABOUT_RADIUS * math.sin(theta)
        return x, y

    # Exit on straight lane: calculate exit position directly
    s_exit = s - bp.intersection_exit_s
    exit_dir = exit_heading_for_movement(direction, bp.turn)

    if exit_dir == Direction.NORTH:
        x = cx - LANE_OFFSET
        y = cy + tangent_dist + s_exit * SCALE
    elif exit_dir == Direction.SOUTH:
        x = cx + LANE_OFFSET
        y = cy - tangent_dist - s_exit * SCALE
    elif exit_dir == Direction.EAST:
        x = cx - tangent_dist - s_exit * SCALE
        y = cy - LANE_OFFSET
    else:  # WEST
        x = cx + tangent_dist + s_exit * SCALE
        y = cy + LANE_OFFSET

    return x, y


# ======================
# Intersection base class
# ======================


class IntersectionModel(ABC):
    """Base class for all four intersection control strategies."""

    def __init__(self, name, center):
        self.name = name
        self.center = center
        self.lanes = {d: [] for d in DIRECTIONS}  # direction -> list[VehicleState]
        self.metrics = IntersectionMetrics()
        self.sim_time = 0.0

    @abstractmethod
    def update_control(self, dt, sim_time):
        """Update control state (signals, who has right of way, etc.)."""
        ...

    @abstractmethod
    def max_s_for_front(self, direction, vehicle, sim_time):
        """
        Return the maximum allowed s for the *front-most approach vehicle* for this
        direction. Others are limited only by car-following.
        """
        ...

    def spawn_vehicle(self, blueprint):
        self.lanes[blueprint.origin].append(VehicleState(blueprint))

    def step(self, dt, sim_time):
        """Advance simulation for this intersection by dt."""
        self.sim_time = sim_time
        self.update_control(dt, sim_time)

        total_queue = 0

        for direction in DIRECTIONS:
            lane = self.lanes[direction]
            if not lane:
                continue

            # Sort by position
            lane.sort(key=lambda v: v.s)

            # Identify the front-most *approach* vehicle (s <= STOP_LINE_S)
            front_approach = None
            for v in lane:
                if v.s <= STOP_LINE_S:
                    if (front_approach is None) or (v.s > front_approach.s):
                        front_approach = v

            # Update vehicles from front to back
            prev_leader = None
            for idx in range(len(lane) - 1, -1, -1):
                v = lane[idx]
                if v is front_approach:
                    control_limit = self.max_s_for_front(direction, v, sim_time)
                else:
                    control_limit = INF

                if prev_leader is None:
                    max_s = control_limit
                else:
                    gap_limit = prev_leader.s - (VEHICLE_LENGTH + MIN_GAP)
                    max_s = min(control_limit, gap_limit)

                self._update_vehicle(v, dt, max_s)
                prev_leader = v

            # Remove vehicles that have left the system
            self._collect_departures(direction, lane, sim_time)
            total_queue += len(lane)

        if total_queue > self.metrics.max_queue:
            self.metrics.max_queue = total_queue

    def _update_vehicle(self, vehicle, dt, max_s):
        """Simple longitudinal motion: accelerate to desired speed, obey max_s."""
        bp = vehicle.blueprint
        desired_v = max(bp.desired_speed, 1.0)
        # Accelerate towards desired speed
        if vehicle.v < desired_v:
            v_candidate = vehicle.v + bp.max_accel * dt
            vehicle.v = min(v_candidate, desired_v)
        # Nominal new s without control limits
        s_nominal = vehicle.s + vehicle.v * dt
        # Enforce control and car-following limit
        vehicle.s = min(s_nominal, max_s)

        # Mark if they entered intersection (cross stop line)
        if (not vehicle.entered) and vehicle.s >= STOP_LINE_S:
            vehicle.entered = True

    def _collect_departures(self, direction, lane, sim_time):
        """Remove vehicles that have gone far enough past the intersection."""
        remaining = []
        for v in lane:
            if v.s > EXIT_S:
                if v.exit_time is None:
                    v.exit_time = sim_time
                self._record_departure(v)
            else:
                remaining.append(v)
        lane[:] = remaining

    def _record_departure(self, vehicle):
        bp = vehicle.blueprint
        if vehicle.exit_time is None:
            return
        travel_time = vehicle.exit_time - bp.arrival_time
        delay = max(travel_time - bp.free_flow_time, 0.0)
        self.metrics.departed_count += 1
        self.metrics.total_delay += delay
        self.metrics.delays.append(delay)

    # ---- Rendering ----

    def vehicle_position(self, direction, vehicle):
        """Default: simple cross intersection mapping."""
        return crossroad_position_with_turn(
            self.center,
            direction,
            vehicle.s,
            vehicle.blueprint.turn,
        )

    def draw(self, surface, font):
        self._draw_roads(surface)
        self._draw_control(surface)
        self._draw_vehicles(surface)
        self._draw_text(surface, font)

    def _draw_roads(self, surface):
        cx, cy = self.center
        road_half_width = 18
        # Road extends from center in both directions
        # Total length in meters divided by 2, then convert to pixels
        road_length_px = int((APPROACH_LENGTH + EXIT_LENGTH + 30) / 2 * SCALE)

        # Vertical road
        rect_v = pygame.Rect(
            cx - road_half_width,
            cy - road_length_px,
            2 * road_half_width,
            2 * road_length_px,
        )
        pygame.draw.rect(surface, ROAD_COLOR, rect_v)
        pygame.draw.rect(surface, ROAD_EDGE_COLOR, rect_v, 2)

        # Horizontal road
        rect_h = pygame.Rect(
            cx - road_length_px,
            cy - road_half_width,
            2 * road_length_px,
            2 * road_half_width,
        )
        pygame.draw.rect(surface, ROAD_COLOR, rect_h)
        pygame.draw.rect(surface, ROAD_EDGE_COLOR, rect_h, 2)

    @abstractmethod
    def _draw_control(self, surface):
        """Draw specific control elements (signals, signs, roundabout island)."""
        ...

    def _draw_vehicles(self, surface):
        for direction in DIRECTIONS:
            for v in self.lanes[direction]:
                # Only render vehicles within visible road bounds
                # Don't render vehicles that have queued beyond spawn or exited far
                # beyond exit
                if SPAWN_S - VEHICLE_LENGTH <= v.s <= EXIT_S + VEHICLE_LENGTH:
                    self._draw_single_vehicle(surface, direction, v)

    def _draw_single_vehicle(self, surface, direction, vehicle):
        x_pix, y_pix = self.vehicle_position(direction, vehicle)
        size = 10  # draw as square
        rect = pygame.Rect(int(x_pix - size / 2), int(y_pix - size / 2), size, size)
        # Color-code by origin direction
        color = VEHICLE_COLORS.get(vehicle.blueprint.origin, VEHICLE_COLOR)
        pygame.draw.rect(surface, color, rect)

    def _draw_text(self, surface, font):
        cx, cy = self.center
        label = font.render(self.name, True, TEXT_COLOR)
        surface.blit(label, (cx - label.get_width() // 2, cy - 120))

        stats = (
            f"N={self.metrics.departed_count} "
            f"mean={self.metrics.mean_delay:.1f}s "
            f"p95={self.metrics.p95_delay:.1f}s "
            f"q_max={self.metrics.max_queue}"
        )
        stats_surf = font.render(stats, True, TEXT_COLOR)
        surface.blit(stats_surf, (cx - stats_surf.get_width() // 2, cy + 120))


# ======================
# Roundabouts
# ======================


class RoundaboutPriorityCirculating(IntersectionModel):
    """
    Single-lane roundabout, priority to circulating traffic.
    Vehicles choose RIGHT / STRAIGHT / LEFT, which correspond to 1st/2nd/3rd exit.
    """

    def __init__(self, center):
        super().__init__("Roundabout (circ. priority)", center)

    def update_control(self, dt, sim_time):
        # No explicit internal state besides vehicles.
        pass

    def step(self, dt, sim_time):
        """Override to enforce car-following in the ring across all directions."""
        self.sim_time = sim_time
        self.update_control(dt, sim_time)

        total_queue = 0

        for direction in DIRECTIONS:
            lane = self.lanes[direction]
            if not lane:
                continue

            # Sort by position
            lane.sort(key=lambda v: v.s)

            # Identify the front-most *approach* vehicle (s <= STOP_LINE_S)
            front_approach = None
            for v in lane:
                if v.s <= STOP_LINE_S:
                    if (front_approach is None) or (v.s > front_approach.s):
                        front_approach = v

            # Update vehicles from front to back
            prev_leader = None
            for idx in range(len(lane) - 1, -1, -1):
                v = lane[idx]
                if v is front_approach:
                    control_limit = self.max_s_for_front(direction, v, sim_time)
                else:
                    control_limit = INF

                if prev_leader is None:
                    max_s = control_limit
                else:
                    gap_limit = prev_leader.s - (VEHICLE_LENGTH + MIN_GAP)
                    max_s = min(control_limit, gap_limit)

                # Additional check: if vehicle is in ring, check for leaders from other
                # directions
                if 0 < v.s <= v.blueprint.intersection_exit_s:
                    ring_leader_limit = self._find_ring_leader_limit(direction, v)
                    max_s = min(max_s, ring_leader_limit)

                self._update_vehicle(v, dt, max_s)
                prev_leader = v

            # Remove vehicles that have left the system
            self._collect_departures(direction, lane, sim_time)
            total_queue += len(lane)

        if total_queue > self.metrics.max_queue:
            self.metrics.max_queue = total_queue

    def _find_ring_leader_limit(self, direction, vehicle):
        """Find the closest vehicle ahead in the ring (from any direction)."""
        my_angle = roundabout_vehicle_angle(vehicle)
        if my_angle is None:
            return INF

        min_limit = INF

        for other_dir in DIRECTIONS:
            for other in self.lanes[other_dir]:
                if other is vehicle:
                    continue

                other_angle = roundabout_vehicle_angle(other)
                if other_angle is None:
                    continue

                # Calculate angular distance (clockwise mathematically =
                # counterclockwise visually)
                angle_diff = (my_angle - other_angle) % (2.0 * math.pi)

                # Only consider vehicles ahead (small positive angle difference)
                if angle_diff > 0 and angle_diff < math.pi:
                    # Convert angular distance to arc length
                    arc_distance = ROUNDABOUT_RADIUS * angle_diff / SCALE
                    gap_limit = vehicle.s + arc_distance - (VEHICLE_LENGTH + MIN_GAP)
                    min_limit = min(min_limit, gap_limit)

        return min_limit

    def _has_conflict(self, direction, vehicle, safe_angle, min_gap_time):
        """Check if there is a circulating vehicle conflicting with entering vehicle."""
        theta_entry = direction_entry_angle(direction)
        bp = vehicle.blueprint

        # approximate time for this vehicle to reach the entry (s=0)
        distance_to_entry = -vehicle.s  # since s<=0
        approach_speed = max(bp.desired_speed, 1.0)
        time_to_entry = max(distance_to_entry / approach_speed, 0.0)

        radius_m = ROUNDABOUT_RADIUS / SCALE

        for d in DIRECTIONS:
            lane = self.lanes[d]
            for other in lane:
                if other is vehicle:
                    continue
                theta_other = roundabout_vehicle_angle(other)
                if theta_other is None:
                    continue  # not in the ring yet
                # angular distance from other to entry, now moving counterclockwise
                # visually (CW mathematically)
                dist_cw = (theta_other - theta_entry) % (2.0 * math.pi)
                if dist_cw <= 0.0 or dist_cw > safe_angle:
                    continue
                # approximate time for that vehicle to reach the entry
                arc = radius_m * dist_cw  # meters now
                other_speed = max(other.v, 1.0)
                time_other_to_entry = arc / other_speed
                # if other will reach entry before or close to this car, conflict
                if time_other_to_entry < time_to_entry + min_gap_time:
                    return True
        return False

    def max_s_for_front(self, direction, vehicle, sim_time):
        # Once the car has passed the stop line, let them go freely.
        if vehicle.s >= STOP_LINE_S:
            return INF

        # Yield if there is conflicting circulating traffic
        safe_angle = 2.0 * math.pi / 3.0
        min_gap_time = 1.0

        if self._has_conflict(direction, vehicle, safe_angle, min_gap_time):
            return STOP_LIMIT_S
        else:
            return INF

    def vehicle_position(self, direction, vehicle):
        return roundabout_vehicle_position(self.center, direction, vehicle)

    def spawn_vehicle(self, blueprint):
        """Override to adjust spawn position for roundabout geometry."""
        # Calculate offset from center to tangent
        tangent_offset = math.sqrt(ROUNDABOUT_RADIUS**2 - LANE_OFFSET**2) / SCALE
        # Spawn closer so vehicles fit within standard road length
        # Standard roads extend APPROACH_LENGTH from center
        # We want spawn to be within that, accounting for tangent offset
        adjusted_spawn = -(APPROACH_LENGTH - tangent_offset)
        v = VehicleState(blueprint)
        v.s = adjusted_spawn
        self.lanes[blueprint.origin].append(v)

    def _collect_departures(self, direction, lane, sim_time):
        """Override to despawn vehicles sooner to fit within standard roads."""
        tangent_offset = math.sqrt(ROUNDABOUT_RADIUS**2 - LANE_OFFSET**2) / SCALE
        # Adjust exit distance to fit within standard road bounds
        adjusted_exit = EXIT_LENGTH - tangent_offset
        remaining = []
        for v in lane:
            # Remove when vehicle has traveled adjusted distance past ring exit
            if v.s > v.blueprint.intersection_exit_s + adjusted_exit:
                if v.exit_time is None:
                    v.exit_time = sim_time
                self._record_departure(v)
            else:
                remaining.append(v)
        lane[:] = remaining

    def _draw_control(self, surface):
        cx, cy = self.center
        # Draw only the circulating lane circle (outermost)
        pygame.draw.circle(
            surface, (100, 100, 100), (cx, cy), int(ROUNDABOUT_RADIUS), width=1
        )


class RoundaboutPriorityEntering(IntersectionModel):
    """
    Roundabout with entering priority:
    entering vehicles have right of way and enter freely,
    circulating vehicles must yield to them.
    """

    def __init__(self, center):
        super().__init__("Roundabout (entering priority)", center)

    def update_control(self, dt, sim_time):
        pass

    def step(self, dt, sim_time):
        """Override to enforce car-following in the ring across all directions."""
        self.sim_time = sim_time
        self.update_control(dt, sim_time)

        total_queue = 0

        for direction in DIRECTIONS:
            lane = self.lanes[direction]
            if not lane:
                continue

            # Sort by position
            lane.sort(key=lambda v: v.s)

            # Identify the front-most *approach* vehicle (s <= STOP_LINE_S)
            front_approach = None
            for v in lane:
                if v.s <= STOP_LINE_S:
                    if (front_approach is None) or (v.s > front_approach.s):
                        front_approach = v

            # Update vehicles from front to back
            prev_leader = None
            for idx in range(len(lane) - 1, -1, -1):
                v = lane[idx]
                if v is front_approach:
                    control_limit = self.max_s_for_front(direction, v, sim_time)
                else:
                    control_limit = INF

                if prev_leader is None:
                    max_s = control_limit
                else:
                    gap_limit = prev_leader.s - (VEHICLE_LENGTH + MIN_GAP)
                    max_s = min(control_limit, gap_limit)

                # Additional check: if vehicle is in ring, check for leaders from other
                # directions
                if 0 < v.s <= v.blueprint.intersection_exit_s:
                    ring_leader_limit = self._find_ring_leader_limit(direction, v)
                    max_s = min(max_s, ring_leader_limit)

                self._update_vehicle(v, dt, max_s)
                prev_leader = v

            # Remove vehicles that have left the system
            self._collect_departures(direction, lane, sim_time)
            total_queue += len(lane)

        if total_queue > self.metrics.max_queue:
            self.metrics.max_queue = total_queue

    def _find_ring_leader_limit(self, direction, vehicle):
        """Find the closest vehicle ahead in the ring (from any direction).
        Also yield to entering vehicles from OTHER directions."""
        my_angle = roundabout_vehicle_angle(vehicle)
        if my_angle is None:
            return INF

        min_limit = INF

        # Check for vehicles ahead in the ring
        for other_dir in DIRECTIONS:
            for other in self.lanes[other_dir]:
                if other is vehicle:
                    continue

                other_angle = roundabout_vehicle_angle(other)
                if other_angle is None:
                    continue

                # Calculate angular distance (clockwise mathematically =
                # counterclockwise visually)
                angle_diff = (my_angle - other_angle) % (2.0 * math.pi)

                # Only consider vehicles ahead (small positive angle difference)
                if angle_diff > 0 and angle_diff < math.pi:
                    # Convert angular distance to arc length
                    arc_distance = ROUNDABOUT_RADIUS * angle_diff / SCALE
                    gap_limit = vehicle.s + arc_distance - (VEHICLE_LENGTH + MIN_GAP)
                    min_limit = min(min_limit, gap_limit)

        # Check for entering vehicles at nearby entry points (from OTHER directions
        # only) Start yielding earlier in the ring (after 1m instead of 4m)
        if vehicle.s > 1.0:
            my_angle = roundabout_vehicle_angle(vehicle)
            if my_angle is not None:
                for other_dir in DIRECTIONS:
                    # CRITICAL: Never yield to vehicles from own direction
                    if other_dir == direction:
                        continue

                    entry_angle = direction_entry_angle(other_dir)
                    angle_to_entry = (entry_angle - my_angle) % (2.0 * math.pi)

                    # Check entry points within 90 degrees ahead (increased from 30)
                    if angle_to_entry <= 0 or angle_to_entry > math.pi / 2.0:
                        continue

                    # Check for entering vehicles at this entry point
                    lane = self.lanes[other_dir]
                    for other in lane:
                        # Yield to vehicles closer to entry (within 5m before entry,
                        # moving)
                        if -5.0 <= other.s <= 0 and other.v > 0.5:
                            # Yield by maintaining distance
                            arc_distance = ROUNDABOUT_RADIUS * angle_to_entry / SCALE
                            yield_limit = (
                                vehicle.s
                                + arc_distance
                                - (VEHICLE_LENGTH + MIN_GAP) * 2.0
                            )
                            min_limit = min(min_limit, yield_limit)
                            break  # Only check front vehicle

        return min_limit

    def max_s_for_front(self, direction, vehicle, sim_time):
        # Entering vehicles have absolute priority - they enter freely
        return INF

    def vehicle_position(self, direction, vehicle):
        return roundabout_vehicle_position(self.center, direction, vehicle)

    def spawn_vehicle(self, blueprint):
        """Override to adjust spawn position for roundabout geometry."""
        # Calculate offset from center to tangent
        tangent_offset = math.sqrt(ROUNDABOUT_RADIUS**2 - LANE_OFFSET**2) / SCALE
        # Spawn closer so vehicles fit within standard road length
        adjusted_spawn = -(APPROACH_LENGTH - tangent_offset)
        v = VehicleState(blueprint)
        v.s = adjusted_spawn
        self.lanes[blueprint.origin].append(v)

    def _collect_departures(self, direction, lane, sim_time):
        """Override to despawn vehicles sooner to fit within standard roads."""
        tangent_offset = math.sqrt(ROUNDABOUT_RADIUS**2 - LANE_OFFSET**2) / SCALE
        # Adjust exit distance to fit within standard road bounds
        adjusted_exit = EXIT_LENGTH - tangent_offset
        remaining = []
        for v in lane:
            # Remove when vehicle has traveled adjusted distance past ring exit
            if v.s > v.blueprint.intersection_exit_s + adjusted_exit:
                if v.exit_time is None:
                    v.exit_time = sim_time
                self._record_departure(v)
            else:
                remaining.append(v)
        lane[:] = remaining

    def _draw_control(self, surface):
        cx, cy = self.center
        # Draw only the circulating lane circle (outermost)
        pygame.draw.circle(
            surface, (100, 100, 100), (cx, cy), int(ROUNDABOUT_RADIUS), width=1
        )


# ======================
# Fixed-time signal
# ======================


class FixedTimeSignalizedIntersection(IntersectionModel):
    """Two-phase fixed-time signal (N/S vs E/W)."""

    def __init__(self, center, cycle_length, green_ns, green_ew):
        super().__init__("Fixed-time signal", center)
        self.cycle_length = cycle_length
        self.green_ns = green_ns
        self.green_ew = green_ew

    def update_control(self, dt, sim_time):
        pass

    def is_ns_green(self, sim_time):
        t = sim_time % self.cycle_length
        return t < self.green_ns

    def is_ew_green(self, sim_time):
        t = sim_time % self.cycle_length
        return self.green_ns <= t < (self.green_ns + self.green_ew)

    def max_s_for_front(self, direction, vehicle, sim_time):
        # Already in the intersection or beyond
        if vehicle.s >= STOP_LINE_S:
            return INF

        ns_green = self.is_ns_green(sim_time)
        ew_green = self.is_ew_green(sim_time)

        # Check if this direction's signal is green
        has_green = False
        if direction in (Direction.NORTH, Direction.SOUTH):
            has_green = ns_green
        else:
            has_green = ew_green

        # Red signal - must stop
        if not has_green:
            return STOP_LIMIT_S

        # Green signal - check for conflicts with opposite direction
        opposite = opposite_direction(direction)
        opposite_lane = self.lanes.get(opposite, [])

        # Check if any vehicle from opposite direction is about to enter or in
        # intersection
        for opp_vehicle in opposite_lane:
            # Only check vehicles that are close to or in the intersection
            if (
                opp_vehicle.s >= STOP_LIMIT_S - 5.0
                and opp_vehicle.s <= opp_vehicle.blueprint.intersection_exit_s
            ):
                # Check if we conflict with this opposite vehicle
                if vehicles_conflict_opposite_dirs(
                    vehicle.blueprint.turn, opp_vehicle.blueprint.turn
                ):
                    # We must yield - check if opposite vehicle is ahead of us or close
                    if (
                        opp_vehicle.s >= vehicle.s - 5.0
                    ):  # Opposite vehicle is ahead or very close
                        return STOP_LIMIT_S

        # Green signal and no conflicts
        return INF

    def _draw_control(self, surface):
        cx, cy = self.center

        ns_green = self.is_ns_green(self.sim_time)
        ew_green = self.is_ew_green(self.sim_time)

        # North approach light
        x_n, y_n = crossroad_position((cx, cy), Direction.NORTH, STOP_LIMIT_S - 2.0)
        color_n = SIGNAL_GREEN if ns_green else SIGNAL_RED
        pygame.draw.circle(surface, color_n, (int(x_n), int(y_n)), 6)

        # South approach light
        x_s, y_s = crossroad_position((cx, cy), Direction.SOUTH, STOP_LIMIT_S - 2.0)
        color_s = SIGNAL_GREEN if ns_green else SIGNAL_RED
        pygame.draw.circle(surface, color_s, (int(x_s), int(y_s)), 6)

        # East approach light
        x_e, y_e = crossroad_position((cx, cy), Direction.EAST, STOP_LIMIT_S - 2.0)
        color_e = SIGNAL_GREEN if ew_green else SIGNAL_RED
        pygame.draw.circle(surface, color_e, (int(x_e), int(y_e)), 6)

        # West approach light
        x_w, y_w = crossroad_position((cx, cy), Direction.WEST, STOP_LIMIT_S - 2.0)
        color_w = SIGNAL_GREEN if ew_green else SIGNAL_RED
        pygame.draw.circle(surface, color_w, (int(x_w), int(y_w)), 6)

        # Label the current phase
        font_small = pygame.font.SysFont("Arial", 14)
        phase_text = "NS GREEN" if ns_green else ("EW GREEN" if ew_green else "ALL RED")
        text_surf = font_small.render(phase_text, True, TEXT_COLOR)
        surface.blit(text_surf, (cx - text_surf.get_width() // 2, cy - 90))


# ======================
# 4-way stop (token per vehicle, no deadlock)
# ======================


class FourWayStopIntersection(IntersectionModel):
    """4-way stop with rotating priority allowing multiple vehicles per direction."""

    def __init__(self, center):
        super().__init__("4-way stop", center)
        self.current_direction = None  # which direction currently has right of way
        self.direction_release_time = None  # when to release the current direction
        self.next_dir_index = 0  # rotation among DIRECTIONS
        self.direction_hold_time = 3.0  # seconds to hold a direction active

    def update_control(self, dt, sim_time):
        # If we have a current direction, check if we should release it
        if self.current_direction is not None:
            # Check if there are still vehicles waiting or in intersection from this
            # direction
            lane = self.lanes[self.current_direction]
            has_active = False

            for v in lane:
                # Vehicle is active if it's approaching the intersection or inside it
                if v.s <= v.blueprint.intersection_exit_s:
                    has_active = True
                    break

            # Release if no more active vehicles OR time expired
            if not has_active or sim_time >= self.direction_release_time:
                self.current_direction = None
                self.direction_release_time = None

        # If no current direction, find next direction with waiting vehicles
        if self.current_direction is None:
            candidates = []  # list of directions with waiting vehicles
            for d in DIRECTIONS:
                lane = self.lanes[d]
                if not lane:
                    continue
                # Check if there's a vehicle waiting to enter
                for v in lane:
                    if v.s <= STOP_LINE_S and v.s >= STOP_LIMIT_S - 1.0:
                        candidates.append(d)
                        break

            if not candidates:
                return

            # Use round-robin to pick next direction
            for i in range(len(DIRECTIONS)):
                cand_dir = DIRECTIONS[(self.next_dir_index + i) % len(DIRECTIONS)]
                if cand_dir in candidates:
                    self.current_direction = cand_dir
                    self.direction_release_time = sim_time + self.direction_hold_time
                    self.next_dir_index = (DIRECTIONS.index(cand_dir) + 1) % len(
                        DIRECTIONS
                    )
                    return

    def max_s_for_front(self, direction, vehicle, sim_time):
        if vehicle.s >= STOP_LINE_S:
            # Once in the intersection, keep going
            return INF

        if self.current_direction is None:
            # No direction has right of way, everyone must stop
            return STOP_LIMIT_S

        if direction == self.current_direction:
            # This direction has right of way
            return INF

        # Check if this direction can go simultaneously with the current direction
        opposite = opposite_direction(direction)
        if self.current_direction == opposite:
            # Opposite direction has right of way - check for turn conflicts
            opposite_lane = self.lanes.get(opposite, [])

            # Check if any vehicle from opposite direction conflicts with us
            for opp_vehicle in opposite_lane:
                # Only check vehicles that are close to or in the intersection
                if (
                    opp_vehicle.s >= STOP_LIMIT_S - 5.0
                    and opp_vehicle.s <= opp_vehicle.blueprint.intersection_exit_s
                ):
                    # Check if we conflict with this opposite vehicle
                    if vehicles_conflict_opposite_dirs(
                        vehicle.blueprint.turn, opp_vehicle.blueprint.turn
                    ):
                        # We must yield
                        return STOP_LIMIT_S

            # No conflicts - can proceed
            return INF

        # Different direction without right of way must stop
        return STOP_LIMIT_S

    def _draw_control(self, surface):
        cx, cy = self.center
        font_small = pygame.font.SysFont("Arial", 14)

        offset = 50
        pos_map = {
            Direction.NORTH: (cx, cy - offset),
            Direction.SOUTH: (cx, cy + offset),
            Direction.EAST: (cx + offset, cy),
            Direction.WEST: (cx - offset, cy),
        }

        for _, pos in pos_map.items():
            text = font_small.render("STOP", True, TEXT_COLOR)
            rect = text.get_rect(center=pos)
            pygame.draw.rect(surface, (150, 0, 0), rect.inflate(10, 6), border_radius=4)
            surface.blit(text, rect)

        # Highlight current direction(s) with right of way
        if self.current_direction is not None:
            pos = pos_map[self.current_direction]
            pygame.draw.circle(surface, (0, 255, 0), pos, 8, width=2)

            # Also highlight opposite direction (can go simultaneously)
            opposite = opposite_direction(self.current_direction)
            pos_opp = pos_map[opposite]
            pygame.draw.circle(surface, (0, 255, 0), pos_opp, 8, width=2)


# ======================
# Simulation wrapper
# ======================


class IntersectionSimulation:
    """Manages all four intersection models and the shared vehicle arrivals."""

    def __init__(self, models, arrival_rates, rng):
        self.models = models
        self.arrival_rates = arrival_rates  # veh/s per direction
        self.rng = rng
        self.sim_time = 0.0
        self.next_vehicle_id = 0

    def update(self, dt):
        self.sim_time += dt

        # Generate new arrivals (Poisson with Bernoulli approx per dt)
        for direction in DIRECTIONS:
            lam = self.arrival_rates[direction]
            if lam <= 0.0:
                continue
            p = lam * dt
            if self.rng.random() < p:
                # Always spawn to consume random seeds identically across models
                self._spawn_vehicle(direction)

        # Step each intersection model
        for model in self.models:
            model.step(dt, self.sim_time)

    def _sample_turn(self):
        # Equal probability for now: can be parameterized later.
        r = self.rng.random()
        if r < 1.0 / 3.0:
            return Turn.RIGHT
        elif r < 2.0 / 3.0:
            return Turn.STRAIGHT
        else:
            return Turn.LEFT

    def _intersection_exit_s_for_turn(self, turn, use_roundabout=False):
        if use_roundabout:
            # For roundabouts, use arc length based on ROUNDABOUT_RADIUS
            # Arc length = radius * angle, where radius is in pixels, convert to meters
            radius_meters = ROUNDABOUT_RADIUS / SCALE

            if turn == Turn.RIGHT:
                # 90 degrees = π/2 radians
                return radius_meters * (math.pi / 2.0)
            elif turn == Turn.STRAIGHT:
                # 180 degrees = π radians
                return radius_meters * math.pi
            else:
                # 270 degrees = 3π/2 radians (LEFT turn)
                return radius_meters * (3.0 * math.pi / 2.0)
        else:
            # For regular intersections (signalized, 4-way stop), use simple multiples
            if turn == Turn.RIGHT:
                return BASE_INTERSECTION_S
            elif turn == Turn.STRAIGHT:
                return 2.0 * BASE_INTERSECTION_S
            else:
                return 3.0 * BASE_INTERSECTION_S

    def _spawn_vehicle(self, direction):
        arrival_time = self.sim_time

        # Sample micro-parameters
        desired_speed = max(
            self.rng.gauss(DEFAULT_DESIRED_SPEED, DEFAULT_DESIRED_SPEED_STD), 3.0
        )
        max_accel = max(self.rng.gauss(DEFAULT_MAX_ACCEL, 0.5), 0.5)
        gap_accept = max(
            self.rng.gauss(DEFAULT_GAP_ACCEPT_MEAN, DEFAULT_GAP_ACCEPT_STD), 0.1
        )

        turn = self._sample_turn()

        # Create separate blueprints for each model type
        # (roundabouts need different intersection_exit_s than regular intersections)
        for model in self.models:
            # Check if this model is a roundabout
            use_roundabout = isinstance(
                model, (RoundaboutPriorityCirculating, RoundaboutPriorityEntering)
            )
            intersection_exit_s = self._intersection_exit_s_for_turn(
                turn, use_roundabout
            )

            # Calculate free-flow time based on actual path length for this design
            # NOTE: For roundabouts, the actual path includes adjusted spawn/exit
            # distances
            # TODO: This currently uses a simplified path_length that doesn't account
            # for roundabout-specific spawn adjustments. For accurate delay metrics,
            # should:
            #
            # - For roundabouts: use (APPROACH_LENGTH - tangent_offset) +
            #   intersection_exit_s + (EXIT_LENGTH - tangent_offset)
            # - For regular: use APPROACH_LENGTH + intersection_exit_s + EXIT_LENGTH
            #
            # For now, using a common baseline for rough comparison:
            path_length = EXIT_S - SPAWN_S
            free_flow_time = path_length / desired_speed

            bp = VehicleBlueprint(
                vehicle_id=self.next_vehicle_id,
                arrival_time=arrival_time,
                origin=direction,
                desired_speed=desired_speed,
                max_accel=max_accel,
                gap_acceptance=gap_accept,
                free_flow_time=free_flow_time,
                turn=turn,
                intersection_exit_s=intersection_exit_s,
            )

            # Only add to queue if under cap (independent per model per direction)
            if len(model.lanes[direction]) < MAX_QUEUE_PER_DIRECTION:
                model.spawn_vehicle(bp)

        self.next_vehicle_id += 1


# ======================
# CLI + main loop
# ======================


@click.command()
@click.option(
    "--rate-north",
    type=float,
    default=1200.0,
    show_default=True,
    help="Arrival rate from North (veh/hour).",
)
@click.option(
    "--rate-south",
    type=float,
    default=1200.0,
    show_default=True,
    help="Arrival rate from South (veh/hour).",
)
@click.option(
    "--rate-east",
    type=float,
    default=1200.0,
    show_default=True,
    help="Arrival rate from East (veh/hour).",
)
@click.option(
    "--rate-west",
    type=float,
    default=1200.0,
    show_default=True,
    help="Arrival rate from West (veh/hour).",
)
@click.option(
    "--sim-speed",
    type=float,
    default=2.0,
    show_default=True,
    help="Simulation speed factor (>1 = faster than real time).",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="Random seed for reproducibility.",
)
@click.option(
    "--cycle-length",
    type=float,
    default=60.0,
    show_default=True,
    help="Signal cycle length (s) for fixed-time intersection.",
)
@click.option(
    "--green-ns",
    type=float,
    default=30.0,
    show_default=True,
    help="Green time for N/S (s) in the fixed-time signal.",
)
@click.option(
    "--green-ew",
    type=float,
    default=30.0,
    show_default=True,
    help="Green time for E/W (s) in the fixed-time signal.",
)
def main(
    rate_north,
    rate_south,
    rate_east,
    rate_west,
    sim_speed,
    seed,
    cycle_length,
    green_ns,
    green_ew,
):
    """Run the interactive GUI simulation comparing four intersection controls."""
    rng = random.Random(seed)

    arrival_rates = {
        Direction.NORTH: rate_north / 3600.0,
        Direction.SOUTH: rate_south / 3600.0,
        Direction.EAST: rate_east / 3600.0,
        Direction.WEST: rate_west / 3600.0,
    }

    pygame.init()
    pygame.display.set_caption("Intersection Control Comparison")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)

    # Centers of the four scenarios (2x2 grid)
    centers = [
        (WINDOW_WIDTH // 4, WINDOW_HEIGHT // 4),  # top-left
        (3 * WINDOW_WIDTH // 4, WINDOW_HEIGHT // 4),  # top-right
        (WINDOW_WIDTH // 4, 3 * WINDOW_HEIGHT // 4),  # bottom-left
        (3 * WINDOW_WIDTH // 4, 3 * WINDOW_HEIGHT // 4),  # bottom-right
    ]

    # Create intersection models
    models = [
        RoundaboutPriorityCirculating(center=centers[0]),
        RoundaboutPriorityEntering(center=centers[1]),
    ]

    if green_ns + green_ew > cycle_length:
        raise ValueError("green_ns + green_ew must be <= cycle_length")

    models.append(
        FixedTimeSignalizedIntersection(
            center=centers[2],
            cycle_length=cycle_length,
            green_ns=green_ns,
            green_ew=green_ew,
        )
    )

    models.append(FourWayStopIntersection(center=centers[3]))

    sim = IntersectionSimulation(models=models, arrival_rates=arrival_rates, rng=rng)

    running = True
    paused = False
    while running:
        real_dt = clock.tick(FPS) / 1000.0
        dt = real_dt * sim_speed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused:
            sim.update(dt)

        screen.fill(BACKGROUND_COLOR)
        for model in models:
            model.draw(screen, font)

        # Global time display and pause indicator
        time_text = font.render(f"Sim time: {sim.sim_time:7.1f}s", True, TEXT_COLOR)
        screen.blit(time_text, (10, 10))

        if paused:
            pause_text = font.render(
                "PAUSED (Press SPACE to continue)", True, (255, 255, 0)
            )
            screen.blit(pause_text, (10, 30))
        else:
            pause_hint = font.render("Press SPACE to pause", True, (150, 150, 150))
            screen.blit(pause_hint, (10, 30))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
