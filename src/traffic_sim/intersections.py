"""Concrete implementations of intersection control strategies."""

import math

import pygame

from traffic_sim.config import (
    APPROACH_LENGTH,
    EXIT_LENGTH,
    INF,
    LANE_OFFSET,
    MIN_GAP,
    ROUNDABOUT_RADIUS,
    SCALE,
    SIGNAL_GREEN,
    SIGNAL_RED,
    STOP_LIMIT_S,
    STOP_LINE_S,
    TEXT_COLOR,
    VEHICLE_LENGTH,
)
from traffic_sim.core import (
    DIRECTIONS,
    Direction,
    IntersectionModel,
    Turn,
    VehicleBlueprint,
    VehicleState,
)
from traffic_sim.geometry import (
    crossroad_position,
    crossroad_position_with_turn,
    direction_entry_angle,
    opposite_direction,
    roundabout_vehicle_angle,
    roundabout_vehicle_position,
)


def vehicles_conflict_opposite_dirs(turn_a: Turn, turn_b: Turn) -> bool:
    """Determine if turn_a must yield to turn_b from opposite directions.

    Args:
        turn_a: Turn maneuver of subject vehicle
        turn_b: Turn maneuver of opposing vehicle

    Returns:
        True if turn_a yields to turn_b
    """
    if turn_a == Turn.LEFT and turn_b in (Turn.STRAIGHT, Turn.RIGHT):
        return True
    return False


class RoundaboutPriorityCirculating(IntersectionModel):
    """Roundabout with standard priority: entering yields to circulating.

    Implements gap acceptance for entering vehicles and circular car-following
    within the ring.
    """

    def __init__(self, center: tuple[int, int]):
        super().__init__("Roundabout (circ. priority)", center)

    def update_control(self, dt: float, sim_time: float) -> None:
        """No active control needed for this intersection type."""
        pass

    def step(self, dt: float, sim_time: float) -> None:
        """Advance simulation with roundabout-specific ring leader logic."""
        self.sim_time = sim_time
        self.update_control(dt, sim_time)

        total_queue = 0
        for direction in DIRECTIONS:
            lane = self.lanes[direction]
            if not lane:
                continue

            lane.sort(key=lambda v: v.s)

            # Identify front vehicle still on approach
            front_approach = None
            for v in lane:
                if v.s <= STOP_LINE_S:
                    if (front_approach is None) or (v.s > front_approach.s):
                        front_approach = v

            # Backward pass: propagate vehicles with appropriate constraints
            prev_leader = None
            for idx in range(len(lane) - 1, -1, -1):
                v = lane[idx]

                # Apply control constraints to front vehicle
                if v is front_approach:
                    control_limit = self.max_s_for_front(direction, v, sim_time)
                else:
                    control_limit = INF

                # Enforce car-following gap
                if prev_leader is None:
                    max_s = control_limit
                else:
                    gap_limit = prev_leader.s - (VEHICLE_LENGTH + MIN_GAP)
                    max_s = min(control_limit, gap_limit)

                # Apply circular ring following constraints
                if 0 < v.s <= v.blueprint.intersection_exit_s:
                    ring_limit = self._find_ring_leader_limit(v)
                    max_s = min(max_s, ring_limit)

                self._update_vehicle(v, dt, max_s)
                prev_leader = v

            self._collect_departures(lane, sim_time)
            total_queue += len(lane)

        if total_queue > self.metrics.max_queue:
            self.metrics.max_queue = total_queue

    def _find_ring_leader_limit(self, vehicle: VehicleState) -> float:
        """Compute position limit based on vehicles ahead in the circular ring.

        Args:
            vehicle: Subject vehicle currently in the ring

        Returns:
            Maximum allowable position to maintain safe gap
        """
        my_angle = roundabout_vehicle_angle(vehicle)
        if my_angle == -999.0:
            return INF

        min_limit = INF
        for other_dir in DIRECTIONS:
            for other in self.lanes[other_dir]:
                if other is vehicle:
                    continue

                other_angle = roundabout_vehicle_angle(other)
                if other_angle == -999.0:
                    continue

                # Check if other vehicle is ahead in circulation direction
                angle_diff = (my_angle - other_angle) % (2.0 * math.pi)
                if 0 < angle_diff < math.pi:
                    arc_dist = ROUNDABOUT_RADIUS * angle_diff / SCALE
                    gap_limit = vehicle.s + arc_dist - (VEHICLE_LENGTH + MIN_GAP)
                    min_limit = min(min_limit, gap_limit)
        return min_limit

    def _has_conflict(
        self,
        direction: Direction,
        vehicle: VehicleState,
        safe_angle: float,
        min_gap: float,
    ) -> bool:
        """Check if entering vehicle conflicts with circulating traffic.

        Args:
            direction: Entry direction
            vehicle: Vehicle attempting to enter
            safe_angle: Angular sector to check for conflicts (radians)
            min_gap: Minimum acceptable time gap (seconds)

        Returns:
            True if a conflict exists
        """
        theta_entry = direction_entry_angle(direction)
        bp = vehicle.blueprint
        dist_to_entry = -vehicle.s
        approach_speed = max(bp.desired_speed, 1.0)
        time_to_entry = max(dist_to_entry / approach_speed, 0.0)
        radius_m = ROUNDABOUT_RADIUS / SCALE

        for d in DIRECTIONS:
            for other in self.lanes[d]:
                if other is vehicle:
                    continue
                theta_other = roundabout_vehicle_angle(other)
                if theta_other == -999.0:
                    continue

                # Check if other vehicle is in critical conflict zone
                dist_cw = (theta_other - theta_entry) % (2.0 * math.pi)
                if dist_cw <= 0.0 or dist_cw > safe_angle:
                    continue

                # Estimate time for other vehicle to reach entry point
                arc = radius_m * dist_cw
                other_speed = max(other.v, 1.0)
                time_other = arc / other_speed

                if time_other < time_to_entry + min_gap:
                    return True
        return False

    def max_s_for_front(
        self, direction: Direction, vehicle: VehicleState, sim_time: float
    ) -> float:
        """Determine stopping constraint for vehicle approaching roundabout entry.

        Args:
            direction: Entry direction
            vehicle: Lead vehicle on approach
            sim_time: Current simulation time

        Returns:
            Maximum position (INF if no stopping required)
        """
        if vehicle.s >= STOP_LINE_S:
            return INF

        safe_angle = 2.0 * math.pi / 3.0
        min_gap_time = 1.0
        if self._has_conflict(direction, vehicle, safe_angle, min_gap_time):
            return STOP_LIMIT_S
        return INF

    def vehicle_position(
        self, direction: Direction, vehicle: VehicleState
    ) -> tuple[float, float]:
        """Map vehicle position to screen coordinates."""
        return roundabout_vehicle_position(self.center, direction, vehicle)

    def spawn_vehicle(self, blueprint: VehicleBlueprint) -> None:
        """Add vehicle to lane with adjusted spawn position for roundabout geometry."""
        tangent_offset = math.sqrt(ROUNDABOUT_RADIUS**2 - LANE_OFFSET**2) / SCALE
        adjusted_spawn = -(APPROACH_LENGTH - tangent_offset)
        v = VehicleState(blueprint)
        v.s = adjusted_spawn
        self.lanes[blueprint.origin].append(v)

    def _collect_departures(self, lane: list[VehicleState], sim_time: float) -> None:
        """Remove departed vehicles with adjusted exit point for roundabout geometry."""
        tangent_offset = math.sqrt(ROUNDABOUT_RADIUS**2 - LANE_OFFSET**2) / SCALE
        adjusted_exit = EXIT_LENGTH - tangent_offset
        remaining = []
        for v in lane:
            if v.s > v.blueprint.intersection_exit_s + adjusted_exit:
                if v.exit_time is None:
                    v.exit_time = sim_time
                self._record_departure(v)
            else:
                remaining.append(v)
        lane[:] = remaining

    def _draw_control(self, surface: pygame.Surface) -> None:
        """Draw roundabout ring outline."""
        cx, cy = self.center
        pygame.draw.circle(
            surface, (100, 100, 100), (cx, cy), int(ROUNDABOUT_RADIUS), width=1
        )


class RoundaboutPriorityEntering(RoundaboutPriorityCirculating):
    """Non-standard roundabout where entering has priority over circulating.

    This inverted priority rule causes circulating vehicles to yield to entering
    traffic, which is rare in practice but useful for comparison studies.
    """

    def __init__(self, center: tuple[int, int]):
        super(RoundaboutPriorityCirculating, self).__init__(
            "Roundabout (enter priority)", center
        )

    def max_s_for_front(
        self, direction: Direction, vehicle: VehicleState, sim_time: float
    ) -> float:
        """No entry restrictions: entering vehicles never stop."""
        return INF

    def _find_ring_leader_limit(self, vehicle: VehicleState) -> float:
        """Compute ring position limit with additional yielding to entering traffic.

        Args:
            vehicle: Circulating vehicle

        Returns:
            Maximum position considering both ring leaders and entering vehicles
        """
        min_limit = super()._find_ring_leader_limit(vehicle)

        # Check if we need to yield to entering vehicles
        if vehicle.s > 1.0:
            my_angle = roundabout_vehicle_angle(vehicle)
            if my_angle != -999.0:
                for other_dir in DIRECTIONS:
                    if other_dir == vehicle.blueprint.origin:
                        continue

                    entry_angle = direction_entry_angle(other_dir)
                    angle_to_entry = (entry_angle - my_angle) % (2.0 * math.pi)

                    # Only consider entry points ahead in next quadrant
                    if angle_to_entry <= 0 or angle_to_entry > math.pi / 2.0:
                        continue

                    # Check if vehicle is actively entering from this direction
                    for other in self.lanes[other_dir]:
                        if -5.0 <= other.s <= 0 and other.v > 0.5:
                            arc_dist = ROUNDABOUT_RADIUS * angle_to_entry / SCALE
                            yield_limit = (
                                vehicle.s + arc_dist - (VEHICLE_LENGTH + MIN_GAP) * 2.0
                            )
                            min_limit = min(min_limit, yield_limit)
                            break
        return min_limit


class FixedTimeSignalizedIntersection(IntersectionModel):
    """Traffic signal with fixed phase durations and no actuation.

    Alternates between north-south and east-west green phases with
    fixed cycle timing.
    """

    def __init__(
        self,
        center: tuple[int, int],
        cycle_length: float,
        green_ns: float,
        green_ew: float,
    ):
        super().__init__("Fixed-time signal", center)
        self.cycle_length = cycle_length
        self.green_ns = green_ns
        self.green_ew = green_ew

    def update_control(self, dt: float, sim_time: float) -> None:
        """No state updates needed for fixed-time operation."""
        pass

    def is_ns_green(self, sim_time: float) -> bool:
        """Check if north-south phase is currently green."""
        t = sim_time % self.cycle_length
        return t < self.green_ns

    def is_ew_green(self, sim_time: float) -> bool:
        """Check if east-west phase is currently green."""
        t = sim_time % self.cycle_length
        return self.green_ns <= t < (self.green_ns + self.green_ew)

    def max_s_for_front(
        self, direction: Direction, vehicle: VehicleState, sim_time: float
    ) -> float:
        """Determine stopping constraint based on signal phase and opposing conflicts.

        Args:
            direction: Approach direction
            vehicle: Lead vehicle
            sim_time: Current simulation time

        Returns:
            Maximum position (STOP_LIMIT_S on red, INF on green with no conflicts)
        """
        if vehicle.s >= STOP_LINE_S:
            return INF

        ns_green = self.is_ns_green(sim_time)
        ew_green = self.is_ew_green(sim_time)

        if direction in (Direction.NORTH, Direction.SOUTH):
            has_green = ns_green
        else:
            has_green = ew_green

        if not has_green:
            return STOP_LIMIT_S

        # Check for conflicts with opposing left-turning traffic
        opposite = opposite_direction(direction)
        opposite_lane = self.lanes.get(opposite, [])
        for opp in opposite_lane:
            if STOP_LIMIT_S - 5.0 <= opp.s <= opp.blueprint.intersection_exit_s:
                if vehicles_conflict_opposite_dirs(
                    vehicle.blueprint.turn, opp.blueprint.turn
                ):
                    if opp.s >= vehicle.s - 5.0:
                        return STOP_LIMIT_S
        return INF

    def vehicle_position(
        self, direction: Direction, vehicle: VehicleState
    ) -> tuple[float, float]:
        """Map vehicle position to screen coordinates."""
        return crossroad_position_with_turn(
            self.center, direction, vehicle.s, vehicle.blueprint.turn
        )

    def _draw_control(self, surface: pygame.Surface) -> None:
        """Render signal indications for all approaches."""
        cx, cy = self.center
        ns_green = self.is_ns_green(self.sim_time)
        ew_green = self.is_ew_green(self.sim_time)

        # Draw signal heads at each approach
        for d in DIRECTIONS:
            x, y = crossroad_position((cx, cy), d, STOP_LIMIT_S - 2.0)
            is_ns = d in (Direction.NORTH, Direction.SOUTH)
            color = (
                SIGNAL_GREEN
                if (is_ns and ns_green) or (not is_ns and ew_green)
                else SIGNAL_RED
            )
            pygame.draw.circle(surface, color, (int(x), int(y)), 6)

        # Display current phase
        font_small = pygame.font.SysFont("Arial", 14)
        phase = "NS GREEN" if ns_green else ("EW GREEN" if ew_green else "ALL RED")
        text = font_small.render(phase, True, TEXT_COLOR)
        surface.blit(text, (cx - text.get_width() // 2, cy - 90))


class FourWayStopIntersection(IntersectionModel):
    """All-way stop-controlled intersection with rotating priority.

    Grants right-of-way sequentially to each direction that has waiting
    vehicles, enforcing stop compliance and turn order.
    """

    def __init__(self, center: tuple[int, int]):
        super().__init__("4-way stop", center)
        self.current_direction: Direction | None = None
        self.direction_release_time: float | None = None
        self.next_dir_index = 0
        self.direction_hold_time = 3.0

    def update_control(self, dt: float, sim_time: float) -> None:
        """Manage turn-taking logic among stopped vehicles."""
        if self.current_direction is not None:
            lane = self.lanes[self.current_direction]
            has_active = any(v.s <= v.blueprint.intersection_exit_s for v in lane)

            # Release current direction if no active vehicles or timeout
            if not has_active or (
                self.direction_release_time and sim_time >= self.direction_release_time
            ):
                self.current_direction = None
                self.direction_release_time = None

        # Assign priority to next waiting direction
        if self.current_direction is None:
            candidates = []
            for d in DIRECTIONS:
                lane = self.lanes[d]
                for v in lane:
                    if STOP_LIMIT_S - 1.0 <= v.s <= STOP_LINE_S:
                        candidates.append(d)
                        break

            if not candidates:
                return

            # Select next direction in round-robin order
            for i in range(len(DIRECTIONS)):
                cand_dir = DIRECTIONS[(self.next_dir_index + i) % len(DIRECTIONS)]
                if cand_dir in candidates:
                    self.current_direction = cand_dir
                    self.direction_release_time = sim_time + self.direction_hold_time
                    self.next_dir_index = (DIRECTIONS.index(cand_dir) + 1) % len(
                        DIRECTIONS
                    )
                    return

    def max_s_for_front(
        self, direction: Direction, vehicle: VehicleState, sim_time: float
    ) -> float:
        """Determine if vehicle can proceed through stop-controlled intersection.

        Args:
            direction: Approach direction
            vehicle: Lead vehicle
            sim_time: Current simulation time

        Returns:
            STOP_LIMIT_S if must stop, INF if cleared to proceed
        """
        if vehicle.s >= STOP_LINE_S:
            return INF
        if self.current_direction is None:
            return STOP_LIMIT_S
        if direction == self.current_direction:
            return INF

        # Check if on opposite approach with compatible turn
        opposite = opposite_direction(direction)
        if self.current_direction == opposite:
            opposite_lane = self.lanes.get(opposite, [])
            for opp in opposite_lane:
                if STOP_LIMIT_S - 5.0 <= opp.s <= opp.blueprint.intersection_exit_s:
                    if vehicles_conflict_opposite_dirs(
                        vehicle.blueprint.turn, opp.blueprint.turn
                    ):
                        return STOP_LIMIT_S
            return INF
        return STOP_LIMIT_S

    def vehicle_position(
        self, direction: Direction, vehicle: VehicleState
    ) -> tuple[float, float]:
        """Map vehicle position to screen coordinates."""
        return crossroad_position_with_turn(
            self.center, direction, vehicle.s, vehicle.blueprint.turn
        )

    def _draw_control(self, surface: pygame.Surface) -> None:
        """Render stop signs and indicate which direction has right-of-way."""
        cx, cy = self.center
        font_small = pygame.font.SysFont("Arial", 14)
        offset = 50
        pos_map = {
            Direction.NORTH: (cx, cy - offset),
            Direction.SOUTH: (cx, cy + offset),
            Direction.EAST: (cx + offset, cy),
            Direction.WEST: (cx - offset, cy),
        }

        # Draw stop signs at all approaches
        for _, pos in pos_map.items():
            text = font_small.render("STOP", True, TEXT_COLOR)
            rect = text.get_rect(center=pos)
            pygame.draw.rect(surface, (150, 0, 0), rect.inflate(10, 6), border_radius=4)
            surface.blit(text, rect)

        # Highlight current direction and opposite with green rings
        if self.current_direction is not None:
            pos = pos_map[self.current_direction]
            pygame.draw.circle(surface, (0, 255, 0), pos, 8, width=2)
            opp = opposite_direction(self.current_direction)
            pygame.draw.circle(surface, (0, 255, 0), pos_map[opp], 8, width=2)
