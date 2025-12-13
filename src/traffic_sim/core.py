"""Core data structures and abstract base class for intersection models."""

from abc import ABC, abstractmethod
from enum import Enum, auto

import pygame

from traffic_sim.config import (
    APPROACH_LENGTH,
    EXIT_LENGTH,
    EXIT_S,
    INF,
    MIN_GAP,
    ROAD_COLOR,
    ROAD_EDGE_COLOR,
    SCALE,
    SPAWN_S,
    STOP_LINE_S,
    TEXT_COLOR,
    VEHICLE_COLOR,
    VEHICLE_LENGTH,
)


class Direction(Enum):
    """Cardinal directions for traffic flow."""

    NORTH = auto()
    EAST = auto()
    SOUTH = auto()
    WEST = auto()


class Turn(Enum):
    """Vehicle turn maneuvers at intersections."""

    RIGHT = auto()
    STRAIGHT = auto()
    LEFT = auto()


DIRECTIONS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]

VEHICLE_COLORS = {
    Direction.NORTH: (100, 150, 255),
    Direction.SOUTH: (255, 100, 100),
    Direction.EAST: (100, 255, 150),
    Direction.WEST: (255, 200, 100),
}


class VehicleBlueprint:
    """Immutable vehicle characteristics sampled at spawn time.

    Parameters are drawn from distributions and remain constant throughout
    the vehicle's lifecycle to represent driver and vehicle heterogeneity.
    """

    def __init__(
        self,
        vehicle_id: int,
        arrival_time: float,
        origin: Direction,
        desired_speed: float,
        max_accel: float,
        gap_acceptance: float,
        free_flow_time: float,
        turn: Turn,
        intersection_exit_s: float,
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
    """Dynamic state of a vehicle during simulation.

    Tracks position, velocity, and state flags that change over time.
    """

    def __init__(self, blueprint: VehicleBlueprint):
        self.blueprint = blueprint
        self.s = SPAWN_S
        self.v = blueprint.desired_speed
        self.entered = False
        self.exit_time: float | None = None


class IntersectionMetrics:
    """Performance metrics for intersection evaluation.

    Tracks throughput, delay statistics, and congestion measures.
    """

    def __init__(self):
        self.departed_count = 0
        self.total_delay = 0.0
        self.delays: list[float] = []
        self.max_queue = 0

    @property
    def mean_delay(self) -> float:
        """Average delay per departed vehicle."""
        if self.departed_count == 0:
            return 0.0
        return self.total_delay / self.departed_count

    @property
    def p95_delay(self) -> float:
        """95th percentile delay."""
        if not self.delays:
            return 0.0
        sorted_delays = sorted(self.delays)
        idx = int(0.95 * (len(sorted_delays) - 1))
        return sorted_delays[idx]


class IntersectionModel(ABC):
    """Abstract base for intersection control strategies.

    Subclasses implement specific control logic (signals, stop signs,
    roundabouts) while sharing common vehicle propagation and metrics.
    """

    def __init__(self, name: str, center: tuple[int, int]):
        self.name = name
        self.center = center
        self.lanes: dict[Direction, list[VehicleState]] = {d: [] for d in DIRECTIONS}
        self.metrics = IntersectionMetrics()
        self.sim_time = 0.0

    @abstractmethod
    def update_control(self, dt: float, sim_time: float) -> None:
        """Update control state (e.g., traffic signal phase transitions)."""
        pass

    @abstractmethod
    def max_s_for_front(
        self, direction: Direction, vehicle: VehicleState, sim_time: float
    ) -> float:
        """Compute maximum allowable position for the leading vehicle.

        Returns INF if no control restriction applies.
        """
        pass

    @abstractmethod
    def vehicle_position(
        self, direction: Direction, vehicle: VehicleState
    ) -> tuple[float, float]:
        """Map vehicle's longitudinal coordinate to screen (x, y)."""
        pass

    @abstractmethod
    def _draw_control(self, surface: pygame.Surface) -> None:
        """Render control-specific visual elements (signals, signs, etc.)."""
        pass

    def spawn_vehicle(self, blueprint: VehicleBlueprint) -> None:
        """Add a new vehicle to the appropriate lane."""
        self.lanes[blueprint.origin].append(VehicleState(blueprint))

    def step(self, dt: float, sim_time: float) -> None:
        """Advance simulation by one timestep."""
        self.sim_time = sim_time
        self.update_control(dt, sim_time)

        total_queue = 0
        for direction in DIRECTIONS:
            lane = self.lanes[direction]
            if not lane:
                continue

            lane.sort(key=lambda v: v.s)

            # Identify the front vehicle still on approach
            front_approach = None
            for v in lane:
                if v.s <= STOP_LINE_S:
                    if (front_approach is None) or (v.s > front_approach.s):
                        front_approach = v

            # Backward pass: propagate each vehicle respecting leader gaps
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

                self._update_vehicle(v, dt, max_s)
                prev_leader = v

            self._collect_departures(lane, sim_time)
            total_queue += len(lane)

        if total_queue > self.metrics.max_queue:
            self.metrics.max_queue = total_queue

    def _update_vehicle(self, vehicle: VehicleState, dt: float, max_s: float) -> None:
        """Accelerate vehicle toward desired speed and advance position."""
        bp = vehicle.blueprint
        desired_v = max(bp.desired_speed, 1.0)

        # Accelerate toward desired speed
        if vehicle.v < desired_v:
            v_candidate = vehicle.v + bp.max_accel * dt
            vehicle.v = min(v_candidate, desired_v)

        # Advance position, respecting max_s constraint
        s_nominal = vehicle.s + vehicle.v * dt
        vehicle.s = min(s_nominal, max_s)

        # Mark entry into intersection zone
        if (not vehicle.entered) and vehicle.s >= STOP_LINE_S:
            vehicle.entered = True

    def _collect_departures(self, lane: list[VehicleState], sim_time: float) -> None:
        """Remove vehicles past exit point and record metrics."""
        remaining = []
        for v in lane:
            if v.s > EXIT_S:
                if v.exit_time is None:
                    v.exit_time = sim_time
                self._record_departure(v)
            else:
                remaining.append(v)
        lane[:] = remaining

    def _record_departure(self, vehicle: VehicleState) -> None:
        """Log departure metrics for completed trips."""
        if vehicle.exit_time is None:
            return
        travel_time = vehicle.exit_time - vehicle.blueprint.arrival_time
        delay = max(travel_time - vehicle.blueprint.free_flow_time, 0.0)
        self.metrics.departed_count += 1
        self.metrics.total_delay += delay
        self.metrics.delays.append(delay)

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Render complete intersection visualization."""
        self._draw_roads(surface)
        self._draw_control(surface)
        self._draw_vehicles(surface)
        self._draw_text(surface, font)

    def _draw_roads(self, surface: pygame.Surface) -> None:
        """Draw road infrastructure (lanes and edges)."""
        cx, cy = self.center
        road_half_width = 18
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

    def _draw_vehicles(self, surface: pygame.Surface) -> None:
        """Render all vehicles within visible range."""
        for direction in DIRECTIONS:
            for v in self.lanes[direction]:
                if SPAWN_S - VEHICLE_LENGTH <= v.s <= EXIT_S + VEHICLE_LENGTH:
                    self._draw_single_vehicle(surface, direction, v)

    def _draw_single_vehicle(
        self, surface: pygame.Surface, direction: Direction, vehicle: VehicleState
    ) -> None:
        """Draw a single vehicle as a colored square."""
        x_pix, y_pix = self.vehicle_position(direction, vehicle)
        size = 10
        rect = pygame.Rect(int(x_pix - size / 2), int(y_pix - size / 2), size, size)
        color = VEHICLE_COLORS.get(vehicle.blueprint.origin, VEHICLE_COLOR)
        pygame.draw.rect(surface, color, rect)

    def _draw_text(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Display intersection name and performance statistics."""
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
