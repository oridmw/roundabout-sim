"""Geometric calculations for road and vehicle positioning."""

import math
from typing import Tuple

from traffic_sim.config import (
    LANE_OFFSET,
    ROUNDABOUT_RADIUS,
    SCALE,
)
from traffic_sim.core import Direction, Turn, VehicleBlueprint, VehicleState


def opposite_direction(d: Direction) -> Direction:
    """Return the opposite cardinal direction."""
    if d == Direction.NORTH:
        return Direction.SOUTH
    if d == Direction.SOUTH:
        return Direction.NORTH
    if d == Direction.EAST:
        return Direction.WEST
    return Direction.EAST


def crossroad_position(
    center: Tuple[int, int], direction: Direction, s_world: float
) -> Tuple[float, float]:
    """Calculate basic (x,y) for a straight road given longitudinal s."""
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


def exit_side_for_cross(origin: Direction, turn: Turn) -> Direction:
    """Determine exit direction based on origin and turn type."""
    if turn == Turn.STRAIGHT:
        return opposite_direction(origin)

    order = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
    idx = order.index(origin)

    if turn == Turn.RIGHT:
        return order[(idx + 1) % 4]
    else:  # Turn.LEFT
        return order[(idx - 1) % 4]


def crossroad_position_with_turn(
    center: Tuple[int, int], origin: Direction, s_world: float, turn: Turn
) -> Tuple[float, float]:
    """Calculate (x,y) for a standard intersection including turns."""
    if turn == Turn.STRAIGHT:
        return crossroad_position(center, origin, s_world)

    lane_offset_m = LANE_OFFSET / SCALE

    if turn == Turn.RIGHT:
        s_origin_corner = -lane_offset_m
        s_exit_corner = +lane_offset_m
    else:  # Turn.LEFT
        s_origin_corner = +lane_offset_m
        s_exit_corner = -lane_offset_m

    if s_world <= s_origin_corner:
        return crossroad_position(center, origin, s_world)

    exit_dir = exit_side_for_cross(origin, turn)
    s_on_exit = s_exit_corner + (s_world - s_origin_corner)
    return crossroad_position(center, exit_dir, s_on_exit)


def direction_entry_angle(direction: Direction) -> float:
    """Return angle (radians) where direction enters roundabout."""
    if direction == Direction.EAST:
        return 0.0
    if direction == Direction.SOUTH:
        return math.pi / 2.0
    if direction == Direction.WEST:
        return math.pi
    return -math.pi / 2.0  # NORTH


def turn_step_count(turn: Turn) -> int:
    """Return number of 90-degree sectors to traverse."""
    if turn == Turn.RIGHT:
        return 1
    if turn == Turn.STRAIGHT:
        return 2
    return 3  # LEFT


def exit_heading_for_movement(origin: Direction, turn: Turn) -> Direction:
    """Return the compass direction of the exit arm."""
    order = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
    idx = order.index(origin)
    steps = turn_step_count(turn)
    dest_side = order[(idx - steps) % len(order)]
    return opposite_direction(dest_side)


def roundabout_vehicle_angle(v: VehicleState) -> float:
    """Return angular position (theta) in roundabout."""
    bp = v.blueprint
    if v.s <= 0.0 or v.s > bp.intersection_exit_s:
        return -999.0  # Sentinel for "not in ring"

    theta_entry = direction_entry_angle(bp.origin)
    steps = turn_step_count(bp.turn)
    delta_theta = steps * (math.pi / 2.0)
    u = max(0.0, min(1.0, v.s / bp.intersection_exit_s))
    return theta_entry - u * delta_theta


def roundabout_vehicle_position(
    center: Tuple[int, int], direction: Direction, vehicle: VehicleState
) -> Tuple[float, float]:
    """Calculate (x,y) for vehicle in roundabout geometry."""
    bp: VehicleBlueprint = vehicle.blueprint
    s = vehicle.s
    cx, cy = center
    tangent_dist = math.sqrt(ROUNDABOUT_RADIUS**2 - LANE_OFFSET**2)

    # Approach
    if s <= 0.0:
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

    # Inside Ring
    if s <= bp.intersection_exit_s:
        theta_entry = direction_entry_angle(direction)
        steps = turn_step_count(bp.turn)
        u = max(0.0, min(1.0, s / bp.intersection_exit_s))
        theta = theta_entry - u * steps * (math.pi / 2.0)
        x = cx + ROUNDABOUT_RADIUS * math.cos(theta)
        y = cy + ROUNDABOUT_RADIUS * math.sin(theta)
        return x, y

    # Exit
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
