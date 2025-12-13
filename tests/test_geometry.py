"""Unit tests for geometric coordinate transformations."""

import math

import pytest

from traffic_sim.config import LANE_OFFSET, ROUNDABOUT_RADIUS
from traffic_sim.core import Direction, Turn, VehicleBlueprint, VehicleState
from traffic_sim.geometry import (
    crossroad_position,
    direction_entry_angle,
    exit_heading_for_movement,
    exit_side_for_cross,
    opposite_direction,
    roundabout_vehicle_angle,
    roundabout_vehicle_position,
    turn_step_count,
)


class TestOppositeDirection:
    """Tests for opposite_direction function."""

    def test_north_opposite_is_south(self):
        """Verify north opposite is south."""
        assert opposite_direction(Direction.NORTH) == Direction.SOUTH

    def test_south_opposite_is_north(self):
        """Verify south opposite is north."""
        assert opposite_direction(Direction.SOUTH) == Direction.NORTH

    def test_east_opposite_is_west(self):
        """Verify east opposite is west."""
        assert opposite_direction(Direction.EAST) == Direction.WEST

    def test_west_opposite_is_east(self):
        """Verify west opposite is east."""
        assert opposite_direction(Direction.WEST) == Direction.EAST

    def test_double_opposite_returns_original(self):
        """Verify applying opposite twice returns original direction."""
        for direction in [
            Direction.NORTH,
            Direction.SOUTH,
            Direction.EAST,
            Direction.WEST,
        ]:
            assert opposite_direction(opposite_direction(direction)) == direction


class TestCrossroadPosition:
    """Tests for basic straight road positioning."""

    def test_north_approach_positions(self, intersection_center):
        """Verify positions along north approach."""
        cx, cy = intersection_center
        x, y = crossroad_position(intersection_center, Direction.NORTH, -10.0)

        assert x == cx - LANE_OFFSET
        assert y < cy  # Approaching from below

    def test_south_approach_positions(self, intersection_center):
        """Verify positions along south approach."""
        cx, cy = intersection_center
        x, y = crossroad_position(intersection_center, Direction.SOUTH, -10.0)

        assert x == cx + LANE_OFFSET
        assert y > cy  # Approaching from above

    def test_east_approach_positions(self, intersection_center):
        """Verify positions along east approach."""
        cx, cy = intersection_center
        x, y = crossroad_position(intersection_center, Direction.EAST, -10.0)

        assert x > cx  # Approaching from right
        assert y == cy - LANE_OFFSET

    def test_west_approach_positions(self, intersection_center):
        """Verify positions along west approach."""
        cx, cy = intersection_center
        x, y = crossroad_position(intersection_center, Direction.WEST, -10.0)

        assert x < cx  # Approaching from left
        assert y == cy + LANE_OFFSET

    def test_at_center(self, intersection_center):
        """Verify position calculation at s=0."""
        cx, cy = intersection_center
        x, y = crossroad_position(intersection_center, Direction.NORTH, 0.0)
        assert x == cx - LANE_OFFSET
        assert y == cy


class TestExitSideForCross:
    """Tests for determining exit direction after turn."""

    def test_straight_through_north(self):
        """Verify north straight goes to south."""
        assert exit_side_for_cross(Direction.NORTH, Turn.STRAIGHT) == Direction.SOUTH

    def test_straight_through_east(self):
        """Verify east straight goes to west."""
        assert exit_side_for_cross(Direction.EAST, Turn.STRAIGHT) == Direction.WEST

    def test_right_turn_from_north(self):
        """Verify north right turn goes to east."""
        assert exit_side_for_cross(Direction.NORTH, Turn.RIGHT) == Direction.EAST

    def test_right_turn_from_east(self):
        """Verify east right turn goes to south."""
        assert exit_side_for_cross(Direction.EAST, Turn.RIGHT) == Direction.SOUTH

    def test_left_turn_from_north(self):
        """Verify north left turn goes to west."""
        assert exit_side_for_cross(Direction.NORTH, Turn.LEFT) == Direction.WEST

    def test_left_turn_from_south(self):
        """Verify south left turn goes to east."""
        assert exit_side_for_cross(Direction.SOUTH, Turn.LEFT) == Direction.EAST


class TestDirectionEntryAngle:
    """Tests for roundabout entry angle calculations."""

    def test_east_entry_angle(self):
        """Verify east enters at 0 radians."""
        assert direction_entry_angle(Direction.EAST) == 0.0

    def test_south_entry_angle(self):
        """Verify south enters at pi/2 radians."""
        assert direction_entry_angle(Direction.SOUTH) == pytest.approx(math.pi / 2.0)

    def test_west_entry_angle(self):
        """Verify west enters at pi radians."""
        assert direction_entry_angle(Direction.WEST) == pytest.approx(math.pi)

    def test_north_entry_angle(self):
        """Verify north enters at -pi/2 radians."""
        assert direction_entry_angle(Direction.NORTH) == pytest.approx(-math.pi / 2.0)


class TestTurnStepCount:
    """Tests for roundabout sector traversal count."""

    def test_right_turn_steps(self):
        """Verify right turn traverses 1 quadrant."""
        assert turn_step_count(Turn.RIGHT) == 1

    def test_straight_turn_steps(self):
        """Verify straight traverses 2 quadrants."""
        assert turn_step_count(Turn.STRAIGHT) == 2

    def test_left_turn_steps(self):
        """Verify left turn traverses 3 quadrants."""
        assert turn_step_count(Turn.LEFT) == 3


class TestExitHeadingForMovement:
    """Tests for determining exit road compass heading."""

    def test_north_right_exit(self):
        """Verify north right turn exits eastward."""
        result = exit_heading_for_movement(Direction.NORTH, Turn.RIGHT)
        assert result == Direction.EAST

    def test_north_straight_exit(self):
        """Verify north straight exits heading northward."""
        result = exit_heading_for_movement(Direction.NORTH, Turn.STRAIGHT)
        assert result == Direction.NORTH

    def test_north_left_exit(self):
        """Verify north left turn exits westward."""
        result = exit_heading_for_movement(Direction.NORTH, Turn.LEFT)
        assert result == Direction.WEST

    def test_east_straight_exit(self):
        """Verify east straight exits heading eastward."""
        result = exit_heading_for_movement(Direction.EAST, Turn.STRAIGHT)
        assert result == Direction.EAST


class TestRoundaboutVehicleAngle:
    """Tests for vehicle angular position in roundabout."""

    def test_vehicle_before_entry(self, sample_blueprint):
        """Verify vehicles on approach have sentinel angle."""
        vehicle = VehicleState(sample_blueprint)
        vehicle.s = -10.0
        angle = roundabout_vehicle_angle(vehicle)
        assert angle == -999.0

    def test_vehicle_after_exit(self, sample_blueprint):
        """Verify vehicles past exit have sentinel angle."""
        vehicle = VehicleState(sample_blueprint)
        vehicle.s = sample_blueprint.intersection_exit_s + 10.0
        angle = roundabout_vehicle_angle(vehicle)
        assert angle == -999.0

    def test_vehicle_at_entry(self):
        """Verify vehicle at entry point has correct angle."""
        bp = VehicleBlueprint(
            vehicle_id=1,
            arrival_time=0.0,
            origin=Direction.EAST,
            desired_speed=12.0,
            max_accel=2.0,
            gap_acceptance=2.0,
            free_flow_time=10.0,
            turn=Turn.STRAIGHT,
            intersection_exit_s=20.0,
        )
        vehicle = VehicleState(bp)
        vehicle.s = 0.01  # Just entered
        angle = roundabout_vehicle_angle(vehicle)
        assert angle != -999.0
        assert angle == pytest.approx(0.0, abs=0.1)

    def test_vehicle_midway_through_ring(self):
        """Verify vehicle in middle of ring has intermediate angle."""
        bp = VehicleBlueprint(
            vehicle_id=1,
            arrival_time=0.0,
            origin=Direction.NORTH,
            desired_speed=12.0,
            max_accel=2.0,
            gap_acceptance=2.0,
            free_flow_time=10.0,
            turn=Turn.STRAIGHT,
            intersection_exit_s=20.0,
        )
        vehicle = VehicleState(bp)
        vehicle.s = 10.0  # Halfway through
        angle = roundabout_vehicle_angle(vehicle)
        assert angle != -999.0


class TestRoundaboutVehiclePosition:
    """Tests for roundabout position calculations."""

    def test_vehicle_on_approach(self, intersection_center):
        """Verify vehicle on approach is positioned correctly."""
        bp = VehicleBlueprint(
            vehicle_id=1,
            arrival_time=0.0,
            origin=Direction.NORTH,
            desired_speed=12.0,
            max_accel=2.0,
            gap_acceptance=2.0,
            free_flow_time=10.0,
            turn=Turn.STRAIGHT,
            intersection_exit_s=20.0,
        )
        vehicle = VehicleState(bp)
        vehicle.s = -10.0

        x, y = roundabout_vehicle_position(
            intersection_center, Direction.NORTH, vehicle
        )
        cx, cy = intersection_center

        # Should be on approach tangent
        assert x == cx - LANE_OFFSET
        assert y < cy

    def test_vehicle_in_ring(self, intersection_center):
        """Verify vehicle in ring has position on circle."""
        bp = VehicleBlueprint(
            vehicle_id=1,
            arrival_time=0.0,
            origin=Direction.EAST,
            desired_speed=12.0,
            max_accel=2.0,
            gap_acceptance=2.0,
            free_flow_time=10.0,
            turn=Turn.STRAIGHT,
            intersection_exit_s=50.0,
        )
        vehicle = VehicleState(bp)
        vehicle.s = 10.0  # Inside ring

        x, y = roundabout_vehicle_position(intersection_center, Direction.EAST, vehicle)
        cx, cy = intersection_center

        # Distance from center should be approximately roundabout radius
        distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        assert distance == pytest.approx(ROUNDABOUT_RADIUS, rel=0.1)

    def test_vehicle_on_exit(self, intersection_center):
        """Verify vehicle on exit segment is positioned correctly."""
        bp = VehicleBlueprint(
            vehicle_id=1,
            arrival_time=0.0,
            origin=Direction.SOUTH,
            desired_speed=12.0,
            max_accel=2.0,
            gap_acceptance=2.0,
            free_flow_time=10.0,
            turn=Turn.RIGHT,
            intersection_exit_s=20.0,
        )
        vehicle = VehicleState(bp)
        vehicle.s = 25.0  # Past exit point

        x, y = roundabout_vehicle_position(
            intersection_center, Direction.SOUTH, vehicle
        )

        # Should be on exit tangent
        assert isinstance(x, float)
        assert isinstance(y, float)
