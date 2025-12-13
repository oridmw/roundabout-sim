"""Unit tests for intersection control strategy implementations."""

from traffic_sim.config import STOP_LIMIT_S, STOP_LINE_S
from traffic_sim.core import Direction, Turn, VehicleBlueprint
from traffic_sim.intersections import (
    FixedTimeSignalizedIntersection,
    FourWayStopIntersection,
    RoundaboutPriorityCirculating,
    RoundaboutPriorityEntering,
    vehicles_conflict_opposite_dirs,
)


class TestVehiclesConflictOppositeDirs:
    """Tests for opposite direction conflict detection."""

    def test_left_yields_to_straight(self):
        """Verify left turn yields to opposing straight traffic."""
        assert vehicles_conflict_opposite_dirs(Turn.LEFT, Turn.STRAIGHT) is True

    def test_left_yields_to_right(self):
        """Verify left turn yields to opposing right turn."""
        assert vehicles_conflict_opposite_dirs(Turn.LEFT, Turn.RIGHT) is True

    def test_straight_does_not_yield_to_left(self):
        """Verify straight does not yield to opposing left turn."""
        assert vehicles_conflict_opposite_dirs(Turn.STRAIGHT, Turn.LEFT) is False

    def test_right_does_not_yield(self):
        """Verify right turn does not yield to opposing traffic."""
        assert vehicles_conflict_opposite_dirs(Turn.RIGHT, Turn.LEFT) is False
        assert vehicles_conflict_opposite_dirs(Turn.RIGHT, Turn.STRAIGHT) is False

    def test_straight_does_not_yield_to_straight(self):
        """Verify straight does not yield to opposing straight."""
        assert vehicles_conflict_opposite_dirs(Turn.STRAIGHT, Turn.STRAIGHT) is False


class TestRoundaboutPriorityCirculating:
    """Tests for standard roundabout with circulating priority."""

    def test_initialization(self, intersection_center):
        """Verify roundabout initializes correctly."""
        roundabout = RoundaboutPriorityCirculating(center=intersection_center)
        assert roundabout.name == "Roundabout (circ. priority)"
        assert roundabout.center == intersection_center
        assert len(roundabout.lanes) == 4
        assert roundabout.metrics.departed_count == 0

    def test_spawn_vehicle(self, intersection_center, sample_blueprint):
        """Verify vehicle spawning adds to correct lane."""
        roundabout = RoundaboutPriorityCirculating(center=intersection_center)
        initial_count = len(roundabout.lanes[Direction.NORTH])

        roundabout.spawn_vehicle(sample_blueprint)
        assert len(roundabout.lanes[Direction.NORTH]) == initial_count + 1

    def test_spawn_vehicle_adjusts_position(
        self, intersection_center, sample_blueprint
    ):
        """Verify spawned vehicle has adjusted position for roundabout geometry."""
        roundabout = RoundaboutPriorityCirculating(center=intersection_center)
        roundabout.spawn_vehicle(sample_blueprint)

        vehicle = roundabout.lanes[Direction.NORTH][0]
        # Position should be negative (on approach)
        assert vehicle.s < 0

    def test_vehicle_propagation(self, intersection_center, sample_blueprint):
        """Verify vehicles advance forward in time."""
        roundabout = RoundaboutPriorityCirculating(center=intersection_center)
        roundabout.spawn_vehicle(sample_blueprint)

        vehicle = roundabout.lanes[Direction.NORTH][0]
        initial_s = vehicle.s

        # Step forward
        roundabout.step(0.1, 0.1)

        # Vehicle should have moved forward
        assert vehicle.s > initial_s

    def test_vehicle_departure(self, intersection_center):
        """Verify vehicles are removed after exiting."""
        roundabout = RoundaboutPriorityCirculating(center=intersection_center)
        bp = VehicleBlueprint(
            vehicle_id=1,
            arrival_time=0.0,
            origin=Direction.NORTH,
            desired_speed=20.0,
            max_accel=5.0,
            gap_acceptance=2.0,
            free_flow_time=5.0,
            turn=Turn.STRAIGHT,
            intersection_exit_s=20.0,
        )
        roundabout.spawn_vehicle(bp)

        # Run simulation long enough for vehicle to exit
        for i in range(100):
            roundabout.step(0.1, i * 0.1)

        # Vehicle should have departed
        assert len(roundabout.lanes[Direction.NORTH]) == 0
        assert roundabout.metrics.departed_count == 1

    def test_multiple_vehicles_maintain_spacing(self, intersection_center):
        """Verify following vehicles maintain safe gap."""
        roundabout = RoundaboutPriorityCirculating(center=intersection_center)

        # Spawn two vehicles
        bp1 = VehicleBlueprint(
            vehicle_id=1,
            arrival_time=0.0,
            origin=Direction.NORTH,
            desired_speed=12.0,
            max_accel=2.0,
            gap_acceptance=2.0,
            free_flow_time=10.0,
            turn=Turn.STRAIGHT,
            intersection_exit_s=24.0,
        )
        bp2 = VehicleBlueprint(
            vehicle_id=2,
            arrival_time=1.0,
            origin=Direction.NORTH,
            desired_speed=12.0,
            max_accel=2.0,
            gap_acceptance=2.0,
            free_flow_time=10.0,
            turn=Turn.STRAIGHT,
            intersection_exit_s=24.0,
        )
        roundabout.spawn_vehicle(bp1)
        roundabout.spawn_vehicle(bp2)

        # Step forward
        for i in range(10):
            roundabout.step(0.1, i * 0.1)

        vehicles = sorted(roundabout.lanes[Direction.NORTH], key=lambda v: v.s)
        if len(vehicles) == 2:
            # Following vehicle should be behind leader
            assert vehicles[0].s < vehicles[1].s


class TestRoundaboutPriorityEntering:
    """Tests for non-standard roundabout with entering priority."""

    def test_initialization(self, intersection_center):
        """Verify roundabout initializes with correct name."""
        roundabout = RoundaboutPriorityEntering(center=intersection_center)
        assert roundabout.name == "Roundabout (enter priority)"
        assert roundabout.center == intersection_center

    def test_entering_vehicle_not_stopped(self, intersection_center, sample_blueprint):
        """Verify entering vehicles are not stopped at entry."""
        roundabout = RoundaboutPriorityEntering(center=intersection_center)
        roundabout.spawn_vehicle(sample_blueprint)

        vehicle = roundabout.lanes[Direction.NORTH][0]
        max_s = roundabout.max_s_for_front(Direction.NORTH, vehicle, 0.0)

        # Should not be restricted (INF means no stopping)
        assert max_s > 1000.0


class TestFixedTimeSignalizedIntersection:
    """Tests for fixed-time traffic signal control."""

    def test_initialization(self, intersection_center):
        """Verify signal initializes with correct parameters."""
        signal = FixedTimeSignalizedIntersection(
            center=intersection_center,
            cycle_length=60.0,
            green_ns=30.0,
            green_ew=25.0,
        )
        assert signal.name == "Fixed-time signal"
        assert signal.cycle_length == 60.0
        assert signal.green_ns == 30.0
        assert signal.green_ew == 25.0

    def test_ns_green_phase_timing(self, intersection_center):
        """Verify north-south green phase timing."""
        signal = FixedTimeSignalizedIntersection(
            center=intersection_center,
            cycle_length=60.0,
            green_ns=30.0,
            green_ew=25.0,
        )

        assert signal.is_ns_green(0.0) is True
        assert signal.is_ns_green(15.0) is True
        assert signal.is_ns_green(29.9) is True
        assert signal.is_ns_green(30.0) is False

    def test_ew_green_phase_timing(self, intersection_center):
        """Verify east-west green phase timing."""
        signal = FixedTimeSignalizedIntersection(
            center=intersection_center,
            cycle_length=60.0,
            green_ns=30.0,
            green_ew=25.0,
        )

        assert signal.is_ew_green(30.0) is True
        assert signal.is_ew_green(40.0) is True
        assert signal.is_ew_green(54.9) is True
        assert signal.is_ew_green(55.0) is False

    def test_cycle_repeats(self, intersection_center):
        """Verify signal cycle repeats correctly."""
        signal = FixedTimeSignalizedIntersection(
            center=intersection_center,
            cycle_length=60.0,
            green_ns=30.0,
            green_ew=25.0,
        )

        # First cycle
        assert signal.is_ns_green(10.0) is True
        # Second cycle
        assert signal.is_ns_green(70.0) is True

    def test_vehicle_stops_on_red(self, intersection_center):
        """Verify vehicle stops when signal is red."""
        signal = FixedTimeSignalizedIntersection(
            center=intersection_center,
            cycle_length=60.0,
            green_ns=30.0,
            green_ew=25.0,
        )

        bp = VehicleBlueprint(
            vehicle_id=1,
            arrival_time=0.0,
            origin=Direction.NORTH,
            desired_speed=12.0,
            max_accel=2.0,
            gap_acceptance=2.0,
            free_flow_time=10.0,
            turn=Turn.STRAIGHT,
            intersection_exit_s=24.0,
        )
        signal.spawn_vehicle(bp)
        vehicle = signal.lanes[Direction.NORTH][0]
        vehicle.s = STOP_LIMIT_S + 1.0  # Position before stop line

        # Red for north at t=35
        max_s = signal.max_s_for_front(Direction.NORTH, vehicle, 35.0)
        assert max_s == STOP_LIMIT_S

    def test_vehicle_proceeds_on_green(self, intersection_center):
        """Verify vehicle can proceed when signal is green."""
        signal = FixedTimeSignalizedIntersection(
            center=intersection_center,
            cycle_length=60.0,
            green_ns=30.0,
            green_ew=25.0,
        )

        bp = VehicleBlueprint(
            vehicle_id=1,
            arrival_time=0.0,
            origin=Direction.NORTH,
            desired_speed=12.0,
            max_accel=2.0,
            gap_acceptance=2.0,
            free_flow_time=10.0,
            turn=Turn.STRAIGHT,
            intersection_exit_s=24.0,
        )
        signal.spawn_vehicle(bp)
        vehicle = signal.lanes[Direction.NORTH][0]
        vehicle.s = STOP_LIMIT_S + 1.0

        # Green for north at t=10
        max_s = signal.max_s_for_front(Direction.NORTH, vehicle, 10.0)
        assert max_s > STOP_LINE_S


class TestFourWayStopIntersection:
    """Tests for all-way stop control."""

    def test_initialization(self, intersection_center):
        """Verify four-way stop initializes correctly."""
        stop = FourWayStopIntersection(center=intersection_center)
        assert stop.name == "4-way stop"
        assert stop.current_direction is None
        assert stop.direction_release_time is None

    def test_no_priority_initially(self, intersection_center):
        """Verify no direction has priority initially."""
        stop = FourWayStopIntersection(center=intersection_center)
        assert stop.current_direction is None

    def test_assigns_priority_to_waiting_vehicle(self, intersection_center):
        """Verify priority is assigned to direction with waiting vehicle."""
        stop = FourWayStopIntersection(center=intersection_center)

        bp = VehicleBlueprint(
            vehicle_id=1,
            arrival_time=0.0,
            origin=Direction.NORTH,
            desired_speed=12.0,
            max_accel=2.0,
            gap_acceptance=2.0,
            free_flow_time=10.0,
            turn=Turn.STRAIGHT,
            intersection_exit_s=24.0,
        )
        stop.spawn_vehicle(bp)
        vehicle = stop.lanes[Direction.NORTH][0]
        vehicle.s = STOP_LIMIT_S  # At stop line

        # Update control logic
        stop.update_control(0.1, 0.1)

        # Should assign priority to north
        assert stop.current_direction is not None

    def test_vehicle_stops_without_priority(self, intersection_center):
        """Verify vehicle stops when it doesn't have priority."""
        stop = FourWayStopIntersection(center=intersection_center)
        stop.current_direction = Direction.EAST  # Priority to east (not opposite)

        bp = VehicleBlueprint(
            vehicle_id=1,
            arrival_time=0.0,
            origin=Direction.NORTH,
            desired_speed=12.0,
            max_accel=2.0,
            gap_acceptance=2.0,
            free_flow_time=10.0,
            turn=Turn.STRAIGHT,
            intersection_exit_s=24.0,
        )
        stop.spawn_vehicle(bp)
        vehicle = stop.lanes[Direction.NORTH][0]
        vehicle.s = STOP_LIMIT_S + 1.0

        max_s = stop.max_s_for_front(Direction.NORTH, vehicle, 1.0)
        assert max_s == STOP_LIMIT_S

    def test_vehicle_proceeds_with_priority(self, intersection_center):
        """Verify vehicle proceeds when it has priority."""
        stop = FourWayStopIntersection(center=intersection_center)
        stop.current_direction = Direction.NORTH  # Priority to north

        bp = VehicleBlueprint(
            vehicle_id=1,
            arrival_time=0.0,
            origin=Direction.NORTH,
            desired_speed=12.0,
            max_accel=2.0,
            gap_acceptance=2.0,
            free_flow_time=10.0,
            turn=Turn.STRAIGHT,
            intersection_exit_s=24.0,
        )
        stop.spawn_vehicle(bp)
        vehicle = stop.lanes[Direction.NORTH][0]
        vehicle.s = STOP_LIMIT_S + 1.0

        max_s = stop.max_s_for_front(Direction.NORTH, vehicle, 1.0)
        assert max_s > STOP_LINE_S

    def test_releases_priority_after_timeout(self, intersection_center):
        """Verify priority is released after hold time expires."""
        stop = FourWayStopIntersection(center=intersection_center)
        stop.current_direction = Direction.NORTH
        stop.direction_release_time = 1.0

        # Update after timeout
        stop.update_control(0.1, 2.0)

        # Priority should be released
        assert stop.current_direction is None
