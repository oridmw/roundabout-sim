"""Unit tests for core data structures and base intersection model."""

from traffic_sim.core import (
    DIRECTIONS,
    Direction,
    IntersectionMetrics,
    Turn,
    VehicleBlueprint,
    VehicleState,
)


class TestDirection:
    """Tests for Direction enum."""

    def test_all_directions_present(self):
        """Verify all cardinal directions are defined."""
        assert Direction.NORTH in DIRECTIONS
        assert Direction.SOUTH in DIRECTIONS
        assert Direction.EAST in DIRECTIONS
        assert Direction.WEST in DIRECTIONS
        assert len(DIRECTIONS) == 4

    def test_directions_are_unique(self):
        """Verify no duplicate directions."""
        assert len(set(DIRECTIONS)) == len(DIRECTIONS)


class TestTurn:
    """Tests for Turn enum."""

    def test_turn_types_exist(self):
        """Verify all turn types are defined."""
        assert Turn.RIGHT
        assert Turn.STRAIGHT
        assert Turn.LEFT

    def test_turns_are_distinct(self):
        """Verify turn types are distinct values."""
        assert Turn.RIGHT != Turn.STRAIGHT
        assert Turn.STRAIGHT != Turn.LEFT
        assert Turn.LEFT != Turn.RIGHT


class TestVehicleBlueprint:
    """Tests for VehicleBlueprint immutable parameters."""

    def test_blueprint_initialization(self, sample_blueprint):
        """Verify blueprint initializes with correct attributes."""
        assert sample_blueprint.id == 1
        assert sample_blueprint.arrival_time == 0.0
        assert sample_blueprint.origin == Direction.NORTH
        assert sample_blueprint.desired_speed == 12.0
        assert sample_blueprint.max_accel == 2.0
        assert sample_blueprint.gap_acceptance == 2.0
        assert sample_blueprint.free_flow_time == 10.0
        assert sample_blueprint.turn == Turn.STRAIGHT
        assert sample_blueprint.intersection_exit_s == 24.0

    def test_blueprint_with_different_turns(self):
        """Verify blueprints can be created with different turn types."""
        for turn in [Turn.RIGHT, Turn.STRAIGHT, Turn.LEFT]:
            bp = VehicleBlueprint(
                vehicle_id=1,
                arrival_time=0.0,
                origin=Direction.SOUTH,
                desired_speed=10.0,
                max_accel=2.0,
                gap_acceptance=2.0,
                free_flow_time=10.0,
                turn=turn,
                intersection_exit_s=20.0,
            )
            assert bp.turn == turn


class TestVehicleState:
    """Tests for VehicleState mutable dynamics."""

    def test_initial_state(self, sample_blueprint):
        """Verify vehicle state initializes correctly from blueprint."""
        state = VehicleState(sample_blueprint)
        assert state.blueprint == sample_blueprint
        assert state.v == sample_blueprint.desired_speed
        assert state.entered is False
        assert state.exit_time is None

    def test_state_mutability(self, sample_blueprint):
        """Verify state properties can be modified."""
        state = VehicleState(sample_blueprint)

        state.s = 10.0
        assert state.s == 10.0

        state.v = 15.0
        assert state.v == 15.0

        state.entered = True
        assert state.entered is True

        state.exit_time = 5.0
        assert state.exit_time == 5.0

    def test_blueprint_remains_unchanged(self, sample_blueprint):
        """Verify blueprint is not modified when state changes."""
        original_speed = sample_blueprint.desired_speed
        state = VehicleState(sample_blueprint)

        state.v = 20.0
        assert sample_blueprint.desired_speed == original_speed


class TestIntersectionMetrics:
    """Tests for IntersectionMetrics statistics tracking."""

    def test_initial_metrics(self):
        """Verify metrics initialize to zero."""
        metrics = IntersectionMetrics()
        assert metrics.departed_count == 0
        assert metrics.total_delay == 0.0
        assert len(metrics.delays) == 0
        assert metrics.max_queue == 0

    def test_mean_delay_with_no_departures(self):
        """Verify mean delay is zero when no vehicles departed."""
        metrics = IntersectionMetrics()
        assert metrics.mean_delay == 0.0

    def test_mean_delay_calculation(self):
        """Verify mean delay computes correctly."""
        metrics = IntersectionMetrics()
        metrics.departed_count = 3
        metrics.total_delay = 15.0
        assert metrics.mean_delay == 5.0

    def test_p95_delay_with_no_data(self):
        """Verify p95 delay is zero with no data."""
        metrics = IntersectionMetrics()
        assert metrics.p95_delay == 0.0

    def test_p95_delay_calculation(self):
        """Verify 95th percentile delay computes correctly."""
        metrics = IntersectionMetrics()
        metrics.delays = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        # 95th percentile of 10 values: int(0.95 * 9) = 8, so index 8 = 9.0
        assert metrics.p95_delay == 9.0

    def test_p95_delay_with_single_value(self):
        """Verify p95 works with single delay value."""
        metrics = IntersectionMetrics()
        metrics.delays = [5.0]
        assert metrics.p95_delay == 5.0

    def test_metrics_accumulation(self):
        """Verify metrics can accumulate over time."""
        metrics = IntersectionMetrics()

        metrics.departed_count += 1
        metrics.total_delay += 3.5
        metrics.delays.append(3.5)

        metrics.departed_count += 1
        metrics.total_delay += 2.5
        metrics.delays.append(2.5)

        assert metrics.departed_count == 2
        assert metrics.total_delay == 6.0
        assert metrics.mean_delay == 3.0
        assert len(metrics.delays) == 2

    def test_max_queue_tracking(self):
        """Verify max queue updates correctly."""
        metrics = IntersectionMetrics()

        metrics.max_queue = 5
        assert metrics.max_queue == 5

        metrics.max_queue = max(metrics.max_queue, 10)
        assert metrics.max_queue == 10

        # Should not decrease
        new_queue = 3
        metrics.max_queue = max(metrics.max_queue, new_queue)
        assert metrics.max_queue == 10
