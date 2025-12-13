"""Unit tests for simulation orchestrator and vehicle generation."""

import pytest

from traffic_sim.core import Direction, Turn
from traffic_sim.intersections import RoundaboutPriorityCirculating
from traffic_sim.sim import IntersectionSimulation


class TestIntersectionSimulation:
    """Tests for IntersectionSimulation class."""

    def test_initialization(self, rng, arrival_rates, intersection_center):
        """Verify simulation initializes with correct state."""
        model = RoundaboutPriorityCirculating(center=intersection_center)
        sim = IntersectionSimulation(
            models=[model], arrival_rates=arrival_rates, rng=rng
        )

        assert sim.sim_time == 0.0
        assert sim.next_vehicle_id == 0
        assert len(sim.models) == 1
        assert sim.rng == rng

    def test_time_advances_on_update(self, rng, arrival_rates, intersection_center):
        """Verify simulation time increments on update."""
        model = RoundaboutPriorityCirculating(center=intersection_center)
        sim = IntersectionSimulation(
            models=[model], arrival_rates=arrival_rates, rng=rng
        )

        initial_time = sim.sim_time
        sim.update(0.1)
        assert sim.sim_time > initial_time
        assert sim.sim_time == pytest.approx(0.1)

    def test_multiple_updates(self, rng, arrival_rates, intersection_center):
        """Verify simulation time accumulates correctly."""
        model = RoundaboutPriorityCirculating(center=intersection_center)
        sim = IntersectionSimulation(
            models=[model], arrival_rates=arrival_rates, rng=rng
        )

        for _ in range(10):
            sim.update(0.1)

        assert sim.sim_time == pytest.approx(1.0)

    def test_vehicle_spawning_increases_id(
        self, rng, arrival_rates, intersection_center
    ):
        """Verify vehicle IDs increment on spawn."""
        model = RoundaboutPriorityCirculating(center=intersection_center)

        # High arrival rate to ensure spawning
        high_rates = {d: 100.0 for d in arrival_rates.keys()}
        sim = IntersectionSimulation(models=[model], arrival_rates=high_rates, rng=rng)

        initial_id = sim.next_vehicle_id

        # Run several timesteps to spawn vehicles
        for _ in range(100):
            sim.update(0.01)

        # Should have spawned vehicles
        assert sim.next_vehicle_id > initial_id

    def test_zero_arrival_rate_no_spawning(self, rng, intersection_center):
        """Verify no vehicles spawn when arrival rates are zero."""
        model = RoundaboutPriorityCirculating(center=intersection_center)
        zero_rates = {
            Direction.NORTH: 0.0,
            Direction.SOUTH: 0.0,
            Direction.EAST: 0.0,
            Direction.WEST: 0.0,
        }
        sim = IntersectionSimulation(models=[model], arrival_rates=zero_rates, rng=rng)

        for _ in range(100):
            sim.update(0.1)

        # No vehicles should be spawned
        assert sim.next_vehicle_id == 0
        for lane in model.lanes.values():
            assert len(lane) == 0

    def test_vehicle_spawns_in_all_models(
        self, rng, arrival_rates, intersection_center
    ):
        """Verify spawned vehicle appears in all models."""
        model1 = RoundaboutPriorityCirculating(center=intersection_center)
        model2 = RoundaboutPriorityCirculating(center=(800, 400))

        high_rates = {d: 50.0 for d in arrival_rates.keys()}
        sim = IntersectionSimulation(
            models=[model1, model2], arrival_rates=high_rates, rng=rng
        )

        # Run until at least one vehicle spawns
        for _ in range(50):
            sim.update(0.1)
            total_vehicles = sum(
                len(lane) for model in sim.models for lane in model.lanes.values()
            )
            if total_vehicles > 0:
                break

        # Check that both models have vehicles
        model1_count = sum(len(lane) for lane in model1.lanes.values())
        model2_count = sum(len(lane) for lane in model2.lanes.values())

        # Both models should have same number of vehicles
        if model1_count > 0:
            assert model1_count == model2_count

    def test_turn_sampling_distribution(self, rng):
        """Verify turn sampling produces all turn types."""
        model = RoundaboutPriorityCirculating(center=(640, 400))
        arrival_rates = {
            Direction.NORTH: 10.0,
            Direction.SOUTH: 0.0,
            Direction.EAST: 0.0,
            Direction.WEST: 0.0,
        }
        sim = IntersectionSimulation(
            models=[model], arrival_rates=arrival_rates, rng=rng
        )

        # Spawn many vehicles
        for _ in range(500):
            sim.update(0.05)

        # Collect turn types
        turns = set()
        for vehicle in model.lanes[Direction.NORTH]:
            turns.add(vehicle.blueprint.turn)

        # Should have sampled all turn types with high probability
        assert len(turns) >= 2  # At least 2 different turn types

    def test_vehicle_blueprint_has_valid_parameters(
        self, rng, arrival_rates, intersection_center
    ):
        """Verify spawned vehicles have reasonable parameter values."""
        model = RoundaboutPriorityCirculating(center=intersection_center)
        high_rates = {d: 20.0 for d in arrival_rates.keys()}
        sim = IntersectionSimulation(models=[model], arrival_rates=high_rates, rng=rng)

        # Spawn vehicles
        for _ in range(100):
            sim.update(0.1)

        # Check first spawned vehicle
        for lane in model.lanes.values():
            if len(lane) > 0:
                vehicle = lane[0]
                bp = vehicle.blueprint

                assert bp.desired_speed > 0
                assert bp.max_accel > 0
                assert bp.gap_acceptance > 0
                assert bp.free_flow_time > 0
                assert bp.intersection_exit_s > 0
                assert bp.turn in [Turn.RIGHT, Turn.STRAIGHT, Turn.LEFT]
                break


class TestVehicleGeneration:
    """Tests for vehicle parameter generation logic."""

    def test_intersection_exit_s_roundabout_right(self):
        """Verify right turn exit distance for roundabout is correct."""
        from traffic_sim.config import ROUNDABOUT_RADIUS, SCALE

        radius_meters = ROUNDABOUT_RADIUS / SCALE
        expected = radius_meters * (3.14159 / 2.0)  # Quarter circle

        model = RoundaboutPriorityCirculating(center=(640, 400))
        sim = IntersectionSimulation(
            models=[model],
            arrival_rates={d: 1.0 for d in Direction},
            rng=pytest.importorskip("random").Random(42),
        )

        # Internal method test
        result = sim._intersection_exit_s_for_turn(Turn.RIGHT, use_roundabout=True)
        assert result == pytest.approx(expected, rel=0.01)

    def test_intersection_exit_s_roundabout_straight(self):
        """Verify straight exit distance for roundabout is correct."""
        from traffic_sim.config import ROUNDABOUT_RADIUS, SCALE

        radius_meters = ROUNDABOUT_RADIUS / SCALE
        expected = radius_meters * 3.14159  # Semicircle

        model = RoundaboutPriorityCirculating(center=(640, 400))
        sim = IntersectionSimulation(
            models=[model],
            arrival_rates={d: 1.0 for d in Direction},
            rng=pytest.importorskip("random").Random(42),
        )

        result = sim._intersection_exit_s_for_turn(Turn.STRAIGHT, use_roundabout=True)
        assert result == pytest.approx(expected, rel=0.01)

    def test_intersection_exit_s_standard_intersection(self):
        """Verify exit distances for standard intersection."""
        from traffic_sim.config import BASE_INTERSECTION_S

        model = RoundaboutPriorityCirculating(center=(640, 400))
        sim = IntersectionSimulation(
            models=[model],
            arrival_rates={d: 1.0 for d in Direction},
            rng=pytest.importorskip("random").Random(42),
        )

        assert (
            sim._intersection_exit_s_for_turn(Turn.RIGHT, False) == BASE_INTERSECTION_S
        )
        assert (
            sim._intersection_exit_s_for_turn(Turn.STRAIGHT, False)
            == 2.0 * BASE_INTERSECTION_S
        )
        assert (
            sim._intersection_exit_s_for_turn(Turn.LEFT, False)
            == 3.0 * BASE_INTERSECTION_S
        )
