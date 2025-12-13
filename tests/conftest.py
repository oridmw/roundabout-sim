"""Shared pytest fixtures for traffic simulation tests."""

import random

import pytest

from traffic_sim.core import Direction, Turn, VehicleBlueprint


@pytest.fixture
def rng():
    """Provide a seeded random number generator for reproducible tests."""
    return random.Random(42)


@pytest.fixture
def sample_blueprint():
    """Provide a standard vehicle blueprint for testing."""
    return VehicleBlueprint(
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


@pytest.fixture
def intersection_center():
    """Provide standard intersection center coordinates."""
    return (640, 400)


@pytest.fixture
def arrival_rates():
    """Provide balanced arrival rates for all directions."""
    return {
        Direction.NORTH: 0.3,
        Direction.SOUTH: 0.3,
        Direction.EAST: 0.3,
        Direction.WEST: 0.3,
    }
