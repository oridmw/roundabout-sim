"""Simulation engine managing multiple intersection models."""

import math
import random
from typing import List

from traffic_sim.config import (
    BASE_INTERSECTION_S,
    DEFAULT_DESIRED_SPEED,
    DEFAULT_DESIRED_SPEED_STD,
    DEFAULT_GAP_ACCEPT_MEAN,
    DEFAULT_GAP_ACCEPT_STD,
    DEFAULT_MAX_ACCEL,
    EXIT_S,
    MAX_QUEUE_PER_DIRECTION,
    ROUNDABOUT_RADIUS,
    SCALE,
    SPAWN_S,
)
from traffic_sim.core import (
    DIRECTIONS,
    Direction,
    IntersectionModel,
    Turn,
    VehicleBlueprint,
)
from traffic_sim.intersections import (
    RoundaboutPriorityCirculating,
    RoundaboutPriorityEntering,
)


class IntersectionSimulation:
    """Manages all intersection models and shared vehicle generation."""

    def __init__(
        self, models: List[IntersectionModel], arrival_rates: dict, rng: random.Random
    ):
        self.models = models
        self.arrival_rates = arrival_rates
        self.rng = rng
        self.sim_time = 0.0
        self.next_vehicle_id = 0

    def update(self, dt: float) -> None:
        self.sim_time += dt

        for direction in DIRECTIONS:
            lam = self.arrival_rates[direction]
            if lam <= 0.0:
                continue
            p = lam * dt
            if self.rng.random() < p:
                self._spawn_vehicle(direction)

        for model in self.models:
            model.step(dt, self.sim_time)

    def _sample_turn(self) -> Turn:
        r = self.rng.random()
        if r < 1.0 / 3.0:
            return Turn.RIGHT
        elif r < 2.0 / 3.0:
            return Turn.STRAIGHT
        else:
            return Turn.LEFT

    def _intersection_exit_s_for_turn(self, turn: Turn, use_roundabout: bool) -> float:
        if use_roundabout:
            radius_meters = ROUNDABOUT_RADIUS / SCALE
            if turn == Turn.RIGHT:
                return radius_meters * (math.pi / 2.0)
            elif turn == Turn.STRAIGHT:
                return radius_meters * math.pi
            else:
                return radius_meters * (3.0 * math.pi / 2.0)
        else:
            if turn == Turn.RIGHT:
                return BASE_INTERSECTION_S
            elif turn == Turn.STRAIGHT:
                return 2.0 * BASE_INTERSECTION_S
            else:
                return 3.0 * BASE_INTERSECTION_S

    def _spawn_vehicle(self, direction: Direction) -> None:
        arrival_time = self.sim_time
        desired_speed = max(
            self.rng.gauss(DEFAULT_DESIRED_SPEED, DEFAULT_DESIRED_SPEED_STD), 3.0
        )
        max_accel = max(self.rng.gauss(DEFAULT_MAX_ACCEL, 0.5), 0.5)
        gap_accept = max(
            self.rng.gauss(DEFAULT_GAP_ACCEPT_MEAN, DEFAULT_GAP_ACCEPT_STD), 0.1
        )
        turn = self._sample_turn()

        path_length = EXIT_S - SPAWN_S
        free_flow_time = path_length / desired_speed

        for model in self.models:
            use_roundabout = isinstance(
                model, (RoundaboutPriorityCirculating, RoundaboutPriorityEntering)
            )
            intersection_exit_s = self._intersection_exit_s_for_turn(
                turn, use_roundabout
            )

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

            if len(model.lanes[direction]) < MAX_QUEUE_PER_DIRECTION:
                model.spawn_vehicle(bp)

        self.next_vehicle_id += 1
