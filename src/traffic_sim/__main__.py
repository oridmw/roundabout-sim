"""Command-line interface and main event loop for traffic intersection simulation."""

import random

import click
import pygame

from traffic_sim.config import (
    BACKGROUND_COLOR,
    FPS,
    TEXT_COLOR,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from traffic_sim.core import Direction
from traffic_sim.intersections import (
    FixedTimeSignalizedIntersection,
    FourWayStopIntersection,
    RoundaboutPriorityCirculating,
    RoundaboutPriorityEntering,
)
from traffic_sim.sim import IntersectionSimulation


@click.command()
@click.option(
    "--rate-north",
    default=800.0,
    show_default=True,
    help="Arrival rate from north (veh/hr)",
)
@click.option(
    "--rate-south",
    default=800.0,
    show_default=True,
    help="Arrival rate from south (veh/hr)",
)
@click.option(
    "--rate-east",
    default=800.0,
    show_default=True,
    help="Arrival rate from east (veh/hr)",
)
@click.option(
    "--rate-west",
    default=800.0,
    show_default=True,
    help="Arrival rate from west (veh/hr)",
)
@click.option(
    "--sim-speed", default=2.0, show_default=True, help="Simulation speed multiplier"
)
@click.option(
    "--seed", default=0, show_default=True, help="Random seed for reproducibility"
)
@click.option(
    "--cycle-length",
    default=60.0,
    show_default=True,
    help="Signal cycle length (seconds)",
)
@click.option(
    "--green-ns",
    default=30.0,
    show_default=True,
    help="Green time for north-south (seconds)",
)
@click.option(
    "--green-ew",
    default=30.0,
    show_default=True,
    help="Green time for east-west (seconds)",
)
def main(
    rate_north: float,
    rate_south: float,
    rate_east: float,
    rate_west: float,
    sim_speed: float,
    seed: int,
    cycle_length: float,
    green_ns: float,
    green_ew: float,
) -> None:
    """Run interactive GUI comparing four intersection control strategies.

    Simulates identical traffic demand across four intersection types:
    - Roundabout with circulating priority (standard)
    - Roundabout with entering priority (non-standard)
    - Fixed-time traffic signal
    - All-way stop control

    Press SPACE to pause/resume. Close window to exit.
    """
    rng = random.Random(seed)

    # Convert hourly rates to per-second rates
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

    # Position four intersections in 2x2 grid
    centers = [
        (WINDOW_WIDTH // 4, WINDOW_HEIGHT // 4),
        (3 * WINDOW_WIDTH // 4, WINDOW_HEIGHT // 4),
        (WINDOW_WIDTH // 4, 3 * WINDOW_HEIGHT // 4),
        (3 * WINDOW_WIDTH // 4, 3 * WINDOW_HEIGHT // 4),
    ]

    models = [
        RoundaboutPriorityCirculating(center=centers[0]),
        RoundaboutPriorityEntering(center=centers[1]),
        FixedTimeSignalizedIntersection(
            center=centers[2],
            cycle_length=cycle_length,
            green_ns=green_ns,
            green_ew=green_ew,
        ),
        FourWayStopIntersection(center=centers[3]),
    ]

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
