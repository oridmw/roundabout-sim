import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm 

# Import simulation packages
from traffic_sim.core import Direction
from traffic_sim.intersections import (
    RoundaboutPriorityCirculating,
    RoundaboutPriorityEntering,
    FixedTimeSignalizedIntersection,
    FourWayStopIntersection
)
from traffic_sim.sim import IntersectionSimulation

# Configuration
SEEDS = 20                # Number of distinct runs per configuration (Monte Carlo)
SIM_DURATION = 3600.0     # Simulation duration in seconds
DT = 0.1                  # Time step
WARMUP = 600.0            # Seconds to discard at start

# Define traffic levels to test (Total incoming veh/hr across all 4 directions)
# We assume symmetric traffic (divide by 4 for per-arm rate)
TRAFFIC_LEVELS = [
    1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400
]

def run_headless_episode(rate_total, seed):
    """
    Runs one simulation episode without the GUI.
    Returns a dictionary of metrics for each model.
    """
    rng = random.Random(seed)
    
    # Symmetric arrival rates
    rate_per_dir = rate_total / 4.0
    arrival_rates = {d: rate_per_dir / 3600.0 for d in Direction}

    # Initialize Models
    models = [
        RoundaboutPriorityCirculating((0,0)),
        RoundaboutPriorityEntering((0,0)),
        FixedTimeSignalizedIntersection((0,0), cycle_length=60, green_ns=30, green_ew=30),
        FourWayStopIntersection((0,0))
    ]
    
    # Initialize Simulation
    sim = IntersectionSimulation(models, arrival_rates, rng)
    
    # Run Loop
    t = 0.0
    while t < SIM_DURATION:
        sim.update(DT)
        t += DT

    # Collect Metrics
    results = []
    for model in models:
        
        # Basic metrics
        mean_delay = model.metrics.mean_delay
        p95_delay = model.metrics.p95_delay
        throughput = model.metrics.departed_count
        max_q = model.metrics.max_queue
        
        results.append({
            "Traffic_Vol": rate_total,
            "Seed": seed,
            "Model": model.name,
            "Mean_Delay": mean_delay,
            "P95_Delay": p95_delay,
            "Throughput": throughput,
            "Max_Queue": max_q
        })
        
    return results

def main():
    print(f"Starting Validation Protocol...")
    print(f"Seeds: {SEEDS} | Traffic Levels: {len(TRAFFIC_LEVELS)} | Duration: {SIM_DURATION}s")
    
    all_data = []
    
    # Use itertools or nested loops
    total_runs = len(TRAFFIC_LEVELS) * SEEDS
    
    with tqdm(total=total_runs) as pbar:
        for vol in TRAFFIC_LEVELS:
            for seed in range(SEEDS):
                # Run the simulation
                data = run_headless_episode(vol, seed)
                all_data.extend(data)
                pbar.update(1)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Save raw data
    df.to_csv("validation_results.csv", index=False)
    print("Simulation complete. Data saved to validation_results.csv")
    
    return df

def plot_results(df):
    """
    Generates validation plots with uncertainty bands (Confidence Intervals).
    """
    sns.set_theme(style="whitegrid")
    
    # 1. Mean Delay vs Traffic Volume
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df, 
        x="Traffic_Vol", 
        y="Mean_Delay", 
        hue="Model", 
        style="Model",
        markers=True, 
        dashes=False,
        errorbar=("ci", 95)  # 95% Confidence Interval
    )
    plt.title("Impact of Traffic Volume on Intersection Delay (95% CI)")
    plt.ylabel("Mean Delay (s)")
    plt.xlabel("Total Traffic Volume (veh/hr)")
    plt.axhline(y=60, color='r', linestyle='--', alpha=0.3, label="Threshold (1 min)")
    plt.legend()
    plt.savefig("validation_delay.png", dpi=300)
    plt.show()

    # 2. P95 Delay (Reliability)
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df, 
        x="Traffic_Vol", 
        y="P95_Delay", 
        hue="Model", 
        errorbar=("ci", 95)
    )
    plt.title("Intersection Reliability (95th Percentile Delay)")
    plt.ylabel("P95 Delay (s)")
    plt.xlabel("Total Traffic Volume (veh/hr)")
    plt.savefig("validation_p95.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    df = main()
    plot_results(df)