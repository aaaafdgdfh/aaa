"""
Visualization Utilities for HG-DRL-ALNS

This module provides visualization tools for:
    - Solution routes and assignments
    - Training convergence curves
    - Operator usage statistics
    - Compartment utilization

Author: HG-DRL-ALNS Project
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visualization disabled.")

from core import ProblemInstance, Solution


# =============================================================================
# Color Schemes
# =============================================================================

# Professional color palette
COLORS = {
    'customer': '#3498DB',      # Blue
    'station': '#E74C3C',       # Red
    'plant': '#2ECC71',         # Green
    'route': [
        '#9B59B6', '#F39C12', '#1ABC9C', '#E91E63',
        '#00BCD4', '#FF5722', '#795548', '#607D8B'
    ],
    'waste_types': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
    'background': '#FAFAFA',
    'grid': '#EEEEEE'
}


# =============================================================================
# Solution Visualization
# =============================================================================

def plot_solution(
    solution: Solution,
    instance: ProblemInstance,
    title: str = "2E-LRP-MC Solution",
    figsize: Tuple[int, int] = (14, 10),
    show_legend: bool = True,
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Visualize a complete solution with routes and assignments.
    
    Args:
        solution: The solution to visualize
        instance: Problem instance
        title: Plot title
        figsize: Figure size
        show_legend: Whether to show legend
        save_path: Path to save figure
        show: Whether to display figure
    
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Cannot visualize.")
        return None
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_facecolor(COLORS['background'])
    
    # Plot customers
    for customer in instance.customers:
        if customer.id in solution.unassigned_customers:
            # Unassigned - hollow marker
            ax.scatter(customer.x, customer.y, s=60, c='white',
                      edgecolors=COLORS['customer'], linewidth=2,
                      marker='o', zorder=3)
        else:
            # Assigned - filled marker
            ax.scatter(customer.x, customer.y, s=60, c=COLORS['customer'],
                      edgecolors='white', linewidth=1,
                      marker='o', zorder=3)
    
    # Plot stations
    for station in instance.stations:
        if station.id in solution.open_stations:
            # Open station - filled square
            ax.scatter(station.x, station.y, s=200, c=COLORS['station'],
                      edgecolors='white', linewidth=2,
                      marker='s', zorder=4)
            ax.annotate(f'S{station.id}', (station.x, station.y),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold')
        else:
            # Closed station - hollow square
            ax.scatter(station.x, station.y, s=150, c='white',
                      edgecolors=COLORS['station'], linewidth=2,
                      marker='s', zorder=4, alpha=0.5)
    
    # Plot plants
    for plant in instance.plants:
        ax.scatter(plant.x, plant.y, s=300, c=COLORS['plant'],
                  edgecolors='white', linewidth=2,
                  marker='^', zorder=4)
        ax.annotate(f'P{plant.id}', (plant.x, plant.y),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, fontweight='bold')
    
    # Plot routes
    route_idx = 0
    for vehicle_id, route in solution.routes.items():
        if route.is_empty:
            continue
        
        color = COLORS['route'][route_idx % len(COLORS['route'])]
        station = instance.get_station(route.station_id)
        
        # Draw route
        points = [(station.x, station.y)]
        for cust_id in route.customers:
            customer = instance.get_customer(cust_id)
            points.append((customer.x, customer.y))
        points.append((station.x, station.y))
        
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.7, zorder=2)
        
        # Draw arrows
        for i in range(len(points) - 1):
            dx = points[i+1][0] - points[i][0]
            dy = points[i+1][1] - points[i][1]
            ax.annotate('', xy=(points[i][0] + dx*0.6, points[i][1] + dy*0.6),
                       xytext=(points[i][0] + dx*0.4, points[i][1] + dy*0.4),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
                       zorder=2)
        
        route_idx += 1
    
    # Plot second echelon (station to plant connections)
    for assignment in solution.second_echelon:
        station = instance.get_station(assignment.station_id)
        plant = instance.get_plant(assignment.plant_id)
        
        ax.plot([station.x, plant.x], [station.y, plant.y],
               color=COLORS['plant'], linewidth=2, linestyle='--',
               alpha=0.5, zorder=1)
    
    # Title and labels
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('X Coordinate', fontsize=10)
    ax.set_ylabel('Y Coordinate', fontsize=10)
    
    # Legend
    if show_legend:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['customer'],
                  markersize=10, label='Customer (Served)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                  markeredgecolor=COLORS['customer'], markersize=10,
                  label='Customer (Unassigned)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['station'],
                  markersize=12, label='Station (Open)'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['plant'],
                  markersize=12, label='Plant'),
            Line2D([0], [0], color=COLORS['route'][0], linewidth=2,
                  label='Vehicle Route'),
            Line2D([0], [0], color=COLORS['plant'], linewidth=2, linestyle='--',
                  label='2nd Echelon')
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                 fontsize=9, framealpha=0.9)
    
    # Add statistics text box
    stats = solution.get_statistics()
    stats_text = (f"Total Cost: {stats['total_cost']:.2f}\n"
                  f"Routes: {stats['num_routes']}\n"
                  f"Open Stations: {stats['num_open_stations']}\n"
                  f"Compartment Util: {stats['compartment_utilization']:.1%}")
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=props)
    
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_compartment_utilization(
    solution: Solution,
    instance: ProblemInstance,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Visualize compartment utilization for each vehicle.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    vehicle_data = []
    
    for vehicle_id, route in solution.routes.items():
        if route.is_empty:
            continue
        
        vehicle = instance.bucket_vehicles[vehicle_id]
        
        # Calculate load per waste type
        loads = {wt: 0.0 for wt in range(instance.num_waste_types)}
        for cust_id in route.customers:
            customer = instance.get_customer(cust_id)
            for wt, demand in customer.demands.items():
                loads[wt] += demand
        
        vehicle_data.append({
            'id': vehicle_id,
            'capacity': vehicle.total_capacity,
            'loads': loads
        })
    
    if not vehicle_data:
        print("No active vehicles to plot")
        return None
    
    # Create stacked bar chart
    x = np.arange(len(vehicle_data))
    width = 0.6
    
    bottoms = np.zeros(len(vehicle_data))
    
    for wt in range(instance.num_waste_types):
        values = [v['loads'].get(wt, 0) for v in vehicle_data]
        color = COLORS['waste_types'][wt % len(COLORS['waste_types'])]
        ax.bar(x, values, width, bottom=bottoms, label=f'Waste Type {wt}',
              color=color, edgecolor='white', linewidth=1)
        bottoms += values
    
    # Add capacity line
    capacities = [v['capacity'] for v in vehicle_data]
    ax.scatter(x, capacities, color='red', marker='_', s=500, linewidths=3,
              zorder=5, label='Capacity')
    
    # Labels
    ax.set_xlabel('Vehicle ID', fontsize=10)
    ax.set_ylabel('Load', fontsize=10)
    ax.set_title('Compartment Utilization by Vehicle', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'V{v["id"]}' for v in vehicle_data])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# Training Visualization
# =============================================================================

def plot_training_curves(
    cost_history: List[float],
    reward_history: Optional[List[float]] = None,
    title: str = "Training Progress",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Plot training convergence curves.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    num_plots = 2 if reward_history else 1
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    
    if num_plots == 1:
        axes = [axes]
    
    # Cost history
    ax = axes[0]
    ax.plot(cost_history, color='#3498DB', linewidth=1, alpha=0.7)
    
    # Add smoothed line
    if len(cost_history) > 20:
        window = min(50, len(cost_history) // 10)
        smoothed = np.convolve(cost_history, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(cost_history)), smoothed,
               color='#E74C3C', linewidth=2, label='Smoothed')
        ax.legend()
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Cost')
    ax.set_title('Cost Convergence')
    ax.grid(True, alpha=0.3)
    
    # Reward history
    if reward_history:
        ax = axes[1]
        ax.plot(reward_history, color='#2ECC71', linewidth=1, alpha=0.7)
        
        if len(reward_history) > 20:
            window = min(50, len(reward_history) // 10)
            smoothed = np.convolve(reward_history, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(reward_history)), smoothed,
                   color='#9B59B6', linewidth=2, label='Smoothed')
            ax.legend()
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_operator_statistics(
    stats: Dict,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Visualize operator usage and success rates.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Destroy operators
    ax = axes[0]
    destroy_stats = stats.get('destroy_operators', {})
    
    if destroy_stats:
        names = list(destroy_stats.keys())
        usage = [destroy_stats[n]['usage'] for n in names]
        success_rate = [destroy_stats[n]['success_rate'] * 100 for n in names]
        
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, usage, width, label='Usage Count',
                      color='#3498DB', alpha=0.8)
        
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, success_rate, width, label='Success Rate (%)',
                       color='#2ECC71', alpha=0.8)
        
        ax.set_xlabel('Operator')
        ax.set_ylabel('Usage Count', color='#3498DB')
        ax2.set_ylabel('Success Rate (%)', color='#2ECC71')
        ax.set_title('Destroy Operators', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace('Removal', '') for n in names],
                          rotation=45, ha='right')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Repair operators
    ax = axes[1]
    repair_stats = stats.get('repair_operators', {})
    
    if repair_stats:
        names = list(repair_stats.keys())
        usage = [repair_stats[n]['usage'] for n in names]
        success_rate = [repair_stats[n]['success_rate'] * 100 for n in names]
        
        x = np.arange(len(names))
        
        bars1 = ax.bar(x - width/2, usage, width, label='Usage Count',
                      color='#E74C3C', alpha=0.8)
        
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, success_rate, width, label='Success Rate (%)',
                       color='#9B59B6', alpha=0.8)
        
        ax.set_xlabel('Operator')
        ax.set_ylabel('Usage Count', color='#E74C3C')
        ax2.set_ylabel('Success Rate (%)', color='#9B59B6')
        ax.set_title('Repair Operators', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace('Repair', '') for n in names],
                          rotation=45, ha='right')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    fig.suptitle('Operator Performance Statistics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# Instance Visualization
# =============================================================================

def plot_instance(
    instance: ProblemInstance,
    title: str = "Problem Instance",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Visualize problem instance (customers, stations, plants).
    """
    if not HAS_MATPLOTLIB:
        return None
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_facecolor(COLORS['background'])
    
    # Plot customers with demand-based size
    demands = [c.total_demand for c in instance.customers]
    max_demand = max(demands) if demands else 1
    sizes = [30 + 100 * (d / max_demand) for d in demands]
    
    for i, customer in enumerate(instance.customers):
        ax.scatter(customer.x, customer.y, s=sizes[i], c=COLORS['customer'],
                  edgecolors='white', linewidth=1, marker='o', zorder=3, alpha=0.7)
    
    # Plot stations
    for station in instance.stations:
        ax.scatter(station.x, station.y, s=200, c=COLORS['station'],
                  edgecolors='white', linewidth=2, marker='s', zorder=4)
        ax.annotate(f'S{station.id}\n({station.capacity:.0f})',
                   (station.x, station.y), xytext=(5, 5),
                   textcoords='offset points', fontsize=8)
    
    # Plot plants
    for plant in instance.plants:
        ax.scatter(plant.x, plant.y, s=300, c=COLORS['plant'],
                  edgecolors='white', linewidth=2, marker='^', zorder=4)
        ax.annotate(f'P{plant.id}\n({plant.capacity:.0f})',
                   (plant.x, plant.y), xytext=(5, 5),
                   textcoords='offset points', fontsize=8)
    
    # Info box
    info_text = (f"Customers: {instance.num_customers}\n"
                 f"Stations: {instance.num_stations}\n"
                 f"Plants: {instance.num_plants}\n"
                 f"Waste Types: {instance.num_waste_types}\n"
                 f"Total Demand: {instance.get_total_demand():.0f}")
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=props)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True, alpha=0.3)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['customer'],
              markersize=10, label='Customer'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['station'],
              markersize=12, label='Station'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['plant'],
              markersize=12, label='Plant'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# Main Demo
# =============================================================================

if __name__ == "__main__":
    from core import create_random_instance, SolutionBuilder
    from alns import ALNSEngine, ALNSConfig
    
    print("Visualization Demo")
    print("=" * 50)
    
    # Create instance
    instance = create_random_instance(num_customers=30, seed=42)
    print(f"Created instance with {instance.num_customers} customers")
    
    # Plot instance
    print("\n1. Plotting problem instance...")
    plot_instance(instance, title="Demo Instance", show=True)
    
    # Build solution
    builder = SolutionBuilder(instance)
    solution = builder.build_greedy_solution()
    solution.rebuild_second_echelon()
    print(f"\n2. Built initial solution with cost: {solution.compute_cost():.2f}")
    
    # Plot solution
    print("\n3. Plotting solution...")
    plot_solution(solution, instance, title="Initial Greedy Solution", show=True)
    
    # Plot compartment utilization
    print("\n4. Plotting compartment utilization...")
    plot_compartment_utilization(solution, instance, show=True)
    
    # Run ALNS and plot convergence
    print("\n5. Running ALNS...")
    config = ALNSConfig(max_iterations=200, verbose=False)
    engine = ALNSEngine(instance, config)
    result = engine.run()
    
    print(f"   ALNS completed: {result.best_cost:.2f}")
    
    # Plot training curves
    print("\n6. Plotting convergence...")
    plot_training_curves(result.cost_history, title="ALNS Convergence", show=True)
    
    # Plot operator statistics
    print("\n7. Plotting operator statistics...")
    stats = engine.get_operator_statistics()
    plot_operator_statistics(stats, show=True)
    
    print("\n✅ Visualization demo complete!")
