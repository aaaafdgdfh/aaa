"""
ALNS Destroy Operators for 2E-LRP-MC

This module implements the Destroy (Removal) operators for the
Adaptive Large Neighborhood Search algorithm.

Destroy Operators:
    1. RandomRemoval - Remove random customers
    2. WorstCostRemoval - Remove customers with highest cost contribution
    3. ClusterRemoval - Remove a cluster of nearby customers
    4. WasteTypeRemoval - Remove all customers of a specific waste type
    5. StationClosureRemoval - Remove all customers from a low-utilization station

Author: HG-DRL-ALNS Project
"""

from abc import ABC, abstractmethod
from typing import List, Set, Tuple, Optional, Dict
import numpy as np
from copy import deepcopy

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_structures import ProblemInstance, Customer, Station, BucketVehicle
from core.solution import Solution, Route


# =============================================================================
# Base Class
# =============================================================================

class DestroyOperator(ABC):
    """
    Abstract base class for Destroy operators.
    
    A destroy operator removes a subset of customers from the solution,
    returning them for later reinsertion by a Repair operator.
    """
    
    def __init__(self, instance: ProblemInstance, name: str = "BaseDestroy"):
        """
        Args:
            instance: The problem instance
            name: Operator name for logging
        """
        self.instance = instance
        self.name = name
    
    @abstractmethod
    def destroy(
        self,
        solution: Solution,
        num_to_remove: int
    ) -> Tuple[Solution, List[int]]:
        """
        Remove customers from the solution.
        
        Args:
            solution: Current solution (will be modified in place or copied)
            num_to_remove: Number of customers to remove
        
        Returns:
            Tuple of (modified_solution, list_of_removed_customer_ids)
        """
        pass
    
    def _get_served_customers(self, solution: Solution) -> List[int]:
        """Get list of all served (assigned) customers."""
        served = []
        for route in solution.routes.values():
            served.extend(route.customers)
        return served


# =============================================================================
# Random Removal
# =============================================================================

class RandomRemoval(DestroyOperator):
    """
    Randomly remove customers from the solution.
    
    Simple baseline operator, useful for diversification.
    """
    
    def __init__(self, instance: ProblemInstance):
        super().__init__(instance, name="RandomRemoval")
    
    def destroy(
        self,
        solution: Solution,
        num_to_remove: int
    ) -> Tuple[Solution, List[int]]:
        """Randomly select and remove customers."""
        served = self._get_served_customers(solution)
        
        if len(served) == 0:
            return solution, []
        
        num_to_remove = min(num_to_remove, len(served))
        
        # Random selection
        removed_ids = list(np.random.choice(served, size=num_to_remove, replace=False))
        
        # Remove from solution
        for cust_id in removed_ids:
            solution.remove_customer(cust_id)
        
        return solution, removed_ids


# =============================================================================
# Worst Cost Removal
# =============================================================================

class WorstCostRemoval(DestroyOperator):
    """
    Remove customers that contribute most to the solution cost.
    
    The "cost contribution" of a customer is the cost savings
    if that customer were removed.
    
    Uses randomization factor to avoid deterministic behavior.
    """
    
    def __init__(
        self,
        instance: ProblemInstance,
        randomization: float = 0.1
    ):
        """
        Args:
            instance: Problem instance
            randomization: Randomization factor [0, 1]. Higher = more random.
        """
        super().__init__(instance, name="WorstCostRemoval")
        self.randomization = randomization
    
    def destroy(
        self,
        solution: Solution,
        num_to_remove: int
    ) -> Tuple[Solution, List[int]]:
        """Remove customers with highest cost contribution."""
        removed_ids = []
        
        for _ in range(num_to_remove):
            # Calculate removal cost for each customer
            customer_costs = []
            
            for vehicle_id, route in solution.routes.items():
                if route.is_empty:
                    continue
                
                vehicle = self.instance.bucket_vehicles[vehicle_id]
                
                for i, cust_id in enumerate(route.customers):
                    # Calculate cost reduction if this customer is removed
                    cost_reduction = self._calculate_removal_savings(
                        solution, vehicle, route, i
                    )
                    customer_costs.append((cust_id, cost_reduction))
            
            if not customer_costs:
                break
            
            # Sort by cost reduction (descending)
            customer_costs.sort(key=lambda x: x[1], reverse=True)
            
            # Apply randomization
            # Select among top candidates with probability based on rank
            num_candidates = max(1, int(len(customer_costs) * 0.3))
            candidates = customer_costs[:num_candidates]
            
            # Weighted random selection (higher cost = higher probability)
            if self.randomization > 0:
                weights = np.array([c[1] for c in candidates])
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    idx = np.random.choice(len(candidates), p=weights)
                else:
                    idx = 0
            else:
                idx = 0
            
            selected_id = candidates[idx][0]
            removed_ids.append(selected_id)
            solution.remove_customer(selected_id)
        
        return solution, removed_ids
    
    def _calculate_removal_savings(
        self,
        solution: Solution,
        vehicle: BucketVehicle,
        route: Route,
        position: int
    ) -> float:
        """Calculate cost savings from removing customer at position."""
        customers = route.customers
        cust_id = customers[position]
        station_id = route.station_id
        
        # Current routing cost contribution
        if len(customers) == 1:
            # Only customer in route - removing saves entire route
            station = self.instance.get_station(station_id)
            dist = 2 * self.instance.customer_station_distance(cust_id, station_id)
            return dist * vehicle.variable_cost_per_unit + vehicle.fixed_cost
        
        # Calculate distance savings
        if position == 0:
            # First customer
            next_cust = customers[1]
            old_dist = (
                self.instance.customer_station_distance(cust_id, station_id) +
                self.instance.customer_distance(cust_id, next_cust)
            )
            new_dist = self.instance.customer_station_distance(next_cust, station_id)
        elif position == len(customers) - 1:
            # Last customer
            prev_cust = customers[-2]
            old_dist = (
                self.instance.customer_distance(prev_cust, cust_id) +
                self.instance.customer_station_distance(cust_id, station_id)
            )
            new_dist = self.instance.customer_station_distance(prev_cust, station_id)
        else:
            # Middle customer
            prev_cust = customers[position - 1]
            next_cust = customers[position + 1]
            old_dist = (
                self.instance.customer_distance(prev_cust, cust_id) +
                self.instance.customer_distance(cust_id, next_cust)
            )
            new_dist = self.instance.customer_distance(prev_cust, next_cust)
        
        return (old_dist - new_dist) * vehicle.variable_cost_per_unit


# =============================================================================
# Cluster Removal
# =============================================================================

class ClusterRemoval(DestroyOperator):
    """
    Remove a cluster of geographically close customers.
    
    Strategy:
        1. Select a random seed customer
        2. Remove the seed and its k nearest neighbors
    
    Effective for restructuring routes in specific areas.
    """
    
    def __init__(self, instance: ProblemInstance):
        super().__init__(instance, name="ClusterRemoval")
    
    def destroy(
        self,
        solution: Solution,
        num_to_remove: int
    ) -> Tuple[Solution, List[int]]:
        """Remove a cluster of nearby customers."""
        served = self._get_served_customers(solution)
        
        if len(served) == 0:
            return solution, []
        
        num_to_remove = min(num_to_remove, len(served))
        
        # Select random seed
        seed_id = np.random.choice(served)
        seed_customer = self.instance.get_customer(seed_id)
        
        # Calculate distances to all other served customers
        distances = []
        for cust_id in served:
            if cust_id != seed_id:
                dist = self.instance.customer_distance(seed_id, cust_id)
                distances.append((cust_id, dist))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        # Select seed + nearest neighbors
        removed_ids = [seed_id]
        for cust_id, _ in distances[:num_to_remove - 1]:
            removed_ids.append(cust_id)
        
        # Remove from solution
        for cust_id in removed_ids:
            solution.remove_customer(cust_id)
        
        return solution, removed_ids


# =============================================================================
# Waste Type Specific Removal
# =============================================================================

class WasteTypeRemoval(DestroyOperator):
    """
    Remove customers to optimize a specific waste type's compartment allocation.
    
    Strategy:
        1. Select a random waste type
        2. Find vehicles with low compartment utilization for that type
        3. Remove customers from those compartments
    
    Specifically designed for multi-compartment problems.
    """
    
    def __init__(
        self,
        instance: ProblemInstance,
        utilization_threshold: float = 0.5
    ):
        """
        Args:
            instance: Problem instance
            utilization_threshold: Remove from compartments below this utilization
        """
        super().__init__(instance, name="WasteTypeRemoval")
        self.utilization_threshold = utilization_threshold
    
    def destroy(
        self,
        solution: Solution,
        num_to_remove: int
    ) -> Tuple[Solution, List[int]]:
        """Remove customers based on waste type compartment utilization."""
        # Select random waste type
        waste_type = np.random.randint(0, self.instance.num_waste_types)
        
        # Find customers to remove
        candidates = []
        
        for vehicle_id, route in solution.routes.items():
            if route.is_empty:
                continue
            
            vehicle = self.instance.bucket_vehicles[vehicle_id]
            
            # Check compartment utilization for this waste type
            for comp in vehicle.compartments:
                if comp.assigned_waste_type == waste_type:
                    # Check if low utilization
                    if comp.utilization < self.utilization_threshold:
                        # Add all customers that contribute to this compartment
                        for cust_id in route.customers:
                            customer = self.instance.get_customer(cust_id)
                            if customer.demand_for_type(waste_type) > 0:
                                candidates.append(cust_id)
        
        # Remove duplicates
        candidates = list(set(candidates))
        
        if len(candidates) == 0:
            # Fallback to random removal
            return RandomRemoval(self.instance).destroy(solution, num_to_remove)
        
        num_to_remove = min(num_to_remove, len(candidates))
        removed_ids = list(np.random.choice(candidates, size=num_to_remove, replace=False))
        
        for cust_id in removed_ids:
            solution.remove_customer(cust_id)
        
        return solution, removed_ids


# =============================================================================
# Station Closure Removal
# =============================================================================

class StationClosureRemoval(DestroyOperator):
    """
    Close a low-utilization station and remove all its customers.
    
    Strategy:
        1. Find the station with lowest utilization
        2. Remove all routes based at that station
        3. Mark station as closed
    
    Forces restructuring of station assignments.
    """
    
    def __init__(self, instance: ProblemInstance):
        super().__init__(instance, name="StationClosureRemoval")
    
    def destroy(
        self,
        solution: Solution,
        num_to_remove: int
    ) -> Tuple[Solution, List[int]]:
        """Close a station and remove its customers."""
        if len(solution.open_stations) <= 1:
            # Can't close the only station - fall back to random
            return RandomRemoval(self.instance).destroy(solution, num_to_remove)
        
        # Calculate station utilization
        station_utils = []
        for station_id in solution.open_stations:
            station = self.instance.get_station(station_id)
            total_load = sum(solution.station_loads[station_id].values())
            utilization = total_load / station.capacity if station.capacity > 0 else 0
            station_utils.append((station_id, utilization))
        
        # Sort by utilization (ascending)
        station_utils.sort(key=lambda x: x[1])
        
        # Select station to close (with some randomization)
        num_candidates = max(1, len(station_utils) // 2)
        candidates = station_utils[:num_candidates]
        selected_station_id = candidates[np.random.randint(len(candidates))][0]
        
        # Find all routes at this station
        removed_ids = []
        routes_to_remove = []
        
        for vehicle_id, route in solution.routes.items():
            if route.station_id == selected_station_id:
                removed_ids.extend(route.customers)
                routes_to_remove.append(vehicle_id)
        
        # Remove routes and customers
        for vehicle_id in routes_to_remove:
            route = solution.routes[vehicle_id]
            for cust_id in route.customers.copy():
                solution.remove_customer(cust_id)
            del solution.routes[vehicle_id]
        
        # Close station
        solution.open_stations.discard(selected_station_id)
        if selected_station_id in solution.station_waste_types:
            del solution.station_waste_types[selected_station_id]
        
        # Limit to requested number
        if len(removed_ids) > num_to_remove:
            # Put back some customers
            to_restore = removed_ids[num_to_remove:]
            removed_ids = removed_ids[:num_to_remove]
            solution.unassigned_customers.update(to_restore)
        
        return solution, removed_ids


# =============================================================================
# Related Removal (Shaw Removal)
# =============================================================================

class RelatedRemoval(DestroyOperator):
    """
    Remove related customers based on multiple similarity measures.
    
    Similarity factors:
        1. Geographic distance
        2. Demand similarity
        3. Same route
        4. Same waste type profile
    
    Also known as Shaw Removal in the literature.
    """
    
    def __init__(
        self,
        instance: ProblemInstance,
        distance_weight: float = 0.4,
        demand_weight: float = 0.2,
        route_weight: float = 0.2,
        waste_profile_weight: float = 0.2,
        randomization: float = 0.1
    ):
        super().__init__(instance, name="RelatedRemoval")
        self.distance_weight = distance_weight
        self.demand_weight = demand_weight
        self.route_weight = route_weight
        self.waste_profile_weight = waste_profile_weight
        self.randomization = randomization
        
        # Precompute max distance for normalization
        self._max_distance = self._compute_max_distance()
        self._max_demand = self._compute_max_demand()
    
    def _compute_max_distance(self) -> float:
        """Compute maximum distance for normalization."""
        max_dist = 0
        for c1 in self.instance.customers:
            for c2 in self.instance.customers:
                if c1.id != c2.id:
                    dist = self.instance.customer_distance(c1.id, c2.id)
                    max_dist = max(max_dist, dist)
        return max_dist if max_dist > 0 else 1.0
    
    def _compute_max_demand(self) -> float:
        """Compute maximum demand for normalization."""
        return max(c.total_demand for c in self.instance.customers) or 1.0
    
    def destroy(
        self,
        solution: Solution,
        num_to_remove: int
    ) -> Tuple[Solution, List[int]]:
        """Remove related customers."""
        served = self._get_served_customers(solution)
        
        if len(served) == 0:
            return solution, []
        
        num_to_remove = min(num_to_remove, len(served))
        
        # Build customer-to-route mapping
        customer_route_map = {}
        for vehicle_id, route in solution.routes.items():
            for cust_id in route.customers:
                customer_route_map[cust_id] = vehicle_id
        
        # Select random seed
        seed_id = np.random.choice(served)
        removed_ids = [seed_id]
        solution.remove_customer(seed_id)
        
        # Iteratively add most related customers
        while len(removed_ids) < num_to_remove:
            remaining = [c for c in self._get_served_customers(solution)]
            if not remaining:
                break
            
            # Calculate relatedness to already removed customers
            relatedness_scores = []
            for cust_id in remaining:
                # Average relatedness to all removed customers
                total_score = 0
                for removed_id in removed_ids:
                    total_score += self._relatedness(
                        cust_id, removed_id, customer_route_map
                    )
                avg_score = total_score / len(removed_ids)
                relatedness_scores.append((cust_id, avg_score))
            
            # Sort by relatedness (higher = more related)
            relatedness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Apply randomization
            num_candidates = max(1, int(len(relatedness_scores) * 0.3))
            candidates = relatedness_scores[:num_candidates]
            
            if self.randomization > 0:
                weights = np.array([c[1] for c in candidates])
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    idx = np.random.choice(len(candidates), p=weights)
                else:
                    idx = 0
            else:
                idx = 0
            
            selected_id = candidates[idx][0]
            removed_ids.append(selected_id)
            solution.remove_customer(selected_id)
        
        return solution, removed_ids
    
    def _relatedness(
        self,
        cust1_id: int,
        cust2_id: int,
        customer_route_map: Dict[int, int]
    ) -> float:
        """Calculate relatedness between two customers."""
        c1 = self.instance.get_customer(cust1_id)
        c2 = self.instance.get_customer(cust2_id)
        
        # Distance similarity (inverse - closer = more related)
        dist = self.instance.customer_distance(cust1_id, cust2_id)
        dist_sim = 1 - (dist / self._max_distance)
        
        # Demand similarity
        demand_diff = abs(c1.total_demand - c2.total_demand)
        demand_sim = 1 - (demand_diff / self._max_demand)
        
        # Same route bonus
        route_sim = 1.0 if customer_route_map.get(cust1_id) == customer_route_map.get(cust2_id) else 0.0
        
        # Waste profile similarity
        waste_sim = self._waste_profile_similarity(c1, c2)
        
        # Weighted sum
        return (
            self.distance_weight * dist_sim +
            self.demand_weight * demand_sim +
            self.route_weight * route_sim +
            self.waste_profile_weight * waste_sim
        )
    
    def _waste_profile_similarity(self, c1: Customer, c2: Customer) -> float:
        """Calculate similarity of waste type profiles."""
        # Cosine similarity of demand vectors
        d1 = np.array([c1.demand_for_type(w) for w in range(self.instance.num_waste_types)])
        d2 = np.array([c2.demand_for_type(w) for w in range(self.instance.num_waste_types)])
        
        norm1 = np.linalg.norm(d1)
        norm2 = np.linalg.norm(d2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(d1, d2) / (norm1 * norm2)


# =============================================================================
# Operator Factory
# =============================================================================

class DestroyOperatorFactory:
    """Factory for creating destroy operators."""
    
    OPERATORS = {
        'random': RandomRemoval,
        'worst': WorstCostRemoval,
        'cluster': ClusterRemoval,
        'waste_type': WasteTypeRemoval,
        'station_closure': StationClosureRemoval,
        'related': RelatedRemoval
    }
    
    @classmethod
    def create(cls, name: str, instance: ProblemInstance, **kwargs) -> DestroyOperator:
        """Create a destroy operator by name."""
        if name not in cls.OPERATORS:
            raise ValueError(f"Unknown destroy operator: {name}. "
                           f"Available: {list(cls.OPERATORS.keys())}")
        return cls.OPERATORS[name](instance, **kwargs)
    
    @classmethod
    def create_all(cls, instance: ProblemInstance) -> List[DestroyOperator]:
        """Create all available destroy operators."""
        return [
            RandomRemoval(instance),
            WorstCostRemoval(instance),
            ClusterRemoval(instance),
            WasteTypeRemoval(instance),
            StationClosureRemoval(instance),
            RelatedRemoval(instance)
        ]


if __name__ == "__main__":
    from core import create_random_instance, SolutionBuilder
    
    print("Testing Destroy Operators...")
    
    # Create instance and initial solution
    instance = create_random_instance(num_customers=30, seed=42)
    builder = SolutionBuilder(instance)
    solution = builder.build_greedy_solution()
    
    print(f"Initial solution: {solution}")
    print(f"Served customers: {len(instance.customers) - len(solution.unassigned_customers)}")
    
    # Test each operator
    operators = DestroyOperatorFactory.create_all(instance)
    
    for op in operators:
        # Work on a copy
        sol_copy = solution.copy()
        
        # Destroy
        modified_sol, removed = op.destroy(sol_copy, num_to_remove=5)
        
        print(f"\n{op.name}:")
        print(f"  Removed: {len(removed)} customers")
        print(f"  Remaining served: {len(instance.customers) - len(modified_sol.unassigned_customers)}")
    
    print("\n✅ Destroy operators test passed!")
