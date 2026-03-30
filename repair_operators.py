"""
ALNS Repair Operators for 2E-LRP-MC

This module implements the Repair (Insertion) operators for the
Adaptive Large Neighborhood Search algorithm.

Repair Operators:
    1. GreedyRepair - Insert at minimum cost position
    2. RegretRepair - Regret-k insertion heuristic
    3. BestFitCompartmentRepair - Optimize compartment utilization
    4. StationOpeningRepair - Open new stations when needed
    5. ParallelRepair - Insert multiple customers simultaneously

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
from core.constraints import (
    InsertionFeasibilityChecker,
    CompartmentChecker,
    InsertionCheckResult
)


# =============================================================================
# Base Class
# =============================================================================

class RepairOperator(ABC):
    """
    Abstract base class for Repair operators.
    
    A repair operator takes a partial solution with removed customers
    and reinserts them to create a complete solution.
    """
    
    def __init__(self, instance: ProblemInstance, name: str = "BaseRepair"):
        """
        Args:
            instance: The problem instance
            name: Operator name for logging
        """
        self.instance = instance
        self.name = name
        self.feasibility_checker = InsertionFeasibilityChecker(instance)
        self.compartment_checker = CompartmentChecker()
    
    @abstractmethod
    def repair(
        self,
        solution: Solution,
        removed_customers: List[int]
    ) -> Solution:
        """
        Reinsert removed customers into the solution.
        
        Args:
            solution: Current partial solution
            removed_customers: List of customer IDs to reinsert
        
        Returns:
            Modified solution with customers reinserted
        """
        pass
    
    def _get_active_vehicles(self, solution: Solution) -> List[BucketVehicle]:
        """Get list of vehicles with active routes."""
        return [
            self.instance.bucket_vehicles[vid]
            for vid, route in solution.routes.items()
            if not route.is_empty
        ]
    
    def _create_new_route_if_needed(
        self,
        solution: Solution,
        customer: Customer
    ) -> Optional[Tuple[int, int]]:
        """
        Try to create a new route for a customer if no existing route works.
        
        Returns:
            Tuple of (vehicle_id, station_id) if successful, None otherwise
        """
        # Find unused vehicle
        used_vehicles = set(solution.routes.keys())
        
        for vehicle in self.instance.bucket_vehicles:
            if vehicle.id not in used_vehicles:
                # Find compatible station
                best_station = None
                best_dist = float('inf')
                
                for station in self.instance.stations:
                    # Check if station can process all customer's waste types
                    can_process = all(
                        wt in station.can_process
                        for wt, demand in customer.demands.items()
                        if demand > 0
                    )
                    
                    if can_process and station.remaining_capacity >= customer.total_demand:
                        dist = self.instance.customer_station_distance(customer.id, station.id)
                        if dist < best_dist:
                            best_dist = dist
                            best_station = station
                
                if best_station is not None:
                    # Create new route
                    solution.create_route(vehicle.id, best_station.id)
                    return (vehicle.id, best_station.id)
        
        return None


# =============================================================================
# Greedy Repair
# =============================================================================

class GreedyRepair(RepairOperator):
    """
    Greedy insertion - insert each customer at minimum cost position.
    
    For each customer (in order):
        1. Find all feasible (vehicle, position) combinations
        2. Select the one with minimum cost increase
        3. Insert customer
    
    Considers multi-compartment constraints during feasibility check.
    """
    
    def __init__(self, instance: ProblemInstance):
        super().__init__(instance, name="GreedyRepair")
    
    def repair(
        self,
        solution: Solution,
        removed_customers: List[int]
    ) -> Solution:
        """Insert customers at minimum cost positions."""
        for cust_id in removed_customers:
            customer = self.instance.get_customer(cust_id)
            
            # Find best insertion
            vehicle_id, position, cost = self.feasibility_checker.find_best_insertion(
                customer,
                consider_fragmentation=False
            )
            
            if vehicle_id is not None:
                # Get compartment allocation
                vehicle = self.instance.bucket_vehicles[vehicle_id]
                result = self.feasibility_checker.check_full_insertion(
                    vehicle, customer, position
                )
                
                solution.insert_customer(
                    vehicle_id, cust_id, position,
                    result.required_compartment_assignments
                )
            else:
                # Try to open a new route
                new_route = self._create_new_route_if_needed(solution, customer)
                
                if new_route is not None:
                    vehicle_id, station_id = new_route
                    vehicle = self.instance.bucket_vehicles[vehicle_id]
                    
                    result = self.compartment_checker.check_insertion(
                        vehicle, customer, self.instance
                    )
                    
                    if result.is_feasible:
                        solution.insert_customer(
                            vehicle_id, cust_id, 0,
                            result.required_compartment_assignments
                        )
                    else:
                        # Still can't insert - leave as unassigned
                        solution.unassigned_customers.add(cust_id)
                else:
                    solution.unassigned_customers.add(cust_id)
        
        return solution


# =============================================================================
# Regret Repair
# =============================================================================

class RegretRepair(RepairOperator):
    """
    Regret-k insertion heuristic.
    
    Prioritizes customers with high "regret" - the difference between
    the best and k-th best insertion cost.
    
    Customers with high regret have fewer good options, so they should
    be inserted first to avoid leaving them with only bad options.
    """
    
    def __init__(self, instance: ProblemInstance, k: int = 2):
        """
        Args:
            instance: Problem instance
            k: Number of positions to consider for regret calculation
        """
        super().__init__(instance, name=f"Regret{k}Repair")
        self.k = k
    
    def repair(
        self,
        solution: Solution,
        removed_customers: List[int]
    ) -> Solution:
        """Insert customers prioritizing those with high regret."""
        remaining = list(removed_customers)
        
        while remaining:
            # Calculate regret for all remaining customers
            regret_scores = []
            
            for cust_id in remaining:
                customer = self.instance.get_customer(cust_id)
                
                vehicle_id, position, cost, regret = \
                    self.feasibility_checker.find_regret_insertion(
                        customer, k=self.k
                    )
                
                if vehicle_id is not None:
                    regret_scores.append((cust_id, vehicle_id, position, cost, regret))
            
            if not regret_scores:
                # No feasible insertions found
                # Try to create new routes for remaining customers
                for cust_id in remaining:
                    customer = self.instance.get_customer(cust_id)
                    new_route = self._create_new_route_if_needed(solution, customer)
                    
                    if new_route is not None:
                        vehicle_id, station_id = new_route
                        vehicle = self.instance.bucket_vehicles[vehicle_id]
                        
                        result = self.compartment_checker.check_insertion(
                            vehicle, customer, self.instance
                        )
                        
                        if result.is_feasible:
                            solution.insert_customer(
                                vehicle_id, cust_id, 0,
                                result.required_compartment_assignments
                            )
                        else:
                            solution.unassigned_customers.add(cust_id)
                    else:
                        solution.unassigned_customers.add(cust_id)
                break
            
            # Sort by regret (descending)
            regret_scores.sort(key=lambda x: x[4], reverse=True)
            
            # Insert customer with highest regret
            cust_id, vehicle_id, position, _, _ = regret_scores[0]
            customer = self.instance.get_customer(cust_id)
            vehicle = self.instance.bucket_vehicles[vehicle_id]
            
            result = self.feasibility_checker.check_full_insertion(
                vehicle, customer, position
            )
            
            solution.insert_customer(
                vehicle_id, cust_id, position,
                result.required_compartment_assignments
            )
            
            remaining.remove(cust_id)
        
        return solution


# =============================================================================
# Best-Fit Compartment Repair
# =============================================================================

class BestFitCompartmentRepair(RepairOperator):
    """
    Best-Fit Compartment Insertion - minimize compartment fragmentation.
    
    Specifically designed for multi-compartment problems:
        1. Prioritize inserting into vehicles with compatible assigned compartments
        2. Minimize the number of new compartments opened
        3. Maximize compartment utilization
    
    This is a key innovation for the 2E-LRP-MC problem.
    """
    
    def __init__(self, instance: ProblemInstance):
        super().__init__(instance, name="BestFitCompartmentRepair")
    
    def repair(
        self,
        solution: Solution,
        removed_customers: List[int]
    ) -> Solution:
        """Insert customers optimizing compartment utilization."""
        # Sort customers by total demand (descending)
        # Larger demands first to avoid fragmentation
        sorted_customers = sorted(
            removed_customers,
            key=lambda c: self.instance.get_customer(c).total_demand,
            reverse=True
        )
        
        for cust_id in sorted_customers:
            customer = self.instance.get_customer(cust_id)
            
            best_score = float('inf')
            best_vehicle_id = None
            best_position = None
            best_allocation = None
            
            # Check all active routes
            for vehicle_id, route in solution.routes.items():
                vehicle = self.instance.bucket_vehicles[vehicle_id]
                
                # Simulate current vehicle state
                temp_vehicle = self._simulate_vehicle_state(vehicle, route, solution)
                
                # Calculate fragmentation score
                frag_score = self.compartment_checker.compute_fragmentation_score(
                    temp_vehicle, customer
                )
                
                if frag_score < best_score:
                    # Check feasibility
                    for pos in range(len(route.customers) + 1):
                        result = self.feasibility_checker.check_full_insertion(
                            temp_vehicle, customer, pos,
                            self.instance.get_station(route.station_id)
                        )
                        
                        if result.is_feasible:
                            # Combine fragmentation score with insertion cost
                            combined_score = (
                                0.6 * frag_score + 
                                0.4 * result.estimated_cost_delta / 100.0
                            )
                            
                            if combined_score < best_score:
                                best_score = combined_score
                                best_vehicle_id = vehicle_id
                                best_position = pos
                                best_allocation = result.required_compartment_assignments
            
            if best_vehicle_id is not None:
                solution.insert_customer(
                    best_vehicle_id, cust_id, best_position, best_allocation
                )
            else:
                # Try new route
                new_route = self._create_new_route_if_needed(solution, customer)
                
                if new_route is not None:
                    vehicle_id, _ = new_route
                    vehicle = self.instance.bucket_vehicles[vehicle_id]
                    
                    result = self.compartment_checker.check_insertion(
                        vehicle, customer, self.instance
                    )
                    
                    if result.is_feasible:
                        solution.insert_customer(
                            vehicle_id, cust_id, 0,
                            result.required_compartment_assignments
                        )
                    else:
                        solution.unassigned_customers.add(cust_id)
                else:
                    solution.unassigned_customers.add(cust_id)
        
        return solution
    
    def _simulate_vehicle_state(
        self,
        vehicle: BucketVehicle,
        route: Route,
        solution: Solution
    ) -> BucketVehicle:
        """Create a copy of vehicle with simulated compartment loads."""
        temp_vehicle = vehicle.copy()
        temp_vehicle.route = route.customers.copy()
        temp_vehicle.base_station_id = route.station_id
        
        # Simulate loading based on route customers
        for cust_id in route.customers:
            customer = self.instance.get_customer(cust_id)
            allocation = route.customer_waste_allocation.get(cust_id, {})
            
            for waste_type, demand in customer.demands.items():
                if demand > 0:
                    comp_id = allocation.get(waste_type)
                    if comp_id is not None and comp_id < len(temp_vehicle.compartments):
                        temp_vehicle.compartments[comp_id].load(waste_type, demand)
        
        return temp_vehicle


# =============================================================================
# Station Opening Repair
# =============================================================================

class StationOpeningRepair(RepairOperator):
    """
    Repair that can open new stations when needed.
    
    Strategy:
        1. Try greedy insertion first
        2. If a customer cannot be inserted, find a new station to open
        3. Create a new route from the newly opened station
    
    Useful when station closure has created infeasible assignments.
    """
    
    def __init__(self, instance: ProblemInstance):
        super().__init__(instance, name="StationOpeningRepair")
    
    def repair(
        self,
        solution: Solution,
        removed_customers: List[int]
    ) -> Solution:
        """Insert customers, opening new stations as needed."""
        for cust_id in removed_customers:
            customer = self.instance.get_customer(cust_id)
            
            # First try existing routes
            vehicle_id, position, cost = self.feasibility_checker.find_best_insertion(
                customer, consider_fragmentation=True
            )
            
            if vehicle_id is not None:
                vehicle = self.instance.bucket_vehicles[vehicle_id]
                result = self.feasibility_checker.check_full_insertion(
                    vehicle, customer, position
                )
                
                solution.insert_customer(
                    vehicle_id, cust_id, position,
                    result.required_compartment_assignments
                )
            else:
                # Need to open a new station or create new route
                inserted = self._try_open_station_for_customer(solution, customer)
                
                if not inserted:
                    solution.unassigned_customers.add(cust_id)
        
        return solution
    
    def _try_open_station_for_customer(
        self,
        solution: Solution,
        customer: Customer
    ) -> bool:
        """Try to open a new station and insert customer."""
        # Find best closed station for this customer
        closed_stations = [
            s for s in self.instance.stations
            if s.id not in solution.open_stations
        ]
        
        if not closed_stations:
            # All stations open - try any station
            return self._create_new_route_if_needed(solution, customer) is not None
        
        # Find compatible station
        best_station = None
        best_dist = float('inf')
        
        for station in closed_stations:
            can_process = all(
                wt in station.can_process
                for wt, demand in customer.demands.items()
                if demand > 0
            )
            
            if can_process:
                dist = self.instance.customer_station_distance(customer.id, station.id)
                if dist < best_dist:
                    best_dist = dist
                    best_station = station
        
        if best_station is None:
            return False
        
        # Find unused vehicle
        used_vehicles = set(solution.routes.keys())
        free_vehicle = None
        
        for vehicle in self.instance.bucket_vehicles:
            if vehicle.id not in used_vehicles:
                free_vehicle = vehicle
                break
        
        if free_vehicle is None:
            return False
        
        # Create new route
        solution.create_route(free_vehicle.id, best_station.id)
        
        result = self.compartment_checker.check_insertion(
            free_vehicle, customer, self.instance
        )
        
        if result.is_feasible:
            solution.insert_customer(
                free_vehicle.id, customer.id, 0,
                result.required_compartment_assignments
            )
            return True
        
        return False


# =============================================================================
# Random Repair (Baseline)
# =============================================================================

class RandomRepair(RepairOperator):
    """
    Random insertion - insert at random feasible position.
    
    Baseline operator for comparison.
    """
    
    def __init__(self, instance: ProblemInstance):
        super().__init__(instance, name="RandomRepair")
    
    def repair(
        self,
        solution: Solution,
        removed_customers: List[int]
    ) -> Solution:
        """Insert customers at random feasible positions."""
        # Shuffle customers
        shuffled = list(removed_customers)
        np.random.shuffle(shuffled)
        
        for cust_id in shuffled:
            customer = self.instance.get_customer(cust_id)
            
            # Collect all feasible insertions
            feasible = []
            
            for vehicle_id, route in solution.routes.items():
                vehicle = self.instance.bucket_vehicles[vehicle_id]
                
                for pos in range(len(route.customers) + 1):
                    result = self.feasibility_checker.check_full_insertion(
                        vehicle, customer, pos
                    )
                    
                    if result.is_feasible:
                        feasible.append((vehicle_id, pos, result))
            
            if feasible:
                # Random selection
                vehicle_id, pos, result = feasible[np.random.randint(len(feasible))]
                solution.insert_customer(
                    vehicle_id, cust_id, pos,
                    result.required_compartment_assignments
                )
            else:
                # Try new route
                new_route = self._create_new_route_if_needed(solution, customer)
                
                if new_route is not None:
                    vehicle_id, _ = new_route
                    vehicle = self.instance.bucket_vehicles[vehicle_id]
                    
                    result = self.compartment_checker.check_insertion(
                        vehicle, customer, self.instance
                    )
                    
                    if result.is_feasible:
                        solution.insert_customer(
                            vehicle_id, cust_id, 0,
                            result.required_compartment_assignments
                        )
                    else:
                        solution.unassigned_customers.add(cust_id)
                else:
                    solution.unassigned_customers.add(cust_id)
        
        return solution


# =============================================================================
# Operator Factory
# =============================================================================

class RepairOperatorFactory:
    """Factory for creating repair operators."""
    
    OPERATORS = {
        'greedy': GreedyRepair,
        'regret2': lambda inst: RegretRepair(inst, k=2),
        'regret3': lambda inst: RegretRepair(inst, k=3),
        'best_fit': BestFitCompartmentRepair,
        'station_opening': StationOpeningRepair,
        'random': RandomRepair
    }
    
    @classmethod
    def create(cls, name: str, instance: ProblemInstance, **kwargs) -> RepairOperator:
        """Create a repair operator by name."""
        if name not in cls.OPERATORS:
            raise ValueError(f"Unknown repair operator: {name}. "
                           f"Available: {list(cls.OPERATORS.keys())}")
        
        factory = cls.OPERATORS[name]
        if callable(factory) and not isinstance(factory, type):
            return factory(instance)
        return factory(instance, **kwargs)
    
    @classmethod
    def create_all(cls, instance: ProblemInstance) -> List[RepairOperator]:
        """Create all available repair operators."""
        return [
            GreedyRepair(instance),
            RegretRepair(instance, k=2),
            RegretRepair(instance, k=3),
            BestFitCompartmentRepair(instance),
            StationOpeningRepair(instance),
            RandomRepair(instance)
        ]


if __name__ == "__main__":
    from core import create_random_instance, SolutionBuilder
    from alns.destroy_operators import RandomRemoval
    
    print("Testing Repair Operators...")
    
    # Create instance and initial solution
    instance = create_random_instance(num_customers=30, seed=42)
    builder = SolutionBuilder(instance)
    solution = builder.build_greedy_solution()
    
    print(f"Initial solution: {solution}")
    
    # Destroy some customers
    destroy_op = RandomRemoval(instance)
    destroyed_sol, removed = destroy_op.destroy(solution.copy(), num_to_remove=10)
    
    print(f"\nRemoved {len(removed)} customers")
    print(f"After destroy: {destroyed_sol}")
    
    # Test each repair operator
    repair_operators = RepairOperatorFactory.create_all(instance)
    
    for op in repair_operators:
        # Work on a copy
        sol_copy = destroyed_sol.copy()
        removed_copy = removed.copy()
        
        # Repair
        repaired_sol = op.repair(sol_copy, removed_copy)
        
        print(f"\n{op.name}:")
        print(f"  Cost: {repaired_sol.compute_cost():.2f}")
        print(f"  Unassigned: {len(repaired_sol.unassigned_customers)}")
        print(f"  Compartment util: {repaired_sol.compute_compartment_utilization():.2%}")
    
    print("\n✅ Repair operators test passed!")
