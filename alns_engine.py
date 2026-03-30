"""
ALNS Engine for 2E-LRP-MC

This module implements the core ALNS (Adaptive Large Neighborhood Search)
algorithm engine that orchestrates the destroy-repair cycle.

Features:
    - Simulated Annealing acceptance criterion
    - Adaptive operator selection (Roulette Wheel)
    - Score-based operator weight updates
    - Integration with RL-based operator selection

Author: HG-DRL-ALNS Project
"""

from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
import time
from copy import deepcopy

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_structures import ProblemInstance
from core.solution import Solution, SolutionBuilder
from alns.destroy_operators import DestroyOperator, DestroyOperatorFactory
from alns.repair_operators import RepairOperator, RepairOperatorFactory


# =============================================================================
# Acceptance Criteria
# =============================================================================

class AcceptanceCriterion(Enum):
    """Types of acceptance criteria."""
    SIMULATED_ANNEALING = auto()
    RECORD_TO_RECORD = auto()
    GREEDY = auto()


@dataclass
class SimulatedAnnealingConfig:
    """Configuration for Simulated Annealing."""
    initial_temperature: float = 100.0
    cooling_rate: float = 0.995
    min_temperature: float = 0.01
    
    def get_temperature(self, iteration: int) -> float:
        """Get temperature at given iteration."""
        temp = self.initial_temperature * (self.cooling_rate ** iteration)
        return max(temp, self.min_temperature)


# =============================================================================
# Operator Scores
# =============================================================================

class OperatorScores:
    """
    Tracks operator performance scores for adaptive selection.
    
    Score categories:
        - New best solution found: σ1 (highest)
        - Better than current solution: σ2
        - Accepted (worse but accepted): σ3
        - Rejected: 0
    """
    
    def __init__(
        self,
        sigma1: float = 33.0,  # New best
        sigma2: float = 9.0,   # Better than current
        sigma3: float = 3.0,   # Accepted worse
        decay_factor: float = 0.8,
        segment_length: int = 100
    ):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.decay_factor = decay_factor
        self.segment_length = segment_length
        
        # Tracking current segment
        self.scores: Dict[str, float] = {}
        self.usage_counts: Dict[str, int] = {}
        self.weights: Dict[str, float] = {}
    
    def initialize(self, operator_names: List[str]):
        """Initialize weights for all operators."""
        for name in operator_names:
            self.scores[name] = 0.0
            self.usage_counts[name] = 0
            self.weights[name] = 1.0
    
    def record_performance(
        self,
        operator_name: str,
        is_new_best: bool,
        is_better: bool,
        is_accepted: bool
    ):
        """Record the performance of an operator application."""
        self.usage_counts[operator_name] = self.usage_counts.get(operator_name, 0) + 1
        
        if is_new_best:
            self.scores[operator_name] = self.scores.get(operator_name, 0) + self.sigma1
        elif is_better:
            self.scores[operator_name] = self.scores.get(operator_name, 0) + self.sigma2
        elif is_accepted:
            self.scores[operator_name] = self.scores.get(operator_name, 0) + self.sigma3
    
    def update_weights(self):
        """Update weights at end of segment."""
        for name in self.scores:
            if self.usage_counts.get(name, 0) > 0:
                # New weight = decay * old_weight + (1-decay) * performance
                performance = self.scores[name] / self.usage_counts[name]
                self.weights[name] = (
                    self.decay_factor * self.weights.get(name, 1.0) +
                    (1 - self.decay_factor) * performance
                )
        
        # Reset segment counters
        self.scores = {name: 0.0 for name in self.scores}
        self.usage_counts = {name: 0 for name in self.usage_counts}
    
    def get_selection_probabilities(self, operator_names: List[str]) -> np.ndarray:
        """Get probability distribution for operator selection."""
        weights = np.array([self.weights.get(name, 1.0) for name in operator_names])
        
        # Ensure non-negative
        weights = np.maximum(weights, 0.01)
        
        # Normalize
        return weights / weights.sum()


# =============================================================================
# ALNS Engine
# =============================================================================

@dataclass
class ALNSConfig:
    """Configuration for ALNS algorithm."""
    max_iterations: int = 10000
    max_time_seconds: float = 300.0
    destroy_ratio_min: float = 0.1
    destroy_ratio_max: float = 0.4
    acceptance: AcceptanceCriterion = AcceptanceCriterion.SIMULATED_ANNEALING
    sa_config: SimulatedAnnealingConfig = field(default_factory=SimulatedAnnealingConfig)
    segment_length: int = 100
    use_adaptive_weights: bool = True
    verbose: bool = True
    log_interval: int = 100


@dataclass
class ALNSResult:
    """Result of ALNS execution."""
    best_solution: Solution
    best_cost: float
    iterations: int
    runtime: float
    cost_history: List[float]
    operator_stats: Dict[str, Dict[str, int]]


class ALNSEngine:
    """
    Adaptive Large Neighborhood Search Engine.
    
    Orchestrates the destroy-repair cycle with:
        - Operator selection (adaptive or RL-based)
        - Acceptance criteria
        - Best solution tracking
    """
    
    def __init__(
        self,
        instance: ProblemInstance,
        config: ALNSConfig = None,
        destroy_operators: List[DestroyOperator] = None,
        repair_operators: List[RepairOperator] = None
    ):
        """
        Args:
            instance: Problem instance
            config: ALNS configuration
            destroy_operators: List of destroy operators (default: all)
            repair_operators: List of repair operators (default: all)
        """
        self.instance = instance
        self.config = config or ALNSConfig()
        
        # Initialize operators
        self.destroy_operators = destroy_operators or DestroyOperatorFactory.create_all(instance)
        self.repair_operators = repair_operators or RepairOperatorFactory.create_all(instance)
        
        # Operator names for tracking
        self.destroy_names = [op.name for op in self.destroy_operators]
        self.repair_names = [op.name for op in self.repair_operators]
        
        # Initialize adaptive weights
        self.destroy_scores = OperatorScores(segment_length=self.config.segment_length)
        self.repair_scores = OperatorScores(segment_length=self.config.segment_length)
        
        self.destroy_scores.initialize(self.destroy_names)
        self.repair_scores.initialize(self.repair_names)
        
        # Statistics
        self.stats = {
            'destroy_usage': {name: 0 for name in self.destroy_names},
            'repair_usage': {name: 0 for name in self.repair_names},
            'destroy_success': {name: 0 for name in self.destroy_names},
            'repair_success': {name: 0 for name in self.repair_names}
        }
    
    def run(self, initial_solution: Solution = None) -> ALNSResult:
        """
        Run ALNS algorithm.
        
        Args:
            initial_solution: Starting solution (if None, builds greedy solution)
        
        Returns:
            ALNSResult with best solution and statistics
        """
        start_time = time.time()
        
        # Initialize solution
        if initial_solution is None:
            builder = SolutionBuilder(self.instance)
            current_solution = builder.build_greedy_solution()
        else:
            current_solution = initial_solution.copy()
        
        best_solution = current_solution.copy()
        best_cost = best_solution.compute_cost()
        current_cost = best_cost
        
        cost_history = [best_cost]
        iteration = 0
        
        if self.config.verbose:
            print(f"ALNS Started - Initial cost: {best_cost:.2f}")
        
        while self._should_continue(iteration, start_time):
            # Select operators
            destroy_idx = self._select_operator(
                self.destroy_names,
                self.destroy_scores
            )
            repair_idx = self._select_operator(
                self.repair_names,
                self.repair_scores
            )
            
            destroy_op = self.destroy_operators[destroy_idx]
            repair_op = self.repair_operators[repair_idx]
            
            # Determine destroy ratio
            destroy_ratio = np.random.uniform(
                self.config.destroy_ratio_min,
                self.config.destroy_ratio_max
            )
            num_to_remove = max(1, int(len(self.instance.customers) * destroy_ratio))
            
            # Apply destroy-repair
            working_solution = current_solution.copy()
            working_solution, removed = destroy_op.destroy(working_solution, num_to_remove)
            working_solution = repair_op.repair(working_solution, removed)
            
            new_cost = working_solution.compute_cost()
            
            # Determine acceptance
            is_new_best = new_cost < best_cost
            is_better = new_cost < current_cost
            is_accepted = self._accept_solution(
                current_cost, new_cost, iteration
            )
            
            # Update statistics
            self.stats['destroy_usage'][destroy_op.name] += 1
            self.stats['repair_usage'][repair_op.name] += 1
            
            if is_accepted:
                self.stats['destroy_success'][destroy_op.name] += 1
                self.stats['repair_success'][repair_op.name] += 1
            
            # Record operator performance
            self.destroy_scores.record_performance(
                destroy_op.name, is_new_best, is_better, is_accepted
            )
            self.repair_scores.record_performance(
                repair_op.name, is_new_best, is_better, is_accepted
            )
            
            # Update solutions
            if is_accepted:
                current_solution = working_solution
                current_cost = new_cost
            
            if is_new_best:
                best_solution = working_solution.copy()
                best_cost = new_cost
            
            cost_history.append(best_cost)
            
            # Update weights at segment end
            if self.config.use_adaptive_weights:
                if (iteration + 1) % self.config.segment_length == 0:
                    self.destroy_scores.update_weights()
                    self.repair_scores.update_weights()
            
            # Logging
            if self.config.verbose and (iteration + 1) % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Iter {iteration + 1}: Best={best_cost:.2f}, "
                      f"Current={current_cost:.2f}, Time={elapsed:.1f}s")
            
            iteration += 1
        
        runtime = time.time() - start_time
        
        if self.config.verbose:
            print(f"ALNS Finished - Best cost: {best_cost:.2f}, "
                  f"Iterations: {iteration}, Time: {runtime:.1f}s")
        
        return ALNSResult(
            best_solution=best_solution,
            best_cost=best_cost,
            iterations=iteration,
            runtime=runtime,
            cost_history=cost_history,
            operator_stats=self.stats
        )
    
    def step(
        self,
        solution: Solution,
        destroy_idx: int,
        repair_idx: int,
        destroy_ratio: float
    ) -> Tuple[Solution, float, bool]:
        """
        Execute a single ALNS step.
        
        Used for RL-based operator selection.
        
        Args:
            solution: Current solution
            destroy_idx: Index of destroy operator to use
            repair_idx: Index of repair operator to use
            destroy_ratio: Ratio of customers to remove
        
        Returns:
            Tuple of (new_solution, new_cost, is_improvement)
        """
        current_cost = solution.compute_cost()
        
        # Get operators
        destroy_op = self.destroy_operators[destroy_idx]
        repair_op = self.repair_operators[repair_idx]
        
        # Determine number to remove
        num_to_remove = max(1, int(len(self.instance.customers) * destroy_ratio))
        
        # Apply destroy-repair
        working_solution = solution.copy()
        working_solution, removed = destroy_op.destroy(working_solution, num_to_remove)
        working_solution = repair_op.repair(working_solution, removed)
        
        new_cost = working_solution.compute_cost()
        is_improvement = new_cost < current_cost
        
        return working_solution, new_cost, is_improvement
    
    def _should_continue(self, iteration: int, start_time: float) -> bool:
        """Check if algorithm should continue."""
        if iteration >= self.config.max_iterations:
            return False
        if time.time() - start_time >= self.config.max_time_seconds:
            return False
        return True
    
    def _select_operator(
        self,
        operator_names: List[str],
        scores: OperatorScores
    ) -> int:
        """Select operator using roulette wheel selection."""
        if self.config.use_adaptive_weights:
            probs = scores.get_selection_probabilities(operator_names)
            return np.random.choice(len(operator_names), p=probs)
        else:
            return np.random.randint(len(operator_names))
    
    def _accept_solution(
        self,
        current_cost: float,
        new_cost: float,
        iteration: int
    ) -> bool:
        """Determine if new solution should be accepted."""
        if new_cost <= current_cost:
            return True
        
        if self.config.acceptance == AcceptanceCriterion.GREEDY:
            return False
        
        elif self.config.acceptance == AcceptanceCriterion.SIMULATED_ANNEALING:
            temperature = self.config.sa_config.get_temperature(iteration)
            delta = new_cost - current_cost
            
            if temperature <= 0:
                return False
            
            probability = np.exp(-delta / temperature)
            return np.random.random() < probability
        
        elif self.config.acceptance == AcceptanceCriterion.RECORD_TO_RECORD:
            # TODO: Implement Record-to-Record Travel
            return False
        
        return False
    
    def get_operator_statistics(self) -> Dict:
        """Get detailed operator statistics."""
        stats = {
            'destroy_operators': {},
            'repair_operators': {}
        }
        
        for name in self.destroy_names:
            usage = self.stats['destroy_usage'][name]
            success = self.stats['destroy_success'][name]
            stats['destroy_operators'][name] = {
                'usage': usage,
                'success': success,
                'success_rate': success / usage if usage > 0 else 0,
                'weight': self.destroy_scores.weights.get(name, 1.0)
            }
        
        for name in self.repair_names:
            usage = self.stats['repair_usage'][name]
            success = self.stats['repair_success'][name]
            stats['repair_operators'][name] = {
                'usage': usage,
                'success': success,
                'success_rate': success / usage if usage > 0 else 0,
                'weight': self.repair_scores.weights.get(name, 1.0)
            }
        
        return stats


if __name__ == "__main__":
    from core import create_random_instance
    
    print("Testing ALNS Engine...")
    
    # Create instance
    instance = create_random_instance(num_customers=50, seed=42)
    
    # Configure ALNS
    config = ALNSConfig(
        max_iterations=500,
        max_time_seconds=60.0,
        verbose=True,
        log_interval=100
    )
    
    # Run ALNS
    engine = ALNSEngine(instance, config)
    result = engine.run()
    
    print(f"\nFinal Results:")
    print(f"  Best cost: {result.best_cost:.2f}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Runtime: {result.runtime:.2f}s")
    print(f"  Unassigned: {len(result.best_solution.unassigned_customers)}")
    
    # Operator statistics
    print(f"\nOperator Statistics:")
    stats = engine.get_operator_statistics()
    
    print("  Destroy operators:")
    for name, data in stats['destroy_operators'].items():
        print(f"    {name}: usage={data['usage']}, "
              f"success_rate={data['success_rate']:.2%}, weight={data['weight']:.2f}")
    
    print("  Repair operators:")
    for name, data in stats['repair_operators'].items():
        print(f"    {name}: usage={data['usage']}, "
              f"success_rate={data['success_rate']:.2%}, weight={data['weight']:.2f}")
    
    print("\n✅ ALNS Engine test passed!")
