"""
Evaluation Script for HG-DRL-ALNS

This script provides comprehensive evaluation tools for:
    - Comparing HG-DRL-ALNS with baselines (Pure ALNS, Greedy)
    - Running experiments on multiple instances
    - Statistical analysis of results
    - Generating evaluation reports

Author: HG-DRL-ALNS Project
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
import numpy as np
from copy import deepcopy

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from core import (
    ProblemInstance,
    create_random_instance,
    Solution,
    SolutionBuilder
)

from alns import (
    ALNSEngine,
    ALNSConfig,
    ALNSResult
)


# =============================================================================
# Evaluation Results
# =============================================================================

@dataclass
class InstanceResult:
    """Results for a single instance."""
    instance_id: int
    seed: int
    num_customers: int
    
    # Method results (method_name -> metrics)
    method_results: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    config: Dict
    instance_results: List[InstanceResult] = field(default_factory=list)
    
    # Aggregated statistics
    summary: Dict = field(default_factory=dict)


# =============================================================================
# Evaluation Methods
# =============================================================================

class Evaluator:
    """
    Evaluator for comparing different solution methods.
    """
    
    def __init__(
        self,
        methods: List[str] = None,
        num_instances: int = 10,
        num_customers_range: Tuple[int, int] = (30, 100),
        time_limit: float = 60.0,
        device: str = 'cpu'
    ):
        """
        Args:
            methods: List of methods to evaluate
            num_instances: Number of instances to evaluate
            num_customers_range: Range of customer counts
            time_limit: Time limit per instance (seconds)
            device: Device for RL methods
        """
        self.methods = methods or ['greedy', 'alns', 'alns_adaptive']
        self.num_instances = num_instances
        self.num_customers_range = num_customers_range
        self.time_limit = time_limit
        self.device = device
        
        # Results storage
        self.results: List[InstanceResult] = []
    
    def evaluate(self, verbose: bool = True) -> EvaluationResult:
        """
        Run evaluation on all instances and methods.
        
        Returns:
            EvaluationResult with all results
        """
        if verbose:
            print("=" * 60)
            print("HG-DRL-ALNS Evaluation")
            print("=" * 60)
            print(f"Methods: {self.methods}")
            print(f"Instances: {self.num_instances}")
            print(f"Customer range: {self.num_customers_range}")
            print(f"Time limit: {self.time_limit}s")
            print("=" * 60)
        
        self.results = []
        
        for i in range(self.num_instances):
            # Generate instance
            seed = 1000 + i
            num_customers = np.random.randint(
                self.num_customers_range[0],
                self.num_customers_range[1] + 1
            )
            
            instance = create_random_instance(
                num_customers=num_customers,
                num_waste_types=3,
                seed=seed
            )
            
            if verbose:
                print(f"\nInstance {i + 1}/{self.num_instances} "
                      f"(n={num_customers}, seed={seed})")
            
            # Evaluate each method
            instance_result = InstanceResult(
                instance_id=i,
                seed=seed,
                num_customers=num_customers
            )
            
            for method in self.methods:
                if verbose:
                    print(f"  Running {method}...", end=" ")
                
                start_time = time.time()
                result = self._run_method(method, instance)
                runtime = time.time() - start_time
                
                result['runtime'] = runtime
                instance_result.method_results[method] = result
                
                if verbose:
                    print(f"Cost: {result['cost']:.2f}, Time: {runtime:.2f}s")
            
            self.results.append(instance_result)
        
        # Generate summary
        summary = self._generate_summary()
        
        eval_result = EvaluationResult(
            config={
                'methods': self.methods,
                'num_instances': self.num_instances,
                'num_customers_range': self.num_customers_range,
                'time_limit': self.time_limit
            },
            instance_results=self.results,
            summary=summary
        )
        
        if verbose:
            self._print_summary(summary)
        
        return eval_result
    
    def _run_method(
        self,
        method: str,
        instance: ProblemInstance
    ) -> Dict:
        """Run a single method on an instance."""
        
        if method == 'greedy':
            return self._run_greedy(instance)
        
        elif method == 'alns':
            return self._run_alns(instance, adaptive=False)
        
        elif method == 'alns_adaptive':
            return self._run_alns(instance, adaptive=True)
        
        elif method == 'hg_drl_alns':
            return self._run_hg_drl_alns(instance)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _run_greedy(self, instance: ProblemInstance) -> Dict:
        """Run greedy construction."""
        builder = SolutionBuilder(instance)
        solution = builder.build_greedy_solution()
        solution.rebuild_second_echelon()
        
        return {
            'cost': solution.compute_cost(),
            'unassigned': len(solution.unassigned_customers),
            'num_routes': len([r for r in solution.routes.values() if not r.is_empty]),
            'compartment_util': solution.compute_compartment_utilization(),
            'iterations': 1
        }
    
    def _run_alns(
        self,
        instance: ProblemInstance,
        adaptive: bool = True
    ) -> Dict:
        """Run ALNS algorithm."""
        config = ALNSConfig(
            max_iterations=1000,
            max_time_seconds=self.time_limit,
            use_adaptive_weights=adaptive,
            verbose=False
        )
        
        engine = ALNSEngine(instance, config)
        result = engine.run()
        
        return {
            'cost': result.best_cost,
            'unassigned': len(result.best_solution.unassigned_customers),
            'num_routes': len([r for r in result.best_solution.routes.values() 
                             if not r.is_empty]),
            'compartment_util': result.best_solution.compute_compartment_utilization(),
            'iterations': result.iterations,
            'cost_history': result.cost_history
        }
    
    def _run_hg_drl_alns(self, instance: ProblemInstance) -> Dict:
        """Run HG-DRL-ALNS (requires trained model)."""
        # Check for trained model
        model_path = PROJECT_ROOT / 'runs' / 'best.pt'
        
        if not model_path.exists():
            # Fall back to ALNS if no model
            print("(No trained model, using ALNS)")
            return self._run_alns(instance, adaptive=True)
        
        # Load model and run
        try:
            from models import create_default_network, HeteroGraphBuilder
            from train import TrainingEnvironment
            
            # Load model
            network = create_default_network(num_waste_types=3)
            checkpoint = torch.load(model_path, map_location=self.device)
            network.load_state_dict(checkpoint['model_state_dict'])
            network.eval()
            network.to(self.device)
            
            # Create environment
            graph_builder = HeteroGraphBuilder(num_waste_types=3)
            env = TrainingEnvironment(instance, graph_builder)
            state = env.reset()
            
            # Run for time limit
            start_time = time.time()
            step = 0
            cost_history = [env.current_cost]
            
            while time.time() - start_time < self.time_limit:
                state = state.to(self.device)
                
                with torch.no_grad():
                    destroy_idx, repair_idx, ratio, *_ = network.sample_action(
                        state, ratio_range=(0.1, 0.4)
                    )
                
                state, _, _, _ = env.step(destroy_idx, repair_idx, ratio)
                cost_history.append(env.best_cost)
                step += 1
            
            return {
                'cost': env.best_cost,
                'unassigned': len(env.best_solution.unassigned_customers),
                'num_routes': len([r for r in env.best_solution.routes.values() 
                                 if not r.is_empty]),
                'compartment_util': env.best_solution.compute_compartment_utilization(),
                'iterations': step,
                'cost_history': cost_history
            }
            
        except Exception as e:
            print(f"Error running HG-DRL-ALNS: {e}")
            return self._run_alns(instance, adaptive=True)
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics."""
        summary = {method: {} for method in self.methods}
        
        for method in self.methods:
            costs = []
            runtimes = []
            unassigned = []
            utilizations = []
            
            for result in self.results:
                if method in result.method_results:
                    mr = result.method_results[method]
                    costs.append(mr['cost'])
                    runtimes.append(mr['runtime'])
                    unassigned.append(mr['unassigned'])
                    utilizations.append(mr['compartment_util'])
            
            if costs:
                summary[method] = {
                    'avg_cost': np.mean(costs),
                    'std_cost': np.std(costs),
                    'min_cost': np.min(costs),
                    'max_cost': np.max(costs),
                    'avg_runtime': np.mean(runtimes),
                    'avg_unassigned': np.mean(unassigned),
                    'avg_utilization': np.mean(utilizations)
                }
        
        # Compute improvements over baseline
        baseline = 'greedy'
        if baseline in summary and summary[baseline]:
            baseline_cost = summary[baseline]['avg_cost']
            
            for method in self.methods:
                if method != baseline and method in summary and summary[method]:
                    improvement = (baseline_cost - summary[method]['avg_cost']) / baseline_cost * 100
                    summary[method]['improvement_vs_greedy'] = improvement
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """Print summary table."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        # Table header
        print(f"{'Method':<20} {'Avg Cost':>12} {'Std':>10} {'Improv.':>10} {'Time':>8}")
        print("-" * 60)
        
        for method in self.methods:
            if method in summary and summary[method]:
                s = summary[method]
                improvement = s.get('improvement_vs_greedy', 0.0)
                print(f"{method:<20} {s['avg_cost']:>12.2f} {s['std_cost']:>10.2f} "
                      f"{improvement:>9.1f}% {s['avg_runtime']:>7.2f}s")
        
        print("=" * 60)
    
    def save_results(self, filepath: str):
        """Save results to JSON file."""
        # Convert to serializable format
        data = {
            'config': {
                'methods': self.methods,
                'num_instances': self.num_instances,
                'num_customers_range': self.num_customers_range,
                'time_limit': self.time_limit
            },
            'instances': []
        }
        
        for result in self.results:
            instance_data = {
                'instance_id': result.instance_id,
                'seed': result.seed,
                'num_customers': result.num_customers,
                'methods': {}
            }
            
            for method, mr in result.method_results.items():
                # Remove non-serializable items
                method_data = {k: v for k, v in mr.items() 
                             if k != 'cost_history'}
                instance_data['methods'][method] = method_data
            
            data['instances'].append(instance_data)
        
        data['summary'] = self._generate_summary()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {filepath}")


# =============================================================================
# Benchmark Instance Generator
# =============================================================================

class BenchmarkGenerator:
    """
    Generate benchmark instances for evaluation.
    """
    
    @staticmethod
    def generate_small_instances(num: int = 5) -> List[ProblemInstance]:
        """Small instances (20-40 customers)."""
        instances = []
        for i in range(num):
            instance = create_random_instance(
                num_customers=np.random.randint(20, 41),
                num_stations=3,
                num_plants=2,
                seed=2000 + i
            )
            instances.append(instance)
        return instances
    
    @staticmethod
    def generate_medium_instances(num: int = 5) -> List[ProblemInstance]:
        """Medium instances (50-100 customers)."""
        instances = []
        for i in range(num):
            instance = create_random_instance(
                num_customers=np.random.randint(50, 101),
                num_stations=5,
                num_plants=3,
                seed=3000 + i
            )
            instances.append(instance)
        return instances
    
    @staticmethod
    def generate_large_instances(num: int = 5) -> List[ProblemInstance]:
        """Large instances (150-250 customers)."""
        instances = []
        for i in range(num):
            instance = create_random_instance(
                num_customers=np.random.randint(150, 251),
                num_stations=8,
                num_plants=4,
                num_bucket_vehicles=30,
                seed=4000 + i
            )
            instances.append(instance)
        return instances


# =============================================================================
# Statistical Tests
# =============================================================================

def wilcoxon_test(
    costs1: List[float],
    costs2: List[float],
    alpha: float = 0.05
) -> Tuple[float, bool]:
    """
    Perform Wilcoxon signed-rank test.
    
    Returns:
        Tuple of (p-value, is_significant)
    """
    try:
        from scipy.stats import wilcoxon
        stat, p_value = wilcoxon(costs1, costs2)
        return p_value, p_value < alpha
    except ImportError:
        print("scipy not installed, skipping statistical test")
        return 1.0, False


def compute_gap(
    actual: float,
    best_known: float
) -> float:
    """Compute optimality gap."""
    if best_known == 0:
        return 0.0
    return (actual - best_known) / best_known * 100


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='HG-DRL-ALNS Evaluation')
    
    parser.add_argument('--num-instances', type=int, default=10,
                       help='Number of instances to evaluate')
    parser.add_argument('--min-customers', type=int, default=30)
    parser.add_argument('--max-customers', type=int, default=100)
    parser.add_argument('--time-limit', type=float, default=60.0,
                       help='Time limit per instance')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['greedy', 'alns', 'alns_adaptive'],
                       help='Methods to evaluate')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = Evaluator(
        methods=args.methods,
        num_instances=args.num_instances,
        num_customers_range=(args.min_customers, args.max_customers),
        time_limit=args.time_limit
    )
    
    result = evaluator.evaluate(verbose=True)
    
    # Save results
    evaluator.save_results(args.output)
    
    print(f"\nEvaluation complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()
