"""
HG-DRL-ALNS Training Script

This script implements the complete training loop for the
Heterogeneous Graph Attention Reinforcement Learning Assisted ALNS algorithm.

Components:
    - HGAT-based state encoding
    - PPO-based operator selection
    - ALNS destroy-repair cycle
    - Training and validation

Author: HG-DRL-ALNS Project
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import (
    ProblemInstance,
    create_random_instance,
    Solution,
    SolutionBuilder
)

from models import (
    HeteroGraphBuilder,
    ActorCriticNetwork,
    create_default_network,
    PPOTrainer,
    ExperienceBuffer,
    Experience,
    RewardCalculator
)

from alns import (
    ALNSEngine,
    ALNSConfig,
    DestroyOperatorFactory,
    RepairOperatorFactory
)


# =============================================================================
# Training Environment
# =============================================================================

class TrainingEnvironment:
    """
    Training environment for HG-DRL-ALNS.
    
    Wraps the ALNS engine and provides RL-compatible interface.
    """
    
    def __init__(
        self,
        instance: ProblemInstance,
        graph_builder: HeteroGraphBuilder,
        destroy_ratio_range: Tuple[float, float] = (0.1, 0.4)
    ):
        self.instance = instance
        self.graph_builder = graph_builder
        self.destroy_ratio_range = destroy_ratio_range
        
        # Create ALNS engine
        self.destroy_operators = DestroyOperatorFactory.create_all(instance)
        self.repair_operators = RepairOperatorFactory.create_all(instance)
        
        self.alns_engine = ALNSEngine(
            instance,
            config=ALNSConfig(use_adaptive_weights=False),
            destroy_operators=self.destroy_operators,
            repair_operators=self.repair_operators
        )
        
        self.num_destroy_ops = len(self.destroy_operators)
        self.num_repair_ops = len(self.repair_operators)
        
        # State tracking
        self.current_solution: Optional[Solution] = None
        self.best_solution: Optional[Solution] = None
        self.best_cost: float = float('inf')
        self.current_cost: float = float('inf')
        self.step_count: int = 0
        
        # Reward calculator
        self.reward_calc = RewardCalculator(
            record_bonus=10.0,
            compactness_weight=0.1,
            use_log_reward=True
        )
    
    def reset(self, solution: Optional[Solution] = None) -> 'HeteroData':
        """
        Reset environment with initial solution.
        
        Returns:
            Initial state as HeteroData graph
        """
        if solution is None:
            builder = SolutionBuilder(self.instance)
            self.current_solution = builder.build_greedy_solution()
        else:
            self.current_solution = solution.copy()
        
        self.best_solution = self.current_solution.copy()
        self.best_cost = self.current_solution.compute_cost()
        self.current_cost = self.best_cost
        self.step_count = 0
        
        return self._get_state()
    
    def step(
        self,
        destroy_idx: int,
        repair_idx: int,
        destroy_ratio: float
    ) -> Tuple['HeteroData', float, bool, Dict]:
        """
        Execute one ALNS step.
        
        Args:
            destroy_idx: Index of destroy operator
            repair_idx: Index of repair operator
            destroy_ratio: Ratio of customers to remove
        
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        old_cost = self.current_cost
        old_compactness = self.current_solution.compute_compartment_utilization()
        
        # Execute ALNS step
        new_solution, new_cost, is_improvement = self.alns_engine.step(
            self.current_solution,
            destroy_idx,
            repair_idx,
            destroy_ratio
        )
        
        new_compactness = new_solution.compute_compartment_utilization()
        
        # Calculate reward
        reward, is_new_record = self.reward_calc.compute_reward(
            old_cost, new_cost, self.best_cost,
            old_compactness, new_compactness
        )
        
        # Update state
        self.current_solution = new_solution
        self.current_cost = new_cost
        
        if is_new_record:
            self.best_solution = new_solution.copy()
            self.best_cost = new_cost
        
        self.step_count += 1
        
        # Info dict
        info = {
            'cost': new_cost,
            'best_cost': self.best_cost,
            'is_improvement': is_improvement,
            'is_new_record': is_new_record,
            'compactness': new_compactness,
            'unassigned': len(new_solution.unassigned_customers),
            'destroy_op': self.destroy_operators[destroy_idx].name,
            'repair_op': self.repair_operators[repair_idx].name
        }
        
        # Episode is never "done" in ALNS (continuous improvement)
        done = False
        
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> 'HeteroData':
        """Convert current solution to graph state."""
        # Extract data from solution
        customer_coords = np.array([c.coords for c in self.instance.customers])
        customer_demands = np.array([
            [c.demand_for_type(w) for w in range(self.instance.num_waste_types)]
            for c in self.instance.customers
        ])
        
        station_coords = np.array([s.coords for s in self.instance.stations])
        station_capacities = np.array([s.capacity for s in self.instance.stations])
        station_fixed_costs = np.array([
            sum(s.fixed_costs.values()) for s in self.instance.stations
        ])
        station_open_flags = np.array([
            1.0 if s.id in self.current_solution.open_stations else 0.0
            for s in self.instance.stations
        ])
        
        plant_coords = np.array([p.coords for p in self.instance.plants])
        plant_capacities = np.array([p.capacity for p in self.instance.plants])
        
        # Customer served flags (1 if served, 0 if unassigned)
        customer_served = np.array([
            0.0 if c.id in self.current_solution.unassigned_customers else 1.0
            for c in self.instance.customers
        ])
        
        # Build graph
        graph = self.graph_builder.build_graph(
            customer_coords=customer_coords,
            customer_demands=customer_demands,
            station_coords=station_coords,
            station_capacities=station_capacities,
            station_fixed_costs=station_fixed_costs,
            station_open_flags=station_open_flags,
            plant_coords=plant_coords,
            plant_capacities=plant_capacities,
            customer_served_flags=customer_served
        )
        
        return graph
    
    def get_action_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action masks for operators.
        
        Returns masks indicating which operators are valid.
        """
        # For now, all operators are valid
        destroy_mask = torch.ones(self.num_destroy_ops, dtype=torch.bool)
        repair_mask = torch.ones(self.num_repair_ops, dtype=torch.bool)
        
        # Mask station closure if only one station
        if len(self.current_solution.open_stations) <= 1:
            # Find index of StationClosureRemoval
            for i, op in enumerate(self.destroy_operators):
                if op.name == "StationClosureRemoval":
                    destroy_mask[i] = False
        
        return destroy_mask, repair_mask


# =============================================================================
# Trainer
# =============================================================================

class HGDRLALNSTrainer:
    """
    Main trainer for HG-DRL-ALNS.
    """
    
    def __init__(
        self,
        config: Dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.config = config
        self.device = device
        
        # Create output directory
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config.get('output_dir', 'runs')) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.output_dir / 'tensorboard')
        
        # Create network
        self.num_waste_types = config.get('num_waste_types', 3)
        self.network = create_default_network(
            num_waste_types=self.num_waste_types,
            num_destroy_ops=6,  # Number of destroy operators
            num_repair_ops=6,   # Number of repair operators
            hidden_dim=config.get('hidden_dim', 128)
        ).to(device)
        
        # PPO Trainer
        self.ppo_trainer = PPOTrainer(
            network=self.network,
            lr=config.get('learning_rate', 3e-4),
            clip_epsilon=config.get('clip_epsilon', 0.2),
            value_coef=config.get('value_coef', 0.5),
            entropy_coef=config.get('entropy_coef', 0.01),
            ppo_epochs=config.get('ppo_epochs', 4),
            mini_batch_size=config.get('mini_batch_size', 32),
            device=device
        )
        
        # Graph builder
        self.graph_builder = HeteroGraphBuilder(num_waste_types=self.num_waste_types)
        
        # Experience buffer
        self.buffer = ExperienceBuffer(
            gamma=config.get('gamma', 0.99),
            gae_lambda=config.get('gae_lambda', 0.95)
        )
        
        # Training state
        self.global_step = 0
        self.best_val_cost = float('inf')
    
    def train(self):
        """Run training loop."""
        config = self.config
        
        num_episodes = config.get('num_episodes', 1000)
        steps_per_episode = config.get('steps_per_episode', 100)
        update_interval = config.get('update_interval', 100)
        val_interval = config.get('val_interval', 50)
        save_interval = config.get('save_interval', 100)
        
        print(f"Starting training with {num_episodes} episodes...")
        print(f"Output directory: {self.output_dir}")
        
        for episode in range(num_episodes):
            # Generate random instance for this episode
            instance = create_random_instance(
                num_customers=config.get('num_customers', 50),
                num_stations=config.get('num_stations', 5),
                num_plants=config.get('num_plants', 2),
                num_bucket_vehicles=config.get('num_bucket_vehicles', 10),
                num_waste_types=self.num_waste_types
            )
            
            # Create environment
            env = TrainingEnvironment(instance, self.graph_builder)
            state = env.reset()
            
            episode_reward = 0.0
            episode_costs = []
            
            for step in range(steps_per_episode):
                # Get action masks
                destroy_mask, repair_mask = env.get_action_masks()
                
                # Move to device
                state = state.to(self.device)
                
                # Sample action
                self.network.eval()
                with torch.no_grad():
                    destroy_idx, repair_idx, ratio, log_p_d, log_p_r, log_p_ratio = \
                        self.network.sample_action(
                            state,
                            destroy_mask.unsqueeze(0).to(self.device),
                            repair_mask.unsqueeze(0).to(self.device),
                            ratio_range=(0.1, 0.4)
                        )
                    
                    # Get value estimate
                    output = self.network(state)
                    value = output.value.item()
                
                # Execute action
                next_state, reward, done, info = env.step(destroy_idx, repair_idx, ratio)
                
                # Store experience
                exp = Experience(
                    state=state.cpu(),
                    destroy_idx=destroy_idx,
                    repair_idx=repair_idx,
                    ratio_sample=ratio,
                    log_prob_destroy=log_p_d.item(),
                    log_prob_repair=log_p_r.item(),
                    log_prob_ratio=log_p_ratio.item(),
                    reward=reward,
                    value=value,
                    done=done
                )
                self.buffer.add(exp)
                
                episode_reward += reward
                episode_costs.append(info['cost'])
                state = next_state
                self.global_step += 1
                
                # PPO update
                if len(self.buffer) >= update_interval:
                    self.network.train()
                    stats = self.ppo_trainer.update(self.buffer, last_value=value)
                    self.buffer.clear()
                    
                    # Log training stats
                    self.writer.add_scalar('train/policy_loss', stats['policy_loss'], self.global_step)
                    self.writer.add_scalar('train/value_loss', stats['value_loss'], self.global_step)
                    self.writer.add_scalar('train/entropy', stats['entropy'], self.global_step)
            
            # Episode logging
            self.writer.add_scalar('episode/reward', episode_reward, episode)
            self.writer.add_scalar('episode/final_cost', episode_costs[-1], episode)
            self.writer.add_scalar('episode/best_cost', env.best_cost, episode)
            self.writer.add_scalar('episode/avg_cost', np.mean(episode_costs), episode)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes} - "
                      f"Reward: {episode_reward:.2f}, "
                      f"Best Cost: {env.best_cost:.2f}")
            
            # Validation
            if (episode + 1) % val_interval == 0:
                val_cost = self._validate()
                self.writer.add_scalar('val/cost', val_cost, episode)
                
                if val_cost < self.best_val_cost:
                    self.best_val_cost = val_cost
                    self._save_checkpoint('best.pt', episode)
                    print(f"  New best validation cost: {val_cost:.2f}")
            
            # Periodic save
            if (episode + 1) % save_interval == 0:
                self._save_checkpoint(f'checkpoint_{episode + 1}.pt', episode)
        
        # Final save
        self._save_checkpoint('final.pt', num_episodes)
        self.writer.close()
        
        print(f"Training complete! Best validation cost: {self.best_val_cost:.2f}")
    
    def _validate(self, num_instances: int = 5) -> float:
        """Run validation on random instances."""
        self.network.eval()
        
        total_cost = 0.0
        
        for _ in range(num_instances):
            instance = create_random_instance(
                num_customers=self.config.get('num_customers', 50),
                num_waste_types=self.num_waste_types,
                seed=np.random.randint(10000)
            )
            
            env = TrainingEnvironment(instance, self.graph_builder)
            state = env.reset()
            
            # Run for some steps
            for _ in range(50):
                state = state.to(self.device)
                
                with torch.no_grad():
                    destroy_idx, repair_idx, ratio, *_ = self.network.sample_action(
                        state, ratio_range=(0.1, 0.4)
                    )
                
                state, _, _, _ = env.step(destroy_idx, repair_idx, ratio)
            
            total_cost += env.best_cost
        
        return total_cost / num_instances
    
    def _save_checkpoint(self, filename: str, episode: int):
        """Save training checkpoint."""
        self.ppo_trainer.save_checkpoint(
            self.output_dir / filename,
            epoch=episode,
            additional_info={
                'global_step': self.global_step,
                'best_val_cost': self.best_val_cost,
                'config': self.config
            }
        )


# =============================================================================
# Main
# =============================================================================

def get_default_config() -> Dict:
    """Get default training configuration."""
    return {
        # Problem
        'num_customers': 50,
        'num_stations': 5,
        'num_plants': 2,
        'num_bucket_vehicles': 10,
        'num_waste_types': 3,
        
        # Training
        'num_episodes': 1000,
        'steps_per_episode': 100,
        'update_interval': 100,
        'val_interval': 50,
        'save_interval': 100,
        
        # Network
        'hidden_dim': 128,
        
        # PPO
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'ppo_epochs': 4,
        'mini_batch_size': 32,
        
        # Output
        'output_dir': 'runs'
    }


def main():
    parser = argparse.ArgumentParser(description='HG-DRL-ALNS Training')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config JSON file')
    parser.add_argument('--num-episodes', type=int, default=None)
    parser.add_argument('--num-customers', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    # Load config
    config = get_default_config()
    
    if args.config:
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Override with command line args
    if args.num_episodes:
        config['num_episodes'] = args.num_episodes
    if args.num_customers:
        config['num_customers'] = args.num_customers
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("HG-DRL-ALNS Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print("=" * 60)
    
    # Create trainer and run
    trainer = HGDRLALNSTrainer(config, device=device)
    trainer.train()


if __name__ == "__main__":
    main()
