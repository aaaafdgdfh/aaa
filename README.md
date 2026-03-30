# HG-DRL-ALNS

## Heterogeneous Graph Attention Reinforcement Learning Assisted ALNS for 2E-LRP-MC

> A novel algorithm combining **Heterogeneous Graph Attention Networks (HGAT)** with **Proximal Policy Optimization (PPO)** to guide **Adaptive Large Neighborhood Search (ALNS)** for solving the Two-Echelon Location-Routing Problem with Multi-Compartment vehicles.

---

## 📁 Project Structure

```
HG-DRL-ALNS/
├── core/                           # Core data structures and constraints
│   ├── __init__.py
│   ├── data_structures.py          # Customer, Station, Vehicle classes
│   ├── constraints.py              # Multi-compartment constraint checking
│   └── solution.py                 # Solution representation
├── models/                         # Neural network models
│   ├── __init__.py
│   ├── hgat_network.py             # HGAT encoder + Actor-Critic
│   └── ppo_trainer.py              # PPO training utilities
├── alns/                           # ALNS algorithm components
│   ├── __init__.py
│   ├── destroy_operators.py        # Destroy (removal) operators
│   ├── repair_operators.py         # Repair (insertion) operators
│   └── alns_engine.py              # ALNS execution engine
├── docs/                           # Documentation
│   ├── HGAT网络架构说明.md
│   └── 约束处理逻辑说明.md
├── train.py                        # Main training script
├── 数学模型.md                      # Original mathematical model
├── 数学模型_修正版.md               # Corrected mathematical model
├── 1-算法方案.md                    # Algorithm design
├── 2-具体实现方案.md                # Implementation details
└── README.md                        # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
cd HG-DRL-ALNS

# Install dependencies
pip install torch torch-geometric numpy tensorboard

# Install PyG dependencies
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Training

```bash
# Run with default config
python train.py

# Run with custom parameters
python train.py --num-episodes 500 --num-customers 100 --output-dir my_runs

# Run with config file
python train.py --config config.json
```

### Testing ALNS

```python
from core import create_random_instance, SolutionBuilder
from alns import ALNSEngine, ALNSConfig

# Create problem instance
instance = create_random_instance(num_customers=50, seed=42)

# Run ALNS
engine = ALNSEngine(instance, ALNSConfig(max_iterations=1000))
result = engine.run()

print(f"Best cost: {result.best_cost}")
```

---

## 🧠 Algorithm Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        HG-DRL-ALNS                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     │
│   │   Solution  │────▶│    HGAT     │────▶│  PPO Agent  │     │
│   │    State    │     │   Encoder   │     │             │     │
│   └─────────────┘     └─────────────┘     └──────┬──────┘     │
│                                                   │             │
│                           ┌───────────────────────┼─────────┐   │
│                           │                       ▼         │   │
│                           │    ┌─────────────────────┐      │   │
│                           │    │  Destroy Operator   │      │   │
│                           │    └──────────┬──────────┘      │   │
│                           │               │                 │   │
│                           │    ┌──────────▼──────────┐      │   │
│                           │    │  Repair Operator    │      │   │
│                           │    └──────────┬──────────┘      │   │
│                           │               │                 │   │
│                           │    ┌──────────▼──────────┐      │   │
│                           │    │  Accept/Reject      │      │   │
│                           │    └──────────┬──────────┘      │   │
│                           │               │                 │   │
│                           └───────────────┼─────────────────┘   │
│                                           │                     │
│                                           ▼                     │
│                              ┌─────────────────────┐           │
│                              │   Updated Solution  │───────────┤
│                              └─────────────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **HGAT Encoder** | Encodes problem state as heterogeneous graph (Customers, Stations, Plants) |
| **PPO Agent** | Selects destroy/repair operators and parameters |
| **Destroy Operators** | Random, Worst-Cost, Cluster, Waste-Type, Station-Closure, Related |
| **Repair Operators** | Greedy, Regret-k, Best-Fit Compartment, Station-Opening |

---

## 📊 Mathematical Model

The 2E-LRP-MC problem is formulated as:

$$\min Z = C_1 + C_2 + C_3 + C_4$$

Where:
- $C_1$: Vehicle fixed costs
- $C_2$: Transportation variable costs
- $C_3$: Facility opening costs
- $C_4$: Facility operating costs

See [数学模型_修正版.md](数学模型_修正版.md) for the complete formulation.

---

## 🔧 Key Features

### Multi-Compartment Constraint Handling

```python
from core import CompartmentChecker, InsertionFeasibilityChecker

checker = InsertionFeasibilityChecker(instance)

# Check if customer can be inserted
result = checker.check_full_insertion(vehicle, customer, position)

if result.is_feasible:
    solution.insert_customer(
        vehicle_id, customer_id, position,
        result.required_compartment_assignments  # Compartment allocation
    )
```

### Best-Fit Compartment Insertion

Minimizes compartment fragmentation:

```python
from alns import BestFitCompartmentRepair

repair_op = BestFitCompartmentRepair(instance)
solution = repair_op.repair(partial_solution, removed_customers)
```

### RL-Guided Operator Selection

```python
# Network selects operators
destroy_idx, repair_idx, ratio = network.sample_action(state)

# Execute ALNS step
new_solution = alns_engine.step(solution, destroy_idx, repair_idx, ratio)
```

---

## 📈 Training Metrics

Training logs are saved to TensorBoard:

```bash
tensorboard --logdir runs/
```

Tracked metrics:
- `train/policy_loss`: PPO policy loss
- `train/value_loss`: Value function loss
- `episode/reward`: Episode cumulative reward
- `episode/best_cost`: Best cost found
- `val/cost`: Validation cost

---

## 📚 Documentation

- [HGAT网络架构说明.md](docs/HGAT网络架构说明.md) - HGAT Architecture
- [约束处理逻辑说明.md](docs/约束处理逻辑说明.md) - Constraint Handling

---

## 🔬 Operators

### Destroy Operators

| Operator | Description |
|----------|-------------|
| RandomRemoval | Randomly remove customers |
| WorstCostRemoval | Remove highest-cost customers |
| ClusterRemoval | Remove geographically clustered customers |
| WasteTypeRemoval | Remove customers with specific waste type |
| StationClosureRemoval | Close a low-utilization station |
| RelatedRemoval | Remove related customers (Shaw) |

### Repair Operators

| Operator | Description |
|----------|-------------|
| GreedyRepair | Insert at minimum cost position |
| RegretRepair | Regret-k insertion heuristic |
| BestFitCompartmentRepair | Optimize compartment utilization |
| StationOpeningRepair | Open new stations if needed |
| RandomRepair | Random feasible insertion |

---

## 📄 License

This project is for academic research purposes.

---

## 📧 Contact

For questions or issues, please open an issue on the repository.
