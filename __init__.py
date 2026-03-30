"""
HG-DRL-ALNS ALNS Package

This package contains the Adaptive Large Neighborhood Search components
for the Two-Echelon Location-Routing Problem with Multi-Compartment vehicles.

Modules:
    - destroy_operators: Destroy (removal) operators
    - repair_operators: Repair (insertion) operators
    - alns_engine: Core ALNS algorithm engine
"""

from .destroy_operators import (
    DestroyOperator,
    RandomRemoval,
    WorstCostRemoval,
    ClusterRemoval,
    WasteTypeRemoval,
    StationClosureRemoval,
    RelatedRemoval,
    DestroyOperatorFactory
)

from .repair_operators import (
    RepairOperator,
    GreedyRepair,
    RegretRepair,
    BestFitCompartmentRepair,
    StationOpeningRepair,
    RandomRepair,
    RepairOperatorFactory
)

from .alns_engine import (
    ALNSConfig,
    ALNSResult,
    ALNSEngine,
    AcceptanceCriterion,
    SimulatedAnnealingConfig,
    OperatorScores
)

__all__ = [
    # Destroy Operators
    'DestroyOperator',
    'RandomRemoval',
    'WorstCostRemoval',
    'ClusterRemoval',
    'WasteTypeRemoval',
    'StationClosureRemoval',
    'RelatedRemoval',
    'DestroyOperatorFactory',
    # Repair Operators
    'RepairOperator',
    'GreedyRepair',
    'RegretRepair',
    'BestFitCompartmentRepair',
    'StationOpeningRepair',
    'RandomRepair',
    'RepairOperatorFactory',
    # Engine
    'ALNSConfig',
    'ALNSResult',
    'ALNSEngine',
    'AcceptanceCriterion',
    'SimulatedAnnealingConfig',
    'OperatorScores'
]
