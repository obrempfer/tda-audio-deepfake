"""Topology-only neural modeling utilities.

Small interpretable models that operate only on explicit topological feature
blocks. This supports flat and staged robust-core-first training while keeping
block-level ablations straightforward.
"""

from .topology_models import (
    FeatureBlock,
    FeatureLayout,
    StageSpec,
    TopologyLinearBaseline,
    TopologyMLP,
    binary_metrics,
    evaluate_block_ablations,
    stack_feature_blocks,
)

__all__ = [
    "FeatureBlock",
    "FeatureLayout",
    "StageSpec",
    "TopologyLinearBaseline",
    "TopologyMLP",
    "binary_metrics",
    "evaluate_block_ablations",
    "stack_feature_blocks",
]
