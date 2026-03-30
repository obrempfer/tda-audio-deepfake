"""Persistent homology computation via Ripser.

Computes Vietoris-Rips persistent homology on feature-space point clouds
derived from sliding-window audio embeddings.
"""

import numpy as np
import numpy.typing as npt
from typing import Optional

try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False

from ..config import TopologyConfig


def compute_persistence(
    point_cloud: npt.NDArray,
    max_dim: int = TopologyConfig.MAX_HOMOLOGY_DIM,
    metric: str = TopologyConfig.DISTANCE_METRIC,
    max_edge_length: Optional[float] = TopologyConfig.MAX_EDGE_LENGTH,
    coeff: int = TopologyConfig.COEFF,
) -> list[npt.NDArray]:
    """Compute Vietoris-Rips persistent homology on a point cloud.

    Args:
        point_cloud: Array of shape (n_points, n_dims). Rows are points.
        max_dim: Highest homological dimension to compute. 1 computes H₀ and H₁.
        metric: Distance metric ('euclidean' or 'precomputed').
            If 'precomputed', point_cloud must be a pairwise distance matrix.
        max_edge_length: Maximum filtration radius. None = auto (Ripser default).
        coeff: Prime coefficient field for homology computation.

    Returns:
        List of persistence diagrams, one per homological dimension.
        Each diagram is an ndarray of shape (n_features, 2) with columns [birth, death].
        Infinite death values are represented as np.inf.

    Raises:
        ImportError: If ripser is not installed.
    """
    if not RIPSER_AVAILABLE:
        raise ImportError("ripser is required for PH computation. pip install ripser")

    kwargs: dict = {
        "maxdim": max_dim,
        "metric": metric,
        "coeff": coeff,
    }
    if max_edge_length is not None:
        kwargs["thresh"] = max_edge_length

    result = ripser(point_cloud, **kwargs)
    return result["dgms"]  # list of (birth, death) arrays, indexed by dimension
