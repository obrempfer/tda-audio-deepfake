"""Persistent homology computation for point-cloud and grid representations."""

import numpy as np
import numpy.typing as npt
from typing import Optional

try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False

from ..config import TopologyConfig


def compute_persistence(
    topological_object: npt.NDArray,
    complex_type: str = TopologyConfig.COMPLEX,
    max_dim: int = TopologyConfig.MAX_HOMOLOGY_DIM,
    metric: str = TopologyConfig.DISTANCE_METRIC,
    cubical_filtration: str = TopologyConfig.CUBICAL_FILTRATION,
    max_edge_length: Optional[float] = TopologyConfig.MAX_EDGE_LENGTH,
    coeff: int = TopologyConfig.COEFF,
) -> list[npt.NDArray]:
    """Compute persistent homology on either a point cloud or a 2-D grid.

    Args:
        topological_object: Point cloud `(n_points, n_dims)` for Vietoris-Rips
            or 2-D grid `(n_rows, n_cols)` for cubical persistence.
        complex_type: Complex family ('vietoris_rips' or 'cubical').
        max_dim: Highest homological dimension to compute. 1 computes H₀ and H₁.
        metric: Distance metric for Vietoris-Rips ('euclidean' or 'precomputed').
            If 'precomputed', topological_object must be a pairwise distance matrix.
        cubical_filtration: Cubical filtration polarity ('sublevel' or 'superlevel').
        max_edge_length: Maximum filtration radius for Vietoris-Rips.
        coeff: Prime coefficient field for homology computation.

    Returns:
        List of persistence diagrams, one per homological dimension.
        Each diagram is an ndarray of shape (n_features, 2) with columns [birth, death].
        Infinite death values are represented as np.inf.

    Raises:
        ImportError: If the required backend is not installed.
    """
    if complex_type == "vietoris_rips":
        return _compute_vietoris_rips_persistence(
            topological_object,
            max_dim=max_dim,
            metric=metric,
            max_edge_length=max_edge_length,
            coeff=coeff,
        )
    if complex_type == "cubical":
        return _compute_cubical_persistence(
            topological_object,
            max_dim=max_dim,
            filtration=cubical_filtration,
            coeff=coeff,
        )
    raise ValueError(f"Unknown complex_type: {complex_type!r}")


def _compute_vietoris_rips_persistence(
    point_cloud: npt.NDArray,
    max_dim: int,
    metric: str,
    max_edge_length: Optional[float],
    coeff: int,
) -> list[npt.NDArray]:
    """Compute Vietoris-Rips persistent homology on a point cloud."""
    if not RIPSER_AVAILABLE:
        raise ImportError("ripser is required for Vietoris-Rips PH. pip install ripser")

    kwargs: dict = {
        "maxdim": max_dim,
        "metric": metric,
        "coeff": coeff,
    }
    if max_edge_length is not None:
        kwargs["thresh"] = max_edge_length

    result = ripser(point_cloud, **kwargs)
    return result["dgms"]


def _compute_cubical_persistence(
    grid: npt.NDArray,
    max_dim: int,
    filtration: str,
    coeff: int,
) -> list[npt.NDArray]:
    """Compute cubical persistent homology on a 2-D grid."""
    if not GUDHI_AVAILABLE:
        raise ImportError("gudhi is required for cubical PH. pip install gudhi")
    if grid.ndim != 2:
        raise ValueError(f"cubical persistence expects a 2-D grid, got shape {grid.shape}")

    if filtration == "superlevel":
        filtered_grid = np.max(grid) - grid
    elif filtration == "sublevel":
        filtered_grid = grid
    else:
        raise ValueError(f"Unknown cubical filtration: {filtration!r}")

    complex_ = gudhi.CubicalComplex(top_dimensional_cells=np.asarray(filtered_grid, dtype=np.float64))
    complex_.persistence(homology_coeff_field=coeff, min_persistence=0)

    diagrams: list[npt.NDArray] = []
    for dim in range(max_dim + 1):
        intervals = complex_.persistence_intervals_in_dimension(dim)
        if intervals.size == 0:
            diagrams.append(np.zeros((0, 2), dtype=np.float64))
        else:
            diagrams.append(np.asarray(intervals, dtype=np.float64).reshape(-1, 2))
    return diagrams
