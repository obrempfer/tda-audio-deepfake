"""Persistent homology computation for point-cloud and grid representations."""

import numpy as np
import numpy.typing as npt
from typing import Optional
from sklearn.neighbors import NearestNeighbors

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
    knn_k: int = TopologyConfig.KNN_K,
    knn_graph_mode: str = TopologyConfig.KNN_GRAPH_MODE,
    max_edge_length: Optional[float] = TopologyConfig.MAX_EDGE_LENGTH,
    coeff: int = TopologyConfig.COEFF,
) -> list[npt.NDArray]:
    """Compute persistent homology on either a point cloud or a 2-D grid.

    Args:
        topological_object: Point cloud `(n_points, n_dims)` for Vietoris-Rips
            or kNN flag persistence, or 2-D grid `(n_rows, n_cols)` for cubical persistence.
        complex_type: Complex family ('vietoris_rips', 'cubical', or 'knn_flag').
        max_dim: Highest homological dimension to compute. 1 computes H₀ and H₁.
        metric: Distance metric for Vietoris-Rips ('euclidean' or 'precomputed').
            If 'precomputed', topological_object must be a pairwise distance matrix.
        cubical_filtration: Cubical filtration polarity ('sublevel' or 'superlevel').
        knn_k: Number of nearest neighbors to retain in the graph before clique expansion.
        knn_graph_mode: Graph symmetrization mode ('union' or 'mutual').
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
    if complex_type in {"knn_flag", "knn_clique", "flag", "clique"}:
        return _compute_knn_flag_persistence(
            topological_object,
            max_dim=max_dim,
            metric=metric,
            knn_k=knn_k,
            graph_mode=knn_graph_mode,
            max_edge_length=max_edge_length,
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


def _compute_knn_flag_persistence(
    point_cloud: npt.NDArray,
    max_dim: int,
    metric: str,
    knn_k: int,
    graph_mode: str,
    max_edge_length: Optional[float],
    coeff: int,
) -> list[npt.NDArray]:
    """Compute persistent homology on a weighted kNN flag/clique complex."""
    if not GUDHI_AVAILABLE:
        raise ImportError("gudhi is required for kNN flag PH. pip install gudhi")
    if point_cloud.ndim != 2:
        raise ValueError(f"kNN flag persistence expects a 2-D point cloud, got shape {point_cloud.shape}")
    if metric == "precomputed":
        raise ValueError("kNN flag persistence expects raw point coordinates, not a precomputed matrix")
    if knn_k <= 0:
        raise ValueError(f"knn_k must be positive, got {knn_k}")

    n_points = point_cloud.shape[0]
    simplex_tree = gudhi.SimplexTree()
    for i in range(n_points):
        simplex_tree.insert([i], filtration=0.0)

    if n_points <= 1:
        simplex_tree.persistence(homology_coeff_field=coeff, min_persistence=0)
        return _simplex_tree_diagrams(simplex_tree, max_dim=max_dim)

    k_eff = min(knn_k, n_points - 1)
    neighbors = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric)
    neighbors.fit(point_cloud)
    distances, indices = neighbors.kneighbors(point_cloud)

    directed = np.full((n_points, n_points), np.inf, dtype=np.float64)
    for i in range(n_points):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):
            directed[i, j] = float(dist)

    if graph_mode == "union":
        adjacency = np.minimum(directed, directed.T)
    elif graph_mode == "mutual":
        adjacency = np.where(np.isfinite(directed) & np.isfinite(directed.T), np.minimum(directed, directed.T), np.inf)
    else:
        raise ValueError(f"Unknown knn_graph_mode: {graph_mode!r}")

    for i in range(n_points):
        for j in range(i + 1, n_points):
            weight = adjacency[i, j]
            if not np.isfinite(weight):
                continue
            if max_edge_length is not None and weight > max_edge_length:
                continue
            simplex_tree.insert([i, j], filtration=float(weight))

    simplex_tree.expansion(max_dim + 1)
    simplex_tree.persistence(homology_coeff_field=coeff, min_persistence=0)
    return _simplex_tree_diagrams(simplex_tree, max_dim=max_dim)


def _simplex_tree_diagrams(simplex_tree: "gudhi.SimplexTree", max_dim: int) -> list[npt.NDArray]:
    """Extract fixed per-dimension diagrams from a Gudhi simplex tree."""
    diagrams: list[npt.NDArray] = []
    for dim in range(max_dim + 1):
        intervals = simplex_tree.persistence_intervals_in_dimension(dim)
        if intervals.size == 0:
            diagrams.append(np.zeros((0, 2), dtype=np.float64))
        else:
            diagrams.append(np.asarray(intervals, dtype=np.float64).reshape(-1, 2))
    return diagrams
