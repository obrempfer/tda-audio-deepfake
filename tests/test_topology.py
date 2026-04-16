"""Tests for persistent homology computation and vectorization."""

import numpy as np
import pytest

from tda_deepfake.topology.persistent_homology import compute_persistence
from tda_deepfake.topology.morse_smale import compute_morse_smale_signature
from tda_deepfake.topology.vectorization import vectorize_diagrams, _summary_statistics_vector


def _random_point_cloud(n: int = 100, d: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d))


def _random_grid(rows: int = 16, cols: int = 24, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, cols))


def test_compute_persistence_returns_diagrams():
    cloud = _random_point_cloud()
    diagrams = compute_persistence(cloud, max_dim=1)
    assert isinstance(diagrams, list)
    assert len(diagrams) == 2  # H0 and H1


def test_persistence_diagram_shape():
    cloud = _random_point_cloud()
    diagrams = compute_persistence(cloud, max_dim=1)
    for dgm in diagrams:
        assert dgm.ndim == 2
        assert dgm.shape[1] == 2  # (birth, death) pairs


def test_vectorize_statistics():
    cloud = _random_point_cloud()
    diagrams = compute_persistence(cloud, max_dim=1)
    vec = vectorize_diagrams(diagrams, method="statistics")
    assert vec.ndim == 1
    # 3 statistics (total, max, entropy) per homological dimension
    assert len(vec) == len(diagrams) * 3


def test_vectorize_fixed_size():
    """Same-shaped clouds should produce same-sized feature vectors."""
    cloud1 = _random_point_cloud(seed=0)
    cloud2 = _random_point_cloud(seed=1)
    dgms1 = compute_persistence(cloud1)
    dgms2 = compute_persistence(cloud2)
    vec1 = vectorize_diagrams(dgms1, method="statistics")
    vec2 = vectorize_diagrams(dgms2, method="statistics")
    assert vec1.shape == vec2.shape


def test_compute_cubical_persistence_returns_diagrams():
    grid = _random_grid()
    diagrams = compute_persistence(grid, complex_type="cubical", max_dim=1)

    assert isinstance(diagrams, list)
    assert len(diagrams) == 2
    for dgm in diagrams:
        assert dgm.ndim == 2
        assert dgm.shape[1] == 2


def test_cubical_statistics_vectorization_fixed_size():
    grid1 = _random_grid(seed=0)
    grid2 = _random_grid(seed=1)
    dgms1 = compute_persistence(grid1, complex_type="cubical", max_dim=1)
    dgms2 = compute_persistence(grid2, complex_type="cubical", max_dim=1)

    vec1 = vectorize_diagrams(dgms1, method="statistics")
    vec2 = vectorize_diagrams(dgms2, method="statistics")

    assert vec1.shape == vec2.shape


def test_compute_knn_flag_persistence_returns_diagrams():
    cloud = _random_point_cloud(n=60, d=6)
    diagrams = compute_persistence(cloud, complex_type="knn_flag", max_dim=1, knn_k=8)

    assert isinstance(diagrams, list)
    assert len(diagrams) == 2
    for dgm in diagrams:
        assert dgm.ndim == 2
        assert dgm.shape[1] == 2


def test_knn_flag_statistics_vectorization_fixed_size():
    cloud1 = _random_point_cloud(seed=0)
    cloud2 = _random_point_cloud(seed=1)
    dgms1 = compute_persistence(cloud1, complex_type="knn_flag", max_dim=1, knn_k=10)
    dgms2 = compute_persistence(cloud2, complex_type="knn_flag", max_dim=1, knn_k=10)

    vec1 = vectorize_diagrams(dgms1, method="statistics")
    vec2 = vectorize_diagrams(dgms2, method="statistics")

    assert vec1.shape == vec2.shape


def test_compute_morse_smale_signature_fixed_length():
    grid1 = _random_grid(seed=0)
    grid2 = _random_grid(seed=1)

    vec1 = compute_morse_smale_signature(grid1, neighborhood_size=3, top_k_basins=4, top_k_extrema=4)
    vec2 = compute_morse_smale_signature(grid2, neighborhood_size=3, top_k_basins=4, top_k_extrema=4)

    assert vec1.ndim == 1
    assert vec1.shape == vec2.shape
    assert np.isfinite(vec1).all()
