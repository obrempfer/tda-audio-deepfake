"""Tests for persistent homology computation and vectorization."""

import numpy as np
import pytest

from tda_deepfake.topology.persistent_homology import compute_persistence
from tda_deepfake.topology.vectorization import vectorize_diagrams, _summary_statistics_vector


def _random_point_cloud(n: int = 100, d: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d))


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
