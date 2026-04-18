"""Persistence diagram vectorization.

Converts raw persistence diagrams (lists of birth/death pairs) into
fixed-size vectors suitable for scikit-learn classifiers.

Supported methods:
- persistence_image: Weighted, smoothed grid (giotto-tda or persim)
- landscape: Piecewise-linear functional summary (giotto-tda)
- statistics: Total persistence, max persistence, diagram entropy
"""

import collections
import collections.abc
from typing import Literal

import numpy as np
import numpy.typing as npt

from ..config import VectorizationConfig

try:
    from gtda.diagrams import PersistenceImage, PersistenceLandscape
    GIOTTO_AVAILABLE = True
except ImportError:
    GIOTTO_AVAILABLE = False

try:
    from persim import PersImage
    PERSIM_AVAILABLE = True
except ImportError:
    PERSIM_AVAILABLE = False


def vectorize_diagrams(
    diagrams: list[npt.NDArray],
    method: Literal["persistence_image", "landscape", "statistics"] = VectorizationConfig.METHOD,
    n_bins: int = VectorizationConfig.PI_N_BINS,
    sigma: float = VectorizationConfig.PI_SIGMA,
) -> npt.NDArray:
    """Convert a list of persistence diagrams into a single flat feature vector.

    Args:
        diagrams: List of persistence diagrams from compute_persistence().
            diagrams[k] has shape (n_features_k, 2) for homological dimension k.
        method: Vectorization strategy. 'persistence_image' is recommended
            for classification with SVMs.
        n_bins: Grid resolution for persistence images (n_bins × n_bins per diagram).
        sigma: Gaussian kernel bandwidth for persistence images.

    Returns:
        1-D feature vector concatenating vectorized representations of all
        homological dimensions. Shape: (n_dims_total,).

    Raises:
        ImportError: If the required library for the chosen method is unavailable.
        ValueError: If an unsupported method is specified.
    """
    if method == "persistence_image":
        return _persistence_image_vector(diagrams, n_bins=n_bins, sigma=sigma)
    elif method == "landscape":
        return _landscape_vector(diagrams)
    elif method == "statistics":
        return _summary_statistics_vector(diagrams)
    else:
        raise ValueError(f"Unknown vectorization method: {method!r}")


def _persistence_image_vector(
    diagrams: list[npt.NDArray],
    n_bins: int,
    sigma: float,
) -> npt.NDArray:
    """Vectorize diagrams via persistence images."""
    if not PERSIM_AVAILABLE and not GIOTTO_AVAILABLE:
        raise ImportError(
            "persim or giotto-tda is required for persistence images. "
            "pip install persim  OR  pip install giotto-tda"
        )

    if GIOTTO_AVAILABLE:
        return _giotto_persistence_image_vector(diagrams, n_bins=n_bins, sigma=sigma)

    vectors = []
    pim = PersImage(spread=sigma, pixels=(n_bins, n_bins), verbose=False)
    for dim, dgm in enumerate(diagrams):
        weight = _homology_weight(dim)
        finite = dgm[~np.isinf(dgm[:, 1])]
        if len(finite) == 0:
            vectors.append(np.zeros(n_bins * n_bins))
        else:
            # persim 0.3.0 still references collections.Iterable, which moved
            # to collections.abc in modern Python.
            if not hasattr(collections, "Iterable"):
                collections.Iterable = collections.abc.Iterable
            img = pim.transform(finite)
            vectors.append(img.flatten() * weight)
    return np.concatenate(vectors)


def _giotto_persistence_image_vector(
    diagrams: list[npt.NDArray],
    n_bins: int,
    sigma: float,
) -> npt.NDArray:
    """Vectorize diagrams with giotto-tda, keeping one block per homology dimension."""
    vectors = []
    for dim, dgm in enumerate(diagrams):
        weight = _homology_weight(dim)
        finite = dgm[~np.isinf(dgm[:, 1])]
        if len(finite) == 0:
            vectors.append(np.zeros(n_bins * n_bins))
            continue

        batch = np.column_stack([finite, np.zeros(len(finite))])[np.newaxis]
        pi = PersistenceImage(sigma=sigma, n_bins=n_bins)
        img = pi.fit_transform(batch)
        vectors.append(img.flatten() * weight)
    return np.concatenate(vectors)


def _landscape_vector(diagrams: list[npt.NDArray]) -> npt.NDArray:
    """Vectorize diagrams via persistence landscapes (giotto-tda)."""
    if not GIOTTO_AVAILABLE:
        raise ImportError("giotto-tda is required for persistence landscapes. pip install giotto-tda")
    # giotto-tda expects a batch dimension; wrap and unwrap
    cfg = VectorizationConfig
    vectors = []
    for dim, dgm in enumerate(diagrams):
        weight = _homology_weight(dim)
        finite = dgm[~np.isinf(dgm[:, 1])]
        if len(finite) == 0:
            vectors.append(np.zeros(cfg.LANDSCAPE_N_LAYERS * cfg.LANDSCAPE_N_BINS))
        else:
            pl = PersistenceLandscape(
                n_layers=cfg.LANDSCAPE_N_LAYERS,
                n_bins=cfg.LANDSCAPE_N_BINS,
            )
            # giotto expects shape (n_samples, n_points, 3) with homology_dimension col
            batch = np.column_stack([finite, np.zeros(len(finite))])[np.newaxis]
            result = pl.fit_transform(batch)
            vectors.append(result.flatten() * weight)
    return np.concatenate(vectors)


def _summary_statistics_vector(diagrams: list[npt.NDArray]) -> npt.NDArray:
    """Compute summary statistics per homological dimension.

    Returns: [total_persistence_H0, max_persistence_H0, entropy_H0,
               total_persistence_H1, max_persistence_H1, entropy_H1, ...]
    """
    stats = []
    for dim, dgm in enumerate(diagrams):
        weight = _homology_weight(dim)
        finite = dgm[~np.isinf(dgm[:, 1])]
        if len(finite) == 0:
            stats.extend([0.0, 0.0, 0.0])
            continue
        persistences = finite[:, 1] - finite[:, 0]
        total = float(np.sum(persistences))
        maximum = float(np.max(persistences))
        # Normalized entropy
        p = persistences / (total + 1e-12)
        entropy = float(-np.sum(p * np.log(p + 1e-12)))
        stats.extend([total * weight, maximum * weight, entropy * weight])
    return np.array(stats)


def _homology_weight(dim: int) -> float:
    """Return scaling weight for a homology dimension."""
    weights = VectorizationConfig.HOMOLOGY_WEIGHTS
    if not weights:
        return 1.0
    if dim >= len(weights):
        return 1.0
    return float(weights[dim])
