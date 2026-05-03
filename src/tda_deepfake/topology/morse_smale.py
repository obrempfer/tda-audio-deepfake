"""Morse-Smale features on 2-D scalar fields.

Preferred implementation uses topopy's approximate Morse-Smale complex. A
discrete local fallback remains available when topopy is unavailable.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.ndimage import maximum_filter, minimum_filter

from ..config import MorseSmaleConfig

try:
    import nglpy
    from topopy import MorseSmaleComplex

    TOPOPY_AVAILABLE = True
except ImportError:
    TOPOPY_AVAILABLE = False


def compute_morse_smale_signature(
    grid: npt.NDArray[np.float64],
    implementation: str = MorseSmaleConfig.IMPLEMENTATION,
    graph_max_neighbors: int = MorseSmaleConfig.GRAPH_MAX_NEIGHBORS,
    graph_relaxed: bool = MorseSmaleConfig.GRAPH_RELAXED,
    normalization: str | None = MorseSmaleConfig.NORMALIZATION,
    simplification: str = MorseSmaleConfig.SIMPLIFICATION,
    neighborhood_size: int = MorseSmaleConfig.NEIGHBORHOOD_SIZE,
    top_k_basins: int = MorseSmaleConfig.TOP_K_BASINS,
    include_extrema_values: bool = MorseSmaleConfig.INCLUDE_EXTREMA_VALUES,
    top_k_extrema: int = MorseSmaleConfig.TOP_K_EXTREMA,
    feature_subset: str = MorseSmaleConfig.FEATURE_SUBSET,
) -> npt.NDArray[np.float64]:
    """Return a fixed-length Morse-Smale or Morse-Smale-inspired signature."""
    if grid.ndim != 2:
        raise ValueError(f"Morse-Smale signature expects a 2-D grid, got {grid.shape}")
    if top_k_basins <= 0 or top_k_extrema <= 0:
        raise ValueError("top_k_basins and top_k_extrema must be positive")

    if implementation == "topopy":
        if not TOPOPY_AVAILABLE:
            raise ImportError("topopy and nglpy are required for implementation='topopy'")
        blocks = _compute_topopy_signature_blocks(
            grid,
            graph_max_neighbors=graph_max_neighbors,
            graph_relaxed=graph_relaxed,
            normalization=normalization,
            simplification=simplification,
            top_k_basins=top_k_basins,
            include_extrema_values=include_extrema_values,
            top_k_extrema=top_k_extrema,
        )
        return _select_signature_blocks(blocks, feature_subset)
    if implementation == "approx":
        blocks = _compute_approx_signature_blocks(
            grid,
            neighborhood_size=neighborhood_size,
            top_k_basins=top_k_basins,
            include_extrema_values=include_extrema_values,
            top_k_extrema=top_k_extrema,
        )
        return _select_signature_blocks(blocks, feature_subset)
    raise ValueError(f"Unknown Morse-Smale implementation: {implementation!r}")


def _compute_topopy_signature_blocks(
    grid: npt.NDArray[np.float64],
    graph_max_neighbors: int,
    graph_relaxed: bool,
    normalization: str | None,
    simplification: str,
    top_k_basins: int,
    include_extrema_values: bool,
    top_k_extrema: int,
) -> list[tuple[str, npt.NDArray[np.float64]]]:
    """Use topopy's MorseSmaleComplex to extract named signature blocks."""
    if graph_max_neighbors <= 0:
        raise ValueError("graph_max_neighbors must be positive")

    rows, cols = grid.shape
    coords = np.array([[r, c] for r in range(rows) for c in range(cols)], dtype=np.float64)
    values = grid.ravel().astype(np.float64)

    graph = nglpy.EmptyRegionGraph(max_neighbors=graph_max_neighbors, relaxed=graph_relaxed)
    msc = MorseSmaleComplex(
        graph=graph,
        gradient="steepest",
        normalization=normalization,
        simplification=simplification,
    )
    msc.build(coords, values)

    partitions = msc.get_partitions()
    stable = msc.get_stable_manifolds()
    unstable = msc.get_unstable_manifolds()
    merge_sequence = msc.get_merge_sequence()

    classifications = [_safe_classification(msc, idx) for idx in range(len(values))]
    maxima_values = np.sort(values[[i for i, c in enumerate(classifications) if c == "maximum"]])[::-1]
    minima_values = np.sort(values[[i for i, c in enumerate(classifications) if c == "minimum"]])

    counts_entropy = np.asarray(
        [
            float(len(partitions)),
            float(len(stable)),
            float(len(unstable)),
            float(sum(c == "maximum" for c in classifications)),
            float(sum(c == "minimum" for c in classifications)),
            float(np.mean(np.abs(np.gradient(grid)[0])) + np.mean(np.abs(np.gradient(grid)[1]))),
            _entropy_from_sizes([len(v) for v in partitions.values()]),
            _entropy_from_sizes([len(v) for v in stable.values()]),
            _entropy_from_sizes([len(v) for v in unstable.values()]),
        ],
        dtype=np.float64,
    )
    basin_fractions = np.concatenate(
        [
            _largest_fraction_vector([len(v) for v in partitions.values()], total=len(values), top_k=top_k_basins),
            _largest_fraction_vector([len(v) for v in stable.values()], total=len(values), top_k=top_k_basins),
            _largest_fraction_vector([len(v) for v in unstable.values()], total=len(values), top_k=top_k_basins),
        ]
    ).astype(np.float64)
    merge_block = np.asarray(
        _top_k_padded(sorted((triplet[0] for triplet in merge_sequence.values()), reverse=True), top_k_extrema),
        dtype=np.float64,
    )
    extrema_block = np.empty(0, dtype=np.float64)
    if include_extrema_values:
        extrema_block = np.asarray(
            [
                *_top_k_padded(maxima_values, top_k_extrema),
                *_top_k_padded(minima_values, top_k_extrema),
            ],
            dtype=np.float64,
        )

    return [
        ("counts_entropy", counts_entropy),
        ("basin_fractions", basin_fractions),
        ("merge_sequence", merge_block),
        ("extrema_values", extrema_block),
    ]


def _safe_classification(msc: "MorseSmaleComplex", idx: int) -> str:
    value = msc.get_classification(idx)
    if value is None:
        return "regular"
    return str(value)


def _compute_approx_signature_blocks(
    grid: npt.NDArray[np.float64],
    neighborhood_size: int,
    top_k_basins: int,
    include_extrema_values: bool,
    top_k_extrema: int,
) -> list[tuple[str, npt.NDArray[np.float64]]]:
    """Fallback local approximation from extrema and steepest-flow basins."""
    if neighborhood_size <= 0 or neighborhood_size % 2 == 0:
        raise ValueError("neighborhood_size must be a positive odd integer")

    maxima_mask, minima_mask = _local_extrema_masks(grid, neighborhood_size=neighborhood_size)
    ascent_labels = _assign_flow_basins(grid, mode="ascent")
    descent_labels = _assign_flow_basins(grid, mode="descent")

    gradient_y, gradient_x = np.gradient(grid)
    gradient_mag = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    maxima_values = np.sort(grid[maxima_mask])[::-1]
    minima_values = np.sort(grid[minima_mask])

    ascent_sizes = _largest_region_fractions(ascent_labels, top_k_basins)
    descent_sizes = _largest_region_fractions(descent_labels, top_k_basins)

    counts_entropy = np.asarray(
        [
            float(np.sum(maxima_mask)),
            float(np.sum(minima_mask)),
            float(np.mean(gradient_mag)),
            float(np.std(gradient_mag)),
            _entropy_from_labels(ascent_labels),
            _entropy_from_labels(descent_labels),
        ],
        dtype=np.float64,
    )
    basin_fractions = np.concatenate([ascent_sizes, descent_sizes]).astype(np.float64)
    merge_block = np.empty(0, dtype=np.float64)
    extrema_block = np.empty(0, dtype=np.float64)
    if include_extrema_values:
        extrema_block = np.asarray(
            [
                *_top_k_padded(maxima_values, top_k_extrema),
                *_top_k_padded(minima_values, top_k_extrema),
            ],
            dtype=np.float64,
        )

    return [
        ("counts_entropy", counts_entropy),
        ("basin_fractions", basin_fractions),
        ("merge_sequence", merge_block),
        ("extrema_values", extrema_block),
    ]


def _select_signature_blocks(
    blocks: list[tuple[str, npt.NDArray[np.float64]]],
    feature_subset: str,
) -> npt.NDArray[np.float64]:
    normalized = feature_subset.strip().lower()
    block_map = {name: values for name, values in blocks if values.size > 0}

    if normalized == "full":
        chosen = [values for _, values in blocks if values.size > 0]
    else:
        if normalized not in {"counts_entropy", "basin_fractions", "merge_sequence", "extrema_values"}:
            raise ValueError(f"Unknown Morse-Smale feature subset: {feature_subset!r}")
        chosen = [block_map.get(normalized, np.empty(0, dtype=np.float64))]

    if not chosen or all(values.size == 0 for values in chosen):
        raise ValueError(
            f"Morse-Smale feature subset {feature_subset!r} is empty for the current configuration"
        )
    return np.concatenate(chosen).astype(np.float64, copy=False)


def _local_extrema_masks(
    grid: npt.NDArray[np.float64],
    neighborhood_size: int,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    max_filtered = maximum_filter(grid, size=neighborhood_size, mode="nearest")
    min_filtered = minimum_filter(grid, size=neighborhood_size, mode="nearest")
    maxima_mask = grid >= max_filtered
    minima_mask = grid <= min_filtered
    return maxima_mask, minima_mask


def _assign_flow_basins(
    grid: npt.NDArray[np.float64],
    mode: str,
) -> npt.NDArray[np.int32]:
    rows, cols = grid.shape
    flat = grid.reshape(-1)
    labels = np.full(flat.shape[0], -1, dtype=np.int32)

    for idx in range(flat.shape[0]):
        if labels[idx] != -1:
            continue
        path = []
        current = idx

        while True:
            if labels[current] != -1:
                label = labels[current]
                break
            path.append(current)
            next_idx = _best_neighbor_index(grid, current, mode=mode)
            if next_idx == current:
                label = current
                break
            current = next_idx

        for visited in path:
            labels[visited] = label

    _, compressed = np.unique(labels, return_inverse=True)
    return compressed.reshape(rows, cols).astype(np.int32)


def _best_neighbor_index(
    grid: npt.NDArray[np.float64],
    flat_index: int,
    mode: str,
) -> int:
    rows, cols = grid.shape
    row, col = divmod(flat_index, cols)
    current_value = grid[row, col]
    best_index = flat_index
    best_delta = 0.0

    for d_row in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            if d_row == 0 and d_col == 0:
                continue
            n_row = row + d_row
            n_col = col + d_col
            if not (0 <= n_row < rows and 0 <= n_col < cols):
                continue
            neighbor_value = grid[n_row, n_col]
            delta = neighbor_value - current_value
            if mode == "ascent":
                if delta > best_delta:
                    best_delta = delta
                    best_index = n_row * cols + n_col
            elif mode == "descent":
                if -delta > best_delta:
                    best_delta = -delta
                    best_index = n_row * cols + n_col
            else:
                raise ValueError(f"Unknown flow mode: {mode!r}")

    return best_index


def _largest_region_fractions(labels: npt.NDArray[np.int32], top_k: int) -> npt.NDArray[np.float64]:
    counts = np.bincount(labels.ravel())
    fractions = np.sort(counts / np.sum(counts))[::-1]
    padded = np.zeros(top_k, dtype=np.float64)
    padded[: min(top_k, len(fractions))] = fractions[:top_k]
    return padded


def _entropy_from_labels(labels: npt.NDArray[np.int32]) -> float:
    counts = np.bincount(labels.ravel()).astype(np.float64)
    probs = counts / np.sum(counts)
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _entropy_from_sizes(sizes: list[int]) -> float:
    if not sizes:
        return 0.0
    counts = np.asarray(sizes, dtype=np.float64)
    probs = counts / np.sum(counts)
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _largest_fraction_vector(sizes: list[int], total: int, top_k: int) -> list[float]:
    if not sizes or total <= 0:
        return [0.0] * top_k
    fractions = sorted((size / total for size in sizes), reverse=True)
    return _top_k_padded(fractions, top_k)


def _top_k_padded(values, k: int) -> list[float]:
    array = np.asarray(list(values), dtype=np.float64)
    padded = np.zeros(k, dtype=np.float64)
    padded[: min(k, len(array))] = array[:k]
    return padded.tolist()
