"""Dimensional ablation for topological anomaly explanation.

Procedure (per flagged sample):
1. Compute the full topological signature using all embedding dimensions.
2. For each named feature group, zero out those dimensions and recompute PH.
3. Measure how much the topological anomaly changes when the group is removed.
4. Report which removal causes the largest collapse in the anomaly signal.

This converts a binary "fake" label into a statement like:
"The anomaly is primarily driven by discontinuity in spectral transition dynamics
(ΔMFCC and Δ²MFCC dimensions), suggesting violations of articulatory smoothness."
"""

import numpy as np
import numpy.typing as npt
from typing import Optional
from scipy.spatial.distance import directed_hausdorff

from ..config import AblationConfig
from ..topology.persistent_homology import compute_persistence
from ..topology.vectorization import vectorize_diagrams


def _diagram_distance(dgms_a: list[npt.NDArray], dgms_b: list[npt.NDArray]) -> float:
    """Compute a simple L2 distance between vectorized persistence diagrams.

    Args:
        dgms_a: Persistence diagrams (list over homological dimensions).
        dgms_b: Persistence diagrams to compare against.

    Returns:
        Scalar distance between the two sets of diagrams.
    """
    vec_a = vectorize_diagrams(dgms_a, method="statistics")
    vec_b = vectorize_diagrams(dgms_b, method="statistics")
    return float(np.linalg.norm(vec_a - vec_b))


class AblationAnalyzer:
    """Runs dimensional ablation on flagged audio samples.

    Args:
        reference_diagrams: Persistence diagrams for a representative set of
            authentic speech samples (used as the comparison baseline).
        feature_groups: Dict mapping group name → list of dimension indices.
            Defaults to AblationConfig.FEATURE_GROUPS.
    """

    def __init__(
        self,
        reference_diagrams: list[npt.NDArray],
        feature_groups: Optional[dict] = None,
    ) -> None:
        self.reference_diagrams = reference_diagrams
        self.feature_groups = feature_groups or AblationConfig.FEATURE_GROUPS

    def analyze(self, point_cloud: npt.NDArray) -> dict:
        """Run ablation analysis on a single flagged point cloud.

        Args:
            point_cloud: Feature matrix of shape (n_points, n_dims) for the
                flagged sample.

        Returns:
            Dict with:
                - 'baseline_distance': anomaly score with all features.
                - 'ablation_scores': dict mapping group name → anomaly score
                  after removing that group.
                - 'most_implicated': name of the group whose removal causes
                  the largest reduction in anomaly score.
                - 'summary': human-readable explanation string.
        """
        # Baseline: full-feature anomaly score
        full_dgms = compute_persistence(point_cloud)
        baseline = _diagram_distance(full_dgms, self.reference_diagrams)

        ablation_scores: dict[str, float] = {}
        for group_name, dim_indices in self.feature_groups.items():
            ablated = point_cloud.copy()
            ablated[:, dim_indices] = 0.0
            ablated_dgms = compute_persistence(ablated)
            ablation_scores[group_name] = _diagram_distance(ablated_dgms, self.reference_diagrams)

        # Group whose removal most reduces the anomaly
        reductions = {g: baseline - score for g, score in ablation_scores.items()}
        most_implicated = max(reductions, key=lambda g: reductions[g])

        summary = _build_summary(baseline, ablation_scores, most_implicated)

        return {
            "baseline_distance": baseline,
            "ablation_scores": ablation_scores,
            "most_implicated": most_implicated,
            "summary": summary,
        }


def _build_summary(
    baseline: float,
    ablation_scores: dict[str, float],
    most_implicated: str,
) -> str:
    """Construct a human-readable ablation explanation.

    Args:
        baseline: Full-feature anomaly distance.
        ablation_scores: Per-group anomaly distances after ablation.
        most_implicated: Name of the most anomalous feature group.

    Returns:
        Explanation string describing which physical property is implicated.
    """
    _EXPLANATIONS = {
        "mfcc_static": "static spectral envelope (vocal tract configuration)",
        "mfcc_delta": "spectral transition rate (vocal tract movement dynamics)",
        "mfcc_delta2": "spectral transition acceleration (continuity / smoothness)",
        "f0": "fundamental frequency dynamics (laryngeal / pitch control)",
        "jitter_shimmer_hnr": "vocal fold micro-perturbations (cycle-to-cycle variability)",
        "formants": "formant frequencies and bandwidths (vowel / resonance structure)",
        "spectral_flux": "frame-to-frame spectral flux (articulatory inertia constraints)",
    }
    description = _EXPLANATIONS.get(most_implicated, most_implicated)
    reduced_score = ablation_scores[most_implicated]
    reduction_pct = 100.0 * (baseline - reduced_score) / (baseline + 1e-12)

    return (
        f"Topological anomaly score: {baseline:.4f}. "
        f"Ablation implicates '{most_implicated}' ({description}): "
        f"removing this group reduces the anomaly by {reduction_pct:.1f}% "
        f"(score → {reduced_score:.4f}). "
        f"This suggests the detection is primarily driven by anomalous {description}."
    )
