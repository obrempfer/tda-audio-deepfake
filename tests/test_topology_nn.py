"""Synthetic tests for topology-only neural models."""

import numpy as np

from tda_deepfake.neural import (
    StageSpec,
    TopologyLinearBaseline,
    TopologyMLP,
    evaluate_block_ablations,
    stack_feature_blocks,
)


def _synthetic_block_dataset(seed: int = 0, n_samples: int = 120):
    rng = np.random.default_rng(seed)
    core = rng.normal(size=(n_samples, 4))
    aux_a = rng.normal(scale=0.2, size=(n_samples, 3))
    aux_b = rng.normal(scale=0.2, size=(n_samples, 2))
    signal = core[:, 0] + 0.8 * core[:, 1] - 0.4 * aux_a[:, 0]
    y = (signal > 0.0).astype(int)
    X, layout = stack_feature_blocks(
        {
            "core_lowband_h1": core,
            "aux_lowband_h0": aux_a,
            "aux_fullfield_h0_h1": aux_b,
        }
    )
    return X, y, layout


def test_stack_feature_blocks_builds_expected_layout():
    X, _, layout = _synthetic_block_dataset()
    assert X.shape[1] == 9
    assert layout.block_names == [
        "core_lowband_h1",
        "aux_lowband_h0",
        "aux_fullfield_h0_h1",
    ]
    assert layout.get_block("core_lowband_h1").start == 0
    assert layout.get_block("core_lowband_h1").end == 4
    assert layout.mask_for_blocks(["core_lowband_h1"]).sum() == 4
    assert layout.mask_without_blocks(["aux_lowband_h0"]).sum() == 6


def test_linear_baseline_and_ablations_smoke():
    X, y, layout = _synthetic_block_dataset()
    model = TopologyLinearBaseline(c=1.0)
    model.fit(X, y)
    metrics = model.evaluate(X, y)
    ablations = evaluate_block_ablations(model, X, y, layout)

    assert metrics["auc"] >= 0.80
    assert "core_lowband_h1" in ablations
    assert "aux_lowband_h0" in ablations
    assert "aux_fullfield_h0_h1" in ablations


def test_staged_topology_mlp_smoke():
    X, y, layout = _synthetic_block_dataset(seed=7, n_samples=160)
    X_train, X_val = X[:120], X[120:]
    y_train, y_val = y[:120], y[120:]

    model = TopologyMLP(
        hidden_layer_sizes=(32, 16),
        feature_dropout=0.05,
        alpha=1e-4,
        batch_size=32,
        random_state=42,
    )
    stages = [
        StageSpec("stage1_core", ("core_lowband_h1",), 8, 1e-3),
        StageSpec("stage2_core_plus_aux_a", ("core_lowband_h1", "aux_lowband_h0"), 6, 5e-4),
        StageSpec(
            "stage3_core_plus_aux_a_plus_aux_b",
            ("core_lowband_h1", "aux_lowband_h0", "aux_fullfield_h0_h1"),
            4,
            2e-4,
        ),
    ]
    model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        layout=layout,
        stages=stages,
        monitor="auc",
        patience=3,
    )

    metrics = model.evaluate(X_val, y_val)
    ablations = evaluate_block_ablations(model, X_val, y_val, layout)

    assert metrics["auc"] >= 0.75
    assert len(model.training_history_) >= 1
    assert len(model.fitted_stages_) == 3
    assert "core_lowband_h1" in ablations
