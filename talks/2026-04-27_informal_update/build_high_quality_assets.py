#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import librosa


ROOT = Path(__file__).resolve().parents[2]
TALK_DIR = ROOT / "talks" / "2026-04-27_informal_update"
ASSET_DIR = TALK_DIR / "assets"
sys.path.insert(0, str(ROOT / "src"))

from tda_deepfake.config import (
    SpectrogramConfig,
    apply_runtime_config,
    export_runtime_config,
    load_config_from_yaml,
)
from tda_deepfake.features import build_raw_mel_spectrogram, postprocess_mel_spectrogram
from tda_deepfake.utils import load_audio

BG = "#f6f1e8"
PANEL = "#fffaf2"
INK = "#13213c"
MUTED = "#5c6b82"
GRID = "#d8cfbf"
BLUE = "#2f7abf"
ORANGE = "#d96a2d"
TEAL = "#2d9d88"
RED = "#ba4454"
GRAY = "#8b98ad"
GOLD = "#d6a441"
GREEN = "#4f9d69"
MAGENTA = "#9a3ea5"


def ensure_dirs() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)


def style_axis(ax: plt.Axes, *, xgrid: bool = False, ygrid: bool = True) -> None:
    ax.set_facecolor(PANEL)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#bdae97")
    ax.spines["bottom"].set_color("#bdae97")
    ax.tick_params(colors=INK, labelsize=10)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.85)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.85)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.24, facecolor=fig.get_facecolor())
    plt.close(fig)


def add_fig_title(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.text(0.035, 0.972, title, fontsize=19, fontweight="bold", color=INK, ha="left", va="top")
    if subtitle:
        fig.text(0.035, 0.928, subtitle, fontsize=10.5, color=MUTED, ha="left", va="top")


def add_badge(ax: plt.Axes, x: float, y: float, text: str, *, color: str, text_color: str = "white") -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=9.5,
        color=text_color,
        bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.2", facecolor=color, edgecolor="none"),
    )


def add_card(ax: plt.Axes, x: float, y: float, w: float, h: float, *, edge: str) -> None:
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=2.2,
        edgecolor=edge,
        facecolor=PANEL,
        path_effects=[pe.withSimplePatchShadow(offset=(2, -2), alpha=0.10)],
    )
    ax.add_patch(card)


def load_demo_cases() -> dict[str, dict]:
    path = ROOT / "docs" / "assets" / "sample_explanation_demo_20260425" / "sample_explanation_demo.json"
    data = json.loads(path.read_text())
    return {case["case_key"]: case for case in data["cases"]}


def get_2019_bonafide_sample() -> tuple[str, Path]:
    protocol = ROOT / "data" / "raw" / "ASVspoof2019_LA" / "ASVspoof2019.LA.cm.dev.trl.txt"
    audio_dir = ROOT / "data" / "raw" / "ASVspoof2019_LA" / "ASVspoof2019_LA_dev" / "flac"
    if not protocol.exists():
        alt_root = Path("/home/obrempfer/Git/obrempfer/tda-audio-deepfake")
        protocol = alt_root / "data" / "raw" / "ASVspoof2019_LA" / "ASVspoof2019.LA.cm.dev.trl.txt"
        audio_dir = alt_root / "data" / "raw" / "ASVspoof2019_LA" / "ASVspoof2019_LA_dev" / "flac"
    with protocol.open() as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[-1].lower() != "bonafide":
                continue
            utt = parts[1]
            path = audio_dir / f"{utt}.flac"
            if path.exists():
                return utt, path
    raise FileNotFoundError("Could not find a 2019 bonafide sample for the presentation assets.")


def resolve_repo_audio_path(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        return path

    parts = list(path.parts)
    try:
        data_idx = parts.index("data")
    except ValueError:
        return path

    candidate = ROOT.joinpath(*parts[data_idx:])
    if candidate.exists():
        return candidate
    return path


def compute_fields(audio_path: Path, *, config_name: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    audio_path = resolve_repo_audio_path(audio_path)
    audio = load_audio(audio_path)
    raw = build_raw_mel_spectrogram(audio)
    raw_db = librosa.power_to_db(raw, ref=np.max)

    snapshot = export_runtime_config()
    try:
        if config_name is not None:
            load_config_from_yaml(str(config_name))
        processed = postprocess_mel_spectrogram(
            raw.copy(),
            log_scale=SpectrogramConfig.LOG_SCALE,
            compression=SpectrogramConfig.COMPRESSION,
            smoothing=SpectrogramConfig.SMOOTHING,
            smoothing_sigma=SpectrogramConfig.SMOOTHING_SIGMA,
            smoothing_axis=SpectrogramConfig.SMOOTHING_AXIS,
            band_mask_mode=SpectrogramConfig.BAND_MASK_MODE,
            band_split_low=SpectrogramConfig.BAND_SPLIT_LOW,
            band_split_high=SpectrogramConfig.BAND_SPLIT_HIGH,
            band_mask_fill=SpectrogramConfig.BAND_MASK_FILL,
            temporal_field_mode=SpectrogramConfig.TEMPORAL_FIELD_MODE,
            temporal_field_sigma=SpectrogramConfig.TEMPORAL_FIELD_SIGMA,
            energy_weighting_mode=SpectrogramConfig.ENERGY_WEIGHTING_MODE,
            energy_weighting_gamma=SpectrogramConfig.ENERGY_WEIGHTING_GAMMA,
            energy_gate_percentile=SpectrogramConfig.ENERGY_GATE_PERCENTILE,
            energy_gate_fill=SpectrogramConfig.ENERGY_GATE_FILL,
            normalize=SpectrogramConfig.NORMALIZE,
            normalization_method=SpectrogramConfig.NORMALIZATION_METHOD,
            max_frames=SpectrogramConfig.MAX_FRAMES,
        )
    finally:
        apply_runtime_config(snapshot)
    return raw_db, processed


def draw_spec(
    ax: plt.Axes,
    grid: np.ndarray,
    *,
    title: str,
    subtitle: str | None = None,
    show_band_split: bool = False,
    cmap: str = "magma",
) -> None:
    ax.set_facecolor(PANEL)
    ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color("#bdae97")
        side.set_linewidth(1.2)
    ax.set_title(title, fontsize=12.5, color=INK, fontweight="bold", pad=8)
    if subtitle:
        ax.text(0.01, 1.02, subtitle, transform=ax.transAxes, fontsize=9.2, color=MUTED, ha="left", va="bottom")
    if show_band_split:
        cutoff = int(np.floor(grid.shape[0] * 0.33))
        ax.axhline(cutoff - 0.5, color="#f2ede4", lw=2.0, ls="--")
        ax.text(
            0.98,
            (cutoff / grid.shape[0]) + 0.015,
            "low-band cutoff",
            transform=ax.transAxes,
            fontsize=8.6,
            color="white",
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=MAGENTA, edgecolor="none", alpha=0.92),
        )

def build_2019_ablation() -> None:
    rows = [
        ("keep_low", 0.075, 0.974, ORANGE),
        ("keep_low_h1", 0.106, 0.956, TEAL),
        ("reference", 0.124, 0.946, BLUE),
        ("keep_low_h0", 0.193, 0.877, RED),
        ("drop_low", 0.265, 0.820, GRAY),
    ]

    fig = plt.figure(figsize=(14, 5.8), facecolor=BG)
    add_fig_title(fig, "2019 LA structural ablations", "Low-band topology wins, H1 is robust, and removing the low band is the main failure mode.")
    ax = fig.add_axes([0.08, 0.23, 0.56, 0.58])
    info = fig.add_axes([0.69, 0.23, 0.23, 0.58], facecolor=PANEL)
    style_axis(ax, xgrid=True, ygrid=False)
    info.axis("off")

    y = np.arange(len(rows))[::-1]
    ax.set_xlim(0, 0.35)
    ax.set_ylim(-0.8, len(rows) - 0.0)
    ax.set_yticks([])
    
    for yi, (label, eer, auc, color) in zip(y, rows):
        ax.hlines(yi, 0, eer, color=color, lw=7, alpha=0.22, capstyle="round")
        ax.scatter([eer], [yi], s=180, color=color, edgecolors="white", linewidths=1.5, zorder=4)
        ax.text(-0.004, yi, label, ha="right", va="center", fontsize=13, color=INK, fontweight="bold")
        ax.text(min(eer + 0.009, 0.277), yi + 0.08, f"EER {eer:.3f}", fontsize=12, color=INK, va="center")
        ypos = (yi + 0.5) / len(rows)
        #info.scatter([0.04], [ypos], s=70, color=color, transform=info.transAxes)
        #info.text(0.10, ypos, label, fontsize=10.0, color=MUTED, va="center", transform=info.transAxes)
        info.text(0.10, ypos, f"AUC {auc:.3f}", fontsize=15, color=INK if label == "keep_low" else MUTED, va="center", transform=info.transAxes)
        if label == "keep_low":
            info.text(0.10, ypos - 0.08, "best overall", fontsize=10.6, color="white", va="center", transform=info.transAxes, bbox=dict(boxstyle="round,pad=0.30", facecolor=ORANGE, edgecolor="none"))
        elif label == "drop_low":
            info.text(0.10, ypos - 0.10, "largest collapse", fontsize=10.6, color=MUTED, va="center", transform=info.transAxes)
    
    #info.text(0.02, 0.97, "AUC by branch", fontsize=12.0, color=INK, fontweight="bold", va="top", transform=info.transAxes)
    fig.text(0.36, 0.16, "H2-only was later tested separately and collapsed to chance: AUC 0.500, EER 0.500.", fontsize=10.3, color=MUTED, ha="center")

    savefig(fig, ASSET_DIR / "ablation_2019.png")


def build_multi_dataset() -> None:
    rows = [
        ("2019 LA in-domain", 0.1025, 0.0896, "keep_low"),
        ("2021 LA transfer", 0.2277, 0.2100, "keep_low_h1  (C=2)"),
        ("2021 LA internal", 0.1222, 0.1096, "keep_low / gate10"),
        ("2021 DF bounded", 0.2844, 0.2658, "keep_low + gate_off"),
    ]

    fig = plt.figure(figsize=(13.0, 5.8), facecolor=BG)
    add_fig_title(fig, "Multi-dataset comparison", "The low-band family keeps winning, but the exact branch shifts with the domain.")
    ax = fig.add_axes([0.08, 0.22, 0.62, 0.59])
    info = fig.add_axes([0.73, 0.22, 0.19, 0.59], facecolor=PANEL)
    style_axis(ax, xgrid=True, ygrid=False)
    info.axis("off")
    ax.set_xlim(0.082, 0.300)
    ax.set_xlabel("EER  (lower is better)")
    ax.set_yticks([])
    ax.set_ylim(-0.8, len(rows) - 0.2)

    y = np.arange(len(rows))[::-1]
    for yi, (label, ref_eer, best_eer, branch) in zip(y, rows):
        ax.plot([best_eer, ref_eer], [yi, yi], color="#c9bba5", lw=5, solid_capstyle="round", zorder=1)
        ax.scatter([ref_eer], [yi], s=160, color=BLUE, edgecolors="white", linewidths=1.4, zorder=3)
        ax.scatter([best_eer], [yi], s=160, color=ORANGE, edgecolors="white", linewidths=1.4, zorder=4)
        ax.text(0.080, yi, label, ha="right", va="center", fontsize=13, color=INK, fontweight="bold")
        ax.text(ref_eer + 0.003, yi - 0.25, f"{ref_eer:.3f}", fontsize=11.0, color=INK, ha="center", va="center_baseline")
        ax.text(best_eer - 0, yi - 0.25, f"{best_eer:.3f}", fontsize=11.0, color=INK, ha="center", va="center_baseline")
        delta = ref_eer - best_eer
        ax.text((ref_eer + best_eer) / 2, yi + 0.19, f"ΔEER {delta:.3f}", fontsize=9.8, color=MUTED, ha="center")
        info.text(0.03, (yi + 0.5) / len(rows), branch, fontsize=11.0, color="white", va="center", bbox=dict(boxstyle="round,pad=0.35", facecolor=TEAL, edgecolor="none"))

    info.text(0.03, 0.97, "Winning branch", fontsize=12.0, color=INK, fontweight="bold", va="top")

    savefig(fig, ASSET_DIR / "multi_dataset_eer.png")


def build_topology_nn() -> None:
    datasets = ["2019 eval", "2021 LA transfer", "2021 DF transfer"]
    linear = np.array([0.1289, 0.2573, 0.3024])
    flat = np.array([0.0844, 0.2299, 0.2746])
    staged = np.array([0.0784, 0.2240, 0.2730])

    heatmap = np.array([
        [0.0095, 0.0112, 0.1715],
        [0.0059, 0.0123, 0.0336],
        [0.0430, 0.0087, 0.0009],
    ])
    row_labels = ["linear", "flat MLP", "staged MLP"]
    col_labels = ["remove H1 core", "remove H0 aux", "remove full aux"]

    fig = plt.figure(figsize=(13.0, 5.9), facecolor=BG)
    add_fig_title(fig, "Topology-only neural models", "Neural heads help over the linear baseline; staged training helps the model rely more on the H1 core.")

    ax1 = fig.add_axes([0.06, 0.16, 0.45, 0.62])
    style_axis(ax1)
    x = np.arange(len(datasets))
    w = 0.22
    ax1.bar(x - w, linear, width=w, color=BLUE, label="linear")
    ax1.bar(x, flat, width=w, color=ORANGE, label="flat MLP")
    ax1.bar(x + w, staged, width=w, color=TEAL, label="staged MLP")
    ax1.set_ylabel("EER")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.set_ylim(0, 0.35)
    ax1.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, fontsize=10.5)
    ax1.set_title("Performance summary", fontsize=13.5, fontweight="bold", pad=33)
    for series_x, series in ((x - w, linear), (x, flat), (x + w, staged)):
        for xpos, val in zip(series_x, series):
            ax1.text(xpos, val + 0.008, f"{val:.3f}", ha="center", fontsize=10.5, color=INK)

    ax2 = fig.add_axes([0.58, 0.23, 0.35, 0.50])
    ax2.set_facecolor(PANEL)
    cmap = LinearSegmentedColormap.from_list("coreheat", ["#fdf5eb", "#f0c38c", "#d96a2d", "#8f2736"])
    im = ax2.imshow(heatmap, cmap=cmap, vmin=0.0, vmax=0.18, aspect="auto")
    ax2.set_xticks(np.arange(len(col_labels)))
    ax2.set_xticklabels(col_labels, rotation=14, ha="right", fontsize=10, color=INK)
    ax2.set_yticks(np.arange(len(row_labels)))
    ax2.set_yticklabels(row_labels, fontsize=11, color=INK)
    ax2.set_title("2021 LA transfer: AUC drop when block is removed", fontsize=12.8, fontweight="bold", color=INK, pad=14)
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            val = heatmap[i, j]
            text_color = "white" if val > 0.08 else INK
            ax2.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=11, color=text_color, fontweight="bold")
    for spine in ax2.spines.values():
        spine.set_color("#bdae97")
    cax = fig.add_axes([0.94, 0.23, 0.015, 0.50])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=9, colors=INK)
    cb.outline.set_edgecolor("#bdae97")

    savefig(fig, ASSET_DIR / "topology_nn_results.png")


def build_case_study() -> None:
    scores = [
        ("full_reference", 0.971, RED),
        ("drop_low", 0.965, RED),
        ("gate_off", 0.069, TEAL),
        ("keep_low", 0.090, TEAL),
        ("keep_low_h1", 0.018, TEAL),
    ]

    fig = plt.figure(figsize=(13.0, 5.6), facecolor=BG)
    add_fig_title(fig, "Sample-level explanation", "2021 LA bona fide example: the broad field says fake, but the low-band/H1 family fixes the decision.")

    card = fig.add_axes([0.06, 0.66, 0.88, 0.16], facecolor=BG)
    card.axis("off")
    box = FancyBboxPatch((0.0, 0.05), 0.98, 0.80, boxstyle="round,pad=0.02,rounding_size=0.03", linewidth=1.6, edgecolor="#d9ccb8", facecolor=PANEL)
    card.add_patch(box)
    card.text(0.03, 0.58, "Sample: LA_E_9616893", fontsize=13.5, fontweight="bold", color=INK, va="center")
    card.text(0.03, 0.30, "True label: bona fide", fontsize=11.5, color=MUTED, va="center")
    card.text(0.43, 0.58, "full_reference → 0.971", fontsize=13.0, color=RED, fontweight="bold", va="center", ha="center")
    card.text(0.43, 0.30, "keep_low_h1 → 0.018", fontsize=13.0, color=TEAL, fontweight="bold", va="center", ha="center")
    card.text(0.71, 0.44, "Decision flip:\nlow-band/H1 moves the sample back\nto the correct side.", fontsize=10.8, color=INK, va="center")

    ax3 = fig.add_axes([0.10, 0.13, 0.76, 0.42])
    style_axis(ax3, xgrid=True, ygrid=False)
    ax3.set_xlim(0, 1.07)
    ax3.set_ylim(-0.5, len(scores) - 0.5)
    ax3.set_xlabel("Predicted fake probability")
    ax3.set_yticks(np.arange(len(scores)))
    ax3.set_yticklabels([s[0] for s in scores], fontsize=11.5, color=INK)
    ax3.axvline(0.5, color=MUTED, linestyle="--", lw=1.4)
    ax3.text(0.505, 0.97, "decision boundary", color=MUTED, fontsize=10, ha="left", va="top", transform=ax3.get_xaxis_transform())

    for i, (name, value, color) in enumerate(scores):
        ax3.barh(i, value, color=color, alpha=0.92)
        x_text = value + 0.015 if value < 0.90 else value + 0.010
        ax3.text(x_text, i, f"{value:.3f}", va="center", ha="left", fontsize=11.5, color=INK, fontweight="bold")
        verdict = "wrong" if value > 0.5 else "correct"
        verdict_color = RED if verdict == "wrong" else GREEN
        add_badge(ax3, 0.93, i, verdict, color=verdict_color)

    ax3.invert_yaxis()

    savefig(fig, ASSET_DIR / "case_2021_bonafide.png")


def build_mel_examples() -> None:
    cases = load_demo_cases()
    bonafide_utt, bonafide_path = get_2019_bonafide_sample()
    spoof_utt = cases["case_2019_fake"]["sample"]["utt_id"]
    spoof_path = Path(cases["case_2019_fake"]["sample"]["audio_path"])
    raw_bf, ref_bf = compute_fields(bonafide_path, config_name=ROOT / "configs" / "experiments" / "cubical_mel_best_field_svm.yaml")
    raw_sp, ref_sp = compute_fields(spoof_path, config_name=ROOT / "configs" / "experiments" / "cubical_mel_best_field_svm.yaml")

    fig = plt.figure(figsize=(13.8, 7.4), facecolor=BG)
    add_fig_title(
        fig,
        "What the spectrogram fields look like",
        "Top row: raw mel spectrograms. Bottom row: the reference field after dB compression, energy gating, and smoothing.",
    )

    panels = [
        ("2019 bona fide · raw mel", raw_bf, f"{bonafide_utt}  ·  bona fide"),
        ("2019 spoof · raw mel", raw_sp, f"{spoof_utt}  ·  spoof"),
        ("2019 bona fide · reference field", ref_bf, "best full-field preprocessing"),
        ("2019 spoof · reference field", ref_sp, "best full-field preprocessing"),
    ]
    positions = [
        [0.05, 0.54, 0.42, 0.28],
        [0.53, 0.54, 0.42, 0.28],
        [0.05, 0.17, 0.42, 0.28],
        [0.53, 0.17, 0.42, 0.28],
    ]

    for (title, grid, subtitle), pos in zip(panels, positions):
        ax = fig.add_axes(pos)
        draw_spec(ax, grid, title=title, subtitle=subtitle, show_band_split=True)

    note = fig.add_axes([0.05, 0.04, 0.90, 0.05], facecolor=BG)
    note.axis("off")
    note.text(
        0.00,
        0.55,
        "Teaching point: the visual difference is not supposed to be obvious to the eye. "
        "The point is that the classifier does not operate on pixels directly; it summarizes stable topological structure in these fields.",
        fontsize=10.4,
        color=MUTED,
        ha="left",
        va="center",
    )
    savefig(fig, ASSET_DIR / "mel_spectrogram_examples.png")


def build_mel_ablation_walkthrough() -> None:
    cases = load_demo_cases()
    sample = cases["case_2019_fake"]["sample"]
    audio_path = Path(sample["audio_path"])

    reference_cfg = ROOT / "configs" / "experiments" / "cubical_mel_best_field_svm.yaml"
    keep_low_cfg = ROOT / "configs" / "experiments" / "ablation" / "cubical_best_band_keep_low.yaml"
    drop_low_cfg = ROOT / "configs" / "experiments" / "ablation" / "cubical_best_band_drop_low.yaml"
    gate_off_cfg = ROOT / "configs" / "experiments" / "ablation" / "cubical_best_band_keep_low_gate_off.yaml"

    raw_db, reference = compute_fields(audio_path, config_name=reference_cfg)
    _, keep_low = compute_fields(audio_path, config_name=keep_low_cfg)
    _, drop_low = compute_fields(audio_path, config_name=drop_low_cfg)
    _, gate_off = compute_fields(audio_path, config_name=gate_off_cfg)

    fig = plt.figure(figsize=(14.0, 7.2), facecolor=BG)
    add_fig_title(
        fig,
        "How the field changes before topology",
        "Same 2019 LA spoof sample, transformed by the exact field settings used in the ablation study.",
    )

    panels = [
        ("Raw mel (dB)", raw_db, "human-readable field"),
        ("Reference field", reference, "gate10, full band"),
        ("keep_low", keep_low, "middle/high bands zeroed"),
        ("drop_low", drop_low, "low band removed"),
        ("keep_low + gate_off", gate_off, "same band mask, no energy gate"),
    ]
    positions = [
        [0.04, 0.55, 0.28, 0.24],
        [0.36, 0.55, 0.28, 0.24],
        [0.68, 0.55, 0.28, 0.24],
        [0.20, 0.18, 0.28, 0.24],
        [0.52, 0.18, 0.28, 0.24],
    ]

    for (title, grid, subtitle), pos in zip(panels, positions):
        ax = fig.add_axes(pos)
        draw_spec(ax, grid, title=title, subtitle=subtitle, show_band_split=True)

    note = fig.add_axes([0.06, 0.03, 0.88, 0.08], facecolor=BG)
    note.axis("off")
    note.text(
        0.00,
        0.72,
        "Why this helps in the talk:",
        fontsize=11.6,
        color=INK,
        fontweight="bold",
        ha="left",
        va="center",
    )
    note.text(
        0.00,
        0.28,
        "You can point to the lower third directly: the successful branch keeps this region, "
        "while the failing ablation deletes it. That makes the low-band story visible before you ever mention AUC or EER.",
        fontsize=10.4,
        color=MUTED,
        ha="left",
        va="center",
    )
    savefig(fig, ASSET_DIR / "mel_ablation_walkthrough.png")


def main() -> None:
    ensure_dirs()
    build_2019_ablation()
    build_multi_dataset()
    build_topology_nn()
    build_case_study()
    build_mel_examples()
    build_mel_ablation_walkthrough()
    print(f"Wrote refreshed assets to {ASSET_DIR}")


if __name__ == "__main__":
    main()
