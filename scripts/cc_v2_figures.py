"""Publication-style figures from frozen per-image measurements, not new fitting."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cc_v2_bootstrap import records

ROOT = Path("docs/benchmarks/cc_v2")
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
METHODS = [
    ("ccv2_sog_large_g0::combined", "Proposed: SoG residual, 3.03M", "#0072B2"),
    ("ccv2_direct_large_g0::combined", "Matched C+ with combined head, 3.03M", "#D55E00"),
    ("gw_ridge1::cheap", "GW + ridge, statistics baseline", "#009E73"),
    ("ccv2_legacy_proposed::upgraded_combined", "v1 mixture + upgraded error head", "#CC79A7"),
    ("ccv2_legacy_baseline::v1_shades_gray", "Classical SoG + learned selector", "#666666"),
]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "svg.fonttype": "none",
    }
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
for ax, domain, title in zip(
    axes,
    ("source_regression", "fresh_all"),
    (
        "SimpleCube++: 462 previously observed test images",
        "INTEL-TAU: 384 fresh images, cameras absent from fitting",
    ),
    strict=True,
):
    for group, label, color in METHODS:
        rr = records(ROOT, group, domain)
        y = np.array([r["selective"]["risk"] for r in rr])
        x = np.array(rr[0]["selective"]["coverage"]) * 100
        mean = y.mean(axis=0)
        ax.plot(x, mean, color=color, label=label, linewidth=2 if "Proposed" in label else 1.6)
        if len(rr) == 3:
            std = y.std(axis=0, ddof=1)
            ax.fill_between(
                x, np.maximum(0, mean - std), mean + std, alpha=0.10, color=color, linewidth=0
            )
    ax.axvline(80, linestyle=":", color="black", linewidth=1)
    ax.set(
        title=title,
        xlabel="Accepted coverage (%)",
        ylabel="Mean reproduction angular error (degrees)",
        xlim=(0, 100),
        ylim=(0, None),
    )
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
fig.suptitle("Compared with matched C+: better transfer risk, worse source accuracy")
for ext in ("png", "svg"):
    fig.savefig(FIGURES / ("risk_coverage." + ext), dpi=180, bbox_inches="tight")
plt.close(fig)

aggregate = json.loads((ROOT / "aggregate.json").read_text(encoding="utf-8"))["domains"]
domains = ["fresh_Canon_5DSR", "fresh_Nikon_D810", "fresh_Sony_IMX135_BLCCSC"]
labels = ["Canon 5DSR", "Nikon D810", "Sony IMX135"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
for ax, metric, title in zip(
    axes, ("mean", "risk80"), ("All images", "At 80% accepted coverage"), strict=True
):
    x = np.arange(3)
    for j, (group, label, color) in enumerate(METHODS[:3]):
        y = [aggregate[d][group][metric]["mean"] for d in domains]
        std = [aggregate[d][group][metric]["std_across_runs"] or 0 for d in domains]
        bars = ax.bar(
            x + (j - 1) * 0.25, y, width=0.24, color=color, label=label, yerr=std, capsize=2
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    ax.set(xticks=x, xticklabels=labels, ylabel="Reproduction angular error (degrees)", title=title)
    ax.set_ylim(0, 8.8)
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", frameon=False, ncol=1)
fig.suptitle("Camera-specific result: proposed loses to matched C+ on Canon")
for ext in ("png", "svg"):
    fig.savefig(FIGURES / ("camera_comparison." + ext), dpi=180, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
for group, label, color in METHODS:
    value = aggregate["fresh_all"][group]
    ax.scatter(
        100 * value["source80_actual_coverage"]["mean"],
        value["source80_error"]["mean"],
        color=color,
        s=85,
        label=label,
    )
    ax.scatter(80, value["risk80"]["mean"], color=color, marker="x", s=70)
    ax.plot(
        [100 * value["source80_actual_coverage"]["mean"], 80],
        [value["source80_error"]["mean"], value["risk80"]["mean"]],
        linestyle=":",
        linewidth=1,
        color=color,
    )
ax.set(
    xlabel="Actual accepted coverage on fresh cameras (%)",
    ylabel="Accepted reproduction angular error (degrees)",
    title="Source-calibrated 80% thresholds do not deliver 80% target coverage",
    xlim=(0, 100),
    ylim=(0, 7),
)
ax.text(
    0.02,
    0.97,
    "Circles: frozen source thresholds\nCrosses: offline fixed-80% ranking",
    transform=ax.transAxes,
    va="top",
    fontsize=9,
)
fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", frameon=False, fontsize=9)
for ext in ("png", "svg"):
    fig.savefig(FIGURES / ("frozen_thresholds." + ext), dpi=180, bbox_inches="tight")
plt.close(fig)
print(
    json.dumps(
        {
            "figures": str(FIGURES),
            "note": "Lines/bars are mean over3seeds when available; shading/error bars are seedSD, not confidence intervals. Statistical baseline has1deterministicfit.",
        }
    )
)
