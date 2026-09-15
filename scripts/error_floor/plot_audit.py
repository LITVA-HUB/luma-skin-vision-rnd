"""Render aggregate TRAIN audit quantiles; never plots absent model residuals."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def main(args):
    instrument = json.loads(args.instrument.read_bytes())
    capture = json.loads(args.capture.read_bytes())
    partial = capture["scope"] != "COMPLETE_TRAIN_CAPTURE_DIAGNOSTIC_NO_TRAINING"
    if partial and not args.allow_partial:
        raise ValueError("Partial capture results require --allow-partial; not a completed-cohort chart")
    args.output.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.titleweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(13, 10.6), gridspec_kw={"height_ratios": [1, 1.7]})
    fig.subplots_adjust(left=.29, right=.95, top=.88, bottom=.24, hspace=.64)
    fig.suptitle("MSKCC TRAIN: reference disagreement and capture variability", fontsize=16,
                 fontweight="bold", x=.5, y=.97)
    fig.text(.5, .936, "Diagnostic distributions — neither panel estimates the deployable model error floor",
             ha="center", fontsize=11, color="#50565e")
    reference = instrument["instrument_repeatability"]
    rows = [
        ("Two recorded assessments\n(pairwise, all 3 pairs/site)", reference["pairwise_delta_e00"]),
        ("Held assessment vs other-two mean\n(single-reading disagreement)",
         reference["held_assessment_vs_other_two_mean"]["delta_e00"]),
        ("Three-reading vs two-reading mean\n(target omission sensitivity)",
         reference["triplet_target_sensitivity_to_omitting_one_assessment"]["delta_e00"])]
    ax = axes[0]
    col = "#245F88"
    for i, (_, s) in enumerate(rows):
        ax.hlines(i, 0, s["p95"], color=col, alpha=.25, linewidth=6)
        ax.scatter(s["median"], i, color=col, s=47, zorder=3)
        ax.scatter(s["mean"], i, facecolor="white", edgecolor=col, marker="D", s=31, zorder=4)
        ax.scatter(s["p90"], i, color=col, marker="|", s=90, zorder=4)
        ax.scatter(s["max"], i, color=col, marker="x", s=28, zorder=4)
        ax.text(10.8, i, f'{s["median"]:.2f} / {s["p95"]:.2f}', va="center", fontsize=10)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(2.55, -.55)
    ax.set_xlim(0, 12.7)
    ax.set_xticks(range(0, 11, 2))
    ax.grid(axis="x", color="#d8dde3", alpha=.65)
    ax.set_xlabel("Instrument ΔE00 · D65 / CIE 1964 10°")
    ax.set_title("A. 248 unique reference triplets · 744 comparisons · 24 people", loc="left", pad=18)
    ax.text(10.8, -.61, "median / p95", fontsize=9, color="#50565e")

    ax = axes[1]
    strata = capture["strata"]["mode_pair"]
    colors = ["#257C70", "#BA6A23"]
    methods = ["original_central_mean_delta_e00", "original_central_median_delta_e00"]
    labels = []
    for i, row in enumerate(strata):
        labels.append(f'{row["group"]}\n{row["pairs"]} pairs · {row["sites"]} sites · {row["people"]} people')
        for offset, method, color in zip([-.15, .15], methods, colors):
            s, y = row[method], i+offset
            ax.hlines(y, 0, s["p95"], color=color, alpha=.3, linewidth=5)
            ax.scatter(s["median"], y, color=color, s=33, zorder=3)
            ax.scatter(s["mean"], y, facecolor="white", edgecolor=color, marker="D", s=25, zorder=4)
            ax.scatter(s["p90"], y, color=color, marker="|", s=64, zorder=4)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels)-.5, -.65)
    max_p95 = max(row[m]["p95"] for row in strata for m in methods)
    ax.set_xlim(0, max_p95*1.07)
    ax.grid(axis="x", color="#d8dde3", alpha=.65)
    ax.set_xlabel("Observed within-site ΔE00 · sRGB D65 / CIE 1931 2° · exact original central ROI")
    n = capture["counts"]
    ax.set_title(f'B. {n["images"]} images · {n["same_site_pairs"]} same-site pairs\n'
                 f'{n["same_mode_pairs"]} same-mode repeats · {n["cross_device_pairs"]} cross-device pairs',
                 loc="left", pad=17, fontsize=11)
    legend = [Line2D([0], [0], color=c, linewidth=5, alpha=.55, label=l)
              for c, l in zip(colors, ["Mean RGB → Lab", "Median RGB → Lab"])]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.53, .156),
               frameon=False, ncol=2, fontsize=10)
    glyphs = [Line2D([0], [0], marker="o", linestyle="", color="#444", label="Median"),
              Line2D([0], [0], marker="D", linestyle="", markerfacecolor="white", color="#444", label="Mean"),
              Line2D([0], [0], marker="|", linestyle="", color="#444", label="p90"),
              Line2D([0], [0], linewidth=5, alpha=.3, color="#444", label="Bar ends at p95"),
              Line2D([0], [0], marker="x", linestyle="", color="#444", label="Max (panel A)")]
    fig.legend(handles=glyphs, loc="lower center", bbox_to_anchor=(.53, .115), frameon=False,
               ncol=5, columnspacing=1.8)
    fig.text(.06, .075, "NP / P = non-polarized / polarized; C / NC = contact / non-contact. NP-NC is clinical close-up.",
             fontsize=9.5, color="#50565e")
    fig.text(.06, .05, "All same-site images reuse one instrument triplet. Crop median is not a verified skin mask.\n"
             "Quantile bars are not confidence intervals. Capture differences cannot be subtracted from model ΔE00.",
             fontsize=9.5, color="#50565e", va="top")
    if partial:
        fig.text(.5, .53, "PARTIAL CAPTURE — DEBUG ONLY", fontsize=27, color="#b94029",
                 alpha=.28, rotation=18, ha="center", va="center", weight="bold")
    stem = "error_floor_diagnostic_partial" if partial else "error_floor_diagnostic"
    png, pdf = args.output/f"{stem}.png", args.output/f"{stem}.pdf"
    fig.savefig(png, dpi=180, facecolor="white")
    fig.savefig(pdf, facecolor="white", metadata={"Title": "MSKCC TRAIN error-floor diagnostic",
                                                "Author": "Luma", "CreationDate": None})
    plt.close(fig)
    receipt = {"scope": capture["scope"], "input_sha256": {
        "instrument": hashlib.sha256(args.instrument.read_bytes()).hexdigest(),
        "capture": hashlib.sha256(args.capture.read_bytes()).hexdigest()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (png, pdf)},
        "model_error_plotted": False, "distribution_display": "precomputed exact aggregate quantiles"}
    (args.output/f"{stem}_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instrument", type=Path, required=True)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    main(p.parse_args())
