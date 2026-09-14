"""Report the camera/support diagnostic, with all controls and matching coverage."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skin_local_search_train import sha, write_json

VIEW_NAMES = {
    "color36": "All 36 color features",
    "rgb_mean": "Mean RGB only",
    "lab3": "Instrument Lab (oracle)",
    "lab_residual": "Color residual after Lab (oracle)",
}
ROLE_NAMES = {"mixed": "Mixed cameras", "slr_to_ipod": "SLR → iPod", "ipod_to_slr": "iPod → SLR"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run, args.output
    result = json.loads((run / "results.json").read_text())
    audit = json.loads((out / "audit.json").read_text())
    assert audit["passed"] and audit["results_sha256"] == sha(run / "results.json")
    assert (
        audit["source_lock_sha256"] == result["source_lock_sha256"] == sha(run / "source_lock.json")
    )
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})
    for mi, (method, color) in enumerate([("linear", "#335A85"), ("rbf", "#147D74")]):
        rows = [r for r in result["classifier_metrics"] if r["method"] == method]
        axes[0].barh(
            np.arange(4) + (mi - 0.5) * 0.32,
            [r["all_people"]["auc"] for r in rows],
            height=0.3,
            color=color,
            label=method.upper(),
        )
    axes[0].set_yticks(np.arange(4), list(VIEW_NAMES.values()))
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1.06)
    axes[0].axvline(0.5, color="#777777", ls=":", lw=1)
    axes[0].set_xlabel("Camera-classification AUC; 24 held-out people in total")
    axes[0].set_title("Camera group is strongly encoded in image features")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -.19), ncol=2, frameon=False)
    for vi, (view, color, label) in enumerate(
        [("color36", "#335A85", "Color features"), ("native_lab", "#BD6B25", "Native Lab")]
    ):
        rows = [r for r in result["support"] if r["view"] == view]
        vals = [100 * r["query_mass_above_radius"] for r in rows]
        pos = np.arange(3) + (vi - 0.5) * 0.32
        axes[1].barh(pos, vals, height=0.3, color=color, label=label)
        for y, value in zip(pos, vals, strict=True):
            axes[1].text(value + 0.6, y, f"{value:.1f}%", va="center", fontsize=9)
    axes[1].set_yticks(np.arange(3), list(ROLE_NAMES.values()))
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 65)
    axes[1].set_xlabel("Query mass above its fit-side 95% neighbor radius (%)")
    axes[1].set_title("Cross-camera feature coverage is weaker")
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, -.19), ncol=2, frameon=False)
    for ax in axes:
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle(
        "ChromaSeed-D · Camera and color-support diagnosis on reused TRAIN",
        fontsize=14,
        x=0.025,
        ha="left",
    )
    fig.text(
        0.025,
        0.025,
        "People are disjoint within folds; cameras use different people. Oracle controls require measured Lab. Skin-color quality is not reevaluated here.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    for suffix in ("png", "svg"):
        fig.savefig(out / f"camera_support.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    lines = [
        "# ChromaSeed-D: camera and target-support diagnosis",
        "",
        "**Progress toward a justified adaptive color model; no new skin-color accuracy claim.** A fixed linear classifier distinguishes all 24 person-held-out camera labels from color36. AUC from native instrument Lab alone is only0.586 (linear) and0.578 (RBF). Camera separation persists after a fit-only linear Lab residualization. This supports testing an acquisition-conditioned color correction, while different people/acquisition per camera prevent a causal conclusion.",
        "",
        "The previous [S study](../chromaseed_selection_stability_v1/report.md) showed selection instability but did not explain the transfer loss. D follows its registered next question. It uses original TRAIN only and does not access prior outer model/results/prediction archives.",
        "",
        "## Fixed camera classifiers: all views and methods",
        "",
        "Three person-disjoint folds over24 people (8 SLR,16 iPod),966 prepared skin-region records. Equal camera/person/site/image fit weights. Each person gets one score averaged equally across their sites. Linear alpha0.1 and RBF alpha0.01 are fixed; no search or threshold tuning. Oracle controls consume measured target Lab and cannot be deployed on an unmeasured selfie.",
        "",
        "| Input | Method | Person AUC | Balanced accuracy, threshold0 | Correct SLR people | Correct iPod people | Fold AUCs |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in result["classifier_metrics"]:
        m = r["all_people"]
        lines.append(
            f"| {VIEW_NAMES[r['view']]} | {r['method']} | {m['auc']:.4f} | {m['balanced_accuracy']:.4f} | {m['slr_correct']}/{m['slr_people']} | {m['ipod_correct']}/{m['ipod_people']} | "
            + ", ".join(f"{f['auc']:.4f}" for f in r["folds"])
            + " |"
        )
    lines.extend(
        [
            "",
            "AUC measures ranking of camera scores, not correctness of predicted skin color. A high AUC can coexist with threshold errors: RBF color36 ranks all people correctly but misses one SLR person at the fixed zero threshold. Fold AUCs and pooled AUC differ when independently fitted fold score scales differ. No population confidence intervals or causal tests are claimed.",
            "",
            "![Camera signal and coverage](camera_support.png)",
            "",
            "## Target-color matching within held folds",
            "",
            "Pair people using equal-site native-Lab centroids, maximizing pair count under each prespecified DeltaE00 caliper and then minimizing total distance. Every person is used at most once per caliper. All pairings stay within the same held fold. Matching conditions evaluation on held targets without fitting classifiers on them; it does not equalize within-person color distributions or other acquisition differences.",
            "",
            "| Caliper | Pairs | Matched people | Unmatched SLR / iPod | Matched images | Mean pair DeltaE00 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in result["matched"]:
        distance = f"{r['pair_de00']['mean']:.4f}" if r["pair_de00"] else "—"
        lines.append(
            f"| {r['caliper']:g} | {r['pairs']} | {r['matched_people']} | {r['unmatched_slr']} / {r['unmatched_ipod']} | {r['matched_images']} | {distance} |"
        )
    lines.extend(
        [
            "",
            "At calipers2 and3 only two pairs/four people are represented. Perfect camera ranking in that subset is therefore limited evidence. All calipers are reported; none was chosen for its favorable ranking.",
            "",
            "| Caliper | Input | Method | Matched-person AUC | Balanced accuracy | SLR score above paired iPod (ties half) |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for r in result["matched"]:
        for s in r["scores"]:
            m = s["metrics"]
            metrics = (
                f"{m['auc']:.4f} | {m['balanced_accuracy']:.4f} | {s['within_pair_slr_above']:.4f}"
                if m
                else "— | — | —"
            )
            lines.append(
                f"| {r['caliper']:g} | {VIEW_NAMES[s['view']]} | {s['method']} | {metrics} |"
            )
    lines.extend(
        [
            "",
            "## Coverage of the historical transfer roles",
            "",
            "Reference distances for fit rows exclude every row from their own person. Color36 uses fit-only weighted standardization and RMS distances; native Lab uses DeltaE00. Radius is the fit reference weighted empirical95th percentile. Query mass weights people and their sites equally. This is descriptive feature geometry, not a calibrated rejection rule or error bound.",
            "",
            "| Role | Space | Fit reference mean / p95 | Query mean / p95 | Query mass above fit radius |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for r in result["support"]:
        f, q = r["fit_reference"], r["query"]
        lines.append(
            f"| {ROLE_NAMES[r['role']]} | {r['view']} | {f['mean']:.4f} / {f['p95']:.4f} | {q['mean']:.4f} / {q['p95']:.4f} | {100 * r['query_mass_above_radius']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Feature mass beyond the fit radius is28.31% for SLR→iPod and54.20% for iPod→SLR, versus1.52% in the mixed role. The corresponding native-Lab masses are3.98%,9.34% and10.61%. The two spaces have different geometries and radii, so their distances are not directly comparable. Still, the feature coverage pattern supports investigating acquisition-dependent transformations before another larger penalty/step search.",
            "",
            "## Measured target coverage",
            "",
            "| Camera | People / rows | Lab 5th percentile | Lab median | Lab 95th percentile |",
            "|---|---:|---|---|---|",
        ]
    )
    for r in result["target_summary"]:
        q = ["(" + ", ".join(f"{v:.2f}" for v in a) + ")" for a in r["lab_quantiles_05_50_95"]]
        lines.append(f"| {r['camera']} | {r['people']} / {r['rows']} | {q[0]} | {q[1]} | {q[2]} |")
    lines.extend(
        [
            "",
            "These are weighted component quantiles, not joint color samples or demographic categories. Native-Lab distributions do differ; the low target-only AUC does not prove identical populations. Person centroid extrema are retained in the machine-readable summary without identifiers.",
            "",
            "## Decision and evidence boundary",
            "",
            "Register a compact shared-kernel model with an acquisition-conditioned residual readout as the next experiment. Fit the gate and residual only on fit-side people, keep a shared-model control, and use exact shared fallback when a fit split contains only one camera group. Share the128-center basis so conditional coefficients add little storage and no second feature-extraction pass. This tests the requested dynamic connections in a bounded way; it does not assume that learning two known camera groups solves unseen-phone transfer. The next series is planned, not launched.",
            "",
            "Do not use oracle Lab-residual features in a deployable predictor, import held-camera labels into training, pick a caliper/radius by outer error, or claim that a camera classifier proves useful skin-color correction. Compactness and approximately0.017ms prepared-feature inference are prior measurements of the earlier model, not results of these diagnostic classifiers. Real ordinary-phone face quality remains unvalidated, so the full user goal remains active.",
            "",
            "## Verification",
            "",
            f"Eleven numerical tests pass. All24 classifiers were independently reconstructed with separate weighted normalizers and SVD solvers; all7728 query scores,8 pooled/24 fold metrics,15 exhaustive matching problems,40 matched metric groups,6 support cases/5796 row distances and2 target summaries passed. Maximum classifier-score drift {audit['maxima']['classifier_score_drift']:.3g}, support-distance drift {audit['maxima']['support_distance_drift']:.3g}. The audit shares the existing verified CIEDE2000 formula and frozen folds.",
            "",
            f"Primary diagnostic time {result['elapsed_seconds']:.3f}s excludes interpreter imports and auditing and is not a production training benchmark. Primary execution and final audit returned exit0. The first audit completed numerical checks but failed to serialize a NumPy integer counter; explicit scalar conversion fixed the report writer and the full audit was rerun successfully. No prior source lock was changed.",
            "",
            "Only TRAIN color/target/patient/site/device arrays were loaded. No images, tokens, legacy validation/calibration/test, prior outer results, external downloads or publication. Stored predictions and pairings use ordinal axes; no direct patient identifiers.",
            "",
            "Related classifier-based distribution diagnostics: [Jang et al. (2022)](https://proceedings.mlr.press/v162/jang22a.html). D is not their sequential hypothesis test. Numerical references: [SciPy assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html), [scikit-learn AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).",
            "",
            "[Protocol](../../research/chromaseed_camera_support_v1_protocol.md) · [Reproduce](reproduce.md) · [Audit](audit.json) · [Verification](verification.json) · [Summary](summary.json) · [Next decision](../../research/chromaseed_camera_support_next_decision.md)",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    shutil.copy2(run / "source_lock.json", out / "source_lock.json")
    write_json(
        out / "summary.json",
        dict(
            source_lock_sha256=sha(run / "source_lock.json"),
            results_sha256=sha(run / "results.json"),
            audit_sha256=sha(out / "audit.json"),
            report_source_sha256=sha(Path(__file__)),
            classifier_metrics=result["classifier_metrics"],
            matched=result["matched"],
            support=result["support"],
            target_summary=result["target_summary"],
        ),
    )
    print("Camera/support report and figure written.")


if __name__ == "__main__":
    main()
