"""Report all frozen G families, controls and actual standalone runtime measurements."""

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

ROOT = Path(__file__).resolve().parents[1]
ROLES = {"mixed": "Mixed", "slr_to_ipod": "SLR → iPod", "ipod_to_slr": "iPod → SLR"}
NAMES = {
    "norm_base": "MSE base",
    "norm_uniform": "MSE uniform",
    "norm_soft": "MSE soft gate",
    "norm_hard": "MSE hard gate",
    "perceptual_base": "Perceptual base",
    "perceptual_uniform": "Perceptual uniform",
    "perceptual_soft": "Perceptual soft gate",
    "perceptual_hard": "Perceptual hard gate",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(result, selected, runtime):
    rows = []
    for role in ROLES:
        for family in NAMES:
            rr = [r for r in result["records"] if r["role"] == role and r["family"] == family]
            tt = [r for r in runtime["records"] if r["role"] == role and r["family"] == family]
            fit = next(
                r
                for r in runtime["standalone_fit_records"]
                if r["role"] == role and r["family"] == family
            )
            choice = selected["roles"][role][family]["selected"]
            assert len(rr) == len(tt) == 3
            rows.append(
                dict(
                    role=role,
                    family=family,
                    residual_lambda=choice["residual_lambda"],
                    rho=choice["rho"],
                    inner_person_mean=choice["person_mean"],
                    metrics={
                        k: float(np.mean([r["metrics"][k] for r in rr]))
                        for k in (
                            "person_mean",
                            "image_mean",
                            "site_person_mean",
                            "p90",
                            "gt5",
                            "gt10",
                        )
                    },
                    people=rr[0]["metrics"]["n_people"],
                    images=rr[0]["metrics"]["n_images"],
                    numeric_bytes=rr[0]["numeric_bytes"],
                    archive_bytes_by_seed=[r["archive_bytes"] for r in rr],
                    active_gate=rr[0]["active_gate"],
                    seed_person_means=[r["metrics"]["person_mean"] for r in rr],
                    canonical_median_us=float(np.median([r["canonical"]["median_us"] for r in tt])),
                    numpy_median_us=float(np.median([r["numpy_only"]["median_us"] for r in tt])),
                    numpy_p95_us=float(np.median([r["numpy_only"]["p95_us"] for r in tt])),
                    numpy_cached_array_bytes=tt[0]["numpy_only"]["cached_array_bytes"],
                    fit_median_ms=fit["median_seconds"] * 1000,
                    fit_rows=fit["n_fit_rows"],
                )
            )
    return rows


def draw(rows, out):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    mixed = [r for r in rows if r["role"] == "mixed"]
    positions = np.arange(8)
    colors = ["#335A85"] * 4 + ["#137E70"] * 4
    ax = axes[0, 0]
    values = [r["metrics"]["person_mean"] for r in mixed]
    ax.scatter(values, positions, c=colors, s=45)
    for value, y in zip(values, positions, strict=True):
        ax.text(value + 0.004, y, f"{value:.4f}", va="center", fontsize=9)
    ax.set_xlim(5.25, 5.50)
    ax.set_title("Mixed role: 6 held people · zoomed x axis")
    ax.set_xlabel("Person mean ΔE00 (lower is better)")
    ax = axes[0, 1]
    for role, marker, color in [("slr_to_ipod", "o", "#335A85"), ("ipod_to_slr", "s", "#BD6B25")]:
        data = [r for r in rows if r["role"] == role]
        ax.scatter(
            [r["metrics"]["person_mean"] for r in data],
            positions,
            marker=marker,
            color=color,
            s=40,
            label=ROLES[role],
        )
    ax.set_xlim(8.2, 8.95)
    ax.set_title("Single-camera fits: routed corrections fall back to base")
    ax.set_xlabel("Person mean ΔE00 (lower is better)")
    ax.legend(loc="lower right", frameon=False)
    ax = axes[1, 0]
    for key, marker, color, name in [
        ("canonical_median_us", "o", "#335A85", "Canonical"),
        ("numpy_median_us", "s", "#137E70", "NumPy only"),
    ]:
        ax.scatter([r[key] for r in mixed], positions, marker=marker, color=color, s=40, label=name)
    ax.set_xlim(0, 28)
    ax.set_title("Mixed-role actual batch-one response, one CPU thread")
    ax.set_xlabel("Median time (µs), prepared color36 input")
    ax.legend(loc="lower left", frameon=False)
    ax = axes[1, 1]
    ax.barh(positions, [r["fit_median_ms"] for r in mixed], color=colors, height=0.55)
    for r, y in zip(mixed, positions, strict=True):
        ax.text(r["fit_median_ms"] + 0.4, y, f"{r['fit_median_ms']:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, 42)
    ax.set_title("Complete fit on 734 prepared rows · seed17")
    ax.set_xlabel("Median time (ms), 3 fits after one warmup")
    for ax in axes.flat:
        ax.set_yticks(positions, list(NAMES.values()))
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Luma ChromaSeed-G · Compact conditional color correction", fontsize=15, x=0.02, ha="left"
    )
    fig.text(
        0.02,
        0.015,
        "Exploratory reused TRAIN roles; seed errors averaged, no inference ensemble. No image preprocessing or ordinary-phone face validation.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    for suffix in ("png", "svg"):
        fig.savefig(out / f"gated_results.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run, args.output
    result, selected = read(run / "results.json"), read(run / "selections.json")
    audit, runtime = read(out / "audit.json"), read(out / "runtime.json")
    assert audit["passed"] and audit["results_sha256"] == sha(run / "results.json")
    assert runtime["audit_sha256"] == sha(out / "audit.json")
    for item in (result, audit, runtime, selected):
        assert item["source_lock_sha256"] == sha(run / "source_lock.json")
    assert runtime["runtime_source_sha256"] == sha(ROOT / "scripts/chromaseed_gated_runtime.py")
    for path, value in runtime["dependencies"].items():
        assert sha(ROOT / path) == value
    rows = aggregate(result, selected, runtime)
    draw(rows, out)
    receipts = [read(p) for p in sorted(run.rglob("receipt.json"))]
    workflow = read(run / "workflow.json")
    lines = [
        "# Luma ChromaSeed-G: a compact conditional color readout",
        "",
        "**A modest mixed-role improvement, with an explicit camera-transfer limit.** The perceptual hard-gate model gives5.2851 ΔE00 versus5.3901 for its exact shared base (−0.1050, about1.95%) and5.3617 for a uniform correction. Active weights occupy21,973 numeric bytes. Its separate NumPy consumer responds in11.7µs; complete mixed-role fitting takes35.53ms on734 prepared rows. These are exploratory results from six held people and reused splits, not a validated phone-face product or a universal winner.",
        "",
        "G follows the [D camera/support diagnostic](../chromaseed_camera_support_v1/report.md). It keeps one128-center kernel basis and adds a small residual whose sign/strength depends on current input features. Fit-side acquisition labels train the gate; query camera labels, measured Lab and query-distribution statistics are never inputs. The two known camera groups contain different people/acquisitions, so neither D nor G isolates a causal camera effect.",
        "",
        "## All selected families and roles",
        "",
        "Only original TRAIN:966 prepared skin-region records,24 people. Mixed fits734 rows/18 people and holds232/6; SLR→iPod fits323/8 and holds643/16; reverse swaps these. Three fixed inner person-disjoint folds choose settings for each family before any final-role score. All24 choices were frozen before all three final banks finished and before72 selected models were evaluated. Roles overlap and have been historically reused. Seeds17/29/43 are repeated fits: the table averages their errors, not their predictions, and does not count seeds as independent people.",
        "",
        "| Family | Mixed ΔE00 ↓ | SLR → iPod ↓ | iPod → SLR ↓ | Mixed numeric bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    for family, name in NAMES.items():
        rr = [r for r in rows if r["family"] == family]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{r['metrics']['person_mean']:.6f}" for r in rr)
            + f" | {rr[0]['numeric_bytes']:,} |"
        )
    lines.extend(
        [
            "",
            "All routed transfer choices return the exact base because their fit side contains one camera group. This is a fit-time fallback, not detection of an unknown camera during deployment. Uniform residual correction remains active and worsens both transfer directions for both bases. G therefore adds no demonstrated unseen-camera benefit.",
            "",
            "The matching base payloads reproduce [P](../chromaseed_perceptual_v1/report.md) exactly. Earlier strong references remain relevant: [full KRR](../skin_local_search_v1/report.md) gives5.3984/8.5845/8.9125 with larger, role-dependent storage; the compact guided RBF gives7.7193 in reverse, better than G's8.3869. [R's dynamic model](../chromaseed_refine_v1/report.md) gives5.3551/9.9089/9.8122 at59,316B. G's mixed result is descriptively lower, but no fresh outer sample selects or confirms a final family.",
            "",
            "![Quality and runtime](gated_results.png)",
            "",
            "## Selection and secondary metrics",
            "",
            "The primary score first averages native ΔE00 across images of each person, then across people. Training instead weights people, sites and their images equally. Site-balanced and image-tail summaries are retained below; p90 is the mean of per-seed image p90 values. Lambda10 with rho0 is a zero-alias placeholder, not a fitted residual penalty.",
            "",
            "| Role | Family | λ / ρ | Inner person ΔE00 | Held image mean | Held site/person mean | Held image p90 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        m = r["metrics"]
        lines.append(
            f"| {ROLES[r['role']]} | {NAMES[r['family']]} | {r['residual_lambda']:g} / {r['rho']:g} | {r['inner_person_mean']:.6f} | {m['image_mean']:.4f} | {m['site_person_mean']:.4f} | {m['p90']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired mixed-role checks",
            "",
            "For each person, average the three seed-specific errors before forming paired differences. Fixed-prediction bootstrap uses20,000 shared resamples of six people. The following95% ranges are descriptive: they omit fit/selection uncertainty and repeated research on these same people. They are not confirmatory population intervals or significance claims.",
            "",
            "| Candidate | Matched reference | Mean difference ↓ | People improved | Descriptive range |",
            "|---|---|---:|---:|---|",
        ]
    )
    for p in audit["paired"]:
        if p["role"] != "mixed":
            continue
        lo, hi = p["descriptive_fixed_prediction_person_range"]
        lines.append(
            f"| {NAMES[p['family']]} | {NAMES[p['reference']]} | {p['person_mean_difference']:+.6f} | {p['people_improved']}/{p['people']} | [{lo:+.6f}, {hi:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "Perceptual hard improves four of six people against its base; its descriptive range includes zero. Soft improves five of six with a very similar mean. Hard routing has a discontinuity at gate score zero; observed ordinary-input quality alone does not establish its stability under color perturbations. That is the next bounded diagnostic, with fixed weights and prespecified perturbations, not another outer-tuned grid.",
            "",
            "## Actual deployment and training cost",
            "",
            "One CPU thread on AMD Ryzen9 7900X, Windows11, NumPy2.5.3. The NumPy-only consumer imports neither Torch nor SciPy and caches converted arrays once. Both implementations were checked on every selected query using actual batch-one calls;20 warmups and3 full query passes per model. Latencies below are medians across three per-seed medians; p95 is the median across seed p95 values. Complete fitting is seed17, three timed fits after one warmup, with every resulting array exactly equal to its frozen payload. No concurrent heavy study process ran during profiling.",
            "",
            "| Role | Family | Numeric B | Canonical µs | NumPy µs / p95 | Cached array B | Full fit ms |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        lines.append(
            f"| {ROLES[r['role']]} | {NAMES[r['family']]} | {r['numeric_bytes']:,} | {r['canonical_median_us']:.2f} | {r['numpy_median_us']:.2f} / {r['numpy_p95_us']:.2f} | {r['numpy_cached_array_bytes']:,} | {r['fit_median_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "The active gate adds1,689 numeric bytes (8.33%) over20,284B; the uniform correction collapses into existing coefficients. The standalone predictor caches44,640B of arrays for an active model versus41,272B for a static model. These exclude Python/NumPy/process memory, temporary arrays, retained input dictionaries and compressed archive overhead;21,973B is not end-to-end RAM. Predictor construction and validation are timed separately in runtime.json; archive loading is excluded. Image decoding, skin localization, color36 extraction, network transport and mobile hardware are outside all quoted latencies.",
            "",
            "Four fixed active probes (both bases × soft/hard, λ0.1/ρ1) ensure active-path costs are measured even when a selected family elsewhere falls back. They add16 full fits including warmups to96 selected fits; no new quality choice was made from these probes.",
            "",
            "| Fixed active mixed probe | NumPy µs | Canonical µs | Full fit ms |",
            "|---|---:|---:|---:|",
        ]
    )
    for p in runtime["fixed_active_records"]:
        lines.append(
            f"| {NAMES[p['family']]} | {p['numpy_only']['median_us']:.2f} | {p['canonical']['median_us']:.2f} | {p['standalone_fit']['median_seconds'] * 1000:.2f} |"
        )
    lines.extend(
        [
            "",
            "## What was fitted and independently checked",
            "",
            f"The primary workflow took{workflow['wall_seconds']:.3f}s including reading, fitting, selection, persistence and evaluation, excluding interpreter imports, audit and profiling. Across12 banks:2,016 stored readouts include864 exact one-camera routed fallbacks and72 unchanged P base controls. Only360 positive residual coefficient solutions were solved, sharing120 algebra operations across three penalties; there were36 perceptual base solves and4 gate fits. Three strengths reuse each coefficient solution. Do not call2,016 configurations2,016 independent trainings.",
            "",
            "Residual fitting uses an analytic weighted ridge solution in the existing whitened kernel basis. It localizes a correction in the readout without global backpropagation or brute-force weight search. The current input modulates effective output coefficients; the model does not learn at query time, add connections, or repeatedly run until convergence. All360 regularized residual objectives are nonincreasing, which is a fit check rather than evidence of generalization.",
            "",
            f"Primary23 numerical tests passed in1.56s; three separate portable-consumer tests passed in0.25s. The audit checks all12 banks/2,016 readouts,72 exact P controls,864 fallbacks,285,600 OOF query rows,24 choices and72 selected models/28,752 final query rows. All72 selected models were independently refitted using direct kernels, analytic perceptual tensors and augmented SVD;12 extra positive-route probes passed. Maximum independent-refit native-Lab drift {audit['maxima']['independent_refit_drift']:.3g}, versus registered0.001 tolerance. Direct OOF/final drift is below2e-12. The independent audit shares the existing verified ΔE00 formula and frozen split helpers.",
            "",
            "Primary PID36812, audit session27067 and runtime returned exit0. Source locks bind the core, runner, primary tests/protocol, inherited dependencies, all P control banks and D verification. The NumPy consumer, portable tests and postprocessing are additionally bound by final verification. Earlier locked sources and results are preserved.",
            "",
            "## Product boundary and next decision",
            "",
            "Luma ChromaSeed is the working family name; G denotes this conditional-readout experiment. Input is36 statistics from an already prepared skin region; output is three native Lab values in the dataset's D65/10° convention. The model does not detect faces, identify people, infer ethnicity, diagnose skin, calibrate an arbitrary camera, or establish cosmetic shade-match accuracy. No real ordinary-phone face benchmark or retailer integration is demonstrated. Dataset/weight commercial-use clearance is not inferred from a code implementation.",
            "",
            "Keep the base and both gates as research candidates. Next measure gate-score margin and output sensitivity to fixed, small synthetic color transformations, using the frozen payloads and no retraining. Report hard/soft/base controls and discontinuities explicitly; synthetic robustness cannot replace independent face acquisition. The broader compact/fast/high-quality goal remains active.",
            "",
            "Conditional expert readouts are established ideas: [Jacobs et al.,1991](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) and feature modulation in [Perez et al.,FiLM](https://arxiv.org/abs/1709.07871). This experiment makes no invention or breakthrough claim. No new data, model weights or third-party code were downloaded; only the original TRAIN arrays were accessed, with legacy validation/calibration/test excluded.",
            "",
            "[Protocol](../../research/chromaseed_gated_v1_protocol.md) · [Reproduce](reproduce.md) · [Audit](audit.json) · [Runtime](runtime.json) · [Summary](summary.json) · [Verification](verification.json) · [Model card](../../architecture/chromaseed_gated_model_card.md) · [Next decision](../../research/chromaseed_gated_next_decision.md)",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    for name in ("source_lock.json", "selections.json"):
        shutil.copy2(run / name, out / name)
    write_json(
        out / "summary.json",
        dict(
            source_lock_sha256=sha(run / "source_lock.json"),
            selection_sha256=sha(run / "selections.json"),
            results_sha256=sha(run / "results.json"),
            audit_sha256=sha(out / "audit.json"),
            runtime_sha256=sha(out / "runtime.json"),
            report_source_sha256=sha(Path(__file__)),
            rows=rows,
            paired=audit["paired"],
            workflow=workflow,
            bank_count=len(receipts),
            goal_status="active",
            evidence="exploratory reused TRAIN; phone-face quality unvalidated",
        ),
    )
    print("G all-family report, summary, manifest copies and figure written.")


if __name__ == "__main__":
    main()
