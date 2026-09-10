"""Build source-backed report; no training or selection choices occur here."""

import hashlib
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from luma_skin_vision.cc.benchmark import indices, load
from luma_skin_vision.experiment import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/benchmarks"
SEEDS = [17, 29, 43]
LABELS = {
    "gray_world": "Gray World + learned selector",
    "max_rgb": "Max RGB + learned selector",
    "shades_gray": "Shades of Gray + learned selector",
    "gray_edge": "Gray Edge + learned selector",
    "baseline_context": "C+ standard context error head",
    "baseline_disagreement": "C + disagreement-only head",
    "baseline_combined": "C + combined error head (strong control)",
    "proposed_context": "Mixture + context-only head",
    "proposed_disagreement": "Mixture + disagreement-only head",
    "proposed_combined": "Proposed mixture + combined head",
}


def run(protocol, method, seed):
    folder = ROOT / f"experiments/runs/cc_{protocol}_{method}_s{seed}"
    return {
        name: json.loads((folder / (name + ".json")).read_text())
        for name in ["config", "training", "inference", "evaluation"]
    }


def values(records, name, domain):
    method = "proposed" if name.startswith("proposed") else "baseline"
    return [records[(method, seed)]["evaluation"][domain][name] for seed in SEEDS]


def avg_sd(a):
    return f"{np.mean(a):.3f} ± {np.std(a, ddof=1):.3f}"


def boot_pair(a, b, groups, repeats=2000):
    rng = np.random.default_rng(20260910)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    chunks = [np.flatnonzero(groups == g) for g in unique]
    result = []
    ea, eb, sa, sb = (np.array(x) for x in [a["errors"], b["errors"], a["scores"], b["scores"]])
    ties = np.array([hashlib.sha256(ident.encode()).hexdigest() for ident in a["ids"]])
    for _ in range(repeats):
        idx = np.concatenate([chunks[i] for i in rng.integers(len(chunks), size=len(chunks))])
        n = max(1, int(0.8 * len(idx)))
        ra = ea[idx][np.lexsort((ties[idx], sa[idx]))[:n]].mean()
        rb = eb[idx][np.lexsort((ties[idx], sb[idx]))[:n]].mean()
        result.append([float(np.mean(ea[idx] - eb[idx])), float(ra - rb)])
    arr = np.array(result)
    return {
        "meaning": "Proposed minus comparator; negative favors Proposed",
        "repeats": repeats,
        "groups": len(unique),
        "full_mean_delta_ci95": np.percentile(arr[:, 0], [2.5, 97.5]).tolist(),
        "risk80_delta_ci95": np.percentile(arr[:, 1], [2.5, 97.5]).tolist(),
        "risk80_delta_point": a["selective"]["fixed"]["80"]["mean"]
        - b["selective"]["fixed"]["80"]["mean"],
    }


def main():
    records = {
        (method, seed): run("official", method, seed)
        for method in ["baseline", "proposed"]
        for seed in SEEDS
    }
    camera = {method: run("camera", method, 17) for method in ["baseline", "proposed"]}
    cache, rows = load(ROOT / "data/processed/cc128")
    ix = indices(rows, "official")
    ids = {r["id"]: r for r in rows}
    group_sets = {p: {rows[i]["group"] for i in ind} for p, ind in ix.items()}
    audit = {
        "image_count": len(rows),
        "official_test": len(ix["test"]),
        "split_counts": {p: len(v) for p, v in ix.items()},
        "capture_date_counts": {p: len(v) for p, v in group_sets.items()},
        "test_dates_overlapping_development": len(
            group_sets["test"] & set.union(*[v for p, v in group_sets.items() if p != "test"])
        ),
        "exact_duplicate_image_hashes": len(rows) - len({r["sha256"] for r in rows}),
        "camera_counts": {p: len(v) for p, v in indices(rows, "camera").items()},
        "near_duplicate_semantic_scene_audit": "NOT MEASURED; capture dates are conservative internal grouping, not verified scene identities",
    }
    write_json(OUT / "public_data_audit.json", audit)
    pairs = {}
    primary = records[("proposed", 17)]["evaluation"]["source"]["proposed_combined"]
    for name in ["baseline_context", "baseline_combined"]:
        comparator = records[("baseline", 17)]["evaluation"]["source"][name]
        assert primary["ids"] == comparator["ids"]
        pairs[name] = boot_pair(primary, comparator, [ids[i]["group"] for i in primary["ids"]])
    write_json(OUT / "public_bootstrap.json", pairs)
    transfer = {}
    for domain in ["sony_pilot", "camera"]:
        if domain == "sony_pilot":
            a = records[("proposed", 17)]["evaluation"][domain]["proposed_combined"]
            b = records[("baseline", 17)]["evaluation"][domain]["gray_world"]
            groups = a["ids"]
        else:
            a = camera["proposed"]["evaluation"]["source"]["proposed_combined"]
            b = camera["baseline"]["evaluation"]["source"]["shades_gray"]
            groups = [ids[i]["group"] for i in a["ids"]]
        assert a["ids"] == b["ids"]
        transfer[domain] = boot_pair(a, b, groups)
    write_json(OUT / "public_transfer_bootstrap.json", transfer)
    evidence = OUT / "public_runs"
    evidence.mkdir(exist_ok=True)
    manifest = []
    for record in list(records.values()) + list(camera.values()):
        config = record["config"]
        folder = ROOT / config["out"]
        dest = evidence / folder.name
        dest.mkdir(exist_ok=True)
        for name in [
            "config.json",
            "training.json",
            "inference.json",
            "evaluation.json",
            "risk_heads.json",
        ]:
            shutil.copy2(folder / name, dest / name)
        manifest.append(
            {
                "run": folder.name,
                "model_sha256": hashlib.sha256((folder / "model.pt").read_bytes()).hexdigest(),
                "data_hashes": config["data_hashes"],
                "source_hash": config["source_hash"],
            }
        )
    write_json(OUT / "public_evidence_manifest.json", manifest)
    lines = [
        "# Public real-data color-constancy milestone",
        "",
        "All numerical model rows below are **REPRODUCED LOCALLY**. Units are degrees; lower is better. No physical ΔE00 or facial accuracy is measured. Seed mean ± sample standard deviation over17/29/43, not a confidence interval and not an ensemble.",
        "",
        "## Real data and license",
        "",
        "SimpleCube++ v2:2234 real images; official test462; development1126 estimator train /119 validation /259 risk-fit /268 calibration. Original data CC BY4.0, publisher MD5 verified. Additional20MB Cube++ metadata supplies capture dates/camera labels. Sony pilot:30 author-selected C5-redistributed INTEL-TAU examples (13,826,580 bytes including JSON), original dataset CC BY-SA4.0; evaluation only. Original INTEL-TAU archives were inaccessible and the full benchmark was not run. See [inventory](../data/public_dataset_inventory.md), [attribution](PUBLIC_DATA_NOTICE.md), [protocol](../research/public_protocol_v1.md).",
        "",
        f"Official test contains{audit['test_dates_overlapping_development']} capture dates also present in development; no exact image-byte duplicates were found. This split is publisher-compatible, not scene-independent. Internal estimator/validation/risk/calibration dates are disjoint. Test scenes were not semantically deduplicated. No camera identity/CCM/test-set adaptation enters prediction; raw decoding does use sensor-level white metadata.",
        "",
        "## Official SimpleCube++ results",
        "",
        "| Method | Recovery mean | Reproduction mean | Reproduction median | Risk at80% | AURC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, label in LABELS.items():
        v = values(records, name, "source")
        stats = [
            [x["recovery"]["mean"] for x in v],
            [x["reproduction"]["mean"] for x in v],
            [x["reproduction"]["median"] for x in v],
            [x["selective"]["fixed"]["80"]["mean"] for x in v],
            [x["selective"]["aurc"] for x in v],
        ]
        lines.append("| " + label + " | " + " | ".join(avg_sd(s) for s in stats) + " |")
    lines += [
        "",
        "Classical estimates are deterministic; their learned selectors share the corresponding baseline encoder. They therefore incur CNN compute for rejection. C+ uses a standard context error head. The combined-error control tests whether the special mixture adds anything beyond error-head features. Same stored capacity966,760 parameters; baseline active964,131 versus Proposed966,760. No imported pretrained weights or claimed FC4/C5 paper reproduction.",
        "",
        "## Fixed diagnostic coverage",
        "",
        "| Accepted target | C+ | Strong combined control | Proposed |",
        "|---|---:|---:|---:|",
    ]
    for cov in ["100", "95", "90", "80", "70", "60"]:
        scores = [
            avg_sd([v["selective"]["fixed"][cov]["mean"] for v in values(records, name, "source")])
            for name in ["baseline_context", "baseline_combined", "proposed_combined"]
        ]
        lines.append(f"| {cov}% | " + " | ".join(scores) + " |")
    lines += [
        "",
        "At80% the accepted count is369/462 =79.87% (floor convention). Full curves are stored per run; fixed-coverage ranking is descriptive, not a deployment threshold. Best/worst quartiles use ceil(n/4); trimean uses NumPy linear percentile convention.",
        "",
        "## Camera transfer",
        "",
        "| Protocol / method | Reproduction mean | Risk80 | Frozen source80 threshold: coverage / risk |",
        "|---|---:|---:|---|",
    ]
    for name in [
        "gray_world",
        "shades_gray",
        "baseline_context",
        "baseline_combined",
        "proposed_combined",
    ]:
        v = values(records, name, "sony_pilot")
        cov = np.mean([x["frozen_source_thresholds"]["80"]["coverage"] for x in v])
        risk = [x["frozen_source_thresholds"]["80"]["mean_reproduction"] for x in v]
        lines.append(
            f"| Sony30 / {LABELS[name]} | {avg_sd([x['reproduction']['mean'] for x in v])} | {avg_sd([x['selective']['fixed']['80']['mean'] for x in v])} | {cov:.1%} / {avg_sd(risk)} |"
        )
    for method in ["baseline", "proposed"]:
        for name in (
            ["gray_world", "shades_gray", "baseline_context", "baseline_combined"]
            if method == "baseline"
            else ["proposed_combined"]
        ):
            x = camera[method]["evaluation"]["source"][name]
            frozen = x["frozen_source_thresholds"]["80"]
            lines.append(
                f"| 550D→600D / {LABELS[name]} | {x['reproduction']['mean']:.3f} | {x['selective']['fixed']['80']['mean']:.3f} | {frozen['coverage']:.1%} / {frozen['mean_reproduction']:.3f} |"
            )
    lines += [
        "",
        "550D→600D uses829 train /72 validation /67 risk-fit /25 calibration and931 held-out600D images with no source-development date overlap. Both models have the same sensor type; this is limited held-out-camera evidence. The25-image calibration subset is particularly weak. Sony is a different sensor, but only30 author-selected samples and unknown scene dependence. Neither protocol establishes generalization to arbitrary phones/ISPs. Source-calibrated error estimates can remain badly wrong on unseen cameras.",
        "",
        "## Tails and uncertainty of the comparison",
        "",
        "| Method (seed17) | Reproduction p95 | Worst25 mean | >10° fraction | >20° fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ["shades_gray", "baseline_context", "baseline_combined", "proposed_combined"]:
        method = "proposed" if name.startswith("proposed") else "baseline"
        x = records[(method, 17)]["evaluation"]["source"][name]["reproduction"]
        lines.append(
            f"| {LABELS[name]} | {x['p95']:.3f} | {x['worst25']:.3f} | {x['over10_fraction']:.1%} | {x['over20_fraction']:.1%} |"
        )
    for name, p in pairs.items():
        lines += [
            "",
            f"Paired2000-resample capture-date bootstrap, primary seed17, Proposed minus{name}: full mean difference95% CI{p['full_mean_delta_ci95']}; risk80 difference95% CI{p['risk80_delta_ci95']}. Negative favors Proposed. Conditional on this data/model; not a model-training uncertainty bound.",
        ]
    lines += [
        "",
        "## Hardware",
        "",
        "| Model seed17 | Active / stored parameters | Checkpoint MB | Training peak allocated MB | Model-only batch1 median ms | Inference peak allocated MB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ["baseline", "proposed"]:
        r = records[(method, 17)]
        t, i = r["training"], r["inference"]
        lines.append(
            f"| {method} | {t['gradient_active_parameters']} / {t['parameters']} | {t['checkpoint_bytes'] / 1e6:.3f} | {t['peak_allocated_mb_including_cache']:.1f} | {i['model_only_median_ms']:.3f} | {i['peak_allocated_mb']:.1f} |"
        )
    lines += [
        "",
        "RTX4060 8GB, FP32, input128×128, 60 epochs each. Training peak includes419MB cached image tensors; CUDA allocator measurements exclude driver/context and unrelated processes. Model-only timings exclude decode, expert bank, transfers and error head; see separate full-path profile. No TensorRT/quantization/public-model ONNX export was performed because special-mechanism advantage is not established.",
        "",
        "## Interpretation and next decision",
        "",
        "A real compact normalization component has now been measured against real illumination references. Selective ranking can reduce within-source error. This does not prove physical surface/skin color reconstruction, cosmetic shade matching, reliable rejection under domain shift, or patentable novelty. Reproduction angular error concerns residual neutral cast, not a measured ΔE00 over scene surfaces.",
        "",
        "The mixture must beat the strong combined control and C+, with stable tail/generalization behavior, to justify continued development. Small point-estimate differences are not evidence of an innovation. Preserve every negative result; prioritize the ordinary compact estimator plus robust error calibration and genuinely diverse permissively licensed cameras, rather than expanding the mixture or tuning this test set. Proprietary facial validation stays future work and is not a prerequisite for the next public-data iteration.",
        "",
        "Published author numbers: **not included as locally reproduced rows**. FC4/Reweight-CC/C5/CCMNet/uncertainty2025/VLM-CC/GC3/GCC/BRE are documented prior-art comparators; no exactly compatible paper-number reproduction is claimed. Camera-blind raw input here differs from camera-CCM, multi-image or foundation-model protocols.",
        "",
        "![Risk coverage](public_risk_coverage.png)",
        "",
        "![Camera transfer](public_camera_transfer.png)",
    ]
    profile = json.loads((OUT / "public_full_path_profile.json").read_text())
    lines += [
        "",
        "## Measured full input path",
        "",
        "Unoptimized FP32 from compressed PNG bytes in RAM, including decode, sensor normalization, mask, resize, expert bank where needed, transfers, CNN, error head and diagonal image correction. Disk I/O and facial analysis excluded.",
        "",
        "| Method | Median ms | p95 ms | CPU expert bank median ms |",
        "|---|---:|---:|---:|",
    ]
    for name, p in profile["results"].items():
        lines.append(
            f"| {LABELS[name]} | {p['full_path_median_ms']:.2f} | {p['full_path_p95_ms']:.2f} | {p['experts_median_ms']:.2f} |"
        )
    lines += [
        "",
        "**Decision: the special mixture has not beaten the strongest baseline on the primary selective endpoint.** The ordinary estimator with combined risk features is better at80/70/60% coverage on the three-seed average. The full-coverage point gain and limited same-sensor camera result are recorded, but do not establish robust superiority. No new patent/novelty claim follows.",
        "",
        "## Reproducibility and protocol accounting",
        "",
        "All eight checkpoints and data-cache hashes are recorded. Training source hashes differ only because Ruff wrapped the Sony decoder line; the exact old full source hash was reconstructed, its Python AST equals the current version, and the [source snapshot/diff](source_snapshots/verification.json) is retained. No model, preprocessing, split or optimizer behavior changed between seed pairs.",
        "",
        "Oracle lower-bound curves and uniform-random expected risk are [diagnostic controls](public_oracle_controls.json), never deployable methods. [Transfer bootstrap](public_transfer_bootstrap.json) adds primary-seed2000-resample Sony-image and Canon-date comparisons. Sony image bootstrap assumes independent examples despite unknown scene grouping; treat it as conditional pilot uncertainty, not a population claim.",
        "",
        "Executed ablations isolate context/disagreement/combined risk features and mixture removal under matched backbones. Not executed: recovery-error-target head, brightness augmentation off, larger1–5M capacity sweep, faithful named-paper training, full scene-independent INTEL-TAU protocol, calibrated physical surfaces. These remain follow-up experiments; the bounded v1 is not an exhaustive model search.",
    ]
    text = re.sub(r"([a-z])([0-9])", r"\1 \2", "\n".join(lines) + "\n")
    text = text.replace("public_protocol_v 1.md", "public_protocol_v1.md")
    (OUT / "public_benchmark_report.md").write_text(text, encoding="utf-8")
    controls = {}
    for name in ["baseline_context", "proposed_combined"]:
        errors = np.array(values(records, name, "source")[0]["errors"])
        controls[name] = {
            "coverage": (np.arange(1, len(errors) + 1) / len(errors)).tolist(),
            "random_expected_risk": float(errors.mean()),
            "oracle_risk": (np.cumsum(np.sort(errors)) / np.arange(1, len(errors) + 1)).tolist(),
            "label": "Seed17 diagnostic oracle (uses test errors, not deployable) and uniform random expected risk",
        }
    write_json(OUT / "public_oracle_controls.json", controls)
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, domain, title in zip(
        axes,
        ["source", "sony_pilot"],
        ["SimpleCube++ official test (462)", "External Sony pilot (30; not full benchmark)"],
    ):
        for name in [
            "gray_world",
            "shades_gray",
            "baseline_context",
            "baseline_combined",
            "proposed_combined",
        ]:
            v = values(records, name, domain)
            curves = np.array([x["selective"]["risk"] for x in v])
            coverage = np.array(v[0]["selective"]["coverage"])
            ax.plot(coverage, curves.mean(0), label=LABELS[name])
        ax.set(
            xlim=(0.1, 1),
            xlabel="Accepted coverage",
            ylabel="Mean reproduction error (degrees)",
            title=title,
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "public_risk_coverage.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    names = ["gray_world", "shades_gray", "baseline_context", "proposed_combined"]
    for j, domain in enumerate(["source", "sony_pilot"]):
        means = [
            np.mean([x["reproduction"]["mean"] for x in values(records, n, domain)]) for n in names
        ]
        ax.bar(
            np.arange(4) + (j - 0.5) * 0.35,
            means,
            width=0.35,
            label="Known-source test" if j == 0 else "Unseen Sony pilot",
        )
    ax.set_xticks(np.arange(4), ["Gray World", "Shades of Gray", "C+", "Proposed"])
    ax.set_ylabel("Mean reproduction error (degrees)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "public_camera_transfer.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"bootstrap": pairs, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
