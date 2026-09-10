"""Build human-readable CC v2 tables from frozen per-image evaluation records."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT = Path("docs/benchmarks/cc_v2")
PRIMARY = "ccv2_sog_large_g0::combined"
MATCHED = "ccv2_direct_large_g0::combined"
METHODS = {
    PRIMARY: "Proposed: SoG residual, combined risk",
    MATCHED: "Strong matched C+: direct, combined risk",
    "ccv2_direct_large_g0::context": "Standard C+: direct, context risk",
    "ccv2_legacy_proposed::upgraded_combined": "V1 mixture, upgraded risk",
    "ccv2_legacy_baseline::upgraded_combined": "V1 direct, upgraded risk",
    "direct_hgb7::combined": "Statistics + HGB, combined risk (1 fit)",
    "gw_ridge1::combined": "GW + ridge, combined risk (1 fit)",
    "gw_ridge1::cheap": "GW + ridge, cheap risk (ablation, 1 fit)",
    "ccv2_legacy_baseline::v1_gray_world": "Gray World + frozen V1 selector",
    "ccv2_legacy_baseline::v1_max_rgb": "Max RGB + frozen V1 selector",
    "ccv2_legacy_baseline::v1_shades_gray": "Shades of Gray + frozen V1 selector",
    "ccv2_legacy_baseline::v1_gray_edge": "Gray Edge + frozen V1 selector",
    "ccv2_gw_large_g0::combined": "GW residual, combined (exploratory, 1 seed)",
}


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def main():
    aggregate = json.loads((ROOT / "aggregate.json").read_text(encoding="utf-8"))
    for name, digest in aggregate["inputs"].items():
        if sha256(ROOT / "runs" / name) != digest:
            raise ValueError("Changed evaluation record: " + name)
    domains = aggregate["domains"]

    def value(domain, group, metric):
        return domains[domain][group][metric]["mean"]

    def records(domain, group):
        return [
            json.loads((ROOT / "runs" / name).read_text(encoding="utf-8"))["domains"][domain][
                group.split("::")[1]
            ]
            for name in domains[domain][group]["source_files"]
        ]

    summary = {}
    for domain in ("source_regression", "fresh_all"):
        summary[domain] = {}
        for group in METHODS:
            rows = records(domain, group)
            stats = {
                metric: {
                    key: float(np.mean([r[metric][key] for r in rows])) for key in rows[0][metric]
                }
                for metric in ("recovery", "reproduction")
            }
            stats["risk80"] = {
                key: float(np.mean([r["selective"]["fixed"]["80"][key] for r in rows]))
                for key in rows[0]["selective"]["fixed"]["80"]
            }
            stats["error_prediction"] = {
                "mean_predicted": float(np.mean([np.mean(r["scores"]) for r in rows])),
                "mae": float(
                    np.mean([np.mean(np.abs(np.array(r["scores"]) - r["errors"])) for r in rows])
                ),
                "spearman": float(
                    np.mean([spearmanr(r["scores"], r["errors"]).statistic for r in rows])
                ),
            }
            summary[domain][group] = stats
    write_json(
        ROOT / "metric_summary.json",
        {
            "aggregate_sha256": sha256(ROOT / "aggregate.json"),
            "script_sha256": sha256(Path(__file__)),
            "domains": summary,
        },
    )

    sections = [
        "# Public camera-transfer improvement — measured CC v2, 2026-09-10",
        """
**The source-selected compact residual method reduces mean reproduction error at 80% accepted coverage on 384 fresh images from three unseen camera models from 5.558° to 3.805° (31.5%) versus the strongest capacity-matched direct C+ control. It does not win on every camera or against every strong alternative.** Source accuracy regresses; Canon transfer loses; a very cheap GW+ridge control remains competitive. This is real component evidence, not facial skin-color validation or a claim of new mathematics/SOTA.

## Data, original licenses and protocol

All estimator, selector and calibration fitting uses **SimpleCube++**, 2,234 real linear images, **CC BY 4.0**. Its official training population is split by capture date into 1,126 estimator training, 119 validation, 259 risk fitting and 268 calibration images. Its 462 official test images were already observed in V1, and 66 of 67 test dates overlap development dates: this is a known-camera/source regression diagnostic, not a fresh scene-independent confirmation.

The fresh test comprises **128 camera-unique field images each from Canon 5DSR, Nikon D810 and Sony IMX135_BLCCSC in INTEL-TAU**. Deterministic archive-path hashes selected IDs before pixels/GT were read; shared-camera/lab categories and the previous Sony 001–030 pilot were excluded. Training sees only source Canon 550D/600D cameras. Sony IMX135 was already observed through the V1 Sony30 pilot; its new128 images are disjoint, but that camera model is not new to the research history. Canon5DSR and NikonD810 are new to development; all three are absent from fitting. No target-camera identity, CCM, extra target images, target-dataset aggregate statistics or adaptation is used at inference. This is a custom three-camera transfer subset, not the authors' full 10-fold protocol or a leave-one-camera-out training study. Known/unseen also changes dataset and scene distribution, so the difference is not solely attributable to camera.

Original INTEL-TAU terms are **CC BY-SA 4.0**, verified through [the original Metax record](https://metax.fairdata.fi/v3/datasets/f0570a3f-3d77-4f44-9ef1-99ab4878f17c). These images are **evaluation-only**; no weights or thresholds were fitted with them. The mirror's MIT badge is ignored. Bounded extraction transferred 4,433,685,559 bytes, not the hundreds-of-GB full collection. Every TIFF/WP pair has frozen mirror CRC, size and local SHA-256 verification. The bounded original-host range probe did not establish whole-archive checksum identity to the pinned mirror; that identity remains **unverified**. See [acquisition and provenance](../data/cc_v2_camera_acquisition.md) and [dataset inventory](../data/public_dataset_inventory.md).

The [original INTEL-TAU paper](https://pure.au.dk/ws/files/301638552/INTEL_TAU_A_Color_Constancy_Dataset.pdf) supports processed linear TIFF decoding and illuminant WP targets. These files are already black-corrected and saturation-normalized; no extra black subtraction or gamma transform is applied. GT chart-acquisition frames are not published. No genuine corresponding surface-color reference is available here: **ΔE00 is not calculated**. Approximate camera CCMs are not substituted for measured reference colorimetry.

## Proposed method and fair controls

The proposed source-selected model uses an unnormalized Shades-of-Gray anchor, a compact residual CNN on channel-normalized input, and restores the anchor to obtain an illuminant. A separately fitted error predictor combines 64 scene-context features and 21 relative hypothesis/spatial features. Its target is held-out log1p reproduction angular error; a separate source split fits a positive scalar calibration. This is an expected-error estimate and a ranking mechanism, not a guaranteed bound on color error.

The normalization/predict/restore construction is known, including [Cotogni and Cusano 2022, Eq. 19](https://arxiv.org/pdf/2207.00292), with direct color-constancy overlap in their 2024 work. Positive-diagonal equivariance is qualified by nondegenerate channels and fixed preprocessing; real sensor spectral differences are not generally diagonal. See [fresh prior art](../research/cc_v2_prior_art.md).

The strongest matched C+ receives the same 3,033,651-parameter MobileNetV3-large backbone, 128×128 input, scratch initialization, source splits, 120 epochs, batch 32, AdamW/cosine budget, gain augmentation=0 and seeds 17/29/43. Both have the same search over five candidate selector models and context/cheap/combined ablations. The ordinary context-only C+ is also retained. The proposed and strong-control combined blocks were locked using source out-of-fold risk before any fresh-target error was evaluated. Calibration data are disjoint from risk fitting.

All numbers below are **REPRODUCED LOCALLY**, averaged across three independently trained seeds unless marked otherwise; this is not a prediction ensemble. Standard learned controls include direct CNNs and statistics→ridge/HGB regressors. No FC4/C5/FFCC weight reproduction or compatible author-number comparison is claimed. Classical full-coverage estimates are identical across selector seeds and use full processed images; their selective curves additionally use frozen V1 learned selectors, which have CNN/expert compute cost.
""",
    ]
    rows = []
    for group, label in METHODS.items():
        rows.append(
            [label]
            + [
                f"{value(d, group, m):.3f}"
                for d, m in [
                    ("source_regression", "mean"),
                    ("source_regression", "risk80"),
                    ("fresh_all", "mean"),
                    ("fresh_all", "risk80"),
                    ("fresh_all", "aurc"),
                ]
            ]
        )
    sections += [
        "## Reproduced comparison (degrees; lower is better)",
        table(
            ["Method", "Source mean", "Source 80%", "Fresh mean", "Fresh 80%", "Fresh AURC"], rows
        ),
    ]
    sections += [
        """
The primary method improves over V1 mixture+upgraded selector on fresh risk80 by 21.2%. However, its source mean/risk80 worsen by 28.5%/39.9% versus strong matched C+. The single-seed GW residual diagnostic has a lower fresh full mean (4.229°) and slightly lower point risk80 (3.745°); it was not promoted after seeing target errors. Cheap GW+ridge also has a lower full mean (4.558°). **There is no universal strongest-baseline victory.**

## Fixed coverage and calibration under shift
""",
        table(
            [
                "Accepted coverage",
                "Proposed",
                "Strong C+",
                "GW+ridge cheap",
                "SoG + learned selector",
            ],
            [
                [f"{c}%"]
                + [
                    f"{value('fresh_all', g, 'risk' + str(c)):.3f}"
                    for g in (
                        PRIMARY,
                        MATCHED,
                        "gw_ridge1::cheap",
                        "ccv2_legacy_baseline::v1_shades_gray",
                    )
                ]
                for c in (100, 95, 90, 80, 70, 60)
            ],
        ),
    ]
    sections += [
        """
Fixed coverage is a retrospective diagnostic using predicted-score ranking, not target-label selection. Counts use floor(N×coverage): nominal80% means307/384=79.95%. A threshold fixed at the source-calibration80% quantile instead accepts **47.66%** of fresh images, with mean accepted error **3.218°**; strong C+ accepts60.16%, error5.540°. Therefore the deployable fixed threshold does not preserve nominal coverage under distribution shift. No finite-sample risk-control guarantee has been implemented.

![Full source and unseen-camera risk-coverage curves](cc_v2/figures/risk_coverage.png)

The curve does not dominate all alternatives at every coverage. In particular, classical SoG with its learned selector is strong at lower coverage. Shading is variability across trained seeds, not a population confidence interval.
"""
    ]
    calrows = []
    for g in (PRIMARY, MATCHED, "gw_ridge1::cheap"):
        r = summary["fresh_all"][g]
        e = r["error_prediction"]
        calrows.append(
            [
                METHODS[g],
                f"{e['mean_predicted']:.3f}",
                f"{r['reproduction']['mean']:.3f}",
                f"{e['mae']:.3f}",
                f"{e['spearman']:.3f}",
            ]
        )
    sections += [
        table(
            ["Fresh error prediction", "Mean predicted°", "Mean observed°", "MAE°", "Spearman"],
            calrows,
        ),
        "These held-out diagnostics are computed without refitting. Mean calibration on source does not establish calibration on new cameras.",
    ]
    sections += [
        "## Cameras unseen during model/selector fitting",
        table(
            ["Camera (128 each)", "Proposed mean", "C+ mean", "Proposed 80%", "C+ 80%"],
            [
                [name]
                + [
                    f"{value('fresh_' + camera, g, m):.3f}"
                    for g, m in (
                        (PRIMARY, "mean"),
                        (MATCHED, "mean"),
                        (PRIMARY, "risk80"),
                        (MATCHED, "risk80"),
                    )
                ]
                for camera, name in (
                    ("Canon_5DSR", "Canon 5DSR"),
                    ("Nikon_D810", "Nikon D810"),
                    ("Sony_IMX135_BLCCSC", "Sony IMX135_BLCCSC"),
                )
            ],
        ),
        "**Canon is a negative result.** Nikon and Sony improve against C+, but the proposed method is not best among all alternatives on each camera. No per-camera threshold was fitted.",
        "![Camera comparison](cc_v2/figures/camera_comparison.png)",
    ]
    sections += ["## Established illumination metrics and catastrophic tails"]
    for metric in ("recovery", "reproduction"):
        keys = ["mean", "median", "trimean", "best25", "worst25", "p90", "p95"]
        sections += [
            f"Fresh {metric} angular error:",
            table(
                ["Method"] + keys,
                [
                    [METHODS[g]] + [f"{summary['fresh_all'][g][metric][k]:.3f}" for k in keys]
                    for g in (PRIMARY, MATCHED, "gw_ridge1::cheap")
                ],
            ),
        ]
    sections += [
        table(
            ["Fresh method", "Full >10°", "80% >10°", "80% >20°", "80% p95°"],
            [
                [
                    METHODS[g],
                    f"{100 * summary['fresh_all'][g]['reproduction']['over10_fraction']:.2f}%",
                    f"{100 * summary['fresh_all'][g]['risk80']['over10_fraction']:.2f}%",
                    f"{100 * summary['fresh_all'][g]['risk80']['over20_fraction']:.2f}%",
                    f"{summary['fresh_all'][g]['risk80']['p95']:.3f}",
                ]
                for g in (PRIMARY, MATCHED, "gw_ridge1::cheap")
            ],
        ),
        "Catastrophic errors remain among accepted examples. Refusal is mandatory for nonfinite, negative and channel-degenerate inputs in the exported interface; the real test contains no such invalid rows, so those contracts are tested with constructed invalid inputs.",
    ]
    sections += [
        """
## Paired uncertainty and mechanism ablations

Paired2,000-draw bootstrap, first-minus-second, conditional on the fitted seeds, gives fresh reproduction mean difference against strong C+ **−0.942° [−1.318,−0.573]** and risk80 difference **−1.753° [−2.132,−1.336]**. These intervals use camera-stratified exact-whitepoint-hash proxy clusters (342 groups); image-level intervals are also saved. Exact reference hashes are only a sensitivity proxy, not verified physical scene IDs. The source uses capture-date clusters. There is no multiple-comparison correction or claim of independence across all real scenes/camera populations.

Against cheap GW+ridge, risk80 difference is **−0.172° [−0.428,+0.147]**: superiority at80% is **not established**. The predeclared descriptive AURC summary is3.392 versus4.190; a secondary AURC interval is−0.798 [−1.287,−0.307]. This supports lower average selective risk over the full curve in this sample but does not turn the80% tie into a win or eliminate the all-coverage accuracy tradeoff.

For the exact same SoG residual estimator, replacing context-only risk with combined relative features improves fresh risk80 from **4.266° to3.805°**, paired difference **−0.461° [−0.610,−0.286]**. Cheap features alone achieve3.863°. Thus there is measured incremental selective value in relative context, while novel error supervision or a new uncertainty principle is not demonstrated.

All17 CNN configurations,6 preserved V1 estimator/control configurations,2 final statistics estimators and594 method/domain records remain in [runs](cc_v2/runs/), [all-results CSV](cc_v2/all_results.csv) and [aggregate](cc_v2/aggregate.json). The original12 statistics candidates and numerically corrected same12-candidate screen are both preserved: standardization amplified roundoff in mathematically zero GW features; the fix and a fitted-model invariance test preceded fresh-error evaluation, and the winning candidate IDs did not change. No failed variant was erased. Source-selection amendments and the final lock are recorded in [protocol](cc_v2/methods_protocol.md).

## Compute and export

The representative seed17 model has **3,033,651 parameters**; checkpoint12,324,047bytes (12.324MB), fitted Ridge head3,758bytes. RTX4060 FP32 batch1 median model-only latency is **4.181ms**. Thumbnail→GPU estimator/features→CPU calibrated head is **6.027ms**. Preloaded648×432 PNG bytes→decode/mask/128thumbnail→score is **24.691ms**, p95 **25.485ms**, excluding disk/ZIP I/O, correction rendering and downstream face processing. These are100 measurements after20 warmups, on Ryzen9 7900X with4 CPU threads; the model receives128×128 input. No4K/phone latency is implied.

Peak PyTorch allocated training memory is **762.34MiB**, including233.45MiB source tensors cached on GPU; peak reserved852MiB. Inference peak allocated28.12MiB, reserved36MiB. These are allocator measurements, **not total CUDA-context/driver/process VRAM**. The matched C+ model-only median is4.083ms and PNG→score30.307ms; sequential CPU timing drift and different fitted head types prevent a robust speed-superiority claim. Raw measurements and exact script snapshot are in [profile](cc_v2/profiles/profile.json).

The FP32 ONNX is12,141,768bytes. ONNX Runtime CPU real-input parity passed with maximum score difference4.30e-6° and identical acceptance flags; constructed invalid inputs were refused. See [export receipt](cc_v2/profiles/export_report.json) for graph/dtype checks and exact vectors. The compact estimator plus Ridge score and fixed threshold is exported after positive matched evidence. **TensorRT, FP16/INT8 accuracy and production smartphone integration remain NOT MEASURED.**

## What this proves for Luma and the next decision

We validated the photometric normalization/reliability core of the future Luma Skin Vision Engine on established color-constancy data with real ground truth, within a documented custom transfer protocol. The measured effect is lower selective reproduction error against matched direct CNN controls using one image and no test-camera calibration or large model. It does not prove arbitrary camera independence, reliable absolute color reconstruction, physically calibrated skin Lab/ΔE00, skin-specific repeatability, cosmetics recommendation quality, or patentability.

Prioritize **C: compact single-image camera transfer**, with relative-feature risk estimation as the measured B ablation and reproduction risk as the limited A proxy. The known equivariant wrapper is an engineering choice, not an invention. Keep the cheap GW+ridge and GW residual controls as serious alternatives. The next experimental milestone should use independent multi-camera **source validation**, strong compact published-method reproduction (e.g. FFCC under separately verified code/data terms), and a new frozen test population; investigate the Canon failure without tuning this now-observed384-image test. Do not prematurely ship a universal correction policy. Skin-specific colorimetric validation remains future work requiring a facial dataset with appropriate reference measurements; its current unavailability does not block this public-data track.

The earlier negative synthetic state2685bf0 and V1 public state7637d6d remain frozen under their original milestone tags. This report supersedes neither their measurements nor their conclusions. [Current status](../research/CURRENT_RND_STATUS.md), [Skolkovo component evidence](../skolkovo/cc_v2_evidence_addendum.md), [reproduction instructions](cc_v2/REPRODUCE.md), [independent metric audit](cc_v2/independent_metric_review.md).
"""
    ]
    Path("docs/benchmarks/cc_v2_report.md").write_text(
        "\n\n".join(s.strip() for s in sections) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
