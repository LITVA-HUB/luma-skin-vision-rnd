"""All-dose GS report; synthetic boundary sensitivity does not measure failure frequency."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from chromaseed_kernel_audit import js, nz
from skin_local_search_train import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
ROLES = {"mixed": "Mixed", "slr_to_ipod": "SLR → iPod", "ipod_to_slr": "iPod → SLR"}
FAMILIES = tuple(
    f"{b}_{r}" for b in ("norm", "perceptual") for r in ("base", "uniform", "soft", "hard")
)
DOSES = np.array([1, 4, 16, 64]) / 255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run, args.output
    result, audit = js(run / "results.json"), js(out / "audit.json")
    assert audit["passed"] and audit["results_sha256"] == sha(run / "results.json")
    assert (
        audit["source_lock_sha256"] == result["source_lock_sha256"] == sha(run / "source_lock.json")
    )
    assert audit["audit_source_sha256"] == sha(ROOT / "scripts/chromaseed_gate_stability_audit.py")
    for path, expected in audit["dependencies"].items():
        assert sha(ROOT / path) == expected
    rows = []
    for role in ROLES:
        for family in FAMILIES:
            rr = [r for r in result["records"] if r["role"] == role and r["family"] == family]
            assert len(rr) == 3
            rows.append(
                dict(
                    role=role,
                    family=family,
                    active_gate=rr[0]["active_gate"],
                    clean_person_mean=float(
                        np.mean([r["transforms"][0]["error"]["person_mean"] for r in rr])
                    ),
                    doses=[
                        dict(
                            dose=float(t),
                            **{
                                k: float(np.mean([r["doses"][i][k]["person_mean"] for r in rr]))
                                for k in ("worst_error", "worst_drift", "worst_error_change")
                            },
                            any_sign_flip=None
                            if not rr[0]["active_gate"]
                            else float(
                                np.mean([r["doses"][i]["any_sign_flip"]["person_mean"] for r in rr])
                            ),
                        )
                        for i, t in enumerate(DOSES)
                    ],
                )
            )
    boundary = []
    for base in ("norm", "perceptual"):
        bb = [b for b in result["boundaries"] if b["base"] == base]
        for control in ("base", "uniform", "soft", "hard"):
            boundary.append(
                dict(
                    base=base,
                    control=control,
                    mean_person_jump=float(
                        np.mean([b["jumps"][control]["person_mean"] for b in bb])
                    ),
                    mean_image_jump=float(np.mean([b["jumps"][control]["image_mean"] for b in bb])),
                    mean_p90_jump=float(np.mean([b["jumps"][control]["p90"] for b in bb])),
                    maximum_jump=float(max(b["jumps"][control]["maximum"] for b in bb)),
                )
            )
    files = [
        nz(run / "boundary" / f"{b}_hard_s{s}.npz")
        for b in ("norm", "perceptual")
        for s in (17, 29, 43)
    ]
    first = files[0]
    for arr in files[1:]:
        for key in (
            "row_indices",
            "query_ordinal",
            "anchor_index",
            "root_t",
            "lower_x",
            "upper_x",
            "crossed",
        ):
            np.testing.assert_array_equal(arr[key], first[key])
    roots = first["root_t"]
    geometry = dict(
        unique_query_anchor_pairs=len(roots),
        distinct_query_rows=len(np.unique(first["query_ordinal"])),
        root_quantiles_0_05_50_95_100=np.quantile(roots, [0, 0.05, 0.5, 0.95, 1]).tolist(),
        roots_below_doses=[int((roots <= t).sum()) for t in DOSES],
        all_six_boundary_input_sets_exactly_equal=True,
    )
    mixed = [r for r in rows if r["role"] == "mixed" and r["family"].startswith("perceptual")]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    colors = ["#335A85", "#8D8D8D", "#168272", "#C46732"]
    for r, color in zip(mixed, colors, strict=True):
        values = [r["clean_person_mean"]] + [d["worst_error"] for d in r["doses"]]
        axes[0, 0].plot(range(5), values, "o-", label=r["family"].split("_")[1], color=color)
    axes[0, 0].set_xticks(range(5), ["original", "0.392%", "1.569%", "6.275%", "25.098%"])
    axes[0, 0].set_ylabel("Worst-of-8 anchors, then person mean ΔE00")
    axes[0, 0].set_title("All controls remain sensitive to synthetic color changes")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_xlabel("Encoded-RGB mixture strength; dose spacing is categorical")
    bb = [b for b in boundary if b["base"] == "perceptual"]
    for i, b in enumerate(bb):
        axes[0, 1].scatter(b["mean_person_jump"], i, color=colors[i], s=50)
        axes[0, 1].plot([b["mean_person_jump"], b["maximum_jump"]], [i, i], color=colors[i], lw=2)
        axes[0, 1].scatter(b["maximum_jump"], i, color=colors[i], marker="|", s=100)
    axes[0, 1].set_yticks(range(4), [b["control"] for b in bb])
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Output jump ΔE00 · dot: person mean; endpoint: maximum")
    axes[0, 1].set_title("Near gate boundary: hard output jumps; smooth controls do not")
    axes[1, 0].hist(100 * roots, bins=np.linspace(0, 26, 14), color="#335A85", edgecolor="white")
    axes[1, 0].axvline(100 * np.median(roots), color="#C46732", ls="--", label="median15.69%")
    axes[1, 0].set_xlabel("Mixture strength needed to reach constructed boundary (%)")
    axes[1, 0].set_ylabel("Query/anchor pairs, repeated inputs counted once")
    axes[1, 0].set_title("534 pairs /185 source rows /6 people; most roots are farther away")
    axes[1, 0].legend(frameon=False)
    flips = mixed[-1]["doses"]
    vv = [d["any_sign_flip"] * 100 for d in flips]
    axes[1, 1].bar(range(4), vv, color="#C46732", width=0.55)
    for i, value in enumerate(vv):
        axes[1, 1].text(i, value + 1.2, f"{value:.2f}%", ha="center")
    axes[1, 1].set_xticks(range(4), ["0.392%", "1.569%", "6.275%", "25.098%"])
    axes[1, 1].set_ylim(0, 95)
    axes[1, 1].set_xlabel("Registered mixture strength")
    axes[1, 1].set_ylabel("Any sign flip among8 anchors, equal-person mean (%)")
    axes[1, 1].set_title("Fixed-dose gate flips are rare at the smallest changes")
    for ax in axes.flat:
        ax.grid(alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Luma ChromaSeed-GS · Frozen-model synthetic stability", fontsize=15, x=0.025, ha="left"
    )
    fig.text(
        0.025,
        0.012,
        "Reused mixed role, six people. Boundary-pair RGB separation ≤0.0002; inputs may first require a much larger color change. No phone-face validation.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    for suffix in ("png", "svg"):
        fig.savefig(out / f"gate_stability.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    lines = [
        "# ChromaSeed-GS: frozen-model color and gate stability",
        "",
        "**Hard routing adds a real numerical discontinuity, while overall color sensitivity affects every control.** On534 deliberately constructed legal boundary pairs, the perceptual hard model's person-mean jump is1.9279 ΔE00 versus0.0110 for soft, with a maximum7.8176 across seeds. The pair separation in encoded RGB is at most0.0002. However, most boundaries require a larger initial transformation: median15.69% color mixture from the original input. This is a mechanism stress test, not an estimate of everyday failure probability.",
        "",
        "No models were fitted or selected. All72 [G models](../chromaseed_gated_v1/report.md) were frozen before33 transformations each. Original TRAIN only,24 people overall, mixed held subset232 rows/six people; SLR→iPod643/16 and reverse323/8. These reused roles overlap. Three seed results average errors/statistics and do not create new people or an inference ensemble.",
        "",
        "## Transformation and interpretation",
        "",
        "Apply v'=(1−t)v+t*a to encoded RGB pixels, where a is one of the eight cube corners and t is1/255,4/255,16/255 or64/255, plus one identity. Every pixel stays in[0,1] without clipping. Quantiles/means receive the same affine map, standard deviations scale by1−t, correlations stay unchanged. Tests compare this feature update with explicit synthetic pixel transformations, including constant channels. Actual photos are not decoded or modified here.",
        "",
        "For each dose, take the worst error or output drift across eight anchors separately for each row, then average within people and across people. Different rows may have different worst anchors. The original instrument target stays fixed under synthetic corruption; that is a stress-test assumption, not new instrument measurements of transformed photographs.25.098% is an especially strong stress condition. All doses and controls are retained, including when hard remains better on a dose-wise mean.",
        "",
        "![Stability diagnosis](gate_stability.png)",
        "",
        "## All families: clean and worst-case reference error",
        "",
        "Values are mean per-person ΔE00 averaged across seeds. Smaller is better. Synthetic worst-case scores must not be compared as if they were ordinary held-photo accuracy. Routed transfer models have no gate and are exact base aliases because the original fit had a single camera group.",
        "",
        "| Role | Family | Original | 1/255 | 4/255 | 16/255 | 64/255 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        values = [r["clean_person_mean"]] + [d["worst_error"] for d in r["doses"]]
        lines.append(
            f"| {ROLES[r['role']]} | {r['family']} | "
            + " | ".join(f"{v:.5f}" for v in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## All families: worst-case answer drift from original",
            "",
            "Same aggregation; these values measure changing answers, not their correctness. A constant predictor would have zero drift but could be useless. Therefore stability cannot replace the original target-error controls.",
            "",
            "| Role | Family | 1/255 | 4/255 | 16/255 | 64/255 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        lines.append(
            f"| {ROLES[r['role']]} | {r['family']} | "
            + " | ".join(f"{d['worst_drift']:.5f}" for d in r["doses"])
            + " |"
        )
    lines.extend(
        [
            "",
            "At16/255, the perceptual base worst-case error is9.8045, soft9.8064 and hard9.8081. Soft's continuity removes the routing jump but does not solve the common color-shift problem. At1/255 and4/255, hard's worst-case person means remain slightly better than soft's; no blanket measured dominance is claimed.",
            "",
            "## Fixed-dose sign changes",
            "",
            "The active soft and hard gates share the same frozen score. Their sign-flip indicator is identical, but soft predictions vary continuously. There is no gate in the other role payloads; absence of gate metrics is not proof of unfamiliar-camera robustness.",
            "",
            "| Strength | Query rows flipping under any anchor | Equal-person mean | SLR / iPod person means | Boundary roots below strength |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    ref = next(
        r
        for r in result["records"]
        if r["role"] == "mixed" and r["family"] == "perceptual_hard" and r["seed"] == 17
    )
    for i, d in enumerate(ref["doses"]):
        f = d["any_sign_flip"]
        group = f["camera_person_mean"]
        lines.append(
            f"| {d['dose'] * 255:.0f}/255 | {round(f['image_mean'] * f['rows'])}/{f['rows']} | {f['person_mean'] * 100:.3f}% | {group['SLR'] * 100:.3f}% / {group['ipod'] * 100:.3f}% | {geometry['roots_below_doses'][i]} |"
        )
    lines.extend(
        [
            "",
            "## Legal boundary pairs",
            "",
            "Find the affine gate-score root along every registered anchor direction, then query t−0.0001 and t+0.0001 only when both lie within the registered0–64/255 path. All534 pairs cross the actual FP32 gate. All six model/seed input-pair arrays match exactly, so3,204 pair evaluations represent534 unique query/anchor pairs,185 original rows and six people. Directions and rows repeat people; no independent-sample uncertainty claim is made.",
            "",
            "Root mixture quantiles minimum/5th/median/95th/maximum: "
            + ", ".join(f"{v * 100:.3f}%" for v in geometry["root_quantiles_0_05_50_95_100"])
            + ". Only two roots are below1/255 and five below4/255. The tiny separation of the pair is not its distance from the original image, and the continuous feature construction is not a tested8-bit/JPEG/camera pipeline.",
            "",
            "| Base loss | Output control | Person-mean jump | Image-mean jump | Mean seed p90 | Maximum across seeds |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for b in boundary:
        lines.append(
            f"| {b['base']} | {b['control']} | {b['mean_person_jump']:.6f} | {b['mean_image_jump']:.6f} | {b['mean_p90_jump']:.6f} | {b['maximum_jump']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Boundary summaries average pair errors within each person, then across people, then seeds. All four controls receive identical input pairs. No target label or output-error maximization was used to find these roots.",
            "",
            "## Unconstrained feature-space boundaries",
            "",
            "Projection to the nearest normalized color36 hyperplane is a separate diagnostic. A projected feature vector can violate quantile/bounds/variance requirements, and passing those necessary checks still does not prove a realizable image. The RMS distance is not a certified pixel robustness radius. Hard's analytic two-sided output limit is evaluated at one fixed projected kernel vector.",
            "",
            "| Loss | Seed | Mean person RMS distance | Rows passing basic checks | Mean person limit jump | Max limit jump |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for b in result["boundaries"]:
        legal = b["unconstrained_basic_bounds"]
        lines.append(
            f"| {b['base']} | {b['seed']} | {b['unconstrained_distance']['person_mean']:.6f} | {round(legal['image_mean'] * legal['rows'])}/{legal['rows']} | {b['unconstrained_jump']['person_mean']:.6f} | {b['unconstrained_jump']['maximum']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Retain the soft gate as the continuous research option alongside its shared base; keep hard as a measured comparison rather than declaring it the default because of a0.0100 clean-error advantage on six people. This is an engineering continuity choice, not a new claim of superior phone accuracy. Numeric payload21,973B and prior~0.012ms prepared-feature response are unchanged; GS does not refit weights or remeasure production latency.",
            "",
            "The next actual learning experiment should address sensitivity with mild fit-side affine color augmentation and an analytical compact readout. Include identical no-augmentation and stronger-ridge controls, preserve the clean-error references and compare to a constant-target baseline so merely flattening the predictor cannot count as success. Keep augmentation generation/weights/selection entirely inside person-disjoint fit folds. Soft conditional readouts and static controls should share the same128-center basis; no query-time learning or synthetic-to-real guarantee. This next series is planned, not launched.",
            "",
            "## Verification and limits",
            "",
            f"Eight numerical tests passed in1.51s after import formatting and before the primary run. The primary diagnostic took{result['elapsed_seconds']:.3f}s excluding interpreter imports and auditing; this is not training or deployment time. Primary PID39468 returned exit0. Independent audit session1166 returned exit0 and took{audit['elapsed_seconds']:.3f}s:72 models,2,376 transform cases/948,816 actual standalone query predictions,288 dose summaries with every person/camera metric,3,204 legal pair evaluations/25,632 matched control predictions and1,392 unconstrained projections.",
            "",
            f"Maximum ordinary native-Lab prediction drift {audit['maxima']['ordinary_prediction']:.3g}; boundary prediction drift {audit['maxima']['boundary_prediction']:.3g}; root drift {audit['maxima']['root']:.3g}; normalized projection drift {audit['maxima']['projection']:.3g}. The audit uses separate feature algebra, the verified NumPy-only consumer, direct kernels and independent metric/projection constructions, while sharing the verified ΔE00 formula and original query indexing. All33 zero/positive transformations and all registered strengths remain available in the local result archive.",
            "",
            "Only the original TRAIN arrays and frozen G artifacts were read. No images/tokens, legacy validation/calibration/test, downloads, new measured target pairs, delegation or publication. Camera groups contain different people; synthetic transforms do not establish real acquisition causality, demographic coverage or ordinary-phone face/shade-match accuracy. The full user goal remains active.",
            "",
            "[Protocol](../../research/chromaseed_gate_stability_v1_protocol.md) · [Reproduce](reproduce.md) · [Audit](audit.json) · [Verification](verification.json) · [Summary](summary.json) · [Next learning decision](../../research/chromaseed_gate_stability_next_decision.md)",
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
            rows=rows,
            boundary=boundary,
            geometry=geometry,
            independent_audit_checks=audit["checks"],
            goal_status="active",
            evidence="synthetic frozen-model sensitivity; no new model or real-phone validation",
        ),
    )
    print("GS all-dose report, figure and summary written.")


if __name__ == "__main__":
    main()
