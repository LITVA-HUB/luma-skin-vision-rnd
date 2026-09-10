"""Audit source-screen receipts and plot measured development-validation results."""

import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
from cc_v2_statistics import read_npz_rows

from luma_skin_vision.cc.core import selective_curve, summarize
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def independent_errors(pred, gt):
    if not np.isfinite(pred).all() or np.any(pred <= 0):
        raise ValueError("Positive finite diagnostic predictions required")
    ratio = gt / pred
    neutral = np.ones_like(ratio) / np.sqrt(3)
    recovery = np.degrees(
        np.arctan2(np.linalg.norm(np.cross(pred, gt), axis=-1), (pred * gt).sum(-1))
    )
    reproduction = np.degrees(
        np.arctan2(np.linalg.norm(np.cross(ratio, neutral), axis=-1), (ratio * neutral).sum(-1))
    )
    return recovery, reproduction


def main():
    target = Path("docs/benchmarks/cc_v3/source_screen")
    if target.exists():
        raise FileExistsError("Preserve existing screen evidence")
    data = Path("data/processed/cc128")
    rows = json.loads((data / "cube_manifest.json").read_text(encoding="utf-8"))
    ix = np.array([i for i, r in enumerate(rows) if r["subset"] == "val"])
    ids = [rows[i]["id"] for i in ix]
    # Recreate the runner's explicit FP32 label conversion, then audit in FP64.
    gt = read_npz_rows(data / "cube.npz", "gt", ix, 2234).astype(np.float32).astype(np.float64)
    expert = read_npz_rows(data / "cube.npz", "experts", ix, 2234).astype(np.float64)
    baseline = {}
    for i, name in enumerate(("Gray World", "Max RGB", "Shades of Gray", "Gray Edge")):
        recovery, reproduction = independent_errors(expert[:, i], gt)
        baseline[name] = {
            "reproduction": summarize(reproduction),
            "recovery": summarize(recovery),
            "scope": "Previously reproduced full-image classical estimator on same119validation images; neural inputs are128px thumbnails",
        }
    methods, histories, sources, configs = {}, {}, {}, []
    for mode in ("direct", "diagonal", "frame"):
        run = Path(f"experiments/runs/ccv3_{mode}_e120_s17")

        def read(name):
            return json.loads((run / name).read_text(encoding="utf-8"))

        config, result, reported = (
            read("config.json"),
            read("result.json"),
            read("best_metrics.json"),
        )
        if config["validation_ids"] != ids or result["status"] != "complete":
            raise ValueError("Incomplete run or mismatched validation identities")
        if config["data_hashes"] != {name: sha256(data / name) for name in config["data_hashes"]}:
            raise ValueError("Source data changed")
        if result["checkpoint_sha256"] != sha256(run / "best.pt"):
            raise ValueError("Checkpoint changed")
        for name, digest in config["scripts_sha256"].items():
            if sha256(run / "source_snapshot/scripts" / name) != digest:
                raise ValueError("Executable snapshot changed")
        history = [
            json.loads(line)
            for line in (run / "history.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if len(history) != 121 or [h["epoch"] for h in history] != list(range(121)):
            raise ValueError("Not a full120epoch comparison")
        if any(
            history[0][k] is not None
            for k in ("train_reproduction", "train_nll", "train_positivity")
        ):
            raise ValueError("Untrained losses must be unmeasured")
        eligible = [h for h in history if h["validation"]["valid_fraction"] >= 0.99]
        best = min(eligible, key=lambda h: h["validation"]["reproduction"]["mean"])
        if best["epoch"] != result["best_epoch"] or best["epoch"] != reported["epoch"]:
            raise ValueError("Wrong checkpoint selection")
        with np.load(run / "best_validation.npz", allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        recovery, reproduction = independent_errors(arrays["pred"].astype(np.float64), gt)
        for name, error in (("recovery", recovery), ("reproduction", reproduction)):
            if not np.allclose(error, arrays[name], rtol=0, atol=1e-6):
                raise ValueError("Independent angle recomputation differs")
            for key, value in summarize(error).items():
                if not np.isclose(value, reported[name][key], rtol=0, atol=1e-6):
                    raise ValueError("Reported summary differs")
        if not arrays["valid"].all():
            raise ValueError(
                "Adapt report for unsupported coverage before plotting; do not hide refusals"
            )
        curve = selective_curve(reproduction, arrays["transport_risk"], ids)
        methods[mode] = {
            "result": result,
            "best_metrics": reported,
            "uncalibrated_validation_risk_curve": curve,
            "risk_limitation": "Raw approximate transported posterior proxy on reused development validation, with checkpoint selection on this set. No calibrated reliability or independent test claim.",
            "files_sha256": {
                name: sha256(run / name)
                for name in (
                    "config.json",
                    "result.json",
                    "history.jsonl",
                    "best_metrics.json",
                    "best_validation.npz",
                )
            },
        }
        histories[mode], sources[mode] = history, run
        comparable = dict(config["arguments"])
        comparable.pop("mode")
        comparable.pop("out")
        configs.append(
            (comparable, config["scripts_sha256"], config["train_ids"], config["loss_weights"])
        )
    if any(c != configs[0] for c in configs[1:]):
        raise ValueError("Three mode budgets/source/data differ")
    target.mkdir(parents=True)
    for mode, run in sources.items():
        destination = target / mode
        destination.mkdir()
        for name in (
            "config.json",
            "result.json",
            "history.jsonl",
            "best_metrics.json",
            "best_validation.npz",
        ):
            shutil.copyfile(run / name, destination / name)
    source_dir = target / "scripts_snapshot"
    source_dir.mkdir()
    for name in configs[0][1]:
        shutil.copyfile(sources["direct"] / "source_snapshot/scripts" / name, source_dir / name)
    write_json(
        target / "summary.json",
        {
            "scope": "REAL SimpleCube++ illuminant GT, CC BY4.0;1126training/119reuseddevelopment validation; one seed17; no test or INTEL labels decoded",
            "classical": baseline,
            "methods": methods,
            "validation_ids": ids,
            "audit": "Independent atan2 recovery/reproduction from source GT; all prediction/summary defects<=1e-6degree; exact best-epoch, IDs,120epochs, code/data/budget/hash equality checked",
            "report_script_sha256": sha256(Path(__file__)),
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    colors = {"direct": "#0072B2", "diagonal": "#009E73", "frame": "#D55E00"}
    for mode, method in methods.items():
        hist = histories[mode]
        axes[0].plot(
            [h["epoch"] for h in hist],
            [h["validation"]["reproduction"]["mean"] for h in hist],
            label=mode,
            color=colors[mode],
            linewidth=1.2,
        )
        curve = method["uncalibrated_validation_risk_curve"]
        axes[1].plot(
            np.array(curve["coverage"]) * 100, curve["risk"], label=mode, color=colors[mode]
        )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
        ax.set_ylabel("Mean reproduction error (degrees)")
    axes[0].set_xlabel("Training epoch")
    axes[0].set_title("Matched 1.22M graph architectures")
    axes[1].set_xlabel("Accepted validation images (%)")
    axes[1].set_xlim(10, 100)
    axes[1].set_title("Uncalibrated posterior risk ordering")
    fig.suptitle("SimpleCube++ development validation (n=119, seed17) — not an independent test")
    fig.savefig(target / "source_screen.png", dpi=160)
    fig.savefig(target / "source_screen.svg")
    plt.close(fig)
    print(
        json.dumps(
            {
                mode: {
                    "mean": m["best_metrics"]["reproduction"]["mean"],
                    "risk80": m["uncalibrated_validation_risk_curve"]["fixed"]["80"]["mean"],
                    "best_epoch": m["result"]["best_epoch"],
                }
                for mode, m in methods.items()
            }
        )
    )


if __name__ == "__main__":
    main()
