"""Publish a LOCAL synthetic engineering report, never a real accuracy claim."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from luma_skin_vision.evaluation import bootstrap_selective

ROOT = Path(__file__).resolve().parents[1]


def main():
    latest = json.loads((ROOT / "artifacts/latest_smoke.json").read_text())
    summary = json.loads(Path(latest["summary"]).read_text())
    out = ROOT / "docs/benchmarks"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    for method, run in summary["runs"].items():
        data = json.loads((ROOT / run / "evaluation_test.json").read_text())
        curve = data["full_risk_coverage"]
        ax.plot([r["coverage"] for r in curve], [r["mean_delta_e00"] for r in curve], label=method)
    ax.set(
        xlabel="Accepted image coverage (offline ranking)",
        ylabel="Mean CIEDE2000 error",
        title="SYNTHETIC ENGINEERING SMOKE — NOT REAL SKIN ACCURACY",
        xlim=(0, 1),
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.savefig(out / "synthetic_risk_coverage.png", dpi=150)
    fig.savefig(out / "synthetic_risk_coverage.svg")
    plt.close(fig)
    records = {}
    for method in ("baseline_c_plus", "proposed_v1"):
        records[method] = json.loads(
            (ROOT / summary["runs"][method] / "predictions_test.json").read_text()
        )
    a, b = records["baseline_c_plus"], records["proposed_v1"]
    assert [r["record_id"] for r in a] == [r["record_id"] for r in b]
    comparison = bootstrap_selective(
        np.array([r["prediction"] for r in a]),
        np.array([r["prediction"] for r in b]),
        np.array([r["target"] for r in a]),
        [r["predicted_error"] for r in a],
        [r["predicted_error"] for r in b],
        [r["record_id"] for r in a],
        [r["subject_id"] for r in a],
        image_ids=[r["image_id"] for r in a],
    )
    evidence = {
        "evidence_kind": "SYNTHETIC",
        "comparison_cplus_minus_proposed": comparison,
        "smoke_summary": summary,
    }
    (out / "synthetic_smoke_evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "plot": str(out / "synthetic_risk_coverage.png"),
                "evidence": str(out / "synthetic_smoke_evidence.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
