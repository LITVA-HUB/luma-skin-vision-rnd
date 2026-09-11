"""Archive a completed V7 seed with standard source-risk and virtual-sensor results."""
import argparse
import json
import shutil
from pathlib import Path

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT=Path(__file__).resolve().parents[1]


def archive(source,target):
    for path in source.rglob("*"):
        relative=path.relative_to(source)
        if not path.is_file() or "source_snapshot" in relative.parts or path.suffix not in (".json",".jsonl",".npz",".md"):
            continue
        dest=target/relative
        if dest.exists():
            if sha256(dest)!=sha256(path):
                raise ValueError("Archive conflict")
        else:
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,dest)


def run(seed):
    base=ROOT/"docs/benchmarks/cc_v7"
    source=ROOT/f"experiments/runs/ccv7_semantic_s{seed}"
    risk=ROOT/f"experiments/runs/ccv7_semantic_risk_s{seed}"
    verification=json.loads((base/f"seed{seed}_verification.json").read_text())
    if verification["status"]!="ALL V7 CHECKPOINTS AND INDEPENDENT ERRORS PASSED" or len(verification["records"])!=10:
        raise ValueError("Missing complete replay")
    result=json.loads((source/"result.json").read_text())
    risks=json.loads((base/f"seed{seed}_source_risk/results.json").read_text())
    stress=json.loads((base/f"seed{seed}_stress/results.json").read_text())
    archive(source,base/f"seed{seed}")
    archive(risk,base/f"seed{seed}_risk")
    by_risk={r["arm"]:r["heads"]["combined"] for r in risks["records"]}
    by_stress={r["model"]:r["metrics"] for r in stress["records"]}
    lines=[f"# V7 seed {seed}: compact semantic/sensor training", "", "Completed source-development screen, not fresh real-camera evidence. Five", "matched 120-epoch arms, 1126 real SimpleCube++ TRAIN / 119 reused VAL rows.", "The teacher uses only TRAIN views; no target-camera data or identity enters", "estimator fitting. All five first-epoch CUDA replays are bitwise equal.", "All ten best/final checkpoints passed independent CPU replay and FP64 rescoring.", "Removing the training-only projection leaves predictions bitwise unchanged.", "", "|Training arm|Best full mean °|Final full mean °|Combined risk80 °|", "|---|---:|---:|---:|"]
    for arm,r in result["arms"].items():
        fixed=by_risk[arm]["selective"]["fixed"]
        lines.append(f"|{arm}|{r['best_mean_reproduction']:.4f}|{r['last_mean_reproduction']:.4f}|{fixed['80']['mean']:.4f}|")
    candidate=result["arms"]["canonical_teacher_sensor"]["best_mean_reproduction"]
    matched=result["arms"]["raw_teacher_sensor"]["best_mean_reproduction"]
    lines += ["",f"Canonical-teacher minus raw-teacher C+ full mean: {candidate-matched:+.4f}°.","A negative difference favors the candidate. This is a single training seed on","reused validation; neither direction establishes novelty or external transfer.","A target corrected by known training illumination is an approximate camera-RGB","view, not a measured intrinsic surface color or facial Lab reference.","","Standard risk heads use 259 held-out RISK rows with five capture-group folds;","268 date-disjoint CAL rows provide a positive scale and fixed thresholds.","Each arm has the same five candidate heads for context, cheap and combined","features. Combined is predeclared primary; the others are archived ablations.","Below, nominal 80% coverage accepts 95/119 = 79.83% per model. All curves and","source-calibrated threshold coverages are in the machine-readable results.","","|Coverage|GT native|GT sensor|Raw teacher + sensor C+|Canonical teacher + sensor|Canonical teacher native|","|---|---:|---:|---:|---:|---:|"]
    for c in (100,95,90,80,70,60):
        values="|".join(f"{by_risk[a]['selective']['fixed'][str(c)]['mean']:.4f}" for a in result["arms"])
        lines.append(f"|{c}%|{values}|")
    lines += ["","Virtual sensor diagnostic; these transformed source images are NOT captures","from real unseen cameras. Original FP64 GT is transformed by the same matrix.","The training/evaluation table above uses the stored FP32 GT promoted to FP64;","that declared precision difference is separately audited, not a model gain.","","|Best checkpoint|Native °|Red gain °|Blue gain °|Small mix °|Training-range mix °|Extrapolation mix °|","|---|---:|---:|---:|---:|---:|---:|"]
    for name,m in by_stress.items():
        if name.endswith("_last"):
            continue
        values="|".join(f"{v['reproduction']['mean']:.4f}" for v in m.values())
        lines.append(f"|{name}|{values}|")
    lines += ["","The archived diagnostics also report prediction equivariance errors, p95 and","catastrophic tails. The preserved SoG baseline must pass the diagonal-gain","consistency check. Zero equivariance error does not imply correct illumination.","All matrix cases and best/final results were retained, including regressions.","","Model: 3,033,651 deployment parameters plus 369,024 training-only projection","parameters. The saved training checkpoint is 13,816,203 bytes; it includes the","projection. Seed17 training peak is 932.71–933.83 MiB including 233.45 MiB source","cache, 116.72 MiB teacher caches and the initial state. Other seeds have their","own allocation records. V7 deployment file size, latency and ONNX/TensorRT","results are NOT MEASURED. The teacher itself is absent at inference.","","Source data: SimpleCube++, original CC BY4.0. Teacher: standard original","DINOv2-S/14, Apache2.0; pretrained-image overlap cannot be independently audited.","Its train-only feature cache took 238.88 seconds on CPU. Three fixed train","images across four view types replayed exactly; all cache rows have hash/ID","bindings. Images/teacher features/weights are excluded from this Git archive.","","Next decision remains the prespecified three-seed comparison and frozen","317-row real INTEL-TAU transfer population (384-row sensitivity). No new target","pixels or GT values have been decoded for this report. No arbitrary-camera,","ordinary phone JPEG/HEIC, facial skin accuracy or finite-sample risk guarantee."]
    destination=base/f"seed{seed}_report.md"
    destination.write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(base/f"seed{seed}_report_sources.json",{"script_sha256":sha256(Path(__file__)),"training_result_sha256":sha256(source/"result.json"),"source_risk_sha256":sha256(base/f"seed{seed}_source_risk/results.json"),"stress_sha256":sha256(base/f"seed{seed}_stress/results.json"),"report_sha256":sha256(destination)})
    print(str(destination))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,required=True)
    run(parser.parse_args().seed)
