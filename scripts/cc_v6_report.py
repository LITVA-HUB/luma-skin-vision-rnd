"""Archive all completed V6 seeds and summarize the negative combination screen."""
import json
import shutil
from pathlib import Path

import numpy as np

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT=Path(__file__).resolve().parents[1]


def run():
    target=ROOT/"docs/benchmarks/cc_v6"
    records=[]
    for seed in (17,29,43):
        source=ROOT/f"experiments/runs/ccv6_sog_s{seed}"
        verification=json.loads((target/f"seed{seed}_verification.json").read_text())
        if verification["status"]!="ALL HASHES AND REPLAYS PASSED" or len(verification["records"])!=8:
            raise ValueError("Missing independent replay")
        manifest=json.loads((source/"artifact_manifest.json").read_text())
        for name,digest in manifest["sha256"].items():
            if sha256(source/name)!=digest:
                raise ValueError("Changed training artifact")
        result=json.loads((source/"result.json").read_text())
        for arm,record in result["arms"].items():
            metric=json.loads((source/arm/"best_metrics.json").read_text())
            records.append({"seed":seed,"arm":arm,**record,"raw_fixed_coverage":metric["stages"]["2"]["risk"]["fixed"] if arm!="point" else None})
        for path in source.rglob("*"):
            relative=path.relative_to(source)
            if not path.is_file() or "source_snapshot" in relative.parts or path.suffix==".pt":
                continue
            if path.suffix not in (".json",".jsonl",".npz",".md"):
                continue
            dest=target/f"seed{seed}"/relative
            if dest.exists():
                if sha256(dest)!=sha256(path):
                    raise ValueError("Existing archive differs")
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,dest)
    summaries={}
    for arm in result["arms"]:
        values=[r for r in records if r["arm"]==arm]
        summaries[arm]={"best_mean":float(np.mean([r["best_mean_reproduction"] for r in values])),"final_mean":float(np.mean([r["last_mean_reproduction"] for r in values])),"risk":{str(c):float(np.mean([r["raw_fixed_coverage"][str(c)]["mean"] for r in values])) for c in (100,95,90,80,70,60)} if arm!="point" else None}
    write_json(target/"three_seed_summary.json",{"records":records,"summary":summaries,"aggregation":"Arithmetic mean of three per-seed metrics, not an ensemble","script_sha256":sha256(Path(__file__))})
    lines=["# V6 completed: canonical-frame combination did not improve the source result","","All 12 primary arms and 24 best/final checkpoints completed. Independent CPU","weight replay and FP64 error rescoring passed for every checkpoint. Same 1126","real SimpleCube++ training / 119 reused validation images, CC BY 4.0; 120 epochs,","three seeds and matched common 20-epoch full states. All three CUDA warmup","replays were bitwise equal. No new test/camera/skin accuracy claim.","","Metrics below are three-seed arithmetic means, not an inference ensemble.","Best checkpoints were selected on these same validation images; risk is raw,","without a separate calibration. 80% coverage means 95/119 accepted per model.","","|Arm|Best full mean °|Final mean °|Raw risk80 °|","|---|---:|---:|---:|"]
    for arm,s in summaries.items():
        risk=f"{s['risk']['80']:.4f}" if s["risk"] else "Not trained"
        lines.append(f"|{arm}|{s['best_mean']:.4f}|{s['final_mean']:.4f}|{risk}|")
    lines += ["","|Coverage|Posterior|Generic action|Physical transport|","|---|---:|---:|---:|"]
    for c in (100,95,90,80,70,60):
        values="|".join(f"{summaries[a]['risk'][str(c)]:.4f}" for a in ("posterior_random","action_random","transport_random"))
        lines.append(f"|{c}%|{values}|")
    lines += ["","The best V6 combination remains worse on this reused source validation than","V5's generic action mean 2.4136° / raw risk80 2.1333°, and physical transport","mean 2.4548° / raw risk80 2.0192°. Those V5 figures are locally reproduced","historical controls, not published author numbers. Canonicalization does not","automatically combine the best properties of an anchor and a learned critic.","","All architectures contain 3,097,189 parameters; saved model state is 12,599,089","bytes. Peak allocated training memory is 859.98–864.07 MiB including the source","cache and common state. No V6 inference timing or export was measured.","","An inherited unused oracle helper clips absolute rather than residual actions.","Separate corrected receipts for all three seeds preserve original artifacts;","all 24 corrected oracle2 minimum arrays have zero change. Training, selection","and achieved error/risk never use the oracle. Corrected receipts, not the old","oracle field, govern any future diagnostic claim.","","Decision: retire V6 as the main accuracy candidate; preserve it as a negative","ablation. Continue the distinct semantic/sensor training screen and the compact","Fourier representation branch. Source-only gains must survive a newly locked","external-camera evaluation before changing product claims.","","[Per-seed metrics, curves and hashes](cc_v6/three_seed_summary.json).","Original per-image predictions and fixed-coverage/tail tables are archived","under cc_v6/seed17, seed29 and seed43; dataset images and weights are excluded."]
    (ROOT/"docs/benchmarks/cc_v6_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(summaries))


if __name__=="__main__":
    run()
