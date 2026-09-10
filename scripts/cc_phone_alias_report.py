"""Independent FP64 phone rescoring and transparent report generation."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "docs/benchmarks/phone_v1_alias"
r = json.loads((B / "results.json").read_text())
rows = r["rows"]
with np.load(ROOT / "data/processed/phone_v1_alias/references.npz") as f:
    gt, good = f["gt"], f["valid"]
records = r["records"]
max_delta = 0.
checks = 0
for name in dict.fromkeys(a["id"] for a in records):
    with np.load(B / "predictions_v1_1" / (name + ".npz")) as f:
        data = {k: f[k] for k in f.files}
    error = np.full(len(rows), np.nan)
    ratio = gt[good].astype(np.float64) / data["pred"][good].astype(np.float64)
    cosine = ratio.sum(-1) / (np.sqrt(3) * np.linalg.norm(ratio, axis=-1))
    error[good] = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    for record in [a for a in records if a["id"] == name]:
        ix = [i for i,row in enumerate(rows) if good[i] and (record["domain"] == "all" or row["camera"] == record["domain"])]
        pairs = [(float(error[ix].mean()),record["reproduction"]["mean"])]
        if record["selective"]:
            order = sorted([i for i in ix if data["valid"][i]], key=lambda i: (data["risk"][i], hashlib.sha256(rows[i]["id"].encode()).hexdigest()))
            for c in (100,95,90,80,70,60):
                k = min(len(order),max(1,int(len(ix)*c/100)))
                saved = record["selective"]["fixed"][str(c)]
                assert saved["accepted"] == k
                assert abs(saved["coverage_planned"]-k/record["planned"]) < 1e-12
                pairs.append((float(error[order[:k]].mean()),saved["mean"]))
        for actual, expected in pairs:
            max_delta = max(max_delta,abs(actual-expected))
            assert abs(actual-expected) < 1e-7
            checks += 1
families = list(dict.fromkeys(a["family"] for a in records))
def aggregate(family, domain="all"):
    a = [v for v in records if v["family"] == family and v["domain"] == domain]
    return {"full":float(np.mean([v["reproduction"]["mean"] for v in a])),
            "risk80":float(np.mean([v["selective"]["fixed"]["80"]["mean"] for v in a])) if a[0]["selective"] else None,
            "n":a[0]["scorable"]}
summary = {d:{f:aggregate(f,d) for f in families} for d in ("all","samsung","oppo")}
(B/"family_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
with (B/"all_methods.csv").open("w",newline="",encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["method","family","domain","scorable","planned","refused","mean_reproduction","median_reproduction","risk80"])
    for v in records:
        writer.writerow([v["id"],v["family"],v["domain"],v["scorable"],v["planned"],v["model_refused"],v["reproduction"]["mean"],v["reproduction"]["median"],v["selective"]["fixed"]["80"]["mean"] if v["selective"] else "NOT MEASURED"])
fig, axes = plt.subplots(1,3,figsize=(14,4.3),sharey=True)
for ax,domain in zip(axes,("all","samsung","oppo")):
    for family,label in [("v2_sog","V2 SoG"),("v2_direct","V2 direct C+"),("gw_ridge1","GW + ridge"),("v5_transport_random","V5 physical critic"),("v5_action_random","V5 action critic")]:
        a = [v for v in records if v["family"] == family and v["domain"] == domain]
        curves = np.array([v["selective"]["curve"] for v in a])
        cov = a[0]["selective"]["coverage_scorable"]
        ax.plot(np.array(cov)*100,curves.mean(0),label=label)
    ax.set(xlim=(20,100),ylim=(0,8),xlabel="Accepted % of scorable inputs",title=domain)
    ax.grid(alpha=.2)
axes[0].set_ylabel("Mean reproduction error (degrees)")
axes[-1].legend(fontsize=8)
fig.suptitle("Frozen source-only transfer to phones; family means, no ensemble")
fig.tight_layout()
fig.savefig(B/"risk_coverage.png",dpi=170)
fig.savefig(B/"risk_coverage.svg")
plt.close(fig)
excluded = Counter(row["reference"].get("reason") for row in rows if not row["reference"]["valid"])
paired = Counter(sum(row["reference"]["valid"] for row in rows if row["scene"] == scene) for scene in {row["scene"] for row in rows})

# Verify alias repair did not alter any previously readable input or reference.
old_rows = json.loads((ROOT/"data/processed/phone_v1/reference_manifest.json").read_text())
with np.load(ROOT/"data/processed/phone_v1/inputs.npz") as a, np.load(ROOT/"data/processed/phone_v1_alias/inputs.npz") as b:
    old_valid = a["valid"]
    for key in ("images","experts"):
        np.testing.assert_array_equal(a[key][old_valid],b[key][old_valid])
with np.load(ROOT/"data/processed/phone_v1/references.npz") as old:
    np.testing.assert_array_equal(old["gt"][old["valid"]],gt[old["valid"]])
valid_references = int(good.sum())
lines = ["# Samsung/Oppo benchmark after documented HDF5 name repair", "", "**All30 frozen source-only methods rerun;79/88 references scorable, all88 NT inputs readable.**", "", "This is an additive file-format repair after v1 results were observed, not a new untouched test. Original v1 locks, predictions and results remain preserved. Original author HDF5 reader takes the first key; the independent repair permits sole camera/MIS names only in verified camera RGB files. All other processing, reference requirements, models, calibration and grouping are unchanged. Previously readable inputs and valid references are byte-identical numerically.", "", "Dataset: Beyond RGB, original CC BY4.0; single quantized demosaiced camera-RGB NT image.37 Oppo and42 Samsung references meet the unchanged gray-patch policy;9 do not. No camera identity, WT capture, spectrum or CCM is a model input. Training was real SimpleCube++ only. No ordinary JPEG/HEIC/iPhone/skin/DeltaE validation. All numbers are REPRODUCED LOCALLY under our custom protocol, not published author numbers.", "", "|Family (means across3 seeds; no ensemble)|Pooled meanВ°|Pooled risk80В°|Samsung risk80В°|Oppo risk80В°|", "|---|---:|---:|---:|---:|"]
for family in families:
    def fmt(x):
        return "untrained" if x is None else f"{x:.3f}"
    lines.append(f"|{family}|{fmt(summary['all'][family]['full'])}|{fmt(summary['all'][family]['risk80'])}|{fmt(summary['samsung'][family]['risk80'])}|{fmt(summary['oppo'][family]['risk80'])}|")
lines += ["", "Paired scene-bootstrap differences (SoG minus comparator; negative favors SoG):", "", "|Comparator|Full mean95% intervalВ°|Risk80 95% intervalВ°|", "|---|---|---|"]
for name,v in r["paired_scene_bootstrap"].items():
    lines.append(f"|{name}|{v['full_mean_ci95']}|{v['risk80_ci95']}|")
lines += ["", "|V2 SoG nominal coverage|Mean accepted errorВ°|Accepted/scorable|Accepted/planned|", "|---|---:|---:|---:|"]
a=[v for v in records if v["family"]=="v2_sog" and v["domain"]=="all"]
for c in (100,95,90,80,70,60):
    v=[x["selective"]["fixed"][str(c)] for x in a]
    lines.append(f"|{c}%|{np.mean([x['mean'] for x in v]):.3f}|{v[0]['accepted']}/{valid_references}|{v[0]['accepted']}/88|")
lines += ["", "Frozen source-calibrated nominal80% thresholds:", "", "|Method|Accepted/scorable|Actual coverage%|MeanВ°|", "|---|---:|---:|---:|"]
for v in [x for x in records if x["domain"]=="all" and x["source_thresholds"]]:
    t=v["source_thresholds"]["80"]
    lines.append(f"|{v['id']}|{t['accepted']}/{valid_references}|{100*t['coverage_scorable']:.1f}|{t['mean']:.3f}|")
lines += ["", "Reference exclusion reasons: `"+json.dumps(dict(excluded))+"`. Scene counts by scorable phone views: `"+json.dumps(dict(paired))+"`. All model_refused counts are0 on scorable inputs. Quality filtering does not certify constant light across separate NT/WT captures, and excludes9/88 planned measurements.", "", f"Independent FP64 cosine/arccos rescoring checks{checks} means, max difference{max_delta:.3g} degrees. Complete all-seed, per-camera recovery/reproduction metrics, tails, curves and source thresholds are in results.json and all_methods.csv. Source80% calibration is not target80% acceptance.", "", "No new latency/VRAM measurement. Original V2:3.034M parameters,12.324MB checkpoint,4.181ms RTX4060 model-only at128px;762MiB train/28MiB inference PyTorch allocations under original scope. V5:3.097M/12.599MB, deployment latency unmeasured. This validates a limited normalization/reliability component, not measured facial skin color or universal camera support.", "", "Scientific decision: retain stronger V2 transfer control, preserve V5 failures, and test V6 combination only on source development. Fourier ridge provides an independent representation-control hypothesis. These observed phone captures must not become a validation set for new methods; confirm improvements on a separate locked sample. Novelty and Skolkovo classification remain unverified."]
(ROOT/"docs/benchmarks/phone_v1_alias_report.md").write_text(chr(10).join(lines)+chr(10),encoding="utf-8")
(B/"independent_verification.json").write_text(json.dumps({"checks":checks,"max_abs_difference_degrees":max_delta,"original_valid_inputs_and_references_unchanged":True,"scorable":valid_references,"planned":len(rows),"report_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2),encoding="utf-8")
print(json.dumps({"checked_means":checks,"max_difference":max_delta,"scorable":valid_references,"summary":summary["all"]}))
