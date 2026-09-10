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
B = ROOT / "docs/benchmarks/phone_v1"
r = json.loads((B / "results.json").read_text())
rows = r["rows"]
with np.load(ROOT / "data/processed/phone_v1/references.npz") as f:
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
lines = ["# First locked Samsung/Oppo transfer benchmark", "", "**Measured custom public-data transfer screen. The newer V5 physical critic loses to the older V2 anchored model on phones.**", "", "All30 methods were fixed before decoding reserved test inputs. A checkpoint-container loader bug was corrected with a separately frozen receipt before any model prediction/error; original executable and lock remain preserved. All results below are REPRODUCED LOCALLY, not author-reported numbers.", "", "Dataset: Beyond RGB original CC BY4.0, Samsung Galaxy S21 Plus and Oppo Find X5 Pro,44 paired official field TEST scenes/88 NT images. Train source: real SimpleCube++ only. Reference policy retained72/88 inputs:37 Samsung and35 Oppo;16 unscorable references are NOT successful model refusals. Nine are strict HDF5-key rejections and seven fail gray-reference quality. Two of72 scorable rows also have unreadable NT keys and are mandatorily refused by every model: the full-population mean includes finite neutral fallback diagnostics, NOT accepted100% operation. Actual maximum coverage is70/72; separate alias-format investigation is required. No estimator saw phone GT, camera metadata, WT chart capture or spectrum. Single quantized demosaiced camera-RGB input; not ordinary JPEG/HEIC. No commercial skin accuracy claim.", "", "|Family (mean of3 seeds for CNNs; no ensemble)|Pooled meanР’В°|Pooled risk80Р’В°|Samsung risk80Р’В°|Oppo risk80Р’В°|", "|---|---:|---:|---:|---:|"]
for f in families:
    def fmt(x):
        return "untrained" if x is None else f"{x:.3f}"
    lines.append(f"|{f}|{fmt(summary['all'][f]['full'])}|{fmt(summary['all'][f]['risk80'])}|{fmt(summary['samsung'][f]['risk80'])}|{fmt(summary['oppo'][f]['risk80'])}|")
lines += ["", "Pooled V2 SoG risk80 improves12.8% versus matched direct C+, but Samsung regresses. The paired scene-bootstrap95% difference interval is[-1.533,+0.378] degrees versus C+ and[-1.103,+0.168] versus GW+ridge. Both include zero: no statistically conclusive dominance from this small, filtered screen. V5 source-development risk gains do not transfer to these phone captures.", "", "Nominal80% means57/72=79.17% of scorable inputs, but only57/88=64.77% of all planned inputs. Per-device accepted counts are29/37 Samsung and28/35 Oppo. Reported source thresholds below are separate: they were never fitted to phones.", "", "|V2 SoG fixed coverage|Mean errorР’В°|Accepted/scorable|Accepted/planned|", "|---|---:|---:|---:|"]
a = [v for v in records if v["family"] == "v2_sog" and v["domain"] == "all"]
for c in (100,95,90,80,70,60):
    v = [x["selective"]["fixed"][str(c)] for x in a]
    lines.append(f"|{c}%|{np.mean([x['mean'] for x in v]):.3f}|{v[0]['accepted']}/72|{v[0]['accepted']}/88|")
lines += ["", "Frozen source-calibrated nominal80% thresholds (actual phone coverage):", "", "|Method|Accepted/scorable|Actual coverage%|MeanР’В°|", "|---|---:|---:|---:|"]
for v in [x for x in records if x["domain"] == "all" and x["source_thresholds"]]:
    t = v["source_thresholds"]["80"]
    lines.append(f"|{v['id']}|{t['accepted']}/72|{100*t['coverage_scorable']:.1f}|{t['mean']:.3f}|")
lines += ["", "Reference exclusion reasons: `"+json.dumps(dict(excluded))+"`. Scenes grouped by number of scorable phone views: `"+json.dumps(dict(paired))+"`. Quality filtering is an engineering policy locked on TRAIN loader captures, not an author official evaluation protocol or an instrument uncertainty guarantee. Separate WT/NT capture illumination consistency remains an assumption.", "", f"Independent FP64 cosine/arccos reconstruction checked{checks} full/coverage means; largest discrepancy{max_delta:.3g} degrees. All90 per-device/method records, tails, recovery metrics, complete risk curves, frozen source thresholds and every seed are in results.json/all_methods.csv. V5 point-only has no trained selective head.", "", "No phone-specific latency measured. Reused V2 measurements:3.034M parameters,12.324MB checkpoint,RTX4060 model-only4.181ms at128px,training762MiB and inference28MiB PyTorch allocations; original timing scope preserved. V5:3.097M parameters,12.599MB checkpoint; V5 deployment latency remains NOT MEASURED.", "", "For Luma: real evidence for the photometric normalization/reliability component on two previously unseen phone models, with explicit failure cases. It does not validate facial skin color, DeltaE00, iPhone, arbitrary ISP pipelines, clinical/cosmetics outcomes or production-safe calibration. TheSkolkovo innovation wording remains a candidate positioning statement.", "", "Next: V6 factorial canonical-frame+physical-evidence experiment, declared before phone errors and trained on source development only. Preserve these phone outputs as observed evaluation, not a model-selection set. Reserve another disjoint acquisition for future confirmation. In parallel investigate a histogram/Fourier representation without a spatial CNN and decision sets instead of a single illuminant; neither is claimed novel merely by implementation.", "", "Reproduction: original method_lock.json plus loader_repair_v1_1.json bind artifacts. Run cc_phone_benchmark.py prepare using the lock on a fresh cache, then cc_phone_benchmark_v1_1.py predict/evaluate with --lock-sha256 f47778497a76782da288fe6740d8ef12388fa080ead9a08db618b7a6de393021. Outputs refuse overwrite. cc_phone_report.py independently verifies means and creates this report. Data acquisition and reference protocol remain separately documented."]
(ROOT/"docs/benchmarks/phone_v1_report.md").write_text(chr(10).join(lines)+chr(10),encoding="utf-8")
(B/"independent_verification.json").write_text(json.dumps({"checks":checks,"max_abs_difference_degrees":max_delta,"scorable":int(good.sum()),"planned":len(rows),"report_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2),encoding="utf-8")
print(json.dumps({"checked_means":checks,"max_difference":max_delta,"scorable":int(good.sum())}))
