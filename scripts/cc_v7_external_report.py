"""Report all frozen methods with paired camera-stratified proxy-cluster intervals."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cc_v7_external_lock import BENCH, ROOT, checked_lock, load
from cc_v7_verify import score

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

PAIRS=[("v7_canonical_teacher_sensor","v7_raw_teacher_sensor"),("v7_gt_sensor","v7_gt_native"),("v7_canonical_teacher_sensor","v2_sog"),("v7_gt_sensor","v2_sog"),("v7_canonical_teacher_sensor","gw_ridge1"),("v7_canonical_teacher_sensor","fourier_ridge_combined")]


def paired_draws(rows,repeats=2000,seed=20260911):
    group_cameras={}
    for row in rows:
        group_cameras.setdefault(row["group"],set()).add(row["camera"])
    if any(len(cameras)>1 for cameras in group_cameras.values()):
        raise ValueError("A reference group spans cameras; stratification would split it")
    strata=[]
    for camera in sorted({r["camera"] for r in rows}):
        groups=sorted({r["group"] for r in rows if r["camera"]==camera})
        members=[np.array([i for i,r in enumerate(rows) if r["camera"]==camera and r["group"]==group]) for group in groups]
        strata.append(members)
    rng=np.random.default_rng(seed)
    return [np.concatenate([members[g] for members in strata for g in rng.integers(0,len(members),len(members))]) for _ in range(repeats)]


def family_measure(family,draw,tie_keys):
    full=float(family["error"][:,draw].mean())
    values=[]
    for e,s,v in zip(family["error"][:,draw],family["score"][:,draw],family["valid"][:,draw]):
        count=max(1,len(draw)*80//100)
        eligible=np.flatnonzero(v)
        if len(eligible)<count:
            return full,None
        order=eligible[np.lexsort((tie_keys[draw][eligible],s[eligible]))]
        values.append(float(e[order[:count]].mean()))
    return full,float(np.mean(values))


def bootstrap(lock):
    population=load(ROOT/lock["population"])
    all_rows=population["rows"]
    chosen=np.array([i for i,r in enumerate(all_rows) if r["primary"]])
    rows=[all_rows[i] for i in chosen]
    with np.load(BENCH/"evaluation/external_references.npz") as references:
        if references["ids"].tolist()!=[r["id"] for r in all_rows]:
            raise ValueError("Reference identity mismatch")
        gt=references["gt"][chosen]
    families={}
    needed={f for pair in PAIRS for f in pair}
    for method in lock["methods"]:
        if method["family"] not in needed:
            continue
        with np.load(BENCH/"predictions"/("external__"+method["id"]+".npz")) as pred:
            family=families.setdefault(method["family"],{"error":[],"score":[],"valid":[]})
            family["error"].append(score(pred["pred"][chosen],gt))
            family["score"].append(pred["scores"][chosen])
            family["valid"].append(pred["valid"][chosen])
    families={f:{k:np.stack(v) for k,v in values.items()} for f,values in families.items()}
    ties=np.array([hashlib.sha256(r["id"].encode()).hexdigest() for r in rows])
    samples=paired_draws(rows)
    estimates={f:family_measure(values,np.arange(len(rows)),ties) for f,values in families.items()}
    series={f:[] for f in families}
    for draw in samples:
        for family,values in families.items():
            series[family].append(family_measure(values,draw,ties))
    results=[]
    for a,b in PAIRS:
        record={"a":a,"b":b,"difference_sign":"negative favors a"}
        for column,name in ((0,"full"),(1,"risk80")):
            delta=[x[column]-y[column] for x,y in zip(series[a],series[b]) if x[column] is not None and y[column] is not None]
            value=estimates[a][column]-estimates[b][column] if estimates[a][column] is not None and estimates[b][column] is not None else None
            record[name]={"difference":value,"replicates_supported":len(delta),"percentile95":np.percentile(delta,[2.5,97.5]).tolist() if len(delta)==2000 else None}
        results.append(record)
    return {"population":"unseen_primary317","bootstrap_draws":2000,"seed":20260911,"unit":"Exact reference-file hash, stratified by camera; not verified scene identity","limitations":"Unknown scene dependence, multiple comparisons and pretrained-image overlap are not corrected","comparisons":results}


def mean_or_none(values):
    return float(np.mean(values)) if all(v is not None for v in values) else None


def aggregate(records):
    result={}
    for family in sorted({r["family"] for r in records}):
        values=[r["metrics"] for r in records if r["family"]==family]
        result[family]={"models":len(values),"full":float(np.mean([v["reproduction"]["mean"] for v in values])),"recovery":float(np.mean([v["recovery"]["mean"] for v in values])),"p95":float(np.mean([v["reproduction"]["p95"] for v in values])),"over10_fraction":float(np.mean([v["reproduction"]["over10_fraction"] for v in values])),"unsupported_max":max(v["invalid_n"] for v in values),"fixed":{str(c):mean_or_none([v["selective"]["fixed"][str(c)]["mean"] for v in values]) for c in (100,95,90,80,70,60)},"source_threshold80_coverage":float(np.mean([v["frozen_source_thresholds"]["80"]["coverage"] for v in values])),"source_threshold80_error":mean_or_none([v["frozen_source_thresholds"]["80"]["mean_reproduction"] for v in values])}
    return result


def run(digest):
    lock=checked_lock(digest)
    out=BENCH/"report"
    if out.exists():
        raise ValueError("Immutable external report exists")
    evaluation=load(BENCH/"evaluation/results.json")
    if evaluation["method_lock_sha256"]!=digest or evaluation["prediction_manifest_sha256"]!=sha256(BENCH/"predictions/manifest.json"):
        raise ValueError("Evaluation source binding changed")
    records=evaluation["records"]
    summary={p:aggregate([r for r in records if r["population"]==p]) for p in sorted({r["population"] for r in records})}
    intervals=bootstrap(lock)
    out.mkdir(parents=True)
    write_json(out/"summary.json",{"aggregation":"Arithmetic mean of per-seed metrics; not inference ensembles; single-model controls retain their own values","populations":summary})
    write_json(out/"paired_intervals.json",intervals)
    fig,ax=plt.subplots(figsize=(9,5))
    plotted=("v7_gt_native","v7_gt_sensor","v7_raw_teacher_sensor","v7_canonical_teacher_sensor","v2_sog","gw_ridge1","fourier_ridge_combined")
    grid=np.arange(1,318)/317
    for family in plotted:
        curves=[]
        for row in records:
            if row["population"]=="unseen_primary" and row["family"]==family:
                curve=row["metrics"]["selective"]
                curves.append(np.interp(grid,curve["coverage"],curve["risk"],left=np.nan,right=np.nan) if len(curve["coverage"]) else np.full_like(grid,np.nan))
        ax.plot(grid*100,np.mean(curves,axis=0),label=family)
    ax.set(xlabel="Accepted coverage (%)",ylabel="Mean reproduction error (degrees)",title="V7: real unseen-camera primary317; mean per-seed risk curves",xlim=(0,100))
    ax.grid(alpha=.2)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out/"risk_coverage.png",dpi=160)
    plt.close(fig)
    primary=summary["unseen_primary"]
    known=summary["known_camera_official_test"]
    def fmt(x):
        return "unsupported" if x is None else f"{x:.4f}"
    lines=["# V7 real benchmark: all frozen methods and camera-transfer results","","All29 methods were locked before new INTEL-TAU pixel/GT decoding. Results are","locally reproduced, not author-published numbers. V7 has five arms by three","seeds; its semantic teacher is training-only. All data/weight/code licensing","and provenance remain separate. No skin-color accuracy or universal camera","independence is inferred from this color-constancy benchmark.","","Primary external317 images:103 Canon5DSR,112 NikonD810,102 SonyIMX135, all camera","models absent from source estimator/risk training. Full384 sensitivity includes","67 rows sharing exact reference-file hashes with historical V2. Reference hashes","are only proxy groups, not proven physical scene identities. Original INTEL-TAU","CC BY-SA4.0 is used for evaluation only; sparse-mirror byte provenance remains","limited. Training uses CC BY4.0 SimpleCube++; standard DINOv2 teacher Apache2.0.","","Known-camera comparison: official SimpleCube TEST462. These image IDs are absent","from fitting but capture dates overlap, and earlier methods already evaluated","this split. It is not a new independent capture-group test. Source camera models","are Canon550D/600D, both present in training. Neither target camera IDs nor CCMs","are fed into prediction, and no target-camera adaptation or calibration occurs.","","Numbers average three per-seed metrics where available, not three-model","ensembles. Lower is better. 80% primary coverage accepts253/317 when supported.","Full means include explicit neutral fallback diagnostics for unsupported rows;","unsupported counts and attainable coverage are preserved per method.","","|Family|Models|Known full °|Unseen full °|Unseen risk80 °|Unseen p95 °|Source80 threshold target coverage|","|---|---:|---:|---:|---:|---:|---:|"]
    for family,values in primary.items():
        lines.append(f"|{family}|{values['models']}|{known[family]['full']:.4f}|{values['full']:.4f}|{fmt(values['fixed']['80'])}|{values['p95']:.4f}|{100*values['source_threshold80_coverage']:.2f}%|")
    lines += ["","![Risk-coverage curve](report/risk_coverage.png)","","|Family|100%|95%|90%|80%|70%|60%|","|---|---:|---:|---:|---:|---:|---:|"]
    for family,values in primary.items():
        lines.append("|"+family+"|"+"|".join(fmt(values["fixed"][str(c)]) for c in (100,95,90,80,70,60))+"|")
    lines += ["","|Unseen primary camera|Family|Full °|Risk80 °|","|---|---|---:|---:|"]
    for population,families in summary.items():
        if population.startswith("unseen_primary/"):
            for family,values in families.items():
                lines.append(f"|{population.split('/')[-1]}|{family}|{values['full']:.4f}|{fmt(values['fixed']['80'])}|")
    lines += ["","|Prespecified contrast A minus B|Full difference and95% proxy-cluster CI|Risk80 difference and95% CI|","|---|---|---|"]
    for comparison in intervals["comparisons"]:
        def display(name):
            value=comparison[name]
            return fmt(value["difference"])+" / "+str(value["percentile95"])
        lines.append(f"|{comparison['a']} minus {comparison['b']}|{display('full')}|{display('risk80')}|")
    candidate=primary["v7_canonical_teacher_sensor"]
    baseline=primary["v7_raw_teacher_sensor"]
    full_delta=candidate["full"]-baseline["full"]
    risk_delta=candidate["fixed"]["80"]-baseline["fixed"]["80"] if candidate["fixed"]["80"] is not None and baseline["fixed"]["80"] is not None else None
    lines += ["",f"Primary canonical-teacher contrast: full {full_delta:+.4f}°; risk80 {fmt(risk_delta)}°","(negative favors candidate). Inspect both the intervals and individual-camera","results; a pooled improvement alone does not establish consistent superiority.","All29 methods, including failures, remain in the results. The strongest observed","control is reported in the complete table; no result is relabeled as reproduced","from an author's publication. Multiple-comparison and scene-dependence limits","remain; teacher pretraining overlap is unknown.","","Model/compute: each V7 student has3,033,651 deployment parameters and369,024","training-only projection parameters. Saved training checkpoint13,816,203 bytes","includes the projection. Peak allocated training memory is about933 MiB with","source/teacher caches and initial state included. V7 inference latency, inference","VRAM, deployed file size and ONNX/TensorRT performance are NOT MEASURED in this","report. No earlier synthetic or V2 timing is substituted for V7 measurement.","","Luma implication: this is component-level evidence about photometric","normalization and reliability on real public illuminant references. Genuine","surface CIEDE2000, facial skin colorimetry and ordinary phone JPEG/HEIC robustness","remain unvalidated. The candidate Skolkovo positioning is technical planning,","not an approved legal classification or established patent novelty.","","[All population summaries](report/summary.json), [paired intervals](report/paired_intervals.json),","[full per-method metrics and curves](evaluation/results.json). Per-image predictions","and separate reference arrays are retained without dataset image publication."]
    (BENCH/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(out/"manifest.json",{"method_lock_sha256":digest,"evaluation_sha256":sha256(BENCH/"evaluation/results.json"),"script_sha256":sha256(Path(__file__)),"report_sha256":sha256(BENCH/"report.md"),"sha256":{p.name:sha256(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({"primary_candidate_full_difference":full_delta,"risk80_difference":risk_delta,"report":str(BENCH/"report.md")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--digest",required=True)
    run(parser.parse_args().digest)
