"""Freeze all29 estimators/selectors before real V7 target image/GT decoding."""
import argparse
import copy
import json
from pathlib import Path

from cc_v7_verify import verify_manifest

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT/"docs/benchmarks/cc_v7_external"
LOCK=BENCH/"method_lock.json"
OLD_LOCK_SHA="f47778497a76782da288fe6740d8ef12388fa080ead9a08db618b7a6de393021"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze():
    if LOCK.exists():
        raise ValueError("Existing immutable real-camera method lock")
    frozen={}
    methods=[]
    def bind(path):
        path=Path(path).resolve()
        relative=path.relative_to(ROOT).as_posix()
        frozen[relative]=sha256(path)
        return relative
    population=ROOT/"docs/data/provenance/cc_v7/external_population.json"
    target=load(population)
    if target["target_errors_observed"] is not False or target["primary_n"]!=317 or target["sensitivity_n"]!=384:
        raise ValueError("Wrong external population status")
    bind(population)
    for seed in (17,29,43):
        run=ROOT/f"experiments/runs/ccv7_semantic_s{seed}"
        risk=ROOT/f"experiments/runs/ccv7_semantic_risk_s{seed}"
        verify_manifest(run,"artifact_manifest.json")
        verify_manifest(risk,"manifest.json")
        config=load(run/"config.json")
        if not config["primary"] or len(config["train_ids"])!=1126:
            raise ValueError("Nonprimary V7 estimator")
        receipt=ROOT/f"docs/benchmarks/cc_v7/seed{seed}_verification.json"
        if len(load(receipt)["records"])!=10 or not load(receipt)["primary"]:
            raise ValueError("V7 checkpoint verification missing")
        bind(receipt)
        for arm in config["arms"]:
            selection=risk/arm/"selection.json"
            info=load(selection)
            weight=run/arm/"best.pt"
            head=risk/arm/"combined.joblib"
            if info["checkpoint_sha256"]!=sha256(weight) or info["heads"]["combined"]["artifact_sha256"]!=sha256(head):
                raise ValueError("V7 weight/head identity mismatch")
            methods.append({"id":f"v7_{arm}_s{seed}","family":"v7_"+arm,"kind":"v7","seed":seed,"mode":"direct","backbone":"large","weight":bind(weight),"head":bind(head),"selection":bind(selection),"config":bind(run/"config.json")})
        bind(risk/"config.json")
    old_path=ROOT/"docs/benchmarks/phone_v1/method_lock.json"
    if sha256(old_path)!=OLD_LOCK_SHA:
        raise ValueError("Historical method lock changed")
    old=load(old_path)
    bind(old_path)
    for item in old["methods"]:
        if item["kind"] not in ("v2","statistics","classical"):
            continue
        method=copy.deepcopy(item)
        for key in ("weight","head","selection","config"):
            if key in method:
                path=ROOT/method[key]
                if sha256(path)!=old["sha256"][method[key]]:
                    raise ValueError("Historical comparator changed")
                bind(path)
        methods.append(method)
    controls=ROOT/"experiments/runs/ccv7_external_controls"
    verify_manifest(controls,"manifest.json")
    control_config=load(controls/"config.json")
    weight=ROOT/control_config["fourier_weight"]
    if sha256(weight)!=control_config["fourier_weight_sha256"]:
        raise ValueError("Fourier control changed")
    for selector in ("raw","combined"):
        method={"id":"fourier_ridge_"+selector,"family":"fourier_ridge_"+selector,"kind":"fourier","selector":selector,"weight":bind(weight)}
        if selector=="combined":
            method["head"]=bind(controls/"fourier_ridge_combined/combined.joblib")
            method["selection"]=bind(controls/"fourier_ridge_combined/selection.json")
        methods.append(method)
    raw_cal=bind(controls/"raw_calibration.json")
    bind(controls/"config.json")
    for path in sorted((ROOT/"src").rglob("*.py")):
        bind(path)
    for name in ("cc_v7_external_lock.py","cc_v7_external_data.py","cc_v7_external_benchmark.py","cc_v7_external_controls.py","cc_v7_external_report.py","cc_v7_risk.py","cc_v2_select.py","cc_v2_statistics.py","cc_v3_ffcc.py","cc_fourier_ridge.py"):
        bind(ROOT/"scripts"/name)
    for name in ("cc_v7_external_methods_protocol.md","cc_v7_transfer_diagnostics_protocol.md"):
        bind(ROOT/"docs/research"/name)
    if len(methods)!=29 or len({m["id"] for m in methods})!=29:
        raise ValueError("Expected29 frozen distinct methods")
    BENCH.mkdir(parents=True,exist_ok=True)
    write_json(LOCK,{"status":"ALL29 METHODS FROZEN BEFORE NEW TARGET DECODE","target_errors_observed":False,"methods":methods,"sha256":frozen,"population":population.relative_to(ROOT).as_posix(),"raw_calibration":raw_cal,"primary_contrast":"v7_canonical_teacher_sensor versus v7_raw_teacher_sensor, plus strongest locally reproduced comparator; all seeds retained","target_data_use":"INTEL-TAU original CC BY-SA4.0 evaluation-only; source training cameras550D/600D differ from external5DSR/D810/IMX135","known_camera_limit":"Official SimpleCube TEST462 overlaps capture dates and was observed by earlier methods; not a new group-independent test"})
    print(json.dumps({"methods":len(methods),"lock_sha256":sha256(LOCK)}))


def checked_lock(digest):
    if sha256(LOCK)!=digest:
        raise ValueError("Explicit method lock digest differs")
    lock=load(LOCK)
    for name,expected in lock["sha256"].items():
        if sha256(ROOT/name)!=expected:
            raise ValueError("Frozen artifact changed: "+name)
    return lock


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("freeze","check"))
    parser.add_argument("--digest")
    args=parser.parse_args()
    if args.action=="freeze":
        freeze()
    else:
        checked_lock(args.digest)
        print("METHOD LOCK VERIFIED")
