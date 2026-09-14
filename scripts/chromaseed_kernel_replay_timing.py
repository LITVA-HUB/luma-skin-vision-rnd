"""One complete process-level replay, distinct from amortized fit-bank timings."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from skin_local_search_train import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def js(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--replay-run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--legacy-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.replay_run.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(ROOT / "scripts/chromaseed_kernel_train.py"), "run",
               "--run", str(args.replay_run.resolve()), "--cache", str(args.cache.resolve()),
               "--legacy-run", str(args.legacy_run.resolve())]
    started = time.perf_counter()
    with (args.replay_run / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    seconds = time.perf_counter() - started
    if sha(args.original_run / "source_lock.json") != sha(args.replay_run / "source_lock.json"):
        raise ValueError("replay source lock differs")
    if sha(args.original_run / "selections.json") != sha(args.replay_run / "selections.json"):
        raise ValueError("replay selections differ")
    compared, entries, maximum = 0, 0, 0.
    for part in ("selected", "evaluated"):
        originals = sorted((args.original_run / part).glob("*/*.npz"))
        replays = sorted((args.replay_run / part).glob("*/*.npz"))
        if [p.relative_to(args.original_run) for p in originals] != [p.relative_to(args.replay_run) for p in replays]:
            raise ValueError("replay file inventory differs")
        for path in originals:
            with np.load(path, allow_pickle=False) as a, np.load(args.replay_run / path.relative_to(args.original_run), allow_pickle=False) as b:
                if a.files != b.files:
                    raise ValueError("array schema changed")
                for name in a.files:
                    if np.issubdtype(a[name].dtype, np.floating):
                        drift = float(np.max(np.abs(a[name] - b[name])))
                        maximum = max(maximum, drift)
                        np.testing.assert_allclose(a[name], b[name], rtol=0, atol=1e-6)
                    else:
                        np.testing.assert_array_equal(a[name], b[name])
                    entries += 1
            compared += 1
    receipt = {"wall_seconds": seconds, "process_exit_code": process.returncode, "repetitions": 1,
               "replay_source_sha256": sha(Path(__file__)), "source_lock_sha256": sha(args.original_run / "source_lock.json"),
               "selection_sha256": sha(args.original_run / "selections.json"), "compared_selected_and_evaluated_files": compared,
               "compared_array_entries": entries, "maximum_replay_array_drift": maximum,
               "readout_configurations_replayed": 4428,
               "scope": "Full train.py run subprocess: interpreter/import/CUDA startup, source/data/legacy hashing, cache load, all 9 inner banks, choices, 3 final banks, selected model export, outer evaluation and persistence. Filesystem cache and driver may be warm. Excludes this wrapper's post-run comparisons, independent audit, plots and runtime benchmarks. This is a repeated timing verification of the same experiment, not 4428 new hypotheses or fresh evidence.",
               "replay_run": str(args.replay_run.resolve())}
    write_json(args.output / "complete_workflow_runtime.json", receipt)
    previous = js(args.replay_run / "progress.json")
    write_json(args.replay_run / "progress.json", {**previous, "status": "replay_compared_complete", "process_exit_code": 0,
                                                 "updated_unix": time.time()})
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
