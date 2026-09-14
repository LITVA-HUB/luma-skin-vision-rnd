"""Lightweight provisional review of completed inner folds; never opens final outputs."""

from __future__ import annotations

import argparse

import numpy as np
from chromaseed_architecture_scale_run import (
    OUT,
    ROOT,
    RUN,
    TIMES,
    VARIANTS,
    WE,
    bank_path,
    load_data,
)
from chromaseed_kernel_audit import js, nz
from chromaseed_refine_audit import error_summary
from skin_local_search_train import roles, sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("mixed", "slr_to_ipod", "ipod_to_slr"), required=True)
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", required=True)
    args = parser.parse_args()
    source = sha(RUN / "source_lock.json")
    data = load_data()
    mask, _ = roles(data["patient"], data["device"])[args.role]
    fit = np.flatnonzero(mask)
    inputs, records = {}, []
    for variant in args.variants:
        by_step = {step: [] for step in TIMES}
        rows = []
        for fold in range(3):
            path = bank_path(args.role, variant, fold)
            receipt = js(path / "receipt.json")
            assert receipt["source_lock_sha256"] == source and receipt["selection_sha256"] is None
            inputs[(path / "receipt.json").relative_to(ROOT).as_posix()] = sha(
                path / "receipt.json"
            )
            for step in TIMES:
                p = path / f"oof_{step}.npz"
                assert sha(p) == receipt["files"][p.name]
                inputs[p.relative_to(ROOT).as_posix()] = sha(p)
                saved = nz(p)
                if step == TIMES[0]:
                    rows.append(saved["row_indices"])
                else:
                    np.testing.assert_array_equal(rows[-1], saved["row_indices"])
                by_step[step].append(saved["predictions"])
        rows = np.concatenate(rows)
        order = np.argsort(rows)
        np.testing.assert_array_equal(rows[order], fit)
        for ri, rate in enumerate((0.00001, 0.0001)):
            for step in TIMES:
                prediction = np.concatenate(by_step[step], axis=1)[:, order]
                errors = [
                    error_summary(
                        prediction[2 * si + ri],
                        data["target"][fit],
                        data["patient"][fit],
                        data["site"][fit],
                    )[0]["person_mean"]
                    for si in range(3)
                ]
                records.append(
                    dict(
                        variant=variant,
                        rate=rate,
                        step=step,
                        person_delta_e00=float(np.mean(errors)),
                        individual_seed_delta_e00=errors,
                    )
                )
    prior = js(WE / "selections.json")["roles"][args.role]
    baseline = next(c for c in prior["candidates"] if c["variant"] == "np")
    value = dict(
        classification="provisional inner-only curve review; no independent model audit or held-role evaluation",
        source_lock_sha256=source,
        role=args.role,
        variants=args.variants,
        records=records,
        rows=len(fit),
        people=len(np.unique(data["patient"][fit])),
        prior_np_inner_delta_e00=baseline["clean"],
        prior_we_policy=prior["policies"]["overall"],
        completed_inner_inputs=inputs,
        analysis_source_sha256=sha(ROOT / "scripts/chromaseed_architecture_scale_inner_review.py"),
    )
    path = OUT / "inner_reviews" / (args.role + "_" + "_".join(args.variants) + ".json")
    if path.exists():
        assert js(path) == value
    else:
        write_json(path, value)
    print("AS INNER-ONLY", args.role, len(fit), "rows", value["people"], "people")
    print("prior NP", baseline["clean"], "prior WE policy", prior["policies"]["overall"]["clean"])
    for variant in args.variants:
        for rate in (0.00001, 0.0001):
            rr = [r for r in records if r["variant"] == variant and r["rate"] == rate]
            print(variant, rate, [(r["step"], round(r["person_delta_e00"], 6)) for r in rr])
    print("PROVISIONAL SNAPSHOT", path)


if __name__ == "__main__":
    main()
