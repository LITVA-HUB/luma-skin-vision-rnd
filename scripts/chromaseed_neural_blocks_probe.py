"""Causal isolation of first-step shape effects, including identical-gradient injection."""

from __future__ import annotations

import numpy as np
import torch
from chromaseed_kernel_audit import js
from chromaseed_local_denoise import Bank, preprocessor
from chromaseed_local_denoise_fit import noise_sequences
from chromaseed_neural_blocks_fit import RetainedBank
from chromaseed_neural_blocks_run import OUT, ROOT, RUN, SEEDS, check_map, load_data, settings
from chromaseed_refine import BankAdamW
from chromaseed_refine_train import sampling_indices, setup
from skin_local_search_train import roles, sha, weights_for, write_json


def difference(a, b):
    a, b = a.detach().cpu().numpy(), b.detach().cpu().numpy()
    return dict(bitwise_equal=bool(np.array_equal(a, b)), maximum=float(np.max(abs(a - b))))


def main():
    assert not (OUT / "verification.json").exists()
    setup("cuda")
    lock = js(RUN / "source_lock.json")
    check_map({**lock["sources"], **lock["input_sha256"]})
    data, records = load_data(), []
    for setting in settings():
        role, family = setting["role"], setting["family"]
        mask, _ = roles(data["patient"], data["device"])[role]
        x, y = data["color"][mask], data["target"][mask]
        prep = preprocessor(x, y)
        xn = (x - prep["x_mean"]) / prep["x_std"]
        yn = ((y - prep["y_mean"]) / prep["y_std"]).astype(np.float32)
        w = weights_for(data["patient"][mask], data["site"][mask])
        index = sampling_indices(w, SEEDS, 1)[:, 0]
        xb, yb = [torch.as_tensor(v[index], device="cuda") for v in (xn, yn)]
        for j in setting["prefixes"]:
            full, subset, injected = (
                Bank(family, SEEDS).cuda(),
                RetainedBank(family, SEEDS, j).cuda(),
                RetainedBank(family, SEEDS, j).cuda(),
            )
            ids = torch.as_tensor(subset.block_indices.astype(np.int64), device="cuda")
            eps = torch.as_tensor(noise_sequences(SEEDS, 1, full.k)[:, 0], device="cuda")
            a = torch.as_tensor(full.a[:-1], dtype=torch.float32, device="cuda")[
                None, :, None, None
            ]
            s = torch.as_tensor(full.s[:-1], dtype=torch.float32, device="cuda")[
                None, :, None, None
            ]
            state = a * yb[:, None] + s * eps if family != "blind4" else torch.zeros_like(eps)
            of = full.local(xb, state)
            os = subset.local(xb, state.index_select(1, ids))
            (of - yb[:, None]).square().mean((2, 3)).sum().backward()
            (os - yb[:, None]).square().mean((2, 3)).sum().backward()
            fg = (
                full.theta.grad.reshape(3, full.k, -1).index_select(1, ids).reshape_as(subset.theta)
            )
            sg = subset.theta.grad.detach().clone()
            injected.theta.grad = fg.detach().clone()
            stats = dict(
                outputs=difference(of.index_select(1, ids), os), gradients=difference(fg, sg)
            )
            opts = [
                BankAdamW(
                    net.parameters(),
                    np.repeat([setting["lr"]] * 3, net.k),
                    weight_decay=0.01,
                    max_norm=5.0,
                )
                for net in (full, subset, injected)
            ]
            correction = torch.as_tensor(
                np.array([1 - 0.9, np.sqrt(1 - 0.999)], np.float32), device="cuda"
            )
            for opt in opts:
                opt.step(correction)
            target = full.theta.reshape(3, full.k, -1).index_select(1, ids).reshape_as(subset.theta)
            stats.update(
                updated_weights=difference(target, subset.theta),
                injected_gradient_update=difference(target, injected.theta),
                first_moment=difference(
                    opts[0]
                    .m[0]
                    .reshape(3, full.k, -1)
                    .index_select(1, ids)
                    .reshape_as(subset.theta),
                    opts[2].m[0],
                ),
            )
            records.append(dict(role=role, family=family, prefix=j, **stats))
    assert len(records) == 18
    value = dict(
        source_lock_sha256=sha(RUN / "source_lock.json"),
        records=records,
        script_sha256=sha(ROOT / "scripts/chromaseed_neural_blocks_probe.py"),
        interpretation="Identical sliced-gradient injection isolates optimizer grouping; not a replacement fit or selector",
    )
    write_json(OUT / "first_step_probe.json", value)
    for key in (
        "outputs",
        "gradients",
        "updated_weights",
        "injected_gradient_update",
        "first_moment",
    ):
        print(
            key,
            dict(
                nonexact=sum(not r[key]["bitwise_equal"] for r in records),
                maximum=max(r[key]["maximum"] for r in records),
            ),
        )


if __name__ == "__main__":
    main()
