"""Fit-only covariance projections with compact analytic joint color readouts."""

from __future__ import annotations

import time

import numpy as np
from chromaseed_affine import color_metrics, design_matrix, export_model, gram_stats, solve_stats
from chromaseed_fast_kernel import streaming_landmarks
from chromaseed_gated import fit_gate, package_model
from chromaseed_gated import predict as g_predict
from chromaseed_kernel import coordinates, fit_normalizer, gaussian_kernel
from chromaseed_perceptual import make_basis, prepare
from scipy.spatial.distance import pdist
from skin_local_search_train import weights_for

FAMILIES = ("norm_static", "norm_joint_soft", "perceptual_static", "perceptual_joint_soft")
SEEDS, ALPHAS = (17, 29, 43), (0.1, 1.0, 10.0)
REPRESENTATIONS = (dict(name="raw", dimension=36, shrinkage=None),) + tuple(
    dict(name=f"d{d}_t{label}", dimension=d, shrinkage=t)
    for d in (8, 16, 36)
    for label, t in (("0", 0.0), ("01", 0.1), ("05", 0.5), ("1", 1.0))
)
POLICIES = ("quality", "compact")


def covariance(z, weights):
    z, w = np.asarray(z, np.float64), np.asarray(weights, np.float64)
    if (
        z.ndim != 2
        or z.shape[1] != 36
        or w.shape != (len(z),)
        or not len(z)
        or np.any(w <= 0)
        or not np.isfinite(z).all()
        or not np.isfinite(w).all()
    ):
        raise ValueError("finite color36 and positive original weights required")
    mean = np.average(z, axis=0, weights=w)
    centered = z - mean
    c = centered.T @ (w[:, None] * centered) / w.sum()
    eigen, vectors = np.linalg.eigh(0.5 * (c + c.T))
    eigen, vectors = eigen[::-1].copy(), vectors[:, ::-1].copy()
    clipped = int((eigen < 0).sum())
    eigen = np.maximum(eigen, 0.0)
    signs = np.where(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(36)] < 0, -1.0, 1.0)
    vectors *= signs
    return dict(
        mean=mean,
        eigenvalues=eigen,
        vectors=vectors,
        clipped=clipped,
        trace=float(np.trace(c)),
        covariance=c,
    )


def projection(z, weights, dimension, shrinkage, spectrum=None):
    if (
        not isinstance(dimension, int)
        or not 1 <= dimension <= 36
        or not np.isfinite(shrinkage)
        or not 0 <= shrinkage <= 1
    ):
        raise ValueError("dimension1..36 and shrinkage0..1 required")
    spectrum = covariance(z, weights) if spectrum is None else spectrum
    eigen = spectrum["eigenvalues"]
    scale = max(spectrum["trace"] / 36, 1e-12)
    denominators = (1 - shrinkage) * eigen[:dimension] + shrinkage * spectrum["trace"] / 36
    floor = 1e-4 * scale
    p = spectrum["vectors"][:, :dimension] / np.sqrt(np.maximum(denominators, floor))
    return dict(
        projection_mean=spectrum["mean"].astype(np.float32), projection=p.astype(np.float32)
    ), dict(
        dimension=dimension,
        shrinkage=shrinkage,
        eigenvalue_floor=floor,
        floored_directions=int(np.sum(denominators < floor)),
        covariance_negative_eigenvalues_clipped=spectrum["clipped"],
        retained_variance_fraction=float(eigen[:dimension].sum() / max(eigen.sum(), 1e-30)),
        cut_eigenvalue_gap=None
        if dimension == 36
        else float((eigen[dimension - 1] - eigen[dimension]) / max(eigen[0], 1e-30)),
    )


def project(model, x):
    z = coordinates(model, x)
    if "projection" not in model:
        return z
    return (
        ((z - model["projection_mean"].astype(np.float64)) @ model["projection"].astype(np.float64))
        .astype(np.float32)
        .astype(np.float64)
    )


def exact_width(x):
    x = np.asarray(x, np.float64)
    if x.ndim != 2 or not len(x) or not 1 <= x.shape[1] <= 36 or not np.isfinite(x).all():
        raise ValueError("finite feature matrix required")
    v = pdist(x, metric="sqeuclidean")
    count, memory = len(v), v.nbytes
    np.divide(v, x.shape[1], out=v)
    np.sqrt(v, out=v)
    v = v[v > 1e-10]
    if len(v):
        index = (len(v) - 1) // 2
        v.partition(index)
        width = max(float(v[index]), 1e-6)
    else:
        width = 1e-6
    return width, dict(
        pair_count=count,
        positive_pairs=len(v),
        condensed_distance_vector_bytes=memory,
        dimension=x.shape[1],
        quadratic_pair_storage=True,
    )


class VariableColumns:
    def __init__(self, x, width):
        self.x = np.asarray(x, np.float64)
        if (
            self.x.ndim != 2
            or not len(self.x)
            or not 1 <= self.x.shape[1] <= 36
            or not np.isfinite(self.x).all()
            or not np.isfinite(width)
            or width <= 0
        ):
            raise ValueError("finite feature matrix and positive width required")
        self.norms = np.einsum("ij,ij->i", self.x, self.x)
        self.denominator = 2 * self.x.shape[1] * float(width) ** 2
        self.entries_evaluated = 0

    def column(self, pivot):
        if not 0 <= pivot < len(self.x):
            raise IndexError("pivot outside fit rows")
        d = np.maximum(self.norms + self.norms[pivot] - 2 * (self.x @ self.x[pivot]), 0)
        value = np.exp(-d / self.denominator)
        value[pivot] = 1
        self.entries_evaluated += len(self.x)
        return value


def projected_basis(q, w, width, seed, rank):
    ids, columns, _, _, info = streaming_landmarks(VariableColumns(q, width), w, rank, seed)
    sub = columns[ids]
    values, vectors = np.linalg.eigh(0.5 * (sub + sub.T))
    keep = values > 1e-8 * max(float(values.max()), np.finfo(float).tiny)
    white = vectors[:, keep] / np.sqrt(values[keep])
    return dict(
        ids=ids, centers=q[ids].astype(np.float32), width=width, whitening=white, z=columns @ white
    ), dict(
        **info,
        effective_rank=int(keep.sum()),
        smallest_retained_eigenvalue=float(values[keep].min()),
        fit_center_indices=ids.tolist(),
    )


def model_id(family, seed, representation, alpha):
    return f"{family}_{representation}_s{seed}_a{ALPHAS.index(alpha)}"


def choose(candidates, policy):
    if policy not in POLICIES or not candidates:
        raise ValueError("registered policy and nonempty grid required")
    if policy == "quality":
        return min(
            candidates, key=lambda r: (r["clean"], r["numeric_bytes"], -r["alpha"], r["order"])
        )
    anchor = min(
        (r for r in candidates if r["representation"] == "raw"),
        key=lambda r: (r["clean"], r["p90"], -r["alpha"]),
    )
    eligible = [
        r
        for r in candidates
        if r["clean"] <= anchor["clean"] + 0.05 and r["p90"] <= anchor["p90"] + 0.10
    ]
    return min(
        eligible, key=lambda r: (r["numeric_bytes"], r["clean"], r["p90"], -r["alpha"], r["order"])
    )


def export(state, projection_payload, basis, theta, beta, joint):
    if not projection_payload:
        return export_model(state, basis, theta, beta, joint)
    dim = basis["whitening"].shape[1]
    base = dict(
        **state["prep"],
        **projection_payload,
        centers=basis["centers"],
        coefficient=(basis["whitening"] @ theta[:dim]).astype(np.float32),
        width=np.array(basis["width"], np.float32),
    )
    if joint and beta is not None:
        return package_model(base, basis["whitening"] @ theta[dim:], beta, "soft", 1.0)
    return base


def predict(model, x):
    if "constant_lab" in model:
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != 36 or not np.isfinite(x).all():
            raise ValueError("finite color36 matrix required")
        return np.broadcast_to(model["constant_lab"].astype(np.float64), (len(x), 3)).copy()
    if "projection" not in model:
        return g_predict(model, x)
    q = project(model, x)
    k = gaussian_kernel(q, model["centers"], float(model["width"]))
    value = k @ model["coefficient"].astype(np.float64)
    if "correction" in model:
        raw = coordinates(model, x)
        s = raw @ model["gate_beta"][1:].astype(np.float64) + float(model["gate_beta"][0])
        value += (
            float(model["rho"])
            * np.clip(s, -1, 1)[:, None]
            * (k @ model["correction"].astype(np.float64))
        )
    return value * model["y_std"] + model["y_mean"]


def representation_state(state, x, rep, spectrum):
    if rep["name"] == "raw":
        return (
            {},
            state["normalized"],
            float(np.float32(state["base"])),
            dict(kind="identity", dimension=36, width_info=state["width_info"]),
        )
    payload, info = projection(
        state["normalized"], state["weights"], rep["dimension"], rep["shrinkage"], spectrum
    )
    q = project({**state["prep"], **payload}, x)
    width, wi = exact_width(q)
    return payload, q, float(np.float32(width)), dict(**info, width_info=wi)


def basis_for(state, payload, q, width, seed, rank):
    if not payload:
        b = make_basis(state, seed, 1, rank, (0.1,))
        return b, dict(
            fit_center_indices=b["ids"].tolist(),
            actual_centers=len(b["ids"]),
            effective_rank=b["whitening"].shape[1],
        )
    return projected_basis(q, state["weights"], width, seed, rank)


def fit_bank(x, y, person, site, camera, rank=128):
    start = time.perf_counter()
    state = prepare(x, y, weights_for(person, site))
    spectrum = covariance(state["normalized"], state["weights"])
    beta = fit_gate(state["normalized"], person, site, camera)
    signal = (
        None if beta is None else state["normalized"] @ beta[1:].astype(np.float64) + float(beta[0])
    )
    structures = (False, True) if beta is not None else (False,)
    metrics = color_metrics(state)
    models, metadata, operations, bases, reps = {}, {}, [], [], []
    for rep in REPRESENTATIONS:
        payload, q, width, info = representation_state(state, x, rep, spectrum)
        reps.append(dict(**rep, width=width, info=info))
        for seed in SEEDS:
            b, bi = basis_for(state, payload, q, width, seed, rank)
            bases.append(dict(representation=rep["name"], seed=seed, **bi))
            for joint in structures:
                design = design_matrix(b["z"], signal, joint)
                solutions, si = solve_stats(
                    gram_stats(design, state["yn"], state["weights"]), metrics, ALPHAS
                )
                operations.append(dict(representation=rep["name"], seed=seed, joint=joint, **si))
                for (loss, alpha), theta in solutions.items():
                    family = f"{loss}_{'joint_soft' if joint else 'static'}"
                    name = model_id(family, seed, rep["name"], alpha)
                    models[name] = export(state, payload, b, theta, beta, joint)
                    metadata[name] = dict(
                        family=family,
                        seed=seed,
                        representation=rep["name"],
                        alpha=alpha,
                        fallback=False,
                    )
            if beta is None:
                for loss in ("norm", "perceptual"):
                    for alpha in ALPHAS:
                        family = loss + "_joint_soft"
                        name = model_id(family, seed, rep["name"], alpha)
                        models[name] = dict(
                            models[model_id(loss + "_static", seed, rep["name"], alpha)]
                        )
                        metadata[name] = dict(
                            family=family,
                            seed=seed,
                            representation=rep["name"],
                            alpha=alpha,
                            fallback=True,
                        )
    assert len(models) == 468
    return models, dict(
        models=metadata,
        operations=operations,
        bases=bases,
        representations=reps,
        new_coefficient_solutions=sum(len(r["solutions"]) for r in operations),
        gram_decompositions=len(operations),
        basis_preparations=39,
        raw_basis_preparations=3,
        projected_basis_preparations=36,
        exact_width_calculations=13,
        covariance_eigendecompositions=1,
        covariance_eigenvalues=spectrum["eigenvalues"].tolist(),
        covariance_trace=spectrum["trace"],
        gate_fits=int(beta is not None),
        auxiliary_baseline_solves=3,
        auxiliary_theta_solves=3,
        n_fit_rows=len(x),
        original_weight_mass=float(state["weights"].sum()),
        fit_bank_seconds=time.perf_counter() - start,
    )


def fit_single(x, y, person, site, camera, family, seed, representation, alpha, rank=128):
    start = time.perf_counter()
    if family not in FAMILIES or alpha not in ALPHAS or seed not in SEEDS:
        raise ValueError("registered family/alpha/seed required")
    rep = next((r for r in REPRESENTATIONS if r["name"] == representation), None)
    if rep is None:
        raise ValueError("registered representation required")
    w = weights_for(person, site)
    if representation == "raw":
        state = prepare(x, y, w)
        spectrum = None
    else:
        prep = fit_normalizer(x, y)
        state = dict(
            prep=prep,
            normalized=coordinates(prep, x),
            weights=w,
            y=np.asarray(y, np.float64),
            yn=(np.asarray(y, np.float64) - prep["y_mean"]) / prep["y_std"],
        )
        spectrum = covariance(state["normalized"], w)
    beta = (
        fit_gate(state["normalized"], person, site, camera)
        if family.endswith("joint_soft")
        else None
    )
    active = beta is not None
    signal = (
        None if beta is None else state["normalized"] @ beta[1:].astype(np.float64) + float(beta[0])
    )
    payload, q, width, info = representation_state(state, x, rep, spectrum)
    b, bi = basis_for(state, payload, q, width, seed, rank)
    loss = family.split("_")[0]
    solutions, si = solve_stats(
        gram_stats(design_matrix(b["z"], signal, active), state["yn"], w),
        color_metrics(state, (loss,)),
        (alpha,),
    )
    model = export(state, payload, b, solutions[(loss, alpha)], beta, active)
    return model, dict(
        fit_seconds=time.perf_counter() - start,
        active_gate=active,
        representation=representation,
        representation_info=info,
        basis=bi,
        **si,
    )
