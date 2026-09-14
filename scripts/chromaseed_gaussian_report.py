"""Describe the frozen TG experiment, including all losses and measured costs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from chromaseed_kernel_audit import js
from skin_local_search_train import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/runs/chromaseed_gaussian_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_gaussian_v1"
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-gaussian-2026-09-13.md"
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
METHODS = ("adam", "tagi_diag", "tagi_full3")
GROUPS = ("raw36", "mean3")
EPOCHS = (1, 4, 16, 64)


def csv_write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def same(rec, role, method, group):
    return (rec["role"], rec["method"], rec["group"]) == (role, method, group)


def main():
    assert not (OUT / "verification.json").exists(), "Sealed reports are read-only"
    result, selection, audit, runtime = (
        js(p)
        for p in (
            RUN / "results.json",
            RUN / "selections.json",
            OUT / "audit.json",
            OUT / "runtime.json",
        )
    )
    assert audit["passed"] and runtime["audit_sha256"] == sha(OUT / "audit.json")
    for obj in (result, audit, runtime):
        assert obj["source_lock_sha256"] == sha(RUN / "source_lock.json")
        assert obj["selection_sha256"] == sha(RUN / "selections.json")
    assert audit["results_sha256"] == runtime["results_sha256"] == sha(RUN / "results.json")
    rows, curves, doses, candidates = [], [], [], []
    for role in ROLES:
        for method in (*METHODS, "fg_norm_static", "constant"):
            for group in ("constant",) if method == "constant" else GROUPS:
                rr = [r for r in result["records"] if same(r, role, method, group)]
                tr = [r for r in runtime["records"] if same(r, role, method, group)]
                fr = [r for r in runtime["standalone_fit_records"] if same(r, role, method, group)]
                chosen = (
                    selection["roles"][role][method][group]["selected"]
                    if method in METHODS
                    else None
                )
                row = dict(
                    role=role,
                    method=method,
                    group=group,
                    parameter=rr[0]["parameter"],
                    epoch=rr[0]["epoch"],
                    seeds=len(rr),
                    inner_clean=None if chosen is None else chosen["clean"],
                    inner_p90=None if chosen is None else chosen["p90"],
                    person_mean=float(np.mean([r["metrics"]["person_mean"] for r in rr])),
                    image_mean=float(np.mean([r["metrics"]["image_mean"] for r in rr])),
                    p90=float(np.mean([r["metrics"]["p90"] for r in rr])),
                    stress4=float(
                        np.mean([r["doses"][1]["worst_error"]["person_mean"] for r in rr])
                    ),
                    median_us=float(np.median([r["numpy_only"]["median_us"] for r in tr])),
                    maximum_seed_p95_us=max(r["numpy_only"]["p95_us"] for r in tr),
                    full_fit_ms=None if not fr else fr[0]["median_seconds"] * 1000,
                    training_parameter_state_bytes=None
                    if not fr
                    else fr[0]["training_parameter_state_bytes"],
                )
                for field in ("numeric_bytes", "archive_bytes"):
                    for op in (min, max):
                        row[field + "_" + op.__name__] = op(r[field] for r in rr)
                for op in (min, max):
                    row["cached_array_bytes_" + op.__name__] = op(
                        r["numpy_only"]["cached_array_bytes"] for r in tr
                    )
                rows.append(row)
                for i, dose in enumerate((1 / 255, 4 / 255, 16 / 255, 64 / 255)):
                    doses.append(
                        dict(
                            role=role,
                            method=method,
                            group=group,
                            dose=dose,
                            worst_person_error=float(
                                np.mean([r["doses"][i]["worst_error"]["person_mean"] for r in rr])
                            ),
                        )
                    )
                for epoch in EPOCHS if method in METHODS else (None,):
                    cr = [
                        r
                        for r in result["curve_records"]
                        if same(r, role, method, group) and r["epoch"] == epoch
                    ]
                    curves.append(
                        dict(
                            role=role,
                            method=method,
                            group=group,
                            epoch=epoch,
                            parameter=rr[0]["parameter"],
                            seeds=len(cr),
                            selected_for_stress=all(r["selected_for_stress"] for r in cr),
                            person_mean=float(np.mean([r["metrics"]["person_mean"] for r in cr])),
                            p90=float(np.mean([r["metrics"]["p90"] for r in cr])),
                        )
                    )
                if method in METHODS:
                    for c in selection["roles"][role][method][group]["candidates"]:
                        candidates.append(
                            dict(
                                role=role,
                                **{k: v for k, v in c.items() if k != "seed_scores"},
                                selected=c["epoch"] == chosen["epoch"]
                                and c["parameter"] == chosen["parameter"],
                            )
                        )
    paired = [
        dict(
            **{k: v for k, v in p.items() if k != "fixed_prediction_person_bootstrap_95"},
            descriptive_95_low=p["fixed_prediction_person_bootstrap_95"][0],
            descriptive_95_high=p["fixed_prediction_person_bootstrap_95"][1],
        )
        for p in audit["paired"]
    ]
    variance = [
        dict(**{k: v for k, v in d.items() if k != "checkpoints"}, **c)
        for d in audit["variance_diagnostics"]
        for c in d["checkpoints"]
    ]
    diagnostics = []
    for method in METHODS:
        dd = [d for d in audit["variance_diagnostics"] if d["method"] == method]
        last = [d["checkpoints"][-1] for d in dd]
        diagnostics.append(
            dict(
                method=method,
                trajectories=len(dd),
                trajectories_with_floor=sum(c["floored_variance_updates"] > 0 for c in last),
                floored_variance_updates=sum(c["floored_variance_updates"] for c in last),
                negative_variance_updates=sum(c["negative_variance_updates"] for c in last),
                minimum_pre_floor_variance=None
                if method == "adam"
                else min(c["minimum_pre_floor_variance"] for c in last),
                minimum_variance=None
                if method == "adam"
                else min(c["minimum_variance"] for c in last),
            )
        )
    counts = dict(
        new_vs_FG=18,
        adverse_vs_FG=sum(
            p["control"] == "fg_norm_static" and p["mean_difference"] > 0 for p in paired
        ),
        gaussian_vs_Adam=12,
        adverse_vs_Adam=sum(p["control"] == "adam" and p["mean_difference"] > 0 for p in paired),
        selected_epoch64=sum(r["epoch"] == 64 for r in rows),
    )
    assert (len(rows), len(curves), len(doses), len(candidates), len(paired), len(variance)) == (
        27,
        81,
        108,
        216,
        30,
        2160,
    )
    summary = dict(
        source_lock_sha256=sha(RUN / "source_lock.json"),
        selection_sha256=sha(RUN / "selections.json"),
        results_sha256=sha(RUN / "results.json"),
        audit_sha256=sha(OUT / "audit.json"),
        runtime_sha256=sha(OUT / "runtime.json"),
        report_source_sha256=sha(Path(__file__)),
        rows=rows,
        curves=curves,
        doses=doses,
        candidates=candidates,
        paired=paired,
        variance_checkpoints=variance,
        variance_diagnostics=diagnostics,
        comparison_counts=counts,
        scope="Seed-average errors, not prediction ensembles. Reused TRAIN people; roles overlap. Clean and stress output counts overlap at identity.",
    )
    write_json(OUT / "summary.json", summary)
    for filename, field in (
        ("selected.csv", "rows"),
        ("learning_curves.csv", "curves"),
        ("all_doses.csv", "doses"),
        ("inner_candidates.csv", "candidates"),
        ("paired.csv", "paired"),
        ("variance_checkpoints.csv", "variance_checkpoints"),
    ):
        csv_write(OUT / filename, summary[field])

    def get(role, method, group):
        return next(r for r in rows if same(r, role, method, group))

    colors = dict(
        adam="#2463a6", tagi_diag="#d87927", tagi_full3="#8061ac", fg_norm_static="#27816b"
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), constrained_layout=True)
    for j, role in enumerate(ROLES):
        for i, group in enumerate(GROUPS):
            ax = axes[i, j]
            for method in METHODS:
                cr = [r for r in curves if same(r, role, method, group)]
                ax.plot(
                    EPOCHS, [r["person_mean"] for r in cr], "o-", color=colors[method], label=method
                )
                selected = get(role, method, group)
                ax.scatter(
                    selected["epoch"],
                    selected["person_mean"],
                    s=95,
                    facecolors="none",
                    edgecolors=colors[method],
                    linewidths=1.5,
                )
            ax.axhline(
                get(role, "fg_norm_static", group)["person_mean"],
                color=colors["fg_norm_static"],
                ls="--",
                label="FG analytic",
            )
            ax.set(
                xscale="log",
                xticks=EPOCHS,
                xticklabels=EPOCHS,
                title=f"{role} / {group}",
                xlabel="Epochs",
                ylabel="Person mean DeltaE00 (lower better)",
            )
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "TG learning curves: fixed inner-selected hyperparameter; rings = frozen epoch\nReused people, seed-average errors; these curves do not select on held roles",
        fontsize=12,
    )
    fig.savefig(OUT / "gaussian_learning_curves.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), constrained_layout=True)
    for ax, field, title, ylabel in zip(
        axes,
        ("numeric_bytes_min", "full_fit_ms", "median_us"),
        ("Deployment payload", "Complete selected-epoch fit", "Prepared-feature response"),
        ("Numeric bytes", "Milliseconds (log scale)", "Microseconds"),
        strict=True,
    ):
        for i, group in enumerate(GROUPS):
            values = [get("mixed", m, group)[field] for m in (*METHODS, "fg_norm_static")]
            ax.bar(
                np.arange(4) + (i - 0.5) * 0.36,
                values,
                width=0.36,
                label=group,
                color=("#355f86", "#e19b50")[i],
            )
        ax.set(
            xticks=np.arange(4),
            xticklabels=("Adam", "diag", "full3", "FG"),
            title=title,
            ylabel=ylabel,
        )
        if field == "full_fit_ms":
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=9)
    fig.suptitle(
        "Mixed role / one CPU thread: smaller network, slower training\nFit: seed17, median of two full repeats after warmup; query: median across three seed medians",
        fontsize=11,
    )
    fig.savefig(OUT / "gaussian_cost.png", dpi=170)
    plt.close(fig)
    table = [
        "| Role | Input | Method | Parameter | Epoch | Mean ΔE00 ↓ | p90 ↓ | Bytes | Fit ms | Query μs |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        fit = "—" if r["full_fit_ms"] is None else f"{r['full_fit_ms']:.2f}"
        table.append(
            f"| {r['role']} | {r['group']} | {r['method']} | {r['parameter']} | {r['epoch']} | {r['person_mean']:.6f} | {r['p90']:.4f} | {r['numeric_bytes_min']:,} | {fit} | {r['median_us']:.2f} |"
        )
    report = (
        """# Luma ChromaSeed TG: локальное гауссовское обучение

Маленькая ReLU-сеть работает: **451 параметр / 1 855 байт** по среднему RGB или **2 563 параметра / 10 564 байта** по 36 цветовым признакам. Ответ занимает около **5,4–6,1 мкс** на одном потоке CPU. Это компактнее и быстрее текущего аналитического FG-consumer, но обучение существенно медленнее. В **16/18** сравнений выбранные новые сети хуже FG; гауссовские варианты хуже соответствующего Adam в **7/12**. Универсального улучшения качества и эффективного обучения не получено.

Сравнение использует одинаковую архитектуру d→64 ReLU→3, начальные средние весов, порядок примеров и person→site→image веса. Три seed, два фиксированных набора входов, три параметра обучения, эпохи1/4/16/64. Внутренние оценки выбрали18 настроек до финального обучения;17 предпочли64 эпохи, reverse/raw36 Adam —16. Adam минимизирует взвешенную квадратичную ошибку; гауссовские процедуры поддерживают приближённые распределения весов. Совпадение архитектуры не означает одинаковую статистическую задачу или объём состояния обучения.

**Данные:** только исходный TRAIN,966 строк/24 человека, измеренный native D65/10° Lab участков кожи. Mixed:734 fit/232 query,18/6 человек; SLR→iPod:323/643,8/16; обратно643/323,16/8. Люди и роли повторно использованы и пересекаются между сценариями; камера связана с составом людей. Это исследовательская оценка, не независимый тест на селфи. Старые validation/calibration/test не открывались. ΔE00 — ошибка цвета, меньше лучше, её нельзя переводить в процент точности. Таблица усредняет ошибки трёх отдельных моделей, не их предсказания.

## Все зафиксированные варианты

Parameter — learning rate у Adam, стандартное отклонение наблюдения в нормализованном Lab у TAGI, alpha=.1 у FG. FG — точный импорт norm_static из предыдущего исследования, без условной или perceptual-поправки. Constant — сбалансированная константа. Размер — числовой payload; полный NPZ, FP64-кэш и состояние обучения отдельно в CSV. Fit включает нормализацию, веса примеров, инициализацию, все выбранные эпохи и экспорт; FG заново считает ширину, центры и readout. Измерены seed17, один прогрев и два полных повтора. Query — медиана по seed-медианам реального NumPy-consumer.

"""
        + "\n".join(table)
        + """

## Что показывают качество и длительное обучение

Mixed/raw36: FG5.438652, Adam5.506610, diag5.548741, full3 5.553386. При 64 эпохах Gaussian не превосходит Adam, хотя первые эпохи здесь лучше. Mixed mean3 ухудшается:6.547224→6.703580/6.982556/6.961798. Прямой raw36-перенос full3 немного улучшает FG8.597000→8.509649, но описательный парный интервал разницы[-.5294,.3301] включает ноль; улучшаются8/16 человек. Обратный raw36-перенос ухудшается8.705018→9.995897,0/8 человек улучшаются. Интервалы условны на фиксированных предсказаниях повторно использованных людей и не учитывают весь процесс выбора модели.

Больше эпох не гарантирует перенос: reverse/raw36 Adam меняется7.4506→7.6293→8.9219→9.5548; внутренний выбор всё равно16. Мы сохраняем этот выбранный результат. Прямой перенос тоже немонотонен. Увеличивать бюджет только из-за17/18 выборов64 без нового протокола нельзя считать установленным решением проблемы.

![All frozen learning curves](gaussian_learning_curves.png)

## Реальная стоимость

На734 mixed fit-строках raw36: FG18.27 мс, Adam2387.04 мс, diag3883.12 мс, full3 4482.17 мс. Mean3:12.16/1994.96/3490.45/4024.02 мс. Это конкретная NumPy FP64 online-реализация на одном потоке CPU, не предел быстродействия метода и не CUDA/cuTAGI benchmark. Батчирование независимых траекторий ускоряет общий поиск, но не подменяет время обучения одной модели. Все72 полных повторных обучения дали побитно одинаковые экспортированные массивы.

Сеть хранит только средние FP32-веса и нормализаторы. Временный постоянный массив параметров при обучении: Adam61 512/10 824 байта для raw36/mean3, Gaussian41 008/7 216. Это средние и два момента Adam либо средние и дисперсии Gaussian; данные, временные матрицы, RNG и память процесса не включены. FP64-кэш consumer20 816/3 659 байт, не весь Python-процесс. Изображения, детектор лица, выделение кожи, подготовка признаков, мобильный runtime и I/O не измерялись. GPU в этой серии не использовался.

![Measured deployment and training costs](gaussian_cost.png)

## Численная устойчивость и воспроизводимость

У диагонального варианта в одной из180 траекторий обнаружены2 отрицательных обновления дисперсии до ограничения и6 881 срабатывание floor=1e-12. Это mixed/inner/fold2/mean3/sigma.1/seed17; минимум до floor−.0026853. Сумма считается по конечным кумулятивным счётчикам каждой траектории, без повторного сложения префиксов. У full3 все180 траекторий без отрицательных дисперсий и без floor; минимум2.1732e-8. Floor был зафиксирован до обучения. Стабильность full3 полезна для корректности приближения, но сама по себе не улучшает точность. Эти дисперсии не являются калиброванной уверенностью в цвете кожи.

540 траекторий,2 160 обученных checkpoint и84 точных FG-контроля в12 банках;13 708 800 обновлений по примерам. Независимый аудит проверил379 100 OOF-выходов,216 кандидатов/18 выборов,94 642 clean-выхода и988 350 stress-выходов. Clean и stress пересекаются по identity: их суммы не означают независимые наблюдения. Все54 финальные траектории независимо обучены скалярным кодом с Cholesky-решением; все216 экспортированных сетей совпали точно, максимум расхождения параметров0. Все фактические consumer проверены. 18 численных тестов прошли до обучения; финальный повтор и хеши находятся в verification.json. Ни новый датасет, ни веса, ни пакеты не загружались.

## Источник метода и следующие действия

[TAGI, Goulet et al., JMLR2021](https://www.jmlr.org/papers/v22/20-1009.html) дал исходную идею приближённых гауссовских распределений и обратного послойного вывода. Наши diag/full3 — самостоятельные ограниченные реализации с локальной линеаризацией ReLU; full3 сохраняет ковариацию трёх выходов, скрытые состояния и параметры остаются диагональными. Это не перебор весов и не воспроизведение авторского cuTAGI benchmark. Формулы и отличия зафиксированы в протоколе.

Следующая гипотеза — проверить аналитическое обучение выходного слоя этой же маленькой ReLU-сети после случайной и коротко обученной основы. В архиве уже есть frozen random/palette-readout; сама идея не новая. Нужна отдельная сравнимая постановка на TG-архитектуре и независимый контроль, без повторного выбора по известным внешним ошибкам. Она пока не запущена. Общее качество на обычных фото лиц остаётся непроверенным; общая цель активна.

[Шесть полных CSV](selected.csv): [эпохи](learning_curves.csv), [stress](all_doses.csv), [внутренние кандидаты](inner_candidates.csv), [парные сравнения](paired.csv), [дисперсии](variance_checkpoints.csv). [Аудит](audit.json) · [Все замеры](runtime.json) · [Машинная сводка](summary.json) · [Воспроизведение](reproduce.md) · [Receipt](verification.json) · [Протокол](../../research/chromaseed_gaussian_v1_protocol.md) · [Модель](../../architecture/chromaseed_gaussian_model_card.md) · [Следующее решение](../../research/chromaseed_gaussian_next_decision.md) · [Форумы и данные](../../research/chromaseed_forum_methods_data_2026-09-13.md).
"""
    )
    (OUT / "report.md").write_text(report, encoding="utf-8")
    card = r"""# ChromaSeed TG model contract

Research regressors from prepared color36 skin-region statistics to native instrument D65/10° Lab3. No identity recognition, face detection, skin segmentation, ordinary-phone color guarantee or cosmetic shade catalogue is included. Same deployment architecture for Adam, TAGI-diag and TAGI-full3: d→64 ReLU→3. Mean3 consumes coordinates27/28/29; raw36 consumes all36. 451/2563 network parameters; 1855/10564 numeric bytes including FP32 normalizers and mean3 uint8 indices. Output is deterministic; posterior variances are discarded and are not a confidence estimate.

Input is finite encoded-sRGB color36 in the existing prepared-feature contract, not linear RGB or arbitrary three-channel images. Mean3 consumer still validates all36 coordinates and ignores the finite unused ones. One row or a batch is accepted. Fit-only population moments are computed in FP64 and stored FP32; x normalization is performed in FP32, then the network computes with cached FP64 weights. Output is rescaled to native Lab. Cached arrays3659/20816 B exclude Python object and process overhead. Numeric storage, compressed NPZ bytes, optimizer state and actual process RAM are distinct quantities.

```python
import sys
from pathlib import Path
import numpy as np

root = Path(r"C:\Users\dimal\Documents\просто\.worktrees\luma-local-search")
sys.path.insert(0, str(root / "scripts"))
from chromaseed_gaussian_numpy import Predictor

path = root / "experiments/runs/chromaseed_gaussian_v1/models/mixed/adam_raw36_s17_h0_e64.npz"
with np.load(path, allow_pickle=False) as archive:
    predictor = Predictor({k: archive[k] for k in archive.files})
# prepared_color36 must come from the existing skin-region preprocessing contract.
# lab = predictor(prepared_color36)
```

The example is a frozen individual model, not a production recommendation. Reported quality averages seed errors; inference does not ensemble. Original TRAIN only966 rows/24 people, reused and overlapping roles; no fresh phone-face evidence. Full3 avoids the observed diag variance-floor failure but does not establish a better quality/cost frontier. Most selected networks lose to the exact FG norm_static controls. Model names and implementation do not establish patent novelty, trademark clearance or Skolkovo admission.

[Report](../benchmarks/chromaseed_gaussian_v1/report.md) · [Protocol and formulas](../research/chromaseed_gaussian_v1_protocol.md) · [Receipt](../benchmarks/chromaseed_gaussian_v1/verification.json).
"""
    (ROOT / "docs/architecture/chromaseed_gaussian_model_card.md").write_text(
        card, encoding="utf-8"
    )
    decision = """# After TG: analytical readout on the same compact network

TG is verified progress, not completion of the compact/fast/high-quality goal. Matched small networks are smaller and faster to query than the current analytic-kernel consumer, but online fitting is much slower and16/18 selected comparisons worsen versus FG. Gaussian inference worsens7/12 comparisons versus matched Adam. All84 FG controls are exact; independent54 trajectories reproduce216 exported checkpoints exactly. Preserve negative results, all epochs and variance counters. Do not enlarge the optimizer grid or replace the frozen reverse epoch using known outer errors.

The proposed next bounded question is whether an analytical output-layer refit can retain this network's compact deployment while avoiding long online adaptation. This is NOT the first frozen/readout idea in the archive: [original palette decision](chromaseed_next_decision.md), [original model contract](../architecture/chromaseed_model_card.md) and `scripts/chromaseed_frozen.py` already cover frozen random/clean/rendered/shuffled encoders with analytical heads. Prior kernel/perceptual studies also contain weighted readout solvers. Inventory these specific controls before any implementation and preserve their measured negative transfer; do not rerun their exposed tests.

Register a small matched study on exactly TG's raw36/mean3 inputs and d→64 ReLU→3 architecture. Include random hidden features and fixed early learned hidden features as controls; compare analytic refit to each corresponding unchanged TG checkpoint. Keep fit-only normalization, person/site weights, seed/order exposure and regularization choices explicit. If comparing standardized versus perceptual readout losses, fix a small grid and a person-disjoint inner selection before final fits. Charge all representation training to full model construction; cached TG hidden layers may help diagnostics but must not produce a misleading zero-cost training claim. Independently reconstruct the linear solution and verify payload bytes and actual consumer. Do not add conditional role-specific fallbacks from known losses.

This follow-up is planned, not implemented or launched. Only original TRAIN is permitted by the current research boundary; no images, new datasets, packages, weights, agents, publication or legacy validation/test access. Forum/data search remains available as a separate route toward suitable licensed evidence. More repeated fits cannot establish ordinary-phone face quality. The full goal is active and not blocked because meaningful local learning work remains.

[TG report](../benchmarks/chromaseed_gaussian_v1/report.md) · [Receipt](../benchmarks/chromaseed_gaussian_v1/verification.json) · [Methods and data leads](chromaseed_forum_methods_data_2026-09-13.md).
"""
    (ROOT / "docs/research/chromaseed_gaussian_next_decision.md").write_text(
        decision, encoding="utf-8"
    )
    reproduce = """# Reproducing and verifying TG

Use the original project's existing `.venv/Scripts/python.exe` from this worktree. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. Only the original `data/processed/skin_mskcc_pixels_v1/train.npz` is allowed. No new installations. Never run G/GS verifier mains; their old receipts are immutable inputs.

Execution record (the canonical run is sealed; do not overwrite it):

1. `python -m pytest tests/test_chromaseed_gaussian.py -q` —18 numerical tests passed before primary fitting in2.41 s.
2. `python scripts/chromaseed_feature_groups_verify.py --cache <original-train.npz>` —fresh parent check read-only, session59348 exit0, unchanged receipt459e6554bfba7ce22c63f7377556648632ba07406d20e1d50a42351d8bbf8908.
3. `python scripts/chromaseed_gaussian_train.py run --run experiments/runs/chromaseed_gaussian_v1 --cache <original-train.npz>` —PID26824/session6175, terminal exit0. Nine inner and three final banks,237.595 s excluding imports/audit/runtime.
4. `python scripts/chromaseed_gaussian_audit.py --run experiments/runs/chromaseed_gaussian_v1 --output docs/benchmarks/chromaseed_gaussian_v1 --cache <original-train.npz>` —session99472 exit0,159.493 s.
5. `python scripts/chromaseed_gaussian_runtime.py --run experiments/runs/chromaseed_gaussian_v1 --output docs/benchmarks/chromaseed_gaussian_v1 --cache <original-train.npz>` —session47215 exit0,138.297 s; no concurrent heavy job.
6. `python scripts/chromaseed_gaussian_report.py` —summary, six CSVs, two figures, model contract and next-decision before sealing.
7. `python scripts/chromaseed_gaussian_verify.py --cache <original-train.npz>` —seals once; subsequent invocation validates the same receipt read-only and recursively checks FG/C/H/X/A without rewriting.

Future full replay requires a new run/output directory and separately named orchestration; train/audit/runtime accept the corresponding paths. Canonical report/verifier source binds this run and must not be edited after sealing. Retain the complete old checkpoint/OOF/evaluation archives. The plan and current-status ledgers remain mutable and are not evidence substitutes.

[Protocol](../../research/chromaseed_gaussian_v1_protocol.md) · [Audit](audit.json) · [Timing](runtime.json) · [Receipt](verification.json).
"""
    (OUT / "reproduce.md").write_text(reproduce, encoding="utf-8")
    shortcut = f"""# Luma ChromaSeed TG: результаты проверки гауссовского обучения

540 траекторий обучения завершены. Маленькая сеть хранит **1,855 КБ** по среднему RGB или **10,564 КБ** по36 признакам; ответ около5,4–6,1 мкс по готовым признакам на одном потоке CPU.

В mixed/raw36: аналитическая FG — **5,439 ΔE00 /18,27 мс обучение**, Adam — **5,507 /2,39 с**, Gaussian diag — **5,549 /3,88 с**, full3 — **5,553 /4,48 с**. Меньше ошибка — лучше. Новые сети хуже FG в16/18 сравнений. Full3 исправляет наблюдавшуюся проблему отрицательной дисперсии diag, но универсального выигрыша точности нет. Размер и ответ улучшились, обучение замедлилось.

Все54 финальные траектории независимо переобучены,216 экспортированных сетей совпали точно;72 полных timing-повтора также точны. Серия завершена, фоновых процессов не осталось. Общая цель активна: качество на обычных фото лиц пока не подтверждено. Аналитический readout той же маленькой сети — следующий запланированный эксперимент, ещё не запущен.

[Все таблицы и графики]({(OUT / "report.md").as_posix()}) · [Проверка]({(OUT / "verification.json").as_posix()}) · [Модель]({(ROOT / "docs/architecture/chromaseed_gaussian_model_card.md").as_posix()}) · [Методы и данные с форумов]({(ROOT / "docs/research/chromaseed_forum_methods_data_2026-09-13.md").as_posix()}).
"""
    SHORTCUT.write_text(shortcut, encoding="utf-8")
    print(
        dict(
            rows=len(rows),
            curves=len(curves),
            candidates=len(candidates),
            variance=len(variance),
            comparisons=counts,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
