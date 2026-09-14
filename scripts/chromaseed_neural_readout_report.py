"""Full NR comparison tables and model-cost report from frozen audited records."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from chromaseed_kernel_audit import js
from skin_local_search_train import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/runs/chromaseed_neural_readout_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_neural_readout_v1"
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-neural-readout-2026-09-13.md"
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
GROUPS = ("raw36", "mean3")
FAMILIES = ("norm", "perceptual")
BASES = (
    "random",
    "adam_e1",
    "adam_e4",
    "adam_e16",
    "tagi_full3_e1",
    "tagi_full3_e4",
    "tagi_full3_e16",
)


def key(r):
    return tuple(r[k] for k in ("role", "group", "family", "basis"))


def csv_write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    assert not (OUT / "verification.json").exists(), "sealed reports are read-only"
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
        assert obj["source_lock_sha256"] == sha(RUN / "source_lock.json") and obj[
            "selection_sha256"
        ] == sha(RUN / "selections.json")
    assert audit["results_sha256"] == runtime["results_sha256"] == sha(RUN / "results.json")
    rows, doses, candidates, policies = [], [], [], []
    for k in sorted({key(r) for r in result["records"]}):
        role, group, family, basis = k
        rr, tr, fr = (
            [r for r in records if key(r) == k]
            for records in (
                result["records"],
                runtime["records"],
                runtime["standalone_fit_records"],
            )
        )
        chosen = (
            selection["roles"][role][group][family]["bases"][basis]["selected"]
            if family in FAMILIES
            else None
        )
        row = dict(
            role=role,
            group=group,
            family=family,
            basis=basis,
            alpha=rr[0]["alpha"],
            seeds=len(rr),
            policy_selected=all(r["policy_selected"] for r in rr),
            inner_clean=None if chosen is None else chosen["clean"],
            inner_p90=None if chosen is None else chosen["p90"],
            person_mean=float(np.mean([r["metrics"]["person_mean"] for r in rr])),
            image_mean=float(np.mean([r["metrics"]["image_mean"] for r in rr])),
            p90=float(np.mean([r["metrics"]["p90"] for r in rr])),
            stress4=float(np.mean([r["doses"][1]["worst_error"]["person_mean"] for r in rr])),
            median_us=float(np.median([r["numpy_only"]["median_us"] for r in tr])),
            maximum_seed_p95_us=max(r["numpy_only"]["p95_us"] for r in tr),
            full_fit_ms=None if not fr else fr[0]["median_seconds"] * 1000,
            representation_training_parameter_state_bytes=None
            if not fr
            else fr[0]["representation_training_parameter_state_bytes"],
        )
        for field in ("numeric_bytes", "archive_bytes"):
            for op in (min, max):
                row[field + "_" + op.__name__] = op(r[field] for r in rr)
        for op in (min, max):
            row["cached_array_bytes_" + op.__name__] = op(
                r["numpy_only"]["cached_array_bytes"] for r in tr
            )
        for field in ("temporary_design_bytes", "temporary_metric_bytes", "temporary_system_bytes"):
            row["head_" + field] = (
                None
                if not fr or fr[0]["head_training_array_bytes"] is None
                else fr[0]["head_training_array_bytes"][field]
            )
        rows.append(row)
        for i, dose in enumerate((1 / 255, 4 / 255, 16 / 255, 64 / 255)):
            doses.append(
                dict(
                    role=role,
                    group=group,
                    family=family,
                    basis=basis,
                    dose=dose,
                    worst_person_error=float(
                        np.mean([r["doses"][i]["worst_error"]["person_mean"] for r in rr])
                    ),
                )
            )
        if chosen is not None:
            for c in selection["roles"][role][group][family]["bases"][basis]["candidates"]:
                candidates.append(
                    dict(
                        role=role,
                        **{k: v for k, v in c.items() if k != "seed_scores"},
                        selected=c["alpha_index"] == chosen["alpha_index"],
                    )
                )

    def get(role, group, family, basis="reference"):
        return next(r for r in rows if key(r) == (role, group, family, basis))

    for role in ROLES:
        for group in GROUPS:
            for family in FAMILIES:
                c = selection["roles"][role][group][family]["policy"]
                r = get(role, group, family, c["basis"])
                baseline = get(role, group, "fg_norm_static")
                policies.append(
                    dict(
                        role=role,
                        group=group,
                        family=family,
                        basis=c["basis"],
                        alpha=c["alpha"],
                        inner_clean=c["clean"],
                        inner_p90=c["p90"],
                        person_mean=r["person_mean"],
                        p90=r["p90"],
                        stress4=r["stress4"],
                        FG_person_mean=baseline["person_mean"],
                        difference_vs_FG=r["person_mean"] - baseline["person_mean"],
                        numeric_bytes=r["numeric_bytes_min"],
                        full_fit_ms=r["full_fit_ms"],
                        median_us=r["median_us"],
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
    counts = {
        kind: dict(
            comparisons=sum(p["kind"] == kind for p in paired),
            adverse=sum(p["kind"] == kind and p["mean_difference"] > 0 for p in paired),
        )
        for kind in ("head_vs_FG", "head_vs_unchanged", "policy_vs_FG")
    }
    alpha_counts = dict(Counter(str(r["alpha"]) for r in rows if r["family"] in FAMILIES))
    assert (len(rows), len(doses), len(candidates), len(policies), len(paired)) == (
        129,
        516,
        252,
        12,
        168,
    )
    summary = dict(
        source_lock_sha256=sha(RUN / "source_lock.json"),
        selection_sha256=sha(RUN / "selections.json"),
        results_sha256=sha(RUN / "results.json"),
        audit_sha256=sha(OUT / "audit.json"),
        runtime_sha256=sha(OUT / "runtime.json"),
        report_source_sha256=sha(Path(__file__)),
        rows=rows,
        doses=doses,
        candidates=candidates,
        policies=policies,
        paired=paired,
        comparison_counts=counts,
        selected_alpha_counts=alpha_counts,
        evidence="Seed-average errors, not ensemble; original TRAIN people and roles reused/confounded.",
    )
    write_json(OUT / "summary.json", summary)
    for filename, field in (
        ("all_models.csv", "rows"),
        ("all_doses.csv", "doses"),
        ("inner_candidates.csv", "candidates"),
        ("policies.csv", "policies"),
        ("paired.csv", "paired"),
    ):
        csv_write(OUT / filename, summary[field])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.1), constrained_layout=True)
    for i, group in enumerate(GROUPS):
        for j, role in enumerate(ROLES):
            ax = axes[i, j]
            for family, color in (("norm", "#27629b"), ("perceptual", "#b26636")):
                values = [get(role, group, family, basis)["person_mean"] for basis in BASES]
                ax.plot(range(7), values, "o-", label=family, color=color)
                policy = next(
                    p
                    for p in policies
                    if (p["role"], p["group"], p["family"]) == (role, group, family)
                )
                ax.scatter(
                    BASES.index(policy["basis"]),
                    policy["person_mean"],
                    s=95,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.5,
                )
            values = [np.nan] + [
                get(role, group, "unchanged", basis)["person_mean"] for basis in BASES[1:]
            ]
            ax.plot(range(7), values, "x--", color="#929292", label="unchanged network")
            ax.axhline(
                get(role, group, "fg_norm_static")["person_mean"],
                color="#298373",
                ls="--",
                label="FG analytic",
            )
            ax.set(
                xticks=range(7),
                xticklabels=("random", "A1", "A4", "A16", "G1", "G4", "G16"),
                title=f"{role} / {group}",
                ylabel="Person mean DeltaE00 (lower better)",
            )
            ax.tick_params(axis="x", labelsize=9)
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "NR: analytical head vs unchanged basis output and FG\nA = Adam, G = full3 Gaussian, number = epochs; rings = frozen inner-selected policies",
        fontsize=12,
    )
    fig.savefig(OUT / "neural_readout_quality.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5), constrained_layout=True)
    for ax, field, title, ylabel in zip(
        axes,
        ("numeric_bytes_min", "full_fit_ms", "median_us"),
        ("Deployment payload", "Complete model construction", "Prepared-feature response"),
        ("Numeric bytes", "Milliseconds (log scale)", "Microseconds"),
        strict=True,
    ):
        for i, group in enumerate(GROUPS):
            chosen = [
                next(
                    p
                    for p in policies
                    if (p["role"], p["group"], p["family"]) == ("mixed", group, f)
                )["basis"]
                for f in FAMILIES
            ]
            cases = [
                get("mixed", group, "fg_norm_static"),
                get("mixed", group, "norm", "random"),
                get("mixed", group, "norm", chosen[0]),
                get("mixed", group, "perceptual", chosen[1]),
            ]
            ax.bar(
                np.arange(4) + (i - 0.5) * 0.36,
                [r[field] for r in cases],
                width=0.36,
                label=group,
                color=("#355f86", "#e19b50")[i],
            )
        ax.set(
            xticks=np.arange(4),
            xticklabels=("FG", "Random\nnorm", "Policy\nnorm", "Policy\nperceptual"),
            title=title,
            ylabel=ylabel,
        )
        if field == "full_fit_ms":
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=9)
    fig.suptitle(
        "Mixed role / one CPU thread: complete fits rebuild the representation\nTwo measured seed17 fits after warmup; query median across three seed medians; image processing excluded",
        fontsize=11,
    )
    fig.savefig(OUT / "neural_readout_cost.png", dpi=170)
    plt.close(fig)
    policy_table = [
        "| Role | Input | Loss | Chosen basis | Error ΔE00 ↓ | FG ↓ | Full fit ms | Query μs |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for p in policies:
        policy_table.append(
            f"| {p['role']} | {p['group']} | {p['family']} | {p['basis']} | {p['person_mean']:.6f} | {p['FG_person_mean']:.6f} | {p['full_fit_ms']:.3f} | {p['median_us']:.2f} |"
        )
    random_table = [
        "| Role | Input | Loss | Error ΔE00 ↓ | Full fit ms | Numeric B |",
        "|---|---|---|---:|---:|---:|",
    ]
    for role in ROLES:
        for group in GROUPS:
            for family in FAMILIES:
                r = get(role, group, family, "random")
                random_table.append(
                    f"| {role} | {group} | {family} | {r['person_mean']:.6f} | {r['full_fit_ms']:.3f} | {r['numeric_bytes_min']} |"
                )
    r0, r3, fg = (
        get("mixed", "raw36", "norm", "random"),
        get("mixed", "mean3", "norm", "random"),
        get("mixed", "raw36", "fg_norm_static"),
    )
    report = (
        f"""# Luma ChromaSeed NR: аналитический выход маленькой сети

Выходной слой действительно обучается аналитически и складывается обратно в прежнюю сеть: **10 564 байта** для36 признаков или **1 855 байт** для среднего RGB. В mixed случайная raw36-основа плюс norm-readout полностью создаётся за **{r0["full_fit_ms"]:.3f} мс**, mean3 — **{r3["full_fit_ms"]:.3f} мс**. Но ошибки соответственно **{r0["person_mean"]:.6f} / {r3["person_mean"]:.6f} ΔE00**. Предыдущий FG raw36 даёт **{fg["person_mean"]:.6f}** при **{fg["full_fit_ms"]:.3f} мс**. Меньше ошибка — лучше; это не процент точности.

Аналитическая замена улучшила исходный обученный выход в54/72 сравнений, но79/84 новых вариантов хуже соответствующего FG. Все12 заранее зафиксированных правил выбора уступают FG. Есть работающий компромисс размера и стоимости, но универсального улучшения всей технологии нет. Для вариантов с обученной основой полное время ниже включает её создание; быстрый пересчёт последнего слоя не считается бесплатным обучением всей модели.

## Протокол и все правила выбора

Одна архитектура d→64 ReLU→3; raw36 и mean3 для всех сценариев. Семь основ: случайная, Adam после1/4/16эпох(rate.001), full3 Gaussian после1/4/16эпох(sigma1). Три seed17/29/43. По каждой основе проверены norm и локальная квадратичная аппроксимация ΔE00, alpha.1/1/10 с незарегуляризованным свободным членом. Нормализаторы и первая матрица при пересчёте выхода неизменны. Данные для новых фотографий не загружались.

Внутренние разбиения по людям зафиксировали84 alpha и12 правил выбора основы до финального обучения. В77/84 случаях выбранalpha10, ещё7 —alpha1. Все12 правил выбралиalpha10. Это граница проверенной сетки; более сильная регуляризация здесь не испытана. Нельзя объявлять найденные настройки глобальным оптимумом или задним числом выбрать удобный внешний результат.

"""
        + "\n".join(policy_table)
        + """

Каждая ошибка сначала усредняется по людям, затем по трём отдельным моделям. Это не ансамбль предсказаний. p90 и все127 сравниваемых варианта на сценарий доступны в CSV, включая проигравшие и неизменённые сети. Правила выбора используют внутреннюю ошибку и заранее определённые tie-breaks; внешние оценки не меняют выбор.

![All bases and objectives](neural_readout_quality.png)

## Случайная основа: полная стоимость без предобучения

"""
        + "\n".join(random_table)
        + f"""

Полное обучение измерено для126 настроек: по3 повтора с нуля, первый прогрев исключён, два остальных образуют медиану. Все **{sum(r["exact_payload_repeats"] for r in runtime["standalone_fit_records"])}/378** экспортированных payload совпали побитно. Измерены нормализация, инициализация, все эпохи основы, подготовка признаков, решение и экспорт; для FG заново рассчитаны ширина/центры/readout. Это NumPy FP64 на одном потоке CPU, не мобильная или CUDA скорость.

![Construction and deployment cost](neural_readout_cost.png)

**Размер** — числовые веса с нормализаторами, не NPZ-файл и не весь процесс Python. Развёрнутый FP64-кэш3659/20816 B для mean3/raw36. Дополнительные скрытые моменты, матрицы метрики и линейные системы нужны только при обучении. Их явные размеры есть в CSV/runtime; сумма таких фазовых массивов не является измерением пикового RAM. Извлечение лица/кожи, цветовых признаков, чтение изображений, I/O и телефонный runtime не входят в микросекунды ответа.

## Проверка результата и ограничения

Исходный TRAIN:966 строк/24 человека с приборным native D65/10° Lab участков кожи. Mixed fit734/query232,18/6человек; SLR→iPod323/643,8/16; обратно643/323,16/8. Люди и роли повторно используются и пересекаются между сценариями; камера связана с составом людей. Это не свежий тест на обычных селфи. Старые validation/calibration/test не открывались. Синтетические33 изменения цвета диагностируют чувствительность, не доказывают естественную цветопередачу камеры.

144 траектории/432 checkpoint и72 случайные основы,3024 аналитических решения,3540 записей в12 банках.979200 обновлений по примерам. Все501500 OOF-выходов,252 оценки/84 выбора/12 правил и5020818 финальных выходов независимо проверены, включая381 фактический consumer.369 совпадающих TG-основ и84 FG-контроля сохранены точно. Независимые144 скалярных обучения восстановили504 основы:503 побитно, у одной максимум7.45058e-9 вw1. Это внутри заранее заданных2e-6atol/rtol; порог не менялся.

Все3024 выходных решения восстановлены другим методом: расширенная задача наименьших квадратов с pivoted QR вместо нормальных уравнений/Cholesky, аналитическая цветовая метрика вместо конечных разностей. Максимальное отличие на fit/query —{audit["maxima"]["qr_prediction"]:.8g} Lab, на выбранном stress —{audit["maxima"]["qr_stress_prediction"]:.8g}; допустимый предел.001. Consumer отличается от сохранённых выходов не более{audit["maxima"]["consumer_prediction"]:.8g}.24 численных теста прошли до обучения; финальный повтор записан в receipt.

Первый аудит завершил вычисления, но остановился на моём дополнительном требовании504 побитных совпадений, которого не было в протоколе. Счётчик заменён на измеряемое число точных совпадений с отдельной записью расхождения, а исходные численные допуски сохранены. Повторный полный аудит прошёл. Код обучения/данные/веса не менялись. Отдельно исправлена описательная ошибка старой TG model card: быстрый Predictor принимает один color36, пакетная функция — отдельный интерфейс; старые измерения уже использовали правильный вызов.

Первичный workflow {js(RUN / "workflow.json")["wall_seconds"]:.3f} с; успешный аудит{audit["wall_seconds"]:.3f} с; profiling{runtime["wall_seconds"]:.3f} с. Эти расходы исследования не равны времени обучения одной модели. Исследование завершено; новых фоновых процессов после проверок не осталось. Общая цель активна, качество на обычных фото лиц ещё не подтверждено.

## Что проверять следующим

В старом ChromaSeed уже были frozen random/palette-readout и аналитические kernel-readout; NR не объявляет эти идеи новыми. Его вклад — проверенное сравнение на одной конкретной компактной TG-архитектуре с полным учётом стоимости. Следующий ограниченный вопрос — влияние более сильной регуляризации на эту же замороженную основу. Поскольку77/84 alpha и все12 правил упёрлись в10, до новой сетки нужно проверить геометрию штрафа и выбор на внутренних людях. Новые результаты нельзя получить перестановкой уже просмотренных внешних строк. Следующая серия пока не запущена.

[Все модели](all_models.csv) · [Все дозы](all_doses.csv) · [Внутренние кандидаты](inner_candidates.csv) · [Правила выбора](policies.csv) · [Парные сравнения](paired.csv) · [JSON](summary.json) · [Аудит](audit.json) · [Замеры](runtime.json) · [Receipt](verification.json) · [Протокол](../../research/chromaseed_neural_readout_v1_protocol.md) · [Воспроизведение](reproduce.md) · [Модель](../../architecture/chromaseed_neural_readout_model_card.md) · [Следующее решение](../../research/chromaseed_neural_readout_next_decision.md) · [Поправка к TG интерфейсу](../../research/chromaseed_gaussian_consumer_erratum.md).
"""
    )
    (OUT / "report.md").write_text(report, encoding="utf-8")
    card = r"""# ChromaSeed NR model contract

Prepared skin-region color36→native instrument D65/10° Lab3. Network d→64 ReLU→3; mean3 uses coordinates27/28/29, raw36 uses all36.451/2563 network parameters,1855/10564 numeric B including FP32 normalizers and optional uint8 indices. Every hidden payload is preserved during analytic fitting; hidden standardization folds into w2/b2 and adds no deployed arrays. All seven representation bases share this capacity. No uncertainty, iterative self-correction or dynamic links are added by NR.

Use the unchanged `chromaseed_gaussian_numpy.Predictor`: one finite shape(36,) vector per call, Lab3 output. It validates every coordinate including finite ignored features. `chromaseed_gaussian.predict` is a separate N×36 batch helper. Inputs follow the existing prepared encoded-sRGB statistics contract, not arbitrary image RGB. The old TG model-card batch statement has a separate bound erratum.

```python
import sys
from pathlib import Path
import numpy as np

root = Path(r"C:\Users\dimal\Documents\просто\.worktrees\luma-local-search")
sys.path.insert(0, str(root / "scripts"))
from chromaseed_gaussian_numpy import Predictor

path = root / "experiments/runs/chromaseed_neural_readout_v1/selected/mixed/norm_random_raw36_s17_a2.npz"
with np.load(path, allow_pickle=False) as archive:
    predictor = Predictor({k: archive[k] for k in archive.files})
# lab = predictor(prepared_color36)
# batch_lab = np.array([predictor(row) for row in prepared_batch])
```

The example is an individual audited random-basis model, not a selected production release. Reported errors average individual seed errors, not ensemble inference. FP32-normalize then FP64 compute; cached arrays20816/3659 B, separate from process RAM. NPZ container size and training design/system/tensor costs are reported separately. Actual face/skin extraction, phone imagery, product-shade mapping and end-to-end latency remain unvalidated.

Original TRAIN966 rows/24 people only; three reused overlapping roles, camera/person confounding. Most NR comparisons and every frozen policy lose to FG. Analytic readout often improves the unchanged short-trained network, but that does not prove superiority over the strongest reference or patent novelty. Preserve all controls and source/input/receipt bindings.

[Report](../benchmarks/chromaseed_neural_readout_v1/report.md) · [Protocol](../research/chromaseed_neural_readout_v1_protocol.md) · [Receipt](../benchmarks/chromaseed_neural_readout_v1/verification.json) · [TG interface correction](../research/chromaseed_gaussian_consumer_erratum.md).
"""
    (ROOT / "docs/architecture/chromaseed_neural_readout_model_card.md").write_text(
        card, encoding="utf-8"
    )
    decision = """# After NR: inspect regularization before another bounded readout study

NR is verified progress, not full-goal completion. Same1855/10564 B network supports a real analytical output fit.54/72 comparisons improve their corresponding unchanged trained output, but79/84 lose to FG and all12 policies lose. Random features make construction much cheaper, with worse color error. Do not promote a universally better model or omit hidden-representation construction from timing.

All12 policies and77/84 per-basis choices selectalpha10, the upper tested boundary;7 select1. This motivates a specific next question about stronger regularization, not proof of overfitting or permission to expand grids indefinitely. Before implementation, inspect the standardized hidden-feature penalty, intercept exemption, effective design spectra/degrees of freedom and original inner alpha curves. Existing W/K/P/FG regularization studies are prior art on different representations; inventory them without rerunning exposed test roles.

If that inspection supports a follow-up, preregister a small extension including the exact old.1/1/10 controls and stronger100/1000 candidates on the same fixed504 NR bases, both losses/groups and all roles. Define inner selection and quality/cost reports before new outer evaluation; do not alter hidden networks or set role-specific thresholds from known losses. Independent augmented QR must verify outputs. Cached bases are legitimate for the diagnostic but full construction timing must still charge all representation learning. Preserve the negative results and parent receipts, including the one non-bitwise independent base reconstruction within original tolerance.

This is planned, not implemented or launched. Ordinary-phone facial/end-to-end quality remains unvalidated and cannot be established by repeatedly reusing these people. Only original TRAIN under the current boundary; no agents/new data/images/weights/packages/publication/exposed validation/test. The full goal remains active and not blocked while meaningful local learning work is available.

[NR report](../benchmarks/chromaseed_neural_readout_v1/report.md) · [Receipt](../benchmarks/chromaseed_neural_readout_v1/verification.json) · [Earlier frozen-palette decision](chromaseed_next_decision.md) · [TG interface erratum](chromaseed_gaussian_consumer_erratum.md).
"""
    (ROOT / "docs/research/chromaseed_neural_readout_next_decision.md").write_text(
        decision, encoding="utf-8"
    )
    reproduce = """# Reproducing and verifying NR

Use the original project `.venv/Scripts/python.exe` from this worktree. Set CUBLAS_WORKSPACE_CONFIG=:4096:8 and OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1. Only original data/processed/skin_mskcc_pixels_v1/train.npz. No installation, no concurrent heavy jobs. Never G/GS verifier mains; TG/FG/C/H/X/A reruns are read-only.

Execution record, not an instruction to overwrite sealed canonical artifacts:

1. `python scripts/chromaseed_gaussian_verify.py --cache <original-train.npz>` —fresh read-only parent, session59007 exit0.
2. `python -m pytest tests/test_chromaseed_neural_readout.py -q` —24 tests before primary,1.47 s. New test batch misuse corrected after checking actual TG consumer; erratum retained separately. Independent QR and representation reference tests pass.
3. `python scripts/chromaseed_neural_readout_train.py run --run experiments/runs/chromaseed_neural_readout_v1 --cache <original-train.npz>` —PID18004/session82004, exit0,61.634 s.
4. `python scripts/chromaseed_neural_readout_audit.py --run experiments/runs/chromaseed_neural_readout_v1 --output docs/benchmarks/chromaseed_neural_readout_v1 --cache <original-train.npz>` —initial79371 exit1 at an overly strict exact-count assertion after all numerical comparisons; corrected only the audit to record exactness under unchanged registered tolerance. Full repeat11895 exit0,141.416 s.503/504 bases bitwise, one w1 difference7.45e-9; all3024 QR refits pass.
5. `python scripts/chromaseed_neural_readout_runtime.py --run experiments/runs/chromaseed_neural_readout_v1 --output docs/benchmarks/chromaseed_neural_readout_v1 --cache <original-train.npz>` —session51100,378 complete fits including126 warmups.
6. `python scripts/chromaseed_neural_readout_report.py` —generates summary/five CSVs/two figures/model card/next decision before sealing.
7. `python scripts/chromaseed_neural_readout_verify.py --cache <original-train.npz>` —seals once, then reruns read-only without rewriting old receipts.

Future numerical replay requires separate run/output paths and separately named orchestration. Canonical report/verifier bind this series and must not be edited after sealing. Preserve the original primary/model/OOF/prediction archives and every control. Mutable plan/status ledgers are not immutable evidence.

[Protocol](../../research/chromaseed_neural_readout_v1_protocol.md) · [Audit](audit.json) · [Timing](runtime.json) · [Receipt](verification.json) · [TG erratum](../../research/chromaseed_gaussian_consumer_erratum.md).
"""
    (OUT / "reproduce.md").write_text(reproduce, encoding="utf-8")
    shortcut = f"""# Luma ChromaSeed NR: быстрый аналитический выход

Модель по случайной основе создаётся с нуля за **{r0["full_fit_ms"]:.3f} мс** при10,6 КБ либо **{r3["full_fit_ms"]:.3f} мс** при1,9 КБ; mixed-ошибка **{r0["person_mean"]:.3f}/{r3["person_mean"]:.3f} ΔE00**. Предыдущая FG raw36: **{fg["full_fit_ms"]:.3f} мс /5,439 ΔE00 /20,3 КБ**. Меньше ошибка — лучше. Это готовые признаки на одном CPU-потоке, не селфи целиком.

Аналитическая замена улучшает исходную сеть в54/72 сравнений, но79/84 новых вариантов и все12 правил выбора уступают FG. Для обученной основы полная стоимость включает её обучение. Получен проверенный компромисс, не универсальный прорыв.

144 независимых пересчёта основы,3024 независимых QR-решения и5,02 млн проверенных выходов;503/504 основы совпали побитно, оставшаяся в исходном допуске.378 полных timing-повторов проверены. Серия завершена; фоновых процессов не осталось. Общая цель активна; следующая проверка регуляризации только запланирована.

[Полный отчёт и графики]({(OUT / "report.md").as_posix()}) · [Receipt]({(OUT / "verification.json").as_posix()}) · [Модель]({(ROOT / "docs/architecture/chromaseed_neural_readout_model_card.md").as_posix()}).
"""
    SHORTCUT.write_text(shortcut, encoding="utf-8")
    print(
        dict(rows=len(rows), policies=len(policies), paired=len(paired), comparisons=counts),
        flush=True,
    )


if __name__ == "__main__":
    main()
