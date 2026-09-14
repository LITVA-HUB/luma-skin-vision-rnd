"""Report all frozen FG choices and costs without selecting on outer outcomes."""

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
RUN = ROOT / "experiments/runs/chromaseed_feature_groups_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_feature_groups_v1"
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-feature-groups-2026-09-13.md"
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
FAMILIES = ("norm_static", "norm_joint_soft", "perceptual_static", "perceptual_joint_soft")
GROUPS = (
    "raw36",
    "mean3",
    "median3",
    "central9",
    "mean_std6",
    "quant27",
    "no_corr33",
    "projected16",
)


def csv_write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    assert runtime["results_sha256"] == audit["results_sha256"] == sha(RUN / "results.json")
    rows, doses, policies = [], [], []
    for role in ROLES:
        for family in (*FAMILIES, "constant"):
            for group in ("constant",) if family == "constant" else GROUPS:
                rr = [
                    r
                    for r in result["records"]
                    if (r["role"], r["family"], r["group"]) == (role, family, group)
                ]
                tr = [
                    r
                    for r in runtime["records"]
                    if (r["role"], r["family"], r["group"]) == (role, family, group)
                ]
                fr = [
                    r
                    for r in runtime["standalone_fit_records"]
                    if (r["role"], r["family"], r["group"]) == (role, family, group)
                ]
                chosen = (
                    None
                    if family == "constant"
                    else selection["roles"][role][family]["groups"][group]["selected"]
                )
                row = dict(
                    role=role,
                    family=family,
                    group=group,
                    alpha=rr[0]["alpha"],
                    seeds=len(rr),
                    inner_clean=None if chosen is None else chosen["clean"],
                    inner_p90=None if chosen is None else chosen["p90"],
                    input_dimensions=rr[0]["input_dimensions"],
                    kernel_dimensions=rr[0]["kernel_dimensions"],
                    active_gate=rr[0]["active_gate"],
                    person_mean=float(np.mean([r["metrics"]["person_mean"] for r in rr])),
                    image_mean=float(np.mean([r["metrics"]["image_mean"] for r in rr])),
                    p90=float(np.mean([r["metrics"]["p90"] for r in rr])),
                    stress4=float(
                        np.mean([r["doses"][1]["worst_error"]["person_mean"] for r in rr])
                    ),
                    median_us=float(np.median([r["numpy_only"]["median_us"] for r in tr])),
                    maximum_seed_p95_us=max(r["numpy_only"]["p95_us"] for r in tr),
                    full_fit_ms=None if not fr else fr[0]["median_seconds"] * 1000,
                )
                for field in ("numeric_bytes", "archive_bytes", "actual_centers"):
                    row[field + "_min"] = min(r[field] for r in rr)
                    row[field + "_max"] = max(r[field] for r in rr)
                for op in (min, max):
                    row["cached_array_bytes_" + op.__name__] = op(
                        r["numpy_only"]["cached_array_bytes"] for r in tr
                    )
                rows.append(row)
                for i, dose in enumerate((1 / 255, 4 / 255, 16 / 255, 64 / 255)):
                    doses.append(
                        dict(
                            role=role,
                            family=family,
                            group=group,
                            dose=dose,
                            worst_person_error=float(
                                np.mean([r["doses"][i]["worst_error"]["person_mean"] for r in rr])
                            ),
                        )
                    )

    def get(role, group, family="perceptual_joint_soft"):
        return next(
            r for r in rows if (r["role"], r["family"], r["group"]) == (role, family, group)
        )

    for role in ROLES:
        for family in FAMILIES:
            for policy in ("quality", "compact"):
                s = selection["roles"][role][family]["policies"][policy]
                r, raw = get(role, s["group"], family), get(role, "raw36", family)
                policies.append(
                    dict(
                        role=role,
                        family=family,
                        policy=policy,
                        group=s["group"],
                        alpha=s["alpha"],
                        inner_clean=s["clean"],
                        inner_p90=s["p90"],
                        inner_numeric_bytes=s["numeric_bytes"],
                        person_mean=r["person_mean"],
                        p90=r["p90"],
                        stress4=r["stress4"],
                        difference_vs_raw=r["person_mean"] - raw["person_mean"],
                        numeric_bytes_min=r["numeric_bytes_min"],
                        numeric_bytes_max=r["numeric_bytes_max"],
                        median_us=r["median_us"],
                        full_fit_ms=r["full_fit_ms"],
                    )
                )
    new = [p for p in audit["paired"] if p["kind"] == "new_subset"]
    counts = dict(
        new_comparisons=len(new),
        new_adverse=sum(p["mean_difference"] > 0 for p in new),
        mixed_adverse=sum(p["role"] == "mixed" and p["mean_difference"] > 0 for p in new),
        transfer_adverse=sum(p["role"] != "mixed" and p["mean_difference"] > 0 for p in new),
        historical_X_comparisons=12,
        policy_adverse=sum(p["difference_vs_raw"] > 0 for p in policies),
        compact_policy_adverse=sum(
            p["policy"] == "compact" and p["difference_vs_raw"] > 0 for p in policies
        ),
    )
    paired = []
    for p in audit["paired"]:
        paired.append(
            {
                **{k: v for k, v in p.items() if k != "fixed_prediction_person_bootstrap_95"},
                "descriptive_95_low": p["fixed_prediction_person_bootstrap_95"][0],
                "descriptive_95_high": p["fixed_prediction_person_bootstrap_95"][1],
            }
        )
    assert (len(rows), len(doses), len(policies), len(paired)) == (99, 396, 24, 84)
    write_json(
        OUT / "summary.json",
        dict(
            source_lock_sha256=sha(RUN / "source_lock.json"),
            selection_sha256=sha(RUN / "selections.json"),
            results_sha256=sha(RUN / "results.json"),
            audit_sha256=sha(OUT / "audit.json"),
            runtime_sha256=sha(OUT / "runtime.json"),
            report_source_sha256=sha(Path(__file__)),
            rows=rows,
            doses=doses,
            policies=policies,
            paired=paired,
            comparison_counts=counts,
            final_query_rows=3834798,
            protocol_count_erratum="docs/research/chromaseed_feature_groups_count_erratum.md",
        ),
    )
    for file, table in (
        ("all_groups.csv", rows),
        ("all_doses.csv", doses),
        ("all_policies.csv", policies),
        ("paired.csv", paired),
    ):
        csv_write(OUT / file, table)

    labels = (
        "Все 36 признаков",
        "Средний RGB · 3",
        "Медианный RGB · 3",
        "Центр. квантили · 9",
        "Среднее + std · 6",
        "Квантили · 27",
        "Без корреляций · 33",
        "Проекция X · 16",
    )
    colors = (
        "#475569",
        "#059669",
        "#0d9488",
        "#0891b2",
        "#d97706",
        "#7c3aed",
        "#db2777",
        "#2563eb",
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for ax, role, title in zip(
        axes,
        ROLES,
        ("Mixed · 6 человек", "SLR → iPod · 16 человек", "iPod → SLR · 8 человек"),
        strict=True,
    ):
        values = [get(role, g)["person_mean"] for g in GROUPS]
        ax.scatter(values, range(8), c=colors, s=48)
        ax.axvline(values[0], color=colors[0], linewidth=1, linestyle="--")
        ax.set_yticks(range(8), labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(min(values) - 0.15, max(values) + 0.4)
        for i, value in enumerate(values):
            ax.annotate(
                f" {value:.3f}", (value, i), xytext=(5, -3), textcoords="offset points", fontsize=9
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Ошибка ΔE00 · меньше лучше")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Luma ChromaSeed FG · один метод, разные цветовые признаки\nТри запуска усреднены по ошибкам; повторно использованные люди, шкалы осей различаются",
        fontsize=12,
    )
    fig.savefig(OUT / "feature_groups_quality.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for ax, field, scale, title in zip(
        axes,
        ("numeric_bytes_max", "full_fit_ms", "median_us"),
        (1000, 1, 1),
        ("Числовые веса, КБ (1000 Б)", "Полное обучение, мс", "Ответ по признакам, мкс"),
        strict=True,
    ):
        values = [get("mixed", g)[field] / scale for g in GROUPS]
        ax.barh(range(8), values, color=colors, height=0.62)
        ax.set_yticks(range(8), labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.2)
        for i, value in enumerate(values):
            ax.text(value + max(values) * 0.02, i, f"{value:.2f}", va="center", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Mixed · perceptual_joint_soft · один поток CPU\nОбучение на 734 строках; время обработки фото и выделения кожи не измерено",
        fontsize=12,
    )
    fig.savefig(OUT / "feature_groups_cost.png", dpi=160)
    plt.close(fig)
    raw, mean, median, central, projected = (
        get("mixed", g) for g in ("raw36", "mean3", "median3", "central9", "projected16")
    )
    table = "\n".join(
        f"| {g} | "
        + " | ".join(f"{get(role, g)['person_mean']:.6f}" for role in ROLES)
        + f" | {get('mixed', g)['numeric_bytes_max']:,} | {get('mixed', g)['full_fit_ms']:.2f} | {get('mixed', g)['median_us']:.1f} |"
        for g in GROUPS
    )
    policy_table = "\n".join(
        f"| {p['role']} | {p['family']} | {p['policy']} | {p['group']} / {p['alpha']:g} | {p['inner_clean']:.4f} | {p['person_mean']:.4f} | {p['difference_vs_raw']:+.4f} |"
        for p in policies
    )
    interval_table = "\n".join(
        f"| {p['role']} | {p['group']} | {p['mean_difference']:+.4f} | [{p['descriptive_95_low']:+.4f}, {p['descriptive_95_high']:+.4f}] | {p['improved_people']}/{p['people']} |"
        for p in paired
        if p["family"] == "perceptual_joint_soft" and p["group"] in ("mean3", "median3", "central9")
    )
    report = f"""# Luma ChromaSeed FG: сколько цветовых признаков нужно компактной модели

Сокращение входа до трёх значений RGB уменьшило числовые веса и улучшило обе исследовательские проверки переноса между камерами. В смешанной выборке качество ухудшилось. Универсальную замену исходной модели эта серия не установила. Это развитие семейства **Luma ChromaSeed**, не подтверждённая точность косметического подбора по селфи.

Для `perceptual_joint_soft` средний RGB даёт **{mean["person_mean"]:.3f}** вместо **{raw["person_mean"]:.3f} ΔE00** в mixed, но **{get("slr_to_ipod", "mean3")["person_mean"]:.3f}** вместо **{get("slr_to_ipod", "raw36")["person_mean"]:.3f}** при SLR → iPod и **{get("ipod_to_slr", "mean3")["person_mean"]:.3f}** вместо **{get("ipod_to_slr", "raw36")["person_mean"]:.3f}** в обратную сторону. Ошибка меньше — лучше. Камера и состав людей здесь связаны, поэтому причинное объяснение «удалили след камеры» остаётся гипотезой.

## Сопоставимые результаты

Одна схема ядра с максимумом 128 центров, одинаковые три seed и сетка alpha. Таблица показывает все восемь представлений в семье `perceptual_joint_soft`. Среднее считается по ошибкам трёх отдельных моделей, не по предсказанию ансамбля. У каждой группы alpha выбран только во внутренних разделениях; [все 99 строк, включая четыре семьи и константы](all_groups.csv) сохраняют выбранные alpha, p90, размеры архивов и кэша. Mixed: 232 изображения / 6 людей; прямой перенос: 643 / 16; обратный: 323 / 8. Люди и роли повторно использованы и пересекаются между проверками.

| Вход | Mixed ΔE00 | SLR → iPod | iPod → SLR | Веса mixed, Б | Обучение mixed, мс | Ответ mixed, мкс |
|---|---:|---:|---:|---:|---:|---:|
{table}

![Сравнение качества](feature_groups_quality.png)

`projected16` — сохранённый контроль X: нужны все 36 исходных признаков, далее проекция. Здесь alpha для X выбирается по правилам FG среди трёх кандидатов; это не прежний выбор целой политики X. Поэтому часть итоговых значений отличается от старого отчёта X, хотя все 432 импортированных массива воспроизведены точно. При одной камере на обучении joint является точной static-копией. Повторяющиеся семейства и политики не являются независимыми подтверждениями.

Из 72 сравнений новых подмножеств с raw **{counts["new_adverse"]} хуже**, включая **{counts["mixed_adverse"]} из 24 mixed** и **{counts["transfer_adverse"]} из 48 переносов**. Это счёт сравнений, не оценка частоты ошибок на новых телефонах. [Все 84 парных сравнения, включая 12 исторических X](paired.csv).

## Выбор по внутренним данным не гарантирует внешнее качество

96 выборов alpha и 24 выбора группы заморожены до финального обучения. Quality минимизирует внутреннюю ошибку; compact допускает +0,05 к внутренней средней и +0,10 к p90 относительно raw, затем минимизирует размер. В mixed обе политики `perceptual_joint_soft` выбрали central9: **{central["person_mean"]:.3f}** на внешней роли против raw **{raw["person_mean"]:.3f}**. Внешние результаты не использованы для замены этих выборов на понравившиеся mean3 или X.

| Роль | Семья | Политика | Группа / alpha | Внутренняя ΔE00 | Внешняя ΔE00 | Разница с raw |
|---|---|---|---|---:|---:|---:|
{policy_table}

Хуже raw оказались {counts["policy_adverse"]} из 24 политик, в том числе {counts["compact_policy_adverse"]} из 12 compact. [Машиночитаемая таблица](all_policies.csv).

## Неопределённость и искажения

Ниже — разница ошибок отдельных людей: подмножество минус raw, сначала средняя по seed. Интервалы получены 20 000 повторными выборками фиксированных предсказаний. Они описательные: не учитывают многократное исследование этих людей, весь поиск моделей и множество сравнений; не являются независимым подтверждением или одновременными доверительными границами.

| Роль | Группа | Разница ΔE00 | Описательный диапазон 95% | Людей с улучшением |
|---|---|---:|---|---:|
{interval_table}

Все 291 модель проверены на identity и 32 заданных RGB-преобразованиях. Четыре силы 1/255, 4/255, 16/255, 64/255 и восемь цветовых якорей сохранены; [396 усреднённых по seed строк](all_doses.csv). Худшая ошибка берётся по преобразованиям для каждой строки перед агрегированием. Это синтетические возмущения готовых статистик, не измеренное изменение реального света и не качество новой камеры.

## Реальные затраты

У mixed mean3 **{mean["numeric_bytes_max"]:,} Б** числовых массивов против raw **{raw["numeric_bytes_max"]:,} Б**: уменьшение на **{100 * (1 - mean["numeric_bytes_max"] / raw["numeric_bytes_max"]):.2f}%**. Для однокамерного обучения mean3 хранит 3 127 Б: условная поправка отсутствует. Кэш mixed mean3 занимает {mean["cached_array_bytes_max"]:,} Б против {raw["cached_array_bytes_max"]:,} Б. Архивы NPZ, числовые массивы и кэш учитываются отдельно; размер весов не равен оперативной памяти процесса.

Полное обучение mean3 на 734 строках занимает **{mean["full_fit_ms"]:.2f} мс**, raw **{raw["full_fit_ms"]:.2f} мс**; median3 **{median["full_fit_ms"]:.2f} мс**. Ответ mean3 **{mean["median_us"]:.1f} мкс**, raw **{raw["median_us"]:.1f} мкс**: уменьшение числа признаков не ускорило этот NumPy-вызов. В индивидуальное обучение включены нормализация, веса примеров, точная ширина ядра, выбор центров, условная ветвь при наличии, целевая метрика и коэффициенты. Квадратичное хранение расстояний при подборе ширины остаётся.

![Размер и время](feature_groups_cost.png)

Замер: Windows, Ryzen 9 7900X, один поток CPU, NumPy {runtime["hardware"]["numpy"]}; GPU не использовался. 291 настоящий вызов потребителя проверен, затем 20 прогревов и три прохода на модель. Для 96 настроек seed17 выполнены четыре полных обучения: одно прогревочное и три измеряемых, всего 384, каждый набор массивов совпал точно. Время первичной серии — {js(RUN / "workflow.json")["wall_seconds"]:.3f} с без импорта процесса, аудита и профилирования. [Все образцы замеров](runtime.json). Декодирование фото, поиск лица, выделение кожи и извлечение статистик не измерены. Текущий API получает готовый color36 даже для подмножеств.

## Независимая проверка и воспроизведение

3 468 записей в 12 банках включают 1 008 точных копий joint/static для одной камеры, 432 импортированных X и 12 констант. Выполнены 2 016 решений коэффициентов, из них 1 728 для сокращённых входов; 336 разложений Gram, 252 подготовки базиса, 84 вычисления ширины, 28 условных функций и 12 целевых метрик, плюс 36 базовых и 36 theta вспомогательных решений. Все 432 raw-контроля точно совпали с A.

Независимый код восстановил геометрию, 491 300 внутренних выходов, 288 оценок кандидатов, все 96/24 выбора и все 3 834 798 финальных выходов. Проверены настоящие однострочные потребители и 288 полных refit через плотные расстояния, SVD и QR. Различий путей центров и эффективного ранга нет. Максимальное расхождение refit **{audit["maxima"]["qr_prediction"]:.8f} Lab** при заранее установленном допуске 0,001; реальный потребитель отличается максимум на {audit["maxima"]["consumer_prediction"]:.3g}. Это проверка численной реализации, не точность по прибору.

**Исправлена арифметика протокола:** исходные 5 441 700 ошибочно посчитаны по 1 700 обучающим строкам; правильный полный объём — `97 × (232 + 643 + 323) × 33 = 3 834 798`. Исходный протокол сохранён. Ни один предусмотренный человек, случай или преобразование не исключён. [Поправка](../../research/chromaseed_feature_groups_count_erratum.md), [аудит](audit.json), [итоговая проверка](verification.json), [воспроизведение](reproduce.md), [карточка модели](../../architecture/chromaseed_feature_groups_model_card.md).

Использован только прежний TRAIN: 966 строк / 24 человека, приборный native D65/10° Lab. Новые фото, веса, датасеты, старые validation/calibration/test не загружались. Обычные лица с телефона, косметический подбор и качество всей цепочки остаются непроверенными. Размер и скорость уже измерены; полная цель остаётся активной. [Следующее решение](../../research/chromaseed_feature_groups_next_decision.md).
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    card = f"""# Luma ChromaSeed FG — research model card

Purpose: estimate instrument-referenced native D65/10° skin Lab from prepared regional color statistics. This study changes the input feature subset of the existing analytic 128-landmark RBF model. It is not face identification, skin segmentation, a photo encoder or a cosmetic shade-match service.

Inputs: finite one-row float32 `color36`, encoded-sRGB quantiles (.01,.05,.1,.25,.5,.75,.9,.95,.99), quantile-major RGB at 0:27, mean 27:30, population std 30:33, correlations RG/RB/GB at 33:36. `mean3` gathers [27,28,29], `median3` [12,13,14], central9 [9:18], mean_std6 [27:33], quant27 [0:27], no_corr33 [0:33]. All 36 caller values must currently be finite. Discarded finite coordinates cannot influence subset predictions. X16 retains all 36 inputs and its original projection. Output is one three-component Lab estimate, without camera labels, query targets, other query rows, error confidence or calibration guarantees.

Training: fit-only population moments, balanced person/site/image weights, exact width, RPCholesky maximum 128 centers, alpha .1/1/10, analytic static/joint and normalized/perceptual readouts. Gate uses only selected fit features/camera labels; one-camera joint exports exact static schema. Twelve banks and three seeds. Ninety-six alpha/group settings and 24 policies fixed before final fits. Original TRAIN only, hash d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0. All 24 people and overlapping evaluation roles are reused exploratory data. Data provenance and permission remain those of the existing cache; no additional data/weight licence is inferred from this implementation.

Deployment: [NumPy consumer](../../scripts/chromaseed_feature_groups_numpy.py), with original projection/gated/kernel NumPy dependencies available from the same scripts directory. Example, run from the worktree with its existing environment:

```python
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path("scripts").resolve()))
from chromaseed_feature_groups_numpy import Predictor

# Demonstration of an evaluated candidate, not a production-selected model.
path = Path("experiments/runs/chromaseed_feature_groups_v1/selected/mixed/perceptual_joint_soft_mean3_s17.npz")
with np.load(path, allow_pickle=False) as archive:
    model = {{k: archive[k] for k in archive.files}}
predict = Predictor(model)
lab = predict(color36)  # Supply the measured region's real features.
```

Mixed mean3: {mean["numeric_bytes_max"]:,} numeric B, {mean["cached_array_bytes_max"]:,} cached-array B, {mean["median_us"]:.1f} us response, {mean["full_fit_ms"]:.2f} ms complete 734-row fit, one CPU thread. Static transfer model is 3,127 B. NPZ archive overhead, runtime dependencies, temporary arrays, Python/process memory and image preprocessing are separate. Reduced files store uint8 indices and FP32 parameters; cached center/readout arithmetic is FP64. No quantization or GPU acceleration claim.

Mixed mean3 error {mean["person_mean"]:.6f} versus raw {raw["person_mean"]:.6f} DeltaE00. The same input group improves both camera-transfer means, with overlapping/confounded people; this does not establish camera invariance. Frozen mixed joint quality/compact select central9, whose outer error is {central["person_mean"]:.6f}. X16 {projected["person_mean"]:.6f} is preserved historical control. No selector was changed after seeing outer errors. All outcomes, adverse results and descriptive intervals are in the [report](../benchmarks/chromaseed_feature_groups_v1/report.md).

Independent audit checks every deployed consumer under 33 transformations and refits all 288 nonconstant models. [Protocol](../research/chromaseed_feature_groups_v1_protocol.md), [count erratum](../research/chromaseed_feature_groups_count_erratum.md), [verification](../benchmarks/chromaseed_feature_groups_v1/verification.json). No ordinary-phone facial quality, subgroup fairness, identity recognition, clinical use, patent novelty or Skolkovo eligibility is established by this series.
"""
    (ROOT / "docs/architecture/chromaseed_feature_groups_model_card.md").write_text(
        card, encoding="utf-8"
    )
    next_decision = """# After FG: separate the training rule from feature content

FG is progress, not goal completion. All registered feature groups were fitted and independently reconstructed. Mean/median RGB produce much smaller models and improve both reused camera-transfer roles, but worsen mixed performance. Removing only correlations or selecting a group on inner folds does not establish universal improvement. Camera labels and person composition are confounded; do not call this causal removal of camera information. Preserve all four families, adverse cases and frozen policies. Existing archive RGB comparisons predate FG; the specific matched nonlinear ablation is the new experiment, not the idea of using RGB.

The next bounded learning question is TAGI-style Gaussian parameter inference versus ordinary gradient backpropagation on identical small networks. This follows the user's forum-method search, not a demonstrated Luma gain. [Checked sources and dataset registry](chromaseed_forum_methods_data_2026-09-13.md). TAGI uses reverse probabilistic inference; it is not Forward-Forward and not weight brute force. Forward-Forward still uses local gradients and needs a separate regression design.

Before any numerical launch, inventory local optimizers and model checkpoints without reopening old test roles; read the primary TAGI equations, specify an implementable regression contract and register a matched protocol. Use both fixed mean3 and raw36 inputs across every existing role, with the same fit-only normalization and one small architecture initially. Keep existing analytic FG models as references. Match data order, seed, parameter capacity and example budgets; report full time, processed examples and error curves, not just nominal steps. Include the storage/runtime cost of any variances during training; do not assume it is part of inference storage if deployment does not need it. No installation or unverified use of cuTAGI is implied. Implement finite/variance and linear-Gaussian reference checks before real fits. A native implementation of an approximate variant must be named honestly, with its differences from the paper documented.

Do not choose mean3 only for the favorable known outer roles, add repeated conditional mixtures to chase these scores, or treat a lower-dimensional color input as calibrated reflectance. A genuine improvement needs robust error/cost evidence; flattening outputs, matching training targets or merely saving bytes does not satisfy the user's full objective.

TAGI/BP is planned, not implemented or launched. New scene/phone face datasets remain a separate prerequisite for independent product-level evidence. UCI has binary skin labels; SCIN has estimated tone categories; neither replaces measured Lab. ENCoDE requires its access process, MST-E disallows ML training, and several forum-recommended releases have unverified licences/access. No new photos, weights, package installation, author contact or publication is authorized by this next-step note. Continue within the original TRAIN-only boundary and no delegation. The user goal remains active and is not blocked: meaningful local model work is available, while phone-face/end-to-end high quality remains unvalidated.

[Full FG result](../benchmarks/chromaseed_feature_groups_v1/report.md) · [Verification](../benchmarks/chromaseed_feature_groups_v1/verification.json).
"""
    (ROOT / "docs/research/chromaseed_feature_groups_next_decision.md").write_text(
        next_decision, encoding="utf-8"
    )
    reproduce = """# Reproducing and verifying FG

Use the original project's existing `.venv/Scripts/python.exe`, not the unused worktree venv. Run from this worktree. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. Only the original `data/processed/skin_mskcc_pixels_v1/train.npz` is allowed; its hash and the source/input chain are checked before numerical work. Never invoke G/GS verifier mains: their older receipts are frozen inputs.

The canonical FG run is sealed. The following is its execution record, not a request to overwrite it:

1. `python -m pytest tests/test_chromaseed_feature_groups.py -q` — 14 tests before the primary run.
2. `python scripts/chromaseed_crossfit_verify.py --cache <original-train.npz>` — existing C receipt checked read-only, unchanged.
3. `python scripts/chromaseed_feature_groups_train.py run --run experiments/runs/chromaseed_feature_groups_v1 --cache <original-train.npz>` — primary PID 39764 / session 44863, terminal exit 0.
4. `python scripts/chromaseed_feature_groups_audit.py --run experiments/runs/chromaseed_feature_groups_v1 --output docs/benchmarks/chromaseed_feature_groups_v1 --cache <original-train.npz>` — session 67225, terminal exit 0. The arithmetic count erratum was written before audit and included among its dependencies.
5. `python scripts/chromaseed_feature_groups_runtime.py --run experiments/runs/chromaseed_feature_groups_v1 --output docs/benchmarks/chromaseed_feature_groups_v1 --cache <original-train.npz>` — session 9942, terminal exit 0; no concurrent heavy job.
6. `python scripts/chromaseed_feature_groups_report.py` — builds summary, four CSVs, two figures, report and model contract before sealing.
7. `python scripts/chromaseed_feature_groups_verify.py --cache <original-train.npz>` — seals once; subsequent invocation checks its existing receipt read-only. It also verifies the C/H/X/A chain without rewriting it.

For a future full replay, use a new run and output directory. The train/audit/runtime tools accept `--run` and `--output` as applicable; replay metadata hashes will differ from the canonical records. The current report/verifier deliberately bind the canonical FG paths and must not be redirected by editing their sealed source. Retain this study as the reference and use separately named orchestration if a replay is justified.

[Registered protocol](../../research/chromaseed_feature_groups_v1_protocol.md) · [Arithmetic erratum](../../research/chromaseed_feature_groups_count_erratum.md) · [Audit](audit.json) · [Timing samples](runtime.json) · [Receipt](verification.json).
"""
    (OUT / "reproduce.md").write_text(reproduce, encoding="utf-8")
    shortcut = f"""# Luma ChromaSeed FG: модели от 3,1 КБ

Проверены восемь вариантов цветовых признаков. Модель по среднему RGB хранит **3,1 КБ** при обучении на одной камере и **4,7 КБ** с условной поправкой в mixed. Исходная — около **20–22 КБ**. В mixed полное обучение занимает **{mean["full_fit_ms"]:.1f} мс**, ответ по готовым признакам **{mean["median_us"]:.1f} мкс** на одном потоке CPU.

Качество неоднозначно: mixed **{mean["person_mean"]:.2f}** против **{raw["person_mean"]:.2f} ΔE00**, прямой перенос **{get("slr_to_ipod", "mean3")["person_mean"]:.2f}** против **{get("slr_to_ipod", "raw36")["person_mean"]:.2f}**, обратный **{get("ipod_to_slr", "mean3")["person_mean"]:.2f}** против **{get("ipod_to_slr", "raw36")["person_mean"]:.2f}**. Меньше — лучше. Модель уменьшилась, но универсального улучшения нет; скорость ответа почти такая же. Это повторно использованные данные участков кожи, не проверка косметического подбора по селфи.

3 468 записей, независимые 288 полных refit и 3,83 млн проверенных выходов; 384 повторных обучения для измерения стоимости. Исправлена арифметическая сумма в протоколе без изменения состава эксперимента. Серия завершена; её фоновых процессов не осталось. Общая цель активна. Следующая проверка TAGI против backprop пока запланирована.

[Все результаты и графики]({(OUT / "report.md").as_posix()}) · [Проверка]({(OUT / "verification.json").as_posix()}) · [Модель и использование]({(ROOT / "docs/architecture/chromaseed_feature_groups_model_card.md").as_posix()}) · [Методы и данные с форумов]({(ROOT / "docs/research/chromaseed_forum_methods_data_2026-09-13.md").as_posix()}).
"""
    SHORTCUT.write_text(shortcut, encoding="utf-8")
    print(
        dict(
            rows=len(rows),
            doses=len(doses),
            policies=len(policies),
            paired=len(paired),
            comparisons=counts,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
