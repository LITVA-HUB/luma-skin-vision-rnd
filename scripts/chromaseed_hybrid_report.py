"""Source-bound H results, all policies/doses, plots, model card and next decision."""

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
RUN = ROOT / "experiments/runs/chromaseed_hybrid_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_hybrid_v1"
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-hybrid-2026-09-13.md"
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
LABELS = dict(
    raw="Исходная",
    projected="Сжатая, общие опоры",
    blend="Смешивание",
    uniform="Поправка",
    support="Взвешенная поправка",
    x_fixed="Сжатая X",
)


def csv_write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
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
    for obj in (result, selection, audit, runtime):
        assert obj["source_lock_sha256"] == sha(RUN / "source_lock.json")
    for obj in (audit, runtime):
        assert obj["results_sha256"] == sha(RUN / "results.json") and obj[
            "selection_sha256"
        ] == sha(RUN / "selections.json")
    rows, doses, inner = [], [], []
    for role in ROLES:
        families = list(dict.fromkeys(r["family"] for r in result["records"] if r["role"] == role))
        for family in families:
            rr = [r for r in result["records"] if r["role"] == role and r["family"] == family]
            tr = [r for r in runtime["records"] if r["role"] == role and r["family"] == family]
            fits = [
                r
                for r in runtime["standalone_fit_records"]
                if r["role"] == role and r["family"] == family
            ]
            r = rr[0]
            rows.append(
                dict(
                    role=role,
                    family=family,
                    kind=r["kind"],
                    alpha=r.get("alpha"),
                    rho=r.get("rho"),
                    power=r.get("power"),
                    person_mean=float(np.mean([v["metrics"]["person_mean"] for v in rr])),
                    image_mean=float(np.mean([v["metrics"]["image_mean"] for v in rr])),
                    p90=float(np.mean([v["metrics"]["p90"] for v in rr])),
                    stress4=float(
                        np.mean([v["doses"][1]["worst_error"]["person_mean"] for v in rr])
                    ),
                    numeric_bytes=r["numeric_bytes"],
                    archive_bytes=r["archive_bytes"],
                    cached_array_bytes=tr[0]["numpy_only"]["cached_array_bytes"],
                    median_us=float(np.median([v["numpy_only"]["median_us"] for v in tr])),
                    maximum_seed_p95_us=max(v["numpy_only"]["p95_us"] for v in tr),
                    fit_ms=None if not fits else fits[0]["median_seconds"] * 1000,
                    seeds=len(rr),
                )
            )
            for i, dose in enumerate((1 / 255, 4 / 255, 16 / 255, 64 / 255)):
                doses.append(
                    dict(
                        role=role,
                        family=family,
                        dose=dose,
                        worst_person_error=float(
                            np.mean([v["doses"][i]["worst_error"]["person_mean"] for v in rr])
                        ),
                    )
                )
        for loss, kinds in selection["roles"][role].items():
            for kind, entry in kinds.items():
                for c in entry["candidates"]:
                    inner.append(
                        dict(
                            role=role,
                            loss=loss,
                            family=kind,
                            **{k: v for k, v in c.items() if k != "seed_scores"},
                            selected=c == entry["selected"],
                        )
                    )
    assert len(rows) == 39 and len(doses) == 156 and len(inner) == 258
    adverse = sum(p["mean_difference"] > 0 for p in audit["paired"])
    transfer_adverse = sum(
        p["role"] != "mixed" and p["mean_difference"] > 0 for p in audit["paired"]
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
        inner_candidates=inner,
        adverse_outer_comparisons=adverse,
        adverse_transfer_comparisons=transfer_adverse,
    )
    write_json(OUT / "summary.json", summary)
    csv_write(OUT / "all_policies.csv", rows)
    csv_write(OUT / "all_doses.csv", doses)
    csv_write(OUT / "all_inner_candidates.csv", inner)

    def get(role, kind):
        family = "x_fixed_perceptual" if kind == "x_fixed" else "perceptual_" + kind
        return next(r for r in rows if r["role"] == role and r["family"] == family)

    kinds = ("raw", "projected", "blend", "uniform", "support", "x_fixed")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), layout="constrained")
    colors = ["#475569", "#8b5cf6", "#0891b2", "#059669", "#ea580c", "#a78bfa"]
    for ax, role, title in zip(
        axes,
        ROLES,
        ("Смешанная: 6 человек", "SLR → iPod: 16 человек", "iPod → SLR: 8 человек"),
        strict=True,
    ):
        values = [get(role, k)["person_mean"] for k in kinds]
        ax.scatter(values, range(6), c=colors, s=60)
        ax.set_yticks(range(6), [LABELS[k] for k in kinds], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(min(values) - 0.22, max(values) + 0.22)
        for position, value in enumerate(values):
            ax.annotate(
                f"{value:.3f}",
                (value, position),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Средняя по людям ΔE00 ↓")
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle(
        "ChromaSeed H · сохранение исходных признаков смягчает, но не устраняет потери переноса",
        fontsize=13,
    )
    fig.savefig(OUT / "hybrid_quality.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for ax, field, title, unit in (
        (axes[0], "numeric_bytes", "Числовые веса", "КБ (1000 Б)"),
        (axes[1], "median_us", "Ответ по готовым color36", "мкс; CPU, 1 поток"),
    ):
        values = [
            get("mixed", k)[field] / (1000 if field == "numeric_bytes" else 1) for k in kinds[:5]
        ]
        bars = ax.barh(range(5), values, color=colors[:5])
        ax.set_yticks(range(5), [LABELS[k] for k in kinds[:5]], fontsize=9)
        ax.invert_yaxis()
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_xlim(0, max(values) * 1.2)
        ax.set_title(title)
        ax.set_xlabel(unit)
    fig.suptitle("Цена дополнительной ветки · смешанная роль, perceptual", fontsize=13)
    fig.savefig(OUT / "hybrid_cost.png", dpi=160)
    plt.close(fig)
    base, corr, weighted = (get("mixed", k) for k in ("raw", "uniform", "support"))
    pair = next(
        p for p in audit["paired"] if p["role"] == "mixed" and p["family"] == "perceptual_uniform"
    )
    table = [
        "| Вариант (perceptual) | Mixed ΔE00 | SLR→iPod | iPod→SLR | Веса mixed, Б | Ответ mixed, мкс | Fit734, мс |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k in kinds:
        a, b, c = (get(role, k) for role in ROLES)
        fit = "—" if a["fit_ms"] is None else f"{a['fit_ms']:.2f}"
        table.append(
            f"| {LABELS[k]} | {a['person_mean']:.6f} | {b['person_mean']:.6f} | {c['person_mean']:.6f} | {a['numeric_bytes']:,} | {a['median_us']:.1f} | {fit} |"
        )
    report = f"""# ChromaSeed H: общие опоры и цветовая поправка

**Прогресс эксперимента; универсального улучшения нет.** Сохранил исходное цветовое представление и добавил обучаемую ветку в 16 координатах. Веса обоих ядер используют одни 128 исходных опорных строк. Сравнил обычное смешивание, обучение поправки и её плавное ослабление по среднему сходству с опорами. Каждая схема могла выбрать нулевую поправку по внутренним данным; все 18 селекторов смешивания/поправок выбрали положительную силу. Это не гарантия качества на внешнем сдвиге.

В смешанной роли обычная perceptual-поправка: **{base["person_mean"]:.6f} → {corr["person_mean"]:.6f} ΔE00**. Размер **{base["numeric_bytes"]:,} → {corr["numeric_bytes"]:,} Б** (+{(corr["numeric_bytes"] / base["numeric_bytes"] - 1) * 100:.2f}%), ответ **{base["median_us"]:.1f} → {corr["median_us"]:.1f} мкс**, полное обучение734 строк **{base["fit_ms"]:.2f} → {corr["fit_ms"]:.2f} мс**. Это одна последовательная серия измерений CPU; дополнительная ветка оказалась медленнее. Две ветки не названы динамической перестройкой связей или новым методом обратного распространения ошибки.

Все 8 mixed-сопоставлений численно улучшились, но **{adverse}/24** сопоставления с исходной моделью ухудшились — **все {transfer_adverse}/16 переносов между камерами**. Для perceptual-поправки прямой перенос {get("slr_to_ipod", "raw")["person_mean"]:.6f} → {get("slr_to_ipod", "uniform")["person_mean"]:.6f}, обратный {get("ipod_to_slr", "raw")["person_mean"]:.6f} → {get("ipod_to_slr", "uniform")["person_mean"]:.6f}. Сглаженное ослабление даёт mixed{weighted["person_mean"]:.6f}, но также не побеждает исходную модель на переносах. Новую ветку универсальной заменой не объявляю.

## Качество и стоимость

{chr(10).join(table)}

![Все три роли](hybrid_quality.png)

![Размер и ответ](hybrid_cost.png)

Таблица показывает отдельные потери, усреднённые по трём seed 17/29/43; это не ансамбль и не три независимых набора людей. Mixed имеет 6 ранее использованных отложенных людей, переносы 16 и 8; роли пересекаются. Смена камеры связана также со сменой людей и распределения. Диапазон парного bootstrap по людям (выборка с возвращением) для mixed perceptual-поправки: {pair["conditional_bootstrap_95"][0]:.4f}…{pair["conditional_bootstrap_95"][1]:.4f} ΔE00, улучшились {pair["people_improve"]}/{pair["people"]}. Он включает ноль; это описательный bootstrap фиксированных предсказаний после многократной исследовательской работы, без учёта повторной подгонки/выбора архитектуры. Некоторые normalized-диапазоны не включают ноль, но также не являются независимым доказательством. Все 24 диапазона доступны в [аудите](audit.json).

## Что именно обучено

Исходный A joint-soft alpha0.1/eta0 воспроизведён точно. У projected-only H и у обеих веток гибрида совпадают исходные landmark-индексы; нормализаторы, проекция, ширины, сглаженный gate и perceptual-метрика обучаются только на fit-части. H projected-only отличается от X, где сжатое ядро само выбирало свои опоры; фиксированный X d16/tau0.5/alpha0.1 сохранён отдельно без переобучения.

В staged-поправках решается регуляризованная задача на остатке от экспортированной FP32 исходной модели. Support h(x)=mean(Kraw)^p с p1/4 включён и в обучающий дизайн, и в ответ. Все 6 support-выборов предпочли p1; p4 сохранён в полной сетке и принудительных проверках. Это гладкая мера сходства с ядром, без порога и без калибровки уверенности. При одной обучающей камере gate отсутствует, но projected-поправка остаётся активной. Входной camera-ID, правильный Lab и статистика соседних запросов не используются.

Гибрид хранит raw-центры один раз и выводит projected-центры при загрузке. У активного mixed варианта числовые веса27,503Б, а кешированные массивы NumPy{corr["cached_array_bytes"]:,}Б; NPZ-контейнер, процесс Python, временные массивы и вызывающий код считаются отдельно. Два ядра и проекция реально вычисляются при ответе; экономия хранения не означает ускорение. Полный fit projected-only также вычисляет исходный readout — это явно включено в его время. Подготовка36 цветовых статистик, поиск лица/кожи и работа на телефоне не измерялись. Точное вычисление ширины всё ещё хранит квадратичный вектор расстояний.

## Объём и воспроизводимость

До реального fit зафиксированы [протокол](../../research/chromaseed_hybrid_v1_protocol.md), исходники и входные SHA. Девять внутренних банков →30 решений →три финальных банка →111 моделей на33 искажениях. Время основного workflow{js(RUN / "workflow.json")["wall_seconds"]:.3f}с, без импорта/audit/runtime; повторного использования H банков не было.

- 2,964 записи:2,880 H,72 импортированных X,12 констант. Внутри H есть72 повторно точно обученных A-контроля и216 rho1 endpoint-алиасов; они не выдаются за независимые новые модели. Сила поправки повторно использует уже найденные веса.
- 936 решений readout,288 Gram-разложений,36 raw+36 projected базисов,24 точные ширины,12 ковариаций,4 gate-fit и36+36 вспомогательных решений исходного построителя.
- 258 записей внутренних кандидатов,30 замороженных выборов,419,900 OOF-предсказаний,1,462,758 финальных предсказаний и444 dose-сводки проверены независимо. Все 33 преобразования законны для цветовых статистик; неизменность целевого Lab при них — синтетическое предположение.
- Независимые weighted-SVD/exhaustive-distance/dense-pivot/QR расчёты:90 выбранных реконструкций и18 принудительно положительных. Максимальное расхождение{audit["maxima"]["qr_prediction"]:.3g} Lab, одноточечный NumPy-потребитель{audit["maxima"]["standalone"]:.3g}; ширины совпали. Аудит{audit["wall_seconds"]:.2f}с.
- Все 111 потребителей измерены реально. 144 полных timing-fit (120 выбранных и24 положительных, включая warmup) точно повторили числовые веса. 15 основных и3 независимых численных теста; окончательная команда и её результат записаны в [verification](verification.json).

[Полная таблица39 вариантов](all_policies.csv) · [Все156 dose-строк](all_doses.csv) · [Все258 внутренних кандидатов](all_inner_candidates.csv) · [JSON-сводка](summary.json) · [Runtime](runtime.json) · [Воспроизведение](reproduce.md).

## Решение и границы

Этот результат поддерживает сохранение исходных признаков: гибрид смягчает провал projected-only, но baseline по-прежнему лучше в переносах. Ни дополнительная ветка, ни её сходство с fit-примерами не решили исходную проблему. Семейство остаётся **Luma ChromaSeed**; H — номер исследовательской схемы, без заявления о патентной новизне или свободе товарного знака.

Использован только original TRAIN: 966 записей/24 человека. Legacy validation/calibration/test не читались. Обычные лица со смартфонов, точность выделения кожи, свет, подбор косметики и end-to-end интеграция не валидированы; это не доказательство готовности технологии или достаточности для «Сколково». Широкая цель активна. [Карточка модели](../../architecture/chromaseed_hybrid_model_card.md) · [Следующее решение](../../research/chromaseed_hybrid_next_decision.md).
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    reproduce = """# Reproduce H locally

Use the existing original-repository Python3.12 environment, NumPy2.5.3/SciPy1.18.1; do not run uv sync or alter older frozen dependency files. Work from the isolated research worktree. PowerShell, one CPU thread:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$hPython='C:\\Users\\dimal\\Documents\\просто\\luma-skin-vision-rnd\\.venv\\Scripts\\python.exe'
$hTrain='C:\\Users\\dimal\\Documents\\просто\\luma-skin-vision-rnd\\data\\processed\\skin_mskcc_pixels_v1\\train.npz'
& $hPython scripts/chromaseed_hybrid_verify.py --cache $hTrain
```

That verifier is read-only when its receipt exists and recursively uses the read-only X/A paths. Never run the old G/GS verifier mains; they rewrite now-bound receipts. The audit/runtime/report commands below target a separate output directory if repeated. Do not overwrite bound archived reports or numeric run files.

For a new reproduction run use the frozen scripts, a new --run directory and --output directory: train script `run --run NEW_RUN --cache TRAIN`, audit script `--run NEW_RUN --cache TRAIN --output NEW_OUTPUT`, then runtime with those same arguments. Freeze9 inner banks before selection and3 final banks; the runner enforces this order. Compare numeric payloads and predictions, not timing bytes or ZIP timestamps. No legacy held-out dataset is needed. The report/verify scripts address the original recorded study and are integrity checks, not general new-run report generators.

[Protocol](../../research/chromaseed_hybrid_v1_protocol.md) · [Report](report.md) · [Verification](verification.json).
"""
    (OUT / "reproduce.md").write_text(reproduce, encoding="utf-8")
    next_text = """# H next decision: cross-fitted residual targets, not a larger projection grid

H is verified progress, not broad-goal completion. Retaining the raw backbone improves the tradeoff relative to projected-only, but all16 transfer comparisons still worsen. Mean-kernel support does not solve the shift and p4 is never selected. Keep exact A and X baselines; do not promote H, retrospectively suppress its branch by outer role, or reinterpret the small reused mixed gain as independent evidence.

A concrete next learning question remains: H learns residuals of the full-fit backbone on its own training rows. Test whether **person-held-out backbone residual targets** produce a more transferable correction, with exact in-sample H, zero correction and ordinary blend controls. This is a planned experiment, not implemented or launched. Derive teacher predictions by an additional nested person-disjoint split entirely within each fitting bank; all target normalization/geometry must respect those boundaries. Subtract teacher predictions in native Lab before converting to the student's original-fit normalized coordinates. Quantify the mismatch between reduced-data teachers and the final full-data backbone. No teacher may use the current bank's query labels, people or camera state.

Before fitting, register the complete teacher/correction grid, matched capacity/budget, original versus cross-fitted residual targets, output strength including zero, all inner decisions, camera-role scores and synthetic stress outcomes. Count extra teacher fits and runtime explicitly. Use the original36-feature/128-center basis and one fixed projected representation so a wider architectural search does not confound the residual-target question. Independently refit positive paths and test person separation. It may fail by exaggerating full-model residuals; record that result too.

The24 historically reused people cannot establish ordinary-phone facial accuracy through further optimization. That part ultimately needs independent instrument-referenced facial acquisition and end-to-end extraction validation; no new-data acquisition, uploads, publication or delegation is authorized here. Only original TRAIN; preserve all old locks, reports and exposed validation/calibration/test exclusions. The broader compact/fast/high-quality goal remains active, not blocked at this point.

[H report](../benchmarks/chromaseed_hybrid_v1/report.md) · [H verification](../benchmarks/chromaseed_hybrid_v1/verification.json) · [Active goal](chromaseed_active_goal.md).
"""
    (ROOT / "docs/research/chromaseed_hybrid_next_decision.md").write_text(
        next_text, encoding="utf-8"
    )
    card = f"""# Luma ChromaSeed H research model card

Purpose: predict the supplied native Lab reference from prepared encoded-sRGB color36 skin-region statistics. Not identity recognition, ethnicity inference, diagnosis, ordinary-phone facial validation or a finished cosmetics matcher. H uses raw128 original landmarks and a16-dimensional shrinkage projection with the same row indices; staged ridge correction, optional continuous mean-kernel support, no iterative inference.

Example mixed perceptual uniform selected model: alpha1/rho1,27,503 numericB,73,088 cached arrays,mean{corr["person_mean"]:.6f} ΔE00 on six reused held people;{corr["median_us"]:.1f}us prepared-feature CPU response,{corr["fit_ms"]:.2f}ms full734-row fit. Both transfer roles worsen vs raw A. No universal deployment recommendation. Distinguish numeric bytes from archive size and process memory.

Weights: [mixed example seed17](../../experiments/runs/chromaseed_hybrid_v1/selected/mixed/perceptual_uniform_s17.npz), [raw A control](../../experiments/runs/chromaseed_hybrid_v1/selected/mixed/perceptual_raw_s17.npz), [projected-only matched control](../../experiments/runs/chromaseed_hybrid_v1/selected/mixed/perceptual_projected_s17.npz). These are individual seed models, not an ensemble. See all families/seeds/roles in the source-bound [results](../../experiments/runs/chromaseed_hybrid_v1/results.json).

Consumer: [H NumPy](../../scripts/chromaseed_hybrid_numpy.py) imports only NumPy and the existing [X NumPy consumer](../../scripts/chromaseed_projection_numpy.py). Load NPZ with allow_pickle=False, construct `Predictor(dict_of_arrays)`, pass one finite float-compatible shape(36,) vector. Ordinary Python imports/working-directory setup still apply. Input27 quantile channels+RGB mean3+std3+RG/RB/GB correlations3 must follow the existing extraction contract; numeric shape alone does not ensure a valid skin crop. Query camera labels/target Lab and batch adaptation are absent.

Hybrid fields extend raw7/11-field payload with latent_projection(36,16),latent_mean(36),latent_width,latent_coefficient(128,3),optional latent_correction(128,3),hybrid_mode uint8,mix FP32,support_power uint8. FP32 normalizers/centers/readouts/projection, FP64 arithmetic with specified latent FP32 rounding. Centers are derived and cached at load. The control endpoints canonicalize rho0 to raw and blend rho1 to projected-only; remaining hybrid modes compute both kernels. Support mean(Kraw)^p is not calibrated uncertainty.

Source, data, weights and commercial terms remain separate. This study does not alter inherited data/license restrictions or clear commercial rights; no participant images/identifiers are exported. Model-family naming is provisional, not a trademark search. [Protocol](../research/chromaseed_hybrid_v1_protocol.md) · [Report](../benchmarks/chromaseed_hybrid_v1/report.md) · [Verification](../benchmarks/chromaseed_hybrid_v1/verification.json).
"""
    (ROOT / "docs/architecture/chromaseed_hybrid_model_card.md").write_text(card, encoding="utf-8")
    shortcut = f"""# Luma ChromaSeed: результат нового эксперимента H

Сохранил исходное представление цвета и добавил сжатую обучаемую поправку с общими 128 опорными примерами. Сравнил её с обычным смешиванием и плавным ослаблением по сходству со знакомыми цветами.

На смешанной выборке ошибка perceptual-поправки **{base["person_mean"]:.3f} → {corr["person_mean"]:.3f} ΔE00**. Числовые веса **27,503Б**, ответ **{corr["median_us"]:.1f}мкс**, обучение734 строк **{corr["fit_ms"]:.1f}мс** на одном потоке CPU. Проверены2,964 записи конфигураций,18 численных тестов, независимый пересчёт и144 повторных полных обучения.

При этом **все 16 сопоставлений переноса между камерами ухудшились**. Дополнительная ветка увеличивает размер и почти удваивает время ответа; ослабление поправки проблему переноса не решило. Это экспериментальное продвижение, без подтверждения универсального улучшения. На обычных лицевых фото качество ещё не проверено; модель не объявляется готовой технологией для заявки в «Сколково».

[Подробные результаты и графики]({(OUT / "report.md").as_posix()}) · [Проверка]({(OUT / "verification.json").as_posix()}) · [Веса и использование]({(ROOT / "docs/architecture/chromaseed_hybrid_model_card.md").as_posix()}) · [Следующий эксперимент]({(ROOT / "docs/research/chromaseed_hybrid_next_decision.md").as_posix()}).

Все задачи этой серии завершены; фонового обучения не осталось. Следующий эксперимент с остатками от прогнозов на исключённых из обучения людях пока только запланирован. Общая цель исследования остаётся активной. Семейство: **Luma ChromaSeed**.
"""
    SHORTCUT.parent.mkdir(parents=True, exist_ok=True)
    SHORTCUT.write_text(shortcut, encoding="utf-8")
    print(
        "H report:39 policy rows,156 doses,258 inner candidates;16/24 adverse,all16 transfer",
        flush=True,
    )


if __name__ == "__main__":
    main()
