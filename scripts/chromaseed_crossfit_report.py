"""C evidence tables and report; neither frozen H settings nor C outcomes are retuned."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from chromaseed_kernel_audit import js
from skin_local_search_train import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/runs/chromaseed_crossfit_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_crossfit_v1"
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-crossfit-2026-09-13.md"
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")


def csv_write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    result, audit, runtime = (
        js(p) for p in (RUN / "results.json", OUT / "audit.json", OUT / "runtime.json")
    )
    assert audit["passed"] and runtime["audit_sha256"] == sha(OUT / "audit.json")
    for o in (result, audit, runtime):
        assert o["source_lock_sha256"] == sha(RUN / "source_lock.json") and o[
            "settings_sha256"
        ] == sha(RUN / "frozen_settings.json")
    rows, doses, diagnostic = [], [], []
    for role in ROLES:
        for family in dict.fromkeys(r["family"] for r in result["records"] if r["role"] == role):
            rr = [r for r in result["records"] if r["role"] == role and r["family"] == family]
            tr = [r for r in runtime["records"] if r["role"] == role and r["family"] == family]
            fr = [
                r
                for r in runtime["full_fit_records"]
                if r["role"] == role and r["family"] == family
            ]
            r = rr[0]
            rows.append(
                dict(
                    role=role,
                    family=family,
                    origin=r["origin"],
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
                    full_fit_ms=None if not fr else fr[0]["median_seconds"] * 1000,
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
        for loss in ("norm", "perceptual"):
            for arm in ("full", "in_matched", "out_person"):
                dd = [
                    d
                    for d in audit["supervision_diagnostics"]
                    if d["role"] == role and d["loss"] == loss and d["arm"] == arm
                ]
                diagnostic.append(
                    dict(
                        role=role,
                        loss=loss,
                        arm=arm,
                        person_error=float(np.mean([d["metrics"]["person_mean"] for d in dd])),
                        native_residual_rms=float(np.mean([d["native_residual_rms"] for d in dd])),
                        full_backbone_discrepancy_rms=float(
                            np.mean([d["full_backbone_discrepancy_rms"] for d in dd])
                        ),
                        mean_signed_L=float(
                            np.mean([d["mean_signed_native_residual"][0] for d in dd])
                        ),
                        mean_signed_a=float(
                            np.mean([d["mean_signed_native_residual"][1] for d in dd])
                        ),
                        mean_signed_b=float(
                            np.mean([d["mean_signed_native_residual"][2] for d in dd])
                        ),
                    )
                )
    assert len(rows) == 63 and len(doses) == 252 and len(diagnostic) == 18
    to_raw = [p for p in audit["paired"] if p["control"].endswith("_raw")]
    to_h = [
        p
        for p in audit["paired"]
        if p["control"].startswith("h_") and not p["control"].endswith("_raw")
    ]
    matched = [p for p in audit["paired"] if p["control"].startswith("in_matched_")]
    counts = dict(
        new_vs_raw_adverse=sum(p["mean_difference"] > 0 for p in to_raw),
        transfer_vs_raw_adverse=sum(
            p["role"] != "mixed" and p["mean_difference"] > 0 for p in to_raw
        ),
        new_vs_H_adverse=sum(p["mean_difference"] > 0 for p in to_h),
        excluded_vs_included_adverse=sum(p["mean_difference"] > 0 for p in matched),
    )
    summary = dict(
        source_lock_sha256=sha(RUN / "source_lock.json"),
        settings_sha256=sha(RUN / "frozen_settings.json"),
        results_sha256=sha(RUN / "results.json"),
        audit_sha256=sha(OUT / "audit.json"),
        runtime_sha256=sha(OUT / "runtime.json"),
        report_source_sha256=sha(Path(__file__)),
        rows=rows,
        doses=doses,
        supervision=diagnostic,
        comparison_counts=counts,
    )
    write_json(OUT / "summary.json", summary)
    for name, table in (
        ("all_policies.csv", rows),
        ("all_doses.csv", doses),
        ("supervision.csv", diagnostic),
    ):
        csv_write(OUT / name, table)

    def get(role, family):
        return next(r for r in rows if r["role"] == role and r["family"] == family)

    families = (
        "h_perceptual_raw",
        "h_perceptual_uniform",
        "in_matched_perceptual_uniform",
        "out_person_perceptual_uniform",
        "h_perceptual_support",
        "in_matched_perceptual_support",
        "out_person_perceptual_support",
    )
    labels = (
        "Исходная без поправки",
        "Поправка H",
        "Поправка: знакомый человек",
        "Поправка: исключённый человек",
        "Взвешенная H",
        "Взвешенная: знакомый",
        "Взвешенная: исключённый",
    )
    colors = ("#475569", "#0891b2", "#059669", "#dc2626", "#8b5cf6", "#10b981", "#f97316")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), layout="constrained")
    for ax, role, title in zip(
        axes,
        ROLES,
        ("Mixed · 6 человек", "SLR → iPod · 16 человек", "iPod → SLR · 8 человек"),
        strict=True,
    ):
        values = [get(role, f)["person_mean"] for f in families]
        ax.scatter(values, range(len(families)), c=colors, s=55)
        ax.set_yticks(range(len(families)), labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(min(values) - 0.12, max(values) + 0.20)
        for i, v in enumerate(values):
            ax.annotate(
                f"{v:.3f}",
                (v, i),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
        ax.set_title(title)
        ax.set_xlabel("Средняя по людям ΔE00 ↓")
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle(
        "ChromaSeed C · одинаковые настройки H, разные источники ошибок для обучения поправки",
        fontsize=13,
    )
    fig.savefig(OUT / "crossfit_quality.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    selected = families[1:4]
    values = [get("mixed", f)["full_fit_ms"] for f in selected]
    bars = ax.barh(range(3), values, color=colors[1:4])
    ax.set_yticks(range(3), labels[1:4])
    ax.invert_yaxis()
    ax.bar_label(bars, fmt="%.1f мс", padding=4)
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel("Полное обучение на 734 строках, включая необходимых учителей")
    ax.set_title("Дополнительная цена обучения · CPU, один поток")
    fig.savefig(OUT / "crossfit_training_cost.png", dpi=160)
    plt.close(fig)
    h, included, excluded = (get("mixed", f) for f in families[1:4])
    table = [
        "| Perceptual-вариант | Mixed ΔE00 | SLR→iPod | iPod→SLR | Fit734, мс | Ответ mixed, мкс |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family, label in zip(families, labels, strict=True):
        a, b, c = (get(role, family) for role in ROLES)
        table.append(
            f"| {label} | {a['person_mean']:.6f} | {b['person_mean']:.6f} | {c['person_mean']:.6f} | {a['full_fit_ms']:.2f} | {a['median_us']:.1f} |"
        )
    pair = next(
        p for p in to_h if p["role"] == "mixed" and p["family"] == "in_matched_perceptual_uniform"
    )
    report = f"""# ChromaSeed C: обучение поправки на исключённых людях

**Рабочая гипотеза не дала устойчивого улучшения.** После проверки H зафиксировал его12 внутренних настроек и изменил только источник ошибок для обучения поправки. В mixed-случае обычная perceptual-поправка H {h["person_mean"]:.6f} ΔE00, контроль на знакомом учителю человеке {included["person_mean"]:.6f}, поправка по исключённому человеку {excluded["person_mean"]:.6f}. Последняя заметно хуже, хотя weighted-вариант немного улучшает mixed. Общего выигрыша по трём ролям нет.

Из24 новых сопоставлений с исходной моделью без поправки ухудшились{counts["new_vs_raw_adverse"]}; **все{counts["transfer_vs_raw_adverse"]}/16 переносов между камерами ухудшились**. Относительно соответствующей H-поправки хуже{counts["new_vs_H_adverse"]}/24; исключение человека хуже matched-контроля в{counts["excluded_vs_included_adverse"]}/12 случаях. Эти условия и все результаты сохраняются: новое имя или малый mixed-выигрыш не превращают проигравшую гипотезу в универсальную технологию.

## Контролируемое сравнение

{chr(10).join(table)}

![Качество при одинаковых настройках](crossfit_quality.png)

Каждая строка — средняя ошибка по людям, затем по отдельным seed17/29/43. Это не ансамбль. Все семьи/оба loss/все33 искажения приведены в [63 вариантах](all_policies.csv) и [252 dose-сводках](all_doses.csv); таблица выше выделяет perceptual-варианты для читаемости. Полные33 трансформации: identity и8 RGB-углов на дозах1/255,4/255,16/255,64/255. Целевой Lab считается неизменным как синтетическое предположение.

В matched mixed perceptual-поправке улучшение против H всего{h["person_mean"] - included["person_mean"]:.6f} ΔE00; улучшились{pair["people_improve"]}/{pair["people"]}, условный bootstrap по людям {pair["conditional_bootstrap_95"][0]:.5f}…{pair["conditional_bootstrap_95"][1]:.5f}. Это описание фиксированных предсказаний на шести многократно использованных людях, не независимое подтверждение; повторная подгонка/выбор архитектуры в диапазоне не учтены. [Все60 парных сравнений](audit.json).

## Как устроен эксперимент

В каждом из трёх role-fit наборов каждый человек по очереди целиком исключается из обучения raw A/H alpha0.1. Учитель использует только оставшиеся n−1 людей для нормализаторов, target-шкалы, ширины, опор, gate и readout. Три seeds и два loss дают252 учителя на42 поднаборах. Учителя предсказывают native Lab всех fit-строк. Out-person выбирает учителя, не видевшего данного человека; in-matched выбирает учителя, исключившего следующего человека той же камеры, но видевшего текущего. Обе схемы используют один пул учителей и одинаковое количество людей/людей каждой камеры. Конкретные люди и количества изображений отличаются; это не устраняет все различия состава данных.

Цели поправки: (native_target − native_teacher_prediction)/student_y_std. Нормализованные выходы разных учителей не смешиваются. Студенческий projected16/tau.5 базис использует те же128 исходных опорных строк и неизменённые настройки uniform/support H; raw-часть обучена на полном n-человеческом fit-наборе и совпадает с H точно. При ответе учителей нет — используется full-backbone и одна поправка, как в H. Сдвиг между учителем на n−1 людях и полной базой всё равно остаётся. [Ошибки и смещения трёх источников целей](supervision.csv) характеризуют обучение, а не качество поправки на новых людях.

Похожая проверка для прежней большой нейросети уже была и не подтвердила превосходство excluded-person. Её отрицательный результат учтён: [старый протокол](../../research/skin_crossfit_correction_protocol_v1.md), [старое решение](../../research/skin_crossfit_correction_next_decision.md). C отдельно проверяет компактную аналитическую модель и исключение одного человека. Метод не объявляется новым изобретением.

## Скорость, размер и проверка

![Полная стоимость обучения](crossfit_training_cost.png)

Для mixed uniform: H **{h["full_fit_ms"]:.2f}мс**, in-matched **{included["full_fit_ms"]:.2f}мс**, out-person **{excluded["full_fit_ms"]:.2f}мс** — увеличение примерно в{excluded["full_fit_ms"] / h["full_fit_ms"]:.1f} раза. Время включает всех18 необходимых учителей выбранных seed/loss, их native-таблицы, маршрутизацию и обучение/экспорт полной студенческой модели. Учителя обоих loss/трёх seeds из большого банка не скрыто повторно используются в измерении отдельного fit. 216 полных повторных обучений (96 C +120 H, с warmup) точно восстановили веса.

Числовые веса C mixed **27,503Б**, cached-массивы **73,088Б**, ответ примерно **{excluded["median_us"]:.1f}мкс** по готовому color36. Схема и число вычислений при ответе не изменены: отличия долей микросекунды не означают ускорение метода. Для однокамерных fit числовые веса24,278Б. Отдельно считаются NPZ, процесс Python, временная память и приватные teacher-архивы, не входящие в deployed-модель. Все36 признаков нужны; лицо/кожа/свет/телефон и подготовка изображения в этих временах не проверены. Точные ширины сохраняют квадратичные векторы расстояний. CPU,1 поток; результата на GPU не заявляется.

Основной workflow **{js(RUN / "workflow.json")["wall_seconds"]:.3f}с**, без imports/audit/runtime и без reuse C-банков. 342 основных решения readout,207 Gram-разложений,144 базиса,48 ширин,3 ковариации,19 gate-fit и135+135 вспомогательных решений. Хранятся183 deployment-записи (111 точных H,72 новых C) и252 teacher-записи; из111 H18 raw заново точно обучены,93 импортированы.

Независимый аудит за{audit["wall_seconds"]:.2f}с проверил границы всех42 исключений,252 QR/SVD учителя,72 QR/SVD поправки,156,504 native teacher-прогноза,20,400 routed-прогнозов,2,411,574 выхода фактического NumPy-потребителя и732 dose-сводки. Максимальный drift учителя{audit["maxima"]["teacher_qr"]:.3g} Lab, поправки{audit["maxima"]["head_qr"]:.3g}, потребителя{audit["maxima"]["final_portable"]:.3g}. Все111 H-веса и1,462,758 H-прогнозов точно сохранены. Девять основных и два независимых численных теста; окончательная команда/вывод записаны в [verification](verification.json).

[Протокол до fit](../../research/chromaseed_crossfit_v1_protocol.md) · [Замороженные H-настройки](../../../experiments/runs/chromaseed_crossfit_v1/frozen_settings.json) · [Runtime](runtime.json) · [JSON-сводка](summary.json) · [Воспроизведение](reproduce.md) · [Карточка и веса](../../architecture/chromaseed_crossfit_model_card.md).

## Решение

Не добавлять стоимость252-учительского исследования в продукт ради этого небольшого mixed-эффекта и не продвигать out-person как улучшенный режим. В этой конфигурации простой переход на ошибки незнакомых учителю людей не устранил перенос, а стоимость отдельного обучения выросла. Это согласуется с необходимостью проверить само цветовое представление, а не продолжать расширять сетку поправок. [Следующий шаг](../../research/chromaseed_crossfit_next_decision.md) пока не реализован и не запущен.

Только original TRAIN966 записей/24 человека; legacy validation/calibration/test не читались. Роли пересекаются и исторически использованы много раз. Обычные лицевые фото со смартфонов и end-to-end технология не валидированы. Общая цель компактной, быстрой и качественной модели остаётся активной; этот результат не доказывает готовность к косметическому рынку или достаточность для «Сколково». Семейство **Luma ChromaSeed**, C — исследовательская серия.
"""
    # Space prose numerals; code paths and mathematical identifiers remain literal.
    for a, b in (
        ("все{", "все {"),
        ("Все60", "Все 60"),
        ("Все111", "Все 111"),
        ("Все36", "Все 36"),
        ("Только original TRAIN966", "Только original TRAIN: 966"),
    ):
        report = report.replace(a, b)
    report = re.sub(r"([А-Яа-яЁё])(?=\d)", r"\1 ", report)
    report = re.sub(r"(\d)(?=[А-Яа-яЁё])", r"\1 ", report)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    reproduce = """# Reproduce C

Use the existing original-repository Python environment, not the unused worktree .venv or uv sync. PowerShell from the research worktree:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$cPython='C:\\Users\\dimal\\Documents\\просто\\luma-skin-vision-rnd\\.venv\\Scripts\\python.exe'
$cTrain='C:\\Users\\dimal\\Documents\\просто\\luma-skin-vision-rnd\\data\\processed\\skin_mskcc_pixels_v1\\train.npz'
& $cPython scripts/chromaseed_crossfit_verify.py --cache $cTrain
```

Existing C/H/X/A verification receipts are read-only. Never run the old G/GS verifier mains or overwrite bound files. For a new reproduction use new directories: crossfit_train.py `run --run NEW_RUN --cache TRAIN`; crossfit_audit.py and crossfit_runtime.py each `--run NEW_RUN --cache TRAIN --output NEW_OUTPUT`. Freeze inherited H settings before fitting all teachers and students. Compare numeric arrays, not timing or ZIP timestamps. The report/verify scripts address the archived study, not arbitrary output paths. Only original TRAIN is required. No new selector or legacy validation/calibration/test access.

[Report](report.md) · [Protocol](../../research/chromaseed_crossfit_v1_protocol.md) · [Verification](verification.json).
"""
    (OUT / "reproduce.md").write_text(reproduce, encoding="utf-8")
    next_text = """# C next decision: inspect and isolate input feature groups

C is verified progress, not broad-goal completion. Fixed-H person-excluded residual targets fail to produce a robust gain. They reuse the same252 teachers as a matched inclusion control, and independent native decoding/routing/QR fits validate the result. In-matched gives only a tiny mixed benefit while teacher-inclusive training costs about14 times H's single fit. The same general mechanism already failed to win in the older large-network experiment. Preserve both failures; do not expand cross-fitting folds, mixing strengths or teacher grids merely to obtain a favorable held score.

The next useful question is in the input representation. Inventory existing direct-RGB/quantile baselines first to avoid repeating an old experiment. Then register a compact feature-group ablation: explicit central RGB statistics versus distribution shape/correlations, with original raw36 and projected16 controls, matched kernels/readouts, complete size/response/full-fit accounting and inner-only selection. A small subset is a hypothesis, not a known camera-invariant representation. Do not infer real-phone accuracy from camera separability or a lower internal score. Fixed encoded-sRGB statistics cannot be assumed to supply calibrated reflectance; any physical color conversion must state and test its assumptions.

This new feature-group study is planned, not implemented or launched. Register the entire grid and source/data boundary before new fits. Preserve all mixed/transfer/stress negatives. There is still a concrete learning question, so the full goal is active, not blocked; compatibility tests and small reused-data gains do not establish compact/fast/high-quality deployment.

Only original TRAIN remains authorized; no new data, agents, publication or uploads. The24 historically reused people cannot establish ordinary-phone facial quality through more searches. That requirement ultimately needs independent instrument-referenced facial acquisition and end-to-end extraction validation. Preserve C/H/X/A read-only receipts and every old source/input binding; never call G/GS verifier mains.

[C report](../benchmarks/chromaseed_crossfit_v1/report.md) · [Verification](../benchmarks/chromaseed_crossfit_v1/verification.json) · [Active goal](chromaseed_active_goal.md).
"""
    (ROOT / "docs/research/chromaseed_crossfit_next_decision.md").write_text(
        next_text, encoding="utf-8"
    )
    card = f"""# Luma ChromaSeed C research model card

Predicts supplied native Lab from prepared skin-region color36. Same H shared128-center raw/projected16 geometry and NumPy deployment schema. Only residual supervision changes: teachers excluding one person, with matched included-person control. No teacher pool, person/camera label or batch adaptation is used at deployment. Neither an identity/ethnicity model nor a diagnosis or proven phone-face estimator.

Mixed examples: [in-matched perceptual uniform seed17](../../experiments/runs/chromaseed_crossfit_v1/selected/mixed/in_matched_perceptual_uniform_s17.npz), [out-person counterpart](../../experiments/runs/chromaseed_crossfit_v1/selected/mixed/out_person_perceptual_uniform_s17.npz), [exact H control](../../experiments/runs/chromaseed_crossfit_v1/selected/mixed/h_perceptual_uniform_s17.npz). These are individual seeds, not ensembles. Numeric weights27,503B, cached arrays73,088B, about23us prepared-feature answer; out-person full fit734rows about{excluded["full_fit_ms"]:.1f}ms including18 teachers for that loss/seed, H{h["full_fit_ms"]:.1f}ms. Transfer remains worse than raw A; no deployment recommendation.

Use the unchanged [H NumPy consumer](../../scripts/chromaseed_hybrid_numpy.py) and its [X NumPy dependency](../../scripts/chromaseed_projection_numpy.py). Load NPZ with allow_pickle=False, instantiate Predictor(model_arrays), pass one finite shape(36,) vector. The exact input quantile/mean/std/correlation contract, FP32 parameters, FP64 arithmetic and latent FP32 rounding remain H's. Teacher archives and private routing receipts are research evidence, not inference dependencies. [H schema card](chromaseed_hybrid_model_card.md).

Source, data and weight licenses remain separate and inherited; this experiment clears no new commercial rights or name/trademark. Only original TRAIN, no image/identifier publication, no legacy validation/calibration/test. Reused24-person evidence is not independent phone-face quality or Skolkovo eligibility. [Protocol](../research/chromaseed_crossfit_v1_protocol.md) · [Report](../benchmarks/chromaseed_crossfit_v1/report.md) · [Verification](../benchmarks/chromaseed_crossfit_v1/verification.json).
"""
    (ROOT / "docs/architecture/chromaseed_crossfit_model_card.md").write_text(
        card, encoding="utf-8"
    )
    shortcut = f"""# Luma ChromaSeed C: проверка обучения на исключённых людях

Обучены252 учителя и72 цветовые поправки при одинаковых настройках H. Учитель либо не видел данного человека, либо видел его, но не видел другого человека той же камеры. Оба режима используют один пул учителей; скрытого дополнительного банка при ответе нет.

Mixed perceptual uniform: прежняя H **{h["person_mean"]:.3f} ΔE00**, знакомый учителю человек **{included["person_mean"]:.3f}**, исключённый **{excluded["person_mean"]:.3f}**. Гипотеза не дала общего улучшения; все16 сравнений переноса с исходной моделью без поправки хуже. Небольшие gains на шести ранее использованных людях не доказывают качество на обычных лицах.

Числовые веса **27,5 КБ**, ответ около **23 мкс** по готовым признакам. Полное обучение с учителями **{excluded["full_fit_ms"]:.0f} мс** вместо H **{h["full_fit_ms"]:.0f} мс**, примерно в{excluded["full_fit_ms"] / h["full_fit_ms"]:.1f} раза дороже. Независимый пересчёт всех учителей/поправок,2,41млн выходов,11 численных тестов и216 повторных обучений прошли.

[Результаты и графики]({(OUT / "report.md").as_posix()}) · [Проверка]({(OUT / "verification.json").as_posix()}) · [Веса и использование]({(ROOT / "docs/architecture/chromaseed_crossfit_model_card.md").as_posix()}) · [Следующее решение]({(ROOT / "docs/research/chromaseed_crossfit_next_decision.md").as_posix()}).

Серия завершена, фоновых обучений C не осталось. Следующая проверка групп цветовых признаков пока только запланирована. Общая цель остаётся активной. Семейство **Luma ChromaSeed**.
"""
    shortcut = shortcut.replace("Небольшие gains", "Небольшие улучшения")
    shortcut = re.sub(r"([А-Яа-яЁё])(?=\d)", r"\1 ", shortcut)
    shortcut = re.sub(r"(\d)(?=[А-Яа-яЁё])", r"\1 ", shortcut)
    SHORTCUT.write_text(shortcut, encoding="utf-8")
    print(
        dict(rows=len(rows), doses=len(doses), supervision=len(diagnostic), comparisons=counts),
        flush=True,
    )


if __name__ == "__main__":
    main()
