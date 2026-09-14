"""Seal WA selected means, matched controls, timing and preserved adverse evidence."""

from __future__ import annotations

import re
import subprocess
import sys

import numpy as np
from chromaseed_kernel_audit import js, nz
from chromaseed_long_training_report import csv_out
from chromaseed_long_training_run import NP
from chromaseed_refine_audit import error_summary
from chromaseed_weight_average_run import OUT, ROOT, RUN, check_map, load_data
from skin_local_search_train import sha, write_json

CARD = ROOT / "docs/architecture/chromaseed_weight_average_model_card.md"
NEXT = ROOT / "docs/research/chromaseed_weight_average_next_decision.md"
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-weight-average-2026-09-13.md"
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
SCRIPTS = (
    "chromaseed_weight_average",
    "chromaseed_weight_average_run",
    "chromaseed_weight_average_audit",
    "chromaseed_weight_average_runtime",
    "chromaseed_weight_average_report",
)


def paired(result):
    data = load_data()
    records = []
    for role in ROLES:
        chosen = [r for r in result["records"] if r["role"] == role and "overall" in r["policies"]]
        for control in ("LT_last", "NP_baseline"):
            diffs = []
            for r in chosen:
                reference = next(
                    c
                    for c in result["records"]
                    if c["role"] == role
                    and c["seed"] == r["seed"]
                    and (("last" in c["policies"]) if control == "LT_last" else c["baseline"])
                )
                a, b = [
                    nz(RUN / "evaluated" / role / f"{name}.npz")
                    for name in (r["name"], reference["name"])
                ]
                np.testing.assert_array_equal(a["row_indices"], b["row_indices"])
                ix = a["row_indices"]
                errors = [
                    error_summary(
                        p["predictions"][0],
                        data["target"][ix],
                        data["patient"][ix],
                        data["site"][ix],
                    )[1]
                    for p in (a, b)
                ]
                diffs.append(errors[0] - errors[1])
            v = np.mean(diffs, axis=0)
            rng = np.random.default_rng(1237101)
            draws = v[rng.integers(len(v), size=(10000, len(v)))].mean(1)
            records.append(
                dict(
                    role=role,
                    reference=control,
                    people=len(v),
                    mean_difference=float(v.mean()),
                    descriptive_ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
                    bootstrap_draws=10000,
                    note="Post-hoc fixed-prediction paired person bootstrap, seed errors averaged, no selection uncertainty or independent confirmation.",
                )
            )
    return records


def main():
    lock, selection, result = [
        js(RUN / n) for n in ("source_lock.json", "selections.json", "results.json")
    ]
    audit, runtime = js(OUT / "audit.json"), js(OUT / "runtime.json")
    source, selected, results = [
        sha(RUN / n) for n in ("source_lock.json", "selections.json", "results.json")
    ]
    check_map(
        {
            **lock["sources"],
            **lock["input_sha256"],
            **audit["artifact_sha256"],
            **audit["dependencies"],
            **runtime["dependencies"],
        }
    )
    assert (
        audit["passed"] and runtime["passed"] and runtime["audit_sha256"] == sha(OUT / "audit.json")
    )
    for obj in (selection, result, audit, runtime):
        assert obj["source_lock_sha256"] == source
    for obj in (result, audit, runtime):
        assert obj["selection_sha256"] == selected
    for obj in (audit, runtime):
        assert obj["results_sha256"] == results
    assert (
        runtime["upstream_fits"],
        runtime["continuation_banks"],
        runtime["exact_parent_checkpoint_payloads"],
        runtime["exact_final_payloads"],
    ) == (27, 9, 810, 189)
    verification = OUT / "verification.json"
    if verification.exists():
        receipt = js(verification)
        assert receipt["passed"] and sha(SHORTCUT) == receipt["shortcut_sha256"]
        check_map({**receipt["artifact_sha256"], **receipt["postprocess_sources"]})
        print(dict(passed=True, mode="read-only", verification_sha256=sha(verification)))
        return
    rows, policies, comparisons = [], [], []
    for role in ROLES:
        names = sorted({r["recipe_name"] for r in result["records"] if r["role"] == role})
        for name in names:
            rr = [r for r in result["records"] if r["role"] == role and r["recipe_name"] == name]
            tt = [r for r in runtime["records"] if r["role"] == role and r["recipe_name"] == name]
            assert len(rr) == len(tt) == 3
            row = dict(
                role=role,
                recipe=name,
                method=rr[0]["method"],
                variants=rr[0]["variants"],
                lr=rr[0]["lr"],
                steps=rr[0]["steps"],
                policies=rr[0]["policies"],
                endpoint_for=rr[0]["endpoint_for"],
                baseline=rr[0]["baseline"],
                person_mean=float(np.mean([r["metrics"]["person_mean"] for r in rr])),
                p90=float(np.mean([r["metrics"]["p90"] for r in rr])),
                stress_largest_dose_worst_error=float(
                    np.mean([r["doses"][-1]["worst_error"]["person_mean"] for r in rr])
                ),
                parameters=643,
                numeric_bytes=2886,
                cached_array_bytes=5456,
                archive_bytes_min=min(r["archive_bytes"] for r in rr),
                archive_bytes_max=max(r["archive_bytes"] for r in rr),
                cpu_median_us=float(np.median([r["median_us"] for r in tt])),
                worst_seed_p95_us=max(r["p95_us"] for r in tt),
                weight_vs_prediction_mean_max_lab=max(
                    r["weight_vs_prediction_mean_max_lab"] for r in rr
                ),
            )
            rows.append(row)
        for policy, c in selection["roles"][role]["policies"].items():
            row = next(r for r in rows if r["role"] == role and r["recipe"] == c["name"])
            policies.append(dict(policy=policy, inner_person_mean=c["clean"], **row))
            if policy in ("pair", "prefix", "tail3"):
                end = next(r for r in rows if r["role"] == role and policy in r["endpoint_for"])
                comparisons.append(
                    dict(
                        role=role,
                        method=policy,
                        recipe=row["recipe"],
                        endpoint_recipe=end["recipe"],
                        average_error=row["person_mean"],
                        endpoint_error=end["person_mean"],
                        difference=row["person_mean"] - end["person_mean"],
                    )
                )
    assert len(rows) == 21 and len(policies) == 15 and len(comparisons) == 9
    fg_records = js(NP / "results.json")["records"]
    fg = {
        role: float(
            np.mean(
                [
                    r["metrics"]["person_mean"]
                    for r in fg_records
                    if r["role"] == role and r["family"] == "fg_norm_static"
                ]
            )
        )
        for role in ROLES
    }
    ci = paired(result)
    outcomes = dict(
        better=sum(c["difference"] < -1e-8 for c in comparisons),
        worse=sum(c["difference"] > 1e-8 for c in comparisons),
        same=sum(abs(c["difference"]) <= 1e-8 for c in comparisons),
    )
    summary = dict(
        rows=rows,
        policies=policies,
        endpoint_comparisons=comparisons,
        endpoint_outcomes=outcomes,
        paired=ci,
        fg_context=fg,
        source_lock_sha256=source,
        selection_sha256=selected,
        results_sha256=results,
        sources=len(lock["sources"]),
        inputs=len(lock["input_sha256"]),
        goal_status="active",
        previous_turn_classification="progress",
        primary_new_gradient_fits=0,
        primary_seconds=js(RUN / "progress.json")["seconds"],
        coverage=audit["counts"],
    )
    write_json(OUT / "summary.json", summary)
    csv_out(OUT / "all_models.csv", rows)
    csv_out(OUT / "policies.csv", policies)
    csv_out(OUT / "endpoint_comparisons.csv", comparisons)
    csv_out(OUT / "paired.csv", ci)
    csv_out(
        OUT / "inner_candidates.csv",
        [dict(role=role, **c) for role, e in selection["roles"].items() for c in e["candidates"]],
    )

    def values(policy):
        return [
            next(p["person_mean"] for p in policies if p["role"] == role and p["policy"] == policy)
            for role in ROLES
        ]

    baseline = [
        next(r["person_mean"] for r in rows if r["role"] == role and r["baseline"])
        for role in ROLES
    ]
    table = ["| Вариант | Mixed | SLR → iPod | iPod → SLR |", "|---|---:|---:|---:|"]
    for label, vals in [
        ("Исходная NP", baseline),
        ("LT: внутренне выбранный одиночный этап", values("last")),
        ("Среднее соседних этапов", values("pair")),
        ("Среднее начальной последовательности этапов", values("prefix")),
        ("Среднее трёх последних этапов", values("tail3")),
        ("WA: общий внутренний выбор", values("overall")),
        ("Прежняя FG,20 284 числовых байта", list(fg.values())),
    ]:
        table.append(f"| {label} | " + " | ".join(f"{v:.5f}" for v in vals) + " |")
    detail = [
        "| Сценарий | Политика | Этапы | Вариации / lr | Внутренняя ΔE00 | Внешняя ΔE00 |",
        "|---|---|---|---|---:|---:|",
    ]
    for p in policies:
        detail.append(
            f"| {p['role']} | {p['policy']} | {','.join(str(s) for s in p['steps'])} | {p['variants']} / {p['lr']} | {p['inner_person_mean']:.5f} | {p['person_mean']:.5f} |"
        )
    times = " / ".join(f"{r['full_construction_seconds']:.3f}" for r in runtime["recipes"])
    cpu = float(np.median([r["median_us"] for r in runtime["records"]]))
    lines = [
        "# Luma ChromaSeed-WA: объединение весов этапов обучения",
        "",
        "Усреднение сохранило643 параметра и прежний однопроходный потребитель, но устойчивого повышения качества не дало. "
        "Общий внутренний выбор оставил LT в mixed, ухудшил прямой перенос и немного улучшил обратный относительно LT. "
        "Оба новых выбранных переноса остаются хуже исходной NP. Предыдущие веса и отрицательные результаты сохранены.",
        "",
        *table,
        "",
        "Меньше ΔE00 — лучше. Это средняя ошибка цвета по людям и затем по ошибкам трёх отдельных seed-моделей. "
        "Не процент точности и не ансамбль прогнозов. Для каждого способа настройки выбраны только на внутренних person-folds. "
        "Общий выбор и все15 политик зафиксированы до новых финальных средних; внешние ошибки не меняли выбор.",
        "",
        *detail,
        "",
        f"Против соответствующего последнего этапа той же траектории усреднение улучшает {outcomes['better']}/9 сравнений и ухудшает {outcomes['worse']}/9. "
        "Этот контроль важен: улучшение относительно короткого LT не означает пользу усреднения, если более длинный одиночный этап уже лучше. "
        "Все девять пар сохранены в endpoint_comparisons.csv. Описательные интервалы для выбранного overall находятся в paired.csv; "
        "они не учитывают неопределённость выбора и не являются независимым подтверждением.",
        "",
        "Проверены91 заранее заданный рецепт на сценарий: один исходный baseline,30 положительных одиночных этапов и60 средних. "
        "Соседние пары включают0+512; prefix использует все положительные сохранённые этапы; tail3 — три последних к заданному сроку. "
        "Веса w0,b0,v0,c0 складываются в FP64 и один раз округляются в FP32. "
        "Нормализаторы и идентичность головы сохраняются. Объединяются только веса одной исходной головы и seed при одном режиме/ставке обучения. "
        "Слои разных seed не смешиваются и нейроны не переставляются. Усреднение весов и ответов нелинейной сети различается; "
        "в тесте есть явный контрпример, численное различие для реальных моделей сохранено в CSV.",
        "",
        "Идея усреднения весов известна по [SWA](https://arxiv.org/abs/1803.05407). Здесь проверяются сохранённые AdamW/cosine-траектории LT; "
        "результаты авторов SWA на Luma не переносились. Более ранние планы EMA-учителя и опыты распределения примеров были проверены перед этой серией.",
        "",
        f"Размер643 параметра/2 886 числовых байт/5 456 байт рабочих массивов. CPU один поток: медиана{cpu:.1f} мкс на готовый color36, "
        "включая нормализацию. Веса FP32, дальнейшие вычисления потребителя FP64. Декодирование фото, выделение кожи и признаки в эти числа времени не входят. "
        "NPZ-служебные байты отдельно. Никакого добавленного повторного прохода или динамической сети в WA нет.",
        "",
        f"Первичный WA-процесс занял{summary['primary_seconds']:.2f} с и переиспользовал сохранённые LT веса; новых градиентных обучений в нём нет. "
        "Чистое объединение готовых весов занимает около0,15 мс на модель, но это не полная цена её получения. "
        f"Полное воспроизведение всех выбранных вариантов и их контролей по сценариям:{times} с. "
        "Для каждого сценария заново обучаются три исходных ND модели, экспортируются NP, выполняется весь18-слотовый LT-банк до32768 "
        "с исходным горизонтом131072 и сохраняются все необходимые этапы. Такой объём требуется объединению всех четырёх политик и их контролей; "
        "это не время отдельного overall-предиктора. Включены нормализация, пулы, выборки, CUDA Graph, проверки родителей и экспорт средних. "
        "Банк не делится на18 для заявления о времени одной модели. Прогретый процесс/GPU, без чтения файлов и извлечения признаков; "
        "первый из трёх повторов исключён только из времени. Исторический LT-процесс до131072 остаётся266,44с, включая подбор и сохранение.",
        "",
        "27 исходных singleton-обучений/NP экспортов воспроизведены точно. Девять полных LT-банков заново получили810 проверенных checkpoint-пакетов "
        "с точными массивами;189 финальных экспортов выбранных моделей и контролей тоже совпали побитово. "
        f"Независимый аудит проверил2457 внутренних средних/копий,464 100 OOF-векторов,273 оценки,15 решений,63 финальные модели "
        f"и все789 525 одиночных ответов при33 преобразованиях. Максимальное расхождение{audit['maximum_native_lab_drift']:.3g} native Lab, допуск2e-8. "
        "Все36 одиночных финальных контролей точно повторили сохранённые LT прогнозы. Десять поведенческих тестов и линтер прошли.",
        "",
        "Только исходный TRAIN966 строк/24 человека. В mixed fit734/18 и оценка232/6; переносы323/8 ↔643/16. "
        "Роли перекрываются и многократно использовались, люди связаны с камерой. Материал — подготовленные участки кожи с инструментальным Lab D65/10°. "
        "Высокая точность обычных селфи и подбора косметики ещё не проверена; широкая цель остаётся активной.",
        "",
        "[Все модели](all_models.csv) · [Политики](policies.csv) · [Внутренние кандидаты](inner_candidates.csv) · "
        "[Сравнение с последним этапом](endpoint_comparisons.csv) · [Парные интервалы](paired.csv) · "
        "[Аудит](audit.json) · [Замеры](runtime.json) · [Сводка](summary.json) · [Проверка](verification.json).",
        "",
        "[Протокол](../../research/chromaseed_weight_average_v1_protocol.md) · [Карточка](../../architecture/chromaseed_weight_average_model_card.md) · "
        "[Следующее решение](../../research/chromaseed_weight_average_next_decision.md).",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    CARD.write_text(
        """# Luma ChromaSeed-WA model card

Same-trajectory equal weight means of frozen LT checkpoints, deployed as the unchanged NP643-parameter36->16ReLU->3 head. Input one finite prepared color36 vector, output native instrument D65/10-degree Lab3. Predictor in scripts/chromaseed_neural_prefix_numpy.py takes(36,); predict batches Nx36. All normalizers and original head metadata preserved; no new gradient training in primary exports. No cross-seed mixing or query-dependent inference loop.

2,886numericB,643parameters,5,456cached-arrayB,about6.4us prepared-feature single-thread CPU. FP32 payload/normalization, FP64 remaining consumer arithmetic. Image/face/skin/feature extraction and archive/runtime/code overhead are additional. Equal weight mean differs from a mean of predictions; no ensemble deployed.

91recipes/role,273 inner candidates,15 frozen policies. Selected overall errors5.716113/8.381401/8.640083 versus LT5.716113/8.294346/8.683471 and NP5.770540/8.294346/8.589532. Both selected averaged transfer outcomes remain worse than initial NP. No universal quality promotion. Original TRAIN966rows/24people only; repeated/confounded exploratory roles, no ordinary-phone facial accuracy claim.

63 final payloads include all selected methods, matching endpoint controls and baseline, not63 independently chosen deployment winners. Follow policies.csv and full source/selection binding; preserve adverse outcomes. Ten tests,2457inner scalar mean checks/464100OOF vectors,63final models/789525actual consumer calls audited.810 parent checkpoint payloads and189 final exports reconstructed bitwise after27 original singleton fits and9 complete LT banks. Pure averaging~0.15ms; all-method/control three-seed construction~6.028/5.856/5.980s with original upstream and fixed18-slot costs, not one-head training time. No new data, assets, packages or external publication. Upstream terms continue to apply.

[Evidence](../benchmarks/chromaseed_weight_average_v1/report.md) · [Protocol](../research/chromaseed_weight_average_v1_protocol.md).
""",
        encoding="utf-8",
    )
    NEXT.write_text(
        """# After WA: stop averaging sweep, revisit compact local information

WA is verified progress through a negative overall result. Same643-parameter/2886B model and~6.4us response retained.273 inner candidates,15 policies,63 final selected/control exports. Overall preserves mixed LT, worsens forward8.29435->8.38140 and improves reverse8.68347->8.64008; both averaged transfers remain worse than NP. Do not select an outer-favorable average or continue enlarging this weight-mean grid. Scalar audit and complete fixed-shape reconstructions pass; no generalization or end-to-end goal completion follows.

Next priority is additional LOCAL information at a small parameter budget, with matched controls. A read-only header inventory of the existing original TRAIN archive confirmed color966x36,tokens966x64x18,hist966x512,color_hist966x548 and other fields. Only archive headers were inspected in this follow-up, no new pixel/feature/target arrays or data were acquired. A corrected rg search then read existing skin_mskcc_pixel_protocol_v1.md: color/histogram ridge and MLP controls already exist. skin_copula_protocol_v1.md already defines approximately1.07M-parameter histogram/copula/patch contexts, with different historical roles. chromaseed_refine_v1_protocol.md already defines patch_mlp and recurrent/dynamic compact patch encoders. Do not describe patch pooling, histogram context, dynamic connections or recurrence as first-use ideas; never open their old validation/calibration/test arrays.

Before implementing, inspect the actual R patch architecture/results and feature definitions, compare its parameter/latency budget with the accepted NP643 head, and preregister a genuinely matched SMALL branch experiment on current original-TRAIN roles. A concrete candidate is shared18->8 local encoding of64 cached tokens, pooled mean/std/max24 dimensions, coupled to color36 through a16-unit head. Estimated architecture count:152 local encoder +1027 head =1179 trainable values, plus measured normalizer/metadata/storage costs. This is an unimplemented estimate, not an acquired or measured model. It may retain local color/gradient distribution information that color36 removes; this is a hypothesis, not evidence of improved camera robustness.

Keep a matched color-only continuation and unchanged NP baseline, equal seed/sample/schedule budgets, fit-only token normalizers and no camera identifier input. Inventory warm-start compatibility; preserve the original NP mapping at step0 when adding the branch and verify it rather than assume floating-point equivalence across shapes. Include actual token encoding/pooling in inference timings and all original NP construction in training costs. Register inner choices before final evaluation; no outer-driven patch width selection. Existing R already tested recurrence/dynamic masks and longer schedules, so use that evidence to avoid repeating its large architecture under a new name. The proposed branch study is planned, not implemented or launched.

Full high-quality ordinary-phone facial and cosmetics objective remains active. Current original-TRAIN-only/no-acquisition/agents/publication scope persists. Preserve all frozen WA/LT/NP/ND/NB evidence. WA report verifier is read-only after sealing; never rerun their bound writers or G/GS mains.

[WA report](../benchmarks/chromaseed_weight_average_v1/report.md) · [R protocol](chromaseed_refine_v1_protocol.md) · [Pixel feature inventory](skin_mskcc_pixel_protocol_v1.md) · [Prior copula context](skin_copula_protocol_v1.md).
""",
        encoding="utf-8",
    )
    test = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_chromaseed_weight_average.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert test.returncode == 0, test.stdout + test.stderr
    lint = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            *(f"scripts/{s}.py" for s in SCRIPTS),
            "tests/test_chromaseed_weight_average.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert lint.returncode == 0, lint.stdout + lint.stderr
    artifacts = dict(audit["artifact_sha256"])
    for p in (*OUT.glob("*"), CARD, NEXT):
        if p.is_file() and p != verification:
            artifacts[p.relative_to(ROOT).as_posix()] = sha(p)
    for p in (OUT / "report.md", CARD, NEXT):
        for link in re.findall(r"\]\(([^)]+)\)", p.read_text(encoding="utf-8")):
            if link.startswith(("https://", "http://")):
                continue
            target = (p.parent / link.split("#")[0]).resolve()
            assert target.is_file() or target == verification, link
    SHORTCUT.parent.mkdir(parents=True, exist_ok=True)
    SHORTCUT.write_text(
        f"# Luma ChromaSeed-WA\n\n[Отчёт]({(OUT / 'report.md').as_posix()})\n\nУсреднение весов:643параметра/2,9КБ,устойчивого выигрыша нет. Выбрано5,71611/8,38140/8,64008; обычные селфи не проверены.\n",
        encoding="utf-8",
    )
    write_json(
        verification,
        dict(
            passed=True,
            source_lock_sha256=source,
            selection_sha256=selected,
            results_sha256=results,
            artifact_sha256=artifacts,
            postprocess_sources={f"scripts/{s}.py": sha(ROOT / f"scripts/{s}.py") for s in SCRIPTS},
            shortcut_sha256=sha(SHORTCUT),
            tests=test.stdout,
            lint=lint.stdout,
            coverage=audit["counts"],
            parent_checkpoint_replays=810,
            final_replays=189,
            goal_status="active",
        ),
    )
    print(
        dict(
            passed=True,
            verification_sha256=sha(verification),
            tests=test.stdout.strip(),
            overall=values("overall"),
            endpoint_outcomes=outcomes,
        )
    )


if __name__ == "__main__":
    main()
