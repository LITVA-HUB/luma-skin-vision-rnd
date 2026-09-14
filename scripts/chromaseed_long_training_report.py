"""Seal LT curves, honest selected outcomes, model card and reconstruction receipts."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys

import numpy as np
from chromaseed_kernel_audit import js, nz
from chromaseed_long_training_run import NP, OUT, ROOT, RUN, check_map, load_data
from chromaseed_refine_audit import error_summary
from skin_local_search_train import sha, write_json

CARD = ROOT / "docs/architecture/chromaseed_long_training_model_card.md"
NEXT = ROOT / "docs/research/chromaseed_long_training_next_decision.md"
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-long-training-2026-09-13.md"
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
SCRIPTS = (
    "chromaseed_long_training_fit",
    "chromaseed_long_training_run",
    "chromaseed_long_training_audit",
    "chromaseed_long_training_runtime",
    "chromaseed_long_training_report",
)


def csv_out(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                for k, v in row.items()
            }
            for row in rows
        )


def pairs(data, result):
    out = []
    for role in ROLES:
        chosen = [r for r in result["records"] if r["role"] == role and r["selected_overall"]]
        for reference in ("NP_baseline", "FG"):
            differences = []
            for rec in chosen:
                saved = nz(RUN / "evaluated" / role / f"{rec['name']}.npz")
                rows, pred = saved["row_indices"], saved["predictions"][0]
                if reference == "NP_baseline":
                    old = nz(RUN / "evaluated" / role / f"v0_r0_t0_s{rec['seed']}.npz")
                else:
                    old = nz(NP / "evaluated" / role / f"fg_norm_static_s{rec['seed']}.npz")
                np.testing.assert_array_equal(rows, old["row_indices"])
                _, actual = error_summary(
                    pred, data["target"][rows], data["patient"][rows], data["site"][rows]
                )
                _, baseline = error_summary(
                    old["predictions"][0],
                    data["target"][rows],
                    data["patient"][rows],
                    data["site"][rows],
                )
                differences.append(actual - baseline)
            per_person = np.mean(differences, axis=0)
            rng = np.random.default_rng(1237101)
            resampled = per_person[
                rng.integers(len(per_person), size=(10000, len(per_person)))
            ].mean(1)
            out.append(
                dict(
                    role=role,
                    reference=reference,
                    people=len(per_person),
                    mean_difference=float(per_person.mean()),
                    descriptive_ci95=np.quantile(resampled, [0.025, 0.975]).tolist(),
                    bootstrap_draws=10000,
                    bootstrap_seed=1237101,
                    interpretation="Post-hoc paired person bootstrap of fixed predictions; seed errors averaged, not ensemble outputs. Reused roles, no selection uncertainty or independent confirmation.",
                )
            )
    return out


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
        runtime["upstream_singleton_fits"] == runtime["upstream_original_and_prefix_bitwise"] == 27
    )
    assert runtime["selected_payload_reconstructions"] == runtime["selected_payloads_bitwise"] == 81
    assert runtime["continuation_banks"] == 6
    verification = OUT / "verification.json"
    if verification.exists():
        receipt = js(verification)
        assert receipt["passed"] and sha(SHORTCUT) == receipt["shortcut_sha256"]
        check_map({**receipt["artifact_sha256"], **receipt["postprocess_sources"]})
        print(dict(passed=True, mode="read-only", verification_sha256=sha(verification)))
        return
    rows, policies, candidates = [], [], []
    for role in ROLES:
        for mode in (0, 16, 256):
            for rate in (0.0001, 0.0003):
                for step in (0, 512, 2048, 8192, 32768, 131072):
                    rr = [
                        r
                        for r in result["records"]
                        if (r["role"], r["variants"], r["lr"], r["step"])
                        == (role, mode, rate, step)
                    ]
                    assert len(rr) == 3
                    tt = [
                        t
                        for t in runtime["records"]
                        if (t["role"], t["variants"], t["lr"], t["step"])
                        == (role, mode, rate, step)
                    ]
                    rows.append(
                        dict(
                            role=role,
                            variants=mode,
                            lr=rate,
                            step=step,
                            baseline_alias=step == 0,
                            selected_per_mode=rr[0]["selected_per_mode"],
                            selected_overall=rr[0]["selected_overall"],
                            person_mean=float(np.mean([r["metrics"]["person_mean"] for r in rr])),
                            p90=float(np.mean([r["metrics"]["p90"] for r in rr])),
                            largest_dose_worst_error=float(
                                np.mean([r["doses"][-1]["worst_error"]["person_mean"] for r in rr])
                            ),
                            parameters=643,
                            numeric_bytes=2886,
                            cached_array_bytes=5456,
                            archive_bytes_min=min(r["archive_bytes"] for r in rr),
                            archive_bytes_max=max(r["archive_bytes"] for r in rr),
                            cpu_median_us=float(np.median([t["median_us"] for t in tt])),
                            worst_seed_p95_us=max(t["p95_us"] for t in tt),
                        )
                    )
        entry = selection["roles"][role]
        for c in entry["candidates"]:
            candidates.append(dict(role=role, **c))
        for policy, c in {**entry["per_mode"], "overall": entry["overall"]}.items():
            row = next(
                r
                for r in rows
                if (r["role"], r["variants"], r["step"], r["lr"])
                == (role, c["variants"], c["step"], c["lr"] or 0.0001)
            )
            policies.append(dict(policy=policy, inner_person_mean=c["clean"], **row))
    assert len(rows) == 108 and len(candidates) == 99 and len(policies) == 12
    banks = [js(p) for p in RUN.rglob("receipt.json")]
    assert len(banks) == 12 and all(b["steps"] == 131072 for b in banks)
    presentations = sum(2 * i["presentations"] for b in banks for i in b["sample_inventory"])
    assert presentations == 216 * 131072 * 64 == 1811939328
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
    paired = pairs(load_data(), result)
    workflow = js(RUN / "progress.json")["seconds"]
    summary = dict(
        rows=rows,
        policies=policies,
        fg_context=fg,
        paired=paired,
        sources=len(lock["sources"]),
        inputs=len(lock["input_sha256"]),
        source_lock_sha256=source,
        selection_sha256=selected,
        results_sha256=results,
        primary_workflow_seconds=workflow,
        completed_bank_seconds=sum(b["full_bank_seconds"] for b in banks),
        full_bank_seconds_range=[
            min(b["full_bank_seconds"] for b in banks),
            max(b["full_bank_seconds"] for b in banks),
        ],
        cuda_peak_allocated_bytes=max(b["cuda_peak_allocated_bytes"] for b in banks),
        actual_continuation_trajectories=216,
        serialized_checkpoints=1296,
        baseline_alias_records=216,
        unique_final_baseline_models=9,
        total_presentations=presentations,
        presentations_per_model=8388608,
        final_pool_rows={b["role"]: b["pool_rows"] for b in banks if b["fold"] is None},
        current_goal_status="active",
        previous_turn_classification="progress",
    )
    write_json(OUT / "summary.json", summary)
    csv_out(OUT / "all_curves.csv", rows)
    csv_out(OUT / "policies.csv", policies)
    csv_out(OUT / "inner_candidates.csv", candidates)
    csv_out(OUT / "paired.csv", paired)
    csv_out(
        OUT / "sample_exposure.csv",
        [dict(role=b["role"], fold=b["fold"], **i) for b in banks for i in b["sample_inventory"]],
    )

    def values(mode, rate, step):
        return [
            next(
                r["person_mean"]
                for r in rows
                if (r["role"], r["variants"], r["lr"], r["step"]) == (role, mode, rate, step)
            )
            for role in ROLES
        ]

    baseline = values(0, 0.0001, 0)
    overall = [
        next(p["person_mean"] for p in policies if p["role"] == role and p["policy"] == "overall")
        for role in ROLES
    ]
    long = values(256, 0.0003, 131072)
    table = ["| Режим | Mixed | SLR → iPod | iPod → SLR |", "|---|---:|---:|---:|"]
    for label, vals in [
        ("NP: исходная маленькая модель", baseline),
        ("LT: выбор по внутренним данным", overall),
        ("LT: 131 072 шага, без добавок, lr=0,0003", values(0, 0.0003, 131072)),
        ("LT: 131 072 шага, 16 вариаций, lr=0,0003", values(16, 0.0003, 131072)),
        ("LT: 131 072 шага, 256 вариаций, lr=0,0003", long),
        ("FG: прежний ориентир, 20 284 числовых байта", list(fg.values())),
    ]:
        table.append(f"| {label} | " + " | ".join(f"{v:.5f}" for v in vals) + " |")
    selected_table = [
        "| Сценарий / число добавок | Шаг | Начальный lr | Внутренняя ΔE00 | Внешняя ΔE00 |",
        "|---|---:|---:|---:|---:|",
    ]
    for p in policies:
        selected_table.append(
            f"| {p['role']} / {p['policy']} | {p['step']} | {'—' if p['step'] == 0 else p['lr']} | {p['inner_person_mean']:.5f} | {p['person_mean']:.5f} |"
        )
    cpu = float(np.median([r["median_us"] for r in runtime["records"]]))
    times = " / ".join(
        f"{r['median_reconstruction_and_validation_seconds']:.3f}" for r in runtime["recipes"]
    )
    lines = [
        "# Luma ChromaSeed-LT: больше примеров и длинное обучение",
        "",
        "Запрос выполнен: 643-параметровая модель продолжала обучение до 131 072 обновлений, по 8 388 608 предъявлений на траекторию. "
        "Добавлено до 256 цифровых цветовых вариаций каждой исходной строки. Длинное обучение улучшило mixed, но ухудшило оба направления переноса при максимальном сроке. "
        "Общий внутренний выбор во всех трёх сценариях предпочёл режим без добавок; универсального улучшения нет.",
        "",
        *table,
        "",
        "Меньше ΔE00 — лучше; это ошибка цвета, не процент точности. Среднее сначала по людям, затем по трём seed-ошибкам отдельных моделей. "
        "Ансамбль не используется. Строки с 131 072 шагами показывают фиксированную часть полной кривой, они не заменяют внутренний выбор. "
        "Все 108 комбинаций role/mode/rate/checkpoint, включая повторные baseline-строки, сохранены в CSV.",
        "",
        f"У режима 256/lr=0,0003 на максимальном сроке mixed {baseline[0]:.5f} → {long[0]:.5f} "
        f"(снижение ошибки {100 * (1 - long[0] / baseline[0]):.2f}%). Переносы {baseline[1]:.5f} → {long[1]:.5f} "
        f"и {baseline[2]:.5f} → {long[2]:.5f} ухудшаются. Даже mixed не превосходит FG {fg['mixed']:.5f}. "
        "Увеличение 16 → 256 вариаций на этом сроке даёт небольшую разницу mixed; число вариаций не равно разнообразию новых людей и камер.",
        "",
        *selected_table,
        "",
        "Все 99 внутренних кандидатов и 12 решений независимо пересчитаны. Overall: mixed — 2048 дополнительных шага без добавок; "
        "SLR → iPod — шаг 0, сохранён исходный NP; iPod → SLR — 2048 без добавок. "
        "Итоговый mixed немного лучше NP, прямой перенос совпал, обратный стал хуже. Ошибки внешних строк не использовались для пересмотра выбора. "
        "Парные описательные интервалы в paired.csv относятся к фиксированным прогнозам и не учитывают неопределённость выбора.",
        "",
        "Использован только оригинальный TRAIN: 966 строк, 24 человека. На стороне mixed fit 734 строки/18 человек, оценка 232/6; "
        "в двух направлениях 323/8 и 643/16 меняются местами. Роли многократно использованы и перекрываются; состав людей связан с камерой. "
        "Проверялись 36 статистик подготовленных участков кожи с инструментальным Lab D65/10°, не повседневные фотографии лица.",
        "",
        "Большой mixed-пул содержит 188 638 вариантов из 734 оригиналов. Вариант 0 точный исходный; остальные — ограниченные RGB-преобразования "
        "с дозой до 4/255. В режимах 16/256 половина вероятностной массы остаётся на оригинале. "
        "Синтетический вариант сохраняет старый Lab по допущению; нового измерения не появляется. "
        "При полном сроке все 188 638 mixed-вариантов действительно встретились каждому seed. "
        "Источники вариантов никогда не переходят из fit в person-disjoint query.",
        "",
        f"Размер остался 643 параметра, 2 886 числовых байт и 5 456 байт кэшированных массивов. Ответ: медиана {cpu:.1f} мкс "
        "на один готовый вектор, CPU один поток Ryzen 9 7900X. Время включает нормализацию, исключает чтение фото, поиск лица и извлечение кожи. "
        "Веса FP32, потребитель считает FP64 после FP32 нормализации; INT8/битовые ядра в LT не применялись. NPZ-служебные байты отдельно в CSV.",
        "",
        f"RTX4060: 12 банков по 18 траекторий, 216 продолжений, суммарно {presentations:,} предъявлений с повторениями. "
        f"Полный первичный процесс — {workflow:.2f} с; сумма завершённых fit-вызовов — {summary['completed_bank_seconds']:.2f} с. "
        f"Каждый банк до 131 072 шагов занял {summary['full_bank_seconds_range'][0]:.2f}–{summary['full_bank_seconds_range'][1]:.2f} с. "
        "Это время всего банка из 18 моделей. Его нельзя делить на 18 и выдавать за время отдельного обучения. "
        "Первичный процесс использовал сохранённые NP warm starts; историческая подготовка признаков и исходное обучение NP/ND в его время не входят.",
        "",
        f"Отдельное полное воспроизведение выбранного рецепта: {times} с для mixed / прямого / обратного переноса. "
        "Каждый повтор заново обучает все три исходных ND singleton-модели, экспортирует NP и, где выбрано продолжение, "
        "выполняет весь банк 18 слотов с исходным горизонтом расписания до 2048 шагов. Стоимость всех трёх исходных моделей и банка сохранена. "
        "При прямом переносе LT не выбран, учтены только три исходных fit. Первый из трёх повторов исключён только из времени. "
        "27 исходных обучений точно воспроизведены; 6 LT-банков и все 81 выбранных экспорта, включая baseline-алиасы, дали точные веса. "
        "Прогретый процесс/GPU; чтение файлов и запуск процесса исключены. В runtime.json отдельно указано время построения и время с проверкой.",
        "",
        f"Аудит: 972 внутренних модели, 183 600 OOF-векторов, 324 финальных модели, все 4 269 672 итоговых вектора и одиночных вызова. "
        f"Максимальное расхождение независимого расчёта {audit['maximum_numpy_audit_drift']:.3g} Lab при допуске 2e-8; "
        f"CUDA/NumPy {audit['maximum_cuda_numpy_drift']:.3g} при допуске 0,002. "
        "Проверены исходные нормализаторы, родительские веса, исходный пул, потоки выборки и число реально использованных вариаций. "
        "Всего сериализовано 1296 checkpoint-моделей: 1080 после обновлений и 216 baseline-алиасов. "
        "На финальной стороне только 9 уникальных исходных весов; алиасы не являются независимыми экспериментами.",
        "",
        "Восемь поведенческих тестов включают источник/массу вариантов, сохранение нормализаторов и исходных весов, префиксы выборок, "
        "исходный горизонт learning rate и точное повторение фиксированного CUDA-банка. "
        "До фиксации источников найдено и исправлено разделение CPU-памяти базовой ставки с изменяемой ставкой оптимизатора; "
        "регрессионный тест сначала воспроизвёл сбой. Все первичные измерения сделаны после исправления. "
        "Исходники и результаты предыдущей NB-серии с отрицательным результатом сохранены.",
        "",
        "Luma ChromaSeed-LT — исследовательская серия. Высокая точность подбора косметики по обычным селфи, коммерческая готовность "
        "и соответствие критериям Сколково этим опытом не установлены. Широкая цель остаётся активной.",
        "",
        "[Все кривые](all_curves.csv) · [Выбранные варианты](policies.csv) · [Внутренние кандидаты](inner_candidates.csv) · "
        "[Реальное число примеров](sample_exposure.csv) · [Парные сравнения](paired.csv) · [Аудит](audit.json) · "
        "[Время](runtime.json) · [Сводка](summary.json) · [Проверка](verification.json).",
        "",
        "[Протокол](../../research/chromaseed_long_training_v1_protocol.md) · [Карточка](../../architecture/chromaseed_long_training_model_card.md) · "
        "[Следующее решение](../../research/chromaseed_long_training_next_decision.md).",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    CARD.write_text(
        """# Luma ChromaSeed-LT model card

Long continuation of the accepted NP643-parameter blind head:36->16ReLU->3. One prepared color36 vector produces instrument-native D65/10-degree Lab. Existing scripts/chromaseed_neural_prefix_numpy.py Predictor accepts one(36,) input; predict batches Nx36. Keep NP metadata and fit-only FP32 normalizers. The historical family='blind4' and original_k/prefix fields retain lineage; LT uses ordinary within-head backprop/AdamW, not a new four-block denoising training pass.

FP32 numerical payload2,886B,643 parameters,5,456B cached consumer arrays, about6.4us single-thread prepared-feature response. NPZ archive, code, runtime, image decoding and face/skin/feature extraction are additional. CPU arithmetic FP64 after FP32 normalization. No new quantization claim.

Only original TRAIN966rows/24people; exact per-fold ND warm starts for inner training, exact NP exports for final fitting. Modes0/16/256 bounded digital variants, unchanged target by synthetic assumption, half clean probability in augmented modes. No new measured people or external images. Six checkpoints through131072;216 trajectories,1296 records including216 baseline aliases. Reused exploratory roles and camera/person confounding remain.

Inner choices prefer no augmentation in all three roles:2048/0/2048 additional steps. Overall errors5.71611/8.29435/8.68347 against initial5.77054/8.29435/8.58953; mixed modestly improves, forward unchanged, reverse worsens. At131072/256variants/lr.0003:5.46195/9.12002/9.13320, an exploratory long-run comparison, not selected deployment settings. There is no universal model promotion or independent facial-phone quality result.

All324 final models use the same NP consumer. Their directory is historically named selected, but only selected_per_mode/selected_overall flags identify the frozen12 policies. Baseline0 duplicates are exact aliases. Consult policies.csv and source/selection/results receipts before choosing a payload; do not choose from known outer accuracy.

Independent audit validates all4,269,672 final vectors and actual consumers. Rebuilt27 original models/NP exports and81 selected payloads are bitwise identical. Full selected construction costs1.425/0.874/1.422seconds per three-seed role recipe; all upstream fits and complete18-slot banks charged, not individual-model latency. Full primary workflow266.44seconds reuses saved warm starts. Eight behavioral tests pass. No photos, pretrained weights, packages or external publication added. Existing upstream licenses apply.

[Evidence](../benchmarks/chromaseed_long_training_v1/report.md) · [Protocol](../research/chromaseed_long_training_v1_protocol.md).
""",
        encoding="utf-8",
    )
    NEXT.write_text(
        """# After LT: diminishing returns from repeated synthetic colors

The user's more-examples/long-training experiment is completed and verified.216 trajectories through131072 updates, each8,388,608 presentations; mixed256-mode actually visits188,638 available variants. Compact capacity643parameters/2886numericB is preserved. Long fixed comparisons improve mixed but worsen transfers, and all overall inner choices reject augmentation. Keep all negative curves and original NP baselines; do not replace the frozen selector with the mixed outer winner. Broad ordinary-phone facial/high-quality goal stays active.

Stop extending this same duration/variant grid just to seek a favorable reused outer result. More repeated digital colors cannot be counted as more measured people, sites or acquisition conditions. LT does not prove a hard architecture accuracy ceiling or that no augmentation can help. The bounded input perturbation/unchanged-target assumption and reused24-person support limit the conclusion.

Next useful local step is an inner-only diagnostic of existing LT checkpoint variation, before further fits. Inventory earlier recurrence/teacher/EMA and sample-allocation work first. This turn's scoped documentation search found earlier skin_color_sampling_protocol_v1.md and skin_sampling_transfer_next_decision.md: balancing/importance sampling already tested elsewhere and not a first-use idea. Historical results from those documents use different models/roles and are not LT baselines; do not open their legacy validation/calibration/test arrays. cc_v4_next_decision.md proposes an EMA teacher, while cc_v5_protocol.md excludes EMA in that series; this scoped search is not proof that averaging has never been tested elsewhere.

A candidate diagnostic, after a separate preregistration, is whether fluctuations between existing same-seed/same-trajectory heads can be reduced by predetermined checkpoint-weight averages at the same643-parameter inference size. Preserve original normalizers and head identity, use matching inner weights only, never average different seeds/heads with unaligned hidden neurons. Compare fixed last-checkpoint and step0 controls, freeze any selector inside original TRAIN before final evaluation, charge all contributing training. This is a planned arithmetic/variance question, not a demonstrated gain, novelty, new method or a launched experiment. It must not become an outer-driven interpolation search. A negative result should close that hypothesis promptly.

Ordinary-camera accuracy ultimately still needs a valid facial/capture evaluation and measured support; this turn adds no new acquisition. Current original-TRAIN-only boundary persists: no new data/images/weights/packages, agents, publication or messages. LT source and evidence are immutable after sealing; only report read-only verification may run. Do not rerun LT primary/audit/runtime, NB/NP/ND bound writers or G/GS verifier mains.

[LT evidence](../benchmarks/chromaseed_long_training_v1/report.md) · [Protocol](chromaseed_long_training_v1_protocol.md).
""",
        encoding="utf-8",
    )
    test = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_chromaseed_long_training.py", "-q"],
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
            "tests/test_chromaseed_long_training.py",
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
            target = (p.parent / link.split("#")[0]).resolve()
            assert target.is_file() or target == verification, link
    SHORTCUT.parent.mkdir(parents=True, exist_ok=True)
    SHORTCUT.write_text(
        f"# Luma ChromaSeed-LT\n\n[Полный отчёт]({(OUT / 'report.md').as_posix()})\n\n643 параметра,2,9КБ; до8,4млн предъявлений на модель. Длинное обучение улучшает mixed, ухудшает перенос. Выбранные результаты5,71611/8,29435/8,68347; обычные селфи не проверены.\n",
        encoding="utf-8",
    )
    receipt = dict(
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
        upstream_fits=27,
        continuation_replay_banks=6,
        exact_selected_payloads=81,
        selection_unchanged=True,
        goal_status="active",
    )
    write_json(verification, receipt)
    print(
        dict(
            passed=True,
            verification_sha256=sha(verification),
            tests=test.stdout.strip(),
            overall=overall,
            long256_rate0003=long,
            numeric_bytes=2886,
        )
    )


if __name__ == "__main__":
    main()
