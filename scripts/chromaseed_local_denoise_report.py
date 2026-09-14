"""Publish local ND evidence and seal it; subsequent calls verify read-only."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from collections import Counter

import numpy as np
from chromaseed_kernel_audit import js
from chromaseed_local_denoise_train import ROOT, RUN
from skin_local_search_train import sha, write_json

OUT = ROOT / "docs/benchmarks/chromaseed_local_denoise_v1"
CARD = ROOT / "docs/architecture/chromaseed_local_denoise_model_card.md"
NEXT = ROOT / "docs/research/chromaseed_local_denoise_next_decision.md"
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-local-denoise-2026-09-13.md"
FAMILIES = ("plain", "local2", "local4", "blind4", "e2e4", "fg_norm_static", "random_head")


def check_map(mapping):
    for path, digest in mapping.items():
        assert sha(ROOT / path) == digest, path


def csv_out(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def main():
    lock, selection, result = [
        js(RUN / name) for name in ("source_lock.json", "selections.json", "results.json")
    ]
    audit, runtime = js(OUT / "audit.json"), js(OUT / "runtime.json")
    check_map(
        {
            **lock["sources"],
            **lock["input_sha256"],
            **audit["artifact_sha256"],
            **audit["dependencies"],
            **runtime["dependencies"],
        }
    )
    assert audit["passed"] and runtime["audit_sha256"] == sha(OUT / "audit.json")
    for obj in (selection, result, audit, runtime):
        assert obj["source_lock_sha256"] == sha(RUN / "source_lock.json")
    for obj in (result, audit, runtime):
        assert obj["selection_sha256"] == sha(RUN / "selections.json")
    for obj in (audit, runtime):
        assert obj["results_sha256"] == sha(RUN / "results.json")
    assert runtime["reconstructions"] == runtime["bitwise_equal_reconstructions"] == 45
    verification = OUT / "verification.json"
    if verification.exists():
        receipt = js(verification)
        check_map(receipt["artifact_sha256"])
        check_map(receipt["postprocess_sources"])
        assert receipt["passed"] and sha(SHORTCUT) == receipt["shortcut_sha256"]
        print(dict(passed=True, mode="read-only", verification_sha256=sha(verification)))
        return
    rows, candidates, curves = [], [], []
    for role in ("mixed", "slr_to_ipod", "ipod_to_slr"):
        for family in FAMILIES:
            rr = [r for r in result["records"] if r["role"] == role and r["family"] == family]
            tt = [r for r in runtime["records"] if r["role"] == role and r["family"] == family]
            ff = [
                r
                for r in runtime["full_fit_records"]
                if r["role"] == role and r["family"] == family
            ]
            stage = np.mean([[s["person_mean"] for s in r["stage_metrics"]] for r in rr], axis=0)
            rows.append(
                dict(
                    role=role,
                    family=family,
                    person_mean=float(np.mean([r["metrics"]["person_mean"] for r in rr])),
                    p90=float(np.mean([r["metrics"]["p90"] for r in rr])),
                    parameters=rr[0]["parameters"],
                    numeric_bytes=rr[0]["numeric_bytes"],
                    archive_bytes_min=min(r["archive_bytes"] for r in rr),
                    archive_bytes_max=max(r["archive_bytes"] for r in rr),
                    step=rr[0]["step"],
                    lr=rr[0]["lr"],
                    cpu_median_us=float(np.median([r["median_us"] for r in tt])),
                    worst_seed_p95_us=max(r["p95_us"] for r in tt),
                    cached_array_bytes=tt[0]["cached_array_bytes"],
                    full_fit_gpu_ms=None if not ff else ff[0]["full_fit_seconds"] * 1000,
                    clean_stage_errors=stage.tolist(),
                    worst_dose64_error=float(
                        np.mean([r["doses"][-1]["worst_error"]["person_mean"] for r in rr])
                    ),
                )
            )
            curves += [
                dict(role=role, family=family, stage=i + 1, person_mean=float(value))
                for i, value in enumerate(stage)
            ]
            if family in selection["roles"][role]:
                entry = selection["roles"][role][family]
                for candidate in entry["candidates"]:
                    candidates.append(
                        dict(
                            role=role,
                            family=family,
                            selected=candidate == entry["selected"],
                            **candidate,
                        )
                    )
    assert len(rows) == 21 and len(candidates) == 90 and len(curves) == 51
    bank_receipts = [
        js(p)
        for p in list((RUN / "inner").glob("*/*/fold*/receipt.json"))
        + list((RUN / "final").glob("*/*/s*/receipt.json"))
    ]
    assert len(bank_receipts) == 90
    accounting = dict(
        completed_fit_banks=90,
        inner_trajectories=270,
        inner_checkpoint_models=810,
        new_final_models=45,
        exact_reference_models=18,
        completed_bank_seconds=sum(r["full_bank_seconds"] for r in bank_receipts),
        completed_model_steps=sum(r["steps"] * len(r["slots"]) for r in bank_receipts),
        peak_cuda_allocated_bytes=max(r["cuda_peak_allocated_bytes"] for r in bank_receipts),
        checkpoint_choices=dict(Counter(c["step"] for c in candidates if c["selected"])),
        audit_seconds=audit["seconds"],
        runtime_seconds=runtime["seconds"],
        interrupted_unfinished_bank_cost_in_completed_sum=False,
    )
    write_json(OUT / "summary.json", dict(rows=rows, accounting=accounting))
    csv_out(OUT / "all_models.csv", rows)
    csv_out(OUT / "inner_candidates.csv", candidates)
    csv_out(OUT / "stage_curves.csv", curves)
    csv_out(OUT / "paired.csv", audit["paired"])
    by = {(r["role"], r["family"]): r for r in rows}

    def q(role, family):
        return by[(role, family)]["person_mean"]

    table = [
        "| Модель | Mixed ΔE00 | SLR → iPod | iPod → SLR | Числовые байты |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        table.append(
            f"| {family} | {q('mixed', family):.6f} | {q('slr_to_ipod', family):.6f} | {q('ipod_to_slr', family):.6f} | {by[('mixed', family)]['numeric_bytes']:,} |"
        )
    times = ["| Модель | CPU ответ, мкс | Полное обучение GPU, мс |", "|---|---:|---:|"]
    for family in FAMILIES:
        row = by[("mixed", family)]
        fit = (
            "не перемерялось" if row["full_fit_gpu_ms"] is None else f"{row['full_fit_gpu_ms']:.3f}"
        )
        times.append(f"| {family} | {row['cpu_median_us']:.2f} | {fit} |")
    lines = [
        "# ChromaSeed-ND: локальное уточнение цвета, 13 сентября 2026",
        "",
        "**Четыре локальных блока дают интересный компромисс размера и качества, но не общую победу.** "
        "На смешанном сценарии local4 хуже обычной сети и FG; при переносах средняя ошибка ниже FG, "
        "однако описательные интервалы сравнений с FG включают ноль. При обратном переносе e2e4 одновременно точнее и быстрее обучается, чем local4. "
        "Метод не назначен универсальной заменой.",
        "",
        "Это собственная непрерывная адаптация идеи NoProp. Локальные блоки учатся восстанавливать зашумлённый Lab, "
        "с локальными градиентами; на применении только признаки изображения и начальное состояние ноль. "
        "Прежний R уже делал повторные поправки; сама рекуррентность здесь не новая.",
        "",
        *table,
        "",
        "Меньше ΔE00 — лучше. Среднее по людям, затем по трём seed; прогнозы seed не усредняются в ансамбль. "
        "Все сценарии повторно используют части оригинального TRAIN (966 изображений / 24 человека). "
        "При переносе вместе с камерой меняются люди. Это подготовленные признаки участков кожи, а не проверка обычных селфи.",
        "",
        "В mixed размеры local4 / FG: 11 368 / 20 284 числовых байта, уменьшение на 43,96%. "
        "Но ответ local4 медленнее. Числовой размер включает веса и нормализацию; архив и рабочие массивы указаны отдельно в CSV. "
        "У blind4 192 сохранённых коэффициента неактивны: его эффективное число параметров 2 572, не 2 764.",
        "",
        *times,
        "",
        "GPU — RTX 4060, CPU — один поток Ryzen 9 7900X. Ответ: медиана трёх seed-медиан реального потребителя, "
        "с нормализацией, без извлечения признаков и поиска лица. Обучение: seed17, одна модель, "
        "полная подготовка, выборка/шум, перенос, создание CUDA Graph, сброшенный прогрев, шаги и экспорт; "
        "из трёх полных повторов первый отброшен только для времени. Все 45 повторов точно воспроизвели веса. "
        "Цена обучения FG здесь не перемерялась; её нельзя сравнивать как замер на той же GPU.",
        "",
        "**Что показали проходы.** local2 в mixed: 5,63062 → 5,72584; дополнительный проход ухудшил ответ. "
        "local4 при SLR → iPod: 8,85964 → 8,33366 → 8,30599 → 8,49549. "
        "При iPod → SLR: 8,48591 → 8,82204 → 8,53029 → 8,32429. "
        "Чистый прогноз каждого блока и скрытое зашумлённое состояние — разные величины; здесь приведена ошибка чистого прогноза. "
        "Ранние стадии не выбирались после просмотра этих внешних оценок.",
        "",
        "Для local4 против e2e4 разность ΔE00 в mixed +0,00370, SLR → iPod −0,12566, iPod → SLR +0,29781. "
        "В последнем случае описательный интервал [0,09186; 0,49902] хуже нуля; "
        "local4 против blind4 там −0,28349, интервал [−0,48525; −0,04730]. "
        "Эти интервалы используют фиксированные прогнозы и повторно виденных людей, не учитывают неопределённость выбора модели "
        "и не являются новой независимой валидацией.",
        "",
        f"**Объём.** 45 внутренних пакетов, 270 траекторий и 810 checkpoint-моделей; 90 оценок настроек, 15 решений, 45 финальных моделей и 18 точных контрольных моделей из NS. "
        f"Выбрано шагов: {accounting['checkpoint_choices']}; ни одного 8192. Сумма времени завершённых обучающих пакетов {accounting['completed_bank_seconds']:.3f} с, "
        f"{accounting['completed_model_steps']:,} принятых модельных шагов. Максимум выделенных CUDA-массивов {accounting['peak_cuda_allocated_bytes'] / 2**20:.2f} МиБ; это не вся память GPU/процесса.",
        "",
        "**Проверка.** Независимое вычисление слоёв/рекуррентности, нормализации, 459 000 внутренних стадийных строк, "
        "всех 90 оценок / 15 решений и 2 016 234 финальных стадийных строк. "
        "830 214 вызовов реального потребителя на всех 33 преобразованиях; все метрики и 48 парных сравнений пересчитаны. "
        f"Максимальный разрыв независимого расчёта и экспорта {audit['max_final_native_lab']:.3g} native Lab. 18 тестов включают разделение локальных градиентов, независимость пакета, "
        "оптимизатор, стадии, экспорт и сброс CUDA Graph.",
        "",
        "**Восстановление запуска.** Первый процесс завершился с WinError 5 при атомарной замене progress.json после 33 завершённых пакетов. "
        "Временная блокировка записи исчезла; вероятна гонка с параллельным чтением файла статуса. "
        "После подтверждения завершения процесса возобновлён тот же замороженный код: 33 пакета проверены и повторно использованы, "
        "незавершённый пакет построен заново. Численные правила и допуски не менялись. "
        "Затраты прерванной части не включены в сумму времени завершённых пакетов; это не полное время работы всех попыток.",
        "",
        "[Все модели](all_models.csv) · [Внутренние кандидаты](inner_candidates.csv) · [Все стадии](stage_curves.csv) · "
        "[Парные сравнения](paired.csv) · [Аудит](audit.json) · [Замеры](runtime.json) · [Проверка](verification.json).",
        "",
        "[Протокол](../../research/chromaseed_local_denoise_v1_protocol.md) · "
        "[Карточка модели](../../architecture/chromaseed_local_denoise_model_card.md) · "
        "[Следующее решение](../../research/chromaseed_local_denoise_next_decision.md).",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    CARD.write_text(
        """# ChromaSeed-ND model card

Research-only five-family comparison on prepared 36-dimensional skin-region color features. Output is native instrument D65/10-degree Lab. Actual consumer scripts/chromaseed_local_denoise_numpy.py accepts exactly one finite (36,) vector; trace returns each block's clean prediction. No target, hidden reference photo or inference seed is passed. Fixed zero initialization and two/four unshared steps for local variants. Backpropagation remains inside each local block.

Plain 2,763 parameters / 11,364 numeric B; local2 2,758 / 11,344; local4, blind4 and e2e4 2,764 / 11,368. Blind4 has192 inactive state-input coefficients, giving2,572 active parameters. FP32 payload, FP32 input normalization, FP64 cached NumPy arithmetic. Local4 cached arrays22,504B, excluding Python objects. Face localization, feature extraction, camera calibration, model code and runtime dependencies are outside payload size and latency.

No universal replacement: local4 mixed error5.700539 exceeds plain5.636588 and FG5.438652. Its transfer errors8.495492/8.324291 improve over FG8.597000/8.705018 on reused, camera/person-confounded roles; descriptive FG comparison intervals include zero. E2e4 reverse8.026476 is stronger and faster to train than local4. No ordinary-phone facial or cosmetic shade-match accuracy established.

Local4 full standalone RTX4060 fits about103/286/287ms for mixed/forward/reverse selected settings; CPU response25.5/25.7/25.9us on one Ryzen7900X thread. All45 complete selected-setting replay fits exactly reproduced payloads. No global backprop-free training or superiority to the original NoProp paper claimed. New data, weights and packages were not used.

Training and selection use original TRAIN only, with person-disjoint inner folds and fixed seed/rate/step grid. Exposed old validation/calibration/test are excluded. Upstream data/code rights remain those in the existing provenance archive; no external NoProp implementation was copied.

[Full evidence](../benchmarks/chromaseed_local_denoise_v1/report.md) · [Protocol](../research/chromaseed_local_denoise_v1_protocol.md).
""",
        encoding="utf-8",
    )
    NEXT.write_text(
        """# After ND: test compact prefixes using inner evidence

ND is verified progress, not broad-goal completion. Block-local noisy-target supervision has been implemented and tested, not merely proposed. Local4 improves both raw36 transfer means against FG with about44% fewer numeric bytes, but the descriptive intervals overlap zero, mixed is worse, and its response is slower. E2e4 beats local4 on reverse transfer at a cheaper selected fit. These outcomes do not justify a universal local-learning advantage. Preserve all positive and negative results and do not expand the step/rate grid against observed external curves.

All five networks supervise clean predictions; saved inner stage outputs allow a different concrete capacity question: can an inner-selected prefix preserve useful accuracy with fewer stored blocks and lower latency? Earlier R tested per-image threshold stopping with shared weights and showed rounding-sensitive exits; it is not the same as removing unused unshared ND blocks. Inventory that distinction and register fixed-prefix selection first. Retain the ORIGINAL schedule denominator K when exporting a prefix: changing K changes its earlier states. For a one-block prefix the initial state is exactly zero, so its unused state-input columns can be removed algebraically. Selection must use inner person folds only, not the favorable known outer stage curve. Account for all original training required to obtain the retained blocks, particularly end-to-end prefixes that received gradients from later blocks.

This prefix study is planned, not implemented or launched in ND. A subsequent per-input early exit would need its own inner-fitted rule and full timing; small state changes alone are not evidence of small true error. Local versus global learning and inference/training-state mismatch remain separate questions; do not mix new objectives, prefix selection and new noise schedules into one untraceable intervention.

Use only original TRAIN under the current boundary. No new data/images/weights/packages, agents, messages or publication. ND report script re-verifies read-only after sealing. Do not rerun ND audit/runtime mains after their outputs are bound: those are measurement writers. Never run old G/GS verifier mains. Ordinary-phone face/end-to-end high quality remains unvalidated, so the full goal stays active.

[ND report](../benchmarks/chromaseed_local_denoise_v1/report.md) · [Prior R](chromaseed_refine_next_decision.md) · [Forum methods/data](chromaseed_forum_update_after_ns_2026-09-13.md).
""",
        encoding="utf-8",
    )
    test = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_chromaseed_local_denoise.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert test.returncode == 0, test.stdout + test.stderr
    script_names = (
        "chromaseed_local_denoise",
        "chromaseed_local_denoise_numpy",
        "chromaseed_local_denoise_fit",
        "chromaseed_local_denoise_train",
        "chromaseed_local_denoise_audit",
        "chromaseed_local_denoise_runtime",
        "chromaseed_local_denoise_report",
    )
    lint = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            *(f"scripts/{name}.py" for name in script_names),
            "tests/test_chromaseed_local_denoise.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert lint.returncode == 0, lint.stdout + lint.stderr
    artifacts = dict(audit["artifact_sha256"])
    for p in (*OUT.glob("*"), CARD, NEXT):
        if p.is_file() and p.name != "verification.json":
            artifacts[p.relative_to(ROOT).as_posix()] = sha(p)
    for p in (OUT / "report.md", CARD, NEXT):
        for link in re.findall(r"\]\(([^)]+)\)", p.read_text(encoding="utf-8")):
            if not link.startswith(("https://", "http://")):
                destination = (p.parent / link.split("#")[0]).resolve()
                assert destination.is_file() or destination == verification, link
    SHORTCUT.parent.mkdir(parents=True, exist_ok=True)
    SHORTCUT.write_text(
        f"# Luma ChromaSeed-ND\n\n[Полный отчёт]({(OUT / 'report.md').as_posix()})\n\n11,4 КБ, локальное уточнение реализовано; общей победы нет. Обычные селфи не проверены.\n",
        encoding="utf-8",
    )
    receipt = dict(
        passed=True,
        source_lock_sha256=sha(RUN / "source_lock.json"),
        selection_sha256=sha(RUN / "selections.json"),
        results_sha256=sha(RUN / "results.json"),
        artifact_sha256=artifacts,
        postprocess_sources={
            f"scripts/{name}.py": sha(ROOT / f"scripts/{name}.py") for name in script_names
        },
        shortcut_sha256=sha(SHORTCUT),
        tests=test.stdout,
        lint=lint.stdout,
        coverage=audit["counts"],
        full_fit_reconstructions=45,
        all_bitwise_equal=True,
    )
    write_json(verification, receipt)
    print(dict(passed=True, verification_sha256=sha(verification), tests=test.stdout.strip()))


if __name__ == "__main__":
    main()
