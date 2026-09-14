"""Seal NP measurements, reproducible tables and report; thereafter read-only."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys

import numpy as np
from chromaseed_kernel_audit import js
from chromaseed_neural_prefix_run import ND, OUT, ROOT, RUN, check_map
from skin_local_search_train import sha, write_json

CARD = ROOT / "docs/architecture/chromaseed_neural_prefix_model_card.md"
NEXT = ROOT / "docs/research/chromaseed_neural_prefix_next_decision.md"
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-neural-prefix-2026-09-13.md"
FAMILIES = ("plain", "local2", "local4", "blind4", "e2e4")
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")


def csv_out(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()
                }
            )


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
    assert audit["passed"] and runtime["audit_sha256"] == sha(OUT / "audit.json")
    for obj in (selection, result, audit, runtime):
        assert obj["source_lock_sha256"] == source
    for obj in (result, audit, runtime):
        assert obj["selection_sha256"] == selected
    for obj in (audit, runtime):
        assert obj["results_sha256"] == results
    assert (
        runtime["reconstructions"]
        == runtime["bitwise_equal_original_reconstructions"]
        == runtime["bitwise_equal_prefix_reconstructions"]
        == 90
    )
    assert runtime["unique_full_fit_settings"] == 30
    verification = OUT / "verification.json"
    if verification.exists():
        receipt = js(verification)
        assert receipt["passed"] and sha(SHORTCUT) == receipt["shortcut_sha256"]
        check_map({**receipt["artifact_sha256"], **receipt["postprocess_sources"]})
        print(dict(passed=True, mode="read-only", verification_sha256=sha(verification)))
        return
    rows, policies, candidates, controls = [], [], [], []
    for role in ROLES:
        for family in (*FAMILIES, "fg_norm_static", "random_head"):
            rr = [r for r in result["records"] if r["role"] == role and r["family"] == family]
            for j in sorted({r["prefix"] for r in rr}, key=lambda x: x or 0):
                mm = [r for r in rr if r["prefix"] == j]
                tt = [
                    t
                    for t in runtime["records"]
                    if t["role"] == role and t["family"] == family and t["prefix"] == j
                ]
                ff = [
                    t
                    for t in runtime["full_fit_records"]
                    if t["role"] == role and t["family"] == family and t["prefix"] == j
                ]
                assert len(mm) == len(tt) == 3 and len(ff) <= 1
                row = dict(
                    role=role,
                    family=family,
                    prefix=j,
                    full_prefix=mm[0]["full_prefix"],
                    policies=mm[0]["policies"],
                    person_mean=float(np.mean([m["metrics"]["person_mean"] for m in mm])),
                    p90=float(np.mean([m["metrics"]["p90"] for m in mm])),
                    numeric_bytes=mm[0]["numeric_bytes"],
                    parameters=mm[0]["parameters"],
                    executed_blocks=mm[0]["executed_blocks"],
                    archive_bytes_min=min(m["archive_bytes"] for m in mm),
                    archive_bytes_max=max(m["archive_bytes"] for m in mm),
                    cpu_median_us=float(np.median([t["median_us"] for t in tt])),
                    worst_seed_p95_us=max(t["p95_us"] for t in tt),
                    cached_array_bytes=tt[0]["cached_array_bytes"],
                    full_fit_gpu_ms=None if not ff else ff[0]["full_fit_seconds"] * 1000,
                    original_fit_gpu_ms=None if not ff else ff[0]["original_fit_seconds"] * 1000,
                    step=mm[0]["step"],
                    lr=mm[0]["lr"],
                    largest_dose_worst_error=float(
                        np.mean([m["doses"][-1]["worst_error"]["person_mean"] for m in mm])
                    ),
                )
                rows.append(row)
                if mm[0]["full_prefix"]:
                    ct = [
                        c
                        for c in runtime["original_ND_controls"]
                        if c["role"] == role and c["family"] == family
                    ]
                    old = [
                        c
                        for c in js(ND / "results.json")["records"]
                        if c["role"] == role and c["family"] == family
                    ]
                    assert len(ct) == len(old) == 3
                    controls.append(
                        dict(
                            role=role,
                            family=family,
                            original_bytes=old[0]["numeric_bytes"],
                            exported_bytes=row["numeric_bytes"],
                            original_us=float(np.median([c["median_us"] for c in ct])),
                            exported_us=row["cpu_median_us"],
                            same_clean_output=True,
                        )
                    )
            if family in FAMILIES:
                entry = selection["roles"][role][family]
                for c in entry["candidates"]:
                    candidates.append(
                        dict(
                            role=role,
                            family=family,
                            quality=c == entry["policies"]["quality"],
                            compact=c == entry["policies"]["compact"],
                            **c,
                        )
                    )
                full = next(
                    r
                    for r in rows
                    if r["role"] == role and r["family"] == family and r["full_prefix"]
                )
                for policy, choice in entry["policies"].items():
                    row = next(
                        r
                        for r in rows
                        if r["role"] == role
                        and r["family"] == family
                        and r["prefix"] == choice["prefix"]
                    )
                    policies.append(
                        dict(
                            policy=policy,
                            inner_person_mean=choice["clean"],
                            error_difference_from_full=row["person_mean"] - full["person_mean"],
                            **row,
                        )
                    )
    assert len(rows) == 51 and len(policies) == 30 and len(candidates) == 45 and len(controls) == 15
    outcomes = {}
    for policy in ("quality", "compact"):
        differences = np.array(
            [p["error_difference_from_full"] for p in policies if p["policy"] == policy]
        )
        outcomes[policy] = dict(
            better=int(np.sum(differences < -1e-8)),
            worse=int(np.sum(differences > 1e-8)),
            same=int(np.sum(abs(differences) <= 1e-8)),
        )
    summary = dict(
        rows=rows,
        policies=policies,
        full_prefix_controls=controls,
        policy_outcomes_vs_full=outcomes,
        source_lock_sha256=source,
        selection_sha256=selected,
        results_sha256=results,
        sources=len(lock["sources"]),
        inputs=len(lock["input_sha256"]),
        coverage=audit["counts"],
        unique_full_fit_settings=30,
        full_reconstructions=90,
        previous_turn_classification="progress",
        goal_status="active",
    )
    write_json(OUT / "summary.json", summary)
    csv_out(OUT / "all_prefixes.csv", rows)
    csv_out(OUT / "policies.csv", policies)
    csv_out(OUT / "inner_candidates.csv", candidates)
    csv_out(OUT / "full_prefix_controls.csv", controls)
    csv_out(OUT / "paired.csv", audit["paired"])

    def policy_row(role, family, policy="compact"):
        return next(
            r
            for r in policies
            if r["role"] == role and r["family"] == family and r["policy"] == policy
        )

    table = [
        "| Компактная политика | J: mixed / вперёд / обратно | Числовые байты | ΔE00 mixed | SLR → iPod | iPod → SLR |",
        "|---|---|---|---:|---:|---:|",
    ]
    for family in FAMILIES:
        pp = [policy_row(role, family) for role in ROLES]
        table.append(
            f"| {family} | {' / '.join(str(p['prefix']) for p in pp)} | {' / '.join(str(p['numeric_bytes']) for p in pp)} | "
            + " | ".join(f"{p['person_mean']:.5f}" for p in pp)
            + " |"
        )
    for family in ("fg_norm_static", "random_head"):
        pp = [
            next(r for r in rows if r["role"] == role and r["family"] == family) for role in ROLES
        ]
        table.append(
            f"| {family} | — | {pp[0]['numeric_bytes']} | "
            + " | ".join(f"{p['person_mean']:.5f}" for p in pp)
            + " |"
        )
    quality_table = [
        "| Политика минимальной внутренней ошибки | J mixed / вперёд / обратно | ΔE00 mixed | SLR → iPod | iPod → SLR |",
        "|---|---|---:|---:|---:|",
    ]
    for family in FAMILIES:
        pp = [policy_row(role, family, "quality") for role in ROLES]
        quality_table.append(
            f"| {family} | {' / '.join(str(p['prefix']) for p in pp)} | "
            + " | ".join(f"{p['person_mean']:.5f}" for p in pp)
            + " |"
        )
    speed = [
        "| Вариант compact | Ответ CPU, мкс: mixed / вперёд / обратно | Полное обучение + экспорт GPU, мс |",
        "|---|---|---|",
    ]
    for family in FAMILIES:
        pp = [policy_row(role, family) for role in ROLES]
        speed.append(
            f"| {family} | {' / '.join(format(p['cpu_median_us'], '.1f') for p in pp)} | "
            + " / ".join(f"{p['full_fit_gpu_ms']:.2f}" for p in pp)
            + " |"
        )
    lines = [
        "# Luma ChromaSeed-NP: сокращение до 643 параметров",
        "",
        "Удаление ненужных блоков и нулевых входов дало измеримое уменьшение размера и времени ответа. "
        "Одна голова blind4: 643 параметра, 2 886 числовых байт, 5 456 байт рабочих массивов, около 6,3 мкс на готовый вектор. "
        "Её внутренне выбранные головы 2/3/2 дают ошибки 5,77054 / 8,29435 / 8,58953. "
        "На mixed результат хуже FG 5,43865; это компромисс качества и размера, универсальная победа не установлена.",
        "",
        "NP использует уже обученные ND модели при прежних learning rate и числе шагов. "
        "Политики выбирают фиксированный J только по внутренним person-folds: минимальная ошибка либо минимальный размер при допуске +0,10 ΔE00. "
        "Допуск предварительный инженерный; он не гарантирует +0,10 на новых людях. "
        "Все 45 кандидатов, 30 решений, 135 экспортов и 18 точных FG/random контролей сохранены.",
        "",
        *table,
        "",
        "Меньше ΔE00 — лучше. Среднее по людям и затем по трём seed-ошибкам; модели не объединяются в ансамбль. "
        "Это три многократно использованных сценария оригинального TRAIN (966 строк/24 человека), с 6/16/8 людьми на стороне оценки. "
        "При переносе камер меняется состав людей. Измерялись признаки подготовленных участков кожи, не повседневные селфи.",
        "",
        *quality_table,
        "",
        f"Относительно полного исходного выхода quality: {outcomes['quality']}; compact: {outcomes['compact']}. "
        "Например, compact local4 в mixed ухудшает 5,70054 → 5,86041, а при обратном переносе 8,32429 → 8,82204. "
        "Для e2e4 выбор quality при обратном переносе ухудшает 8,02648 → 8,62946. "
        "Его compact J=1 даёт 7,76477, но это не основание задним числом менять политику на остальных сценариях.",
        "",
        "У blind4 внутренние политики совпали. Против FG разности ΔE00: +0,33189 / −0,30265 / −0,11549; "
        "описательные 95% интервалы соответственно [0,04558; 0,67546], [−1,03385; 0,47472], [−1,02685; 0,48288]. "
        "Парный bootstrap фиксированных прогнозов не учитывает неопределённость выбора и не является независимым подтверждением.",
        "",
        "**Точная оптимизация.** У первого блока состояние всегда нулевое: его три входных ряда весов удалены. "
        "В blind4 состояние не использовалось ни при обучении, ни при применении: для любого J достаточно одной J-й головы. "
        "Полный выход blind4 сохранён с 11 368 → 2 886 числовых байт и 28,1–28,4 → 6,3 мкс. "
        "Полные local4/e2e4 сохраняют тот же ответ при 11 178 байтах и примерно 19 мкс вместо 25 мкс. "
        "Исходный знаменатель расписания K сохранён; последнее неиспользуемое обновление состояния пропущено.",
        "",
        *speed,
        "",
        "CPU: один поток Ryzen 9 7900X, с нормализацией, без поиска лица, декодирования изображения и извлечения признаков. "
        "Числовые байты включают FP32 веса/нормализаторы и два uint8; family-строка и ZIP-служебные данные отдельно в CSV. "
        "Вычисления потребителя FP64 после FP32 нормализации, а не INT8-ядра. "
        "Полное обучение измерено на RTX4060, seed17, для 30 уникальных {quality, compact, full} настроек: "
        "каждая заново обучает ВСЮ исходную ND сеть, затем экспортирует префикс. "
        "Все 90 повторов точно воспроизвели исходные и сокращённые веса. Первый из трёх повторов исключён только из времени. "
        "Время включает подготовку, все исходные блоки, оптимизатор/CUDA Graph и оба экспорта; прогретый процесс/GPU-контекст. "
        "Обрезка модели здесь не доказала ускорения обучения. Цена обучения FG заново не измерялась.",
        "",
        f"**Проверка.** 135 внутренних баз,405 экспортов и76 500 строк префиксов; все45 кандидатов и30 решений пересчитаны независимо. "
        f"Все153 финальные модели,5 049 модель-преобразований и2 016 234 одиночных вызова проверены против независимого исходного ND расчёта. "
        f"Максимальное расхождение {audit['max_consumer_native_lab']:.3g} native Lab; допуск2e-8. "
        "20 поведенческих тестов проверяют расписание, чистый выход, точное удаление весов, формат и порог выбора. "
        "Новых первичных обучений нет;90 повторов выполнены для честных замеров полной стоимости.",
        "",
        "[Все префиксы](all_prefixes.csv) · [Политики](policies.csv) · [Внутренние кандидаты](inner_candidates.csv) · "
        "[Точные полные контролы](full_prefix_controls.csv) · [Парные сравнения](paired.csv) · "
        "[Аудит](audit.json) · [Замеры](runtime.json) · [Проверка](verification.json).",
        "",
        "[Протокол](../../research/chromaseed_neural_prefix_v1_protocol.md) · [Карточка](../../architecture/chromaseed_neural_prefix_model_card.md) · "
        "[Следующее решение](../../research/chromaseed_neural_prefix_next_decision.md).",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    CARD.write_text(
        """# Luma ChromaSeed-NP model card

Exact fixed-prefix exports of existing ND networks; no new primary training. One finite prepared color36 vector -> native instrument D65/10-degree Lab3. Standalone NumPy Predictor in scripts/chromaseed_neural_prefix_numpy.py takes one (36,) input; module predict takes nonempty Nx36. FP32 normalization, cached FP64 arithmetic. Input images, segmentation/detection, feature extraction and runtime/code overhead are not included in model size or latency.

First state-input rows are pruned because inference starts at zero. Blind4 exports only its selected head, since none consumes state. Preserve original schedule denominator K and clean-head semantics for other prefixes. The final unused state update is removed. All original training is still required and charged in this study, including e2e gradients from later heads. No backprop-free or training acceleration claim.

Single h16 head:643 parameters /2,886 numeric B /5,456 cached-array B /~6.3us one-thread response. Two metadata uint8 scalars count in payload; NPZ archives and family strings are separate. Blind compact/quality policies choose heads2/3/2 inside the inner folds; errors mixed5.77054 /forward8.29435 /reverse8.58953. FG5.43865/8.59700/8.70502. Mixed is worse; transfer uncertainty does not establish superiority. Complete original blind4 fits plus prefix export~304–309ms on RTX4060. Full controls and all positive/negative prefixes are retained.

Conditional inner selection follows unchanged ND rate/checkpoint decisions. Compact tolerance+0.10 applies to measured inner error only and is not a cosmetic requirement. Reused original TRAIN roles overlap, and people/camera are confounded. No ordinary-phone facial accuracy, cosmetic shade-match success, clinical use, novelty or Skolkovo eligibility established by these measurements. Existing upstream data/code licenses apply; no new assets or external source implementation adopted.

All153 exports/references,405 inner exports and2,016,234 actual final consumer calls independently verified.90 full reconstructions yield bitwise identical original and prefix weights;20 behavioral tests pass. Source and evidence are sealed; report script re-verifies read-only.

[Evidence](../benchmarks/chromaseed_neural_prefix_v1/report.md) · [Protocol](../research/chromaseed_neural_prefix_v1_protocol.md).
""",
        encoding="utf-8",
    )
    NEXT.write_text(
        """# After NP: test the actual construction cost of independent retained blocks

NP is verified progress, not broad-goal completion. Exact export reduces blind4 to one643-parameter/2,886B head and ~6.3us response, and full local/e2e inference also accelerates without changing outputs. Conditional compact selection sometimes worsens external error despite the inner+0.10 tolerance. Preserve all policies and full controls; no outer-driven grid expansion or generic accuracy win.

Next inventory and preregister a narrowly matched training-cost study: can independent local/blind blocks that are actually retained be trained without evaluating discarded blocks, while reproducing the original parameters/predictions? Existing ND local-gradient/per-block optimizer and bank-independence tests are prior work, not a first use of independent learning. Original NP charges full original training; do not retroactively change that result.

For local2/local4 preserve ORIGINAL K, selected rate/steps, original block identities, initialization, weighted row samples, noise values and per-block clipping/AdamW. Deployment-zero state is NOT training-zero state: local first blocks train with noise, so their state weights cannot simply be removed during fitting. For blind4 the selected head is independent, but keep its original initialization and shared samples. Subsetted generation must preserve the exact random-number coordinates; changing array shape can change the stream. First prove behavior on synthetic CPU/GPU fixtures and compare against full-network fits. Register original-full controls and complete fit timings, without dividing bank time or hiding preparation. E2e4 is NOT independently trainable this way: later heads send gradients to earlier ones, so retain full e2e training and do not assume equality. This study is planned, not implemented or launched.

A per-input stopping rule is a separate future question; R already tested threshold exits. Small self-corrections are not a bound on true skin-color error. No new loss, stopping selector, noise schedule or external data in the same construction-cost experiment. Current original-TRAIN-only boundary remains; no new data/images/weights/packages, agents, messages or publication. Ordinary-phone face/end-to-end high quality still unvalidated; goal active.

NP report reruns read-only after sealing. NP audit/runtime refuse writes after a verification receipt exists. Preserve NP/ND and all older bindings; never run old G/GS verifier mains, nor sealed ND audit/runtime writers.

[NP report](../benchmarks/chromaseed_neural_prefix_v1/report.md) · [ND decision](chromaseed_local_denoise_next_decision.md) · [Forum leads](chromaseed_forum_update_after_ns_2026-09-13.md).
""",
        encoding="utf-8",
    )
    test = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_chromaseed_neural_prefix.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert test.returncode == 0, test.stdout + test.stderr
    scripts = (
        "chromaseed_neural_prefix_numpy",
        "chromaseed_neural_prefix_run",
        "chromaseed_neural_prefix_audit",
        "chromaseed_neural_prefix_runtime",
        "chromaseed_neural_prefix_report",
    )
    # Preserve already frozen/measurement-bound code. These two imports have no numerical effect.
    lint_exceptions = {
        "scripts/chromaseed_neural_prefix_run.py": "F401: unused SEEDS import; primary already frozen",
        "scripts/chromaseed_neural_prefix_audit.py": "F401: unused ND_OUT import; bound by runtime",
    }
    lint = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--per-file-ignores",
            "scripts/chromaseed_neural_prefix_run.py:F401,scripts/chromaseed_neural_prefix_audit.py:F401",
            *(f"scripts/{s}.py" for s in scripts),
            "tests/test_chromaseed_neural_prefix.py",
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
                target = (p.parent / link.split("#")[0]).resolve()
                assert target.is_file() or target == verification, link
    SHORTCUT.parent.mkdir(parents=True, exist_ok=True)
    SHORTCUT.write_text(
        f"# Luma ChromaSeed-NP\n\n[Полный отчёт]({(OUT / 'report.md').as_posix()})\n\n643 параметра,2,9 КБ числовых данных,~6,3 мкс по готовым признакам. Компромисс размера/качества; селфи не проверены.\n",
        encoding="utf-8",
    )
    receipt = dict(
        passed=True,
        source_lock_sha256=source,
        selection_sha256=selected,
        results_sha256=results,
        artifact_sha256=artifacts,
        postprocess_sources={f"scripts/{s}.py": sha(ROOT / f"scripts/{s}.py") for s in scripts},
        shortcut_sha256=sha(SHORTCUT),
        tests=test.stdout,
        lint=lint.stdout,
        lint_exceptions=lint_exceptions,
        coverage=audit["counts"],
        full_fit_reconstructions=90,
        all_original_and_prefix_bitwise_equal=True,
    )
    write_json(verification, receipt)
    print(
        dict(
            passed=True,
            verification_sha256=sha(verification),
            tests=test.stdout.strip(),
            outcomes=outcomes,
        )
    )


if __name__ == "__main__":
    main()
