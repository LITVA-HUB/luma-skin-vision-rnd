"""WE local result report and immutable read-only verification after sealing."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
from chromaseed_kernel_audit import js, nz
from chromaseed_refine_audit import error_summary
from chromaseed_widen_early_run import (
    CAPS,
    OUT,
    ROOT,
    RUN,
    WIDE,
    WIDE_OUT,
    check_map,
    load_data,
)
from skin_local_search_train import sha, write_json

ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")


def main():
    seal = OUT / "verification.json"
    if seal.exists():
        v = js(seal)
        check_map(
            {**v["sources"], **v["inputs"], **v["postprocess_sources"], **v["artifact_sha256"]}
        )
        print("WE READ-ONLY VERIFIED", sha(seal), flush=True)
        return
    lock, selection, result = [
        js(RUN / n) for n in ("source_lock.json", "selections.json", "results.json")
    ]
    audit, runtime = js(OUT / "audit.json"), js(OUT / "runtime.json")
    assert (
        audit["passed"]
        and runtime["passed"]
        and audit["results_sha256"] == runtime["results_sha256"] == sha(RUN / "results.json")
    )
    assert runtime["audit_sha256"] == sha(OUT / "audit.json")
    check_map(
        {
            **lock["sources"],
            **lock["input_sha256"],
            **audit["artifact_sha256"],
            **audit["dependencies"],
            **runtime["dependencies"],
        }
    )
    commands = [
        [sys.executable, "-m", "pytest", "tests/test_chromaseed_widen_early.py", "-q"],
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "scripts/chromaseed_widen_early_fit.py",
            "scripts/chromaseed_widen_early_run.py",
            "scripts/chromaseed_widen_early_audit.py",
            "scripts/chromaseed_widen_early_runtime.py",
            "scripts/chromaseed_widen_early_report.py",
            "tests/test_chromaseed_widen_early.py",
        ],
    ]
    checks = []
    for command in commands:
        p = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        checks.append(
            dict(command=command[1:], exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
        )
        assert p.returncode == 0, p.stdout + p.stderr
    write_json(OUT / "checks.json", dict(passed=True, checks=checks))
    data = load_data()
    old_result = js(WIDE / "results.json")
    old_summary = js(WIDE_OUT / "summary.json")
    old_runtime = js(WIDE_OUT / "runtime.json")
    groups = [
        ("np", "np"),
        ("tiny", "tiny"),
        *((kind, v) for v in CAPS for kind in ("wide", "new")),
    ]
    rows = []
    people = {}
    for kind, variant in groups:
        recs = [r for r in result["records"] if r["kind"] == kind and r["variant"] == variant]
        rt = [r for r in runtime["records"] if r["kind"] == kind and r["variant"] == variant]
        row = dict(
            kind=kind,
            variant=variant,
            parameters=recs[0]["parameters"],
            numeric_bytes=recs[0]["numeric_bytes"],
            median_cpu_us=float(np.median([r["median_us"] for r in rt])),
            roles={},
        )
        for role in ROLES:
            rr = [r for r in recs if r["role"] == role]
            assert len(rr) == 3
            per = []
            for r in rr:
                saved = nz(RUN / "evaluated" / role / f"{r['name']}.npz")
                ix = saved["row_indices"]
                per.append(
                    error_summary(
                        saved["predictions"][0],
                        data["target"][ix],
                        data["patient"][ix],
                        data["site"][ix],
                    )[1]
                )
            people[kind, variant, role] = np.mean(per, axis=0)
            recipe = next(
                r
                for r in (runtime if kind == "new" else old_runtime)["recipes"]
                if r["role"] == role and r["variant"] == variant
            )
            row["roles"][role] = dict(
                delta_e00=float(np.mean([r["metrics"]["person_mean"] for r in rr])),
                seed_errors=[r["metrics"]["person_mean"] for r in rr],
                step=rr[0]["step"],
                lr=rr[0]["lr"],
                unchanged_setting=rr[0]["unchanged_setting"],
                complete_three_seed_recipe_seconds=recipe[
                    "median_reconstruction_and_validation_seconds"
                ],
                training_timing_source="WE fresh" if kind == "new" else "WIDE historical",
            )
        rows.append(row)
    comparisons = []
    for vi, variant in enumerate(CAPS):
        for ri, role in enumerate(ROLES):
            diff = people["new", variant, role] - people["wide", variant, role]
            draws = np.random.default_rng(317003 + 100 * vi + ri).integers(
                0, len(diff), (20000, len(diff))
            )
            ci = np.quantile(diff[draws].mean(axis=1), [0.025, 0.975])
            comparisons.append(
                dict(
                    variant=variant,
                    role=role,
                    people=len(diff),
                    new_minus_wide=float(diff.mean()),
                    conditional_paired_person_95=ci.tolist(),
                )
            )
    overall = {}
    for role in ROLES:
        new = [r for r in result["records"] if r["role"] == role and "overall" in r["policies"]]
        old = [r for r in old_result["records"] if r["role"] == role and "overall" in r["policies"]]
        assert len(new) == len(old) == 3
        c = selection["roles"][role]["policies"]["overall"]
        overall[role] = dict(
            variant=c["variant"],
            step=c["step"],
            lr=c["lr"],
            inner_delta_e00=c["clean"],
            new_delta_e00=float(np.mean([r["metrics"]["person_mean"] for r in new])),
            wide_delta_e00=float(np.mean([r["metrics"]["person_mean"] for r in old])),
        )
    banks = [js(p) for p in sorted((RUN / "inner").rglob("receipt.json"))] + [
        js(p) for p in sorted((RUN / "final").rglob("receipt.json"))
    ]
    assert len(banks) == 108 + audit["counts"]["final_banks"]
    presentations = sum(b["steps"] * 64 * 6 for b in banks)
    counts = dict(
        better=sum(c["new_minus_wide"] < -1e-9 for c in comparisons),
        worse=sum(c["new_minus_wide"] > 1e-9 for c in comparisons),
        same=sum(abs(c["new_minus_wide"]) <= 1e-9 for c in comparisons),
    )
    chosen = [selection["roles"][role]["policies"][v] for role in ROLES for v in CAPS]
    summary = dict(
        rows=rows,
        comparisons=comparisons,
        overall=overall,
        comparison_counts=counts,
        selected_steps=[c["step"] for c in chosen],
        selected_rates=[c["lr"] for c in chosen],
        old_fg=old_summary["historical_fg"],
        research_banks=len(banks),
        research_trajectories=6 * len(banks),
        research_presentations=presentations,
        research_bank_seconds=sum(b["full_bank_seconds"] for b in banks),
        research_workflow_seconds=js(RUN / "job.json")["seconds"],
        audit_counts=audit["counts"],
        runtime_counts={
            k: runtime[k]
            for k in (
                "upstream_singleton_fits",
                "continuation_banks",
                "selected_payloads_bitwise",
                "all_bank_payloads_bitwise",
            )
        },
        limitations="Historically reused original TRAIN roles; tiny grid unchanged; conditional bootstrap does not correct adaptive reuse or establish ordinary-phone accuracy.",
    )
    write_json(OUT / "summary.json", summary)
    lines = [
        "# Luma ChromaSeed: точная настройка обучения крупных моделей",
        "",
        f"Расширение скоростей обучения и проверка более ранних остановок дали {counts['better']} улучшений, {counts['worse']} ухудшений и {counts['same']} неизменных результатов в 12 сравнениях с прежними настройками тех же крупных моделей. Выбор настроек сделан внутри обучающей части и сохранён до внешней оценки.",
        "",
        "Архитектура и размер моделей не менялись. Новый диапазон: 0,00001 / 0,00003 / 0,0001 / 0,001; ранние точки 32 и 128 шагов, далее 512 и 2048. Старые точки 8192 шагов для прежних скоростей сохранены. Все цифры ниже — ошибка ΔE00, **меньше лучше**, средняя по людям и затем по трём отдельным запускам.",
        "",
        "| Вариант | Параметры | Модель, КБ | Смешанная | SLR → iPod | iPod → SLR | CPU, мкс |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        label = ("WE " if r["kind"] == "new" else "WIDE " if r["kind"] == "wide" else "") + r[
            "variant"
        ]
        errors = " | ".join(f"{r['roles'][k]['delta_e00']:.4f}" for k in ROLES)
        lines.append(
            f"| {label} | {r['parameters']:,} | {r['numeric_bytes'] / 1000:.2f} | {errors} | {r['median_cpu_us']:.1f} |"
        )
    lines.extend(
        [
            "",
            "NP и tiny — фиксированные прежние контроли. WIDE — прежние настройки данного размера; WE — настройки после расширения поиска. Tiny не проходил новый подбор, поэтому таблица не доказывает чистый эффект размера. Исторический сильный контроль FG на тех же ролях: "
            + " / ".join(f"{old_summary['historical_fg'][k]:.4f}" for k in ROLES)
            + " ΔE00.",
            "",
            "## Что выбрано внутри обучения",
            "",
            "| Роль | Модель | Шаги | Скорость | Ошибка WIDE policy | Ошибка WE policy |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for role, c in overall.items():
        lines.append(
            f"| {role} | {c['variant']} | {c['step']} | {c['lr']} | {c['wide_delta_e00']:.4f} | {c['new_delta_e00']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Общий выбор модели — отдельное решение по внутренним данным. Нельзя подменять его более красивой строкой внешней таблицы. Все 222 оценки кандидатов и 15 решений доступны в selections.json.",
            "",
            "| Модель WE | Шаги: смешанная / SLR→iPod / iPod→SLR | Скорости | Полная сборка трёх запусков, с |",
            "| --- | --- | --- | --- |",
        ]
    )
    for r in [v for v in rows if v["kind"] == "new"]:
        steps = " / ".join(str(r["roles"][k]["step"]) for k in ROLES)
        rates = " / ".join(str(r["roles"][k]["lr"]) for k in ROLES)
        times = " / ".join(
            f"{r['roles'][k]['complete_three_seed_recipe_seconds']:.3f}" for k in ROLES
        )
        lines.append(f"| {r['variant']} | {steps} | {rates} | {times} |")
    lines.extend(
        [
            "",
            "Стоимость включает три исходных обучения ND, экспорт NP, весь пакет из шести продолжений и проверку весов. Время не делится на шесть. Сравнение со старым временем обучения использует исторические замеры WIDE. Ответ CPU включает обработку всех 64 участков; декодирование фото и извлечение признаков исключены.",
            "",
            "## Проверка и ограничения",
            "",
            f"Исследование выполнило {summary['research_banks']} пакетов / {summary['research_trajectories']} траекторий и {presentations:,} предъявлений примеров. Это повторения существующих 966 наблюдений 24 людей. Полное время банков: {summary['research_bank_seconds']:.1f} с; первичный процесс с сохранением и оценкой: {summary['research_workflow_seconds']:.1f} с.",
            "",
            f"Независимо проверены 3240 внутренних моделей / 612000 ответов, все решения и 1 186 020 вызовов конечного потребителя. 432 старых контрольных экспорта и 1080 повторных контрольных моделей совпали побитно. Максимальное численное расхождение: {audit['maximum_native_lab']:.3g} Lab. Повторены 108 исходных обучений, 108 выбранных и 216 полных банковых экспортов совпали побитно. Четыре регрессионных теста и линтер прошли.",
            "",
            "Группы людей и камер переиспользованы из прошлых исследований. Внешние группы содержат 6, 16 и 8 людей; интервалы в summary.json условны на сделанном выборе и не устраняют смещение от многократного исследования данных. Обучение на меньшей скорости ограничено 2048 шагами: выбор верхней границы означает, что более длинный прогон ещё не проверен. Обычные селфи покупателей и полный путь от фотографии до подбора косметики здесь не валидированы.",
            "",
            "P8 остаётся с завершённым первичным обучением и отложенным аудитом. Все старые результаты и отрицательные сравнения сохранены; новые данные и внешние веса не использовались.",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    card = ROOT / "docs/architecture/chromaseed_widen_early_model_card.md"
    card.write_text(
        "# ChromaSeed WE model card\n\nUnchanged WIDE patch/color regression architecture and consumer. WE changes only optimization rate pairs and candidate checkpoints. Model capacities30915/60611/110979/832259, FP32 exports and cached FP64 CPU arithmetic. No ensemble inference. Inputs are color36 plus64x18 skin-patch descriptors; output instrument-native D65/10-degree Lab.\n\nOriginal TRAIN966rows24people only; person-disjoint inner folds, historically reused outer roles. Tiny remains a fixed old control. See ../benchmarks/chromaseed_widen_early_v1/report.md and summary.json for the full selected outcomes, adverse cases and measured construction/response costs. Synthetic affine stress is target-invariance by assumption. No ordinary-phone face/end-to-end quality or patent/Skolkovo claim.\n",
        encoding="utf-8",
    )
    decision = ROOT / "docs/research/chromaseed_widen_early_next_decision.md"
    boundary = sum(c["step"] == 2048 and c["lr"] in (0.00001, 0.00003) for c in chosen)
    decision.write_text(
        f"# Next quality decision after WE\n\nPrevious turn is verified progress; broad goal active. Latest explicit user steering prioritizes quality, allows roughly five million parameters and requests revisiting previously unsuccessful architectures at larger scale. {boundary}/12 selected larger-capacity WE settings reach the2048 boundary at new lower rates. Preserve that optimization evidence without treating it as a mandate to repeat another small-model schedule study.\n\nNext preregister a matched scaling comparison of patch pooling, four-pass soft attention and dynamic patch selection, with an ordinary widened model as a control. Include small versions under the same training and warm-start rules so a change of initialization or schedule is not mislabeled a pure size effect. The earlier R, PatchVotes and WIDE implementations are prior work, not newly invented mechanisms. The approximately five-million-parameter series is not yet trained or validated by this WE receipt. Use fit-only selection, preserve negative curves and measure actual memory and runtime. Longer lower-rate training remains a secondary question; any replay of WE must retain its original8192 horizon and optimizer prefix.\n\nWE primary/audit/runtime/report/card/decision are sealed; only report.py may reverify read-only. Preserve WIDE/P8/WA/LT/NP/ND and older writers. P8 audit/runtime remain explicitly deferred. Only original TRAIN, no new assets/delegation/publication. Ordinary-phone face quality is still unvalidated; software checks cannot complete the full goal.\n",
        encoding="utf-8",
    )
    post = {
        p: sha(ROOT / p)
        for p in (
            "scripts/chromaseed_widen_early_audit.py",
            "scripts/chromaseed_widen_early_runtime.py",
            "scripts/chromaseed_widen_early_report.py",
            card.relative_to(ROOT).as_posix(),
            decision.relative_to(ROOT).as_posix(),
        )
    }
    artifacts = dict(audit["artifact_sha256"])
    for name in ("audit.json", "runtime.json", "checks.json", "summary.json", "report.md"):
        p = OUT / name
        artifacts[p.relative_to(ROOT).as_posix()] = sha(p)
    value = dict(
        passed=True,
        source_lock_sha256=sha(RUN / "source_lock.json"),
        selection_sha256=sha(RUN / "selections.json"),
        results_sha256=sha(RUN / "results.json"),
        sources=lock["sources"],
        inputs=lock["input_sha256"],
        postprocess_sources=post,
        artifact_sha256=artifacts,
        classification="verified progress; broad quality goal active",
    )
    check_map({**value["sources"], **value["inputs"], **post, **artifacts})
    write_json(seal, value)
    print("WE SEALED", sha(seal), flush=True)
    print("WE QUALITY", overall, "comparisons", counts, flush=True)


if __name__ == "__main__":
    main()
