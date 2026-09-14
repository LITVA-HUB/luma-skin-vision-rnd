"""Seal NB negative equivalence evidence; later calls only verify."""

from __future__ import annotations

import csv
import json
import subprocess
import sys

from chromaseed_kernel_audit import js
from chromaseed_neural_blocks_run import OUT, ROOT, RUN, check_map
from skin_local_search_train import sha, write_json

CARD = ROOT / "docs/architecture/chromaseed_neural_blocks_model_card.md"
NEXT = ROOT / "docs/research/chromaseed_neural_blocks_next_decision.md"


def csv_out(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(
            {
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                for k, v in r.items()
            }
            for r in rows
        )


def main():
    lock, diag, result = [
        js(RUN / n)
        for n in ("source_lock.json", "diagnostic_source_lock.json", "diagnostic_results.json")
    ]
    audit, runtime, probe = [
        js(OUT / n) for n in ("audit.json", "runtime.json", "first_step_probe.json")
    ]
    check_map(
        {
            **lock["sources"],
            **lock["input_sha256"],
            **diag["sources"],
            **diag["input_sha256"],
            **audit["artifact_sha256"],
            **audit["dependencies"],
            **runtime["dependencies"],
        }
    )
    assert (
        audit["passed"]
        and not audit["equivalence_accepted"]
        and runtime["reconstruction_calls"] == 108
    )
    assert runtime["audit_sha256"] == sha(OUT / "audit.json")
    assert probe["script_sha256"] == sha(ROOT / "scripts/chromaseed_neural_blocks_probe.py")
    for obj in (result, audit, runtime, probe):
        assert obj["source_lock_sha256"] == sha(RUN / "source_lock.json")
    for obj in (audit, runtime):
        assert obj["diagnostic_results_sha256"] == sha(RUN / "diagnostic_results.json")
    verify = OUT / "verification.json"
    if verify.exists():
        receipt = js(verify)
        check_map({**receipt["artifact_sha256"], **receipt["postprocess_sources"]})
        assert receipt["passed"] and not receipt["equivalence_accepted"]
        print(
            dict(
                passed=True,
                mode="read-only",
                equivalence_accepted=False,
                verification_sha256=sha(verify),
            )
        )
        return
    rows = []
    for r in result["records"]:
        rows.append(
            {
                k: r[k]
                for k in (
                    "role",
                    "family",
                    "mode",
                    "prefix",
                    "seed",
                    "step",
                    "selected_step",
                    "policies",
                    "numeric_bytes",
                    "parameters",
                    "metrics",
                    "raw_comparison",
                    "full_control_comparison",
                    "np_comparison",
                )
            }
        )
    timed = []
    for r in runtime["records"]:
        timed.append(
            dict(
                role=r["role"],
                family=r["family"],
                prefix=r["prefix"],
                policies=r["policies"],
                full_ms=r["full_seconds"] * 1000,
                subset_ms=r["subset_seconds"] * 1000,
                speedup=r["speedup"],
                guards_pass=sum(p["paired_comparison"]["all_guards_pass"] for p in r["repeats"]),
                bitwise_pairs=sum(p["paired_comparison"]["bitwise_equal"] for p in r["repeats"]),
            )
        )
    csv_out(OUT / "all_checkpoints.csv", rows)
    csv_out(OUT / "paired_fit_times.csv", timed)
    subset = [r for r in result["records"] if r["mode"] == "subset"]
    groups = []
    for family in ("local2", "local4", "blind4"):
        for step in (512, 2048, 8192):
            rr = [r for r in subset if r["family"] == family and r["step"] == step]
            groups.append(
                dict(
                    family=family,
                    step=step,
                    models=len(rr),
                    exact=sum(r["raw_comparison"]["bitwise_equal"] for r in rr),
                    guard_pass=sum(r["full_control_comparison"]["all_guards_pass"] for r in rr),
                    maximum_weight_drift=max(r["raw_comparison"]["max_weight_drift"] for r in rr),
                    maximum_prediction_drift=max(
                        r["full_control_comparison"]["max_prediction_drift"] for r in rr
                    ),
                )
            )
    csv_out(OUT / "equivalence_groups.csv", groups)
    summary = dict(
        equivalence_accepted=False,
        primary_exit_code=1,
        coverage=audit["counts"],
        groups=groups,
        timings=timed,
        maximum_raw_weight_drift=max(r["raw_comparison"]["max_weight_drift"] for r in subset),
        maximum_prediction_drift=max(
            r["full_control_comparison"]["max_prediction_drift"] for r in subset
        ),
        maximum_person_error_change=max(
            abs(r["full_control_comparison"]["error_difference"]) for r in subset
        ),
        sources=len(lock["sources"]),
        inputs=len(lock["input_sha256"]),
        goal_status="active",
    )
    write_json(OUT / "summary.json", summary)
    lines = [
        "# Luma ChromaSeed-NB: сокращение обучения не прошло проверку эквивалентности",
        "",
        "Обучение только сохраняемых блоков не принимается как точная замена прежней процедуры. "
        "Все27 пакетов/81 траектория обучены до8192 шагов, получены243 checkpoint-модели. "
        "19 из162 сокращённых checkpoint-вариантов вышли за исходные допуски;63 совпали побитово. "
        "Среди54 настроек на ранее выбранном числе шагов53 проходят сравнение с новым полным пакетом. "
        "Прежние NP модели сохранены как рабочий вариант.",
        "",
        "Первичный процесс завершился с exit1 на проверке весов mixed/local2/8192 после завершения всех обучений. "
        "Код, данные и допуски не менялись. Отдельная диагностика рассчитала все243 варианта, включая неудачные, "
        "в отдельных файлах; это завершение исследования провала, а не успешное прохождение первичного допуска.",
        "",
        "| Семейство | Шаги | Побитово /18 | В допуске /18 | Максимальный разрыв весов | Максимальный разрыв native Lab |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['family']} | {r['step']} | {r['exact']} | {r['guard_pass']} | {r['maximum_weight_drift']:.6g} | {r['maximum_prediction_drift']:.6g} |"
        for r in groups
    ]
    lines += [
        "",
        "Допуски прежние: веса atol/rtol2e-6, максимальный разрыв native Lab0,002, "
        "изменение среднего по людям ΔE00 не более0,002. Максимумы фактически: веса0,20007, "
        "прогноз3,35671 native Lab и изменение среднего ΔE00 0,07211. "
        "Это не новая точность модели: ожидалось совпадение с полным обучением, а поздние точки не использовались для выбора.",
        "",
        "**Причина на первом шаге.** В18 контрольных пробах исходные выходы совпадают побитово. "
        "В11 пробах градиенты различаются до7,45e-9, а обновлённые веса — до4,66e-10. "
        "Если подать сокращённому оптимизатору точно тот же срез полного градиента, все18 обновлений и первых моментов совпадают. "
        "Таким образом, в этих пробах различие локализовано в вычислении градиента при другой форме пакета. "
        "Это не доказательство конкретного механизма каждого позднего расхождения; конкретные CUDA-ядра не профилировались.",
        "",
        "Даже новый полный пакет из трёх seed не всегда повторяет прежние одиночные NP fits: "
        "из81 сравнения выбранных точек с NP75 проходят допуски. "
        "Шесть отказов относятся к трём базовым случаям, каждый повторён как full и full-prefix subset. "
        "Детерминированность внутри одной формы вычисления не гарантирует совпадения после изменения формы пакета.",
        "",
        "| Сценарий | Семейство / J | Полное обучение, мс | Только блоки, мс | Ускорение |",
        "|---|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r['role']} | {r['family']} / {r['prefix']} | {r['full_ms']:.2f} | {r['subset_ms']:.2f} | {r['speedup']:.3f}× |"
        for r in timed
    ]
    lines += [
        "",
        "108 полных одиночных построений:18 настроек x3 пары x2 процедуры, seed17; первый прогрев исключён только из времени. "
        "В этих коротких выбранных настройках54/54 пары проходят допуски,24 совпадают побитово; это не отменяет провал других seed/длинных траекторий. "
        "Ускорение лишь0,976–1,084×. Для blind около269–275 →251–255мс, с подготовкой, выборками, шумом и экспортом. "
        "Затраты пакета не делились на число моделей. Ответ/размер использует прежний NP потребитель; нового выигрыша инференса нет.",
        "",
        "**Проверки.** Независимый аудит проверил все243 модели,3 202 254 строки прогнозов, "
        "2 134 836 одиночных вызовов сокращённого потребителя,8019 преобразований и972 сводки доз. "
        f"Максимальный разрыв независимого расчёта с сохранённым собственным прогнозом {audit['maximum_native_lab_audit_drift']:.3g} native Lab. "
        "20 коротких поведенческих тестов проходят; они не заменяют провалившуюся проверку8192 шагов. "
        "Использовались только исходные TRAIN-признаки966 кадров/24 людей; сценарии повторяются, камеры и люди смешаны как факторы. "
        "Обычные селфи и подбор косметики не валидированы.",
        "",
        "Новая просьба пользователя — больше примеров и дольше учить лёгкую модель. "
        "Следующая серия будет продолжать проверенную643-параметрическую NP голову с большим набором синтетических вариаций исходных обучающих строк. "
        "Это отдельная проверка качества, с неизменными людьми на стороне оценки и фиксированной формой обучающего пакета.",
        "",
        "[Все checkpoint-варианты](all_checkpoints.csv) · [Допуски](equivalence_groups.csv) · [Полные замеры](paired_fit_times.csv) · "
        "[Проба первого шага](first_step_probe.json) · [Аудит](audit.json) · [Верификация](verification.json).",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    CARD.write_text(
        """# ChromaSeed-NB construction study card

NB is a rejected exact-construction optimization, not a new accepted skin model. It trains only independent retained local/blind heads with original K/noise/block identities. Deployment remains the unchanged NP consumer and payload.19/162 checkpoint pairs fail original guards; one of54 selected-step pairs fails against fresh full control.63 pairs are bitwise exact. Fixed NP weights are the fallback.

Maximum raw weight drift0.200071, native Lab prediction drift3.356707, person DeltaE00 change0.072105. First-step probes isolate small differences to gradient computation when batch shape changes; identical-gradient optimizer injection removes them in all18 probes. Long nonlinear training can magnify differences, but no universal mechanism for every failure is proved. Three-seed full-bank versus old singleton differences also occur.

Paired seed17 selected-setting reconstruction times improve at most~8.4%, sometimes worsen.108 complete calls retain all correctness checks; short timing correctness does not establish general long-training equivalence. Existing TRAIN-only provenance applies. No ordinary-phone face/shade-match accuracy or Skolkovo qualification established. Goal active.

[Evidence](../benchmarks/chromaseed_neural_blocks_v1/report.md).
""",
        encoding="utf-8",
    )
    NEXT.write_text(
        """# After NB: user's longer-training / more-examples request

User steered the active task: «поробуй лекгой модели дать больше примеров долго обучать». NB's immutable negative equivalence evidence is preserved; do not keep optimizing dropped-block replay ahead of this request. Accepted NP643-parameter heads remain the starting point.

Next preregister LT: continue the exact NP blind-head warm start on each legal fit split, with original fit-only normalizers. Increase synthetic variant count and training steps while retaining no-augmentation/short/baseline controls. Original measured people/labels do not multiply when variants are added. No external data acquisition authorized by the selected implementation; no old held/test borrowing. Earlier palette pretraining and A's16 affine copies already exist, so do not claim first augmentation. Explicitly distinguish a larger variation bank from more measured people.

The planned larger comparison is 0/16/256 added bounded color variations per original fit row, same original-image weights and augmentation probability; longer checkpoints up to131072 updates, with inner-only LR/checkpoint/policy selection and intermediate errors. Both variation count and training duration must have controls. Warm starts for inner folds come only from the matching original ND inner models, never from full-role models. Include the original NP prediction as step0 and preserve full upstream training cost when timing.

NB proves that changing batch shape can alter longer learned trajectories despite deterministic execution. Keep a fixed training-bank shape/slot order and original schedule horizon for exact replay, and charge the full bank rather than dividing its cost. Any later standalone optimization needs its own correctness evidence. First implement/test/freeze the protocol and data transform; LT is planned, not implemented/launched at NB sealing.

NB report re-verifies read-only after sealing. Never rerun frozen NB primary or bound diagnostic/audit/probe/runtime writers; preserve all older NP/ND sources and outputs, never G/GS mains. Original TRAIN-only/no new images/weights/packages/agents/messages/publication boundary persists. Ordinary-phone face/end-to-end high quality unvalidated; broad goal active, not blocked.
""",
        encoding="utf-8",
    )
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_chromaseed_neural_blocks.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert tests.returncode == 0, tests.stdout + tests.stderr
    names = ("fit", "run", "diagnose", "audit", "probe", "runtime", "report")
    files = [f"scripts/chromaseed_neural_blocks_{n}.py" for n in names]
    lint = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *files, "tests/test_chromaseed_neural_blocks.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert lint.returncode == 0, lint.stdout + lint.stderr
    artifacts = dict(audit["artifact_sha256"])
    for p in (*OUT.glob("*"), CARD, NEXT):
        if p.is_file() and p.name != "verification.json":
            artifacts[p.relative_to(ROOT).as_posix()] = sha(p)
    receipt = dict(
        passed=True,
        equivalence_accepted=False,
        primary_exit_code=1,
        source_lock_sha256=sha(RUN / "source_lock.json"),
        diagnostic_results_sha256=sha(RUN / "diagnostic_results.json"),
        artifact_sha256=artifacts,
        postprocess_sources={f: sha(ROOT / f) for f in files},
        tests=tests.stdout,
        lint=lint.stdout,
        counts=audit["counts"],
        complete_reconstruction_calls=108,
    )
    write_json(verify, receipt)
    print(
        dict(
            passed=True,
            equivalence_accepted=False,
            verification_sha256=sha(verify),
            tests=tests.stdout.strip(),
        )
    )


if __name__ == "__main__":
    main()
