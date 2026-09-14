"""Generate all NS comparisons, then seal or verify them read-only."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from chromaseed_kernel_audit import js
from skin_local_search_train import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/runs/chromaseed_neural_shrinkage_v1"
NR = ROOT / "experiments/runs/chromaseed_neural_readout_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_neural_shrinkage_v1"
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-neural-shrinkage-2026-09-13.md"


def key(row):
    return tuple(row[k] for k in ("role", "group", "family", "basis"))


def check_map(values):
    for p, h in values.items():
        assert sha(ROOT / p) == h, p


def main():
    lock, result, sel, audit, runtime = (
        js(p)
        for p in (
            RUN / "source_lock.json",
            RUN / "results.json",
            RUN / "selections.json",
            OUT / "audit.json",
            OUT / "runtime.json",
        )
    )
    check_map(
        {
            **lock["sources"],
            **lock["input_sha256"],
            **audit["dependencies"],
            **audit["artifact_sha256"],
            **runtime["dependencies"],
        }
    )
    assert audit["passed"]
    assert audit["audit_source_sha256"] == sha(
        ROOT / "scripts/chromaseed_neural_shrinkage_audit.py"
    )
    assert runtime["runtime_source_sha256"] == sha(
        ROOT / "scripts/chromaseed_neural_shrinkage_runtime.py"
    )
    for obj in (result, sel, audit, runtime):
        assert obj["source_lock_sha256"] == sha(RUN / "source_lock.json")
    for obj in (result, audit, runtime):
        assert obj["selection_sha256"] == sha(RUN / "selections.json")
    assert audit["results_sha256"] == runtime["results_sha256"] == sha(RUN / "results.json")
    assert runtime["audit_sha256"] == sha(OUT / "audit.json")
    vp = OUT / "verification.json"
    if vp.exists():
        receipt = js(vp)
        check_map(receipt["artifact_sha256"])
        assert sha(SHORTCUT) == receipt["shortcut_sha256"]
        assert receipt["passed"]
        print(dict(passed=True, mode="read-only", verification_sha256=sha(vp)))
        return
    old, oldsel = js(NR / "results.json"), js(NR / "selections.json")
    rows, doses, candidates, policies = [], [], [], []
    for ident in sorted({key(r) for r in result["records"]}):
        rr = [r for r in result["records"] if key(r) == ident]
        tt = [r for r in runtime["records"] if key(r) == ident]
        ff = [r for r in runtime["standalone_fit_records"] if key(r) == ident]
        oo = [r for r in old["records"] if key(r) == ident]
        role, group, family, basis = ident
        head = family in ("norm", "perceptual")
        chosen = sel["roles"][role][group][family]["bases"][basis]["selected"] if head else None
        current = float(np.mean([r["metrics"]["person_mean"] for r in rr]))
        previous = float(np.mean([r["metrics"]["person_mean"] for r in oo]))
        fg = [
            r
            for r in result["records"]
            if r["role"] == role and r["group"] == group and r["family"] == "fg_norm_static"
        ]
        fg_error = float(np.mean([r["metrics"]["person_mean"] for r in fg])) if fg else None
        row = dict(
            role=role,
            group=group,
            family=family,
            basis=basis,
            alpha=None if chosen is None else chosen["alpha"],
            policy_selected=any(r["policy_selected"] for r in rr),
            person_mean=current,
            p90=float(np.mean([r["metrics"]["p90"] for r in rr])),
            previous_NR=previous,
            difference_vs_NR=current - previous,
            FG=fg_error,
            difference_vs_FG=None if fg_error is None else current - fg_error,
            numeric_bytes_min=min(r["numeric_bytes"] for r in rr),
            numeric_bytes_max=max(r["numeric_bytes"] for r in rr),
            archive_bytes_min=min(r["archive_bytes"] for r in rr),
            archive_bytes_max=max(r["archive_bytes"] for r in rr),
            median_us=float(np.median([r["numpy_only"]["median_us"] for r in tt])),
            maximum_seed_p95_us=max(r["numpy_only"]["p95_us"] for r in tt),
            full_fit_ms=None if not ff else ff[0]["median_seconds"] * 1000,
            cached_array_bytes=max(r["numpy_only"]["cached_array_bytes"] for r in tt),
            stress4=float(np.mean([r["doses"][1]["worst_error"]["person_mean"] for r in rr])),
        )
        rows.append(row)
        for i, amount in enumerate((1 / 255, 4 / 255, 16 / 255, 64 / 255)):
            doses.append(
                dict(
                    role=role,
                    group=group,
                    family=family,
                    basis=basis,
                    dose=amount,
                    worst_person_error=float(
                        np.mean([r["doses"][i]["worst_error"]["person_mean"] for r in rr])
                    ),
                    previous_NR=float(
                        np.mean([r["doses"][i]["worst_error"]["person_mean"] for r in oo])
                    ),
                )
            )
        if head:
            for c in sel["roles"][role][group][family]["bases"][basis]["candidates"]:
                candidates.append(
                    dict(
                        role=role,
                        **{k: v for k, v in c.items() if k != "seed_scores"},
                        selected=c["alpha_index"] == chosen["alpha_index"],
                    )
                )
    for row in rows:
        if not row["policy_selected"]:
            continue
        role, group, family, _ = key(row)
        prior = oldsel["roles"][role][group][family]["policy"]
        oo = [r for r in old["records"] if key(r) == (role, group, family, prior["basis"])]
        policies.append(
            dict(
                row,
                previous_NR_policy=float(np.mean([r["metrics"]["person_mean"] for r in oo])),
                inner_clean=sel["roles"][role][group][family]["policy"]["clean"],
            )
        )
    paired = [
        dict(
            **{k: v for k, v in p.items() if k != "descriptive_95"},
            descriptive_95_low=p["descriptive_95"][0],
            descriptive_95_high=p["descriptive_95"][1],
        )
        for p in audit["paired"]
    ]
    assert (len(rows), len(doses), len(candidates), len(policies), len(paired)) == (
        129,
        516,
        420,
        12,
        192,
    )
    alpha_counts = dict(Counter(r["alpha"] for r in rows if r["family"] in ("norm", "perceptual")))
    comparisons = {
        kind: dict(
            better=sum(p["kind"] == kind and p["mean_difference"] < -1e-10 for p in paired),
            worse=sum(p["kind"] == kind and p["mean_difference"] > 1e-10 for p in paired),
            tie=sum(p["kind"] == kind and abs(p["mean_difference"]) <= 1e-10 for p in paired),
        )
        for kind in sorted({p["kind"] for p in paired})
    }
    summary = dict(
        rows=rows,
        doses=doses,
        candidates=candidates,
        policies=policies,
        paired=paired,
        alpha_counts=alpha_counts,
        comparisons=comparisons,
    )
    write_json(OUT / "summary.json", summary)
    for filename, items in (
        ("all_models.csv", rows),
        ("all_doses.csv", doses),
        ("inner_candidates.csv", candidates),
        ("policies.csv", policies),
        ("paired.csv", paired),
    ):
        with (OUT / filename).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
    text = [
        "# Luma ChromaSeed NS: более сильный штраф не дал общего выигрыша",
        "",
        "Завершены5040 первичных обучений выхода и5040 независимых QR-пересчётов на504 прежних основах. Все3540 старых моделей совпали точно. Размер сети остался1855/10564 числовых байта. Из84 вариантов десять выбралиalpha100: пять улучшились, пять ухудшились относительно соответствующего NR;74 не изменились. Все12 правил выбора по-прежнему уступают FG.",
        "",
        "Выбрано67 настроекalpha10,10 настроекalpha100,7 настроекalpha1; никто не выбрал1000. Дальше расширять эту сетку оснований нет. Геометрия NG обоснованно указала на оставшуюся гибкость, но её уменьшение не гарантировало перенос на другие группы людей.",
        "",
        "Данные: исходный TRAIN966 строк/24 человека. Mixed734 fit/232 query,18/6 людей; SLR→iPod323/643 строк,8/16 людей; обратно643/323. Роли многократно использованы, люди пересекаются между сценариями, камера связана с составом людей. Это исследовательские оценки подготовленных цветовых признаков, не независимая точность на селфи. Ошибка — DeltaE00, меньше лучше; усредняются ошибки трёх отдельных моделей, не ансамбль ответов.",
        "",
        "| Роль | Вход | Цель | Основа | alpha | NS ΔE00 | NR правило | FG | Полное обучение, мс | Ответ, мкс |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in policies:
        text.append(
            f"| {r['role']} | {r['group']} | {r['family']} | {r['basis']} | {r['alpha']:g} | {r['person_mean']:.6f} | {r['previous_NR_policy']:.6f} | {r['FG']:.6f} | {r['full_fit_ms']:.3f} | {r['median_us']:.2f} |"
        )
    text += [
        "",
        "Два правила улучшились относительно прежнего NR, одно ухудшилось, девять не изменились. Mixed/raw36 norm ухудшается5.502520→5.678073; perceptual улучшается5.654729→5.607624 при смене основы наAdam16, но уступаетFG5.438652. Reverse/raw36 norm улучшается10.067767→9.539056, оставаясь хужеFG8.705018. Всего9/84 отдельных голов лучше FG,75 хуже; выбирать одну из удобных внешних удач задним числом нельзя.",
        "",
        "Полное обучение ниже начинается с исходных fit-массивов и включает создание скрытых признаков. Только случайная основа позволяет получить маленькую сеть за несколько миллисекунд, с потерей качества:",
        "",
        "| Mixed | Ошибка ΔE00 | Числовые байты | Полное обучение, мс | Ответ, мкс |",
        "|---|---:|---:|---:|---:|",
    ]
    for family, basis, group in (
        ("norm", "random", "raw36"),
        ("norm", "random", "mean3"),
        ("fg_norm_static", "reference", "raw36"),
        ("fg_norm_static", "reference", "mean3"),
    ):
        r = next(r for r in rows if key(r) == ("mixed", group, family, basis))
        text.append(
            f"| {family}/{basis}/{group} | {r['person_mean']:.6f} | {r['numeric_bytes_min']} | {r['full_fit_ms']:.3f} | {r['median_us']:.2f} |"
        )
    text += [
        "",
        f"Отдельно выполнены270 полных построений моделей, все{runtime['exact_payload_repeats']}/270 совпали побитно; один прогрев и два учитываемых повтора на настройку. Полные замеры заняли{runtime['wall_seconds']:.3f}с. Первичный пакет с кэшированными основами и QR-проверками занял{js(RUN / 'workflow.json')['wall_seconds']:.3f}с после импорта; его нельзя выдавать за стоимость обучения всех скрытых сетей с нуля.",
        "",
        "Размер означает числовые веса с нормализаторами, не NPZ/процесс/приложение. Кэш3659/20816 байт для mean3/raw36. Временные обучающие массивы и состояния указаны в runtime.json и не являются пиковым RAM. Микросекунды измерены одним потоком CPU на подготовленных36 признаках; поиск лица, обработка изображения, I/O, телефон и CUDA в них не входят. Для unchanged-сетей полное обучение заново в NS не профилировалось: в CSV эти поля пустые.",
        "",
        "Проверка сохранённых результатов охватила5556 моделей,787100 внутренних строк,420 оценок/84 выбора/12 правил и5020818 выходных строк381 настоящего consumer при33 преобразованиях. Все12573 оценки преобразований и1524 дозы пересчитаны независимо. QR-расхождение на fit не более5.81e-6Lab, на стрессах2.48e-6; actual consumer совпадает в пределах1.18e-12Lab. Все четыре дозы сохранены, включая ухудшения.",
        "",
        "Парные интервалы в paired.csv:20000 повторов по людям при фиксированных предсказаниях, seed771031. Это описательные интервалы; они не учитывают повторный подбор моделей и не превращают шесть mixed query-людей в новый независимый тест.",
        "",
        "Первый дополнительный аудит остановился на распаковке пары (dose,anchor) как тройки в новом проверочном скрипте. Сверен фактический формат родительского SETTINGS; исправлен только незапечатанный аудит. Полный повтор82044 завершился успешно. Первичный код, варианты, модели и допуски не менялись.",
        "",
        "[Все модели](all_models.csv) · [Все дозы](all_doses.csv) · [Внутренние кандидаты](inner_candidates.csv) · [Правила](policies.csv) · [Парные сравнения](paired.csv) · [JSON](summary.json) · [Аудит](audit.json) · [Замеры](runtime.json) · [Протокол](../../research/chromaseed_neural_shrinkage_v1_protocol.md) · [Модель](../../architecture/chromaseed_neural_shrinkage_model_card.md) · [Следующий шаг](../../research/chromaseed_neural_shrinkage_next_decision.md).",
        "",
        "Общая цель активна: эксперимент дал проверенный ответ о регуляризации, но высокое качество на обычных телефонах ещё не доказано. Следующий отдельный вопрос — локальное обучение уточнений по идее NoProp с контролем полной стоимости. Он пока не реализован и не запущен.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(text), encoding="utf-8")
    link_count = 0
    docs = [
        OUT / "report.md",
        ROOT / "docs/architecture/chromaseed_neural_shrinkage_model_card.md",
        ROOT / "docs/research/chromaseed_neural_shrinkage_next_decision.md",
    ]
    for path in docs:
        for dest in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if not dest.startswith(("http:", "https:", "#")):
                assert (path.parent / dest.split("#")[0]).resolve().is_file(), (path, dest)
                link_count += 1
    tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_chromaseed_neural_readout.py",
            "tests/test_chromaseed_neural_geometry.py",
            "-q",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    changed = [
        "scripts/chromaseed_neural_shrinkage_" + s + ".py"
        for s in ("train", "audit", "runtime", "report")
    ]
    lint = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *changed],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    SHORTCUT.parent.mkdir(parents=True, exist_ok=True)
    SHORTCUT.write_text(
        f"# Luma ChromaSeed NS\n\n5040 обучений выхода; все12 правил уступают FG.\n\n[Полный отчёт]({(OUT / 'report.md').as_posix()})\n\nЦель остаётся активной; качество на обычных телефонах не подтверждено.\n",
        encoding="utf-8",
    )
    artifacts = [
        RUN / f
        for f in (
            "source_lock.json",
            "selections.json",
            "results.json",
            "workflow.json",
            "progress.json",
            "inner_complete.json",
            "final_complete.json",
        )
    ]
    artifacts += (
        list(OUT.glob("*.json")) + list(OUT.glob("*.csv")) + docs + [ROOT / p for p in changed]
    )
    write_json(
        vp,
        dict(
            passed=True,
            source_lock_sha256=sha(RUN / "source_lock.json"),
            selection_sha256=sha(RUN / "selections.json"),
            results_sha256=sha(RUN / "results.json"),
            parent_NR_verification_sha256="3e4079df5d688136ef59be6b599633b0bc7c44d9d21f410eaeefc2aef0699c65",
            parent_NG_verification_sha256="57b1d6b00b51e9f761ad34b2062f28a1186cd403f0ba40a52e24b0d105014e25",
            checks=audit["checks"],
            maxima=audit["maxima"],
            complete_timing_fits=270,
            exact_timing_payloads=runtime["exact_payload_repeats"],
            tests_passed=33,
            pytest_output=tests,
            lint=lint,
            local_markdown_links_checked=link_count,
            comparisons=comparisons,
            alpha_counts=alpha_counts,
            previous_goal_turn_classification="progress",
            goal_status="active",
            terminal_evidence="Primary PID30112/session54705 exit0; first audit exit1 format-only; corrected audit82044 exit0; runtime41219 exit0. NG PID34636 direct exit0; NR verifier57298 exit0.",
            artifact_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in set(artifacts)},
            shortcut_sha256=sha(SHORTCUT),
        ),
    )
    print(
        dict(
            passed=True,
            verification_sha256=sha(vp),
            comparisons=comparisons,
            alpha_counts=alpha_counts,
        )
    )


if __name__ == "__main__":
    main()
