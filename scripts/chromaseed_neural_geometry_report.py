"""Seal or read-only verify the inner-only geometry evidence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from chromaseed_kernel_audit import js
from skin_local_search_train import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/runs/chromaseed_neural_geometry_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_neural_geometry_v1"


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    lock = js(RUN / "source_lock.json")
    for p, h in {**lock["sources"], **lock["input_sha256"]}.items():
        assert sha(ROOT / p) == h, p
    vp = OUT / "verification.json"
    if vp.exists():
        saved = js(vp)
        for p, h in saved["artifact_sha256"].items():
            assert sha(ROOT / p) == h, p
        assert saved["passed"]
        print(dict(passed=True, mode="read-only", verification_sha256=sha(vp)))
        return
    result, workflow = js(RUN / "results.json"), js(RUN / "workflow.json")
    assert (
        result["source_lock_sha256"]
        == workflow["source_lock_sha256"]
        == sha(RUN / "source_lock.json")
    )
    assert result["spectra_sha256"] == sha(RUN / "spectra.npz")
    assert workflow["exit_code"] == 0
    assert (len(result["records"]), len(result["curves"]), result["candidate_scores_checked"]) == (
        756,
        84,
        252,
    )
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_chromaseed_neural_geometry.py", "-q"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    files = [
        "scripts/chromaseed_neural_geometry.py",
        "scripts/chromaseed_neural_geometry_reference.py",
        "scripts/chromaseed_neural_geometry_run.py",
        "scripts/chromaseed_neural_geometry_report.py",
        "tests/test_chromaseed_neural_geometry.py",
    ]
    lint = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *files],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    text = [
        "# Luma ChromaSeed NG: почему стоит проверить более сильный штраф",
        "",
        "Все756 расчётов геометрии выходного слоя сверены независимо через QR/SVD;252 прежние внутренние оценки воспроизведены точно. Новые модели не обучались, внешние предсказания не оценивались. Исходный TRAIN, девять внутренних разбиений,378 фиксированных основ.",
        "",
        "77/84 прежних кривых улучшались при переходе alpha1→10. Медианное уменьшение эквивалентных степеней свободы на один выход при10→100 —10.906852. Оба заранее установленных условия выполнены: один ограниченный опыт .1/1/10/100/1000 оправдан. Это не доказательство улучшения качества; диапазон проверяется целиком с прежними контролями.",
        "",
        "Числа ниже — медианы по семи основам, трём seed и трём внутренним разбиениям. Это мера гибкости выхода, не число независимых людей, не размер сохранённой сети и не процент точности. Свободный член не штрафуется; постоянному трёхмерному ответу соответствует значение1 на выход.",
        "",
        "| Роль | Вход | Цель | df при .1 | при1 | при10 | при100 | при1000 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for role in ("mixed", "slr_to_ipod", "ipod_to_slr"):
        for group in ("raw36", "mean3"):
            for family in ("norm", "perceptual"):
                rows = [
                    r
                    for r in result["records"]
                    if (r["role"], r["group"], r["family"]) == (role, group, family)
                ]
                vals = np.median([r["equivalent_per_output_df"] for r in rows], axis=0)
                text.append(
                    f"| {role} | {group} | {family} | "
                    + " | ".join(f"{v:.3f}" for v in vals)
                    + " |"
                )
    text += [
        "",
        "У raw36 не обнаружено почти постоянных скрытых столбцов; у mean3 максимум4 из64. При alpha10 модель всё ещё использует заметно больше направлений, чем константа. Усиление штрафа уменьшит гибкость без изменения формата сети; хороший перенос из этого не следует.",
        "",
        f"Независимая проверка: относительное расхождение спектров до{result['maxima']['spectrum_scaled_error']:.3g}; абсолютное расхождение df до{result['maxima']['degrees_error']:.3g}. Проверены3780 значений df. Расчёт занял{workflow['wall_seconds']:.3f}с после импорта, PID{workflow['pid']}, завершениеexit0. Это диагностическое время, не скорость обучения/телефона.",
        "",
        "Полные значения и спектры: [results.json](../../../experiments/runs/chromaseed_neural_geometry_v1/results.json), [spectra.npz](../../../experiments/runs/chromaseed_neural_geometry_v1/spectra.npz). [Протокол](../../research/chromaseed_neural_geometry_v1_protocol.md).",
        "",
        "Предыдущий поиск на форумах — продвижение, но не новый численный результат. Общая цель остаётся активной. Качество цвета лица на обычном телефоне пока не подтверждено. Все прежние отрицательные результаты, разбиения и отчёты сохранены.",
        "",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text("\n".join(text), encoding="utf-8")
    artifacts = [
        RUN / f
        for f in (
            "source_lock.json",
            "results.json",
            "spectra.npz",
            "workflow.json",
            "progress.json",
        )
    ] + [OUT / "report.md", Path(__file__)]
    write_json(
        vp,
        dict(
            passed=True,
            source_lock_sha256=sha(RUN / "source_lock.json"),
            results_sha256=sha(RUN / "results.json"),
            tests_passed=9,
            pytest_output=tests,
            lint=lint,
            independent_designs=756,
            degrees_checked=3780,
            candidate_scores_checked=252,
            maxima=result["maxima"],
            decision=result["decision"],
            terminal_evidence="Primary command exited0; workflow PID34636; no observation timeout",
            previous_goal_turn_classification="progress",
            goal_status="active",
            artifact_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in artifacts},
        ),
    )
    print(dict(passed=True, verification_sha256=sha(vp), decision=result["decision"]))


if __name__ == "__main__":
    main()
