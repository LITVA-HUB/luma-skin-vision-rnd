"""Expanded-grid report: separate search effects from correction effects."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chromaseed_kernel_audit import js
from skin_local_search_train import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
FAMILIES = ("norm_mse", "constant_de2", "local_de2", "local_irls", "midpoint_irls")
SHORT = ("Norm", "Global", "Local", "IRLS", "Midpoint")


def report(run, parent, output):
    lock = js(run / "source_lock.json")
    source = sha(run / "source_lock.json")
    audit, runtime, selections = js(output / "audit.json"), js(output / "runtime.json"), js(run / "selections.json")
    current, previous, old_selections = js(run / "results.json"), js(parent / "results.json"), js(parent / "selections.json")
    assert audit["passed"] and audit["checks"]["exact_parent_payloads"] == 2916
    assert sha(parent / "results.json") == lock["parent_bindings"]["results.json"]
    assert sha(parent / "selections.json") == lock["parent_bindings"]["selections.json"]
    for d in (audit, runtime, current, selections):
        assert d["source_lock_sha256"] == source
    assert runtime["audit_sha256"] == sha(output / "audit.json")
    assert runtime["runtime_source_sha256"] == sha(ROOT / "scripts/chromaseed_weak_ridge_runtime.py")
    assert audit["audit_source_sha256"] == sha(ROOT / "scripts/chromaseed_weak_ridge_audit.py")
    for rel, expected in lock["sources"].items():
        assert sha(ROOT / rel) == expected
    rows, curves = [], []
    for role in ROLES:
        for family in FAMILIES:
            measured = [r for r in current["records"] if r["role"] == role and r["family"] == family]
            old = [r for r in previous["records"] if r["role"] == role and r["family"] == family]
            baseline = np.mean([r["metrics"]["person_mean"] for r in current["records"] if r["role"] == role and r["family"] == "norm_mse"])
            error = float(np.mean([r["metrics"]["person_mean"] for r in measured]))
            old_error = float(np.mean([r["metrics"]["person_mean"] for r in old]))
            fit = next(r for r in runtime["standalone_fit_records"] if r["role"] == role and r["family"] == family)
            inference = [r["median_us"] for r in runtime["records"] if r["role"] == role and r["family"] == family]
            chosen, old_chosen = selections["roles"][role][family]["selected"], old_selections["roles"][role][family]["selected"]
            rows.append({"role": role, "family": family, "person_mean": error, "previous_P_person_mean": old_error,
                         "change_vs_matching_P": error - old_error, "change_vs_W_norm": float(error - baseline),
                         "inner_person_mean": chosen["person_mean"], "previous_P_inner_person_mean": old_chosen["person_mean"],
                         "inner_change_vs_matching_P": chosen["person_mean"] - old_chosen["person_mean"],
                         "alpha": chosen["alpha"], "width_factor": chosen["width_factor"], "steps": chosen["steps"],
                         "numeric_bytes": measured[0]["numeric_bytes"], "fit_median_ms": fit["median_seconds"] * 1000,
                         "executed_solves": fit["details"][0]["executed_solves"], "inference_us": float(np.mean(inference)),
                         "p90": float(np.mean([r["metrics"]["p90"] for r in measured]))})
            for alpha in lock["alphas"]:
                candidate = min((c for c in selections["roles"][role][family]["candidates"] if c["alpha"] == alpha), key=lambda c: c["person_mean"])
                curves.append({"role": role, "family": family, "alpha": alpha, "person_mean": candidate["person_mean"],
                               "steps": candidate["steps"], "width_factor": candidate["width_factor"]})
    receipts = [js(p) for p in sorted(run.glob("inner/*/fold*/receipt.json")) + sorted(run.glob("final/*/bank/receipt.json"))]
    ops = [op for r in receipts for op in r["operations"]]
    cost = {"stored_configurations": sum(r["readout_configurations"] for r in receipts),
            "bank_fit_seconds": sum(r["fit_bank_seconds"] for r in receipts), "bank_wall_seconds": sum(r["bank_wall_seconds"] for r in receipts),
            "main_wall_seconds": js(run / "workflow.json")["main_wall_seconds"],
            "coupled_solves": sum(o.get("executed_solves", 0) for o in ops),
            "iterative_solves": sum(o["executed_solves"] for o in ops if o["kind"] == "iterative"),
            "exact_P_controls": sum(r["parent_control"]["identical_payloads"] for r in receipts),
            "tensor_eigenvalue_clips": sum(r["tensor_info"]["clipped_eigenvalues"] for r in receipts) + sum(o.get("tensor_eigenvalue_clips", 0) for o in ops)}
    summary = {"source_lock_sha256": source, "selection_sha256": sha(run / "selections.json"), "report_source_sha256": sha(Path(__file__)),
               "parent_results_sha256": sha(parent / "results.json"), "rows": rows, "inner_alpha_curves": curves, "cost": cost,
               "outer_regressions_vs_matching_P": sum(r["change_vs_matching_P"] > 0 for r in rows),
               "inner_improvements_vs_matching_P": sum(r["inner_change_vs_matching_P"] < 0 for r in rows),
               "evidence": "historically reused overlapping original TRAIN; no fresh confirmation"}
    write_json(output / "summary.json", summary)
    for name in ("source_lock.json", "selections.json"):
        shutil.copyfile(run / name, output / name)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4), layout="constrained")
    for ax, role in zip(axes, ROLES, strict=True):
        local = [r for r in rows if r["role"] == role]
        ax.bar(np.arange(5) - .18, [r["inner_change_vs_matching_P"] for r in local], .36, label="Inner change", color="#126d80")
        ax.bar(np.arange(5) + .18, [r["change_vs_matching_P"] for r in local], .36, label="Outer change", color="#cb6042")
        ax.axhline(0, color="#777", lw=1)
        ax.set_xticks(np.arange(5), SHORT, rotation=20)
        ax.set_title(role)
        ax.set_ylabel("ΔE00 change versus matching P family")
        ax.grid(axis="y", alpha=.18)
    axes[0].legend(fontsize=8)
    fig.suptitle("Broader penalty search: inner scores improve while outer errors often increase; reused TRAIN", fontsize=12)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"search_transfer.{suffix}", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), layout="constrained")
    for ax, role in zip(axes, ROLES, strict=True):
        for family, label in zip(FAMILIES, SHORT, strict=True):
            curve = [c for c in curves if c["role"] == role and c["family"] == family]
            ax.plot([c["alpha"] for c in curve], [c["person_mean"] for c in curve], "o-", markersize=3, label=label)
        ax.set_xscale("log")
        ax.axvline(.1, color="#888", ls="--", lw=1)
        ax.set_title(role)
        ax.set_xlabel("Ridge alpha; dashed = previous lower bound")
        ax.set_ylabel("Inner person ΔE00; lower is better")
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=8)
    fig.suptitle("At each alpha, width and correction count selected on inner people only; mean over3 seeds", fontsize=12)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"penalty.{suffix}", dpi=170)
    plt.close(fig)

    def row_for(role, family):
        return next(r for r in rows if r["role"] == role and r["family"] == family)

    text = ["# Luma ChromaSeed-W: слабая регуляризация не дала общего выигрыша", "",
            f"**Измеренный итог:** все15 внутренних оценок улучшились после расширения сетки, но {summary['outer_regressions_vs_matching_P']} из15 внешних сравнений с соответствующим прежним методом ухудшились. Более длинная коррекция стала выбираться, однако надёжной замены прежним кандидатам не получено.", "",
            "Изменён только диапазон ridge-alpha:0,0001 /0,0003 /0,001 /0,003 /0,01 /0,03 /0,1 /1 /10. Все2 916 прежних конфигураций P независимо проверены на точное совпадение числовых массивов. Это позволяет отделить новые настройки от изменения реализации.", "",
            "Объём модели прежний: **20 284 числовых байта**, один проход при ответе. Итерации выполняются при обучении выходных коэффициентов и не добавляют динамического графа на каждом запросе. Все роли — многократно использованный исходный TRAIN,24 человека; это не новая независимая проверка обычных лиц со смартфонов.", "",
            "## Все выбранные методы", "", "Ошибки person-balanced ΔE00, средние трёх отдельных seed17/29/43. Меньше — лучше. Seed не заменяет нового человека и не означает ансамбль.", "",
            "| Метод | mixed | SLR→iPod | iPod→SLR |", "|---|---:|---:|---:|"]
    for family in FAMILIES:
        text.append(f"| {family} | " + " | ".join(f"{row_for(role,family)['person_mean']:.4f}" for role in ROLES) + " |")
    text += ["", "Прежний нормированный P-контроль:5,4387 /8,5970 /8,7050. Прежние общие цветовые веса P:5,3901 /8,6548 /8,3869. Слабее регуляризованные общие веса W ухудшили все три этих сравнения. Guided RBF7,7193 на обратном переносе и ChromaSeed-R5,3551 на mixed также остаются более сильными старыми ориентирами в отдельных ролях.", "",
             "Коррекция с фиксированной геометрией улучшает новый W-контроль в прямом и обратном переносе (9,5273→9,4134 и8,9189→8,5290), но это не превосходство над прежними лучшими кандидатами. На mixed оба вида коррекции хуже нового нормированного контроля. Обновление геометрии с16 шагами даёт обратное8,5144, что всё ещё хуже прежних общих весов8,3869.", "",
             "![Внутренние улучшения и внешние ухудшения](search_transfer.png)", "",
             "Все внутренние изменения отрицательны, но внешние часто положительны. Ширина тоже выбиралась заново: новые mixed/SLR→iPod решения взяли×2 вместо прежней×1. Поэтому нельзя приписывать весь эффект только alpha или только числу шагов. Это результат зарегистрированной совместной процедуры выбора на более широкой сетке.", "",
             "## Что выбрано и сколько это стоит", "",
             "| Роль / метод | alpha | Ширина | Запрошено шагов | Выполнено решений | Fit, мс | Ответ, мкс |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        text.append(f"| {r['role']} / {r['family']} | {r['alpha']:g} | {r['width_factor']:g} | {r['steps']} | {r['executed_solves']} | {r['fit_median_ms']:.2f} | {r['inference_us']:.2f} |")
    text += ["", "Все выбранные alpha теперь находятся внутри расширенного диапазона, а не на его нижней границе. Само устранение границы не обеспечило переносимость.", "",
             "![Внутренняя ошибка по регуляризации](penalty.png)", "",
             "На графике для каждого alpha показана лучшая внутренняя конфигурация по ширине/шагам. У итеративных семей сохраняется нулевой вариант. Кривые используют только внутренние оценки, а не дополнительные оценки непринятых кандидатов на внешних людях.", "",
             "Тайминг: Ryzen9 7900X, один поток CPU. Полное обучение выбранной модели seed17 с нуля, включая балансировку/нормализацию/точную ширину/опоры/readout; один прогрев и три замера. I/O и поиск настроек исключены. При ответе20 прогревов и три прохода по всем подготовленным color36, нормализация включена, обработка фото/лица исключена. FP32-хранение и FP64-арифметика; размер не включает библиотеки/заголовки файла.", "",
             "Запрошенные16 шагов могут означать меньше выполненных решений после отклонения направления. Например, выбранный midpoint SLR→iPod остановился после5 решений. Обратный midpoint выполнил16 и обучался111,33мс, без преимущества над прежним P128-G по качеству.", "",
             "## Проверка и объём исследования", "",
             f"8 748 сохранённых readout-конфигураций в12 общих банках:6 561 внутренних и2 187 финальных.2 916 — воспроизведённые старые контроли,5 832 — новые alpha. Выполнено{cost['coupled_solves']} связанных матричных решений, из них{cost['iterative_solves']} итеративных, в1 944 траекториях. Это не8 748 независимых полных обучений. Fit банков{cost['bank_fit_seconds']:.2f}с; с контролями/предсказаниями/записью{cost['bank_wall_seconds']:.2f}с; main-проход{cost['main_wall_seconds']:.2f}с без запуска интерпретатора/импортов, отдельного аудита и профилирования.", "",
             "26 численных тестов прошли. Итоговый аудит независимо проверил все2 916 старых payload,1 239 300 внутренних прогнозов,15 решений выбора,45 финальных SVD-перерешиваний/17 970 внешних прогнозов и18 срезов положительной коррекции при alpha0,0001. Максимальный финальный дрейф независимого решения0,000369Lab, контрольных срезов0,000520Lab — ниже заранее заданного0,002. Сглаженные обучающие цели с регуляризацией не возрастали во всех1 944 траекториях. Численная проверка не превращает выборку в независимое подтверждение.", "",
             "Профилирование:45 моделей при ответе,60 выбранных полных fit с прогревами (массивы совпали точно), ещё24 фиксированных16-шаговых fit при seed17/width1/alpha0,0001. Дополнительные fit измеряли только стоимость, без новых внешних оценок. Их сырые времена и реальные числа решений — в [runtime.json](runtime.json).", "",
             "Использованы только color,target,patient,site,device исходного TRAIN. Изображения/tokens и legacy validation/calibration/test исключены. Люди разнесены внутри каждой процедуры, но исторические роли перекрываются; камеры также меняют распределение людей/цвета. Приложение, старые артефакты и источник данных не изменены.", "",
             "**Решение:** не принимать расширение диапазона как улучшение качества. Следующий полезный вопрос — устойчивость внутреннего выбора к отдельным людям и их повторной выборке. Перед новой тренировкой проверить это на уже сохранённых OOF, не выбирая новое правило по внешней таблице. Эта диагностика пока запланирована. Общая цель активна, качество на обычных лицах остаётся неустановленным.", "",
             "[Протокол](../../research/chromaseed_weak_ridge_v1_protocol.md) · [Аудит](audit.json) · [Таблицы](summary.json) · [Воспроизведение](reproduce.md) · [Карточка](../../architecture/chromaseed_weak_ridge_model_card.md) · [Следующее решение](../../research/chromaseed_weak_ridge_next_decision.md).", "",
             "Контекст методов: [Nyström computational regularization](https://arxiv.org/abs/1507.04717), [FALKON](https://arxiv.org/abs/1705.10958), прежний [P-протокол и источники CIEDE2000/IRLS](../../research/chromaseed_perceptual_v1_protocol.md). Теоретические гарантии этих работ не служат гарантией нашего применения к коже."]
    (output / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"REPORT15 rows; {summary['inner_improvements_vs_matching_P']} inner improvements, {summary['outer_regressions_vs_matching_P']} outer regressions versus matching P", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for option in ("run", "parent", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    report(args.run, args.parent, args.output)
