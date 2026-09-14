"""Measured P-study report and figures; no fitting or selection changes."""
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
LABELS = {"norm_mse": "Нормированная ошибка", "constant_de2": "Общие цветовые веса", "local_de2": "Веса каждого оттенка",
          "local_irls": "Коррекция с фиксированной геометрией", "midpoint_irls": "Коррекция с обновлением геометрии"}


def report(run, output):
    source = sha(run / "source_lock.json")
    selection = js(run / "selections.json")
    audit, runtime, results = js(output / "audit.json"), js(output / "runtime.json"), js(run / "results.json")
    assert audit["passed"]
    for item in (selection, audit, runtime, results):
        assert item["source_lock_sha256"] == source
    assert runtime["audit_sha256"] == sha(output / "audit.json")
    assert runtime["runtime_source_sha256"] == sha(ROOT / "scripts/chromaseed_perceptual_runtime.py")
    assert audit["audit_source_sha256"] == sha(ROOT / "scripts/chromaseed_perceptual_audit.py")
    for path, expected in js(run / "source_lock.json")["sources"].items():
        assert sha(ROOT / path) == expected
    rows, inner = [], []
    for role in ROLES:
        for family in FAMILIES:
            records = [r for r in results["records"] if r["role"] == role and r["family"] == family]
            timings = [r for r in runtime["records"] if r["role"] == role and r["family"] == family]
            fit = next(r for r in runtime["standalone_fit_records"] if r["role"] == role and r["family"] == family)
            chosen = selection["roles"][role][family]["selected"]
            rows.append({"role": role, "family": family, "person_mean": float(np.mean([r["metrics"]["person_mean"] for r in records])),
                         "p90": float(np.mean([r["metrics"]["p90"] for r in records])), "numeric_bytes": records[0]["numeric_bytes"],
                         "fit_median_ms": fit["median_seconds"] * 1000, "inference_us": float(np.mean([r["median_us"] for r in timings])),
                         "steps": chosen["steps"], "width_factor": chosen["width_factor"], "alpha": chosen["alpha"], "inner_person_mean": chosen["person_mean"]})
            if family.endswith("irls"):
                for step in (0, 1, 4, 16):
                    best = min((c for c in selection["roles"][role][family]["candidates"] if c["steps"] == step), key=lambda c: c["person_mean"])
                    inner.append({"role": role, "family": family, "step": step, "person_mean": best["person_mean"], "alpha": best["alpha"], "width_factor": best["width_factor"]})
    receipts = [js(p) for p in sorted(run.glob("inner/*/fold*/receipt.json")) + sorted(run.glob("final/*/bank/receipt.json"))]
    operations = [op for receipt in receipts for op in receipt["operations"]]
    cost = {"stored_configurations": sum(r["readout_configurations"] for r in receipts), "bank_fit_seconds": sum(r["fit_bank_seconds"] for r in receipts),
            "bank_wall_seconds": sum(r["bank_wall_seconds"] for r in receipts), "main_wall_seconds": js(run / "workflow.json")["main_wall_seconds"],
            "coupled_solves": sum(o.get("executed_solves", 0) for o in operations),
            "iterative_solves": sum(o["executed_solves"] for o in operations if o["kind"] == "iterative"),
            "identical_baselines": sum(r["parent_control"]["identical_payloads"] for r in receipts),
            "all_tensor_eigenvalue_clips": sum(r["tensor_info"]["clipped_eigenvalues"] for r in receipts) + sum(o.get("tensor_eigenvalue_clips", 0) for o in operations)}
    write_json(output / "summary.json", {"source_lock_sha256": source, "selection_sha256": sha(run / "selections.json"), "report_source_sha256": sha(Path(__file__)),
                                        "rows": rows, "inner_checkpoint_curves": inner, "cost": cost, "evidence": "historically reused TRAIN; seeds are not independent people"})
    shutil.copyfile(run / "source_lock.json", output / "source_lock.json")
    shutil.copyfile(run / "selections.json", output / "selections.json")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), layout="constrained")
    for ax, role in zip(axes, ROLES, strict=True):
        for i, family in enumerate(("constant_de2", "local_de2")):
            pair = next(p for p in audit["paired"] if p["role"] == role and p["family"] == family)
            value = pair["person_mean_difference"]
            lo, hi = pair["descriptive_bootstrap_interval"]
            ax.errorbar(value, i, xerr=[[value - lo], [hi - value]], fmt="o", color=("#126d80", "#cb6042")[i], capsize=4, markersize=7)
        ax.axvline(0, color="#777", lw=1, ls="--")
        ax.set_yticks([0, 1], ["Global weights", "Local weights"])
        ax.set_title(role)
        ax.set_xlabel("Person ΔE00 change; lower is better")
        ax.set_ylim(-.6, 1.6)
        ax.grid(axis="x", alpha=.18)
    fig.suptitle("Mixed transfer results: descriptive person bootstrap; historically reused TRAIN", fontsize=13)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"quality.{suffix}", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.0), layout="constrained")
    colors = {"local_irls": "#126d80", "midpoint_irls": "#cb6042"}
    for column, role in enumerate(ROLES):
        receipt = js(run / "final" / role / "bank/receipt.json")
        for family in ("local_irls", "midpoint_irls"):
            curve = [r for r in inner if r["role"] == role and r["family"] == family]
            axes[0, column].plot([r["step"] for r in curve], [r["person_mean"] for r in curve], "o-", color=colors[family], label=family)
            op = next(o for o in receipt["operations"] if o["kind"] == "iterative" and o["family"] == family and o["seed"] == 17 and o["width_index"] == 1 and o["alpha_index"] == 0)
            values = np.array([h["objective"] for h in op["trajectory"]])
            axes[1, column].plot(np.arange(len(values)), values / values[0], "o-", markersize=3, color=colors[family])
        axes[0, column].set_title(role)
        axes[0, column].set_ylabel("Inner held-person ΔE00")
        axes[1, column].set_ylabel("Fit objective / initial value")
        for row in range(2):
            axes[row, column].set_xlabel("Requested correction steps")
            axes[row, column].set_xticks([0, 1, 4, 16])
            axes[row, column].grid(alpha=.18)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("More correction lowers regularized fit objectives; all inner choices retain zero steps", fontsize=13)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"steps.{suffix}", dpi=170)
    plt.close(fig)

    def row_for(role, family):
        return next(r for r in rows if r["role"] == role and r["family"] == family)

    lines = ["# Luma ChromaSeed-P: цветовые веса и повторная коррекция", "",
             "**Измеренный итог:** общие цветовые веса немного улучшили смешанное разбиение и обратный перенос, но ухудшили SLR→iPod. Итеративное уточнение проиграло контролю во внутреннем отборе: все шесть выборов сохранили ноль шагов. Универсального улучшения точности нет.", "",
             "Размер числовой части всех кандидатов — **20 284 байта**, граф ответа прежний. Меняется только обучение выходных коэффициентов. Это исследование цвета подготовленного участка, не распознавание личности и не проверенная модель обычных фотографий лиц.", "",
             "## Все первичные результаты", "", "Средние person-balanced ΔE00 трёх отдельно обученных seed17/29/43; это не ансамбль. Ниже — исторически использованные роли исходного TRAIN,24 человека суммарно. Разбиения перекрываются, перенос камеры меняет также людей и распределение цветов. Это не свежая внешняя проверка.", "",
             "| Метод | mixed,6 чел. | SLR→iPod,16 чел. | iPod→SLR,8 чел. |", "|---|---:|---:|---:|"]
    for family in FAMILIES:
        lines.append(f"| {LABELS[family]} | " + " | ".join(f"{row_for(role, family)['person_mean']:.4f}" for role in ROLES) + " |")
    lines += ["", "Два последних ряда совпадают с контролем именно потому, что выбран нулевой шаг. Нельзя приписывать им пользу от итераций. Общие веса дают5,3901 вместо5,4387 на mixed и8,3869 вместо8,7050 на обратном переносе; прямой перенос ухудшается8,5970→8,6548. Локальные веса не выигрывают у общих во всех ролях.", "",
              "![Изменение ошибки относительно контроля](quality.png)", "",
              "Интервалы на графике — описательный bootstrap по людям с20 000 повторов, после усреднения парных различий трёх seed. Многократное исследование тех же данных не учтено как независимое подтверждение. Даже интервал ниже нуля на обратном переносе не доказывает общий выигрыш на новых клиентах.", "",
              "Сильные старые результаты сохранены: guided RBF7,7193 на обратном переносе остаётся лучше новых кандидатов; ChromaSeed-R5,3551 на mixed также ниже новой ошибки. Полное старое ядро mixed5,3984 — близкий ориентир, а не заменённое доказательство качества.", "",
              "## Что дали1,4 и16 шагов", "",
              "Для каждой итеративной семьи совместно выбирались ширина, регуляризация и число шагов, включая0. На каждом фиксированном числе шагов график показывает лучшую внутреннюю ошибку по зарегистрированной сетке. Все15 первичных выборов, включая статические методы, взяли alpha0,1 и множитель ширины1. Это нижняя граница текущей сетки регуляризации, а не доказанный глобальный оптимум.", "",
              "![Внутреннее качество и обучающая целевая функция](steps.png)", "",
              "Нижние кривые относятся к заранее фиксированным финальным обучающим выборкам, seed17/width1/alpha0,1, и нормированы на собственное начальное значение каждой целевой функции. Они включают регуляризацию. Фиксированная геометрия оптимизирует локальное приближение, обновляемая — принимает шаги по сглаженной настоящей ΔE00 с регуляризацией. Убывание этих величин не гарантирует улучшения на других людях.", "",
              "Аудит проверил все648 траекторий: целевая функция нигде не возросла. Дополнительно18 положительных контрольных срезов1/4/16 независимо пересчитаны аналитической геометрией и расширенным SVD-решением. То есть отрицательный результат не объясняется обнаруженной ошибкой матричного решателя.", "",
              "## Реальные затраты", "", "Ryzen9 7900X, один CPU-поток, FP64-вычисления/FP32-хранение. Полный fit одной выбранной модели seed17: балансировка людей/участков, нормализация, точная ширина, опоры и решение; один прогрев и три замера. Загрузка/запись/поиск исключены. Время ответа — среднее трёх per-model медиан для batch-one,20 прогревов/3 прохода; обработка фотографии и лица исключена.", "",
              "| Роль / метод | Fit, мс | Ответ, мкс | Выбранные шаги |", "|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['role']} / {r['family']} | {r['fit_median_ms']:.2f} | {r['inference_us']:.2f} | {r['steps']} |")
    lines += ["", "Отдельные замеры фиксированных16-шаговых конфигураций из численной проверки, seed17/width1/alpha0,1. Это дополнительные замеры затрат, без новых оценок на внешних строках и без выбора по ним:", "",
              "| Роль / метод | Fit, мс | Реально выполнено коррекций |", "|---|---:|---:|"]
    for r in runtime["fixed_step_fit_records"]:
        lines.append(f"| {r['role']} / {r['family']} | {r['median_seconds']*1000:.2f} | {r['details'][0]['executed_solves']} |")
    lines += ["", "При отклонённом направлении последующие контрольные шаги копируются без нового решения. Поэтому запрошенные16 шагов не всегда означают16 выполненных коррекций. Все60 профилировочных повторов выбранных моделей с прогревами сохранили массивы точно; ещё24 fit с прогревами измерили фиксированные положительные шаги.", "",
              "## Воспроизводимость и пределы вывода", "",
              f"2916 сохранённых конфигураций в12 общих банках:2187 внутренних и729 финальных, затем45 выбранных моделей. Общая линейная алгебра переиспользуется. Реально выполнено{cost['coupled_solves']} связанных решений, включая{cost['iterative_solves']} итеративных. Fit банков{cost['bank_fit_seconds']:.2f}с; с проверкой прежних контролей/запросами/записью{cost['bank_wall_seconds']:.2f}с. Main-проход{cost['main_wall_seconds']:.2f}с без запуска интерпретатора/импортов, последующего аудита и профилирования.", "",
              "Все324 нормированных контрольных модели совпали числовыми массивами с KF128. Независимый аудит проверил413 100 внутренних прогнозов,15 решений выбора, все45 финальных моделей/17 970 прогнозов и18 положительных срезов коррекции. Максимальное отличие финального SVD-перерешивания2,40×10⁻⁵Lab; положительных срезов1,63×10⁻⁵Lab. Проекций отрицательных собственных значений цветового приближения не понадобилось.", "",
              "Использован только исходный TRAIN: изображения не загружались, старые validation/calibration/test исключены. Никаких новых данных, внешних весов или изменений приложения. Научная новизна, пригодность к подбору товарного оттенка и точность на обычных лицах со смартфонов этим не установлены. Общая цель остаётся активной.", "",
              "Следующий вопрос — отдельно проверить более слабую регуляризацию: все текущие выборы упёрлись в нижнюю границу0,1. Это план следующей зарегистрированной серии, а не уже полученный результат и не повод менять текущие протокол/таблицы.", "",
              "[Протокол](../../research/chromaseed_perceptual_v1_protocol.md) · [Аудит](audit.json) · [Все метрики](summary.json) · [Замеры](runtime.json) · [Воспроизведение](reproduce.md) · [Карточка модели](../../architecture/chromaseed_perceptual_model_card.md).", "",
              "Методическая основа: [Sharma et al.2005](https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/ciede2000noteCRNA.pdf), [O'Leary1990](https://www.cs.umd.edu/users/oleary/reprints/j30.pdf), [Dong and Yang](https://arxiv.org/abs/1903.11202). Локальное приближение и его применение здесь — проверяемая инженерная гипотеза; гарантии других алгоритмов на неё автоматически не переносятся."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"REPORT generated {len(rows)} primary rows and {len(inner)} inner checkpoint points", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.run, args.output)
