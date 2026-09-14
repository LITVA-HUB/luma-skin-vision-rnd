"""Report KF search failures alongside the exact KE implementation gain."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skin_local_search_train import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
ARMS = ("dense_exact", "column_exact", "column_pairs1024", "column_pairs4096", "column_pairs16384")
NAMES = {"dense_exact": "Полное ядро + точная медиана", "column_exact": "Столбцы + точная медиана",
         "column_pairs1024": "Столбцы + 1024 пары", "column_pairs4096": "Столбцы + 4096 пар", "column_pairs16384": "Столбцы + 16384 пары",
         "condensed_exact": "Столбцы + точная медиана SciPy"}


def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize(run, condensed_run, output):
    results, choices = js(run / "results.json"), js(run / "selections.json")
    audit, runtime, exact = (js(output / name) for name in ("audit.json", "runtime.json", "condensed_exact.json"))
    assert audit["passed"] and exact["passed"]
    assert runtime["audit_sha256"] == exact["parent_audit_sha256"] == sha(output / "audit.json")
    assert runtime["runtime_source_sha256"] == sha(ROOT / "scripts/chromaseed_fast_kernel_runtime.py")
    assert audit["audit_source_sha256"] == sha(ROOT / "scripts/chromaseed_fast_kernel_audit.py")
    for folder in (run, condensed_run):
        for path, expected in js(folder / "source_lock.json")["sources"].items():
            assert sha(ROOT / path) == expected
    assert exact["source_lock_sha256"] == sha(condensed_run / "source_lock.json")
    assert exact["source_run_lock_sha256"] == sha(run / "source_lock.json")
    rows, policies, bank_costs = [], [], []
    for role in ROLES:
        for arm in ARMS:
            for rank in (64, 128, 256):
                records = [r for r in results["records"] if (r["role"], r["arm"], r["rank"]) == (role, arm, rank)]
                timing = [r for r in runtime["records"] if (r["role"], r["arm"], r["rank"]) == (role, arm, rank)]
                fit = next(r for r in runtime["standalone_fit_records"] if (r["role"], r["arm"], r["rank"]) == (role, arm, rank))
                selected = choices["roles"][role]["primary"][arm][str(rank)]
                row = {"role": role, "arm": arm, "rank": rank, "numeric_bytes_min": min(r["numeric_bytes"] for r in records),
                       "numeric_bytes_max": max(r["numeric_bytes"] for r in records), "actual_centers": [r["actual_centers"] for r in records],
                       "seed_person_means": [r["metrics"]["person_mean"] for r in records], "width_index": selected["width_index"],
                       "alpha_index": selected["alpha_index"], "inner_person_mean": selected["selected"]["person_mean"],
                       "standalone_seed17_median_ms": fit["median_ms"], "cpu_median_us_mean": float(np.mean([r["median_us"] for r in timing])),
                       "cpu_p95_us_mean": float(np.mean([r["p95_us"] for r in timing]))}
                for metric in ("person_mean", "image_mean", "site_person_mean", "p90", "gt5", "gt10"):
                    row[metric] = float(np.mean([r["metrics"][metric] for r in records]))
                rows.append(row)
        for rank in (64, 128, 256):
            arm = choices["roles"][role]["fast"][str(rank)]["selected"]["arm"]
            row = next(r for r in rows if (r["role"], r["arm"], r["rank"]) == (role, arm, rank))
            policies.append({"role": role, "policy": "fast_arm", "rank": rank, "arm": arm, "person_mean": row["person_mean"]})
        selected = choices["roles"][role]["size"]["selected"]
        row = next(r for r in rows if (r["role"], r["arm"], r["rank"]) == (role, selected["arm"], selected["rank"]))
        policies.append({"role": role, "policy": "size", "rank": selected["rank"], "arm": selected["arm"], "person_mean": row["person_mean"]})
    for path in sorted((run / "inner").glob("*/fold*/receipt.json")) + sorted((run / "final").glob("*/bank/receipt.json")):
        receipt = js(path)
        bank_costs.append({"stage": path.relative_to(run).parts[0], "role": receipt["role"], "readouts": receipt["readout_configurations"],
                           "fit_seconds": receipt["fit_bank_seconds"], "wall_seconds": receipt["bank_wall_seconds_including_parent_reproduction_and_persistence"],
                           "old_k_reproduction": receipt["old_k_reproduction"]})
    summary = {"family": "Luma ChromaSeed-KF + exact KE follow-up", "evidence": results["evidence"], "rows": rows, "policies": policies,
               "source_lock_sha256": sha(run / "source_lock.json"), "selection_sha256": sha(run / "selections.json"),
               "condensed_source_lock_sha256": sha(condensed_run / "source_lock.json"), "report_source_sha256": sha(Path(__file__)),
               "cost": {"inner_readouts": sum(r["readouts"] for r in bank_costs if r["stage"] == "inner"),
                        "final_readouts": sum(r["readouts"] for r in bank_costs if r["stage"] == "final"),
                        "bank_fit_seconds": sum(r["fit_seconds"] for r in bank_costs), "bank_wall_seconds": sum(r["wall_seconds"] for r in bank_costs),
                        "main_search_seconds": js(run / "workflow.json")["main_wall_seconds"]}, "bank_costs": bank_costs}
    write_json(output / "summary.json", summary)
    shutil.copyfile(run / "source_lock.json", output / "source_lock.json")
    shutil.copyfile(run / "selections.json", output / "selections.json")
    shutil.copyfile(condensed_run / "source_lock.json", output / "condensed_source_lock.json")
    plot(summary, runtime, exact, output)
    report(summary, choices, audit, runtime, exact, output)


def row_for(summary, role, arm, rank):
    return next(r for r in summary["rows"] if (r["role"], r["arm"], r["rank"]) == (role, arm, rank))


def plot(summary, runtime, exact, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    colors = {"dense_exact": "#233649", "column_pairs1024": "#c24e52", "column_pairs4096": "#138b87", "column_pairs16384": "#6689c0"}
    for ax, role in zip(axes, ROLES, strict=True):
        for arm, color in colors.items():
            values = [row_for(summary, role, arm, k)["person_mean"] for k in (64, 128, 256)]
            ax.plot([64, 128, 256], values, marker="o", color=color, label=arm.replace("column_pairs", "Pairs ").replace("dense_exact", "Exact"))
        selected = next(p for p in summary["policies"] if p["role"] == role and p["policy"] == "size")
        ax.scatter([selected["rank"]], [selected["person_mean"]], s=100, marker="x", linewidths=2, color="#9d245e", label="Inner size policy")
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 128, 256], ["64", "128", "256"])
        ax.set_xlabel("Nominal landmark count")
        ax.set_ylabel("Person-balanced native ΔE00 ↓")
        ax.set_title({"mixed": "Mixed: 6 held people", "slr_to_ipod": "SLR → iPod: 16 held people", "ipod_to_slr": "iPod → SLR: 8 held people"}[role])
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=8)
    figure.suptitle("Sampled choices do not guarantee outer quality; means over 3 seeds on historically reused TRAIN roles", fontsize=12)
    figure.savefig(output / "accuracy.png", dpi=160)
    figure.savefig(output / "accuracy.svg")
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    plot_series = [(arm, [r for r in runtime["synthetic_scaling"] if r["arm"] == arm]) for arm in ("dense_exact", "column_exact", "column_pairs4096")]
    plot_series.append(("condensed_exact", exact["synthetic_scaling"]))
    palette = ["#617286", "#ab668b", "#c49b4a", "#11897f"]
    for (name, values), color in zip(plot_series, palette, strict=True):
        xs = [r["n_rows"] for r in values]
        label = {"dense_exact": "Dense + exact", "column_exact": "Columns + original exact", "column_pairs4096": "Columns + 4096 pairs", "condensed_exact": "Columns + condensed exact"}[name]
        axes[0].plot(xs, [r["median_ms"] for r in values], marker="o", color=color, label=label)
        axes[1].plot(xs, [r["incremental_tracemalloc_peak_bytes"] / 1024**2 for r in values], marker="o", color=color, label=label)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1024, 4096, 8192], ["1024", "4096", "8192"])
        ax.set_xlabel("Synthetic fit rows; fixed 128 centers")
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Standalone fit median, ms ↓")
    axes[1].set_ylabel("Incremental traced allocation peak, MiB ↓")
    axes[0].legend(fontsize=8)
    figure.suptitle("SYNTHETIC scaling; not skin accuracy. Traced allocations exclude preloaded data/runtime, not process RSS.", fontsize=11)
    figure.savefig(output / "scaling.png", dpi=160)
    figure.savefig(output / "scaling.svg")
    plt.close(figure)


def report(summary, choices, audit, runtime, exact, output):
    fits = exact["standalone_fit_records"]

    def get_fit(role, rank, kind):
        return next(r["median_ms"] for r in fits if (r["role"], r["rank"], r["implementation"]) == (role, rank, kind))

    old_ms, new_ms = get_fit("mixed", 128, "dense_exact"), get_fit("mixed", 128, "condensed_exact")
    quality = [row_for(summary, role, "column_exact", 128)["person_mean"] for role in ROLES]
    latency = float(np.mean([r["median_us"] for r in exact["inference"] if r["role"] == "mixed" and r["rank"] == 128]))
    lines = ["# Luma ChromaSeed: ускорение точного обучения", "",
             f"**Измеренный результат:** обучение модели на 128 опорах и 734 подготовленных примерах ускорено с **{old_ms:.2f} до {new_ms:.2f} мс**, в **{old_ms / new_ms:.2f} раза**, с сохранением всех числовых массивов проверенных моделей. Размер остаётся **20 284 байта**. Медиана CPU-ответа на готовые признаки **{latency:.2f} мкс**. Это время ядра цвета, без получения признаков из фотографии и поиска лица.", "",
             "Серия KF сначала проверила выборку пар и автоматический выбор размера. Эти политики прошли внутренний отбор, но часть внешних ошибок выросла. Затем отдельный заранее описанный опыт KE заменил только реализацию точной медианы: скомпилированные расстояния верхнего треугольника, без приближения. Это стандартная инженерная оптимизация, не новая архитектура или доказанная научная новизна.", "",
             "Все роли — повторно использованный исходный TRAIN, не новая независимая проверка. Качество на обычных лицах со смартфонов неизвестно. [Решение и следующий опыт](../../research/chromaseed_fast_kernel_next_decision.md).", "",
             "## Что дал точный вариант KE", "",
             f"Воспроизведены **81 внутренняя и 27 финальных моделей** с ранее зафиксированными настройками `column_exact`. Все **{exact['checks']['exact_payload_matches']}/108** наборов числовых массивов совпали точно, включая опоры и коэффициенты. Проверены **{exact['checks']['query_rows']}** строк ответов: максимальное отличие **{exact['maxima']['prediction_component_drift']:.3g} Lab**. Относительное отличие неокруглённой медианы {exact['maxima']['relative_width_difference']:.3g}, после FP32-округления — {exact['maxima']['rounded_width_ulps']} ULP. Гиперпараметры не подбирались заново.", "",
             f"Ошибка модели128 в трёх ролях: **{quality[0]:.4f} / {quality[1]:.4f} / {quality[2]:.4f} ΔE00**. Это сохранение прежнего качества, не новый выигрыш по точности. Предыдущий guided RBF всё ещё сильнее на обратном переносе (7,7193). Числовая эквивалентность относится к уже независимо проверенным KF-моделям; второй внешней лабораторной репликации здесь нет.", "",
             "Парные замеры одной модели seed17, каждый раз с нуля: балансировка людей/участков, нормализация, точная ширина, опоры и решение. Один прогрев и три измерения; загрузка файлов, поиск гиперпараметров и сохранение исключены. Один CPU-поток Ryzen9 7900X. SciPy1.18.1 уже был установлен.", "",
             "| Роль / опоры | Полная матрица, мс | Только столбцы, мс | Столбцы + точная медиана SciPy, мс |", "|---|---:|---:|---:|"]
    for role in ROLES:
        for rank in (64, 128, 256):
            lines.append(f"| {role} / {rank} | {get_fit(role, rank, 'dense_exact'):.2f} | {get_fit(role, rank, 'column_exact'):.2f} | {get_fit(role, rank, 'condensed_exact'):.2f} |")
    lines.extend(["", "FP32 — формат хранения; ядра и линейная алгебра вычисляются в FP64. Числовые байты исключают заголовки NPZ, Python/NumPy, временные массивы и обработку изображения. SciPy требуется для нового обучения, стандартный сохранённый предиктор использует NumPy.", "",
                  "## Полная первичная таблица KF", "", "Ниже средние ошибок трёх отдельно построенных моделей. Это не ансамбль; seed не заменяет независимых людей. Параметры каждого метода/размера выбраны на внутренних разбиениях по людям.", "",
                  "| Метод / опоры | Смешанные, 6 чел. | SLR → iPod, 16 чел. | iPod → SLR, 8 чел. |", "|---|---:|---:|---:|"])
    for arm in ARMS:
        for rank in (64, 128, 256):
            values = [row_for(summary, role, arm, rank)["person_mean"] for role in ROLES]
            lines.append(f"| {NAMES[arm]} / {rank} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines.extend(["", "![Точность по размерам и способам выбора ширины](accuracy.png)", "",
                  "Все девять политик быстрого выбора приняли1024 пары. В смешанной роли128 ошибка выросла с5,4387 до5,9161 — примерно на0,4774ΔE00. Внутренний допуск +0,05 по средней ошибке и +0,10 поp90 не стал гарантией на других людях. Одновременно выбранная ширина сменилась с×1 на×2 при очень близких внутренних средних; это наблюдение, не изоляция единственной причины потери.", "",
                  "Автоматический размер выбрал64 опоры для смешанной роли и128 для обеих направленных ролей:", "",
                  "| Роль | Опоры | Метод | Внешняя ошибка ΔE00 |", "|---|---:|---|---:|"])
    for row in summary["policies"]:
        if row["policy"] == "size":
            lines.append(f"| {row['role']} | {row['rank']} | {row['arm']} | {row['person_mean']:.4f} |")
    lines.extend(["", "На смешанной роли уменьшение до64 не сохраняет качество прежней модели128. Эта адаптация происходит при выборе конфигурации обучения, а не внутри каждого ответа. Гипотеза полезна как отрицательный результат; рекламировать её как надёжную автоматическую настройку нельзя.", "",
                  "256 опор приблизили смешанное качество к полному ядру, но ухудшили обратный перенос относительно128 и удвоили числовой размер примерно до40КБ. Увеличение модели не даёт универсальной победы. Все размеры, дополнительные метрики и времена лежат в [summary.json](summary.json).", "",
                  "## Масштабирование времени и памяти", "",
                  "Искусственные числовые данные, без утверждений о реалистичности кожи. N=1024/4096/8192,rank128. Для каждого случая один прогрев, три независимых замера и отдельный fit с измерением выделений памяти. Точное и выборочное обучение здесь измеряются целиком. Величина памяти — пиковое приращение отслеживаемых выделений Python/NumPy над уже загруженными данными и средой, не RSS процесса. Видимость NumPy-памяти отдельно проверена на массиве8МиБ.", "",
                  "| N | Метод | Fit, мс | Пик выделений, МиБ |", "|---:|---|---:|---:|"])
    for n in (1024, 4096, 8192):
        for row in [r for r in runtime["synthetic_scaling"] if r["n_rows"] == n]:
            lines.append(f"| {n} | {NAMES[row['arm']]} | {row['median_ms']:.2f} | {row['incremental_tracemalloc_peak_bytes'] / 1024**2:.2f} |")
        row = next(r for r in exact["synthetic_scaling"] if r["n_rows"] == n)
        lines.append(f"| {n} | {NAMES['condensed_exact']} | {row['median_ms']:.2f} | {row['incremental_tracemalloc_peak_bytes'] / 1024**2:.2f} |")
    lines.extend(["", "![Искусственное масштабирование](scaling.png)", "",
                  "Точные condensed-расстояния всё ещё требуют квадратичной памяти поN. На8192 строках они сокращают затраты, но это не линейный по памяти алгоритм и не доказательство пригодности для миллионов примеров. Выборочная ширина значительно экономнее; её риск по качеству показан выше. Замеры KE и первичные KF-контроли собраны отдельно на том же компьютере, парные реальные замеры выполнены в одной серии.", "",
                  "## Объём работы и проверка", "",
                  f"KF:3645 внутренних +1215 финальных аналитических readout-конфигураций в12 банках,135 выбранных моделей. Совместные вычисления переиспользуются; это не4860 отдельных полных обучений. Сумма fit-времени банков {summary['cost']['bank_fit_seconds']:.2f}с; с проверкой старых контролей, запросами и записью {summary['cost']['bank_wall_seconds']:.2f}с. Полный main-проход {summary['cost']['main_search_seconds']:.2f}с, включая выбор/оценку/запись, исключая запуск интерпретатора, аудит и последующие профилирования.", "",
                  f"Аудит KF проверил4860 нормализаций/ширин/наборов опор,10935 контрольных OOF-наблюдений,972 сравнения точных реализаций,45+9+3 решений выбора. Все135 финальных моделей перерешены независимыми прямыми матрицами/SVD, все53910 финальных прогнозов пересчитаны. Максимальный дрейф перерешивания {audit['maxima']['svd_component_drift']:.3g}Lab; максимальный дрейф независимого предиктора {audit['maxima']['prediction_component_drift']:.3g}.", "",
                  "Отдельные измерения KF:180 самостоятельных fit с прогревами и45 искусственных fit, включая пробы памяти. KE:108 воспроизведённых моделей,108 парных fit с прогревами и15 искусственных fit. Повторы и эквивалентные реализации не увеличивают число независимых людей или новых гипотез.", "",
                  "Использован только исходный TRAIN,966 записей/24 человека, без загрузки изображений. Старые validation/calibration/test исключены. Все текущие роли исторически использованы и перекрываются. Исследовательская цель остаётся активной: теперь целесообразно проверять именно качество цветового решения, а не считать ускорение его доказательством.", "",
                  "Методы: [RPCholesky](https://arxiv.org/abs/2207.06503), [Nyström regularization](https://arxiv.org/abs/1507.04717), [официальная документация SciPy pdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html). [Протокол KF](../../research/chromaseed_fast_kernel_v1_protocol.md) · [протокол KE](../../research/chromaseed_condensed_exact_v1_protocol.md) · [аудит](audit.json) · [точное воспроизведение](condensed_exact.json) · [воспроизведение серии](reproduce.md) · [карточка](../../architecture/chromaseed_fast_kernel_model_card.md).", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    for name in ("run", "condensed-run", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run, args.condensed_run, args.output)


if __name__ == "__main__":
    main()
