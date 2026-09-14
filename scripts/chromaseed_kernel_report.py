"""Source-linked descriptive report of the completed frozen K experiment."""
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
ROLES = {"mixed": "Смешанные камеры (6 человек)", "slr_to_ipod": "SLR → iPod (16 человек)", "ipod_to_slr": "iPod → SLR (8 человек)"}
EN_ROLES = {"mixed": "Mixed cameras; 6 held people", "slr_to_ipod": "SLR → iPod; 16 held people", "ipod_to_slr": "iPod → SLR; 8 held people"}
LABELS = {"exact": "Полное ядро", "nys_random": "Nyström random", "nys_pivot": "Nyström greedy",
          "nys_rpchol": "Nyström RPCholesky", "project_rpchol": "Проекция RPCholesky",
          "adaptive_project": "Проекция с ранним выходом", "blend_nys_rpchol_64": "Nyström64 + guided RBF",
          "blend_project_rpchol_128": "Проекция128 + guided RBF"}


def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def aggregate(run, output):
    results = js(run / "results.json")
    choices = js(run / "selections.json")
    audit = js(output / "audit.json")
    runtime = js(output / "runtime.json")
    workflow = js(output / "complete_workflow_runtime.json")
    if not audit["passed"] or runtime["audit_sha256"] != sha(output / "audit.json"):
        raise ValueError("runtime must refer to the current passing audit")
    for record in (results, audit, runtime, workflow):
        if record["source_lock_sha256"] != sha(run / "source_lock.json") or record["selection_sha256"] != sha(run / "selections.json"):
            raise ValueError("source or selection mismatch")
    if audit["audit_source_sha256"] != sha(ROOT / "scripts/chromaseed_kernel_audit.py"):
        raise ValueError("audit implementation changed")
    if runtime["runtime_source_sha256"] != sha(ROOT / "scripts/chromaseed_kernel_runtime.py"):
        raise ValueError("timing implementation changed")
    rows = []
    configurations = list(dict.fromkeys((r["family"], r["rank"]) for r in results["records"] if r["family"] in LABELS))
    for role in ROLES:
        for family, rank in configurations:
            records = [r for r in results["records"] if (r["role"], r["family"], r["rank"]) == (role, family, rank)]
            timing = [r for r in runtime["records"] if (r["role"], r["family"], r["rank"]) == (role, family, rank)]
            if len(records) != len(timing):
                raise ValueError("timing/model count mismatch")
            row = {"role": role, "family": family, "rank": rank, "seeds": [r["seed"] for r in records],
                   "numeric_bytes": records[0]["numeric_bytes"], "archive_bytes_mean": float(np.mean([r["archive_bytes"] for r in records])),
                   "seed_person_means": [r["metrics"]["person_mean"] for r in records],
                   "cpu_median_us_mean": float(np.mean([r["median_us"] for r in timing])),
                   "cpu_p95_us_mean": float(np.mean([r["p95_us"] for r in timing]))}
            for metric in ("person_mean", "image_mean", "site_person_mean", "p90", "gt5", "gt10"):
                row[metric] = float(np.mean([r["metrics"][metric] for r in records]))
            if family == "adaptive_project":
                row.update({"tolerance": records[0]["tolerance"], "mean_centers": float(np.mean([r["mean_centers"] for r in records])),
                            "tolerance_met_fraction": float(np.mean([r["tolerance_met_fraction"] for r in records]))})
            elif family.startswith("blend_"):
                row["rho"] = records[0]["rho"]
            else:
                selected = choices["roles"][role]["primary"][family][str(rank)]
                row.update({"width_index": selected["width_index"], "alpha_index": selected["alpha_index"],
                            "inner_person_mean": selected["selected"]["person_mean"]})
            rows.append(row)
    costs = []
    paths = sorted((run / "inner").glob("*/fold*/receipt.json")) + sorted((run / "final").glob("*/bank/receipt.json"))
    for path in paths:
        receipt = js(path)
        costs.append({"stage": path.relative_to(run).parts[0], "role": receipt["role"],
                      "n_fit_rows": receipt["n_fit_rows"], "readouts": receipt["readout_configurations"],
                      "fit_bank_seconds": receipt["fit_bank_seconds"],
                      "bank_wall_seconds_including_persistence_and_queries": receipt["bank_wall_seconds_including_persistence_and_queries"]})
    summary = {"family": "Luma ChromaSeed-K v1", "evidence": results["evidence"], "rows": rows,
               "source_lock_sha256": results["source_lock_sha256"], "selection_sha256": results["selection_sha256"],
               "report_source_sha256": sha(Path(__file__)), "cost": {
                   "inner_banks": sum(c["stage"] == "inner" for c in costs), "final_banks": sum(c["stage"] == "final" for c in costs),
                   "inner_readouts": sum(c["readouts"] for c in costs if c["stage"] == "inner"),
                   "final_readouts": sum(c["readouts"] for c in costs if c["stage"] == "final"),
                   "total_bank_fit_seconds": sum(c["fit_bank_seconds"] for c in costs),
                   "total_bank_wall_seconds": sum(c["bank_wall_seconds_including_persistence_and_queries"] for c in costs),
                   "complete_replay_wall_seconds": workflow["wall_seconds"], "selected_primary_models": 123,
                   "adaptive_models": 9, "blend_models": 18, "reference_reproductions": 3,
                   "standalone_timing_refits_including_warmups": 4 * len(runtime["standalone_fit_records"])}, "bank_costs": costs}
    write_json(output / "summary.json", summary)
    for name in ("source_lock.json", "selections.json"):
        shutil.copyfile(run / name, output / name)
    plot(summary, output)
    report(summary, audit, runtime, workflow, output)


def one(summary, role, family, rank):
    return next(r for r in summary["rows"] if (r["role"], r["family"], r["rank"]) == (role, family, rank))


def plot(summary, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    legacy = js(ROOT / "docs/benchmarks/skin_local_search_v1/summary.json")["rows"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    colors = {"nys_random": "#929baa", "nys_pivot": "#a15b84", "nys_rpchol": "#167d8d", "project_rpchol": "#de9229"}
    for ax, role in zip(axes, ROLES, strict=True):
        for family, color in colors.items():
            rows = sorted([r for r in summary["rows"] if r["role"] == role and r["family"] == family], key=lambda r: r["rank"])
            ax.plot([r["numeric_bytes"] / 1024 for r in rows], [r["person_mean"] for r in rows], marker="o", color=color,
                    label={"project_rpchol": "Teacher projection"}.get(family, LABELS[family]))
        exact = one(summary, role, "exact", 0)
        guided = next(r for r in legacy if r["protocol"] == role and r["method"] == "guided_rbf")
        ax.scatter([exact["numeric_bytes"] / 1024], [exact["person_mean"]], color="#20334a", marker="*", s=115, label="Full kernel")
        ax.axhline(exact["person_mean"], color="#20334a", ls="--", lw=1, alpha=.65)
        ax.scatter([10996 / 1024], [guided["person_mean"]], marker="X", color="#a44747", s=65, label="Previous guided RBF")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Numeric payload, KiB (1 KiB = 1024 B)")
        ax.set_ylabel("Person-balanced native ΔE00 ↓")
        ax.set_title(EN_ROLES[role])
        ax.grid(alpha=.16)
    axes[0].legend(fontsize=8, loc="upper right")
    figure.suptitle("ChromaSeed-K: reused exploratory TRAIN roles; averages over seeds, not an ensemble", fontsize=12)
    figure.savefig(output / "quality_size.png", dpi=160)
    figure.savefig(output / "quality_size.svg")
    plt.close(figure)
    figure, ax = plt.subplots(figsize=(9, 4.7), constrained_layout=True)
    choices = [("exact", 0, "Full kernel"), ("nys_rpchol", 128, "Nyström 128"),
               ("project_rpchol", 128, "Projection 128"), ("adaptive_project", 128, "Adaptive projection")]
    values = [one(summary, "mixed", f, k)["cpu_median_us_mean"] for f, k, _ in choices]
    bars = ax.barh([label for _, _, label in choices], values, color=["#526477", "#167d8d", "#de9229", "#a15b84"])
    ax.bar_label(bars, labels=[f"{v:.1f} µs" for v in values], padding=5)
    ax.set_xlim(0, max(values) * 1.19)
    ax.invert_yaxis()
    ax.set_xlabel("Mean of per-model median latency (µs); one CPU thread")
    ax.grid(axis="x", alpha=.15)
    ax.set_title("Measured batch-one kernel execution: mixed role, Ryzen 9 7900X\nNormalization included; image feature extraction and file load excluded", fontsize=11)
    figure.savefig(output / "runtime.png", dpi=160)
    figure.savefig(output / "runtime.svg")
    plt.close(figure)


def report(summary, audit, runtime, workflow, output):
    cost = summary["cost"]
    rp = one(summary, "mixed", "nys_rpchol", 128)
    exact = one(summary, "mixed", "exact", 0)
    fits = runtime["standalone_fit_records"]
    nys_fit = next(r["median_ms"] for r in fits if r["family"] == "nys_rpchol" and r["rank"] == 128)
    proj_fit = next(r["median_ms"] for r in fits if r["family"] == "project_rpchol" and r["rank"] == 128 and r["exact_backend"] == "cuda")
    lines = ["# Luma ChromaSeed-K v1: компактные ядра", "",
             "Измеренные исследовательские результаты на повторно использованном исходном TRAIN. Это не новая независимая проверка на обычных лицах со смартфонов. ΔE00 измеряет ошибку цвета относительно инструментального Lab; меньше — лучше.", "",
             f"Получен полезный компромисс: **Nyström RPCholesky, 128 опор, {rp['numeric_bytes']:,} байт чисел**. Смешанная ошибка **{rp['person_mean']:.4f}** против **{exact['person_mean']:.4f}** у полного ядра размером {exact['numeric_bytes']:,} байт. Модель меньше в {exact['numeric_bytes'] / rp['numeric_bytes']:.2f} раза, но это небольшой проигрыш по ошибке, а не доказанная эквивалентность. Простая проекция учителя даёт близкий результат и дольше обучается. Устойчивого превосходства над всеми прежними сильными моделями нет.", "",
             f"Медиана ответа Nyström128: **{rp['cpu_median_us_mean']:.2f} мкс**, самостоятельное обучение на 734 готовых строках: **{nys_fit:.2f} мс**. Хранение FP32, арифметика ядра FP64. Время не включает получение признаков из фотографии. [Решение и следующий опыт](../../research/chromaseed_kernel_next_decision.md).", "",
             "## Точность и размер", "",
             "Параметры выбирались только на внутренних разбиениях по людям. В каждой ячейке средняя ошибка отдельно построенных моделей; три seed для случайных методов, один для полного ядра и greedy. Это не ансамбль. Выбор роли также меняет людей и распределение цветов; перенос не изолирует влияние камеры.", "",
             "| Модель / опоры | Смешанные, 6 чел. | SLR → iPod, 16 чел. | iPod → SLR, 8 чел. | Числовые байты |",
             "|---|---:|---:|---:|---:|"]
    for row in [r for r in summary["rows"] if r["role"] == "mixed"]:
        family, rank = row["family"], row["rank"]
        variants = [one(summary, role, family, rank) for role in ROLES]
        sizes = [r["numeric_bytes"] for r in variants]
        size = str(sizes[0]) if len(set(sizes)) == 1 else f"{min(sizes)}–{max(sizes)}"
        label = LABELS[family] + (f" / {rank}" if rank and not family.startswith("blend_") else "")
        lines.append(f"| {label} | " + " | ".join(f"{r['person_mean']:.4f}" for r in variants) + f" | {size} |")
    legacy = js(ROOT / "docs/benchmarks/skin_local_search_v1/summary.json")["rows"]
    for method in ("guided_rbf", "mlp"):
        values = [next(r["person_mean"] for r in legacy if r["protocol"] == role and r["method"] == method) for role in ROLES]
        lines.append(f"| Прежний {method} | " + " | ".join(f"{v:.4f}" for v in values) + " | 10996 |")
    lines.extend(["", "![Все размеры и методы](quality_size.png)", "",
                  "Размер — сумма числовых массивов модели, включая нормализацию и опоры. Служебные поля NPZ, интерпретатор, временные FP64-массивы и пиковая RAM в это число не входят. Размер самого архива и остальные метрики, включая p90, сохранены в [summary.json](summary.json). Случайные опоры могут оказаться лучше в одной роли и хуже в другой; выбирать красивый результат из таблицы как независимое подтверждение нельзя.", "",
                  "Описательная парная разность проекции128 относительно полного ядра (20 000 повторных выборок людей; это не поправка на множество прежних опытов):", "",
                  "| Роль | Разность ΔE00 | Описательный 95% интервал |", "|---|---:|---:|"])
    for row in audit["paired"]:
        if row["family"] == "project_rpchol" and row["rank"] == 128:
            lo, hi = row["descriptive_person_bootstrap_95"]
            lines.append(f"| {ROLES[row['role']]} | {row['mean_delta_vs_exact']:+.4f} | [{lo:+.4f}; {hi:+.4f}] |")
    lines.extend(["", "## Уточнение и объединение моделей", "",
                  "Все три внутренних выбора раннего выхода оставили допуск 0, то есть принудительно все 128 опор. Экономии вычислений не было: реальный пошаговый путь занимает больше времени и хранит 27 508 байт против 20 284 у фиксированной модели. Диагностическая граница сравнивает приближение с обученным полным ядром; она не ограничивает ошибку относительно истинного цвета кожи. При максимальном числе опор модель возвращает ответ даже без выполнения допуска. Численные проверки границы прошли, формальная сертификация округлений не заявляется.", "",
                  "Смешивание с guided RBF включало нулевую коррекцию. Внутренний выбор коэффициента guided для Nyström64: 0,5 / 0,25 / 0,5; для проекции128: 0,5 / 0 / 0,25 в порядке трёх ролей. На обратном переносе смешивание помогает по сравнению с новым ядром, но прежний guided RBF всё ещё лучше. На смешанной роли коррекция ухудшает проекцию128. При нулевом коэффициенте ненужная модель отсутствует в файле и не вызывается при измерении скорости.", "",
                  "## Время и вычислительная стоимость", "",
                  "Ryzen 9 7900X, один CPU-поток; RTX 4060 8 ГБ для полных спектральных решений. Для каждого из 150 выбранных моделей: 20 прогревов и три прохода по всем внешним строкам, batch=1. Перед замером проверено совпадение с сохранёнными прогнозами. Старые KRR и guided RBF также повторно измерены своей неизменённой реализацией FP32.", "",
                  "| Модель, смешанная роль | Медиана, мкс | p95, мкс |", "|---|---:|---:|"])
    for family, rank in (("exact", 0), ("nys_rpchol", 128), ("project_rpchol", 128), ("adaptive_project", 128), ("blend_project_rpchol_128", 128)):
        row = one(summary, "mixed", family, rank)
        lines.append(f"| {LABELS[family]} | {row['cpu_median_us_mean']:.2f} | {row['cpu_p95_us_mean']:.2f} |")
    for family in ("legacy_krr", "legacy_guided_rbf"):
        rows = [r for r in runtime["records"] if r["role"] == "mixed" and r["family"] == family]
        lines.append(f"| Прежний {family.removeprefix('legacy_')} (FP32) | {np.mean([r['median_us'] for r in rows]):.2f} | {np.mean([r['p95_us'] for r in rows]):.2f} |")
    lines.extend(["", "Здесь усреднены медианы и p95 отдельных моделей, а не объединены все отсчёты в одну выборку. Включены нормализация и полный алгоритм вызова, исключены загрузка файла, обработка лица/изображения и сеть. Создание объекта адаптивной модели измерено отдельно. Микросекундные значения зависят от конкретной реализации и компьютера.", "", "![Реальное выполнение моделей](runtime.png)", "",
                  "Самостоятельный fit: один прогрев и три повторения каждого выбранного fixed-варианта seed17 на 734 строках. Каждый повтор заново вычисляет веса людей/участков, нормализацию, медиану расстояний, полное Gram-ядро, опоры и решение. Ни медиана, ни матрицы не переиспользуются. В проекции включён полный fit учителя, в Nyström учитель не строится. Выбор гиперпараметров и сохранение файлов исключены.", "",
                  "| Модель | Полное решение | Медиана fit, мс |", "|---|---|---:|"])
    for row in fits:
        if row["rank"] == 128 or row["family"] == "exact":
            backend = row["exact_backend"] or "учитель не требуется"
            lines.append(f"| {LABELS[row['family']]} | {backend} | {row['median_ms']:.2f} |")
    lines.extend(["", f"Проекция128 с CUDA-учителем стоит {proj_fit:.2f} мс против {nys_fit:.2f} мс у Nyström128. Текущая реализация даже компактного метода материализует полную fit-матрицу: результат на 734 строках не доказывает такую же эффективность на миллионах примеров.", "",
                  f"В первичной серии было **{cost['inner_readouts']} внутренних и {cost['final_readouts']} финальных аналитических readout-конфигураций**, всего 4428 в 12 банках. Банки переиспользуют ядра, опоры и разложения. Их суммарное fit-время {cost['total_bank_fit_seconds']:.3f} с, с запросами и выгрузкой внутри банков — {cost['total_bank_wall_seconds']:.3f} с. Это не 4428 независимых полных запусков обучения.", "",
                  f"Дополнительный полный процессный повтор занял **{workflow['wall_seconds']:.2f} с**: запуск Python/CUDA, чтение и проверка источников, все внутренние банки, выбор, финальные банки, внешний расчёт и запись. Одна повторная оценка времени, файловый кэш мог быть прогрет. В неё не входят независимый аудит и построение отчёта. Сравнены {workflow['compared_selected_and_evaluated_files']} выбранных/оценочных файлов, максимальное отличие массивов {workflow['maximum_replay_array_drift']:.3g}. Повтор не считается новой серией гипотез.", "",
                  "## Проверка и пределы результата", "",
                  f"Независимый численный аудит проверил {audit['checks']['bank_model_normalizers_and_centers']} нормализаций/наборов опор, {audit['checks']['oof_probe_rows']} внутренних контрольных строк, 51 первичный выбор и 9 выборов дополнительных политик. 123 модели заново решены через прямую систему/SVD; максимальное отличие Lab-компоненты {audit['maximum_independent_svd_refit_component_drift']:.3g}. Пересчитаны 150 финальных моделей и {audit['checks']['final_prediction_rows']} строк прогнозов; дрейф независимого предиктора {audit['maximum_independent_prediction_component_drift']:.3g}. Это отдельная реализация проверки в этом проекте, не внешняя лабораторная репликация.", "",
                  "Использован только исходный TRAIN: 966 изображений/24 человека, числовые признаки цвета без загрузки изображений. Внутренние разбиения и нормализация разделены по людям. Старые validation/calibration/test исключены. Все три роли уже многократно использованы и перекрываются; получить из них независимую проверку повторным запуском нельзя.", "",
                  "Обучение на палитре из предыдущей серии остаётся отдельным контролем: небольшая польза чистой палитры не стала устойчивым выигрышем перед более долгим обучением на реальной коже. K-серия не использует palette pretraining и не приписывает ему свои результаты. Библиотечные методы Nyström и RPCholesky известны; это измеренная инженерная работа, не доказанная новизна изобретения. Код, веса и данные имеют отдельные условия использования.", "",
                  "Источники метода: [Rudi et al., Nyström regularization](https://arxiv.org/abs/1507.04717), [Chen et al., randomized pivoted Cholesky](https://arxiv.org/abs/2207.06503). [Зафиксированный протокол](../../research/chromaseed_kernel_v1_protocol.md) · [аудит](audit.json) · [время](runtime.json) · [полный повтор](complete_workflow_runtime.json) · [воспроизведение](reproduce.md) · [карточка модели](../../architecture/chromaseed_kernel_model_card.md).", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.run, args.output)


if __name__ == "__main__":
    main()
