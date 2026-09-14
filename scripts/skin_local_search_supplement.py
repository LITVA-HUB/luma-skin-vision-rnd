"""Publish aggregate ablations and a standalone scientific comparison figure."""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/runs/skin_local_search_v1"
OUT = ROOT / "docs/benchmarks/skin_local_search_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    summary = read(OUT / "summary.json")
    precision = read(RUN / "precision/evaluation.json")
    compact = read(ROOT / "experiments/runs/skin_local_search_compact_v1/evaluation.json")
    compact_inner = read(ROOT / "experiments/runs/skin_local_search_compact_v1/aggregate.json")
    rows = []
    for p in ("mixed", "slr_to_ipod", "ipod_to_slr"):
        for m in ("ridge", "krr", "random_rbf", "guided_rbf", "mlp"):
            for f in ("fp32_reference", "fp16", "int8"):
                a = [r for r in precision["models"] if r["protocol"] == p and r["method"] == m and r["format"] == f]
                rows.append({"protocol": p, "method": m, "format": f,
                             "person_mean": float(np.mean([r["person_delta_e"] for r in a])),
                             "numeric_bytes": a[0]["stored_numeric_bytes"],
                             "max_prediction_drift_delta_e00": max(r["deviation_from_fp32"]["delta_e00_max"] for r in a)})
    (OUT / "precision_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for name in ("runtime_device_audit.json", "historical_reference_audit.json"):
        (OUT / name).write_bytes((RUN / name).read_bytes())
    (OUT / "compact_inner_summary.json").write_text(json.dumps(compact_inner, indent=2), encoding="utf-8")
    # Compact receipts contain aggregate metrics only; no row identifiers are copied.
    compact_rows = []
    for p in ("mixed", "slr_to_ipod", "ipod_to_slr"):
        for m in ("random_rbf", "guided_rbf"):
            for count in (8, 16, 32, 64):
                a = [r for r in compact["models"] if r["protocol"] == p and r["family"] == m and r["atoms"] == count]
                if len(a) != 3:
                    raise ValueError("Expected three compact seeds")
                compact_rows.append({"protocol": p, "method": m, "count": count,
                                     "person_mean": float(np.mean([r["outer_metrics"]["person_mean"] for r in a])),
                                     "image_mean": float(np.mean([r["outer_metrics"]["image_mean"] for r in a])),
                                     "numeric_bytes": compact_inner["protocols"][p][m]["numeric_bytes_by_count"][str(count)],
                                     "selected_by_inner": a[0]["selected_by_inner"]})
    (OUT / "compact_summary.json").write_text(json.dumps(compact_rows, indent=2), encoding="utf-8")
    lines = ["# Сокращение модели и разрядности", "",
             "Дополнительные опыты задуманы после внутреннего подбора основной серии, до внешней проверки. Все варианты сохранены. Выбор по внешней ошибке не выполнялся.", "",
             "## Число элементов", "",
             "Из обученных моделей на 64 элемента взяты первые K элементов и заново аналитически рассчитаны выходные веса. Порядок, нормализация и alpha исходного подбора сохранены. В стоимость входят исходный поиск 64 элементов и дополнительный пересчёт; это не замер обучения K элементов с нуля.", "",
             "| Протокол | Метод | K | Байты FP32 | Отложенные люди ΔE00 | Выбран K внутренней проверкой |", "|---|---|---:|---:|---:|---|"]
    for r in compact_rows:
        lines.append(f"| {r['protocol']} | {r['method']} | {r['count']} | {r['numeric_bytes']} | {r['person_mean']:.4f} | {'да' if r['selected_by_inner'] else ''} |")
    lines += ["", "## Разрядность хранения", "",
              "FP16 хранит все числовые массивы в 16 битах. INT8 хранит веса/центры с масштабами, нормализацию — в FP32. Перед вычислением оба варианта разворачиваются в FP32. Уменьшение файла не равно уменьшению рабочей памяти или ускорению native INT8/FP16 kernels.", "",
              "Drift — отклонение предсказания от исходного FP32, а не ошибка относительно прибора. Таблица содержит максимум среди изображений и seeds. Формат не выбирался по этой проверке.", "",
              "| Протокол | Метод | Хранение | Числовые байты | Отложенные люди ΔE00 | Максимальный drift ΔE00 |", "|---|---|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['protocol']} | {r['method']} | {r['format']} | {r['numeric_bytes']} | {r['person_mean']:.4f} | {r['max_prediction_drift_delta_e00']:.4f} |")
    lines += ["", "[Протокол сокращения](../../research/skin_local_search_compact_protocol.md) · [Протокол разрядности](../../research/skin_local_search_precision_protocol.md) · [Основной отчёт](report.md)", ""]
    (OUT / "ablations.md").write_text("\n".join(lines), encoding="utf-8")

    methods = ("ridge", "krr", "random_rbf", "guided_rbf", "mlp")
    labels = ("Линейная", "Полное ядро", "Случайные RBF", "Поиск по ошибке", "Нейросеть")
    colors = ("#b7bec8", "#6e8495", "#87bdba", "#137e8c", "#cf8145")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), sharey=True, layout="constrained")
    for ax, p, title in zip(axes, ("mixed", "slr_to_ipod", "ipod_to_slr"),
                           ("Смешанные камеры · 18 → 6 людей", "SLR → iPod · 8 → 16 людей", "iPod → SLR · 16 → 8 людей")):
        values = [next(r["person_mean"] for r in summary["rows"] if r["protocol"] == p and r["method"] == m) for m in methods]
        bars = ax.bar(np.arange(5), values, color=colors, width=.75)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
        ax.set_xticks(np.arange(5), labels, rotation=35, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 12)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Ошибка цвета ΔE00 · меньше лучше")
    fig.suptitle("Быстрое обучение удалось; перенос между камерами остаётся слабым", fontsize=14)
    fig.supxlabel("Исходный TRAIN, повторно использованные группы · исследовательский результат · среднее отдельных моделей по seeds", fontsize=9)
    fig.savefig(OUT / "comparison.png", dpi=170)
    fig.savefig(OUT / "comparison.svg")
    plt.close(fig)
    print("Aggregate ablations and comparison.png/.svg generated.")


if __name__ == "__main__":
    main()
