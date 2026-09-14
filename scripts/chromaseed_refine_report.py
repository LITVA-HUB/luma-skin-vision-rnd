"""Aggregate the locked ChromaSeed-R study without selecting from outer data."""

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
LABELS = {"stats_mlp": "Statistics MLP", "patch_mlp": "Patch MLP", "recur_soft": "Refine all 64",
          "recur_top16": "Refine top 16", "recur_dynamic": "Refine dynamic"}
ROLES = {"mixed": "Mixed cameras, 18/6 people", "slr_to_ipod": "SLR → iPod, 8/16 people", "ipod_to_slr": "iPod → SLR, 16/8 people"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize(run, output):
    results = read_json(run / "results.json")
    choices = read_json(run / "selections.json")
    audit = read_json(output / "audit.json")
    timing = read_json(output / "runtime.json")
    if not audit["passed"] or audit["selection_sha256"] != sha(run / "selections.json"):
        raise ValueError("passing audit of these exact selections required")
    rows = []
    bank_costs = []
    for receipt_path in sorted((run / "inner").glob("*/*/*/receipt.json")) + sorted((run / "final").glob("*/*/receipt.json")):
        receipt = read_json(receipt_path)
        bank_costs.append({"stage": receipt_path.relative_to(run).parts[0], "role": receipt["role"], "family": receipt["family"],
                           "steps": receipt["steps"], "n_models": len(receipt["slots"]), "bank_seconds": receipt["bank_total_seconds"],
                           "setup_seconds": receipt["setup_seconds"], "peak_allocated_bytes": receipt["cuda_peak_allocated_bytes"]})
    for role in ROLES:
        for family in LABELS:
            records = [r for r in results["records"] if r["role"] == role and r["family"] == family]
            selected = choices["roles"][role][family]
            final_cost = next(c for c in bank_costs if c["stage"] == "final" and c["role"] == role and c["family"] == family)
            runtime = [r for r in timing["records"] if r["role"] == role and r["family"] == family]
            row = {"role": role, "family": family, "seeds": 3, "lr": selected["lr"], "steps": selected["checkpoint"],
                   "exit_threshold": selected["exit_threshold"], "inner_person_mean": selected["selected_fixed"]["person_mean"],
                   "learned_parameters": records[0]["learned_parameters"], "numeric_payload_bytes": records[0]["numeric_payload_bytes"],
                   "archive_bytes_mean": float(np.mean([r["archive_bytes"] for r in records])),
                   "mean_passes": float(np.mean([r["mean_passes"] for r in records])),
                   "mean_connections_full_trace": float(np.mean([r["mean_connections_full_trace"] for r in records])),
                   "final_bank_seconds_three_models": final_cost["bank_seconds"],
                   "final_amortized_seconds_per_fit": final_cost["bank_seconds"] / 3,
                   "pass_person_means": np.mean([[p["person_mean"] for p in r["pass_metrics"]] for r in records], axis=0).tolist()}
            for policy in ("fixed", "adaptive"):
                for metric in ("person_mean", "image_mean", "site_person_mean", "p90", "gt5", "gt10"):
                    row[f"{policy}_{metric}"] = float(np.mean([r[f"{policy}_metrics"][metric] for r in records]))
                row[f"{policy}_seed_person_means"] = [r[f"{policy}_metrics"]["person_mean"] for r in records]
                row[f"{policy}_cpu_median_us_mean"] = float(np.mean([r["median_us"] for r in runtime if r["policy"] == policy]))
                row[f"{policy}_cpu_p95_us_mean"] = float(np.mean([r["p95_us"] for r in runtime if r["policy"] == policy]))
            if family == "recur_dynamic":
                diagnostics = []
                for seed in (17, 29, 43):
                    with np.load(run / "evaluated" / role / family / f"seed{seed}.npz", allow_pickle=False) as z:
                        counts = z["connections"]
                    diagnostics.append({"seed": seed, "unique_counts_per_pass": [len(np.unique(counts[:, p])) for p in range(4)],
                                        "input_count_std_per_pass": counts.std(0).tolist(), "min": float(counts.min()),
                                        "max": float(counts.max()), "fraction_minimum4": float(np.mean(counts == 4)),
                                        "fraction_all64": float(np.mean(counts == 64))})
                row["dynamic_connection_diagnostics"] = diagnostics
            rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"family": "Luma ChromaSeed-R v1", "evidence": results["evidence"], "rows": rows,
               "source_lock_sha256": results["source_lock_sha256"], "selection_sha256": results["selection_sha256"],
               "cost": {"inner_banks": sum(c["stage"] == "inner" for c in bank_costs),
                        "inner_model_traces": sum(c["n_models"] for c in bank_costs if c["stage"] == "inner"),
                        "final_banks": sum(c["stage"] == "final" for c in bank_costs),
                        "final_refits": sum(c["n_models"] for c in bank_costs if c["stage"] == "final"),
                        "total_bank_training_seconds": sum(c["bank_seconds"] for c in bank_costs),
                        "maximum_cuda_allocated_bytes": max(c["peak_allocated_bytes"] for c in bank_costs),
                        "optimizer_updates_across_independent_models": sum(c["steps"] * c["n_models"] for c in bank_costs)},
               "bank_costs": bank_costs}
    write_json(output / "summary.json", summary)
    for name in ("source_lock.json", "selections.json"):
        shutil.copyfile(run / name, output / name)
    plot(summary, choices, output)
    report(summary, audit, output)
    return summary


def plot(summary, choices, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    reference = read_json(ROOT / "docs/benchmarks/skin_local_search_v1/summary.json")["rows"]
    for ax, role in zip(axes, ROLES, strict=True):
        rows = [r for r in summary["rows"] if r["role"] == role]
        values = np.array([r["adaptive_person_mean"] for r in rows])
        low = values - np.array([min(r["adaptive_seed_person_means"]) for r in rows])
        high = np.array([max(r["adaptive_seed_person_means"]) for r in rows]) - values
        ax.bar(np.arange(5), values, color=["#677587", "#21918c", "#4262a8", "#9276ab", "#ef9b24"])
        ax.errorbar(np.arange(5), values, yerr=np.stack((low, high)), fmt="none", color="#26334b", capsize=4)
        krr = next(r["person_mean"] for r in reference if r["protocol"] == role and r["method"] == "krr")
        guided = next(r["person_mean"] for r in reference if r["protocol"] == role and r["method"] == "guided_rbf")
        ax.axhline(krr, ls="--", color="#203040", lw=1.2, label="Previous KRR")
        ax.axhline(guided, ls=":", color="#aa4455", lw=1.2, label="Previous guided RBF")
        ax.set_xticks(np.arange(5), [LABELS[r["family"]] for r in rows], rotation=28, ha="right")
        ax.set_title(ROLES[role])
        ax.set_ylim(0, max(float((values + high).max()), krr, guided) * 1.18)
        ax.set_ylabel("Person-balanced native ΔE00 ↓")
        ax.grid(axis="y", alpha=.15)
    axes[0].legend(fontsize=8, loc="upper left")
    figure.suptitle("ChromaSeed-R: exploratory reused TRAIN roles; bars average 3 seeds, whiskers show seed range", fontsize=12)
    figure.savefig(output / "accuracy.png", dpi=160)
    figure.savefig(output / "accuracy.svg")
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, role in zip(axes, ROLES, strict=True):
        for family in LABELS:
            candidates = choices["roles"][role][family]["candidates"]
            checkpoints = [512, 2048, 8192]
            best_at_budget = [min(c["person_mean"] for c in candidates if c["checkpoint"] == step) for step in checkpoints]
            ax.plot(checkpoints, best_at_budget, marker="o", label=LABELS[family])
        ax.set_xscale("log", base=2)
        ax.set_xticks(checkpoints, [str(v) for v in checkpoints])
        ax.set_title(ROLES[role])
        ax.set_xlabel("Optimizer updates per model")
        ax.set_ylabel("Inner OOF person ΔE00 ↓")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    figure.suptitle("Longer training: minimum of three registered LRs at each budget; inner data only", fontsize=12)
    figure.savefig(output / "inner_training.png", dpi=160)
    figure.savefig(output / "inner_training.svg")
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for ax, role in zip(axes, ROLES, strict=True):
        for row in summary["rows"]:
            if row["role"] == role and row["family"].startswith("recur_"):
                ax.plot([1, 2, 3, 4], row["pass_person_means"], marker="o", label=LABELS[row["family"]])
        ax.set_xticks([1, 2, 3, 4])
        ax.set_title(ROLES[role])
        ax.set_xlabel("Actually applied shared refinement passes")
        ax.set_ylabel("Exploratory outer person ΔE00 ↓")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    figure.suptitle("Prediction at each internal pass; these outer curves did not select stopping thresholds", fontsize=12)
    figure.savefig(output / "refinement.png", dpi=160)
    figure.savefig(output / "refinement.svg")
    plt.close(figure)


def report(summary, audit, output):
    cost = summary["cost"]
    lines = ["# Luma ChromaSeed-R: результаты серии v1", "",
             "Измеренный исследовательский результат на повторно использованных разбиениях исходного TRAIN. Это не новая независимая проверка на лицах со смартфонов. Чем меньше ΔE00, тем точнее цвет.", "",
             "Повторное уточнение дало небольшой выигрыш на смешанных камерах, но устойчивого превосходства над прежними сильными моделями нет. Во всех 15 сочетаниях семейства и протокола внутренний выбор остановился на 512 шагах; увеличение до 2048/8192 не помогло. На обратном переносе поздние поправки усиливали ошибку. [Вывод и следующий эксперимент](../../research/chromaseed_refine_next_decision.md).", "",
             f"Обучены **{cost['inner_model_traces']} внутренних вариантов и {cost['final_refits']} финальных моделей**. До 8192 шагов на модель, три seed и три скорости обучения. Внутренние разбиения выбирали и скорость, и длительность обучения, затем порог раннего выхода. Суммарное время банков обучения: **{cost['total_bank_training_seconds']:.1f} с**, максимум выделенной CUDA-памяти: **{cost['maximum_cuda_allocated_bytes'] / 1024**2:.1f} МиБ**. Время в сумме не включает полный аудит, выгрузку прогнозов и измерение CPU.", "",
             "## Точность после зафиксированного выбора", "",
             "Среднее ошибок трех отдельно обученных моделей; это не ансамбль. Протоколы перекрываются, seed не заменяют людей.", "",
             "| Модель | Смешанные камеры, 6 человек | SLR → iPod, 16 человек | iPod → SLR, 8 человек | FP32, байт |",
             "|---|---:|---:|---:|---:|"]
    for family in LABELS:
        rr = [next(r for r in summary["rows"] if r["role"] == role and r["family"] == family) for role in ROLES]
        lines.append(f"| {LABELS[family]} | {rr[0]['adaptive_person_mean']:.3f} | {rr[1]['adaptive_person_mean']:.3f} | {rr[2]['adaptive_person_mean']:.3f} | {rr[0]['numeric_payload_bytes']:,} |")
    reference = read_json(ROOT / "docs/benchmarks/skin_local_search_v1/summary.json")["rows"]
    for family in ("krr", "guided_rbf", "random_rbf", "mlp"):
        rr = [next(r for r in reference if r["protocol"] == role and r["method"] == family) for role in ROLES]
        lines.append(f"| Предыдущий {family} | {rr[0]['person_mean']:.3f} | {rr[1]['person_mean']:.3f} | {rr[2]['person_mean']:.3f} | {int(rr[0]['numeric_bytes']):,}* |")
    lines.extend(["", "*Числа старых моделей взяты из ранее проверенного отчета с теми же ролями. Размер KRR зависит от объема fit. Новый размер включает веса, нормализацию, аналитический старт и числовой порог; архив и полный конвейер обработки фото имеют отдельный размер.", "", "![Точность](accuracy.png)", "",
                  "## Настройки и фактическая работа модели", "",
                  "| Протокол | Модель | Шаги / LR | ΔE, 4 прохода или статическая | ΔE, выбранный выход | Проходов | CPU, мкс, фикс. → адапт. | Обучение 3 моделей, с |",
                  "|---|---|---|---:|---:|---:|---:|---:|"])
    for r in summary["rows"]:
        lines.append(f"| {r['role']} | {LABELS[r['family']]} | {r['steps']} / {r['lr']} | {r['fixed_person_mean']:.3f} | {r['adaptive_person_mean']:.3f} | {r['mean_passes']:.2f} | {r['fixed_cpu_median_us_mean']:.1f} → {r['adaptive_cpu_median_us_mean']:.1f} | {r['final_bank_seconds_three_models']:.2f} |")
    lines.extend(["", "CPU: NumPy FP32, один вход, с нормализацией и настоящим прекращением цикла; среднее медиан трех seed, прогрев 20 примеров и три полных повтора всех held-строк. Извлечение признаков, детекция лица, браузер и телефон сюда не входят. Обучение — реальное время банка из трех независимых сетей; деление на три дает амортизированную стоимость, не время одиночного запуска.", "", "![Длительность обучения](inner_training.png)", "", "![Повторное уточнение](refinement.png)", "",
                  "## Динамические связи", "",
                  "Все варианты сначала кодируют и оценивают 64 участка. Маска выбирает связи для объединения признаков; новые обучаемые веса во время ответа не создаются. В NumPy объединяются действительно выбранные участки, но кодирование остается плотным. Уменьшение числа связей само по себе не означает ускорение всего конвейера.", "",
                  "| Протокол | Seed | Число связей, min–max | Доля проходов только с 4 | Разных чисел связей по входам, проходы 1/2/3/4 |",
                  "|---|---:|---:|---:|---|"])
    for row in summary["rows"]:
        for d in row.get("dynamic_connection_diagnostics", []):
            lines.append(f"| {row['role']} | {d['seed']} | {d['min']:.0f}–{d['max']:.0f} | {d['fraction_minimum4']:.1%} | {' / '.join(map(str, d['unique_counts_per_pass']))} |")
    lines.extend(["", "## Проверка и пределы выводов", "",
                  f"Независимый аудит проверил {audit['checks']['artifact_hashes']} хешей файлов, {audit['checks']['normalizer_anchor_sets']} наборов нормализации/аналитического старта, все {audit['checks']['selection_decisions']} решения выбора и {audit['checks']['outer_models']} финальных моделей. Внутренних контрольных прогнозов: {audit['checks']['inner_prediction_probe_rows']}; финальных строк, проверенных отдельной NumPy-реализацией: {audit['checks']['outer_prediction_rows']}. Максимальное расхождение компоненты native Lab с CUDA: {audit['maximum_numpy_vs_cuda_native_lab_component_drift']:.7f}.", "",
                  "Парные интервалы по людям — описательные: выборка мала, данные уже использовались, протоколы перекрываются. Нельзя выдавать эти интервалы за подтверждение нового алгоритма или точности на обычных фото лиц. Модели оценивают цвет подготовленных участков кожи, а не личность человека. Палитровое предобучение из v1 в этой серии не применялось: надежного переноса оно не показало.", "",
                  "| Протокол | A − B | Δ среднего, меньше 0 лучше A | Описательный 95% интервал | Людей лучше A |",
                  "|---|---|---:|---|---:|"])
    for p in audit["paired"]:
        lo, hi = p["descriptive_person_bootstrap_95"]
        lines.append(f"| {p['role']} | {p['a']} − {p['b']} | {p['mean_a_minus_b']:.3f} | [{lo:.3f}, {hi:.3f}] | {p['people_a_better']}/{p['n_people']} |")
    lines.extend(["", "Повторное вычисление и условная маршрутизация — известные идеи: [Adaptive Computation Time](https://arxiv.org/abs/1603.08983), [Universal Transformers](https://arxiv.org/abs/1807.03819), [CondConv](https://arxiv.org/abs/1904.04971). Реализация их сочетания для цвета кожи сама по себе не доказывает научную новизну. Действующие ограничения лицензии датасета и весов не менялись.", "",
                  "Воспроизведение: [reproduce.md](reproduce.md). Полные настройки и машинные результаты: [summary.json](summary.json), [selections.json](selections.json), [audit.json](audit.json), [runtime.json](runtime.json)."])
    if (output / "precision.json").exists():
        lines.extend(["", "Дополнительное сжатие весов: [числа](precision.json). На этих весах арифметика остается FP32 после декодирования; уменьшение файла не означает ускорение вычислений."])
    if (output / "precision.md").exists():
        lines.extend(["", "[Разбор FP16/INT8 и переключения раннего выхода](precision.md)."])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run, args.output)


if __name__ == "__main__":
    main()
