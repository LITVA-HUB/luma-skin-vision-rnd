"""Aggregate ChromaSeed transfer evidence, retaining controls and learning curves."""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/runs/chromaseed_v1"
FROZEN = ROOT / "experiments/runs/chromaseed_frozen_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_v1"
PROTOCOLS = ("mixed", "slr_to_ipod", "ipod_to_slr")
ARMS = ("scratch", "clean_palette", "rendered_palette", "shuffled_palette", "skin_long")
LABELS = {"scratch": "С нуля", "clean_palette": "Чистая палитра", "rendered_palette": "Палитра + камера",
          "shuffled_palette": "Перемешанные ответы", "skin_long": "Дольше на коже", "random_basis": "Случайная основа"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(records, primary=True):
    keys = sorted({(r["protocol"], r["arm"], r.get("step", 0)) for r in records})
    rows = []
    for protocol, arm, step in keys:
        matched = [r for r in records if (r["protocol"], r["arm"], r.get("step", 0)) == (protocol, arm, step)]
        row = {"protocol": protocol, "arm": arm, "step": step, "seeds": len(matched),
               "person_mean": float(np.mean([r["metrics"]["person_mean"] for r in matched])),
               "image_mean": float(np.mean([r["metrics"]["image_mean"] for r in matched])),
               "p90": float(np.mean([r["metrics"]["p90"] for r in matched])),
               "numeric_bytes": matched[0]["numeric_bytes"], "serialized_bytes": matched[0]["serialized_bytes"],
               "seed_person_means": [r["metrics"]["person_mean"] for r in matched]}
        if primary:
            row["adapt_seconds"] = float(np.mean([r["adapt_seconds"] for r in matched]))
            row["inherited_fit_seconds"] = float(np.mean([r["inherited_fit_seconds"] for r in matched]))
        else:
            row["fit_seconds"] = float(np.mean([r["fit_seconds"] for r in matched]))
        rows.append(row)
    return rows


def main():
    global RUN, FROZEN, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--frozen-run", type=Path, default=FROZEN)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    RUN, FROZEN, OUT = args.run, args.frozen_run, args.output
    OUT.mkdir(parents=True, exist_ok=True)
    evaluation = read(RUN / "evaluation.json")
    choices = read(RUN / "frozen_selections.json")
    primary_rows = aggregate(evaluation["models"])
    frozen_rows = aggregate(read(FROZEN / "evaluation.json")["models"], primary=False)
    pretraining = read(RUN / "pretrain_manifest.json")
    synthetic = []
    for arm in ARMS[1:4]:
        rows = [r for r in pretraining["models"] if r["arm"] == arm]
        synthetic.append({"arm": arm, "pretrain_seconds": float(np.mean([r["fit_seconds"] for r in rows])),
                          **{style: float(np.mean([r["synthetic_validation"][style]["mean_delta_e00"] for r in rows]))
                             for style in ("clean", "rendered")}})
    zero_shot = []
    for protocol in PROTOCOLS:
        for arm in ARMS[1:4]:
            rows = [r for r in evaluation["zero_shot_diagnostic"] if r["protocol"] == protocol and r["arm"] == arm]
            zero_shot.append({"protocol": protocol, "arm": arm,
                              "person_mean": float(np.mean([r["metrics"]["person_mean"] for r in rows]))})
    trace_records = [read(p) for p in RUN.glob("*/inner/fold*/*/trace.json")]
    final_trace_records = [read(p) for p in RUN.glob("*/final/*/trace.json")]
    costs = {"palette_preparation_seconds": pretraining["palette_generation_seconds"],
             "synthetic_pretrain_seconds_sum": sum(r["fit_seconds"] for r in pretraining["models"]),
             "real_trace_count": len(trace_records) + len(final_trace_records),
             "real_trace_fit_seconds_sum": sum(r["fit_seconds"] for r in trace_records + final_trace_records),
             "real_fit_phase_wall_seconds": read(RUN / "fit_completion.json")["seconds_this_invocation"],
             "frozen_head_fit_phase_wall_seconds": read(FROZEN / "fit_completion.json")["seconds"]}
    summary = {"model_family": "Luma ChromaSeed v1", "evidence": evaluation["evidence"],
               "full_finetuning": primary_rows, "frozen_readout": frozen_rows,
               "synthetic_diagnostics": synthetic, "zero_shot_diagnostics": zero_shot, "costs": costs}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name in ("source_lock.json", "frozen_selections.json", "pretrain_manifest.json"):
        (OUT / name).write_bytes((RUN / name).read_bytes())
    (OUT / "frozen_readout_source_lock.json").write_bytes((FROZEN / "source_lock.json").read_bytes())
    lines = ["# Luma ChromaSeed v1 — палитра → кожа", "",
             "Исследовательское имя модели: **Luma ChromaSeed**. Размер всех сравниваемых вариантов — 2749 чисел с нормализацией, 10 996 байт FP32. Учим цветовую основу на синтетических поверхностях, затем проверяем перенос на инструментально размеченные участки кожи.", "",
             "Это не подтверждение точности на обычных селфи: используются только исходные TRAIN966 фото/24 человека и уже известные пересекающиеся группы. Старые validation/calibration/test не открывались. Ни изображения, ни веса не отправлялись внешним сервисам.", "",
             "## Предобучение: только синтетика", "",
             "32 768 обучающих и 4096 отдельных проверочных цветов. Цель — исходный Lab поверхности под D65, наблюдение — чистый или искажённый RGB. Симулятор приблизителен и не моделирует спектральную физику кожи/реального сенсора. Эти ошибки нельзя сравнивать напрямую с ошибками реальной кожи.", "",
             "| Основа | Чистая синтетическая проверка ΔE00 | С искажениями ΔE00 | Предобучение одной модели, с |", "|---|---:|---:|---:|"]
    for r in synthetic:
        lines.append(f"| {LABELS[r['arm']]} | {r['clean']:.3f} | {r['rendered']:.3f} | {r['pretrain_seconds']:.3f} |")
    lines += ["", "![Пример палитры и искажений](palette.png)", "",
              "## Перенос с дообучением всей сети", "",
              "Основная точка — 1024 шага адаптации; batch256. Скорость обучения выбрана только по person-held-out внутренним фолдам, усредняя3 seeds. Предобученные варианты получают ещё2048 шага синтетики. Контроль «Дольше на коже» получает2048 дополнительных шагов реальных данных текущего fit, затем тот же сброс оптимизатора и1024 шага. «Перемешанные ответы» сохраняет вычислительный бюджет синтетики, но разрушает правильное соответствие цвета и ответа.", "",
              "Ошибка — инструментальный native Lab ΔE00, сначала средняя по человеку, затем по людям; меньше лучше. Значения усредняют отдельные модели по seeds, а не ансамбль.", "",
              "| Протокол | Вариант | Внутренний ΔE00 | Отложенные люди ΔE00 | По снимкам | Адаптация, с | До неё, с |", "|---|---|---:|---:|---:|---:|---:|"]
    for protocol in PROTOCOLS:
        for arm in ARMS:
            r = next(v for v in primary_rows if v["protocol"] == protocol and v["arm"] == arm and v["step"] == 1024)
            inner = choices["protocols"][protocol][arm]["inner_person_mean"]
            lines.append(f"| {protocol} | {LABELS[arm]} | {inner:.3f} | {r['person_mean']:.3f} | {r['image_mean']:.3f} | {r['adapt_seconds']:.3f} | {r['inherited_fit_seconds']:.3f} |")
    lines += ["", "![Кривые адаптации](transfer_curves.png)", "",
              "На графике показаны дополнительные шаги адаптации, а не полная стоимость обучения. LR для всех трёх точек один, выбран по внутренней1024-точке; ранние точки не имеют отдельного подбора. Предобучение и дополнительная реальная тренировка не бесплатны.", "",
              "## Замороженная основа + аналитические выходные веса", "",
              "Здесь скрытый слой вообще не дообучается на коже. Веса выхода вычисляются через ridge, а вспомогательная нормализация признаков поглощается весами. Объём модели остаётся прежним. Случайная основа — обязательный контроль: если она не хуже цветовой, выигрыш нельзя приписать палитре.", "",
              "| Протокол | Основа | Отложенные люди ΔE00 | CPU fit, мс |", "|---|---|---:|---:|"]
    for r in frozen_rows:
        lines.append(f"| {r['protocol']} | {LABELS[r['arm']]} | {r['person_mean']:.3f} | {r['fit_seconds']*1000:.3f} |")
    lines += ["", "Время analytic-head серии записывалось при параллельно работающем основном GPU-эксперименте. Это ориентир реализации, не выделенный benchmark CPU. Код NumPy float64, один поток BLAS. Общая стоимость цветового предобучения учитывается отдельно.", "",
              "## Цена эксперимента и воспроизводимость", "",
              f"Генерация палитры: {costs['palette_preparation_seconds']:.2f} с; сумма9 synthetic fits: {costs['synthetic_pretrain_seconds_sum']:.2f} с. {costs['real_trace_count']} реальных training traces (включая отдельные warmups), сумма fit-времени {costs['real_trace_fit_seconds_sum']:.2f} с, wall time основной real-fit фазы {costs['real_fit_phase_wall_seconds']:.2f} с. Вторичная frozen-head фаза: {costs['frozen_head_fit_phase_wall_seconds']:.2f} с. Это измерения RTX4060/локального CPU по готовым36 признакам, без извлечения лица и фотографических признаков.", "",
              "Никакого выбора renderer strength, числа pretrain steps или итогового варианта по внешним данным не выполнялось. Сохранены отрицательные контроли,3 seeds, кривые и zero-shot диагностика. Подтверждение на новых лицах и новом освещении остаётся отдельной задачей.", "",
              "Связанные идеи известны: [domain randomization](https://arxiv.org/abs/1703.06907); [нелинейная обработка камеры после white balance](https://openaccess.thecvf.com/content_CVPR_2019/html/Afifi_When_Color_Constancy_Goes_Wrong_Correcting_Improperly_White-Balanced_Images_CVPR_2019_paper.html). Работы не воспроизводились целиком, сторонние модели/данные не загружались. Название рабочее, наличие прав на товарный знак не проверялось.", "",
              "[Основной протокол](../../research/chromaseed_v1_protocol.md) · [Протокол замороженной основы](../../research/chromaseed_frozen_protocol.md) · [Точные агрегаты](summary.json) · [Независимый аудит](audit.json)", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    with np.load(RUN / "palette/validation.npz", allow_pickle=False) as d:
        rgb, observed = d["canonical_rgb"], d["observed_rgb"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 2.8), layout="constrained")
    indices = np.arange(18) * 37
    for ax, values, title in zip(axes, (rgb, observed), ("Исходные цвета поверхности — правильные ответы", "Те же цвета после имитации света и камеры — наблюдения")):
        ax.imshow(values[indices][None], aspect="auto", interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("SYNTHETIC · цветовые образцы, не фотографии кожи", fontsize=10)
    fig.savefig(OUT / "palette.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    colors = ("#64748b", "#0d9488", "#2563eb", "#9ca3af", "#d97706")
    for ax, protocol in zip(axes, PROTOCOLS):
        for arm, color in zip(ARMS, colors):
            rows = sorted([r for r in primary_rows if r["protocol"] == protocol and r["arm"] == arm], key=lambda r: r["step"])
            ax.plot([r["step"] for r in rows], [r["person_mean"] for r in rows], "o-", label=LABELS[arm], color=color)
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 256, 1024], ["64", "256", "1024"])
        ax.set_xlabel("Шаги адаптации к коже")
        ax.set_ylabel("Отложенные люди · ΔE00")
        ax.set_title(protocol)
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=9)
    fig.suptitle("Luma ChromaSeed · перенос палитры на кожу\nПовторно использованные TRAIN-группы, исследовательская проверка", fontsize=12)
    fig.savefig(OUT / "transfer_curves.png", dpi=170)
    fig.savefig(OUT / "transfer_curves.svg")
    plt.close(fig)
    print("ChromaSeed aggregate report and figures saved.")


if __name__ == "__main__":
    main()
