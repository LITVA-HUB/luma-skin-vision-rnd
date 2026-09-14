"""All-policy A evidence: augmentation is not a universal quality improvement."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from chromaseed_kernel_audit import js
from skin_local_search_train import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
LABELS = ("Mixed · 6 people", "SLR → iPod · 16 people", "iPod → SLR · 8 people")
FAMILIES = ("norm_static", "norm_joint_soft", "perceptual_static", "perceptual_joint_soft")
CONTROLS = ("g_norm_base", "g_norm_soft", "g_perceptual_base", "g_perceptual_soft", "constant")


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run, args.output
    result, selected, workflow = (
        js(run / p) for p in ("results.json", "selections.json", "workflow.json")
    )
    audit, runtime = (js(out / p) for p in ("audit.json", "runtime.json"))
    assert audit["passed"] and audit["results_sha256"] == sha(run / "results.json")
    assert runtime["audit_sha256"] == sha(out / "audit.json")
    for path, h in audit["dependencies"].items():
        assert sha(ROOT / path) == h
    groups = [(f, p) for f in FAMILIES for p in ("clean", "guarded")] + [
        (f, "reference") for f in CONTROLS
    ]
    rows, doses, candidates = [], [], []
    for role in ROLES:
        for family, policy in groups:
            rr = [
                r
                for r in result["records"]
                if (r["role"], r["family"], r["policy"]) == (role, family, policy)
            ]
            tt = [
                r
                for r in runtime["records"]
                if (r["role"], r["family"], r["policy"]) == (role, family, policy)
            ]
            ff = [
                r
                for r in runtime["standalone_fit_records"]
                if (r["role"], r["family"], r["policy"]) == (role, family, policy)
            ]
            assert len(rr) == len(tt) == (1 if family == "constant" else 3)
            rows.append(
                dict(
                    role=role,
                    family=family,
                    policy=policy,
                    seeds=len(rr),
                    people=rr[0]["metrics"]["n_people"],
                    images=rr[0]["metrics"]["n_images"],
                    alpha=rr[0].get("alpha"),
                    eta=rr[0].get("eta"),
                    clean_person_mean=float(np.mean([r["metrics"]["person_mean"] for r in rr])),
                    clean_image_mean=float(np.mean([r["metrics"]["image_mean"] for r in rr])),
                    clean_p90=float(np.mean([r["metrics"]["p90"] for r in rr])),
                    stress4_person_mean=float(
                        np.mean([r["doses"][1]["worst_error"]["person_mean"] for r in rr])
                    ),
                    drift4_person_mean=float(
                        np.mean([r["doses"][1]["worst_drift"]["person_mean"] for r in rr])
                    ),
                    active_gate=rr[0]["active_gate"],
                    numeric_bytes=rr[0]["numeric_bytes"],
                    cached_array_bytes=tt[0]["numpy_only"]["cached_array_bytes"],
                    median_of_seed_median_us=float(
                        np.median([r["numpy_only"]["median_us"] for r in tt])
                    ),
                    max_seed_p95_us=float(max(r["numpy_only"]["p95_us"] for r in tt)),
                    seed17_full_fit_ms=None if not ff else ff[0]["median_seconds"] * 1000,
                )
            )
            for i, t in enumerate((1 / 255, 4 / 255, 16 / 255, 64 / 255)):
                doses.append(
                    dict(
                        role=role,
                        family=family,
                        policy=policy,
                        dose=t,
                        **{
                            k: float(np.mean([r["doses"][i][k]["person_mean"] for r in rr]))
                            for k in ("worst_error", "worst_drift", "worst_error_change")
                        },
                    )
                )
        for family in FAMILIES:
            selection = selected["roles"][role][family]
            for c in selection["candidates"]:
                candidates.append(
                    dict(
                        role=role,
                        family=family,
                        alpha=c["alpha"],
                        eta=c["eta"],
                        clean=c["clean"],
                        robust=c["robust"],
                        feasible=c["clean"] <= selection["zero_eta_anchor_clean"] + 0.05,
                        **{
                            f"selected_{p}": (c["alpha"], c["eta"])
                            == (selection["policies"][p]["alpha"], selection["policies"][p]["eta"])
                            for p in ("clean", "guarded")
                        },
                    )
                )
    assert len(rows) == 39 and len(doses) == 156 and len(candidates) == 144
    csv_write(out / "all_policies.csv", rows)
    csv_write(out / "all_doses.csv", doses)
    csv_write(out / "all_inner_candidates.csv", candidates)

    def one(role, family, policy):
        return next(
            r for r in rows if (r["role"], r["family"], r["policy"]) == (role, family, policy)
        )

    joint = one("mixed", "perceptual_joint_soft", "clean")
    base = one("mixed", "perceptual_static", "clean")
    old = one("mixed", "g_perceptual_soft", "reference")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    plot_cases = (
        ("perceptual_static", "clean", "Static"),
        ("g_perceptual_soft", "reference", "G soft"),
        ("perceptual_joint_soft", "clean", "Joint η=0"),
        ("perceptual_joint_soft", "guarded", "Joint η=.75"),
    )
    for j, (role, label) in enumerate(zip(ROLES, LABELS, strict=True)):
        for i, key in enumerate(("clean_person_mean", "stress4_person_mean")):
            values = [one(role, f, p)[key] for f, p, _ in plot_cases]
            ax = axes[i, j]
            ax.plot(range(4), values, color="#81929e", lw=1)
            ax.scatter(
                range(4), values, c=["#48627a", "#6d81a2", "#148879", "#df8f33"], s=60, zorder=3
            )
            for k, v in enumerate(values):
                ax.annotate(
                    f"{v:.3f}",
                    (k, v),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )
            ax.set_xticks(range(4), [r[2] for r in plot_cases], rotation=15)
            low, high = min(values), max(values)
            gap = max(high - low, 0.03)
            ax.set_ylim(low - 0.35 * gap, high + 0.6 * gap)
            ax.grid(axis="y", alpha=0.2)
            if i == 0:
                ax.set_title(label)
            if j == 0:
                ax.set_ylabel("Clean ΔE00 ↓" if i == 0 else "Worst of 8 at 4/255 · ΔE00 ↓")
    fig.suptitle(
        "Luma ChromaSeed-A · matched perceptual controls\nReused TRAIN; mean errors across seeds; each panel has its own vertical scale",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    fig.text(
        0.5,
        0.012,
        "Synthetic affine stress preserves source targets by assumption. No new phone-face measurements; no independent confidence claim.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(out / "affine_quality.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (role, label) in zip(axes, zip(ROLES, LABELS, strict=True), strict=True):
        for f, p, label2 in plot_cases:
            dd = [r for r in doses if (r["role"], r["family"], r["policy"]) == (role, f, p)]
            ax.plot(
                [1, 4, 16, 64], [d["worst_error"] for d in dd], marker="o", label=label2, lw=1.4
            )
        ax.set_xscale("log", base=4)
        ax.set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
        ax.set_xlabel("Mixture amount × 255")
        ax.set_title(label)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Worst-of-eight person ΔE00 ↓")
    axes[1].legend(fontsize=8)
    fig.suptitle("All registered doses · no dose selected from final results", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "affine_all_doses.png", dpi=170)
    plt.close(fig)
    header = "# Luma ChromaSeed-A: совместное обучение и цветовые искажения\n\n"
    text = (
        header
        + f"Серия завершена и независимо проверена. На смешанном разбиении совместное обучение плавной поправки даёт **{joint['clean_person_mean']:.6f} ΔE00** против {old['clean_person_mean']:.6f} у прежнего G soft и {base['clean_person_mean']:.6f} у статической основы. Это небольшой исследовательский результат на шести ранее использованных людях. При выборе по обычной ошибке все12 решений оставили **η=0**, то есть без добавления искажённых цветов. Польза здесь связана с совместным обучением коэффициентов, а не с подтверждённым переносом палитры на лица.\n\n"
    )
    text += f"Активные веса занимают **{joint['numeric_bytes']:,} байта**, массивы потребителя — {joint['cached_array_bytes']:,} байта; ответ по готовому вектору36 признаков — **{joint['median_of_seed_median_us']:.1f} мкс**, полное обучение на734 исходных строках — **{joint['seed17_full_fit_ms']:.2f} мс** (CPU Ryzen9 7900X, один поток). Это не время обработки фотографии и не замер на телефоне. Все112 повторных полных обучений точно воспроизвели сохранённые массивы.\n\n"
    text += "## Что именно проверялось\n\nЧетыре семейства: статическое и совместное плавное, каждое с обычной и общей перцепционной квадратичной функцией потерь.128 общих опорных цветов; исходные нормировки, ширина ядра, центры и классификатор условий съёмки обучаются только на исходных строках обучения. Совместная модель обучает сразу оба блока [Z,s(x)Z], без цикла обратного распространения ошибки. На разбиениях с одной камерой она точно совпадает с соответствующей новой статической моделью. Это свойство состава обучающих данных; модель не определяет автоматически неизвестный телефон.\n\n"
    text += "16 цифровых вариантов на исходную строку: восемь углов RGB-куба × дозы1/255 и4/255. Квантили, средний цвет и стандартные отклонения преобразуются согласованно; исходная инструментальная цель сохраняется как синтетическое допущение. Общий вес каждого исходного примера неизменен. Новых людей, фотографий и физических измерений не добавлено. Исследована доля искажений η∈{0,.25,.5,.75}, α∈{.1,1,10}, три случайных базиса. Полная сетка, включая сильную регуляризацию, сохранена.\n\n"
    text += "Две заранее заданные политики: clean минимизирует обычную внутреннюю ошибку; guarded минимизирует худшую ошибку восьми искажений4/255, допуская внутреннюю обычную ошибку не более чем на0.05 выше лучшего η=0. Все12 clean выбрали η=0, все12 guarded — η=.75, α=.1 везде. Ограничение0.05 относится к внутреннему выбору, **не гарантирует** такое же ухудшение на отложенных людях.\n\n"
    text += "## Сопоставимые результаты\n\nСреднее ΔE00 по человеку, затем среднее ошибок трёх базисов; меньше лучше. Это не ансамбль и не увеличение числа независимых участников. Исходные роли используют966 записей24 людей и многократно пересекались между прошлыми сериями. При переносе между камерами одновременно меняются люди и распределение.\n\n"
    text += "| Семейство / политика | Mixed | SLR → iPod | iPod → SLR |\n|---|---:|---:|---:|\n"
    for f, p in groups:
        text += (
            f"| {f} / {p} | "
            + " | ".join(f"{one(role, f, p)['clean_person_mean']:.6f}" for role in ROLES)
            + " |\n"
        )
    text += "\n![Обычная ошибка и мягкий стресс](affine_quality.png)\n\n"
    text += "| Совместная перцепционная модель | Mixed clean | Mixed стресс4/255 | SLR→iPod clean | SLR→iPod стресс4/255 |\n|---|---:|---:|---:|---:|\n"
    for p in ("clean", "guarded"):
        aa, bb = (
            one("mixed", "perceptual_joint_soft", p),
            one("slr_to_ipod", "perceptual_joint_soft", p),
        )
        text += f"| {p} | {aa['clean_person_mean']:.6f} | {aa['stress4_person_mean']:.6f} | {bb['clean_person_mean']:.6f} | {bb['stress4_person_mean']:.6f} |\n"
    text += "\nИскажения немного улучшают смешанный стресс-тест, но ухудшают прямой перенос SLR→iPod и по обычной, и по стрессовой ошибке. На обратном переносе стресс улучшается, а обычная ошибка меняется мало и разнонаправленно. Поэтому вариант guarded не повышается до универсальной основной модели. Постоянный Lab даёт нулевую изменчивость, но большую ошибку: одна устойчивость ответа недостаточна.\n\n"
    text += "Внутренняя смешанная ошибка совместной перцепционной модели без искажений 4.464672, у G soft 4.431187, у статической основы 4.448451. Поэтому улучшение внешнего результата 5.272642 не разрешает объявить архитектуру победителем по независимому выбору. Сравнение с G soft: −0.022476 ΔE00, лучше у 4/6 людей; описательный интервал ресэмплирования фиксированных прогнозов [−0.163324,+0.109194] включает 0. Это не новая подтверждающая проверка и не учёт всей истории выбора архитектур.\n\n"
    text += "Сохранены более старые ориентиры: полный KRR5.3984/8.5845/8.9125; guided RBF5.8013/10.0280/7.7193; динамический R5.3551/9.9089/9.8122; G hard5.2851/8.6548/8.3869. Они получены в прошлых сериях на тех же исторических ролях, здесь не переобучались. У G hard предыдущая GS-проверка выявила разрыв ответа на границе переключения; текущая плавная модель не использует такое жёсткое переключение. A не превосходит лучшие старые ориентиры на всех ролях.\n\n"
    text += "## Все дозы и вычислительная стоимость\n\n![Все фиксированные дозы](affine_all_doses.png)\n\n"
    text += f"Общий первичный процесс прочтения/обучения/выбора/сохранения/оценки занял {workflow['wall_seconds']:.3f}с, без импорта библиотек и независимого аудита.1,884 сохранённых конфигурации в12 банках — не1,884 независимых обучения: в них576 точных однокамерных копий,144 импортированных G-контроля и12 констант. Выполнено1,152 новых решений коэффициентов,192 разложения матриц,36 построений базиса,4 обучения плавного переключателя; вспомогательный код ещё вычисляет36 базовых решений и36 theta-решений.\n\n"
    text += (
        "| Mixed, seed17 | Полное обучение, мс | Ответ, мкс¹ | Веса, байт |\n|---|---:|---:|---:|\n"
    )
    for f in FAMILIES:
        for p in ("clean", "guarded"):
            r = one("mixed", f, p)
            text += f"| {f} / {p} | {r['seed17_full_fit_ms']:.3f} | {r['median_of_seed_median_us']:.1f} | {r['numeric_bytes']} |\n"
    text += "\n¹ Для ответа приведена медиана медиан трёх базисов,20 прогревов и3 прохода по каждому запросу. Обучение — медиана трёх полных повторов после одного прогрева, включая подготовку признаков из готовых color36, нормировки, расстояния, базис, варианты и коэффициенты. Декодирование фотографии, обнаружение лица, маска кожи, извлечение color36, импорт, ввод/вывод, поиск сетки и память библиотек исключены. Добавление вариантов делает обучение дороже; GPU здесь не использовался. Фактические все роли/семейства/политики и разброс времени сохранены в runtime.json.\n\n"
    text += f"## Независимая проверка\n\n23 численных теста проверены отдельной командой; окончательный результат фиксирует verification.json. Аудит воспроизводит2,402,100 внутренних ответов, все144 пары оценок кандидатов,24 выбора и15 групп контроля. Все111 итоговых моделей ×33 преобразования =1,462,758 ответов,444 набора метрик по дозе/человеку/камере.72 выбранных экземпляра и12 положительных проб переобучены независимым QR/SVD способом; расхождение максимум **{audit['maxima']['qr_prediction']:.3g} Lab**, допустимо0.001. Исходные нормировки проверены по исходным строкам, базисы сопоставлены с замороженным проверенным G.72 статических η=0/α=.1 полезных контроля точно совпали с G. Ни одна запись решения не нарушила проверку нормальных уравнений, отрицательных собственных значений Gram не было.\n\n"
    text += "Общий высокий уровень качества на обычных фотографиях лиц пока не доказан. Здесь подготовленные статистики участков кожи и цели native Lab исходного набора, не подбор оттенка продукта, распознавание личности или готовый SDK обработки лица. Семейство называется **Luma ChromaSeed**; A — обозначение этой серии, а не утверждение о новизне или правовой охране названия. Для резидентства и коммерческого обещания нельзя превращать эти числа в доказательство точности на телефонах.\n\n"
    text += "[Протокол](../../research/chromaseed_affine_v1_protocol.md) · [Все политики](all_policies.csv) · [Все дозы](all_doses.csv) · [Полная внутренняя сетка](all_inner_candidates.csv) · [Аудит](audit.json) · [Скорость](runtime.json) · [Проверка файлов](verification.json) · [Воспроизведение](reproduce.md) · [Карточка модели](../../architecture/chromaseed_affine_model_card.md) · [Следующее решение](../../research/chromaseed_affine_next_decision.md)\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    write_json(
        out / "summary.json",
        dict(
            source_lock_sha256=sha(run / "source_lock.json"),
            selection_sha256=sha(run / "selections.json"),
            results_sha256=sha(run / "results.json"),
            audit_sha256=sha(out / "audit.json"),
            runtime_sha256=sha(out / "runtime.json"),
            report_source_sha256=sha(Path(__file__)),
            rows=rows,
            doses=doses,
            inner_candidates=candidates,
            workflow_seconds=workflow["wall_seconds"],
            paired=audit["paired"],
            conclusion="Small exploratory joint-learning gain, not universal augmentation improvement. Clean selects eta0; guarded eta.75 worsens forward transfer. Ordinary-phone face quality unvalidated; full goal active.",
        ),
    )
    print("A REPORT:39 aggregate policies,156 dose cases,144 inner candidates", flush=True)


if __name__ == "__main__":
    main()
