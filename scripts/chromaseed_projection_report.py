"""X evidence: compact mixed-role gain with adverse camera-transfer results."""

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
CONTROLS = (
    "g_norm_soft",
    "g_perceptual_soft",
    "a_norm_joint_guarded",
    "a_perceptual_joint_guarded",
    "constant",
)


def csv_write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run, args.output
    result, selected, lock = (
        js(run / p) for p in ("results.json", "selections.json", "source_lock.json")
    )
    audit, runtime = js(out / "audit.json"), js(out / "runtime.json")
    assert audit["passed"] and audit["results_sha256"] == sha(run / "results.json")
    assert runtime["audit_sha256"] == sha(out / "audit.json")
    for p, h in {**lock["sources"], **lock["input_sha256"], **audit["dependencies"]}.items():
        assert sha(ROOT / p) == h, p
    parent = ROOT / "experiments/runs/chromaseed_affine_v1/results.json"
    parent_result = js(parent)
    groups = [(f, p) for f in FAMILIES for p in ("quality", "compact")] + [
        (f, "reference") for f in CONTROLS
    ]
    rows, doses, raw, inner = [], [], [], []
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
                    representation=rr[0].get("representation", "reference"),
                    alpha=rr[0].get("alpha"),
                    seeds=len(rr),
                    people=rr[0]["metrics"]["n_people"],
                    images=rr[0]["metrics"]["n_images"],
                    person_mean=float(np.mean([r["metrics"]["person_mean"] for r in rr])),
                    image_mean=float(np.mean([r["metrics"]["image_mean"] for r in rr])),
                    p90=float(np.mean([r["metrics"]["p90"] for r in rr])),
                    stress4=float(
                        np.mean([r["doses"][1]["worst_error"]["person_mean"] for r in rr])
                    ),
                    numeric_bytes=rr[0]["numeric_bytes"],
                    cached_array_bytes=tt[0]["numpy_only"]["cached_array_bytes"],
                    centers=rr[0]["actual_centers"],
                    latent_dimension=rr[0]["latent_dimension"],
                    active_gate=rr[0]["active_gate"],
                    response_us=float(np.median([t["numpy_only"]["median_us"] for t in tt])),
                    max_seed_p95_us=float(max(t["numpy_only"]["p95_us"] for t in tt)),
                    fit_ms=None if not ff else ff[0]["median_seconds"] * 1000,
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
            rr = [
                r
                for r in parent_result["records"]
                if (r["role"], r["family"], r["policy"]) == (role, family, "clean")
            ]
            assert len(rr) == 3
            raw.append(
                dict(
                    role=role,
                    family=family,
                    person_mean=float(np.mean([r["metrics"]["person_mean"] for r in rr])),
                    p90=float(np.mean([r["metrics"]["p90"] for r in rr])),
                    numeric_bytes=rr[0]["numeric_bytes"],
                    doses=[
                        float(np.mean([r["doses"][i]["worst_error"]["person_mean"] for r in rr]))
                        for i in range(4)
                    ],
                )
            )
            saved = selected["roles"][role][family]
            anchor = saved["raw_anchor"]
            for c in saved["candidates"]:
                inner.append(
                    dict(
                        role=role,
                        family=family,
                        representation=c["representation"],
                        alpha=c["alpha"],
                        clean=c["clean"],
                        p90=c["p90"],
                        numeric_bytes=c["numeric_bytes"],
                        compact_feasible=c["clean"] <= anchor["clean"] + 0.05
                        and c["p90"] <= anchor["p90"] + 0.10,
                        **{
                            f"selected_{p}": (c["representation"], c["alpha"])
                            == (
                                saved["policies"][p]["representation"],
                                saved["policies"][p]["alpha"],
                            )
                            for p in ("quality", "compact")
                        },
                    )
                )
    assert len(rows) == 39 and len(doses) == 156 and len(inner) == 468 and len(raw) == 12
    csv_write(out / "all_policies.csv", rows)
    csv_write(out / "all_doses.csv", doses)
    csv_write(out / "all_inner_candidates.csv", inner)

    def one(role, family, policy):
        return next(
            r for r in rows if (r["role"], r["family"], r["policy"]) == (role, family, policy)
        )

    def prior(role, family):
        return next(r for r in raw if (r["role"], r["family"]) == (role, family))

    selected_x = one("mixed", "perceptual_joint_soft", "compact")
    prior_x = prior("mixed", "perceptual_joint_soft")
    pairs = audit["paired"]
    adverse = sum(r["mean_difference"] > 0 for r in pairs)
    rotation = []
    for role in ROLES:
        for family in FAMILIES:
            c = selected["roles"][role][family]["policies"]["quality"]
            if c["representation"] == "d36_t1":
                baseline = next(
                    r
                    for r in selected["roles"][role][family]["candidates"]
                    if r["representation"] == "raw" and r["alpha"] == c["alpha"]
                )
                rotation.append(
                    dict(
                        role=role,
                        family=family,
                        inner_difference=c["clean"] - baseline["clean"],
                        extra_numeric_bytes=c["numeric_bytes"] - baseline["numeric_bytes"],
                    )
                )
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, role, label in zip(axes, ROLES, LABELS, strict=True):
        old = prior(role, "perceptual_joint_soft")
        qq, cc = (one(role, "perceptual_joint_soft", p) for p in ("quality", "compact"))
        cases = [
            (old, "Raw A", "#607a92"),
            (qq, "X quality", "#159183"),
            (cc, "X compact", "#d1842d"),
        ]
        for r, name, color in cases:
            ax.scatter(r["numeric_bytes"] / 1000, r["person_mean"], s=65, color=color, zorder=3)
            offset = (-3, 13) if name != "X compact" else (-3, -25)
            ax.annotate(
                f"{name}\n{r['person_mean']:.3f}",
                (r["numeric_bytes"] / 1000, r["person_mean"]),
                xytext=offset,
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        vals = [r[0]["person_mean"] for r in cases]
        gap = max(max(vals) - min(vals), 0.04)
        ax.set_ylim(min(vals) - 0.65 * gap, max(vals) + 0.65 * gap)
        ax.set_xlim(4, 31)
        ax.grid(alpha=0.2)
        ax.set_title(label)
        ax.set_xlabel("Numeric KB (1000 bytes)")
    axes[0].set_ylabel("Person mean ΔE00 ↓")
    fig.suptitle(
        "Luma ChromaSeed-X · smaller mixed model, adverse camera transfer\nPerceptual joint family · reused TRAIN · independent vertical scales",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    fig.text(
        0.5,
        0.018,
        "Each role has its own inner-selected settings. Repeated seeds do not add independent people; no phone-face validation.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(out / "projection_tradeoff.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    for ax, role, label in zip(axes, ROLES, LABELS, strict=True):
        ax.plot(
            [1, 4, 16, 64],
            prior(role, "perceptual_joint_soft")["doses"],
            marker="o",
            label="Raw A",
            color="#607a92",
        )
        for policy, color in (("quality", "#159183"), ("compact", "#d1842d")):
            dd = [
                r
                for r in doses
                if (r["role"], r["family"], r["policy"]) == (role, "perceptual_joint_soft", policy)
            ]
            ax.plot(
                [1, 4, 16, 64],
                [r["worst_error"] for r in dd],
                marker="o",
                label="X " + policy,
                color=color,
            )
        ax.set_xscale("log", base=4)
        ax.set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
        ax.set_xlabel("Mixture amount × 255")
        ax.set_title(label)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Worst-of-eight person ΔE00 ↓")
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "All fixed synthetic doses · unchanged physical targets are an assumption", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(out / "projection_stress.png", dpi=170)
    plt.close(fig)
    text = "# Luma ChromaSeed-X: сжатие цветовых признаков\n\n"
    text += f"На смешанном разбиении совместная перцепционная модель с 16 признаками уменьшилась с **{prior_x['numeric_bytes']:,} до {selected_x['numeric_bytes']:,} байт** ({100 * (1 - selected_x['numeric_bytes'] / prior_x['numeric_bytes']):.2f}% меньше). Обычная ошибка снизилась с **{prior_x['person_mean']:.6f} до {selected_x['person_mean']:.6f} ΔE00**. Это исследовательский выигрыш на шести ранее использованных людях; описательный интервал парного сравнения включает ноль.\n\n"
    text += f"При этом перенос между камерами ухудшился. Из 24 сравнений семейства/политики/роли с прежним A clean хуже стали **{adverse}**, поэтому X не заменяет прежние модели как универсальное решение. Меньший вариант для смешанных данных — полезный кандидат, не доказательство качества обычных селфи.\n\n"
    text += "## Что обучалось и как выбиралось\n\nВектор из 36 статистик участка кожи переводится в 8, 16 или 36 координат. Ковариация и проекция оценены только по исходным обучающим людям, с выравниванием веса людей/участков/изображений. Проверены четыре степени сглаживания ковариации: τ=0/.1/.5/1, плюс точный исходный вариант. Данные, физические Lab-цели и нормировки исходных строк сохранены. В X не добавлялись искусственные цвета при обучении.\n\n"
    text += "Каждое представление заново определяет ширину ядра и до 128 опорных примеров. Два вида потерь × статическое/совместное плавное решение × три α и три базиса. Совместная модель учит оба блока коэффициентов сразу; плавный сигнал по-прежнему использует исходные 36 признаков. При обучении на одной камере совместный вариант точно совпадает со статическим. Это не распознавание неизвестной камеры на запросе.\n\n"
    text += "Политика quality минимизирует внутреннюю ошибку по человеку. Compact выбирает меньший численный объём при внутренней ошибке не более +0.05 и p90 не более +0.10 относительно того же лучшего исходного кандидата. Размер при выборе — максимум среди внутренних моделей. Ограничения относятся к внутренним разбиениям, не гарантируют качество на внешних людях. Все 24 решения заморожены до финального обучения; на каждой роли получены собственные настройки. Все выбрали α=.1.\n\n"
    text += "## Полные сопоставления\n\nСреднее ошибок трёх базисов, затем одинаковый вес людей; это не ансамбль и не три независимые группы участников. Всего исходных TRAIN 966 строк/24 человека. Роли многократно использовались и пересекаются; смена камеры сопровождается сменой людей/распределения.\n\n"
    text += "| Семейство / политика | Mixed | SLR→iPod | iPod→SLR |\n|---|---:|---:|---:|\n"
    for family in FAMILIES:
        text += (
            f"| {family} / raw A clean | "
            + " | ".join(f"{prior(role, family)['person_mean']:.6f}" for role in ROLES)
            + " |\n"
        )
        for policy in ("quality", "compact"):
            text += (
                f"| {family} / {policy} | "
                + " | ".join(f"{one(role, family, policy)['person_mean']:.6f}" for role in ROLES)
                + " |\n"
            )
    for family in CONTROLS:
        text += (
            f"| {family} / reference | "
            + " | ".join(f"{one(role, family, 'reference')['person_mean']:.6f}" for role in ROLES)
            + " |\n"
        )
    text += "\n![Размер и ошибка](projection_tradeoff.png)\n\n"
    text += "| Совместная перцепционная модель | Проекция | Веса, байт | Обычная ΔE00 | Стресс4/255 |\n|---|---|---:|---:|---:|\n"
    for role in ROLES:
        for policy in ("quality", "compact"):
            r = one(role, "perceptual_joint_soft", policy)
            text += f"| {role} / {policy} | {r['representation']} | {r['numeric_bytes']} | {r['person_mean']:.6f} | {r['stress4']:.6f} |\n"
    text += "\nВосемь признаков дали особенно плохой SLR→iPod результат, хотя внутренние допуски были соблюдены. Сжатие потеряло полезную для переноса информацию или изменило геометрию неблагоприятно; эти результаты не позволяют отделить причины от различий людей/съёмки. На обратном переносе обе политики тоже уступили прежней основе. Ограничение внутренней ошибки нельзя представлять как внешнюю гарантию.\n\n"
    best_pair = next(
        r
        for r in pairs
        if (r["role"], r["family"], r["policy"]) == ("mixed", "perceptual_joint_soft", "compact")
    )
    ci = best_pair["fixed_prediction_person_bootstrap_95"]
    text += f"Смешанный compact улучшил {best_pair['improved_people']} из {best_pair['people']} людей. Разность с A clean {best_pair['mean_difference']:.6f}, описательный 95%-диапазон ресэмплирования фиксированных прогнозов [{ci[0]:.6f}, {ci[1]:.6f}]. Он не учитывает всю историю выбора моделей и не является новой подтверждающей проверкой.\n\n"
    for r in rotation:
        text += f"У {r['role']}/{r['family']} quality выбрал полное вращение d36_t1: внутренняя разность с raw того же α всего {r['inner_difference']:.3g}, при этом добавлено {r['extra_numeric_bytes']} байт. Это численно почти одинаковая геометрия; выбор обусловлен строгим сравнением оценок, полезный выигрыш от вращения не доказан. Максимальное отличие матриц ядра полного вращения во всех обучающих банках {audit['maxima']['full_rotation_fit_kernel']:.3g}. Выбор после просмотра внешних оценок не менялся.\n\n"
    text += "Старые ориентиры также сохранены: полный KRR 5.3984/8.5845/8.9125; guided RBF 5.8013/10.0280/7.7193; динамический R 5.3551/9.9089/9.8122; G hard 5.2851/8.6548/8.3869. Здесь они не переобучались. У G hard ранее обнаружен разрыв на границе переключения. X не превосходит эти ориентиры на всех ролях.\n\n"
    text += "## Реальная стоимость\n\n| Mixed | Проекция | Полное обучение, мс¹ | Ответ, мкс² | Массивы исполнителя, байт |\n|---|---|---:|---:|---:|\n"
    for family in FAMILIES:
        for policy in ("quality", "compact"):
            r = one("mixed", family, policy)
            text += f"| {family}/{policy} | {r['representation']} | {r['fit_ms']:.3f} | {r['response_us']:.1f} | {r['cached_array_bytes']} |\n"
    for family in ("g_perceptual_soft", "a_perceptual_joint_guarded"):
        r = one("mixed", family, "reference")
        text += (
            f"| {family} | raw control | — | {r['response_us']:.1f} | {r['cached_array_bytes']} |\n"
        )
    text += f"\nАктивный compact: {selected_x['fit_ms']:.3f} мс полного обучения на 734 исходных строках, {selected_x['response_us']:.1f} мкс ответа, {selected_x['cached_array_bytes']:,} байт сохранённых массивов исполнителя. Проекция добавляет операцию; ответ оказался медленнее исходных контрольных моделей в том же исполнителе. Уменьшение веса не следует выдавать за ускорение. Прежнее A clean обучение 34.600 мс измерено в другом запуске, поэтому отношение этих времён не является новым парным ускорением.\n\n"
    text += "¹ Seed17, медиана трёх полных повторов после одного прогрева. ² Медиана медиан трёх базисов, 20 прогревов/3 прохода по запросам. Ryzen 9 7900X, один CPU-поток. Обучение включает нормировки, ковариацию, проекцию, точную ширину, опоры, переключатель и коэффициенты. Исключены импорт/ввод-вывод/поиск сетки/подготовка фотографии. Ответ начинается с готового color36; это не телефонный или полный лицевой замер. Размер весов не равен памяти процесса; отдельно нужны библиотеки, временные массивы и исходный словарь загрузчика.\n\n"
    text += "## Проверка и все дозы\n\n![Все фиксированные дозы](projection_stress.png)\n\n"
    workflow = js(run / "workflow.json")
    text += f"Первичный процесс занял {workflow['wall_seconds']:.3f} с без импорта/аудита/замера скорости. 5,772 сохранённых readout-конфигурации — это 3,744 новых решения коэффициентов, 1,872 точные однокамерные копии, 144 импортированных контроля и 12 констант. 624 разложения Gram, 468 базисов, 156 расчётов ширины, 12 разложений ковариации и 4 переключателя; исходные вспомогательные функции дополнительно вычисляют 36+36 решений. Все 432 raw-конфигурации точно воспроизводят A eta0.\n\n"
    text += f"Независимый аудит проверил 144 проекции через SVD и их оператор расстояния, все 156 точных ширин, 468 плотных траекторий выбора опор, 817,700 внутренних прогнозов, 468 пар оценок/24 выбора и 1,462,758 итоговых ответов/444 набора метрик по дозе. Заново обучены 72 выбранных экземпляра и 12 положительных проб: SVD-проекция, плотное ядро/опоры, независимый переключатель, аналитическая цветовая метрика, QR/SVD-решение. Максимальный разрыв с исходным обучением {audit['maxima']['qr_prediction']:.3g} Lab при допуске .001; с реальным исполнителем {audit['maxima']['standalone_prediction']:.3g}. Все 112 повторных полных обучений воспроизвели массивы точно. Основные 12 тестов и 3 теста независимого решения проходят; окончательная проверка связывает исходники и артефакты.\n\n"
    text += "Рабочее семейство — **Luma ChromaSeed**, X обозначает эту серию. Это цветовой компонент по подготовленным статистикам кожи с native Lab D65/10° целями набора. Точность на обычных лицах/телефонах, выбор оттенка косметики и готовность коммерческого SDK не подтверждены. Новые данные, загрузки, публикация и старые validation/calibration/test не использовались.\n\n"
    text += "[Протокол](../../research/chromaseed_projection_v1_protocol.md) · [Все политики](all_policies.csv) · [Все дозы](all_doses.csv) · [Полная сетка](all_inner_candidates.csv) · [Аудит](audit.json) · [Стоимость](runtime.json) · [Проверка файлов](verification.json) · [Воспроизведение](reproduce.md) · [Карточка/веса](../../architecture/chromaseed_projection_model_card.md) · [Следующее решение](../../research/chromaseed_projection_next_decision.md).\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    write_json(
        out / "summary.json",
        dict(
            source_lock_sha256=sha(run / "source_lock.json"),
            selection_sha256=sha(run / "selections.json"),
            results_sha256=sha(run / "results.json"),
            parent_A_results_sha256=sha(parent),
            audit_sha256=sha(out / "audit.json"),
            runtime_sha256=sha(out / "runtime.json"),
            report_source_sha256=sha(Path(__file__)),
            rows=rows,
            doses=doses,
            inner_candidates=inner,
            raw_A_controls=raw,
            paired=pairs,
            adverse_outer_comparisons=adverse,
            rotation_selected_cases=rotation,
            workflow_seconds=workflow["wall_seconds"],
            conclusion="Mixed compact joint candidate saves35.46% weights with small exploratory clean gain; camera transfer worsens and inference is slower. No universal promotion or phone-face validation; full goal active.",
        ),
    )
    print(
        f"X REPORT:39 policies,468 inner candidates; {adverse}/24 adverse matched outer comparisons",
        flush=True,
    )


if __name__ == "__main__":
    main()
