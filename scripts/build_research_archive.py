"""Build the public research paper's static figures and complete evidence index.

Reads archived reports/aggregate JSON only. Does not decode datasets, train models,
rescore the exposed test, or change any frozen evidence. Python >=3.11.
"""

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "publication"
FIG = OUT / "figures"
NAVY, TEAL, CORAL, GOLD = "#162c46", "#087e8b", "#c74a48", "#b77e19"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": NAVY,
        "text.color": NAVY,
        "axes.titleweight": "bold",
        "savefig.facecolor": "#fcfcfa",
        "figure.facecolor": "#fcfcfa",
        "axes.facecolor": "#fcfcfa",
        "svg.fonttype": "none",
    }
)

NOTES = {
    "cc_v2": "Якорный остаточный метод: выигрыш у matched C+ на внешних камерах, регрессия на части доменов.",
    "cc_v3": "Полный цветовой базис и граф: проигрыш обычному RGB при одинаковой ёмкости.",
    "cc_v4": "Оценка исправленного цвета и повторное уточнение: механизм transport не выиграл source screen.",
    "cc_v5": "Парное обучение критика и равное число запросов: устойчивого преимущества повторных проходов нет.",
    "cc_v6": "Скрещивание канонического базиса и физического критика: улучшения не сложились.",
    "cc_v7": "Семантический teacher и виртуальный сенсор: стандартные контроли остаются конкурентными.",
    "cc_v7_external": "29 заранее зафиксированных методов; новые изображения внешних камер, с оговоркой о сценах и истории камер.",
    "fourier_ridge_v1": "Выпуклая регрессия в пространстве цветовых гистограмм проверяет необходимость пространственной CNN.",
    "fourier_ridge_v2": "Отмена глобального prior и проверка границы регуляризации; отдельный source screen.",
    "phone_v1": "Первый фиксированный Samsung/Oppo тест; строгий HDF5 loader ограничивал достижимое покрытие.",
    "phone_v1_alias": "Документированное исправление HDF5 имён: 79/88 пригодных эталонов. Это повторный, уже раскрытый тест.",
    "projector_probe": "Проверка геометрии допустимых поправок; привилегированный диагностический контроль, не skin accuracy.",
    "public_runs": "Первый реальный SimpleCube++ benchmark: классические и компактные нейросетевые контроли.",
    "source_snapshots": "Архив исходников при исправлении формата. Это служебная доказательная папка, не поколение модели.",
    "skin_appearance_inverse_v1": "Обратная задача: предсказывать распределение наблюдения по цвету и инвертировать его; универсального выигрыша нет.",
    "skin_branch_combination_v1": "Равновесные пары уже обученных моделей: source gains при удвоенной стоимости, не новый независимый результат.",
    "skin_capture_v1": "Скрытые условия съёмки и смесь четырёх гипотез. Сильный компактный source baseline: 929 297 параметров.",
    "skin_capture_support_v1": "Перемешивание реально наблюдавшихся патчей одного участка кожи; одинаковый Lab не означает регистрацию пикселей.",
    "skin_color_sampling_v1": "Распределение бюджета по измеренному цвету и людям; небольшие внутренние улучшения, слабое отличие от обычной балансировки.",
    "skin_color_sampling_mass_v1": "Шесть дополнительных контролей массы людей к sampling screen; включены в общий отчёт, не считать дважды.",
    "skin_copula_v1": "Ранги, copula и абсолютные гистограммы. Удаление абсолютного цвета ухудшает точность.",
    "skin_correction_transfer_v1": "Финальная проверка: все три correction head ухудшили оба unseen-camera направления. Исследования остановлены.",
    "skin_correspondence_v1": "Повторяемость реальных приборных измерений и многовидовые диагностики; это не известный предел ошибки модели.",
    "skin_crossfit_correction_v1": "Внутренний выигрыш 7,40% при +188 035 параметрах. Гипотеза превосходства OOF не подтвердилась.",
    "skin_distribution_v1": "Условные распределения цвета и решения по ожидаемой ошибке; универсального выигрыша над прямой регрессией нет.",
    "skin_expert_anchor_v1": "Удаление и явная супервизия экспертов; отдельный forward gain, общей победы нет.",
    "skin_gradient_transfer_v1": "Градиенты людей, перемешанные группы и малые шаги: уменьшение TRAIN loss не гарантирует уменьшения ΔE00.",
    "skin_graph_support_v1": "Комбинация графа и реальных paired patches помогает слабому transfer baseline, но не сильнейшим рецептам.",
    "skin_he_xyz_v1": "Реальные парные региональные RGB/XYZ, 20 тестовых людей. XYZ RMSE, без выдуманного ΔE00 при неизвестном белом.",
    "skin_issa_v1": "Измеренные спектры кожи: oracle compression с полным спектром на входе, не точность фотографии.",
    "skin_local_reference_v1": "Локальная аффинная регрессия на TRAIN leave-person-out улучшает ridge. Известный статистический механизм.",
    "skin_local_reference_transfer_v1": "Перенос локального эталонного банка на камеры не побеждает сильные нейронные контроли.",
    "skin_local_teacher_v1": "Локальное соответствие teacher/RGB: aligned, global и shuffled контроли, отрицательный перенос.",
    "skin_loss_field_v1": "Предсказывать поле цветовой ошибки вместо единственного ответа; 15 625 кандидатов не дали общей победы.",
    "skin_material_image_v1": "Физический спектральный декодер на реальных изображениях: диапазон представимых цветов не объясняет основной провал.",
    "skin_mskcc_pixel_ablation_v2": "Локальные/глобальные голоса и усиленная робастная итерация. Изъятие контекста не помогло.",
    "skin_mskcc_pixels_v1": "Переход от готовых региональных статистик к JPEG-to-Lab; CNN, patch votes, fusion и robust recurrence.",
    "skin_mskcc_selective_v1": "Основной независимый тест: 400 изображений / 10 новых людей. 4,457 ΔE00; обычная fusion сильнее.",
    "skin_mskcc_summary_v1": "Первый MSKCC контроль по трём опубликованным медианным image-Lab признакам, только source validation.",
    "skin_neural_reference_v1": "Более ёмкий neural reference head в лимите +200k; снижение TRAIN ошибки не перенеслось универсально.",
    "skin_nuisance_v1": "Learned graph, grid, global graph и constant bias; инвариантность не должна уничтожать полезный абсолютный сигнал.",
    "skin_offset_diagnostic_v1": "Привилегированные offsets по эталонам других evaluation людей; не результат без калибровки.",
    "skin_pair_v1": "Paired invariance, quotient, output consistency и VICReg; пара снимков не устранила зависимость от условий съёмки.",
    "skin_patch_likelihood_v1": "Распределение патчей против среднего изображения; сложность не принесла универсальной точности.",
    "skin_relational_probe_v1": "Сравнение участков и ложных пар; 1 421 пара не создаёт новых независимых людей или cross-camera пар.",
    "skin_risk_cross_v1": "Перекрёстные сочетания цвета и риска: 90 endpoints из существующих моделей, не 90 новых обучений.",
    "skin_sampling_transfer_v1": "Скрещивание цветовой и person/site балансировки: небольшие known gains не переносятся в обе стороны.",
    "skin_shared_bias_v1": "Общий bias нескольких снимков против их согласованности; согласие не гарантирует правильный цвет.",
    "skin_spatial_v1": "Свёрточные и recurrent graph ветви с геометрическими контролями. Дополнительные проходы не дали общего выигрыша.",
    "skin_spectral_probe_v1": "22 500 спектров из трёх TRAIN лиц. Представимость материала проверена отдельно от восстановления по RGB.",
    "skin_support_curve_v1": "6/12/18 обучающих людей при одинаковом числе обновлений: больше людей помогло, pixel adapter не убедил.",
    "skin_teacher_readout_v1": "Frozen foundation features, ridge и shuffled controls: перенос семантики не стал универсальной колориметрией.",
    "skin_train_branch_v1": "Граф/свёртка только при обучении; направленные gains без универсальной победы и независимого подтверждения.",
}


def read(path):
    return (ROOT / path).read_text(encoding="utf-8-sig")


def load(path):
    return json.loads(read(path))


def write(path, value):
    p = OUT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value, encoding="utf-8", newline="\n")


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=155, bbox_inches="tight")
    plt.close(fig)


def tables(text):
    blocks = re.findall(r"(?:^\|[^\n]*\n){2,}", text + "\n", re.M)
    result = []
    for b in blocks:
        rows = [
            [s.strip() for s in line.strip().strip("|").split("|")]
            for line in b.strip().splitlines()
        ]
        if len(rows) >= 3 and all(re.fullmatch(r"[:\- ]+", s) for s in rows[1]):
            result.append({"headers": rows[0], "rows": rows[2:]})
    return result


def overview_figures():
    testpath = "docs/benchmarks/skin_mskcc_selective_v1/test_results.json"
    test = load(testpath)
    methods = [
        ("C+ · same color core", test["risk_models"]["ensemble__C_plus"], NAVY),
        ("Proposed · disagreement", test["risk_models"]["ensemble__Proposed"], TEAL),
        ("Ordinary fusion · 6 models", test["color_comparators"]["fusion_ensemble"], CORAL),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.7))
    for i, (name, r, color) in enumerate(methods):
        vals = [r["full"]["mean"], next(x["mean"] for x in r["coverage"] if x["coverage"] == 0.8)]
        x = np.arange(2) + (i - 1) * 0.24
        ax[0].bar(x, vals, 0.22, label=name, color=color)
        for xx, yy in zip(x, vals):
            ax[0].text(xx, yy + 0.055, f"{yy:.3f}", ha="center", fontsize=10)
        cov = sorted(r["coverage"], key=lambda x: x["coverage"])
        ax[1].plot(
            [v["coverage"] * 100 for v in cov],
            [v["mean"] for v in cov],
            "o-",
            color=color,
            label=name,
        )
    ax[0].set(
        xticks=[0, 1],
        xticklabels=["All images", "80% accepted"],
        ylabel="Mean skin error · ΔE00 ↓",
        ylim=(0, 5.2),
    )
    ax[1].set(
        xlabel="Accepted coverage · %",
        ylabel="Mean skin error · ΔE00 ↓",
        title="Selective comparison",
        ylim=(3.9, 4.7),
    )
    ax[0].set_title("Independent, known-camera test")
    ax[1].legend(fontsize=9, loc="upper left")
    fig.suptitle(
        "Real instrument skin references · 400 images / 10 new people", fontsize=16, y=1.04
    )
    fig.text(
        0.02,
        -0.035,
        "Proposed − C+ at 80%: −0.174 ΔE00; patient-cluster 95% CI [−0.568, +0.165]. No convincing mechanism win.",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, "independent_skin")

    transfer = load("docs/benchmarks/skin_correction_transfer_v1/summary.json")["groups"]
    internal = load("docs/benchmarks/skin_crossfit_correction_v1/summary.json")["rows"]
    panels = [
        ("Internal screen · 6 reused people", internal),
        (
            "SLR → unseen iPod · source",
            [x for x in transfer if x["protocol"] == "from_SLR" and x["domain"] == "unseen"],
        ),
        (
            "iPod → unseen SLR · source",
            [x for x in transfer if x["protocol"] == "from_ipod" and x["domain"] == "unseen"],
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (title, rs) in zip(axes, panels):
        for i, r in enumerate(rs):
            ax.bar(
                i,
                r["mean"],
                color=NAVY
                if r["arm"] == "base"
                else TEAL
                if r["arm"] == "in_matched"
                else "#a4b5be",
            )
            ax.scatter([i] * 3, r["seed_means"], color="white", edgecolor=NAVY, s=20, zorder=3)
            ax.text(i, max(r["seed_means"]) + 0.22, f"{r['mean']:.3f}", ha="center", fontsize=10)
        ax.set(
            title=title,
            xticks=range(len(rs)),
            xticklabels=[r["arm"].replace("_", "\n") for r in rs],
            ylim=(0, 9),
        )
    axes[0].set_ylabel("Mean skin error · ΔE00 ↓")
    fig.suptitle("A local gain failed the camera-transfer follow-up", fontsize=17, y=1.04)
    fig.text(
        0.015,
        -0.04,
        "Different evaluation cohorts: compare heads WITHIN each panel. Bars = mean of 3 individual seeds; dots = seed means, not confidence intervals.",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, "correction_falsifier")

    labels = [
        "Synthetic\nengineering",
        "Real color\nconstancy",
        "Phone RAW-like\ncomponent test",
        "Instrument skin\nindependent test",
        "Skin mechanism\nsearch",
        "Final transfer\nfalsifier",
    ]
    fig, ax = plt.subplots(figsize=(14, 2.9))
    ax.axis("off")
    for i, label in enumerate(labels):
        ax.text(
            i,
            0.58,
            f"{i + 1:02}",
            ha="center",
            fontsize=24,
            color=TEAL if i < 4 else CORAL,
            weight="bold",
        )
        ax.text(i, 0.2, label, ha="center", fontsize=12)
        if i < 5:
            ax.annotate(
                "",
                xy=(i + 0.78, 0.61),
                xytext=(i + 0.23, 0.61),
                arrowprops={"arrowstyle": "->", "color": "#bac6cd"},
            )
    ax.set(xlim=(-0.5, 5.5), ylim=(-0.25, 1.05))
    ax.text(-0.5, 1.02, "LUMA / RESEARCH ATLAS", fontsize=22, weight="bold")
    ax.text(
        -0.5,
        -0.22,
        "Completed research archive · 2026-09-10—11 · Evidence, controls and negative results",
        fontsize=12,
    )
    save(fig, "research_journey")
    return {
        "independent": testpath,
        "correction": "docs/benchmarks/skin_correction_transfer_v1/summary.json",
        "internal": "docs/benchmarks/skin_crossfit_correction_v1/summary.json",
    }


def report_for(d):
    overrides = {
        "public_runs": "public_benchmark_report.md",
        "fourier_ridge_v1": "fourier_representation_report.md",
        "fourier_ridge_v2": "fourier_representation_report.md",
        "skin_color_sampling_mass_v1": "skin_color_sampling_v1/report.md",
    }
    if d.name in overrides:
        return ROOT / "docs/benchmarks" / overrides[d.name]
    for p in [
        d / "report.md",
        ROOT / "docs/benchmarks" / f"{d.name}_report.md",
        d / "seed17_report.md",
    ]:
        if p.exists():
            return p
    return None


def family_plot(card):
    """Plot literal rounded report cells, never pretend they are pooled raw data."""
    cc_columns = {
        "public_runs": 2,
        "cc_v2": 4,
        "cc_v3": 3,
        "cc_v4": 2,
        "cc_v5": 1,
        "cc_v6": 1,
        "cc_v7": 1,
        "cc_v7_external": 3,
        "phone_v1": 1,
        "phone_v1_alias": 1,
        "fourier_ridge_v1": 3,
        "fourier_ridge_v2": 3,
    }
    is_cc = card["id"] in cc_columns
    if (not card["id"].startswith("skin_") and not is_cc) or card["id"] in {
        "skin_issa_v1",
        "skin_he_xyz_v1",
        "skin_spectral_probe_v1",
        "skin_correspondence_v1",
        "skin_gradient_transfer_v1",
        "skin_offset_diagnostic_v1",
        "skin_relational_probe_v1",
        "skin_mskcc_selective_v1",
        "skin_color_sampling_mass_v1",
    }:
        return None
    for tab in card["tables"]:
        indices = (
            [cc_columns[card["id"]]]
            if is_cc
            else [
                i
                for i, h in enumerate(tab["headers"])
                if re.search(r"(?i)mean", h)
                and not re.search(r"(?i)(80|train|seed|patient|site|secondary)", h)
            ]
        )
        if not indices:
            continue
        idx = indices[0]
        vals = []
        labels = []
        for row in tab["rows"]:
            if len(row) != len(tab["headers"]):
                continue
            value = re.match(r"\*?\*?([0-9]+\.[0-9]+)", row[idx])
            if value:
                vals.append(float(value[1]))
                labels.append(
                    row[0]
                    if is_cc and not card["id"].startswith("fourier")
                    else " / ".join(row[:idx])
                )
        if not vals:
            continue
        fig, ax = plt.subplots(figsize=(11, max(3.5, 0.30 * len(vals) + 1.5)))
        y = np.arange(len(vals))
        ax.barh(y, vals, color=TEAL, height=0.65)
        for yy, v in zip(y, vals):
            ax.text(v + 0.035, yy, f"{v:.4f}", va="center", fontsize=9)
        ax.set(
            yticks=y,
            yticklabels=labels,
            xlabel=(tab["headers"][idx] + " · reproduction angular error / degrees ↓")
            if is_cc
            else "Mean skin error · ΔE00 ↓",
            xlim=(0, max(vals) * 1.16),
        )
        ax.invert_yaxis()
        ax.set_title(card["id"].replace("skin_", "").replace("_", " "), loc="left", pad=18)
        fig.text(
            0.015,
            0.002,
            (
                "Color constancy only, NOT skin accuracy. Scope/selection: linked report. First comparison table; rounded archived cells."
                if is_cc
                else "Exploratory source/internal results. Protocols & budgets: linked report. First mean-error table only; all tables retained in catalog."
            ),
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        name = "family_" + card["id"]
        save(fig, name)
        return "figures/" + name + ".png"
    return None


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    sources = overview_figures()
    cards = []
    # First appearance in Git establishes archive chronology, not experiment time.
    first = {}
    commits = subprocess.check_output(
        ["git", "log", "--reverse", "--format=COMMIT %H %s", "--name-only"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).splitlines()
    order = 0
    sha = ""
    subject = ""
    history = []
    for line in commits:
        if line.startswith("COMMIT "):
            _, sha, subject = line.split(" ", 2)
            order += 1
            history.append({"original_commit": sha, "subject": subject, "order": order})
        elif line.startswith("docs/benchmarks/"):
            parts = line.split("/")
            if len(parts) > 3:
                first.setdefault(parts[2], (order, sha))
    for d in sorted((ROOT / "docs/benchmarks").iterdir()):
        if not d.is_dir():
            continue
        report = report_for(d)
        t = report.read_text(encoding="utf-8-sig") if report else ""
        rel = report.relative_to(ROOT).as_posix() if report else None
        card = {
            "id": d.name,
            "introduced_order": first.get(d.name, (999, ""))[0],
            "introduced_original_commit": first.get(d.name, (999, ""))[1],
            "directory": d.relative_to(ROOT).as_posix(),
            "report": rel,
            "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest() if report else None,
            "title": next(
                (line.lstrip("# ") for line in t.splitlines() if line.startswith("# ")), d.name
            ),
            "observation_ru": NOTES.get(d.name, "См. исходные артефакты."),
            "tables": tables(t),
            "figures": [p.relative_to(ROOT).as_posix() for p in sorted(d.rglob("*.png"))],
            "json_files": sum(1 for _ in d.rglob("*.json")),
        }
        card["publication_figure"] = family_plot(card)
        cards.append(card)
    cards.sort(key=lambda c: (c["introduced_order"], c["id"]))
    write(
        "catalog.json",
        json.dumps(
            {
                "scope": "53 benchmark directories, not 53 independent architectures; no duplicate fit total",
                "families": cards,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    write("original_history.json", json.dumps(history, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Атлас всех поколений и экспериментов",
        "",
        f"В архиве **{len(cards)} каталога бенчмарков**. Это не число независимых архитектур или обучений: здесь также общие контроли, переоценки и служебные папки.",
        "",
        "Порядок — первое появление каталога в исходной Git-истории. Исторические слова «next», «unopened» и «active» описывают момент написания; исследования остановлены 11 сентября 2026 по запросу автора проекта.",
        "",
        "[Главная](../../README.md) · [Исследовательская работа](RESEARCH_REPORT_RU.md) · [Галерея](GALLERY.md) · [Все таблицы JSON](catalog.json) · [История](HISTORY.md)",
        "",
        "## Синтетический нулевой этап",
        "",
        "[Отчёт](../benchmarks/benchmark_results.md), [baseline](../benchmarks/baseline_report.md), [абляции](../benchmarks/ablation_report.md). Proposed проиграл классическому A2. 48 тестов / 34 smoke-команды относятся к тому этапу; синтетическая точность не подтверждает реальную.",
        "",
        "| № | Семейство / исходный отчёт | Наблюдение |",
        "|---:|---|---|",
    ]
    gallery = [
        "# Галерея исследовательских поколений",
        "",
        "Все изображения — графики измерений или схемы. Фотографий участников здесь нет. Графики не образуют единую шкалу прогресса: протоколы и единицы различаются.",
        "",
        "[Атлас](GENERATIONS.md) · [Основной отчёт](RESEARCH_REPORT_RU.md)",
        "",
        "![Путь исследования](figures/research_journey.png)",
        "",
        "![Независимый тест](figures/independent_skin.png)",
        "",
        "![Последняя проверка](figures/correction_falsifier.png)",
    ]
    all_rows = []
    for n, c in enumerate(cards, 1):
        target = "../../" + (c["report"] or c["directory"])
        lines.append(f"| {n:02} | [{c['id']}]({target}) | {c['observation_ru']} |")
        gallery += [
            "",
            f"## {n:02} · {c['id']}",
            "",
            c["observation_ru"],
            "",
            f"[Методика, все результаты и ограничения]({target})",
            "",
        ]
        figs = ([c["publication_figure"]] if c["publication_figure"] else []) + [
            "../../" + p for p in c["figures"]
        ]
        if figs:
            gallery += [f"![{c['id']} — {i + 1}]({p})\n" for i, p in enumerate(dict.fromkeys(figs))]
        else:
            gallery += [
                "Числовые таблицы и/или диагностические доказательства находятся в отчёте. Отсутствующая метрика не заменена придуманной диаграммой."
            ]
        for ti, tab in enumerate(c["tables"]):
            for ri, row in enumerate(tab["rows"]):
                all_rows.append(
                    {
                        "family": c["id"],
                        "report": c["report"],
                        "table": ti + 1,
                        "row": ri + 1,
                        "headers": json.dumps(tab["headers"], ensure_ascii=False),
                        "cells": json.dumps(row, ensure_ascii=False),
                    }
                )
    write("GENERATIONS.md", "\n".join(lines) + "\n")
    write("GALLERY.md", "\n".join(gallery) + "\n")
    with (OUT / "all_report_tables.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "report", "table", "row", "headers", "cells"])
        w.writeheader()
        w.writerows(all_rows)
    hist = [
        "# Полная исходная история исследования",
        "",
        "Хэши ниже относятся к неизменённому локальному исследовательскому репозиторию. При публикации отфильтрованы только явно перечисленные сторонние архивные файлы; соответствие публичным коммитам приведено в `commit_map.json`. Исторические source locks остаются исходными, не переписанными задним числом.",
        "",
        "| № | Исходный коммит | Что было зафиксировано |",
        "|---:|---|---|",
    ]
    hist += [f"| {x['order']} | `{x['original_commit'][:10]}` | {x['subject']} |" for x in history]
    write("HISTORY.md", "\n".join(hist) + "\n")
    manifest = {
        "kind": "report-only generation; no experiment rerun",
        "benchmark_directories": len(cards),
        "table_rows": len(all_rows),
        "sources": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources.values()
        },
        "report_sources": {c["report"]: c["report_sha256"] for c in cards if c["report"]},
        "figure_count_png": len(list(FIG.glob("*.png"))),
    }
    write("figure_provenance.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k not in {"sources", "report_sources"}},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
