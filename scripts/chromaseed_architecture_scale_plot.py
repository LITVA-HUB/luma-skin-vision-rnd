"""Standalone scientific figure from a provisional, source-bound inner review."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from chromaseed_architecture_scale_run import OUT, ROOT, RUN, check_map
from chromaseed_kernel_audit import js
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, ScalarFormatter
from skin_local_search_train import sha, write_json

STYLE = {
    "patch_small": ("Участки · 17,4 тыс.", "#246B9A", "o"),
    "patch5m": ("Участки · 4,96 млн", "#D77825", "s"),
    "soft_small": ("4 уточнения · 15,2 тыс.", "#78803C", "^"),
    "soft5m": ("4 уточнения · 4,85 млн", "#A34E78", "D"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("review", type=Path)
    args = parser.parse_args()
    data = js(args.review)
    assert data["source_lock_sha256"] == sha(RUN / "source_lock.json")
    check_map(data["completed_inner_inputs"])
    assert data["role"] == "mixed" and data["rows"] == 734 and data["people"] == 18
    assert set(data["variants"]) <= STYLE.keys()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelcolor": "#272C31",
            "text.color": "#272C31",
            "xtick.color": "#4D555D",
            "ytick.color": "#4D555D",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 8), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.975, top=0.685, bottom=0.26, wspace=0.13)
    fig.text(0.06, 0.956, "LUMA / ChromaSeed AS", fontsize=12, color="#687078")
    fig.text(0.97, 0.966, "✿", fontsize=22, ha="right", va="top", color="#687078")
    fig.text(0.06, 0.90, "Размер модели и продолжительность обучения", fontsize=21, weight="bold")
    fig.text(
        0.06,
        0.85,
        "Внутренняя ошибка ΔE00 · меньше лучше · 3 проверенные точки обучения",
        fontsize=12,
    )
    handles = []
    bounds = []
    for variant in data["variants"]:
        label, color, marker = STYLE[variant]
        handles.append(Line2D([], [], color=color, marker=marker, lw=2, label=label))
    handles.extend(
        [
            Line2D([], [], color="#414A53", ls="--", lw=1.3, label="Исходная NP"),
            Line2D([], [], color="#8D949B", ls=":", lw=1.5, label="Прежний выбор WE"),
        ]
    )
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.06, 0.805),
        ncol=3,
        frameon=False,
        fontsize=10.5,
        handlelength=2.4,
        columnspacing=2.0,
    )
    for axis, rate in zip(axes, (1e-5, 1e-4), strict=True):
        for variant in data["variants"]:
            _, color, marker = STYLE[variant]
            records = sorted(
                [r for r in data["records"] if r["variant"] == variant and r["rate"] == rate],
                key=lambda r: r["step"],
            )
            assert len(records) == 3 and [r["step"] for r in records] == [128, 512, 2048]
            x = np.array([r["step"] for r in records])
            seeds = np.array([r["individual_seed_delta_e00"] for r in records])
            assert seeds.shape == (3, 3) and np.isfinite(seeds).all()
            mean = seeds.mean(1)
            np.testing.assert_allclose(
                mean, [r["person_delta_e00"] for r in records], atol=1e-12, rtol=0
            )
            axis.fill_between(x, seeds.min(1), seeds.max(1), color=color, alpha=0.08, linewidth=0)
            axis.plot(x, mean, color=color, marker=marker, lw=2, ms=7)
            bounds.extend(seeds.ravel().tolist())
        axis.axhline(data["prior_np_inner_delta_e00"], color="#414A53", ls="--", lw=1.3, zorder=0)
        axis.axhline(data["prior_we_policy"]["clean"], color="#8D949B", ls=":", lw=1.5, zorder=0)
        axis.set_xscale("log", base=2)
        axis.set_xlim(96, 2730)
        axis.xaxis.set_major_locator(FixedLocator([128, 512, 2048]))
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.minorticks_off()
        axis.grid(axis="y", color="#E8EBEE", lw=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color("#BBC1C7")
        axis.set_xlabel("Шаги обучения", labelpad=10)
        axis.set_title(
            "Скорость 0,00001" if rate == 1e-5 else "Скорость 0,0001", fontsize=13, pad=14
        )
    low = min(bounds + [data["prior_we_policy"]["clean"]])
    high = max(bounds)
    axes[0].set_ylim(np.floor((low - 0.035) * 20) / 20, np.ceil((high + 0.035) * 20) / 20)
    axes[0].set_ylabel("Ошибка ΔE00", labelpad=10)
    fig.text(
        0.06,
        0.152,
        "734 наблюдения · 18 людей · три внутренних разбиения без пересечения людей",
        fontsize=11,
    )
    fig.text(
        0.06,
        0.111,
        "Линии — среднее трёх запусков. Заливка — их минимум–максимум, не доверительный интервал.",
        fontsize=10.5,
        color="#555E66",
    )
    fig.text(
        0.06,
        0.073,
        "Предварительный снимок: внешняя оценка AS ещё не проведена. NP и WE — прежние контроли;",
        fontsize=10.5,
        color="#555E66",
    )
    fig.text(
        0.06,
        0.041,
        "у них другой объём подбора настроек. Остальные архитектуры серии продолжают обучение.",
        fontsize=10.5,
        color="#555E66",
    )
    base = OUT / "figures" / args.review.stem
    base.parent.mkdir(parents=True, exist_ok=True)
    png, svg = base.with_suffix(".png"), base.with_suffix(".svg")
    fig.savefig(png, dpi=160, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    write_json(
        base.with_suffix(".figure.json"),
        dict(
            provisional=True,
            review_path=str(args.review.resolve()),
            review_sha256=sha(args.review),
            generator_sha256=sha(ROOT / "scripts/chromaseed_architecture_scale_plot.py"),
            png_sha256=sha(png),
            svg_sha256=sha(svg),
            variants=data["variants"],
            points=len(data["records"]),
            population="734 observations /18 people, original TRAIN inner folds",
            band="min/max across3seeds, not a confidence interval",
        ),
    )
    print("AS SCIENTIFIC FIGURE", png)


if __name__ == "__main__":
    main()
