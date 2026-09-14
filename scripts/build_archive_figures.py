"""Render source-backed archive figures and explicit AS layer cards; no model execution."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/archive/2026-09-14"
FIG = OUT / "figures"
SOURCES = {}


def load(path):
    p = ROOT / path
    SOURCES[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text(encoding="utf-8-sig"))


def save(fig, name):
    fig.savefig(
        FIG / f"{name}.png",
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "Luma archive; saved evidence only"},
    )
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    svg = FIG / f"{name}.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    plt.close(fig)


def specs(variant):
    if variant == "pool5m":
        return [("token1", 18, 512), ("mlp1", 1572, 3072), ("head", 3072, 3)]
    a, e, h = (32, 24, 48) if variant.endswith("small") else (384, 256, 1120)
    layers = [("token1", 18, a), ("token2", a, e)]
    if variant.startswith("patch"):
        h1, h2, h3 = (96, 64, 48) if variant.endswith("small") else (1792, 1536, 1024)
        return layers + [("mlp1", 36 + e, h1), ("mlp2", h1, h2), ("mlp3", h2, h3), ("head", h3, 3)]
    return layers + [
        ("context", 36 + e, h),
        ("query", h + 3, e),
        ("key", e, e),
        ("update1", 2 * h + e + 3, h),
        ("update2", h, h),
        ("head", h, 3),
    ]


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fcfcfa",
            "axes.facecolor": "#fcfcfa",
            "text.color": "#162c46",
            "axes.labelcolor": "#162c46",
            "svg.hashsalt": "luma-archive-2026-09-14",
        }
    )
    asummary = load("docs/benchmarks/chromaseed_architecture_scale_v1/summary.json")
    groups = load("docs/archive/2026-09-14/hr_quality_groups.json")
    roles = ["mixed", "slr_to_ipod", "ipod_to_slr"]
    variants = ["pool5m__unit", "pool5m__unit", "patch5m__linear"]
    previous = [asummary["overall"][r]["delta_e00"] for r in roles]
    current = [
        next(
            g["mean_person_delta_e00_across_seeds"]
            for g in groups
            if g["role"] == r and g["variant"] == v
        )
        for r, v in zip(roles, variants)
    ]
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    x = np.arange(3)
    for d, values, label, color in [
        (-0.18, previous, "AS · выбранный рецепт", "#087e8b"),
        (0.18, current, "HR · выбранный рецепт", "#b77e19"),
    ]:
        bars = ax.bar(x + d, values, width=0.32, label=label, color=color)
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=5, fontsize=10)
    ax.set(
        xticks=x,
        xticklabels=["Mixed", "SLR → iPod", "iPod → SLR"],
        ylim=(0, 11.5),
        ylabel="Средний по людям ΔE00 · меньше лучше",
        title="Расширение head не дало общего улучшения",
    )
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.text(
        0,
        -0.24,
        "Повторно используемые TRAIN-роли · среднее ошибок 3 seeds, не ансамбль\nHR: качество проверено; полный runtime/replay не завершён",
        transform=ax.transAxes,
        fontsize=9,
    )
    save(fig, "hr_selected")
    seg = load(
        "docs/archive/2026-09-14/evidence/data_growth_2026_09_14/facial_skin_v1/test_results.json"
    )
    values = [100 * seg["paired_128"][k]["mean_image_iou"] for k in ["uci_color_only", "cnn"]]
    fig, ax = plt.subplots(figsize=(9.4, 4.0))
    bars = ax.barh(
        ["UCI · только цвет", "Seg1 · U-Net"], values, color=["#c1b7a8", "#087e8b"], height=0.5
    )
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=7)
    ax.set(
        xlim=(0, 107),
        xlabel="Средний IoU маски по изображениям, %",
        title="Выделение кожи · одна и та же подвыборка 128 TEST-фото",
    )
    ax.text(
        0,
        -0.28,
        "Это пересечение масок, не процент правильно распознанных лиц.\nНативный приборный цвет селфи этой проверкой не измеряется.",
        transform=ax.transAxes,
        fontsize=9,
    )
    save(fig, "segmentation")
    expected = {
        "patch_small": 17374,
        "patch5m": 4962566,
        "soft_small": 15246,
        "soft5m": 4846822,
        "dynamic_small": 15246,
        "dynamic5m": 4846822,
        "pool5m": 4851846,
    }
    for variant, count in expected.items():
        layers = specs(variant)
        total = 643 + sum((a + (name != "key")) * b for name, a, b in layers)
        assert total == count
        lines = [
            f"# Luma ChromaSeed AS / HR · {variant}",
            "",
            "[Архитектуры](../ARCHITECTURES.md) · [AS-серия](../experiments/chromaseed_architecture_scale_v1.md) · [HR: все результаты](../HR_ALL_MODELS.md)",
            "",
            f"**{count:,} параметров**, включая frozen NP-якорь 643. Семейство residual-регрессии нативного Lab; вход color36 +64×18 токенов подготовленного участка. Это не самостоятельный face detector или система подбора косметики.",
            "",
            "## Полная спецификация слоёв",
            "",
            "| Слой | Вход → выход | Bias | Параметры |",
            "|---|---|---|---:|",
        ]
        for name, a, b in layers:
            lines.append(
                f"| `{name}` | {a} → {b} | {'нет' if name == 'key' else 'да'} | {(a + (name != 'key')) * b:,} |"
            )
        lines += [
            "| Frozen NP anchor | color36 → Lab3 | В составе родителя | 643 |",
            f"| **Всего** | | | **{total:,}** |",
            "",
            "## Прямой проход",
            "",
        ]
        if variant == "pool5m":
            lines += [
                "Token1 +ReLU; mean/std/max по 64 токенам. Concat 36+3×512=1572. MLP +ReLU, head3, прибавление ограниченной поправки к NP. Std содержит +1e-6 под корнем."
            ]
        elif variant.startswith("patch"):
            lines += [
                "Два token-слоя с SiLU; mean по 64 токенам; concat с color36; три MLP-слоя с SiLU; head3 и прибавление поправки к NP. Один ответ, рекуррентных шагов нет."
            ]
        else:
            lines += [
                "Два token-слоя с SiLU, context из concat(color36,mean(tokens)). Keys без bias вычисляются один раз. Четыре прохода общими весами: query(state,pred) → scores/√e → attention → pooled tokens → update MLP; state=.5state+.5tanh(update), pred+=.25head(state).",
                "",
                "Dynamic: hard mask score>0 OR top4 и sigmoid straight-through при обучении; все 64 токена всё равно кодируются и оцениваются."
                if variant.startswith("dynamic")
                else "Soft: attention использует все токены. Общие между проходами веса не умножают параметрический размер на четыре.",
            ]
        lines += [
            "",
            "## Три версии выходной функции",
            "",
            "| Версия | Формула | Новые веса |",
            "|---|---|---:|",
            "| unit / исходный AS | tanh(z) | 0 |",
            "| wide / HR | 4·tanh(z/4) | 0 |",
            "| linear / HR | z | 0 |",
            "",
            "Изменяется residual output head. Остальные нелинейности и рекуррентный state update сохранены. Все три имеют value0/derivative1 в начале координат. HR unit использует точный AS-контроль.",
            "",
            "## Обучение и выбор",
            "",
            "NP заморожен; FP32 residual bank из 6 слотов (seeds17/29/43 ×LR1e-5/1e-4). Batch64, AdamW decay.01, clip5; фиксированный cosine horizon8192. Loss=.6 mean(all-pass MSE)+.4 last-pass MSE+.001 gate penalty. Выбор шага/LR только по INNER. CUDA Graph ускоряет исполнение одинакового правила; в NumPy ответ считается FP64 после FP32-нормализации.",
            "",
            "## Все сохранённые AS-результаты этого варианта",
            "",
            "| Роль | Выбранный шаг | LR | ΔE00 | CPU median, мкс |",
            "|---|---:|---:|---:|---:|",
        ]
        row = next(r for r in asummary["rows"] if r["variant"] == variant)
        for role, r in row["roles"].items():
            lines.append(
                f"| {role} | {r['step']} | {r['lr']} | {r['delta_e00']:.9f} | {r['cpu_median_us']} |"
            )
        lines += [
            "",
            "Это среднее ошибок отдельных seeds на повторно используемых TRAIN-ролях, не независимая точность селфи. CPU время относится к готовым признакам; декодирование и сегментация исключены. Общий победитель определяется отдельным INNER-выбором, не минимумом внешних чисел этой карточки.",
            "",
            "## HR: все head / role результаты",
            "",
            "| Head | Роль | Средняя ΔE00 |",
            "|---|---|---:|",
        ]
        for r in groups:
            if r["variant"].startswith(variant + "__"):
                lines.append(
                    f"| {r['variant'].split('__')[1]} | {r['role']} | {r['mean_person_delta_e00_across_seeds']:.9f} |"
                )
        lines += [
            "",
            "**Статус HR:** аудит качества принят; runtime оборван, финального seal нет. Все seeds, хэши и выбранные checkpoints доступны в [HR records](../evidence/chromaseed_head_range_v1/results.json).",
            "",
            "## Реализация и тесты",
            "",
            "- [AS: точные конструкторы, forward, fit и consumer](../modules/scripts__chromaseed_architecture_scale.md)",
            "- [HR: unit/wide/linear](../modules/scripts__chromaseed_head_range.md)",
            "- [Все тесты соответствующих серий](../EXPERIMENTS.md)",
            "",
            "Таблица размерностей — документальная расшифровка specs/capacity; при сборке архива модель не создаётся и не исполняется.",
        ]
        (OUT / "models" / f"{variant}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    source = ROOT / "scripts/chromaseed_architecture_scale.py"
    SOURCES[source.relative_to(ROOT).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    provenance = {
        "scope": "Static plots from saved metrics; no inference/training",
        "sources": SOURCES,
        "figures": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(FIG.iterdir())
            if p.suffix in [".png", ".svg"]
        },
    }
    (OUT / "figure_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    gallery = [
        "# Галерея сохранённых измерений",
        "",
        "[Архив](README.md) · [Все серии](EXPERIMENTS.md)",
        "",
        "Графики разных задач не складываются в общую кривую прогресса. Единицы и выборки определены в исходном отчёте рядом с каждым изображением. Обложка в эту галерею измерений не включена.",
        "",
        "![AS и HR](figures/hr_selected.png)",
        "",
        "![Seg1](figures/segmentation.png)",
    ]
    for d in sorted((ROOT / "docs/benchmarks").iterdir()):
        if d.is_dir():
            plots = sorted(d.rglob("*.png"))
            if plots:
                gallery += [
                    "",
                    f"## {d.name}",
                    "",
                    f"[Все результаты, методика и ограничения](experiments/{d.name}.md)",
                    "",
                ]
                gallery += [
                    f"![{d.name} · {p.stem}](../../../{p.relative_to(ROOT).as_posix()})\n"
                    for p in plots
                ]
    (OUT / "GALLERY.md").write_text(
        "\n".join(gallery).rstrip() + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"new_metric_figures": 2, "as_architecture_cards": 7, "source_bound": True}))


if __name__ == "__main__":
    main()
