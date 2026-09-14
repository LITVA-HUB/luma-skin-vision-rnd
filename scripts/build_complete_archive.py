"""Build the 2026-09-14 documentation atlas using saved text and Python AST only.

Never imports model/training modules, reads participant arrays, or runs experiments.
Run from any directory with Python >= 3.11. Existing scientific evidence is immutable.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/archive/2026-09-14"
PUB = ROOT / "docs/publication"


def read(p):
    return p.read_text(encoding="utf-8-sig")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, s):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s.rstrip() + "\n", encoding="utf-8", newline="\n")


def dump(p, obj):
    write(p, json.dumps(obj, ensure_ascii=False, indent=2))


def esc(s):
    return str(s).replace("|", "&#124;").replace("\n", " ")


def rel(p):
    return p.relative_to(ROOT).as_posix()


def slug(p):
    return rel(p).replace("/", "__").removesuffix(".py")


def table_extract(t):
    tables = []
    current = []
    fenced = False
    for line in t.splitlines() + [""]:
        if line.strip().startswith("```"):
            fenced = not fenced
        if not fenced and line.strip().startswith("|"):
            current.append([s.strip() for s in line.strip().strip("|").split("|")])
        elif current:
            if len(current) > 1 and all(re.fullmatch(r"[:\- ]+", c) for c in current[1]):
                tables.append({"headers": current[0], "rows": current[2:]})
            current = []
    return tables


def report_for(d):
    overrides = {
        "public_runs": "public_benchmark_report.md",
        "fourier_ridge_v1": "fourier_representation_report.md",
        "fourier_ridge_v2": "fourier_representation_report.md",
        "skin_color_sampling_mass_v1": "skin_color_sampling_v1/report.md",
    }
    if d.name in overrides:
        return ROOT / "docs/benchmarks" / overrides[d.name]
    return next(
        (
            p
            for p in [d / "report.md", d.parent / f"{d.name}_report.md", d / "seed17_report.md"]
            if p.exists()
        ),
        None,
    )


def source_page(p):
    text = read(p)
    tree = ast.parse(text)
    nodes = []
    classes = []
    constants = []
    imports = []
    architecture_nodes = []
    tests = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id.isupper() for t in targets):
                constants.append(
                    {"line": node.lineno, "source": ast.get_source_segment(text, node)}
                )
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.get_source_segment(text, node))
        if isinstance(node, ast.ClassDef):
            classes.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "bases": [ast.unparse(b) for b in node.bases],
                }
            )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nodes.append(
                {
                    "name": node.name,
                    "kind": type(node).__name__,
                    "line": node.lineno,
                    "end_line": node.end_lineno,
                    "docstring": ast.get_docstring(node) or "",
                }
            )
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                tests.append(
                    {
                        "name": node.name,
                        "line": node.lineno,
                        "end_line": node.end_lineno,
                        "docstring": ast.get_docstring(node) or "",
                        "decorators": [ast.unparse(d) for d in node.decorator_list],
                        "assertions": [
                            ast.unparse(n) for n in ast.walk(node) if isinstance(n, ast.Assert)
                        ],
                        "source": ast.get_source_segment(text, node),
                    }
                )
            if node.name in {
                "__init__",
                "forward",
                "__call__",
                "specs",
                "capacity",
                "gate",
                "fit",
                "predict",
                "skin_loss",
                "loss",
                "color36",
                "render",
                "export",
            }:
                architecture_nodes.append(node)
    identity = slug(p)
    is_test = p.name.startswith("test_")
    category = "tests" if is_test else "modules"
    page = OUT / category / f"{identity}.md"
    back = "TESTS.md" if is_test else "SOURCE_INDEX.md"
    lines = [
        f"# `{rel(p)}`",
        "",
        f"[Архив](../README.md) · [Индекс](../{back}) · [Полный исходник](../../../../{rel(p)})",
        "",
        "> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.",
        "",
        ast.get_docstring(tree)
        or "Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.",
        "",
        f"SHA-256 исходника: `{sha(p)}`. Строк: **{len(text.splitlines())}**.",
        "",
    ]
    if imports:
        lines += ["## Зависимости", "", "```python", *imports, "```", ""]
    if constants:
        lines += [
            "## Зафиксированные конфигурации и константы",
            "",
            "Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.",
            "",
        ]
        for c in constants:
            lines += [
                f"[Строка {c['line']}](../../../../{rel(p)}#L{c['line']})",
                "",
                "```python",
                c["source"],
                "```",
                "",
            ]
    if classes:
        lines += [
            "## Классы и наследование",
            "",
            "Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.",
            "",
            "| Класс | Базовые классы | Исходник |",
            "|---|---|---|",
        ]
        lines += [
            f"| `{c['name']}` | {esc(', '.join(c['bases']) or '—')} | [L{c['line']}](../../../../{rel(p)}#L{c['line']}) |"
            for c in classes
        ]
    lines += [
        "",
        "## Определения верхнего уровня",
        "",
        "| Имя | Вид | Назначение из docstring | Исходник |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| `{n['name']}` | {n['kind']} | {esc(n['docstring'] or 'См. реализацию')} | [L{n['line']}](../../../../{rel(p)}#L{n['line']}) |"
        for n in nodes
    ]
    if not is_test and architecture_nodes:
        lines += [
            "",
            "## Устройство, вычисление ответа и обучение",
            "",
            "Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.",
            "",
        ]
        for n in sorted(architecture_nodes, key=lambda n: n.lineno):
            lines += [
                f"<details><summary>{n.name} · L{n.lineno}–{n.end_lineno}</summary>",
                "",
                "```python",
                ast.get_source_segment(text, n),
                "```",
                "",
                "</details>",
                "",
            ]
    if tests:
        lines += [
            "",
            f"## Все тестовые определения ({len(tests)})",
            "",
            "Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.",
            "",
        ]
        for t in tests:
            lines += [
                f"### `{t['name']}` · L{t['line']}",
                "",
                t["docstring"] or "Проверяемые условия перечислены в теле теста ниже.",
                "",
                "<details><summary>Условия, параметризация и полное тело теста</summary>",
                "",
                "```python",
                *["@" + d for d in t["decorators"]],
                t["source"],
                "```",
                "",
                "</details>",
                "",
            ]
    write(page, "\n".join(lines))
    return {
        "path": rel(p),
        "sha256": sha(p),
        "lines": len(text.splitlines()),
        "docstring": ast.get_docstring(tree) or "",
        "classes": classes,
        "definitions": nodes,
        "configuration_declarations": constants,
        "imports": imports,
        "page": page.relative_to(OUT).as_posix(),
        "tests": [{k: v for k, v in t.items() if k != "source"} for t in tests],
        "is_test_module": is_test,
    }


def catalog(modules):
    legacy = {c["id"]: c for c in json.loads(read(PUB / "catalog.json"))["families"]}
    cards = []
    rows = []
    documentation = [
        p
        for p in (ROOT / "docs").rglob("*.md")
        if "archive" not in p.parts and "publication" not in p.parts
    ]
    index = [
        "# Все серии и эксперименты",
        "",
        "[Главная архива](README.md) · [Механизмы моделей](ARCHITECTURES.md) · [Тесты](TESTS.md) · [Наблюдения](OBSERVATIONS.md)",
        "",
        "Каталог охватывает все папки `docs/benchmarks`. Служебные папки, переоценки, подбор гиперпараметров и самостоятельные архитектуры различаются по описанию. Подготовленные P3/Seg2 и отдельные диагностики перечислены в [статусе остановки](STOP_STATUS.md).",
        "",
        "| Серия | Отчёт и статус доказательств | Таблиц / строк |",
        "|---|---|---|",
    ]
    observations = [
        "# Наблюдения, решения и отрицательные результаты",
        "",
        "[Главная](README.md) · [Синтез результатов](RESULTS.md)",
        "",
        "Полный указатель исходных исследовательских заметок: неудачные гипотезы, исправления, источники, планы и ограничения. Старые слова `active`, `next`, `unopened` описывают момент записи и не возобновляют работу. Текущий статус — [остановлено](STOP_STATUS.md).",
        "",
        "| Документ | Заголовок |",
        "|---|---|",
    ]
    for p in sorted(documentation):
        if any(s in p.parts for s in ["research", "data", "architecture", "superpowers", "ip"]):
            heading = next(
                (line.lstrip("# ") for line in read(p).splitlines() if line.startswith("#")), p.stem
            )
            observations.append(f"| [{p.name}](../../../{rel(p)}) | {esc(heading)} |")
    observations += [
        "",
        "Полный исторический журнал продолжений: [RESEARCH_CONTINUATION_HISTORY.md](evidence/RESEARCH_CONTINUATION_HISTORY.md). Старые команды в нём не исполняются.",
    ]
    write(OUT / "OBSERVATIONS.md", "\n".join(observations))
    for d in sorted((ROOT / "docs/benchmarks").iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        report = report_for(d)
        stem = re.sub(r"_v\d+$", "", name)
        related = [p for p in documentation if stem in p.name or name in p.as_posix()]
        related_modules = [m for m in modules if stem in Path(m["path"]).stem]
        # Explicit aliases where the benchmark name differs from implementation.
        aliases = {
            "facial_skin_v1": ["skin_face_segment", "skin_face_train", "skin_face_evaluate"],
            "skin_mskcc_pixels_v1": ["skin_mskcc_vote"],
            "skin_mskcc_pixel_ablation_v2": ["skin_mskcc_vote_v2"],
            "skin_capture_v1": ["skin_capture_model"],
            "public_runs": ["cc/model.py", "cc/benchmark.py"],
        }
        related_modules += [
            m
            for m in modules
            if any(a in m["path"] for a in aliases.get(name, [])) and m not in related_modules
        ]
        tables = table_extract(read(report)) if report else []
        facts = []
        for p in sorted(d.glob("*.json")):
            j = json.loads(read(p))
            if isinstance(j, dict):
                facts.append(
                    {
                        "path": rel(p),
                        "sha256": sha(p),
                        "fields": {
                            k: v
                            for k, v in j.items()
                            if k
                            in {
                                "passed",
                                "status",
                                "primary_exit_code",
                                "equivalence_accepted",
                                "seconds",
                                "scope",
                                "test_now_exposed",
                            }
                        },
                    }
                )
        c = legacy.get(name, {}).copy()
        c.update(
            id=name,
            directory=rel(d),
            report=rel(report) if report else None,
            report_sha256=sha(report) if report else None,
            tables=tables,
            publication_figure=c.get("publication_figure"),
            evidence_flags=facts,
            related_documents=[rel(p) for p in related],
            source_modules=[m["path"] for m in related_modules],
            observation_ru=c.get(
                "observation_ru",
                "См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.",
            ),
        )
        count = sum(len(t["rows"]) for t in tables)
        status = (
            "Есть исходный отчёт" if report else "Служебная папка / неполный этап; см. артефакты"
        )
        if name == "chromaseed_head_range_v1":
            status = "Качество проверено; runtime оборван, итоговая печать отсутствует"
        if name == "chromaseed_neural_blocks_v1":
            status = "Эквивалентность не принята; отрицательная диагностика сохранена"
        c["publication_status"] = status
        cards.append(c)
        index.append(f"| [{name}](experiments/{name}.md) | {status} | {len(tables)} / {count} |")
        lines = [
            f"# {name}",
            "",
            "[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)",
            "",
            f"**Статус документации:** {status}.",
            "",
            c["observation_ru"],
            "",
            f"[Полная папка артефактов](../../../../{rel(d)})",
            "",
        ]
        if report:
            lines += [
                f"[Полный исходный отчёт: методика, все результаты, ограничения](../../../../{rel(report)})",
                "",
                f"SHA-256 отчёта: `{sha(report)}`.",
                "",
            ]
        lines += [
            "## Архитектура, протокол и решения",
            "",
            "Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.",
            "",
        ]
        lines += [f"- [{p.name}](../../../../{rel(p)})" for p in related]
        lines += [
            "",
            "## Реализация и все связанные тесты",
            "",
            "Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.",
            "",
        ]
        lines += [f"- [{m['path']}](../{m['page']})" for m in related_modules]
        lines += [
            "",
            "## Сохранённые проверки",
            "",
            "Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.",
            "",
            "| Артефакт | Зафиксированные поля | SHA-256 |",
            "|---|---|---|",
        ]
        lines += [
            f"| [{Path(f['path']).name}](../../../../{f['path']}) | {esc(json.dumps(f['fields'], ensure_ascii=False))} | `{f['sha256']}` |"
            for f in facts
        ]
        lines += [
            "",
            "## Все таблицы исходного отчёта",
            "",
            "Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.",
            "",
        ]
        for ti, tab in enumerate(tables, 1):
            lines += [
                f"### Таблица {ti}",
                "",
                "| " + " | ".join(tab["headers"]) + " |",
                "| " + " | ".join(["---"] * len(tab["headers"])) + " |",
            ]
            for ri, row in enumerate(tab["rows"], 1):
                lines.append("| " + " | ".join(row) + " |")
                rows.append(
                    {
                        "family": name,
                        "report": rel(report),
                        "table": ti,
                        "row": ri,
                        "headers": json.dumps(tab["headers"], ensure_ascii=False),
                        "cells": json.dumps(row, ensure_ascii=False),
                    }
                )
            lines.append("")
        if not tables:
            lines += [
                "У этой папки нет табличного итогового отчёта. Отсутствующие результаты не восстановлены из предположений.",
                "",
            ]
        write(OUT / "experiments" / f"{name}.md", "\n".join(lines))
    write(OUT / "EXPERIMENTS.md", "\n".join(index))
    dump(
        PUB / "catalog.json",
        {
            "scope": f"{len(cards)} benchmark directories, not independent architectures or unique fits; updated 2026-09-14",
            "families": cards,
        },
    )
    with (PUB / "all_report_tables.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "report", "table", "row", "headers", "cells"])
        w.writeheader()
        w.writerows(rows)
    return cards, rows


def run_index():
    """Index every retained run, including those without a benchmark directory."""
    metadata = json.loads(read(OUT / "run_metadata_catalog.json"))
    intro = [
        "# Полный указатель каталогов запусков",
        "",
        "[Архив](README.md) · [88 каталогов отчётов](EXPERIMENTS.md) · [Индекс всех файлов и хэшей](run_metadata_catalog.json)",
        "",
        "39 исходных run-каталогов включают полноценные серии, preflight, precision, compression, timing и superseded receipts. Они пересекаются с каталогом отчётов и не прибавляются к нему как 39 новых архитектур. Исходные JSON/логи скопированы без изменения. Исключены подробные Windows desktop snapshots, для них оставлены хэши и причины исключения. Массивы и фотографии не копируются.",
        "",
        "| Каталог | Текстовых артефактов | Опубликовано |",
        "|---|---:|---:|",
    ]
    for name, count in sorted(metadata["families"].items()):
        files = [r for r in metadata["files"] if r["original_relative_path"].split("/")[0] == name]
        available = sum(r["published_path"] is not None for r in files)
        intro.append(f"| [{name}](runs/{name}.md) | {count} | {available} |")
        lines = [
            f"# Run archive · {name}",
            "",
            "[Все run-каталоги](../RUNS.md) · [Архитектуры](../ARCHITECTURES.md) · [Точка остановки](../STOP_STATUS.md)",
            "",
            "Наличие файла не означает успешный финальный опыт. Проверяйте completion, audit, verification и исходный протокол. Superseded/failed receipts сохраняют прежнюю ошибку, а не заменяют её исправленной записью.",
            "",
            "| Исходный артефакт | Байты | SHA-256 |",
            "|---|---:|---|",
        ]
        for row in files:
            source = row["original_relative_path"]
            target = row["published_path"]
            label = (
                f"[{source}](../../../../{target})"
                if target
                else f"`{source}` — {row['exclusion_reason']}"
            )
            lines.append(f"| {label} | {row['bytes']} | `{row['sha256']}` |")
        write(OUT / "runs" / f"{name}.md", "\n".join(lines))
    write(OUT / "RUNS.md", "\n".join(intro))


def model_records(obj, source, pointer=""):
    """Literal record extraction; repeated records remain repeated evidence."""
    if isinstance(obj, dict):
        if (
            isinstance(obj.get("variant"), str)
            and isinstance(obj.get("seed"), (int, float))
            and any(k in obj for k in ["metrics", "parameters", "model_sha256", "numeric_bytes"])
        ):
            metrics = obj.get("metrics", {})
            yield {
                "source": source,
                "json_pointer": pointer,
                "variant": obj["variant"],
                "role": obj.get("role", ""),
                "seed": obj["seed"],
                "step": obj.get("step", obj.get("steps", "")),
                "lr": obj.get("lr", ""),
                "parameters": obj.get("parameters", ""),
                "numeric_bytes": obj.get("numeric_bytes", ""),
                "model_artifact": obj.get("model", ""),
                "model_sha256": obj.get("model_sha256", ""),
                "person_mean_delta_e00": metrics.get("person_mean", "")
                if isinstance(metrics, dict)
                else "",
            }
        for k, v in obj.items():
            escaped = str(k).replace("~", "~0").replace("/", "~1")
            yield from model_records(v, source, pointer + "/" + escaped)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from model_records(value, source, pointer + f"/{i}")


def main():
    sources = sorted(
        p
        for top in ["src", "scripts", "tests", "apps", "docs"]
        for p in (ROOT / top).rglob("*.py")
        if not any(s in p.parts for s in ["node_modules", "__pycache__", ".venv", "dist"])
    )
    modules = [source_page(p) for p in sources]
    dump(OUT / "source_catalog.json", modules)
    tests = [
        {"module": m["path"], "page": m["page"], **t}
        for m in modules
        if m["is_test_module"]
        for t in m["tests"]
    ]
    dump(
        OUT / "test_catalog.json",
        {"scope": "AST definitions, not executed/expanded pytest cases", "definitions": tests},
    )
    for only_tests, filename, title in [
        (False, "SOURCE_INDEX.md", "Код и устройство всех реализаций"),
        (True, "TESTS.md", "Полный каталог тестов"),
    ]:
        selected = [m for m in modules if m["is_test_module"] == only_tests]
        lines = [
            f"# {title}",
            "",
            "[Архив](README.md) · [Архитектуры](ARCHITECTURES.md) · [Все серии](EXPERIMENTS.md)",
            "",
            f"**{len(selected)} модулей.** {len(tests)} тестовых определений во всех тестовых модулях. Подсчёт выполнен статическим AST-разбором без импорта и исполнения исследовательских модулей. Число тестовых функций не равно числу запусков или параметризованных cases.",
            "",
            "При остановке исследования новые тесты моделей не запускались. Исторические результаты проверок находятся в JSON каждой серии и в протоколах. Проверки самого архива публикуются отдельно в `validation.json`.",
            "",
            "| Модуль | Строк | Классов | Тестовых определений |",
            "|---|---:|---:|---:|",
        ]
        lines += [
            f"| [{m['path']}]({m['page']}) | {m['lines']} | {len(m['classes'])} | {len(m['tests'])} |"
            for m in selected
        ]
        write(OUT / filename, "\n".join(lines))
    cards, rows = catalog(modules)
    run_index()
    # Each JSON evidence object retains its source path and hash; no metric pooling.
    evidence = []
    models = []
    for p in sorted((OUT / "evidence").rglob("*.json")):
        obj = json.loads(read(p))
        models.extend(model_records(obj, rel(p)))
        evidence.append(
            {
                "path": rel(p),
                "sha256": sha(p),
                "bytes": p.stat().st_size,
                "top_level_fields": list(obj) if isinstance(obj, dict) else None,
            }
        )
    dump(OUT / "evidence_catalog.json", evidence)
    fields = [
        "source",
        "json_pointer",
        "variant",
        "role",
        "seed",
        "step",
        "lr",
        "parameters",
        "numeric_bytes",
        "model_artifact",
        "model_sha256",
        "person_mean_delta_e00",
    ]
    with (OUT / "model_records.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(models)
    hr = json.loads(read(OUT / "evidence/chromaseed_head_range_v1/results.json"))
    groups = {}
    for r in hr["records"]:
        key = (r["role"], r["variant"])
        groups.setdefault(key, []).append(r)
    quality = []
    for (role, variant), records in sorted(groups.items()):
        values = [r["metrics"]["person_mean"] for r in records]
        quality.append(
            {
                "role": role,
                "variant": variant,
                "seeds": [r.get("seed") for r in records],
                "individual_seed_errors": values,
                "mean_person_delta_e00_across_seeds": sum(values) / len(values),
                "parameters": records[0].get("parameters"),
                "numeric_bytes": records[0].get("numeric_bytes"),
                "scope": "Reused TRAIN-only roles; seed mean, not ensemble; quality audited, runtime incomplete",
            }
        )
    dump(OUT / "hr_quality_groups.json", quality)
    hr_lines = [
        "# HR: все выбранные варианты и исходные записи",
        "",
        "[Описание HR](ARCHITECTURES.md#hr-диапазон-выхода) · [Остановка](STOP_STATUS.md) · [Все 207 исходных записей](evidence/chromaseed_head_range_v1/results.json)",
        "",
        "Это среднее ошибок отдельных seeds, не ансамблевый прогноз. Метрика — средний по людям ΔE00 на повторно используемых ролях исходного TRAIN. Выбор по INNER; по этим значениям новые победители не назначаются. В записях есть 189 вариантов для ответов и 18 унаследованных контрольных записей.",
        "",
        "| Роль | Вариант | Seeds | Параметры | Средний ΔE00 |",
        "|---|---|---|---:|---:|",
    ]
    hr_lines += [
        f"| {r['role']} | `{r['variant']}` | {esc(r['seeds'])} | {r['parameters']} | {r['mean_person_delta_e00_across_seeds']:.9f} |"
        for r in quality
    ]
    write(OUT / "HR_ALL_MODELS.md", "\n".join(hr_lines))
    stats = {
        "benchmark_directories": len(cards),
        "literal_report_tables": sum(len(c["tables"]) for c in cards),
        "literal_report_rows": len(rows),
        "python_modules": len(modules),
        "source_modules": sum(not m["is_test_module"] for m in modules),
        "test_modules": sum(m["is_test_module"] for m in modules),
        "test_definitions": len(tests),
        "class_definitions_including_helpers": sum(len(m["classes"]) for m in modules),
        "source_lines": sum(m["lines"] for m in modules),
        "external_evidence_json": len(evidence),
        "hr_result_records": len(hr["records"]),
        "hr_role_variant_groups": len(quality),
        "counting_scope": "files, declarations and literal records; never distinct models, people or unique experiments",
    }
    dump(OUT / "inventory.json", stats)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
