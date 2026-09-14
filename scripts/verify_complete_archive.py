"""Verify the publication without importing or running any research implementation."""

import ast
import csv
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/archive/2026-09-14"


def load(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    from build_complete_archive import table_extract

    manifest = load(OUT / "import_manifest.json")
    for row in manifest["exact_copies"]:
        p = ROOT / row["path"]
        assert p.is_file(), row["path"]
        assert sha(p) == row["sha256"], ("import hash mismatch", row["path"])
        assert p.stat().st_size == row["bytes"], row["path"]
    catalog = load(ROOT / "docs/publication/catalog.json")["families"]
    actual = {p.name for p in (ROOT / "docs/benchmarks").iterdir() if p.is_dir()}
    assert {r["id"] for r in catalog} == actual
    table_rows = 0
    for c in catalog:
        assert (OUT / "experiments" / f"{c['id']}.md").is_file()
        if c["report"]:
            p = ROOT / c["report"]
            assert sha(p) == c["report_sha256"], c["id"]
            assert table_extract(p.read_text(encoding="utf-8-sig")) == c["tables"], c["id"]
            table_rows += sum(len(t["rows"]) for t in c["tables"])
    with (ROOT / "docs/publication/all_report_tables.csv").open(encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == table_rows
    for row in csv_rows:
        c = next(c for c in catalog if c["id"] == row["family"])
        t = c["tables"][int(row["table"]) - 1]
        assert json.loads(row["headers"]) == t["headers"]
        assert json.loads(row["cells"]) == t["rows"][int(row["row"]) - 1]
    modules = load(OUT / "source_catalog.json")
    actual_py = {
        p.relative_to(ROOT).as_posix()
        for top in ["src", "scripts", "tests", "apps", "docs"]
        for p in (ROOT / top).rglob("*.py")
        if not any(s in p.parts for s in ["node_modules", "__pycache__", ".venv", "dist"])
    }
    assert {m["path"] for m in modules} == actual_py
    test_count = 0
    for m in modules:
        p = ROOT / m["path"]
        assert sha(p) == m["sha256"], m["path"]
        assert (OUT / m["page"]).is_file(), m["page"]
        tree = ast.parse(p.read_text(encoding="utf-8-sig"))
        functions = [
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
        ]
        assert len(functions) == len(m["tests"]), m["path"]
        if m["is_test_module"]:
            test_count += len(functions)
    assert len(load(OUT / "test_catalog.json")["definitions"]) == test_count
    missing = []
    links = 0
    for p in [ROOT / "README.md", *OUT.rglob("*.md")]:
        if "evidence" in p.relative_to(OUT).parts if p != ROOT / "README.md" else False:
            continue  # immutable source notes may refer to local data stores
        t = re.sub(r"```[^\n]*\n.*?```", "", p.read_text(encoding="utf-8-sig"), flags=re.S)
        for link in re.findall(r"\]\(([^)]+)\)", t):
            if re.match(r"[a-z]+:|#", link, re.I):
                continue
            dest = unquote(link.split("#")[0].split(' "')[0].strip("<>"))
            if dest:
                links += 1
                if not (p.parent / dest).is_file() and not (p.parent / dest).is_dir():
                    missing.append((p.relative_to(ROOT).as_posix(), dest))
    assert not missing, missing[:40]
    figure = load(OUT / "figure_provenance.json")
    for path, digest in figure["sources"].items():
        assert sha(ROOT / path) == digest, path
    for name, digest in figure["figures"].items():
        assert sha(OUT / "figures" / name) == digest, name
    design = load(OUT / "design_manifest.json")
    assert sha(OUT / design["asset"]) == design["sha256"]
    pause = load(ROOT / "docs/research/PAUSE_2026-09-14.json")
    assert pause["research_state"] == "paused_by_user"
    assert pause["last_runtime_response_receipts"] == 126
    assert not pause["runtime_final_seal_exists"]
    hr = load(OUT / "evidence/chromaseed_head_range_v1/results.json")
    assert len(hr["records"]) == 207
    assert (
        sha(OUT / "evidence/chromaseed_head_range_v1/results.json")
        == "14d4613402f1c93293a545d7780adf17e2cbd6e0ae17bfbeb9ccdec2a756eb74"
    )
    audit = load(ROOT / "docs/benchmarks/chromaseed_head_range_v1/audit.json")
    assert audit["passed"]
    failure = load(
        OUT / "evidence/chromaseed_head_range_v1/verification_v1/host_recovery_v1/completion.json"
    )
    assert not failure["passed"] and failure["completed_host_checks"] == 127
    seg = load(OUT / "evidence/data_growth_2026_09_14/facial_skin_v1/test_results.json")
    assert seg["test"]["images"] == 2000 and seg["ground_truth_color_accuracy"] is None
    assert round(seg["test"]["mean_image_iou"], 8) == 0.94195626
    old = load(ROOT / "docs/benchmarks/skin_mskcc_selective_v1/test_results.json")
    assert old["population"] == {"images": 400, "people": 10}
    assert (
        old["color_comparators"]["fusion_ensemble"]["full"]["mean"]
        < old["risk_models"]["ensemble__Proposed"]["full"]["mean"]
    )
    # Scan newly imported text and new documentation, not private directories.
    forbidden = re.compile(
        r"(?:ISIC_\d{6,}|gh[pousr]_[A-Za-z0-9]{24,}|sk-proj-[A-Za-z0-9_-]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|photo_2026-09-14)"
    )
    scan_paths = (
        {ROOT / r["path"] for r in manifest["exact_copies"]}
        | set(OUT.rglob("*"))
        | {ROOT / "README.md"}
    )
    hits = []
    for p in scan_paths:
        if p.is_file() and p.suffix in [
            ".md",
            ".json",
            ".csv",
            ".py",
            ".txt",
            ".log",
            ".js",
            ".jsx",
            ".html",
            ".toml",
        ]:
            if p.name == "source_catalog.json" or p.name.endswith("verify_complete_archive.md"):
                continue  # literal scanner signatures appear in the static source atlas
            if p.name == "verify_complete_archive.py":
                continue
            if forbidden.search(p.read_text(encoding="utf-8-sig", errors="replace")):
                hits.append(p.relative_to(ROOT).as_posix())
    assert not hits, ("publication scan", hits)
    result = {
        "status": "PASS",
        "scope": "Publication-only: hashes, AST, literal tables, links, plot provenance and key evidence assertions. No model tests/training/evaluation rerun.",
        "exact_imported_files": len(manifest["exact_copies"]),
        "benchmark_directories": len(catalog),
        "literal_table_rows": table_rows,
        "python_modules": len(modules),
        "test_definitions_not_executed": test_count,
        "local_links_checked": links,
        "new_metric_figures": len(figure["figures"]) // 2,
        "forbidden_text_hits": len(hits),
    }
    (OUT / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
