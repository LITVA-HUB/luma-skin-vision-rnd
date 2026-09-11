"""Validate the publication against archived evidence; no data access or training."""

import csv
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

from build_research_archive import tables

ROOT = Path(__file__).resolve().parents[1]


def main():
    pub = ROOT / "docs/publication"
    catalog = json.loads((pub / "catalog.json").read_text(encoding="utf-8"))["families"]
    actual = {p.name for p in (ROOT / "docs/benchmarks").iterdir() if p.is_dir()}
    assert {c["id"] for c in catalog} == actual
    count = 0
    for c in catalog:
        if not c["report"]:
            continue
        p = ROOT / c["report"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == c["report_sha256"], c["id"]
        assert tables(p.read_text(encoding="utf-8-sig")) == c["tables"], c["id"]
        count += sum(len(t["rows"]) for t in c["tables"])
        if c["publication_figure"]:
            assert (pub / c["publication_figure"]).is_file()
    with (pub / "all_report_tables.csv").open(encoding="utf-8", newline="") as f:
        assert len(list(csv.DictReader(f))) == count
    provenance = json.loads((pub / "figure_provenance.json").read_text(encoding="utf-8"))
    for p, h in provenance["sources"].items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h, p
    test = json.loads(
        (ROOT / "docs/benchmarks/skin_mskcc_selective_v1/test_results.json").read_text()
    )
    assert len(test["color_comparators"]) == 41
    assert test["population"] == {"images": 400, "people": 10}
    assert round(test["risk_models"]["ensemble__Proposed"]["full"]["mean"], 4) == 4.4570
    assert round(test["color_comparators"]["fusion_ensemble"]["full"]["mean"], 4) == 4.3005
    ci = test["paired"]["ensemble"]["proposed_minus_C_plus_risk80_ci95"]
    assert ci[0] < 0 < ci[1]
    transfer = json.loads(
        (ROOT / "docs/benchmarks/skin_correction_transfer_v1/summary.json").read_text()
    )
    assert (
        transfer["all_heads_worsen_both_unseen_means"]
        and not transfer["strongest_local_baseline_beaten"]
    )
    missing = []
    links = 0
    for p in [ROOT / "README.md", *pub.glob("*.md")]:
        for link in re.findall(r"\]\(([^)]+)\)", p.read_text(encoding="utf-8")):
            if re.match(r"[a-z]+:|#", link):
                continue
            dest = unquote(link.split("#")[0].split(' "')[0].strip("<>"))
            if dest:
                links += 1
                if not (p.parent / dest).exists():
                    missing.append((p.relative_to(ROOT).as_posix(), dest))
    assert not missing, missing
    print(
        json.dumps(
            {
                "status": "PASS",
                "benchmark_directories": len(catalog),
                "literal_table_rows": count,
                "publication_local_links": links,
                "scope": "report-only; no experiment rerun",
            }
        )
    )


if __name__ == "__main__":
    main()
