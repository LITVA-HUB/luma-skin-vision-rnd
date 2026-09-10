"""Freeze metadata-only remainder population; no image decoding or GT parsing."""
import json
from collections import Counter
from pathlib import Path

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT=Path(__file__).resolve().parents[1]


def select(new_records,old_records):
    old_images={f["sha256"] for r in old_records for f in r["files"] if f["path"].endswith(".tiff")}
    old_references={f["sha256"] for r in old_records for f in r["files"] if f["path"].endswith(".wp")}
    rows=[]
    for record in new_records:
        images=[f for f in record["files"] if f["path"].endswith(".tiff")]
        refs=[f for f in record["files"] if f["path"].endswith(".wp")]
        if len(images)!=1 or len(refs)!=1:
            raise ValueError("Expected one image/reference pair")
        image,reference=images[0],refs[0]
        if image["sha256"] in old_images:
            raise ValueError("Historical image overlap")
        history_linked=reference["sha256"] in old_references
        rows.append({"id":image["path"],"camera":image["path"].split("/")[0],"image":image,"reference":reference,"group":reference["sha256"],"history_linked_reference":history_linked,"primary":not history_linked,"role":"external_test_only"})
    if len({r["id"] for r in rows})!=len(rows) or len({r["image"]["sha256"] for r in rows})!=len(rows):
        raise ValueError("Duplicate image identities")
    return sorted(rows,key=lambda r:r["id"])


def run():
    out=ROOT/"docs/data/provenance/cc_v7/external_population.json"
    if out.exists():
        raise ValueError("Existing immutable population lock")
    new=ROOT/"docs/data/provenance/cc_v3/verification_report.json"
    old=ROOT/"docs/data/provenance/cc_v2/verification_report.json"
    new_report=json.loads(new.read_text())
    old_report=json.loads(old.read_text())
    rows=select(new_report["records"],old_report["records"])
    counts=dict(Counter(r["camera"] for r in rows if r["primary"]))
    if len(rows)!=384 or counts!={"Canon_5DSR":103,"Nikon_D810":112,"Sony_IMX135_BLCCSC":102}:
        raise ValueError("Population differs from prior metadata audit")
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,{"status":"METADATA POPULATION FROZEN; METHOD LOCK STILL REQUIRED BEFORE DECODING","new_image_pixels_or_gt_values_read":False,"target_errors_observed":False,"primary_n":317,"primary_cameras":counts,"sensitivity_n":384,"historical_reference_linked_rows":67,"group_limit":"Exact reference-file hash only, not a verified physical scene ID; camera-local white points cannot establish cross-camera scene identity","rights":"Original INTEL-TAU CC BY-SA4.0; evaluation only in V7, no target model/risk fitting; original multipart byte identity still unverified for sparse mirror","source_verification_sha256":{new.relative_to(ROOT).as_posix():sha256(new),old.relative_to(ROOT).as_posix():sha256(old)},"script_sha256":sha256(Path(__file__)),"rows":rows})
    print(json.dumps({"primary":317,"by_camera":counts,"sensitivity":384}))


if __name__=="__main__":
    run()
