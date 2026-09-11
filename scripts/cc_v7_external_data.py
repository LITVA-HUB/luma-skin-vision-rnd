"""Decode exact frozen INTEL remainder and known-camera official source test."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import DATA_HASHES
from cc_v7_external_lock import LOCK, ROOT, checked_lock, load

from luma_skin_vision.cc.core import experts, unit
from luma_skin_vision.cc.data import sample
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

CACHE=ROOT/"data/processed/cc_v7_external128"


def publisher_pair(root,row):
    payload={}
    for kind in ("image","reference"):
        record=row[kind]
        path=(root/record["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Image/reference escaped data root")
        content=path.read_bytes()
        if len(content)!=record["bytes"] or hashlib.sha256(content).hexdigest()!=record["sha256"]:
            raise ValueError("Acquired data bytes changed")
        payload[kind]=content
    raw=cv2.imdecode(np.frombuffer(payload["image"],np.uint8),cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype!=np.uint16 or raw.ndim!=3 or raw.shape[-1]!=3:
        raise ValueError("Expected publisher linear uint16 RGB TIFF")
    words=payload["reference"].decode("utf-8-sig").split()
    if len(words)!=3:
        raise ValueError("Expected exactly three whitepoint components")
    gt=np.array([float(v) for v in words])
    if not np.isfinite(gt).all() or np.any(gt<=0):
        raise ValueError("Invalid publisher ground truth")
    rgb=raw[...,::-1].astype(np.float32)/65535
    rgb[rgb.max(axis=-1)>=.98]=0
    return rgb,unit(gt),list(raw.shape)


def prepare(digest):
    lock=checked_lock(digest)
    if CACHE.exists():
        raise ValueError("Immutable evaluation cache exists")
    data=ROOT/"data/processed/cc128"
    if {k:sha256(data/k) for k in DATA_HASHES}!=DATA_HASHES:
        raise ValueError("Source cache identity changed")
    population=load(ROOT/lock["population"])
    root=ROOT/"data/public/intel_tau_v3"
    CACHE.mkdir(parents=True)
    started=time.perf_counter()
    images,gt,classic,rows=[],[],[],[]
    for row in population["rows"]:
        rgb,label,shape=publisher_pair(root,row)
        images.append(sample(rgb,128))
        gt.append(label)
        classic.append(experts(rgb))
        rows.append({"id":row["id"],"camera":row["camera"],"group":row["group"],"primary":row["primary"],"history_linked_reference":row["history_linked_reference"],"subset":"test","shape":shape,"image_sha256":row["image"]["sha256"],"gt_sha256":row["reference"]["sha256"]})
        if len(rows)%32==0:
            print(json.dumps({"external_prepared":len(rows),"seconds":round(time.perf_counter()-started,1)}),flush=True)
    if len(rows)!=384 or sum(r["primary"] for r in rows)!=317:
        raise ValueError("Target population changed")
    np.savez_compressed(CACHE/"external.npz",images=np.stack(images).astype(np.float16),gt=np.stack(gt),experts=np.stack(classic))
    write_json(CACHE/"external_manifest.json",rows)
    source_rows=load(data/"cube_manifest.json")
    ix=np.array([i for i,r in enumerate(source_rows) if r["subset"]=="test"],dtype=np.int64)
    if len(ix)!=462:
        raise ValueError("Official source test population changed")
    values={key:read_npz_rows(data/"cube.npz",key,ix,2234) for key in ("images","gt","experts")}
    np.savez_compressed(CACHE/"source_test.npz",**values)
    write_json(CACHE/"source_test_manifest.json",[source_rows[i] for i in ix])
    write_json(CACHE/"preparation.json",{"status":"COMPLETE LOCKED EVALUATION CACHES","method_lock_sha256":sha256(LOCK),"external_rows":384,"primary_rows":317,"official_source_test_rows":462,"script_sha256":sha256(Path(__file__)),"seconds":time.perf_counter()-started,"preprocessing":"Publisher linear uint16 RGB /65535; max>=.98 masked; no GT/camera metadata used; sample area128, per-image95-percentile exposure, clamp0to4, FP16 cache. Classical algorithms use same full-resolution masked RGB.","license":"INTEL original CC BY-SA4.0 evaluation-only; SimpleCube original CC BY4.0","provenance_limit":"Sparse mirror pinned bytes verified; whole publisher multipart byte identity not verified; exact GT-hash groups are not confirmed physical scenes","known_camera_limit":"Official TEST date overlap and earlier-method exposure are disclosed","sha256":{p.name:sha256(p) for p in CACHE.iterdir() if p.is_file()}})
    print(json.dumps({"external":384,"primary":317,"known_camera":462,"seconds":time.perf_counter()-started}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--digest",required=True)
    prepare(parser.parse_args().digest)
