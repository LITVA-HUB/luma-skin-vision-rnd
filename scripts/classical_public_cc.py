import json
from pathlib import Path

from luma_skin_vision.cc.benchmark import indices, load
from luma_skin_vision.cc.core import EXPERT_NAMES, angular, reproduction, summarize
from luma_skin_vision.experiment import source_identity, write_json

root = Path(__file__).resolve().parents[1]
cache, rows = load(root / "data/processed/cc128")
result = {
    "source": source_identity(),
    "status": "REPRODUCED LOCALLY",
    "dataset": "SimpleCube++ v2 real illuminant GT",
    "protocols": {},
}
for protocol in ["official", "camera"]:
    ix = indices(rows, protocol)
    result["protocols"][protocol] = {"counts": {k: len(v) for k, v in ix.items()}, "methods": {}}
    for i, name in enumerate(EXPERT_NAMES):
        p, g = cache["experts"][ix["test"], i], cache["gt"][ix["test"]]
        result["protocols"][protocol]["methods"][name] = {
            "recovery": summarize(angular(p, g)),
            "reproduction": summarize(reproduction(p, g)),
        }
out = root / "docs/benchmarks/public_classical_initial.json"
write_json(out, result)
print(json.dumps(result, indent=2))
