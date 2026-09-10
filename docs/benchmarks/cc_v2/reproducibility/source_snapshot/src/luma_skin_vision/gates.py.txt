"""Require explicit measured-protocol review before fitting non-synthetic targets."""

import json
import math
from pathlib import Path

from luma_skin_vision.data import sha256
from luma_skin_vision.evaluation import repeatability


def measurement_gate(manifest, rows):
    if all(r.data_kind == "SYNTHETIC" for r in rows):
        return {"decision": "NOT APPLICABLE", "reason": "SYNTHETIC engineering mode; G1 not passed"}
    approval_path = Path(manifest).parent / "measurement_approval.json"
    if not approval_path.exists():
        raise ValueError("GATE 1: measurement_approval.json required before real-target training")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if (
        approval.get("schema_version") != "1.0"
        or approval.get("dataset_hash") != sha256(manifest)
        or approval.get("decision") != "PASS"
        or not approval.get("reviewer_role")
        or not approval.get("protocol_id")
        or approval.get("reference_accuracy_reviewed") is not True
        or approval.get("registration_reviewed") is not True
    ):
        raise ValueError("GATE 1: missing review evidence or manifest binding mismatch")
    limit = approval.get("repeatability_p95_limit")
    if not isinstance(limit, (float, int)) or not math.isfinite(limit) or limit <= 0:
        raise ValueError("GATE 1: positive preregistered repeatability limit required")
    refs = {r.reference_measurement_id: r.reference_repeats_lab for r in rows}
    measured = repeatability(list(refs.values()))
    if measured["p95_pair_delta_e00"] > limit:
        raise ValueError("GATE 1: reference repeatability exceeds approved threshold")
    if any(r.makeup_protocol != "none" for r in rows):
        raise ValueError("GATE 1: primary operating protocol requires no makeup")
    return {
        **approval,
        "measured_repeatability": measured,
        "approval_sha256": sha256(approval_path),
    }
