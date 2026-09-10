"""V6 evaluation in original camera RGB; actions trained in a residual frame."""
import numpy as np
import torch
from cc_v4_experiment import action_rgb, oracle_errors, refinement_summary, risk_summary

from luma_skin_vision.cc.core import angular, reproduction, summarize


@torch.no_grad()
def evaluate(model, x, gt, batch):
    model.eval()
    arrays = {key: [] for key in ("point_action", "trajectory_actions", "trajectory_risk", "valid")}
    observed_query_counts = set()
    for start in range(0, len(x), batch):
        cache = model.encode(x[start:start + batch])
        selected = model.select(cache, steps=4)
        observed_query_counts.add(selected["query_count"])
        arrays["point_action"].append((cache["point_action"] + cache["anchor_logchroma"]).cpu().numpy())
        for key in ("trajectory_actions", "trajectory_risk", "valid"):
            arrays[key].append(selected["absolute_trajectory_actions" if key == "trajectory_actions" else key].cpu().numpy())
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    label = gt.cpu().numpy().astype(np.float64)
    point = action_rgb(arrays["point_action"].astype(np.float64))
    trajectory = action_rgb(arrays["trajectory_actions"].astype(np.float64))
    arrays["base_pred"] = point
    arrays["base_reproduction"] = reproduction(point, label)
    arrays["trajectory_reproduction"] = np.stack([reproduction(trajectory[:, step], label) for step in range(4)], -1)
    arrays["trajectory_recovery"] = np.stack([angular(trajectory[:, step], label) for step in range(4)], -1)
    arrays["oracle2"], arrays["oracle4"] = oracle_errors(arrays["point_action"], arrays["trajectory_actions"], label)
    arrays["pred"] = trajectory[:, 1]
    arrays["reproduction"] = arrays["trajectory_reproduction"][:, 1]
    arrays["risk"] = arrays["trajectory_risk"][:, 1]
    if observed_query_counts != {103}:
        raise ValueError("Four-stage policy changed its observed query budget")
    metrics = {"base_reproduction": summarize(arrays["base_reproduction"]),
               "observed_four_stage_queries": sorted(observed_query_counts),
               "valid_fraction": float(arrays["valid"].mean()), "stages": {},
               "refinement": refinement_summary(arrays["trajectory_reproduction"], arrays["trajectory_risk"]),
               "oracle2_mean": float(arrays["oracle2"].mean()), "oracle4_mean": float(arrays["oracle4"].mean())}
    for stage in (1, 2, 4):
        err = arrays["trajectory_reproduction"][:, stage - 1]
        metrics["stages"][str(stage)] = {"reproduction": summarize(err),
            "recovery": summarize(arrays["trajectory_recovery"][:, stage - 1]),
            "risk": risk_summary(err, arrays["trajectory_risk"][:, stage - 1])}
    return metrics, arrays

