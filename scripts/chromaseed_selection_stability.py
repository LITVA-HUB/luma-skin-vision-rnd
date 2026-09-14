"""Sensitivity of model selection conditional on fixed person-held-out predictions."""
from __future__ import annotations

import numpy as np

from luma_skin_vision.color import delta_e00


def person_losses(predictions, target, person):
    predictions, target, person = np.asarray(predictions), np.asarray(target), np.asarray(person)
    if predictions.ndim != 3 or predictions.shape[1:] != target.shape or target.shape != (len(person), 3):
        raise ValueError("seed/image/Lab shapes differ")
    if not np.isfinite(predictions).all() or not np.isfinite(target).all():
        raise ValueError("nonfinite colors")
    errors = delta_e00(predictions, target[None, :, :])
    return np.array([errors[:, person == p].mean(axis=1).mean() for p in np.unique(person)])


def bootstrap_counts(camera, repeats, seed):
    camera = np.asarray(camera)
    if camera.ndim != 1 or len(camera) < 2 or repeats < 1:
        raise ValueError("invalid person bootstrap dimensions")
    rng = np.random.default_rng(seed)
    counts = np.zeros((repeats, len(camera)), dtype=np.int32)
    for label in np.unique(camera):
        members = np.flatnonzero(camera == label)
        draws = members[rng.integers(len(members), size=(repeats, len(members)))]
        for member in members:
            counts[:, member] = (draws == member).sum(axis=1)
    return counts


def tie_order(configs):
    return np.array(sorted(range(len(configs)), key=lambda i: (
        configs[i]['steps'], configs[i]['alpha'], configs[i]['width_factor'])))


def winner_indices(scores, configs):
    order = tie_order(configs)
    return order[np.argmin(np.asarray(scores)[:, order], axis=1)]


def original_ranks(scores, original, configs):
    position = np.argsort(tie_order(configs))
    score = scores[:, original, None]
    return 1 + ((scores < score) | ((scores == score) & (position < position[original]))).sum(axis=1)


def distribution(values):
    return dict(mean=float(np.mean(values)), minimum=float(np.min(values)),
                median=float(np.median(values)), p95=float(np.quantile(values, .95)),
                maximum=float(np.max(values)))


def paired_range(full_gap, samples):
    return dict(mean=float(full_gap), conditional_percentiles_2_5_97_5=np.quantile(samples, [.025, .975]).tolist(),
                fraction_below_zero=float(np.mean(samples < 0.)))


def diagnose(losses, configs, counts, parent_index):
    losses, counts = np.asarray(losses, dtype=np.float64), np.asarray(counts)
    if (losses.ndim != 2 or min(losses.shape) < 2 or len(configs) != len(losses)
            or not np.isfinite(losses).all() or np.any(losses < 0.)
            or counts.ndim != 2 or counts.shape[1] != losses.shape[1] or len(counts) < 1
            or not np.isfinite(counts).all() or np.any(counts < 0)
            or not np.all(counts == counts.astype(np.int64))
            or not np.all(counts.sum(axis=1) == losses.shape[1])
            or parent_index not in range(len(configs))):
        raise ValueError("invalid candidate/person/count matrix")
    full = losses.mean(axis=1)
    order = tie_order(configs)
    ranking = order[np.argsort(full[order], kind='stable')]
    original, runner = int(ranking[0]), int(ranking[1])
    deletion = (losses.sum(axis=1)[None, :] - losses.T) / (losses.shape[1] - 1)
    resampled = counts @ losses.T / losses.shape[1]
    dw, bw = winner_indices(deletion, configs), winner_indices(resampled, configs)
    br, dr = original_ranks(resampled, original, configs), original_ranks(deletion, original, configs)
    parent_gap = resampled[:, original] - resampled[:, parent_index]
    runner_gap = resampled[:, original] - resampled[:, runner]
    result = dict(original_index=original, runner_up_index=runner, parent_index=int(parent_index),
                  n_people=losses.shape[1], n_candidates=len(configs), bootstrap_repeats=len(counts),
                  full_person_mean=float(full[original]),
                  deletion_switches=int((dw != original).sum()),
                  deletion_original_rank=distribution(dr),
                  deletion_full_oof_regret=distribution(full[dw] - full[original]),
                  bootstrap_original_frequency=float(np.mean(bw == original)),
                  bootstrap_unique_winners=len(np.unique(bw)),
                  bootstrap_weak_alpha_frequency=float(np.mean(np.array([c['alpha'] for c in configs])[bw] < .1)),
                  bootstrap_positive_steps_frequency=float(np.mean(np.array([c['steps'] for c in configs])[bw] > 0)),
                  bootstrap_original_rank=distribution(br),
                  bootstrap_full_oof_regret=distribution(full[bw] - full[original]),
                  paired_vs_parent=paired_range(full[original] - full[parent_index], parent_gap),
                  paired_vs_runner_up=paired_range(full[original] - full[runner], runner_gap),
                  candidates=[dict(**c, index=i, full_person_mean=float(full[i]),
                                   deletion_wins=int(np.sum(dw == i)), bootstrap_wins=int(np.sum(bw == i)))
                              for i, c in enumerate(configs)])
    trace = dict(deletion_winners=dw, bootstrap_winners=bw,
                 deletion_original_ranks=dr, bootstrap_original_ranks=br,
                 gap_vs_parent=parent_gap, gap_vs_runner_up=runner_gap)
    return result, trace
