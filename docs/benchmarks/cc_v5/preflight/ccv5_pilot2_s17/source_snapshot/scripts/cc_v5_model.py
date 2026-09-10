# V5 exact V4 architecture: skip identity 4x4 pooling for strict CUDA backward.
"""Experimental correction-conditioned evidence model in linear camera RGB.

SYNTHETIC correctness checks are not benchmark accuracy or novelty evidence.
Actions are log(R/G), log(B/G) of positive candidate illuminants; their costs
are neutral reproduction costs, never physical surface DeltaE. The 16 initial
local offsets are an explicit proposal prior, with maximum Euclidean norm .06.

Transport uses corrected simplex RGB, a nonlinear transformation of observed
log colors. The action null holds that color at the point while retaining the
relative action input; it controls for a generic action-conditioned router.
All modes contain the same modules. The posterior/direct router fixes its last
two inputs to zero: 128 first-layer scalar weights therefore have no effect in
those modes. Equal declared parameter counts do not imply equal active graphs.
The direct mode uses the point for inference; posterior weights only diagnose it.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_large

ANGLE_NORM_EPS = 1e-8
COLOR_FLOOR = 1e-12
MAX_ACTION = 4.0


def _action_structure(actions, batch):
    if (
        actions.ndim != 3 or actions.shape[0] != batch or actions.shape[1] == 0
        or actions.shape[2] != 2 or not actions.is_floating_point()
    ):
        raise ValueError("Expected nonempty floating NxKx2 actions matching the batch")


def _assert_action_domain(actions):
    # A device-side assertion avoids copying a scalar to Python for each GPU
    # query. Shapes are checked separately; no image-dependent Python loops.
    torch._assert_async(
        (torch.isfinite(actions) & (actions.abs() <= MAX_ACTION)).all(),
        "Actions must be finite with absolute log coordinate <= 4",
    )


def _costs(hypotheses, actions):
    residual = hypotheses[:, None] - actions[:, :, None]
    logs = torch.stack((residual[..., 0], torch.zeros_like(residual[..., 0]), residual[..., 1]), -1)
    ratio = (logs - logs.amax(-1, keepdim=True)).exp()
    red, green, blue = ratio.unbind(-1)
    squared_difference = (red - green).square() + (green - blue).square() + (blue - red).square()
    # This is exactly 1 - (sum ratio)^2 / (3 * sum ratio^2), evaluated without
    # subtracting nearly equal numbers. It is smooth and exactly zero at null.
    sin2 = squared_difference / (3 * ratio.square().sum(-1))
    # atan2(sqrt(sum pairwise differences^2), sum ratio) is the reproduction
    # angle. Only its norm is smoothed to permit finite second derivatives at
    # exact equality. Bias is <= degrees(ANGLE_NORM_EPS) because max ratio=1;
    # independent exact atan2 geometry must be used for measured GT metrics.
    angular = torch.atan2((squared_difference + ANGLE_NORM_EPS**2).sqrt(), ratio.sum(-1))
    return {"angular": angular * (180 / math.pi), "sin2": sin2}


def analytic_costs(hypotheses, actions):
    """Return angular degrees and exact sin² costs, each NxKxH.

    hypotheses: finite floating NxHx2; actions: finite floating NxKx2 with
    absolute coordinates <= 4. Compute in FP64 if either input is double,
    otherwise FP32, including under autocast. Positive illuminants are implied
    by exp([aR, 0, aB]); no ground-truth-specific API or parameter is needed.
    """
    if (
        hypotheses.ndim != 3 or hypotheses.shape[0] == 0 or hypotheses.shape[1] == 0
        or hypotheses.shape[2] != 2 or not hypotheses.is_floating_point()
    ):
        raise ValueError("Expected nonempty floating NxHx2 hypotheses")
    _action_structure(actions, hypotheses.shape[0])
    if actions.device != hypotheses.device:
        raise ValueError("Hypotheses and actions must be on the same device")
    _assert_action_domain(actions)
    torch._assert_async(torch.isfinite(hypotheses).all(), "Hypotheses must be finite")
    dtype = torch.float64 if torch.float64 in (hypotheses.dtype, actions.dtype) else torch.float32
    with torch.autocast(device_type=hypotheses.device.type, enabled=False):
        return _costs(hypotheses.to(dtype), actions.to(dtype))


def _illuminant(actions):
    log_rgb = torch.stack((actions[..., 0], torch.zeros_like(actions[..., 0]), actions[..., 1]), -1)
    rgb = (log_rgb - log_rgb.amax(-1, keepdim=True)).exp()
    return rgb / rgb.norm(dim=-1, keepdim=True)


def corrected_simplex(logchroma, action):
    """Physical diagonal correction followed by centered simplex RGB.

    Inputs have broadcast-compatible leading dimensions and final dimension2.
    For mean/RMS log chroma z and correction a, p=softmax([zR-aR,0,zB-aB]);
    return (3*pR-1,3*pB-1). Bounded output (-1,2) retains actual image color.
    Its nonlinear dependence on a is not an affine log-input reparameterization;
    this fact alone does not establish novelty or superior empirical accuracy.
    """
    corrected = logchroma - action
    logits = torch.stack((corrected[..., 0], torch.zeros_like(corrected[..., 0]), corrected[..., 1]), -1)
    return 3 * logits.softmax(-1)[..., [0, 2]] - 1


class CorrectionEvidenceNet(nn.Module):
    """One cached encoder and an action-conditioned 16-cell evidence router.

    Inputs must be nonempty floating NCHW RGB with both spatial dimensions >=32.
    Negative/nonfinite pixels, all-black rows, or globally absent channels mark
    the corresponding row invalid. Invalid rows use finite neutral diagnostics
    and are never accepted. Zero local cells use a declared finite color floor.

    Standard MobileNet BatchNorm is preserved. Training a single32x32 image is
    unsupported (one value/channel); use eval for deployment or a larger batch.
    Network arithmetic follows parameter dtype: FP32 by default, FP64 after
    net.double(). Input common scaling precedes FP32 conversion, preserving
    finite extreme FP64 exposures. Autocast is explicitly disabled here.
    """

    def __init__(self, mode="transport"):
        super().__init__()
        if mode not in {"transport", "action", "posterior", "direct"}:
            raise ValueError("mode must be transport, action, posterior or direct")
        self.mode = mode
        self.backbone = mobilenet_v3_large(weights=None).features
        self.context_head = nn.Sequential(nn.Linear(960, 64), nn.SiLU())
        self.point_head = nn.Linear(64, 2)
        self.local_project = nn.Linear(960, 48)
        self.vote_head = nn.Sequential(nn.Linear(118, 64), nn.SiLU(), nn.Linear(64, 2))
        self.router = nn.Sequential(nn.Linear(118, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 4), torch.linspace(-1, 1, 4), indexing="ij")
        xy = torch.stack((xx.flatten(), yy.flatten()), -1)
        self.register_buffer("coordinates", xy)
        self.register_buffer("proposal_prior", (xy * (0.06 / math.sqrt(2)) / 0.5).atanh())
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 5), torch.linspace(-1, 1, 5), indexing="ij")
        self.register_buffer("search_offsets", torch.stack((xx.flatten(), yy.flatten()), -1))
        nn.init.zeros_(self.point_head.weight)
        nn.init.zeros_(self.point_head.bias)
        nn.init.zeros_(self.vote_head[-1].weight)
        nn.init.zeros_(self.vote_head[-1].bias)

    def encode(self, image):
        if (
            image.ndim != 4 or image.shape[0] == 0 or image.shape[1] != 3
            or min(image.shape[-2:]) < 32 or not image.is_floating_point()
        ):
            raise ValueError("Expected nonempty floating NCHW RGB with H/W >= 32")
        if self.backbone.training and image.shape[0] == 1 and image.shape[-2:] == (32, 32):
            raise ValueError("A single 32x32 image requires eval() or a larger training batch for BatchNorm")
        dtype = self.point_head.weight.dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Model parameters must be FP32 or FP64")
        with torch.autocast(device_type=image.device.type, enabled=False):
            # Keep supplied FP64 until after common scaling. Promote narrower
            # inputs before moments; half squares would overflow/underflow.
            raw = image.to(torch.float64 if image.dtype == torch.float64 else torch.float32)
            valid = torch.isfinite(raw).all((1, 2, 3)) & (raw >= 0).all((1, 2, 3))
            clean = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
            scale = clean.amax((1, 2, 3), keepdim=True)
            valid = valid & (clean.amax((-2, -1)) > 0).all(-1)
            scaled = clean / torch.where(scale > 0, scale, torch.ones_like(scale))
            scaled = scaled.to(dtype)
            # Invalid images must neither generate NaNs nor leak a sanitized
            # partial image as a valid prediction. Acceptance uses original data.
            scaled = torch.where(valid[:, None, None, None], scaled, torch.ones_like(scaled))
            global_rms = scaled.square().mean((1, 2, 3), keepdim=True).clamp_min(COLOR_FLOOR**2).sqrt()
            features = self.backbone(scaled / global_rms)
            context = self.context_head(features.mean((-2, -1)))
            point = self.point_head(context).clamp(-2, 2)
            point = torch.where(valid[:, None], point, torch.zeros_like(point))
            local_features = self.local_project((features if features.shape[-2:] == (4, 4) else F.adaptive_avg_pool2d(features, (4, 4))).flatten(2).transpose(1, 2))
            means = F.adaptive_avg_pool2d(scaled, (4, 4)).flatten(2).transpose(1, 2)
            rms = F.adaptive_avg_pool2d(scaled.square(), (4, 4)).clamp_min(COLOR_FLOOR**2).sqrt().flatten(2).transpose(1, 2)
            mean_log = means.clamp_min(COLOR_FLOOR).log()
            rms_log = rms.clamp_min(COLOR_FLOOR).log()
            mean_logchroma = mean_log[..., [0, 2]] - mean_log[..., 1:2]
            rms_logchroma = rms_log[..., [0, 2]] - rms_log[..., 1:2]
            vote_input = torch.cat((
                local_features, context[:, None].expand(-1, 16, -1),
                self.coordinates[None].expand(image.shape[0], -1, -1), mean_logchroma, rms_logchroma,
            ), -1)
            local_actions = point[:, None] + 0.5 * (self.vote_head(vote_input) + self.proposal_prior[None]).tanh()
            return {
                "point_action": point, "local_actions": local_actions,
                "local_features": local_features, "context": context,
                "mean_logchroma": mean_logchroma, "rms_logchroma": rms_logchroma,
                "valid": valid,
            }

    def query(self, cache, actions):
        """Predict action risks using cached image color and local hypotheses.

        Transport physically corrects pooled mean/RMS colors by the queried
        action, normalizes them onto the simplex, and gives the router action-
        point. Action mode holds simplex colors at the point, retaining relative
        action. Posterior additionally fixes relative action to zero. None has
        label access. Parameters/hypotheses remain attached for mixed derivatives.
        """
        point = cache["point_action"]
        _action_structure(actions, point.shape[0])
        _assert_action_domain(actions)
        actions = actions.to(device=point.device, dtype=point.dtype)
        with torch.autocast(device_type=point.device.type, enabled=False):
            return self._query(cache, actions)

    def _query(self, cache, actions):
        point = cache["point_action"]
        queries = actions.shape[1]
        if self.mode == "transport":
            corrected_mean = corrected_simplex(cache["mean_logchroma"][:, None], actions[:, :, None])
            corrected_rms = corrected_simplex(cache["rms_logchroma"][:, None], actions[:, :, None])
        else:
            corrected_mean = corrected_simplex(cache["mean_logchroma"], point[:, None])[:, None].expand(-1, queries, -1, -1)
            corrected_rms = corrected_simplex(cache["rms_logchroma"], point[:, None])[:, None].expand(-1, queries, -1, -1)
        relative = actions - point[:, None] if self.mode in {"transport", "action"} else torch.zeros_like(actions)
        router_input = torch.cat((
            cache["local_features"][:, None].expand(-1, queries, -1, -1),
            cache["context"][:, None, None].expand(-1, queries, 16, -1),
            corrected_mean, corrected_rms, relative[:, :, None].expand(-1, -1, 16, -1),
        ), -1)
        weights = self.router(router_input).squeeze(-1).softmax(-1)
        costs = _costs(cache["local_actions"], actions)
        angular = (weights * costs["angular"]).sum(-1)
        sin2 = (weights * costs["sin2"]).sum(-1)
        valid = cache["valid"][:, None]
        # Conservative diagnostic conventions for refused inputs. These finite
        # values are not evidence that the underlying image has a valid GT cost.
        angular = torch.where(valid, angular, torch.full_like(angular, 90))
        sin2 = torch.where(valid, sin2, torch.full_like(sin2, 2 / 3))
        return {"angular_risk": angular, "sin2_risk": sin2, "weights": weights}

    def select(self, cache, steps=2):
        """Deterministic cached 5x5 search with 25/51/103 total candidates.

        Later stages include both the preceding decision (grid center) and the
        original point (extra candidate). Predicted risk cannot increase across
        stages; this does not promise lower true error. Direct uses one point.
        """
        if steps not in (1, 2, 4):
            raise ValueError("steps must be 1, 2 or 4")
        point = cache["point_action"]
        with torch.autocast(device_type=point.device.type, enabled=False):
            rows = torch.arange(point.shape[0], device=point.device)
            current = point
            trajectory_actions, trajectory_risk = [], []
            if self.mode == "direct":
                risk = self._query(cache, point[:, None])["angular_risk"][:, 0]
                trajectory_actions.append(point)
                trajectory_risk.append(risk)
                query_count = 1
            else:
                query_count = 0
                for stage, radius in enumerate((0.24, 0.06, 0.03, 0.015)[:steps]):
                    candidates = (current[:, None] + radius * self.search_offsets[None]).clamp(-2, 2)
                    if stage:
                        candidates = torch.cat((candidates, point[:, None]), 1)
                    risks = self._query(cache, candidates)["angular_risk"]
                    chosen = risks.argmin(-1)
                    current = candidates[rows, chosen]
                    risk = risks[rows, chosen]
                    trajectory_actions.append(current)
                    trajectory_risk.append(risk)
                    query_count += candidates.shape[1]
            trajectory_actions = torch.stack(trajectory_actions, 1)
            trajectory_risk = torch.stack(trajectory_risk, 1)
            valid = cache["valid"]
            # Invalid ties must not invent a non-neutral result at a grid edge.
            trajectory_actions = torch.where(valid[:, None, None], trajectory_actions, torch.zeros_like(trajectory_actions))
            action = trajectory_actions[:, -1]
            return {
                "pred": _illuminant(action), "action": action, "risk": trajectory_risk[:, -1],
                "base_pred": _illuminant(point), "valid": valid,
                "query_count": query_count, "trajectory_actions": trajectory_actions,
                "trajectory_risk": trajectory_risk,
            }

    def forward(self, image):
        cache = self.encode(image)
        return {**self.select(cache), "context": cache["context"]}
