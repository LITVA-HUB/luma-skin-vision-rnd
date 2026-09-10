"""Experimental color-frame graph posterior; constructed correctness is not accuracy evidence.

Frames are expressed after a positive per-image common scale. Projective GL(3)
equivariance only holds where the selected triple and full-rank validity stay
unchanged. Floating-point ties, rank refusal, nonpositive outputs and diagnostic
fallbacks are outside that claim. No pretrained components or camera metadata.
"""

import itertools
import math

import torch
from torch import nn
from torch.nn import functional as F


def vmf_log_normalizer(concentration):
    """log(k / (4*pi*sinh(k))) on S², stable at zero and large k."""
    k = concentration
    safe = k.clamp_min(1e-4)
    regular = safe.log() - math.log(2 * math.pi) - safe - torch.log(-torch.expm1(-2 * safe))
    small = k.clamp_max(1e-3)
    series = -math.log(4 * math.pi) - small.square() / 6 + small.pow(4) / 180
    return torch.where(k < 1e-3, series, regular)


def canonical_target(frame, gt):
    """Normalize B^-1 GT with an FP64 solve. Singular frames are caller errors.

    Use output['frame_valid'] before calling for external targets. The likelihood
    helper does this masking itself, and rejects nonfinite/nonpositive GT.
    """
    if frame.shape != (gt.shape[0], 3, 3) or gt.shape[-1] != 3:
        raise ValueError("Expected frame Nx3x3 and gt Nx3")
    if not torch.isfinite(frame).all() or not torch.isfinite(gt).all() or not (gt > 0).all():
        raise ValueError("canonical_target requires finite frame and positive finite GT")
    # Both common scales are projectively immaterial. Remove them before the
    # solve, then remove solved-vector magnitude before its Euclidean norm.
    # This avoids absolute normalize eps changing tiny GT and norm overflow for
    # huge GT, including opposing finite GT/frame scales such as1e200/1e-200.
    frame64, gt64 = frame.double(), gt.double()
    frame_scale = frame64.abs().amax((-2, -1), keepdim=True)
    if not (frame_scale > 0).all():
        raise ValueError("canonical_target requires a nonsingular frame")
    frame64 = frame64 / frame_scale
    gt64 = gt64 / gt64.amax(-1, keepdim=True)
    solved = torch.linalg.solve(frame64, gt64.unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(solved).all():
        raise ValueError("canonical_target solve is nonfinite; frame is numerically unsupported")
    solved = solved / solved.abs().amax(-1, keepdim=True)
    return solved / solved.norm(dim=-1, keepdim=True)


def posterior_nll(output, gt, reduction="mean"):
    """Canonical directional NLL; mean over valid frames/GT, never silent zero.

    Point positivity does not mask a valid-frame likelihood. reduction='none'
    returns N rows, NaN for excluded rows; 'mean' raises if all are excluded.
    This is a density in canonical solid angle, not camera-space solid angle.
    """
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be mean or none")
    mask = output["frame_valid"] & torch.isfinite(gt).all(-1) & (gt > 0).all(-1)
    if not mask.any():
        if reduction == "none":
            return output["logits"].new_full((gt.shape[0],), float("nan"))
        raise ValueError("posterior_nll has no valid frame/GT rows")
    q = canonical_target(output["frame"][mask], gt[mask]).to(output["directions"].dtype)
    k = output["concentration"][mask]
    directions = output["directions"][mask]
    dot = (directions * q[:, None]).sum(-1)
    small_k = k.clamp_max(1.)
    small_density = vmf_log_normalizer(small_k) + small_k * dot
    safe_k = k.clamp_min(1.)
    # Combine normalizer and exponent before evaluation to avoid subtracting k
    # from k at a concentrated mode. For unit directions, dot-1=-||mu-q||²/2;
    # squared chord length also preserves tiny angular offsets lost by dot-1.
    chord_squared = (directions - q[:, None]).square().sum(-1)
    regular_density = safe_k.log() - math.log(2 * math.pi)
    regular_density = regular_density - torch.log(-torch.expm1(-2 * safe_k))
    regular_density = regular_density - (0.5 * safe_k) * chord_squared
    log_density = torch.where(k >= 1., regular_density, small_density)
    components = log_density + output["logits"][mask].log_softmax(-1)
    nll = -torch.logsumexp(components, dim=-1)
    if reduction == "mean":
        return nll.mean()
    result = nll.new_full((gt.shape[0],), float("nan"))
    return result.masked_scatter(mask, nll)


def camera_posterior_nll(output, gt, reduction="mean"):
    """Pushforward density NLL on camera S², using J=abs(det B)/||Bq||³.

    Canonical NLLs from different frames are not comparable densities. This
    correction has no learned-parameter gradient because B and GT are inputs.
    Real reproduction errors remain the architecture-selection metric.
    """
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be mean or none")
    canonical = posterior_nll(output, gt, reduction="none")
    mask = torch.isfinite(canonical)
    if not mask.any():
        if reduction == "none":
            return canonical
        raise ValueError("camera_posterior_nll has no valid frame/GT rows")
    frame = output["frame"][mask].double()
    # The common frame magnitude cancels exactly from the spherical Jacobian;
    # remove it before computing ||Bq|| to avoid finite-scale norm overflow.
    frame = frame / frame.abs().amax((-2, -1), keepdim=True)
    q = canonical_target(frame, gt[mask])
    length = (frame @ q.unsqueeze(-1)).squeeze(-1).norm(dim=-1)
    log_jacobian = torch.linalg.slogdet(frame).logabsdet - 3 * length.log()
    values = canonical[mask] + log_jacobian.to(canonical.dtype)
    if reduction == "mean":
        return values.mean()
    return canonical.masked_scatter(mask, values)


def _vmf_quadrature(directions, concentration):
    """32 equal-weight nodes: four midpoint CDF quantiles x eight azimuths.

    For vMF about mu, z=cos(theta) has CDF inverse
    1 + log(u+(1-u)*exp(-2*k))/k. A fixed least-aligned coordinate
    axis constructs the tangent basis. This low-order quadrature is approximate
    and can have orientation/validity-boundary artifacts, especially at low k.
    """
    dtype, device = directions.dtype, directions.device
    u = (torch.arange(4, dtype=dtype, device=device) + 0.5) / 4
    phi = torch.arange(8, dtype=dtype, device=device) * (2 * math.pi / 8)
    k = concentration[..., None].clamp_min(1e-5)
    z = 1 + torch.log(u + (1 - u) * torch.exp(-2 * k)) / k
    z = torch.where(concentration[..., None] < 1e-4, 2 * u - 1, z)
    axis = F.one_hot(directions.abs().argmin(-1), num_classes=3).to(dtype)
    tangent = F.normalize(torch.linalg.cross(directions, axis, dim=-1), dim=-1)
    second = torch.linalg.cross(directions, tangent, dim=-1)
    radial = phi.cos()[None, None, :, None] * tangent[:, :, None]
    radial = radial + phi.sin()[None, None, :, None] * second[:, :, None]
    radius = (1 - z.square()).clamp_min(1e-12).sqrt()
    nodes = z[..., None, None] * directions[:, :, None, None]
    nodes = nodes + radius[..., None, None] * radial[:, :, None]
    return nodes.flatten(2, 3)


def transported_risk(frame, pred, directions, logits, concentration, frame_valid=None):
    """Camera reproduction-angle spread proxy in degrees, preserving invalid mass.

    Each of K*32 hypotheses is transported by B. Nonpositive or nonfinite mapped
    hypotheses receive cost90 degrees (a diagnostic convention, not a physical
    expectation). No renormalization of the remaining posterior mass is applied.
    """
    dtype = directions.dtype
    nodes = _vmf_quadrature(directions, concentration)
    mapped = torch.einsum("nij,nkqj->nkqi", frame.to(dtype), nodes)
    valid = torch.isfinite(mapped).all(-1) & (mapped > 0).all(-1)
    if frame_valid is not None:
        valid = valid & frame_valid[:, None, None]
    safe = torch.where(valid[..., None], mapped, torch.ones_like(mapped))
    ratio = safe / pred[:, None, None].clamp_min(torch.finfo(dtype).tiny)
    ratio = F.normalize(ratio, dim=-1)
    neutral = torch.ones_like(ratio) / math.sqrt(3)
    sine = torch.linalg.cross(ratio, neutral, dim=-1).norm(dim=-1)
    cosine = (ratio * neutral).sum(-1)
    degrees = torch.atan2(sine, cosine) * (180 / math.pi)
    cost = torch.where(valid, degrees, torch.full_like(degrees, 90))
    weights = logits.softmax(-1)
    risk = (cost.mean(-1) * weights).sum(-1)
    mass = ((~valid).to(dtype).mean(-1) * weights).sum(-1)
    if frame_valid is not None:
        # Explicit refusal values avoid a softmax-roundoff mass such as0.99999994.
        risk = torch.where(frame_valid, risk, torch.full_like(risk, 90))
        mass = torch.where(frame_valid, mass, torch.ones_like(mass))
    return risk, mass


class EdgeDiffusionBlock(nn.Module):
    """Directed eight-neighbor messages gated by embeddings and token differences."""

    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.gate = nn.Sequential(nn.Linear(2 * width + 11, width // 2), nn.SiLU(), nn.Linear(width // 2, 1), nn.Sigmoid())
        self.value = nn.Linear(width + 11, width)
        self.update = nn.Sequential(nn.Linear(2 * width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))

    def forward(self, h, features, source, target, degree):
        z = self.norm(h)
        delta = features[:, source] - features[:, target]
        gate = self.gate(torch.cat([z[:, source], z[:, target], delta], dim=-1))
        value = self.value(torch.cat([z[:, source] - z[:, target], delta], dim=-1))
        aggregate = torch.zeros_like(h).index_add(1, target, gate * value)
        aggregate = aggregate / degree[None, :, None]
        return h + self.update(torch.cat([z, aggregate], dim=-1))


class ColorFramePosteriorNet(nn.Module):
    def __init__(self, mode="frame", width=192, layers=4, hypotheses=8):
        super().__init__()
        if mode not in {"direct", "diagonal", "frame"}:
            raise ValueError("mode must be direct, diagonal or frame")
        if width < 2 or layers < 1 or hypotheses < 1:
            raise ValueError("width>=2, layers>=1 and hypotheses>=1 required")
        self.mode = mode
        self.hypotheses = hypotheses
        self.condition_limit = 1e6
        self.register_buffer("triples", torch.tensor(list(itertools.combinations(range(16), 3))))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 8), torch.linspace(-1, 1, 8), indexing="ij")
        self.register_buffer("coordinates", torch.stack([xx.flatten(), yy.flatten()], dim=-1))
        edges = [(a, b) for a in range(64) for b in range(64) if a != b and max(abs(a // 8 - b // 8), abs(a % 8 - b % 8)) == 1]
        self.register_buffer("source", torch.tensor([a for a, _ in edges]))
        self.register_buffer("target", torch.tensor([b for _, b in edges]))
        self.register_buffer("degree", torch.bincount(self.target, minlength=64).float())
        self.embedding = nn.Linear(11, width)
        self.blocks = nn.ModuleList([EdgeDiffusionBlock(width) for _ in range(layers)])
        self.context_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.SiLU())
        self.point_head = nn.Linear(64, 3)
        self.direction_head = nn.Linear(64, 3 * hypotheses)
        self.logit_head = nn.Linear(64, hypotheses)
        self.concentration_head = nn.Linear(64, hypotheses)
        nn.init.zeros_(self.point_head.weight)
        nn.init.zeros_(self.point_head.bias)

    def _frame_and_pixels(self, x):
        # Float64 also prevents overflow when scaling large finite FP32 pixels.
        raw = x.double()
        input_valid = torch.isfinite(raw).all((1, 2, 3)) & (raw >= 0).all((1, 2, 3))
        clean = torch.nan_to_num(raw, nan=0., posinf=0., neginf=0.).clamp_min(0)
        scale = clean.amax((1, 2, 3))
        input_valid = input_valid & (scale > 0)
        scaled = clean / torch.where(scale > 0, scale, torch.ones_like(scale))[:, None, None, None]
        mean = scaled.mean((-2, -1))
        fallback = F.normalize(mean.clamp_min(1e-8), dim=-1)
        identity = torch.eye(3, dtype=raw.dtype, device=x.device).expand(x.shape[0], -1, -1)
        indices = torch.full((x.shape[0], 3), -1, dtype=torch.long, device=x.device)
        margin = torch.zeros_like(scale)
        if self.mode == "frame":
            patches = F.adaptive_avg_pool2d(scaled, (4, 4)).flatten(2).transpose(1, 2)
            candidates = patches[:, self.triples].transpose(-1, -2)
            volumes = torch.linalg.det(candidates).abs()
            chosen = volumes.argmax(-1)  # combinations lexicographic; first exact tie
            indices = self.triples[chosen]
            frame = candidates[torch.arange(x.shape[0], device=x.device), chosen]
            top = volumes.topk(2, dim=-1).values
            margin = (top[:, 0] - top[:, 1]) / top[:, 0].clamp_min(1e-30)
        elif self.mode == "diagonal":
            frame = torch.diag_embed(mean)
        else:
            frame = identity
        singular = torch.linalg.svdvals(frame)
        condition = (singular[:, 0] / singular[:, -1].clamp_min(1e-30)).clamp_max(1e30)
        frame_valid = input_valid & (singular[:, -1] > singular[:, 0] / self.condition_limit) & (singular[:, 0] > 0)
        safe_frame = torch.where(frame_valid[:, None, None], frame, identity)
        canonical = torch.linalg.solve(safe_frame, scaled.flatten(2)).reshape_as(scaled)
        return canonical, fallback, safe_frame, {
            "input_valid": input_valid, "frame_valid": frame_valid,
            "frame_singular_values": singular, "frame_condition": condition,
            "frame_indices": indices, "frame_volume_margin": margin, "input_scale": scale,
        }

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or not x.is_floating_point() or min(x.shape[-2:]) < 8 or x.shape[0] == 0:
            raise ValueError("Expected nonempty floating NCHW RGB, H/W >= 8")
        canonical, fallback, frame, diagnostics = self._frame_and_pixels(x)
        dtype = self.embedding.weight.dtype
        # Population standard deviation; floor avoids unstable sqrt derivative at zero.
        means = F.adaptive_avg_pool2d(canonical, (8, 8))
        variance = F.adaptive_avg_pool2d(canonical.square(), (8, 8)) - means.square()
        std = variance.clamp_min(1e-24).sqrt()
        asinh = F.adaptive_avg_pool2d(canonical.asinh(), (8, 8))
        stats = torch.cat([means, std, asinh], dim=1).flatten(2).transpose(1, 2).to(dtype)
        coordinates = self.coordinates.to(dtype).expand(x.shape[0], -1, -1)
        features = torch.cat([stats, coordinates], dim=-1)
        h = self.embedding(features)
        for block in self.blocks:
            h = block(h, features, self.source, self.target, self.degree.to(dtype))
        context = self.context_head(h.mean(1))
        canonical_mean = canonical.mean((-2, -1)).to(dtype)
        anchor = F.normalize(canonical_mean, dim=-1)
        point = F.normalize(anchor + self.point_head(context), dim=-1)
        raw_pred = torch.einsum("nij,nj->ni", frame.to(dtype), point)
        point_valid = torch.isfinite(raw_pred).all(-1) & (raw_pred > 0).all(-1)
        valid = diagnostics["frame_valid"] & point_valid
        pred = torch.where(valid[:, None], F.normalize(raw_pred, dim=-1), fallback.to(dtype))
        direction_raw = anchor[:, None] + self.direction_head(context).reshape(-1, self.hypotheses, 3)
        directions = F.normalize(direction_raw, dim=-1)
        # Exact zero direction has a deterministic canonical axis; no GL claim is
        # attached to positivity fallback. Avoid an undefined sphere density.
        axis = torch.zeros_like(directions)
        axis[..., 0] = 1
        directions = torch.where(direction_raw.norm(dim=-1, keepdim=True) > 1e-12, directions, axis)
        logits = self.logit_head(context)
        concentration = F.softplus(self.concentration_head(context)) + 1e-4
        risk, mass = transported_risk(frame, pred, directions, logits, concentration, diagnostics["frame_valid"])
        return {
            "pred": pred, "raw_pred": raw_pred, "context": context, "valid": valid,
            "point_valid": point_valid, "frame": frame, "canonical_mean": canonical_mean,
            "directions": directions, "logits": logits, "concentration": concentration,
            "transport_risk": risk, "invalid_posterior_mass": mass, **diagnostics,
        }
