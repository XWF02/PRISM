"""Entropy-guided latent-domain construction (PRISM Eqs. 8-16)."""
from __future__ import annotations

import math
import torch

_EPS = 1e-8


def squared_euclidean(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distances, shape [N, K]."""
    return (
        x.square().sum(dim=1, keepdim=True)
        + centers.square().sum(dim=1).unsqueeze(0)
        - 2.0 * (x @ centers.t())
    ).clamp_min(0.0)


@torch.no_grad()
def entropy_weighted_kmeanspp(
    z: torch.Tensor,
    entropy: torch.Tensor,
    n_centers: int,
    seed: int = 42,
) -> torch.Tensor:
    """Entropy-weighted K-means++ initialization.

    The manuscript names entropy-weighted K-means++ but does not give its
    sampling probability explicitly. This implementation uses p_1 proportional
    to entropy and subsequent p_m proportional to entropy times the minimum
    squared distance to an already selected center, the standard K-means++
    distance-weighting pattern with entropy weighting.
    """
    if n_centers <= 0:
        return z.new_empty((0, z.shape[1]))
    if z.shape[0] < n_centers:
        raise ValueError("Need at least as many source samples as extra centroids")

    base_weights = entropy.detach().float().clamp_min(0).cpu()
    if base_weights.sum().item() <= _EPS:
        base_weights = torch.ones_like(base_weights)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    min_dist = torch.full((z.shape[0],), float("inf"), dtype=torch.float32, device=z.device)
    centers = []

    for center_idx in range(n_centers):
        weights = base_weights if center_idx == 0 else base_weights * min_dist.detach().cpu()
        if not torch.isfinite(weights).all() or weights.sum().item() <= _EPS:
            weights = base_weights.clone()
        if weights.sum().item() <= _EPS:
            weights = torch.ones_like(weights)
        selected = int(torch.multinomial(weights / weights.sum(), 1, generator=generator).item())
        center = z[selected].detach().clone()
        centers.append(center)
        dist = (z - center.unsqueeze(0)).square().sum(dim=1)
        min_dist = torch.minimum(min_dist, dist)

    return torch.stack(centers, dim=0)


@torch.no_grad()
def characterize_latent_domains(
    z: torch.Tensor,
    class_logits: torch.Tensor,
    num_domains: int,
    tau: float = 1.0,
    soft_updates: int = 3,
    hard_updates: int = 3,
    seed: int = 42,
):
    """Compute class-guided and entropy-guided centroids and hard pseudo-labels.

    z: bottleneck features [N,d]; class_logits: emotion logits [N,C].
    Returns zero-based pseudo-domain labels [N] and final centroids [K,d].
    """
    z = z.detach().float()
    probabilities = class_logits.detach().float().softmax(dim=1)
    n_samples, _ = z.shape
    n_classes = probabilities.shape[1]
    if n_samples == 0 or num_domains < n_classes or tau <= 0:
        raise ValueError("Require non-empty samples, K >= C, and tau > 0")
    if soft_updates < 0 or hard_updates < 0:
        raise ValueError("Centroid update counts cannot be negative")

    # Eq. (8): class-probability-weighted centroids for the first C domains.
    class_mass = probabilities.sum(dim=0).unsqueeze(1)
    class_centers = (probabilities.t() @ z) / (class_mass + _EPS)

    if num_domains > n_classes:
        # Eq. (9): normalized predictive entropy.
        entropy = -(
            probabilities * probabilities.clamp_min(_EPS).log()
        ).sum(dim=1) / math.log(n_classes)
        n_extra = num_domains - n_classes
        extra_centers = entropy_weighted_kmeanspp(z, entropy, n_extra, seed=seed)

        # Eqs. (10)-(11): entropy-gated soft assignment and R centroid updates.
        for _ in range(soft_updates):
            distance = squared_euclidean(z, extra_centers)
            assignment = torch.softmax(-distance / tau, dim=1)
            # The factor M appears in Eq. (10). It cancels in each weighted
            # centroid ratio but is retained here to mirror the manuscript.
            delta = entropy.unsqueeze(1) * n_extra * assignment
            mass = delta.sum(dim=0).unsqueeze(1)
            updated = (delta.t() @ z) / (mass + _EPS)
            extra_centers = torch.where(mass > _EPS, updated, extra_centers)

        # Eq. (12): concatenate class-guided and refined extra centroids.
        centers = torch.cat([class_centers, extra_centers], dim=0)
    else:
        centers = class_centers

    # Eq. (13): initial nearest-centroid assignment.
    labels = squared_euclidean(z, centers).argmin(dim=1)

    # Eqs. (14)-(16): Q hard assignment / centroid refinement iterations.
    for _ in range(hard_updates):
        counts = torch.bincount(labels, minlength=num_domains).to(z.dtype).unsqueeze(1)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, z)
        updated_centers = sums / counts.clamp_min(1.0)
        centers = torch.where(counts > 0, updated_centers, centers)
        labels = squared_euclidean(z, centers).argmin(dim=1)

    return labels, centers
