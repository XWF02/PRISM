"""PRISM optimization following Eqs. (4)-(5), (17)-(19), and Algorithm 1."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from latent_domain import characterize_latent_domains


def _make_loader(x, y, batch_size: int, shuffle: bool):
    indices = torch.arange(len(y), dtype=torch.long)
    dataset = TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long).view(-1),
        indices,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, drop_last=False)


@torch.no_grad()
def _extract_all(model, ordered_loader, device):
    model.eval()
    features, class_logits, original_indices = [], [], []
    for xb, _yb, idx in ordered_loader:
        cls_logits, _joint_logits, _domain_logits, z = model(xb.to(device))
        features.append(z.detach().cpu())
        class_logits.append(cls_logits.detach().cpu())
        original_indices.append(idx.cpu())
    return (torch.cat(features, dim=0), torch.cat(class_logits, dim=0),
            torch.cat(original_indices, dim=0))


def _run_fine_grained_epoch(model, loader, pseudo_labels, optimizer, scheduler,
                            device, epoch_number, cycle_number):
    """One pass minimizing L_dc only, as the first stage of Algorithm 1."""
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    for xb, yb, idx in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        # Pseudo labels are stored on CPU in original sample order. Move the
        # indexed batch to the model device before combining it with y_b.
        domain_batch = pseudo_labels[idx].to(device)
        joint_target = domain_batch * model.num_classes + yb

        emotion_logits, joint_logits, _domain_logits, _z = model(xb)
        loss_dc = F.cross_entropy(joint_logits, joint_target)
        optimizer.zero_grad(set_to_none=True)
        loss_dc.backward()
        optimizer.step()

        total_loss += loss_dc.item() * len(yb)
        correct += (emotion_logits.argmax(dim=1) == yb).sum().item()
        count += len(yb)

    scheduler.step()
    return {
        "epoch": epoch_number,
        "outer_cycle": cycle_number,
        "stage": "fine_grained_feature_update",
        "loss": total_loss / max(count, 1),
        "train_accuracy": correct / max(count, 1),
    }


def _run_domain_invariant_epoch(model, loader, pseudo_labels, optimizer,
                                scheduler, device, lambda1, lambda2,
                                epoch_number, cycle_number):
    """One pass minimizing Eq. (19), with GRL only on the L_dom branch."""
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    for xb, yb, idx in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        domain_batch = pseudo_labels[idx].to(device)
        joint_target = domain_batch * model.num_classes + yb

        emotion_logits, joint_logits, domain_logits, _z = model(xb, grl_scale=1.0)
        loss_cls = F.cross_entropy(emotion_logits, yb)
        loss_dc = F.cross_entropy(joint_logits, joint_target)
        loss_dom = F.cross_entropy(domain_logits, domain_batch)
        loss = loss_cls + lambda1 * loss_dc + lambda2 * loss_dom

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(yb)
        correct += (emotion_logits.argmax(dim=1) == yb).sum().item()
        count += len(yb)

    scheduler.step()
    return {
        "epoch": epoch_number,
        "outer_cycle": cycle_number,
        "stage": "domain_invariant_representation_learning",
        "loss": total_loss / max(count, 1),
        "train_accuracy": correct / max(count, 1),
    }


def train_prism(
    model,
    x_train,
    y_train,
    *,
    device="cpu",
    epochs: int = 80,
    batch_size: int = 96,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-3,
    lambda1: float = 0.1,
    lambda2: float = 0.05,
    tau: float = 1.0,
    update_every: int = 8,
    soft_updates: int = 3,
    hard_updates: int = 3,
    seed: int = 42,
    verbose: bool = False,
):
    """Train PRISM for one fixed K using source-training data only.

    The manuscript specifies 80 epochs, with pseudo-domain labels refreshed
    every 8 epochs (T=10 cycles). To preserve the stated total of 80 epochs
    while following Algorithm 1's stage order, each 8-epoch cycle consists of:
      (1) one epoch of fine-grained L_dc update;
      (2) latent-domain characterization;
      (3) seven epochs of the full objective in Eq. (19).
    Thus label refreshes are eight epochs apart, and the history has exactly
    80 optimizer epochs. Pseudo-labels are indexed by stable sample IDs.
    """
    x = torch.as_tensor(x_train, dtype=torch.float32)
    y = torch.as_tensor(y_train, dtype=torch.long).view(-1)
    if x.ndim != 2 or len(x) != len(y) or len(y) == 0:
        raise ValueError("x_train must be non-empty [N,D] and y_train must be [N]")
    if epochs <= 0 or update_every < 2 or epochs % update_every != 0:
        raise ValueError("Require positive epochs, update_every >= 2, and epochs divisible by update_every")
    if y.min().item() < 0 or y.max().item() >= model.num_classes:
        raise ValueError("Training labels must be zero-based in [0, C-1]")

    train_loader = _make_loader(x, y, batch_size, shuffle=True)
    ordered_loader = _make_loader(x, y, batch_size, shuffle=False)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Eq. (4): d_i^(0)=1 in the paper; stored as zero-based index 0 here.
    pseudo_labels = torch.zeros(len(y), dtype=torch.long)
    history = []
    cycles = epochs // update_every
    epoch_number = 0

    for cycle in range(cycles):
        # Stage 1: Fine-grained feature update using the previous pseudo labels.
        epoch_number += 1
        item = _run_fine_grained_epoch(
            model, train_loader, pseudo_labels, optimizer, scheduler,
            device, epoch_number, cycle + 1,
        )
        history.append(item)
        if verbose:
            print(f"epoch {epoch_number:03d}/{epochs} | {item['stage']} | "
                  f"loss={item['loss']:.4f} | acc={item['train_accuracy']:.4f}", flush=True)

        # Stage 2: characterize domains from source-training samples only.
        z, class_logits, original_indices = _extract_all(model, ordered_loader, device)
        updated_labels, _centers = characterize_latent_domains(
            z, class_logits, model.num_domains, tau=tau,
            soft_updates=soft_updates, hard_updates=hard_updates,
            seed=seed + cycle,
        )
        pseudo_labels[original_indices] = updated_labels.cpu()

        # Stage 3: seven epochs in the default 8-epoch cycle, using fixed labels.
        for _ in range(update_every - 1):
            epoch_number += 1
            item = _run_domain_invariant_epoch(
                model, train_loader, pseudo_labels, optimizer, scheduler,
                device, lambda1, lambda2, epoch_number, cycle + 1,
            )
            history.append(item)
            if verbose:
                print(f"epoch {epoch_number:03d}/{epochs} | {item['stage']} | "
                      f"loss={item['loss']:.4f} | acc={item['train_accuracy']:.4f}", flush=True)

    if epoch_number != epochs:
        raise RuntimeError(f"Internal epoch accounting error: {epoch_number} != {epochs}")
    return model, history


@torch.no_grad()
def evaluate(model, x, y, device="cpu", batch_size=96):
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long).view(-1)
    if len(x) == 0 or len(x) != len(y):
        raise ValueError("Evaluation data must be non-empty and have matching x/y lengths")
    loader = _make_loader(x, y, batch_size, shuffle=False)
    model.to(device)
    model.eval()
    predictions, truths = [], []
    for xb, yb, _idx in loader:
        logits = model.predict(xb.to(device))
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        truths.append(yb.numpy())
    predictions = np.concatenate(predictions)
    truths = np.concatenate(truths)
    accuracy = float((predictions == truths).mean() * 100.0)
    return accuracy, predictions, truths
