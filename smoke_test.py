"""CPU smoke tests for PRISM modules; synthetic data only, not EEG reproduction."""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiment_protocol import (
    load_dataset_npz, materialize_subject_samples, run_outer_fold,
    split_source_subjects,
)
from latent_domain import characterize_latent_domains
from preprocess_de import mde_sequence, seed_seediv_window_mde
from prism_model import PRISM
from training import evaluate, train_prism


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(7)

    # MDE formula and overlapping-window boundary: DE[5]-DE[4] must be kept.
    de = rng.normal(size=(15, 62, 5)).astype(np.float32)
    mde = mde_sequence(de)
    assert mde.shape == (15, 62, 10)
    np.testing.assert_allclose(mde[0, :, 5:], 0.0, atol=1e-7)
    np.testing.assert_allclose(mde[5, :, 5:], de[5] - de[4], atol=1e-6)
    windows = seed_seediv_window_mde(de, window=10, stride=5)
    assert windows.shape == (2, 620)
    # The first epoch in window 2 retains its within-trial predecessor diff.
    second_window = mde[5:15].mean(axis=0).reshape(-1)
    np.testing.assert_allclose(windows[1], second_window, atol=1e-6)

    # Dataset-specific input dimensions and 256-D bottleneck.
    seed_model = PRISM(channels=62, num_classes=3, num_domains=3)
    faced_model = PRISM(channels=32, num_classes=9, num_domains=9)
    with torch.no_grad():
        out_seed = seed_model(torch.randn(3, 620))
        out_faced = faced_model(torch.randn(2, 320))
    assert out_seed[0].shape == (3, 3) and out_seed[3].shape == (3, 256)
    assert out_faced[0].shape == (2, 9) and out_faced[3].shape == (2, 256)

    # Entropy-guided domain construction.
    z = torch.randn(48, 256)
    class_logits = torch.randn(48, 3)
    pseudo, centers = characterize_latent_domains(
        z, class_logits, num_domains=5, tau=1.0,
        soft_updates=3, hard_updates=3, seed=11,
    )
    assert pseudo.shape == (48,) and centers.shape == (5, 256)
    assert int(pseudo.min()) >= 0 and int(pseudo.max()) < 5

    # Full small training run: 8 epochs = one refinement cycle, 1+7 stages.
    x_train = rng.normal(size=(48, 20)).astype(np.float32)
    y_train = np.tile(np.arange(3, dtype=np.int64), 16)
    small_model = PRISM(channels=2, num_classes=3, num_domains=4, dropout=0.1)
    small_model, history = train_prism(
        small_model, x_train, y_train, device="cpu", epochs=8,
        batch_size=12, update_every=8, soft_updates=3, hard_updates=3,
        seed=5, verbose=False,
    )
    assert len(history) == 8
    assert history[0]["stage"] == "fine_grained_feature_update"
    assert all(item["stage"] == "domain_invariant_representation_learning" for item in history[1:])
    accuracy, predictions, truths = evaluate(small_model, x_train, y_train, device="cpu")
    assert 0.0 <= accuracy <= 100.0 and predictions.shape == truths.shape == (48,)

    # Trial-packed data: split by subject first, then create 10-step windows.
    trial_de = rng.normal(size=(4 * 15, 62, 5)).astype(np.float32)
    offsets = np.arange(0, 4 * 15 + 1, 15, dtype=np.int64)
    trial_y = np.array([0, 1, 2, 0], dtype=np.int64)
    trial_subject = np.array(["s1", "s1", "s2", "s2"])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tiny_seed_trials.npz"
        np.savez_compressed(path, de=trial_de, offsets=offsets,
                            y=trial_y, subject=trial_subject)
        data = load_dataset_npz(path, "seed")
        selected = materialize_subject_samples(data, ["s1"], "seed")
        assert selected["x"].shape == (4, 620)
        assert set(selected["subject"].tolist()) == {"s1"}

        faced_de = rng.normal(size=(2 * 3, 32, 5)).astype(np.float32)
        faced_offsets = np.array([0, 3, 6], dtype=np.int64)
        faced_path = Path(tmp) / "tiny_faced_trials.npz"
        np.savez_compressed(
            faced_path, de=faced_de, offsets=faced_offsets,
            y=np.array([0, 1]), subject=np.array(["f1", "f2"]),
        )
        faced_data = load_dataset_npz(faced_path, "faced")
        faced_samples = materialize_subject_samples(faced_data, ["f1"], "faced")
        assert faced_samples["x"].shape == (3, 320)

    train_subjects, val_subjects = split_source_subjects(
        [f"s{i}" for i in range(10)], seed=3, val_fraction=0.2
    )
    assert set(train_subjects).isdisjoint(val_subjects)
    assert set(train_subjects) | set(val_subjects) == {f"s{i}" for i in range(10)}

    # End-to-end nested source-only K selection on a tiny synthetic sample set.
    # This is a protocol/software test, not a performance reproduction.
    protocol_x = rng.normal(size=(8 * 9, 620)).astype(np.float32)
    protocol_y = np.tile(np.array([0, 1, 2], dtype=np.int64), 8 * 3)
    protocol_subject = np.repeat(np.array([f"p{i}" for i in range(8)]), 9)
    protocol_data = {
        "mode": "mde_samples", "x": protocol_x, "y": protocol_y,
        "subject": protocol_subject,
        "session": np.zeros(len(protocol_y), dtype=np.int64),
        "trial": np.arange(len(protocol_y), dtype=np.int64),
    }
    with tempfile.TemporaryDirectory() as tmp:
        args = SimpleNamespace(
            seed=17, fold_seed_offset=0, val_fraction=0.25,
            device="cpu", dropout=0.1, epochs=8, batch_size=96,
            lr=2e-4, weight_decay=1e-3, lambda1=0.1, lambda2=0.05,
            tau=1.0, update_every=8, soft_updates=3, hard_updates=3,
            out=tmp, verbose=False,
        )
        fold_result = run_outer_fold(
            protocol_data, ["p0"], "seed", args, "smoke_fold"
        )
        assert fold_result["selected_K"] in (3, 4, 5)
        assert len(fold_result["predictions"]) == 9
        assert set(fold_result["target_subjects"]) == {"p0"}
        assert len(fold_result["source_validation_subjects_for_K"]) == 2

    print("PASS: MDE, overlapping windows, 62/32-channel model, latent domains,")
    print("      8-epoch training cycle, nested source-only K selection, evaluation,")
    print("      and subject-first trial preprocessing.")
    print("NOTE: Synthetic smoke test only; it does not reproduce reported EEG accuracies.")


if __name__ == "__main__":
    main()
