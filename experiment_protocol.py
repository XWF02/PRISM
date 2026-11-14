"""Nested source-only K selection and subject-independent evaluation for PRISM."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from prism_model import PRISM
from preprocess_de import seed_seediv_window_mde
from training import evaluate, train_prism

DATASET_CONFIG = {
    "seed": {"channels": 62, "classes": 3},
    "seed-iv": {"channels": 62, "classes": 4},
    "faced": {"channels": 32, "classes": 9},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _zero_base_labels(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    values = set(np.unique(labels).tolist())
    if values == set(range(1, num_classes + 1)):
        labels = labels - 1
        values = set(np.unique(labels).tolist())
    if not values or not values.issubset(set(range(num_classes))):
        raise ValueError(f"Labels must be zero-based [0,{num_classes-1}] or one-based [1,{num_classes}]")
    return labels


def load_dataset_npz(path: str | Path, dataset: str) -> dict:
    """Load either trial-packed DE or already constructed MDE samples.

    Preferred SEED/SEED-IV format (construct windows only after each subject
    split):
      de: [total_epochs, 62, 5], offsets: [n_trials+1],
      y: [n_trials], subject: [n_trials]. Optional session/trial per trial.

    Sample format (also used for FACED because its exact window reduction is
    not specified in the manuscript):
      x: [N, 620] or [N, 320], y: [N], subject: [N].
    """
    cfg = DATASET_CONFIG[dataset]
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        if {"de", "offsets", "y", "subject"}.issubset(keys):
            de = np.asarray(archive["de"], dtype=np.float32)
            offsets = np.asarray(archive["offsets"], dtype=np.int64).reshape(-1)
            y = _zero_base_labels(archive["y"], cfg["classes"])
            subject = np.asarray(archive["subject"])
            n_trials = len(y)
            session = np.asarray(archive["session"]) if "session" in keys else np.zeros(n_trials, dtype=np.int64)
            trial = np.asarray(archive["trial"]) if "trial" in keys else np.arange(n_trials)
            if de.ndim != 3 or de.shape[1:] != (cfg["channels"], 5):
                raise ValueError(f"de must have shape [total_epochs,{cfg['channels']},5]")
            if not np.isfinite(de).all():
                raise ValueError("de contains NaN or Inf")
            if len(offsets) != n_trials + 1 or offsets[0] != 0 or offsets[-1] != len(de):
                raise ValueError("offsets must start at 0, end at len(de), and have n_trials+1 entries")
            if np.any(np.diff(offsets) <= 0):
                raise ValueError("Every trial must contain at least one DE epoch")
            if not (len(subject) == len(session) == len(trial) == n_trials):
                raise ValueError("y, subject, session and trial must have one entry per trial")
            return {
                "mode": "trial_de", "de": de, "offsets": offsets, "y": y,
                "subject": subject, "session": session, "trial": trial,
            }

        if {"x", "y", "subject"}.issubset(keys):
            x = np.asarray(archive["x"], dtype=np.float32)
            y = _zero_base_labels(archive["y"], cfg["classes"])
            subject = np.asarray(archive["subject"])
            session = np.asarray(archive["session"]) if "session" in keys else np.zeros(len(y), dtype=np.int64)
            trial = np.asarray(archive["trial"]) if "trial" in keys else np.arange(len(y))
        else:
            raise ValueError("NPZ must contain either (de, offsets, y, subject) or (x, y, subject)")

    expected_dim = cfg["channels"] * 5 * 2
    if x.ndim != 2 or x.shape[1] != expected_dim:
        raise ValueError(f"{dataset} expects x shape [N,{expected_dim}], got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("x contains NaN or Inf")
    if not (len(x) == len(y) == len(subject) == len(session) == len(trial)):
        raise ValueError("x, y, subject, session and trial must have the same N")
    return {
        "mode": "mde_samples", "x": x, "y": y,
        "subject": subject, "session": session, "trial": trial,
    }


def _subject_mask(subject_array: np.ndarray, selected_subjects) -> np.ndarray:
    selected = {str(s) for s in selected_subjects}
    return np.asarray([str(s) in selected for s in subject_array], dtype=bool)


def materialize_subject_samples(data: dict, selected_subjects, dataset: str) -> dict:
    """Select subjects first, then construct SEED/SEED-IV windows if needed."""
    mask = _subject_mask(data["subject"], selected_subjects)
    cfg = DATASET_CONFIG[dataset]

    if data["mode"] == "mde_samples":
        if not mask.any():
            raise ValueError(f"No samples found for subjects {list(selected_subjects)}")
        return {
            "x": data["x"][mask], "y": data["y"][mask],
            "subject": data["subject"][mask], "session": data["session"][mask],
            "trial": data["trial"][mask],
        }

    # Trial-level DE path: outer/source subject splits have already been made.
    # SEED/SEED-IV use the stated 10-step, stride-5 windows. For FACED, the
    # manuscript gives no sliding-window length; therefore each epoch-wise
    # MDE vector is retained as one sample (no extra temporal windowing).
    from preprocess_de import mde_sequence
    windows, labels, subjects, sessions, trials = [], [], [], [], []
    for trial_idx in np.flatnonzero(mask):
        start, end = data["offsets"][trial_idx:trial_idx + 2]
        de_trial = data["de"][start:end]
        if dataset == "faced":
            trial_windows = mde_sequence(de_trial).reshape(len(de_trial), -1)
        else:
            trial_windows = seed_seediv_window_mde(de_trial, window=10, stride=5, pooling="mean")
        if len(trial_windows) == 0:
            continue
        windows.append(trial_windows)
        labels.append(np.full(len(trial_windows), data["y"][trial_idx], dtype=np.int64))
        subjects.append(np.repeat(data["subject"][trial_idx], len(trial_windows)))
        sessions.append(np.repeat(data["session"][trial_idx], len(trial_windows)))
        trials.append(np.repeat(data["trial"][trial_idx], len(trial_windows)))

    if not windows:
        raise ValueError("Selected subjects produced no 10-step windows; check trial lengths and split")
    x = np.concatenate(windows).astype(np.float32, copy=False)
    expected = cfg["channels"] * 10
    if x.shape[1] != expected:
        raise RuntimeError(f"Internal MDE dimension error: expected {expected}, got {x.shape[1]}")
    return {
        "x": x, "y": np.concatenate(labels),
        "subject": np.concatenate(subjects),
        "session": np.concatenate(sessions),
        "trial": np.concatenate(trials),
    }


def split_source_subjects(subjects, seed: int = 42, val_fraction: float = 0.2):
    """Subject-level source train/validation split; val_fraction is not specified in paper."""
    subjects = np.asarray(list(subjects)).copy()
    if len(subjects) < 2:
        raise ValueError("At least two source subjects are required for source-only K validation")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_val = max(1, min(len(subjects) - 1, int(round(len(subjects) * val_fraction))))
    return subjects[n_val:].tolist(), subjects[:n_val].tolist()


def source_normalize(x_train: np.ndarray, x_other: np.ndarray):
    """Fit feature-wise statistics on source training samples only."""
    mean = x_train.mean(axis=0, keepdims=True)
    std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-6)
    x_train_norm = ((x_train - mean) / std).astype(np.float32)
    x_other_norm = ((x_other - mean) / std).astype(np.float32)
    return x_train_norm, x_other_norm, mean, std


def fit_candidate(k, x_train, y_train, x_val, y_val, cfg, args, seed):
    set_seed(seed)
    model = PRISM(cfg["channels"], cfg["classes"], k, dropout=args.dropout)
    model, _history = train_prism(
        model, x_train, y_train, device=args.device, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.lr,
        weight_decay=args.weight_decay, lambda1=args.lambda1,
        lambda2=args.lambda2, tau=args.tau,
        update_every=args.update_every, soft_updates=args.soft_updates,
        hard_updates=args.hard_updates, seed=seed, verbose=False,
    )
    score, _, _ = evaluate(model, x_val, y_val, args.device, args.batch_size)
    return score


def run_outer_fold(data, target_subjects, dataset, args, fold_name):
    cfg = DATASET_CONFIG[dataset]
    all_subjects = np.unique(data["subject"]).tolist()
    target_keys = {str(s) for s in target_subjects}
    source_subjects = [s for s in all_subjects if str(s) not in target_keys]
    train_subjects, val_subjects = split_source_subjects(
        source_subjects, seed=args.seed + args.fold_seed_offset,
        val_fraction=args.val_fraction,
    )

    # For trial-level SEED/SEED-IV data, each subject split is made BEFORE
    # constructing windows, as required by Section 4.1 of the manuscript.
    train_data = materialize_subject_samples(data, train_subjects, dataset)
    val_data = materialize_subject_samples(data, val_subjects, dataset)
    x_train, y_train = train_data["x"], train_data["y"]
    x_val, y_val = val_data["x"], val_data["y"]
    x_train_norm, x_val_norm, _val_mean, _val_std = source_normalize(x_train, x_val)

    candidate_ks = [cfg["classes"] + offset for offset in range(3)]
    validation_scores = {}
    for k in candidate_ks:
        score = fit_candidate(
            k, x_train_norm, y_train, x_val_norm, y_val, cfg, args,
            args.seed + args.fold_seed_offset + k,
        )
        validation_scores[str(k)] = score
        print(f"{fold_name}: source-validation K={k}: {score:.2f}%", flush=True)

    # Select K exclusively from source validation. Target test labels are not
    # used in model selection, preprocessing statistics, or training.
    selected_k = max(candidate_ks, key=lambda k: (validation_scores[str(k)], -k))

    final_source_data = materialize_subject_samples(data, source_subjects, dataset)
    target_data = materialize_subject_samples(data, target_subjects, dataset)
    x_source, y_source = final_source_data["x"], final_source_data["y"]
    x_test, y_test = target_data["x"], target_data["y"]
    x_source_norm, x_test_norm, mean, std = source_normalize(x_source, x_test)

    final_seed = args.seed + args.fold_seed_offset + selected_k + 1000
    set_seed(final_seed)
    model = PRISM(cfg["channels"], cfg["classes"], selected_k, dropout=args.dropout)
    model, history = train_prism(
        model, x_source_norm, y_source, device=args.device,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        lambda1=args.lambda1, lambda2=args.lambda2, tau=args.tau,
        update_every=args.update_every, soft_updates=args.soft_updates,
        hard_updates=args.hard_updates, seed=final_seed, verbose=args.verbose,
    )
    test_accuracy, predictions, truths = evaluate(
        model, x_test_norm, y_test, args.device, args.batch_size
    )

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "selected_K": int(selected_k),
        "dataset": dataset,
        "target_subjects": [str(s) for s in target_subjects],
        "source_subjects": [str(s) for s in source_subjects],
        "normalization_mean": mean,
        "normalization_std": std,
        "seed": int(final_seed),
        "settings": _settings_dict(args),
    }, output_dir / f"{fold_name}.pt")

    return {
        "fold": fold_name,
        "data_mode": data["mode"],
        "target_subjects": [str(s) for s in target_subjects],
        "source_train_subjects_for_K": [str(s) for s in train_subjects],
        "source_validation_subjects_for_K": [str(s) for s in val_subjects],
        "source_subjects_for_final_fit": [str(s) for s in source_subjects],
        "candidate_K_source_validation_accuracy": validation_scores,
        "selected_K": int(selected_k),
        "target_test_accuracy": test_accuracy,
        "n_test_samples": int(len(truths)),
        "settings": _settings_dict(args),
        "final_seed": int(final_seed),
        "predictions": predictions.tolist(),
        "labels": truths.tolist(),
        "history": history,
        "normalization_mean": mean.reshape(-1).tolist(),
        "normalization_std": std.reshape(-1).tolist(),
    }


def _settings_dict(args):
    return {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "tau": args.tau,
        "T": args.epochs // args.update_every,
        "update_every": args.update_every,
        "R_soft_updates": args.soft_updates,
        "Q_hard_updates": args.hard_updates,
        "source_validation_fraction": args.val_fraction,
        "seed": args.seed,
    }


def build_faced_folds(subjects, n_folds: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(list(subjects)).copy()
    rng.shuffle(shuffled)
    return [fold.tolist() for fold in np.array_split(shuffled, n_folds)]


def load_faced_folds(path: str | Path, subjects):
    with open(path, "r", encoding="utf-8") as file:
        raw_folds = json.load(file)
    subject_lookup = {str(subject): subject for subject in subjects}
    flat = [str(subject) for fold in raw_folds for subject in fold]
    expected = set(subject_lookup)
    if len(raw_folds) != 10 or len(flat) != len(set(flat)) or set(flat) != expected:
        raise ValueError("FACED fold file must contain 10 disjoint folds covering every subject exactly once")
    # Map JSON strings/numbers back to the original subject dtype. This avoids
    # int-vs-string mismatches in np.isin / subject selection.
    return [[subject_lookup[str(subject)] for subject in fold] for fold in raw_folds]


def main():
    parser = argparse.ArgumentParser(description="PRISM nested source-only K selection and evaluation")
    parser.add_argument("--dataset", required=True, choices=DATASET_CONFIG)
    parser.add_argument("--data", required=True, help="Dataset NPZ; see README for both supported schemas")
    parser.add_argument("--out", default="reviewer_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Source-subject validation fraction; manuscript does not specify its exact value")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lambda1", type=float, default=0.1)
    parser.add_argument("--lambda2", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--update-every", type=int, default=8)
    parser.add_argument("--soft-updates", type=int, default=3)
    parser.add_argument("--hard-updates", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--subject", default=None, help="Run one held-out SEED/SEED-IV subject only")
    parser.add_argument("--faced-folds-json", default=None,
                        help="Optional original FACED fold membership: JSON list of 10 subject lists")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.epochs != 80 or args.batch_size != 96 or args.update_every != 8:
        print("WARNING: supplied epoch/batch/update settings differ from manuscript defaults", flush=True)
    if args.epochs % args.update_every != 0:
        raise ValueError("epochs must be divisible by update-every")

    data = load_dataset_npz(args.data, args.dataset)
    subjects = np.unique(data["subject"]).tolist()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("seed", "seed-iv"):
        if args.subject is None:
            target_subjects = subjects
        else:
            matched = [s for s in subjects if str(s) == str(args.subject)]
            if not matched:
                raise ValueError(f"Unknown subject {args.subject}; available IDs: {subjects}")
            target_subjects = matched
        outer_folds = [(f"subject_{subject}", [subject]) for subject in target_subjects]
    else:
        if args.faced_folds_json:
            folds = load_faced_folds(args.faced_folds_json, subjects)
        else:
            folds = build_faced_folds(subjects, n_folds=10, seed=args.seed)
        with open(output_dir / "faced_subject_folds.json", "w", encoding="utf-8") as file:
            json.dump([[str(s) for s in fold] for fold in folds], file, indent=2)
        outer_folds = [(f"fold_{index + 1:02d}", fold) for index, fold in enumerate(folds)]

    fold_results = []
    for fold_index, (fold_name, targets) in enumerate(outer_folds):
        args.fold_seed_offset = fold_index * 10000
        result = run_outer_fold(data, targets, args.dataset, args, fold_name)
        fold_results.append(result)
        summary_result = {
            key: value for key, value in result.items()
            if key not in {"predictions", "labels", "history", "normalization_mean", "normalization_std"}
        }
        with open(output_dir / f"{fold_name}.json", "w", encoding="utf-8") as file:
            json.dump(summary_result, file, indent=2, ensure_ascii=False)
        np.savez_compressed(
            output_dir / f"{fold_name}_predictions.npz",
            predictions=np.asarray(result["predictions"], dtype=np.int64),
            labels=np.asarray(result["labels"], dtype=np.int64),
        )
        print(
            f"{fold_name}: selected K={result['selected_K']}, "
            f"target accuracy={result['target_test_accuracy']:.2f}%",
            flush=True,
        )

    scores = np.asarray([item["target_test_accuracy"] for item in fold_results], dtype=np.float64)
    summary = {
        "dataset": args.dataset,
        "n_outer_folds": int(len(scores)),
        "mean_accuracy": float(scores.mean()),
        "sd_accuracy_sample": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "fold_results": [
            {key: value for key, value in item.items()
             if key not in {"predictions", "labels", "history", "normalization_mean", "normalization_std"}}
            for item in fold_results
        ],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(f"Overall: {summary['mean_accuracy']:.2f} ± {summary['sd_accuracy_sample']:.2f}%", flush=True)


if __name__ == "__main__":
    main()
