# PRISM Core Implementation

This repository provides the core implementation of **PRISM** for cross-subject EEG emotion recognition. It covers the MDE representation, spatial-frequency attention, entropy-guided latent-domain modeling, domain-invariant representation learning, and subject-independent evaluation protocols described in the manuscript.

## Code organization

| File | Function | Manuscript correspondence |
|---|---|---|
| `preprocess_de.py` | Constructs 0th-order DE and first-order temporal differential DE; prepares within-trial windows | Section 3.2, Eqs. (1)–(3); Section 4.1 |
| `prism_model.py` | Spatial-frequency attention, residual bottleneck, emotion/domain classifiers, and GRL | Sections 3.2–3.4 |
| `latent_domain.py` | Class-guided centroids, predictive entropy, entropy-weighted K-means++, soft assignment, and centroid refinement | Section 3.3.3, Eqs. (8)–(16) |
| `training.py` | Fine-grained feature update and domain-invariant optimization with the joint objective | Sections 3.3.1 and 3.4, Eqs. (4)–(5), (17)–(19); Algorithm 1 |
| `experiment_protocol.py` | Source-validation selection of K, nested LOSO, and subject-independent FACED 10-fold evaluation | Section 4.2 |
| `smoke_test.py` | Checks the main modules and execution flow | — |
| `CODE_MAP.md` | Equation- and section-level code index | — |
| `requirements.txt` | Python dependencies | — |

## Main configuration

| Parameter | Setting |
|---|---:|
| MDE input dimension | 620 (SEED/SEED-IV); 320 (FACED) |
| Bottleneck | 620/320 → 512 → 256 → 256 |
| Optimizer | Adam |
| Learning rate | `2e-4` |
| Weight decay | `1e-3` |
| Dropout | `0.3` |
| Batch size | `96` |
| Training epochs | `80` |
| Joint objective | `L = L_cls + 0.1 L_dc + 0.05 L_dom` |
| Temperature | `tau = 1.0` |
| Refinement | `T = 10`, `R = 3`, `Q = 3` |
| Pseudo-domain update interval | `8` epochs |
| K candidates | SEED `{3, 4, 5}`; SEED-IV `{4, 5, 6}`; FACED `{9, 10, 11}` |

## Data format

Prepare the official DE features in an NPZ file. For trial-wise DE input, use:

```python
np.savez_compressed(
    "seed_trials.npz",
    de=de_steps,             # [total_epochs, 62, 5]
    offsets=trial_offsets,   # [number_of_trials + 1]
    y=trial_labels,          # one label per trial
    subject=trial_subjects,  # one subject ID per trial
    session=trial_sessions,  # optional
    trial=trial_ids,          # optional
)
```

For SEED-IV, use the same structure with four emotion classes. For FACED, use 32 channels and nine emotion classes. Alternatively, provide preconstructed MDE samples:

```python
np.savez_compressed(
    "faced_mde.npz",
    x=mde_samples,  # [N, 320]
    y=labels,        # [N]
    subject=subject_ids,
)
```

The sample-based format also supports SEED and SEED-IV with 620-dimensional inputs. Labels may be zero-based or one-based; the loader converts them to zero-based indices.

## Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Check the implementation:

```bash
python smoke_test.py
```

Run SEED:

```bash
python experiment_protocol.py --dataset seed --data seed_trials.npz
```

Run SEED-IV:

```bash
python experiment_protocol.py --dataset seed-iv --data seed_iv_trials.npz
```

Run FACED:

```bash
python experiment_protocol.py --dataset faced --data faced_mde.npz
```

The scripts save fold-level K-selection results, evaluation metrics, predictions, and summary results in the output directory. Use `--help` to view the available options.
