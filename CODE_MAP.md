# PRISM Manuscript-to-Code Map

| Manuscript section / equation | Implementation |
|---|---|
| Section 3.2, Eqs. (1)–(3): 0th-order DE and first-order temporal differential DE | `preprocess_de.py:mde_sequence` |
| Section 4.1: within-trial 10-step, stride-5 SEED/SEED-IV windows | `preprocess_de.py:seed_seediv_window_mde` |
| Section 3.2: spatial/channel attention and learnable frequency-band weighting | `prism_model.py:SpatialFrequencyAttention` |
| Section 3.2: residual bottleneck and emotion classifier | `prism_model.py:PRISM`, `ResidualMLPBlock` |
| Section 3.3.1, Eqs. (4)–(5): pseudo domain–class labels and fine-grained feature update | `training.py:_run_fine_grained_epoch` |
| Section 3.3.3, Eqs. (8)–(9): class-guided centroids and predictive entropy | `latent_domain.py:characterize_latent_domains` |
| Section 3.3.3, Eqs. (10)–(12): entropy-weighted K-means++ and soft centroid refinement | `latent_domain.py:entropy_weighted_kmeanspp`, `characterize_latent_domains` |
| Section 3.3.3, Eqs. (13)–(16): nearest-centroid assignment and hard centroid refinement | `latent_domain.py:characterize_latent_domains` |
| Sections 3.3.3–3.4, Eqs. (17)–(19): domain classifier, GRL, and joint objective | `prism_model.py:GradientReversal`, `training.py:_run_domain_invariant_epoch` |
| Algorithm 1: iterative PRISM training | `training.py:train_prism` |
| Section 4.2: source-validation K selection, nested LOSO, and FACED 10-fold evaluation | `experiment_protocol.py` |
