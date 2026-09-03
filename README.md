# Z-to-tau-tau DGPO reconstruction toy

This repository implements a GPU-capable closure study for choosing among ambiguous invisible reconstructions in a minimal `Z -> tau tau` event model. The policy never receives truth `x`, truth `C`, or the true tau direction. It sees only the two smeared visible four-momenta and samples a two-dimensional tangent-plane action that defines a candidate tau axis.

The implemented chain is:

1. Sample `(c_A, c_B)` from `p(c_A,c_B|C) = (1 + C c_A c_B)/4`.
2. Generate the visible and invisible tau-decay systems and verify four-momentum closure.
3. Smear the two visible four-momenta and construct an observed seed direction.
4. Train a conditional normalizing flow on the exact sphere-log-map target at nominal `C0 = 0.60`.
5. Reconstruct `y = c_A,reco c_B,reco` explicitly from each candidate tau four-momentum.
6. Train `s_r(y) = E[t(X)|Y=y; pi_r]`, where `t(x)=x/(1+C0 x)`, and freeze it within each local update.
7. Require pre-DGPO agreement among score Fisher, fine-binned Poisson Fisher, and nominal pseudo-experiment width.
8. Compare the complete frozen/iterative x trust/no-trust 2x2 ablation. Both use the exact event-replacement Fisher reward; iterative trust is local `KL(pi_phi || pi_r)` and global `KL(pi_phi || pi_0)` remains diagnostic only.
9. Select checkpoints only with an independent fixed nominal validation sample and the configured 80-bin direct reconstructed-level Fisher, then fit separate off-nominal pseudo-data with independently calibrated Poisson templates.

The final parameter estimate and its asymmetric 68% interval come from the Poisson forward-folding likelihood and `Delta(-2 log L)=1`. The pipeline deliberately stops before DGPO if the pre-training Fisher closure gate fails.

Metric roles are deliberately separated: active surrogate Fisher and reward are training quantities; an independently re-estimated policy-score Fisher is a refresh diagnostic; high-statistics direct binned validation Fisher selects the checkpoint; and the independent pseudo-experiment `Std(C_hat)` is the final scientific validation. Final pseudo-experiments are run once after selection and never participate in early stopping.

## Install and check the GPU

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA GPU')"
```

## Run over SSH

All physics, detector, flow, DGPO, inference, and plotting controls are in [`config/default.yaml`](config/default.yaml).

```bash
tmux new -s ztautau-dgpo
source .venv/bin/activate
toy-dgpo run --config config/default.yaml --device cuda
```

Detach with `Ctrl-b d` and reconnect with:

```bash
tmux attach -t ztautau-dgpo
```

Training and evaluation can be split across allocations:

```bash
toy-dgpo train --config config/default.yaml --device cuda
toy-dgpo evaluate --config config/default.yaml --device cuda
toy-dgpo diagnose --config config/default.yaml --device cuda
toy-dgpo closure --config config/default.yaml --device cuda
```

`diagnose` does not update the policies. It loads the baseline and enabled optimized checkpoints, trains independent diagnostic-only policy-specific score models on new nominal MC, reuses the existing off-nominal summary, and writes `diagnosis.md`.

`closure` is strictly read-only with respect to every checkpoint. It uses binned policy-specific Poisson intensity scores and direct-versus-reweighted high-statistics samples, without training any flow or score model. It writes plots `19_extended_score_closure.png` through `23_asimov_likelihood_examples.png` and updates `diagnosis.md`.

The terminal reports progress for baseline flow MLE, every score refresh, each local DGPO round, high-statistics policy calibration, independent final score estimation, and all off-nominal pseudo-experiments.

## Outputs

The output directory contains the resolved YAML, versioned checkpoints, Fisher gate result, per-toy results, summaries, and:

- `00_truth_physics.png`
- `01_invisible_kinematics.png`
- `01b_observed_z_by_C.png`
- `02_baseline_reconstruction.png`
- `03_score_fisher_closure.png`
- `04_candidate_event_display.png`
- `05_training.png`
- `06_response_before_after.png`
- `07_offnominal_closure.png`
- `08_final_dashboard.png`
- `09_fisher_decomposition.png` through `18_reward_hacking_diagnosis.png`
- `diagnosis_metrics.csv`, `17_offnominal_detailed.csv`, `diagnosis_metrics.json`, and `diagnosis.md`
- `19_extended_score_closure.png` through `23_asimov_likelihood_examples.png`
- `extended_score_closure.csv`, `template_closure_metrics.csv`, and `statistical_closure.json`
- `ablation_study/roundwise_refresh_closure.png`
- `ablation_study/validation_early_stopping.png`
- `ablation_study/fisher_vs_PE_by_C.png` plus its CSV/JSON values
- `ablation_study/final_2x2_ablation.png`

The `ablation_study/` subdirectory contains the controlled 2x2 score-refresh and KL-trust comparison:

- `00_training_diagnostics.png`, the focused figures `01_ablation_fisher.png` through `08_final_ablation_dashboard.png`, and `09_checkpoint_selection_summary.png`
- `final_ablation_metrics.csv`, `final_ablation_metrics.json`, and `ablation_report.md`
- per-round stale-gap CSV/JSON and independent diagnostic score checkpoints for both iterative policies

The main comparison plots and the ablation plots are both regenerated during evaluation. Their method order comes from `ablation.policy_order`; evaluation fails instead of silently producing a partial comparison if any enabled checkpoint or summary is missing.

Validation histories and best/final comparisons are written to `checkpoint_validation_*.json`, `checkpoint_validation_*.csv`, and `checkpoint_selection_summary.{json,csv}`. Each optimized-policy checkpoint directory contains both `best_validation_policy.pt` and `final_policy.pt`; the root policy checkpoint used by downstream inference is always the best-validation version.

Round-aligned reference-policy, updated-policy, and active-score checkpoints are stored under `checkpoints/iterative_refresh_*/`. The default maximum budget is 50 optimized epochs (`max_refresh_rounds: 10`, five epochs per round), with configurable two-round patience and a 0.2% validation-Fisher improvement threshold. The ablation enforces all five policies, equal maximum optimized budgets, zero global KL in the loss, and a 20/40/60/80/100-bin stability scan around the primary 80-bin selection value.

`config/smoke.yaml` is a software-path test only. Its small ensemble cannot establish scientific coverage.

## Full spin-matrix extension

The spin-matrix study is an additive workflow and never overwrites the existing `C_nn` checkpoints. Its right-handed convention is fixed to `(r, n, k)`, with

```text
n   = normalize(z_beam x k_A)
k_A = k_A,  r_A = n x k_A
k_B = -k_A, r_B = n x k_B
```

The saved truth and reconstructed polarimeters use component order `(h^r, h^n, h^k)`. Consequently, the original observable and score remain exactly

```text
x = h_A^n h_B^n
t_Cnn = x / (1 + 0.60 x)
```

Run the passive 15-parameter impact, conditioning, correlation, and null-cross-talk study without training a policy:

```bash
toy-dgpo spin-passive --spin-config config/spin_matrix.yaml --device cuda
```

Train the separate Cdiag multi-measurement policy and then evaluate all comparisons:

```bash
toy-dgpo spin-run --spin-config config/spin_matrix.yaml --device cuda
```

Training and evaluation may also be separated:

```bash
toy-dgpo spin-train --spin-config config/spin_matrix.yaml --device cuda
toy-dgpo spin-evaluate --spin-config config/spin_matrix.yaml --device cuda
```

The Cdiag configuration remains the independent `(C_nn, C_rr, C_kk)` study. Its checkpoints stay under `outputs/ztautau_spin_matrix/`; the original `C_nn` study stays under `outputs/ztautau/`.

The polarization extension uses the decorrelated basis

```text
B_plus_n  = (B_A_n + B_B_n) / sqrt(2)
B_minus_n = (B_A_n - B_B_n) / sqrt(2)
```

and the dedicated BC5 target `(C_nn, C_rr, C_kk, B_plus_n, B_minus_n)`. Run it only after the independent Cdiag checkpoint exists:

```bash
toy-dgpo spin-run --spin-config config/spin_bc5.yaml --device cuda
```

This writes to `outputs/ztautau_spin_bc5/` and refuses to overwrite an existing controlled checkpoint directory unless `multi_training.allow_overwrite` is explicitly enabled. The fixed and adaptive controllers use the same maximum DGPO budget. The reward remains event-level rank-one replacement with grouped leave-one-out advantage and no KL term, but now minimizes the baseline-whitened objective `Tr(F0 F^-1) / d` in float64.

Adaptive refresh keeps one fixed nominal monitor sample separate from score training, policy optimization, round-reference construction, direct validation, and final evaluation. NMSE is only a warning. The refresh trigger is confirmed by drift in the independent joint binned validation Fisher, or forced by the configured maximum interval. Score models warm-start across refreshes; every third refresh also trains a fresh-init diagnostic score for path-dependence checks. Best BC5 checkpoints minimize independent direct-validation `J`, never NMSE or tau-axis error.

The separate `outputs/ztautau_spin_matrix/` directory contains:

- `spin_matrix_information_transfer.png`
- `spin_matrix_correlations.png`
- `multi_measurement_training.png`
- `multi_measurement_tradeoff.png`
- `cnn_vs_multitask_response.png`
- `spin_matrix_summary.png`
- full Fisher/eigenvalue/condition diagnostics and null-cross-talk CSV/JSON files

The isolated BC5 workflow additionally writes:

- `adaptive_refresh_diagnostics.png`
- `fixed_vs_adaptive_refresh.png`
- `BC5_profiled_precision.png`
- `BC5_fisher_eigenmodes.png`
- `full_spin_passive_transfer_after_BC5.png`
- `multi_measurement_summary.png`

Every figure has a CSV or JSON source-data companion. The baseline BC5 Fisher report includes `sigma(B_plus_n)`, `sigma(B_minus_n)`, and the full eigenvalue/eigenvector decomposition before policy training.

## Consolidated conditional-precision study

The primary spin-measurement study is isolated in `config/spin_conditional.yaml` and writes only to `outputs/ztautau_spin_conditional/`. The earlier Cnn, Cdiag, and exploratory BC5 outputs above remain reproducible historical studies and are not overwritten. The consolidated comparison trains exactly three no-trust policies from the same frozen baseline:

```text
Cnn   = (C_nn)
Cdiag = (C_nn, C_rr, C_kk)
BC5   = (C_nn, C_rr, C_kk, B_A_n, B_B_n)
```

The primary per-parameter uncertainty is conditional, `sigma_a = 1 / sqrt(F_aa)`, with the other spin coefficients fixed. The multi-target objective is `J_cond = mean(F0_aa / F_aa)`. Its event replacement uses only diagonal score squares; no Fisher inverse, jointly profiled covariance, Fisher eigenmode, or KL term enters the reward or checkpoint selection.

Each fixed two-epoch local update is accepted only when a freshly trained diagnostic score on an independent fixed high-statistics sample improves validation `J_cond` beyond a tolerance derived from the configured minimum and the split-sample validation noise. Rejected trials roll back to the current accepted policy, optionally retry once at lower learning rate, and count toward early stopping. NMSE, tau-axis error, and KL to the initial baseline remain diagnostics only.

Run training and evaluation together with:

```bash
toy-dgpo spin-conditional-run --spin-config config/spin_conditional.yaml --device cuda
```

Or split the stages:

```bash
toy-dgpo spin-conditional-train --spin-config config/spin_conditional.yaml --device cuda
toy-dgpo spin-conditional-evaluate --spin-config config/spin_conditional.yaml --device cuda
```

The six primary figures are `conditional_multitarget_summary.png`, `full15_conditional_passive_transfer.png`, `measurement_accept_reject_training.png`, `conditional_score_closure.png`, `target_information_decomposition.png`, and `final_conditional_summary_table.png`. Each has editable SVG and CSV/JSON source-data companions. `final_Cnn_pseudoexperiment_closure.{csv,json}` compares the predicted conditional Fisher width with the independent nominal pseudo-experiment `Std(C_hat)` using exact nominal reweighting.

## Validation

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
toy-dgpo run --config config/smoke.yaml --device cpu
toy-dgpo spin-run --spin-config config/spin_bc5_smoke.yaml --device cpu
toy-dgpo spin-conditional-run --spin-config config/spin_conditional_smoke.yaml --device cpu
```

The spin smoke commands require the preceding smoke `C_nn` and Cdiag checkpoints. Smoke outputs validate software paths only and are not physics-performance evidence.
