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

## Validation

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
toy-dgpo run --config config/smoke.yaml --device cpu
```
