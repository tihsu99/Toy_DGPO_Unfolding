# Toy DGPO unfolding

GPU-capable toy pipeline for testing whether reconstruction optimized at one nominal physics point remains calibrated away from it. The default configuration uses only one nominal calibration sample at `C0 = 0.60`, trains a baseline and two DGPO variants, then evaluates independent pseudo-data at `C_true = 0.20, 0.40, 0.60, 0.80, 0.90`.

The final parameter estimate uses the required reconstructed-level Poisson forward-folding likelihood:

1. Reconstruct the nominal response split with each frozen policy.
2. Build every reconstructed template from that same nominal paired sample using
   `w(C) = (1 + C*x) / (1 + C0*x)`.
3. Fit each off-nominal pseudo-dataset with the binned Poisson deviance.
4. Quote asymmetric 68% errors from `Delta(-2 log L) = 1`, and validate them with pulls and empirical coverage.

Iterative Bayesian (D'Agostini) unfolding is retained as an explicit diagnostic. The output compares the generated truth spectrum, nominal prior, and unfolded spectrum, but unfolding no longer supplies the final parameter estimate.

The reconstructed-score estimator is not frozen at the baseline policy. Each DGPO epoch re-estimates `E[s0(x) | y]` on the independent nominal score split for the current policy before applying the group policy update. A truth-bin conditional-calibration loss constrains reconstruction bias without training on off-nominal pseudo-data.

No off-nominal sample is used for training, response calibration, priors, or fit templates. Off-nominal events are generated only as blind pseudo-data.

## Install

Create an environment on the GPU host and install the package. Install the PyTorch build appropriate for that host first if its CUDA stack requires a site-specific wheel.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

Confirm that PyTorch sees the GPU:

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA GPU')"
```

## Run over SSH

All experiment controls are in [`config/default.yaml`](config/default.yaml). For a long run, use `tmux`:

```bash
tmux new -s toy-dgpo
source .venv/bin/activate
toy-dgpo run --config config/default.yaml --device cuda
```

Detach with `Ctrl-b d` and reconnect with:

```bash
tmux attach -t toy-dgpo
```

The terminal shows progress bars for baseline training, score calibration, each DGPO policy, and the full closure ensemble. To preserve work across separate batch allocations, train and evaluate independently:

```bash
toy-dgpo train --config config/default.yaml --device cuda
toy-dgpo evaluate --config config/default.yaml --device cuda
```

Both commands must use the same configuration and output directory. `evaluate` loads checkpoints from `<output_dir>/checkpoints/`.

## Outputs

The default output directory is `outputs/default/` and contains:

- `resolved_config.yaml`: exact configuration used;
- `checkpoints/`: each policy and its corresponding reconstructed-score model;
- `pseudo_experiments.csv`: one forward-folded estimate and asymmetric interval per toy;
- `summary.csv` and `summary.json`: bias, normalized bias, precision, RMSE, pulls, and coverage;
- `policy_diagnostics.csv`: reconstruction correlation/MSE, response effective rank, singular values, and reconstructed Fisher information;
- `00_truth_generation.png`: generated truth distributions against the analytic density;
- `01_reco_vs_truth.png`: reconstructed output versus generated truth for every policy;
- `02_z_distributions.png`: all detector features at every tested `C`;
- `03_response_matrices.png` and `04_response_singular_values.png`: response and conditioning diagnostics;
- `05_unfolding_diagnostics.png`: generated truth, nominal prior, and D'Agostini unfolding;
- `06_forward_likelihood_scans.png`: representative Poisson likelihood scans;
- `12_bias_linearity.png` through `16_C_hat_offnominal.png`: the mandatory closure figures.

The likelihood currently treats the nominal reconstructed templates as exact. It therefore does not include finite nominal-response/template MC uncertainty; increase `data.nominal_events` until that contribution is negligible, or add it as a separate nuisance/systematic study.

## Validation

```bash
python3 -m unittest discover -s tests
toy-dgpo run --config config/smoke.yaml
```

The smoke configuration is only a software check; three pseudo-experiments are not a scientific closure result.
