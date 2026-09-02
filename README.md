# Toy DGPO unfolding

GPU-capable toy pipeline for testing whether reconstruction optimized at one nominal physics point remains calibrated away from it. The default configuration uses only one nominal calibration sample at `C0 = 0.60`, trains a baseline and two DGPO variants, then evaluates independent pseudo-data at `C_true = 0.20, 0.40, 0.60, 0.80, 0.90`.

The final parameter estimate is based on iterative Bayesian (D'Agostini) unfolding:

1. Build `P(reco bin | truth bin)` from the nominal response split only.
2. Unfold every pseudo-dataset for the configured number of iterations.
3. Propagate Poisson data fluctuations through the unfolding with a numerical Jacobian, retaining the full truth-bin covariance.
4. Fit the unfolded truth spectrum to nominal-MC templates reweighted exactly by
   `w(C) = (1 + C*x) / (1 + C0*x)`.
5. Quote asymmetric 68% errors from `Delta chi2 = 1`, and validate them with pulls and empirical coverage.

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
- `checkpoints/`: baseline, score-model, Fisher-DGPO, and bias-controlled-DGPO weights;
- `pseudo_experiments.csv`: one unfolded estimate and asymmetric interval per toy;
- `summary.csv` and `summary.json`: bias, normalized bias, precision, RMSE, pulls, and coverage;
- `12_bias_linearity.png` through `16_C_hat_offnominal.png`: the mandatory closure figures.

The reported unfolding covariance contains pseudo-data counting uncertainty. It does not include finite nominal-response MC uncertainty; increase `data.nominal_events` until that contribution is negligible, or treat it as a separate systematic study.

## Validation

```bash
python3 -m unittest discover -s tests
toy-dgpo run --config config/smoke.yaml
```

The smoke configuration is only a software check; three pseudo-experiments are not a scientific closure result.
