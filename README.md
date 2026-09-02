# Z-to-tau-tau DGPO reconstruction toy

This repository implements a GPU-capable closure study for choosing among ambiguous invisible reconstructions in a minimal `Z -> tau tau` event model. The policy never receives truth `x`, truth `C`, or the true tau direction. It sees only the two smeared visible four-momenta and samples a two-dimensional tangent-plane action that defines a candidate tau axis.

The implemented chain is:

1. Sample `(c_A, c_B)` from `p(c_A,c_B|C) = (1 + C c_A c_B)/4`.
2. Generate the visible and invisible tau-decay systems and verify four-momentum closure.
3. Smear the two visible four-momenta and construct an observed seed direction.
4. Train a conditional normalizing flow on the exact sphere-log-map target at nominal `C0 = 0.60`.
5. Reconstruct `y = c_A,reco c_B,reco` explicitly from each candidate tau four-momentum.
6. Train and freeze `s_ref(y) = E[t(X)|Y=y]`, where `t(x)=x/(1+C0 x)`.
7. Require pre-DGPO agreement among score Fisher, fine-binned Poisson Fisher, and nominal pseudo-experiment width.
8. Optimize candidate preferences with the exact event-replacement Fisher reward and, for the trusted policy, `KL(q_phi || q_ref)`.
9. Fit independent off-nominal pseudo-data with nominal-only, analytically reweighted, reconstructed-level Poisson templates.

The final parameter estimate and its asymmetric 68% interval come from the Poisson forward-folding likelihood and `Delta(-2 log L)=1`. The pipeline deliberately stops before DGPO if the pre-training Fisher closure gate fails.

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
```

The terminal reports progress for baseline flow MLE, frozen score regression, the pre-DGPO closure toys, each DGPO policy, and all off-nominal pseudo-experiments.

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

`config/smoke.yaml` is a software-path test only. Its small ensemble cannot establish scientific coverage.

## Validation

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
toy-dgpo run --config config/smoke.yaml --device cpu
```
