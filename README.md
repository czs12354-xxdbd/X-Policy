# X-Policy

X-Policy is a vision-language-action policy for closed-loop, long-horizon robot manipulation. This repository is the standalone source release of the model evaluated on VLA-Arena.

[Checkpoint](https://huggingface.co/ChenShu55/X-Policy-VLA-Arena) · [VLA-Arena](https://github.com/PKU-Alignment/VLA-Arena) · [OpenPI](https://github.com/Physical-Intelligence/openpi)

## What changed

X-Policy augments the inherited VLA policy with:

- eight recurrent cross-replan memory tokens, split into four fast and four slow slots;
- ordered subgoal-progress and transition supervision for workflow retention;
- conditioning on the previous five executed actions;
- a zero-initialized policy boundary that preserves the parent policy at initialization; and
- target-preserving PCGrad for the auxiliary training stream.

The released checkpoint is step **18,000**. Its formal VLA-Arena run used five-step replanning, exactly **1,700 episodes**, and produced **924 successes**. The cell-mean success rate is **53.21%** and the episode-weighted success rate is **54.35%**. The complete machine-readable breakdown is in [`results/vla_arena.json`](results/vla_arena.json).

| Level | Cell-mean success rate |
| --- | ---: |
| L0 | 76.18% |
| L1 | 50.18% |
| L2 | 33.27% |

## Installation

X-Policy requires Python 3.11 and an NVIDIA GPU for the released JAX configuration.

```bash
git clone https://github.com/czs12354-xxdbd/X-Policy.git
cd X-Policy
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e packages/openpi-client
pip install -e .
```

The default dependency selects the CUDA 12 build of JAX. Adjust the JAX installation separately if your platform differs.

## Download the checkpoint

```bash
pip install -U huggingface_hub
huggingface-cli download ChenShu55/X-Policy-VLA-Arena \
  --local-dir checkpoints/x-policy
```

The repository includes the small VLA-Arena normalization asset under `assets/`; model weights remain on Hugging Face.

## Serve the policy

```bash
python scripts/serve_policy.py \
  --policy.config x_policy_vla_arena \
  --policy.dir checkpoints/x-policy
```

The server uses the public config name `x_policy_vla_arena`. The VLA-Arena evaluator can connect to it over the OpenPI websocket client.

## Training

The released model was trained with a target L0 stream and an auxiliary L1 stream. Edit the dataset, semantic-phase metadata, and parent-checkpoint paths in [`configs/x_policy_vla_arena.yaml`](configs/x_policy_vla_arena.yaml), then run:

```bash
python trainer.py --config configs/x_policy_vla_arena.yaml
```

The configuration uses batches of 32 for each stream, a peak learning rate of `1e-5`, 30,000 scheduled steps, suite-balanced target sampling, and persistent-memory-only auxiliary gradients. The published winner was selected at step 18,000.

## Evaluation

Install VLA-Arena in the same environment, then use `evaluator.py` or the benchmark's evaluation launcher with:

- config: `x_policy_vla_arena`
- checkpoint: `checkpoints/x-policy`
- replan steps: `5`

The checked-in result is a formal 33-cell evaluation. It does not use action smoothing, adaptive replanning, or checkpoint selection during evaluation.

## Repository layout

```text
assets/                 normalization statistics
configs/                public training configuration
packages/openpi-client/ websocket inference client
results/                formal VLA-Arena result
scripts/                training and policy-serving entry points
src/openpi/             model, policy, and data pipeline implementation
```

## License and attribution

This project is released under the Apache License 2.0. It is derived from OpenPI and the VLA-Arena OpenPI integration; original notices are retained in source files. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
