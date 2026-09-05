# makemore

A reimplementation of the Makemore lessons in Andrej Karpathy's
[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) video
series. This project follows the series' progression through character-level
language modeling with PyTorch, building models that generate new names from a
dataset of 32,032 examples.

The notebooks start with bigram counts and progress to a multilayer perceptron
with hand-built neural-network components. They are intended as a hands-on
learning exercise rather than a reusable library.

## Notebooks

| Notebook | Topics |
| --- | --- |
| `makemore_v1.ipynb` | Data preparation, a count-based bigram model, and an equivalent single-layer neural network |
| `makemore_v2.ipynb` | Character embeddings, context windows, an MLP, train/dev/test splits, hyperparameter search, and name generation |
| `makemore_v3.ipynb` | A small module system, custom embedding/linear/tanh/batch-normalization layers, deeper networks, and training diagnostics |

The notebooks are intended to be read and run in order.

## Getting started

This project uses Python 3.14 and [uv](https://docs.astral.sh/uv/) for dependency
management.

```bash
git clone https://github.com/rahuliyer/makemore.git
cd makemore
uv sync
uv run jupyter lab
```

Open `makemore_v1.ipynb` in JupyterLab to begin. The notebooks load the included
dataset from `data/names.txt`, so no additional downloads are required.

## Project structure

```text
.
├── data/
│   └── names.txt       # Training data: one name per line
├── makemore_v1.ipynb  # Bigram language model
├── makemore_v2.ipynb  # MLP language model
├── makemore_v3.ipynb  # Deeper network and diagnostics
├── pyproject.toml
└── uv.lock
```

## What this project covers

- Turning text into character-level training examples
- Estimating and sampling from bigram probabilities
- Training models with gradient descent and cross-entropy loss
- Learning character embeddings and fixed-length context representations
- Evaluating model choices with train, development, and test splits
- Implementing core neural-network layers directly with PyTorch tensors
- Inspecting activations, gradients, and update-to-data ratios

## Acknowledgments

This project follows Andrej Karpathy's Makemore lectures and is based on the
materials in the official [`karpathy/nn-zero-to-hero`](https://github.com/karpathy/nn-zero-to-hero)
and [`karpathy/makemore`](https://github.com/karpathy/makemore) repositories.
