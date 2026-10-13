# Kev

An open-source learning project exploring system one decision models in Rust.
The aim is to understand transformers by implementing small examples, then
training, adapting and serving models locally. The interactive visual
prototype is implemented in JavaScript. The Rust implementation has started
with one inspectable self-attention head; the full model is still to come.

## What we want to learn

- Build a tiny transformer from scratch in Rust and follow its computations.
- Train for accurate, calibrated decisions: predictions assigned 80% confidence
  should be correct about 80% of the time across comparable examples. Measure
  accuracy and calibration separately, and investigate uncertainty and abstention.
- Adapt compact open models for specific decision tasks and compare them with
  the educational implementation. Fine-tuning existing models and training a
  tiny model from scratch are separate experiments.
- Understand inference, serving and performance on local hardware before
  considering cloud resources.
- Explore extracting relationships and choosing nodes or paths in the
  voyagershell knowledge graph, with explicit evaluation tasks.

“System one” describes the intended fast, structured decisions. No particular
published architecture called Decision Transformer has been selected. Jev is a
source of curiosity and public research leads, not a claim to reproduce its
private training method or use its weights.

## Model and application

Kev is intended to be a general transformer-based typed decision model. A task
will supply context, a question and an allowed answer shape. Graph-building
will be its first training application; graph categories and relationship
names belong in task data rather than being hard-coded output labels.

The harness will retrieve candidates, sequence dependent decisions, batch
independent decisions and validate proposed graph changes. A general model
architecture does not itself supply general decision-making competence: that
requires training and evaluation across the tasks we want it to handle.

## A live understanding map

Build an interactive visual explanation alongside the model: inputs, intermediate
representations, attention, outputs, loss, gradients and parameter updates.
Connect inference measurements to latency, memory use and hardware activity.

Show actual recorded computations where possible. Label illustrations and
simplifications so an animation cannot be mistaken for measured execution.
The purpose is to understand each step well enough to explain it.

## Learning from use

A small decision harness should eventually record inputs, model/version,
outputs, confidence, observed outcomes and timings. Keep private traces and
large datasets out of this repository. Select and document shareable examples.

Investigate converting suitable traces into training data. Reinforcement learning
also needs a defined environment, actions, rewards and evaluation; traces alone
do not establish that training setup. First learn what a simple supervised
baseline can do, then compare other objectives and training methods.

## How we work through Kev

One small, understandable change at a time:

1. Name the question and show the proposed computation or behavior.
2. Implement that slice in its own branch/worktree and check it.
3. Commit and push it so the code is readable on GitHub.
4. Walk through before/after, pseudocode beside actual code, how it works,
   why this approach, what was proved and what remains uncertain.
5. Discuss and refine it before adding the next concept.

Understanding each operation is the purpose. Keep walkthroughs beside each
experiment; do not treat a working demo as proof that the learner understands it.

## Running

The first Rust slice uses the standard library only. From this directory, with
Rust 1.85 or newer (tested with 1.99.0):

```sh
cargo run --offline --example attention
cargo run --offline --example attention -- --uniform
cargo test --offline
```

Read [the first Rust walkthrough](walkthroughs/001-rust-attention.md) beside
[src/attention.rs](src/attention.rs). It follows queries, keys, values, scores,
attention weights and outputs. Inputs and weights are hand-set. This slice
has no text encoder, complete transformer, typed output head or training loop.

For the browser, start with [001 — Visual network](experiments/001-visual-network/README.md):
a small attention-based classifier with editable inputs, two heads, typed choices
and a real training step on the final layer. Run instructions are in that folder.

## Progress

- Initial direction captured on 13 October 2026.
- Interactive visual prototype added on 14 October 2026.
- Rust self-attention core and terminal trace example added as the next slice.
- Dataset creation and wiring Rust traces into the visual are subsequent steps.
- Actual numerical calculations, with animation timing labelled illustrative.
- No pretrained model, whole-network training, benchmark, calibration or
  integration with voyagershell claimed.
