# Kev

An open-source learning project exploring system one decision models in Rust.
The aim is to understand transformers by implementing small examples, then
training, adapting and serving models locally. Kev is planned, not implemented.

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

## First session

**Question:** Can we trace one small decision from inputs to scores in Rust?

1. Choose a tiny synthetic classification task and define its expected outputs.
2. Implement a small forward pass with explicit intermediate values. This first
   slice can be a basic neural network; do not call it a transformer yet.
3. Show one worked prediction and compare it with a hand calculation.
4. Spend the final 15–20 minutes explaining what was learned and the next question.

A training loop, transformer attention, calibrated confidence, local serving and
hardware visualisation are subsequent slices, not requirements for day one.

## Running

There is no runnable code yet. Add exact setup and run commands with the first
implementation, including toolchain and hardware requirements.

## Progress

- Initial direction captured on 13 October 2026.
- Next: choose the tiny task and implement one observable forward pass.
- No model trained, benchmark run or integration with voyagershell claimed.
