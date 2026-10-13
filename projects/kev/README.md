# Kev

An open-source learning project exploring system one decision models in Rust.
The aim is to understand transformers by implementing small examples, then
training, adapting and serving models locally. The current implementation is one inspectable Rust self-attention head.
It runs natively or in the browser through WebAssembly, with a TypeScript
interactive diagram. The full model is still to come.

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
2. Implement that slice on a working branch in the canonical checkout and check it.
3. Commit and push it so the code is readable on GitHub.
4. Walk through before/after, pseudocode beside actual code, how it works,
   why this approach, what was proved and what remains uncertain.
5. Discuss and refine it before adding the next concept.

Understanding each operation is the purpose. Keep walkthroughs beside each
experiment; do not treat a working demo as proof that the learner understands it.

## Running

The model is a Rust library, not an HTTP service. The terminal example and the
browser call the same forward function. This slice has no text encoder, complete
transformer, typed output head, dataset or training loop.

With Rust 1.85 or newer (tested with 1.99.0), from this directory:

```sh
cargo run --locked --example attention
cargo run --locked --example attention -- --uniform
cargo test --locked
```

The native model has no runtime dependencies. The optional browser build adds
wasm-bindgen. See [web/README.md](web/README.md) for setup, build and serving.
The browser remains local: no account, API key or hosted model is needed.

## Code and walkthroughs

| File | Responsibility |
| --- | --- |
| [src/lib.rs](src/lib.rs) | Module boundary and optional browser feature |
| [src/attention.rs](src/attention.rs) | Self-attention head, trace and errors |
| [src/math.rs](src/math.rs) | Private projection and softmax helpers |
| [src/example.rs](src/example.rs) | Shared synthetic example |
| [src/browser.rs](src/browser.rs) | Typed WebAssembly boundary |
| [web/src/model.ts](web/src/model.ts) | Read and release Rust traces |
| [web/src/graph.ts](web/src/graph.ts) | SVG representation of those traces |
| [web/src/app.ts](web/src/app.ts) | Interaction and illustrative playback |

Start with the [Rust attention walkthrough](walkthroughs/001-rust-attention.md),
then [Rust in the browser and official patterns](walkthroughs/002-rust-browser.md).
The web app's **Code walkthrough** opens pseudocode beside the actual Rust source,
embedded at build time.

The earlier standalone JavaScript classifier has been replaced. Git history
preserves that experiment; there is now one attention implementation.

## Progress

- Initial direction captured on 13 October 2026.
- First interactive prototype and Rust attention slice added on 14 October 2026.
- Browser now displays the Rust self-attention trace, including all query rows.
- Inputs and matrices can be edited; inference does not learn from those edits.
- Training, typed decisions and dataset creation remain subsequent slices.
- No pretrained model, benchmark, calibration, GPU performance or integration
  with voyagershell claimed.
