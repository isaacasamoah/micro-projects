# 001 — Watch a decision form

**Question:** What does attention contribute to a small neural classifier, and how does a decision head differ from next-token generation?

An interactive HTML/JavaScript prototype with real, inspectable arithmetic. It is the visual sketch for Kev's eventual from-scratch Rust implementation. This is a small attention-based classifier, not a full transformer or a pretrained model.

## Run

From this directory:

```sh
python3 -m http.server 8795 --bind 127.0.0.1
```

Open http://127.0.0.1:8795. Python is only serving files; all computation happens locally in the browser. No build, package installation, external script, API key or GPU is required. Serve the folder rather than opening the HTML with `file://`, because it uses JavaScript modules.

## Explore

The graph is the interface. There are no separate control panels or value tables.

1. Move the clearance/progress sliders inside an input node. Watch attention and the decision change.
2. Move either attention head's query sliders. Negative values favor lower feature values. Set all four query coordinates to zero for uniform attention and, with the initial classifier weights, uniform decisions.
3. Hover a node or pathway for its computation and current values. Click it to keep the details open next to it. Keyboard focus, then Enter or Space, works too.
4. Click a connection into the hidden or decision layer to edit its weight. Click a hidden neuron to edit its bias.
5. Click Left, Forward or Right to choose the training target. Click Loss to change the learning rate.
6. Play inference to follow the forward flow. Play training to follow that flow, the loss, the backward gradient flow and one parameter update. Stop before the update to cancel it.
7. Open **Code ↗** for pseudocode beside the actual running functions on a separate page.

The six small A/B nodes preserve both head outputs individually. The green dot marks the current decision; the pink outline marks the training target. Thin dashed connections have zero weight, but can still be selected and edited.

Reset restores the original inputs, all parameters, target and learning rate. Changes live in memory; reloading resets everything. There is no automatic persistence or background training. On smaller screens, the graph itself scrolls.

## The network

- Three input tokens, each with five coordinates: two features and three route-identity bits.
- Shared key projection: 5 → 2. Shared value projection: 5 → 3.
- Two independent query vectors, each attending across all three tokens using scaled dot-product attention. This is query pooling, not token-to-token self-attention.
- Concatenate two three-coordinate outputs: 6 coordinates.
- Dense 6 → 3 layer, bias and ReLU.
- Dense 3 → 3 decision layer, bias, softmax, argmax.
- Output labels: `Left | Forward | Right`. The UI previews the future Rust enum; the implementation currently returns a member of a JavaScript label list.

The initial sparse projections deliberately make values route-identity vectors, so a head's output initially equals its three attention weights. That is a teaching choice, not a general property of attention. Queries, hidden weights and output weights are hand-set initially; nothing arrived pretrained.

Training implements exact cross-entropy gradients for the final 3 → 3 weights and biases only. Attention, key/value projections and the hidden layer remain frozen. Editing a query is a manual parameter change, not training. A complete from-scratch transformer, end-to-end backpropagation, dataset, optimizer schedule, calibration and generalization evaluation are later experiments.

Animation replays the computation at a readable speed. The numerical values are real; the dots are not hardware timing, a causal explanation, or an independently measured signal. Pink backward flow shows the final-layer gradient paths. Earlier layers stay frozen. Inference never changes weights.

## Files to read

- [model.mjs](model.mjs): the full computation and one training update.
- [app.mjs](app.mjs): graph construction, node/path editing and playback.
- [index.html](index.html), [style.css](style.css): full-screen graph surface.
- [walkthrough.html](walkthrough.html), [walkthrough.mjs](walkthrough.mjs): separate pseudocode and imported function source.
- [WALKTHROUGH.md](WALKTHROUGH.md): before/after, pseudocode, design judgment, proof and questions.
- [model.test.mjs](model.test.mjs): numerical checks with independent cases.

## Verify

With Node 18+:

```sh
node --test model.test.mjs
```

Tests verify softmax stability, uniform attention for a zero query, feature-specific head response, closed-set output, loss decrease on one example, frozen upstream values and analytic gradients against central finite differences.

Browser checks cover graph controls, inference without weight changes, training cancellation and the delayed parameter update. With the original scene, one Right-target update reduced loss from 1.315445 to 1.094645 at learning rate 0.2; earlier layers stayed unchanged. See [WALKTHROUGH.md](WALKTHROUGH.md) for the observed checks and their limits. These are observations on a synthetic scene, not accuracy or calibration results.

## What typed output buys us

A finite label head cannot invent a fourth category or malformed text. It requires one forward computation rather than autoregressive decoding, but this prototype measures no performance advantage. It can still choose an unsafe or incorrect label. Three normalized scores are not automatically calibrated probabilities. More complex types need explicit architectural choices; a finite enum does not solve arbitrary structured output.

Next: refine the visual together, then implement one agreed operation in Rust, push it, and walk through that change before adding the next.
