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

1. Play a forward pass or step through each stage.
2. Edit route clearance/progress. Watch attention and the decision change.
3. Edit either head's query. Negative values favor lower feature values.
4. Reset the model and choose Equal evidence: both heads and final probabilities become uniform. The tie chooses Left by enum order.
5. Choose a training target and take one step. Inspect the actual loss, gradients and weight update.
6. Read pseudocode next to the running functions in the walkthrough panel.

Reset model preserves the input scene but restores all matrices and queries. Original scene changes inputs only. Changes live in memory; reloading resets everything. There is no automatic persistence or background training.

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

The six-stage animation is replay at a readable speed. The numerical values are real; the dots are not hardware timing, a causal explanation, or an independently measured signal. The gradient arrow illustrates the actual update's scope. Inference never changes weights.

## Files to read

- [model.mjs](model.mjs): the full computation and one training update.
- [app.mjs](app.mjs): controls, network diagram, intermediate-value views and function walkthrough.
- [index.html](index.html), [style.css](style.css): reading surface.
- [WALKTHROUGH.md](WALKTHROUGH.md): before/after, pseudocode, design judgment, proof and questions.
- [model.test.mjs](model.test.mjs): numerical checks with independent cases.

## Verify

With Node 18+:

```sh
node --test model.test.mjs
```

Tests verify softmax stability, uniform attention for a zero query, feature-specific head response, closed-set output, loss decrease on one example, frozen upstream values and analytic gradients against central finite differences.

Browser checks: Chrome controls produced uniform outputs for equal evidence. One Right-target update reduced loss from 1.098612 to 0.920506 at learning rate 0.2; attention stayed unchanged. These are observations on one synthetic scene, not accuracy or calibration results.

## What typed output buys us

A finite label head cannot invent a fourth category or malformed text. It requires one forward computation rather than autoregressive decoding, but this prototype measures no performance advantage. It can still choose an unsafe or incorrect label. Three normalized scores are not automatically calibrated probabilities. More complex types need explicit architectural choices; a finite enum does not solve arbitrary structured output.

Next: refine the visual together, then implement one agreed operation in Rust, push it, and walk through that change before adding the next.
