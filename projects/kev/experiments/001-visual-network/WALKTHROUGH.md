# Walkthrough: an attention-based decision network

## Before → after

Before, Kev described the learning direction but had no runnable example. Now there is an interactive browser network with visible intermediate numbers and a small training operation. Rust implementation follows after we agree on the concepts and visual.

## The path through the code

`index.html` → `app.mjs` → `forward` in `model.mjs` → attention, hidden layer and decision → SVG and value tables.

The training button calls `trainOutput`, replaces the output parameters, then recomputes the forward pass. The code panel uses each imported function's own source, so the displayed implementation stays tied to what runs.

## 1. Turn the scene into numbers

Before: there was no defined example input. After: three routes each have clearance, progress and route identity.

```text
For each route:
  retain the two measured features
  append a one-hot route identity
  project the token into a key and a value
```

Read `forward`, `project` and `createModel` in [model.mjs](model.mjs). The feature values are user-controlled synthetic inputs, not observations from a robot or learned embeddings. The identity bits keep the outputs grounded in the three labels.

Judgment: **agree for this first slice**. Two inputs and three choices make the network traceable. A camera encoder or language tokenizer would add a separate learning problem too early.

## 2. Ask two different questions of the same evidence

Before: raw features had no mechanism to combine evidence selectively. After: one initial query emphasizes clearance, the other progress.

```text
For each head:
  compare its query with all keys
  scale the scores and normalize with softmax
  blend value vectors using those weights
```

Read `attend` and `softmax` in [model.mjs](model.mjs). The scale is the square root of the key dimension. Subtracting the maximum before exponentiation prevents numerical overflow without changing softmax's result.

Judgment: **agree with a scope limit**. This is query attention over tokens. A full transformer would require more machinery: token-wise queries, residual connections, normalization and additional blocks. We should introduce those as individually explained changes.

Because the initial values are one-hot route identities, the weighted sum looks exactly like the attention distribution. Learned value projections would usually make the two different; the UI exposes Wv so this is not hidden.

## 3. Produce a typed choice

```text
Join the head outputs without averaging them
Mix the six coordinates into three hidden features
Apply a bias and ReLU
Mix hidden features into one score per allowed label
Normalize scores and choose the largest
```

Read `forward` in [model.mjs](model.mjs). The matrices start with readable hand-set values. They are not claimed to be trained or optimal.

Judgment: **agree for a finite decision**. A classification head builds the permitted labels into the architecture. Generating JSON with a language model would be a different mechanism. The eventual Rust return value can be an enum; JavaScript enforces membership by indexing the fixed label list here.

The output is structurally valid, but correctness and calibration remain unproved. A tie resolves to the first label. This example has no hard safety constraint, and attention weights alone do not explain a decision causally.

## 4. Learn from one supplied label

```text
Run the scene and compare probabilities with its target
Compute the final-layer gradient
Move weights and biases opposite that gradient
Run the same scene again and compare loss
```

Read `crossEntropy` and `trainOutput` in [model.mjs](model.mjs). For cross-entropy with softmax, the logit gradient is `p − oneHot(target)`. Each weight gradient multiplies this error by its input hidden activation. The new parameter is `old − learningRate × gradient`.

Judgment: **agree as a bounded demonstration**. The analytic update is real, but it trains only the final layer. The visual labels this boundary. Training the whole network from scratch will require a separate backward pass through ReLU, projections and attention.

No dataset is implied. Repeated updates on one scene can memorize a label and damage other decisions. A lower displayed loss is not evidence of generalization or calibration.

## 5. Read the visual honestly

The six stage buttons choose a view; the model's small numerical forward pass runs immediately. Playback illustrates the route of information at a human-readable speed. All activations, attention weights, gradients and updated parameters are actual model values.

Judgment: **agree for understanding**, with illustrative timing stated clearly. Profiling CPU/GPU execution would be another experiment. The SVG is a conceptual network view, not a hardware diagram.

## What we checked

Five Node tests passed: stable softmax, a hand-computable uniform attention case, feature-specific query behavior, output-layer learning with immutable inputs, and analytic gradients matched against central finite differences.

In Chrome, equal evidence gave three probabilities of one third. One training step targeting Right reduced loss from 1.098612 to 0.920506 at learning rate 0.2. Attention stayed unchanged and the decision became Right. The reset restored initial model parameters.

These checks establish the small numerical and interaction claims. They do not establish dataset accuracy, calibrated scores, end-to-end training, or a trained transformer.

## Questions for our walkthrough

1. **What changed, and how does it work?** What happens when Left's clearance rises? The first query gives it more attention; its weighted value then flows through the hidden and decision layers. The other head initially ignores clearance because its query's clearance coefficient is zero.
2. **Why does this matter?** Why can the network never output a fourth direction? There are exactly three output scores, and the selection indexes those three labels. That guarantees shape, not that the selected route is useful or safe.
3. **Why this approach?** Why are we training only the output layer first? Its gradients can be inspected independently and checked by finite differences. Once that is intuitive, a later slice can trace gradients farther back without hiding several new concepts at once.

## Next decision

Refine the visual and the teaching task together. Then choose one Rust operation to implement, push, and explain before moving on.
