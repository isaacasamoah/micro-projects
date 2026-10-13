# First Rust slice: follow one self-attention head

## Before → after

Before, Kev's runnable network lived in the JavaScript visual experiment. Its two queries were hand-set vectors that pooled the input routes.

Now Kev also has a small Rust library and a runnable terminal example. Every input vector produces its own query, key and value through shared projection matrices. Each query attends across the whole sequence. The function returns the actual queries, keys, values, scaled scores, attention weights and output vectors.

This is one self-attention building block, not yet the complete transformer or typed decision model. The browser still runs its existing JavaScript example. No backend connection or dataset has been added.

## Where the computation lives

[`examples/attention.rs`](../examples/attention.rs) supplies synthetic numeric vectors and hand-set matrices → [`AttentionHead::forward`](../src/attention.rs) projects inputs, computes attention and returns an `AttentionTrace` → the example prints that trace.

The library contains no route labels, graph categories or graph-specific rules. `INPUT`, `KEY` and `VALUE` are numerical feature dimensions. They do not specify how many answer choices a future decision task may offer. Sequence length can vary between calls.

## Pseudocode beside the code

Read [attention.rs](../src/attention.rs) in order. These expressions come from the implementation; the full functions include shape types, finite-number checks and trace construction.

| What happens | Rust expression |
| --- | --- |
| Combine input coordinates using one column of a weight matrix. | `input.iter().zip(weights).map(|(x, row)| x * row[column]).sum()` |
| Make a query for each input token. Repeat with separate matrices for keys and values. | `inputs.iter().map(|x| project(x, &self.query_weights)).collect()` |
| Compare a query with a key, scaled by the square root of the key width. | `query.iter().zip(key).map(|(q, k)| q * k).sum::<f64>() / scale` |
| Normalize each query's scores across the tokens it can attend to. | `scores.iter().map(|row| softmax(row)).collect::<Result<_, _>>()?` |
| Blend each value coordinate using the attention weights. | `row.iter().zip(&values).map(|(weight, value)| weight * value[column]).sum()` |

`softmax` subtracts the largest score before exponentiation, then divides each exponential by their sum. The shift preserves the distribution while avoiding overflow from exponentiating a large positive score.

Rust's array types connect the dimensions: `[f64; INPUT]` can be multiplied by `[[f64; OUTPUT]; INPUT]`. A mismatched input width cannot be passed to this function. A `Vec` holds the variable number of tokens. `Result` reports an empty sequence, zero feature width or non-finite calculation to the caller.

## A computation to follow by hand

Run from the Kev directory:

```sh
cargo run --offline --example attention
```

The inputs are `[1, 0]`, `[0, 1]` and `[1, 1]`. These are synthetic numerical tokens, not tokenized words. Query and key matrices are identity matrices. The value matrix maps them to `[1, 0, 1]`, `[0, 1, -1]` and `[1, 1, 0]`.

For token 0, the Rust run produced:

```text
query:   [1, 0]
scores:  [0.707107, 0.000000, 0.707107]
weights: [0.401112, 0.197776, 0.401112]
output:  [0.802224, 0.598888, 0.203336]
```

The output is a blend of value vectors. It is not a probability distribution over decisions. Attention weights sum to one, but the resulting output coordinates do not have to sum to one and can be negative.

Now run the deliberately different case:

```sh
cargo run --offline --example attention -- --uniform
```

This zeros the query matrix. Every score becomes zero, each attention weight becomes one third, and every output is the mean value vector: `[0.666667, 0.666667, 0]`. Nothing was trained; we deliberately changed a parameter to expose its effect.

## Why start here?

**Judgment: agree for this first learning slice.** Explicit CPU arithmetic and small dimensions make the computation inspectable. No tensor framework, accelerator, model download or runtime dependency hides the operations.

The trace is returned by the same function that performs the calculation. A later visualization can read those values directly. Today there is no serialization protocol, server or browser integration; defining those before we understand the numerical component would add another problem to this slice.

This implements unmasked attention: every input can attend to every other input. There is no positional encoding yet, so reordering inputs simply reorders the corresponding outputs. It does not understand word order or text. Multi-head composition, residuals, normalization, feed-forward layers, input encoding, typed decision heads and training are still to come.

The mathematical reference is [Attention Is All You Need, section 3.2.1](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1). The Rust is a small implementation of the attention formula, not a reproduction of the paper's complete model or results.

## What was checked

Run from the Kev directory:

```sh
cargo fmt --check
cargo test --offline
cargo clippy --offline --all-targets -- -D warnings
```

On Rust 1.99.0, seven numerical/boundary tests and one compile-fail documentation test passed. Clippy reported no warnings. Both terminal examples ran successfully.

The checks include a hand-computed two-token answer, zero-query mean values, one-token behavior, softmax shift invariance, a non-square projection, permutation behavior, unchanged parameters during inference, and invalid input rejection. The compile-fail case checks that incompatible input widths are rejected by Rust.

These checks establish the calculations and terminal example. They do not establish learned ability, calibration, speed, GPU behavior or browser integration.

## Questions to walk through together

1. **What changed, and how does it work?** Where does each token's query come from now? Its own input vector is projected using the shared query matrix. The earlier visual's pooling queries were separate, hand-set vectors.
2. **Why does it matter?** Why are attention weights different from outputs? The weights decide how much of each value vector to blend. The non-identity value matrix makes that distinction visible in the printed numbers.
3. **Why this approach?** Why implement one head before the whole model? It gives us a hand-checkable unit whose trace we understand. We can then compose multiple heads and explain exactly what the added operation contributes.
