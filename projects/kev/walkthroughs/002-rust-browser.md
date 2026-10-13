# Rust self-attention in the browser

## Before → after

The browser previously ran a separate JavaScript classifier: fixed pooling
queries, a hidden layer, and a trainable final decision layer. The Rust library
implemented self-attention but was only visible through a terminal trace.

The browser now executes that Rust library as WebAssembly. Its TypeScript graph
shows every token's projected Q, K and V, the full score matrix, row-wise attention
weights, and one value mixture per token. Selecting a token highlights its path.
The obsolete JavaScript model and its training control were removed together.

The model is still a general numerical building block. It has no graph-task
labels or text encoder. Its hand-set weights have not learned anything.

## What is Kev here?

Kev's computation is a Rust library. It can be called by a native program, as the
terminal example does. For this small visual experiment it is compiled to
WebAssembly and called locally by the browser. TypeScript owns interaction and
rendering. There is no HTTP model service, remote inference or API key.

That is our design choice, not a mandate from either language's handbook.
It gives this example one implementation without adding a server. Future native
GPU execution, hosted inference or an API can be evaluated when a real model
needs them. This browser build makes no GPU-performance claim.

## Official guidance → concrete changes

| Source | Pattern applied | Where to read it |
| --- | --- | --- |
| [Rust Book: modules](https://doc.rust-lang.org/book/ch07-02-defining-modules-to-control-scope-and-privacy.html) | Keep implementation details private and expose a small public interface. | [lib.rs](../src/lib.rs), [math.rs](../src/math.rs) |
| [Rust Book: borrowing](https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html) | Read borrowed inputs and parameters without transferring ownership. | `AttentionHead::forward(&self, inputs: &[…])` in [attention.rs](../src/attention.rs) |
| [Rust Book: Result](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html) | Represent recoverable failures in the return type and propagate with `?`. | Core errors and [browser.rs](../src/browser.rs) array validation |
| [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html), [strict](https://www.typescriptlang.org/tsconfig/strict.html) | Use static checking before execution, with strict null and type checks. | [tsconfig.json](../web/tsconfig.json) |
| [TypeScript: narrowing](https://www.typescriptlang.org/docs/handbook/2/narrowing.html) | Narrow DOM objects with `instanceof`; use a discriminated union for playback. | [app.ts](../web/src/app.ts) |
| [TypeScript: compiler options](https://www.typescriptlang.org/docs/handbook/modules/guides/choosing-compiler-options.html) | Match module configuration to the runtime. Browser ESM uses explicit `.js` imports, package `type: module`, and NodeNext checking. | [package.json](../web/package.json), [tsconfig.json](../web/tsconfig.json) |
| [Rust's WebAssembly page](https://rust-lang.org/what/wasm/), [wasm-bindgen without a bundler](https://wasm-bindgen.github.io/wasm-bindgen/examples/without-a-bundler.html) | Generate browser ES-module bindings and initialize Wasm before calling exports. | [build.mjs](../web/build.mjs), [model.ts](../web/src/model.ts) |
| [wasm-bindgen exported Rust types](https://wasm-bindgen.github.io/wasm-bindgen/reference/types/exported-rust-types.html) | Export a concrete Rust trace with generated TypeScript declarations. Release it after copying arrays. | `BrowserTrace` and `infer` |

`noUncheckedIndexedAccess` also makes missing array coordinates explicit. The
`ESNext.Disposable` declaration library is included because generated bindings
expose `Symbol.dispose`; the application explicitly calls `free()` in `finally`.
It does not rely on a JavaScript cast or `any` to hide that boundary.

The handbooks do not prescribe this folder structure or a frontend framework.
We kept the browser as native ES modules with three responsibilities: the Rust
boundary, graph rendering, and interaction. Source conventions live in
[CLAUDE.md](../CLAUDE.md). Cargo and npm lockfiles pin the resolved dependencies.

## Walk the code

The browser's **Code walkthrough** places explanation beside actual Rust source.
The build embeds the current files, so the displayed code follows the build.

| Pseudocode | Actual implementation |
| --- | --- |
| Validate flat browser arrays and turn rows into fixed-width Rust arrays. | `rows` and `matrix` in [browser.rs](../src/browser.rs) |
| Run exactly the native attention calculation. | `head.forward(&rows::<2>(inputs)?)?` |
| Copy each returned trace array, then release its Rust owner. | `infer` / `finally { result.free(); }` in [model.ts](../web/src/model.ts) |
| Build node labels and paths from returned numbers. | `renderGraph` in [graph.ts](../web/src/graph.ts) |
| Try an edited parameter on a candidate copy; keep it only if Rust accepts it. | `updateParameter` in [app.ts](../web/src/app.ts) |
| Compute once, then replay four explanation stages while weights stay fixed. | The inference button and `Playback` union in [app.ts](../web/src/app.ts) |

Three tokens produce three queries in **one head**. Each score row is one query
against every key, including its own. Softmax normalizes within that row. Each
output is the weighted sum of the value vectors, not a distribution over actions.

The native core permits different token counts and compile-time feature widths.
This visual deliberately fixes three tokens, input/key widths of two, and a value
width of three. The browser adapter fixes those feature widths but accepts a
variable token count. All tokens use the same projection matrices.

## A small experiment

1. Follow token 0. The output starts near `[0.802224, 0.598888, 0.203336]`.
2. Follow token 1. Its query and attention row differ despite sharing Wq/Wk/Wv.
3. Set all Wq cells to zero. Scores become zero, every weight becomes one third,
   and every output is `[2/3, 2/3, 0]`.
4. Edit Wv. Attention probabilities stay the same, but the mixed output changes.
5. Play inference. It explains the same calculation without updating weights.

**Judgment:** this is a suitable next learning slice. It removes the mismatch
between what the visual teaches and what Rust implements. The tradeoff is that
the old classifier's training demonstration is gone until we implement training
in Rust. We have not silently substituted manual matrix edits for backpropagation.

## Verification

From `projects/kev`:

```sh
cargo fmt --check
cargo test --locked --all-features
cargo clippy --locked --all-targets --all-features -- -D warnings
cd web
npm run build
npm run check
npm test
npm run format:check
```

Native tests cover independent numerical answers, stable softmax, shape types,
uniform attention, input permutation, frozen parameters and invalid values.
The WebAssembly tests use the built artifact and actual frontend boundary. They
compare all score, attention and output rows against the native terminal example
in normal and uniform cases. Native output is printed to six decimals, so that
comparison allows its rounding; the independent two-token check uses 1e-12.

On 14 October 2026, seven numerical/boundary tests and one compile-fail doctest
passed, plus six WebAssembly/boundary tests. Strict TypeScript, formatting and
Clippy checks passed.

Chrome at `http://127.0.0.1:8795/`, controlled through Claude Has Hands, loaded
`generated/kev_bg.wasm`. Eighteen scripted interaction assertions passed:

- Edited input values changed the Rust output without replacing the slider DOM.
- Keyboard selection followed another query and retained focus after redraw.
- All three matrix editors opened; zeroing Wq produced three mean-value outputs.
- Invalid matrix input was restored. Wv edits affected the output, and its
  inspector showed the actual weighted sum.
- Reset restored the example and cancelled a pending animation timer.
- The four playback stages and completion were observed. Playback did not change
  the original outputs. The application console contained no errors.

A screenshot was inspected for layout and readable values. Interaction checks
used DOM mouse/keyboard/input events in the real Chrome page; they do not prove
physical pointer dragging or screen-reader behavior. Those remain useful things
to try by hand. Animation is explanatory, not GPU or execution-time telemetry.

This proves a locally served, Rust-powered forward-pass visual. No full
transformer, decision head, training, hosted deployment or GPU execution is claimed.
