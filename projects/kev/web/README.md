# Kev in the browser

A local interactive view of the Rust self-attention head. Three synthetic tokens,
two features each, one attention head. Every query compares with every key;
softmax weights mix value vectors into three outputs. These are not decision
probabilities, and there is no training in this slice.

## Setup

Install Rust, Node.js and Python 3. The checked environment uses Rust 1.99.0 and
Node 24.13.1. Add Rust's WebAssembly target and the binding generator once:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked
```

From this folder:

```sh
npm ci
npm run build
npm run serve
```

Open http://127.0.0.1:8795/ . Serve over HTTP; opening index.html as a file will
not load WebAssembly correctly. `npm run build` compiles Rust, generates its
JavaScript/TypeScript bindings, compiles TypeScript and copies the static site
into ignored `dist/`. The CLI version must match Cargo.toml's wasm-bindgen version.
The build resolves Rust tools from CARGO_HOME or the usual ~/.cargo/bin directory.

No bundler, frontend framework, HTTP model API or remote inference is required.
The static files could later be hosted, with computation still on the reader's
machine. This does not establish a deployment approach for future GPU models.

## Workspace

Model is the centre. Move near the left, right or top edge and click the revealed
arrow to slide into Data, Evaluation or Capabilities. The reverse arrow returns
to Model. Tab reveals navigation controls; touch layouts show them continuously.
Reduced motion disables sliding. Leaving Model stops playback and closes details
without changing its parameters.

Data browses the canonical `../data/graph-decisions.seed.json` copied during
build. Labels are authored proposals, with no train/test assignment yet. The
other panels are experiment outlines, not fabricated results or working harnesses.
Model values remain the actual Rust trace, independent of the selected data case.

## Explore

- Drag input sliders: all Q/K/V vectors and attention rows are recomputed in Rust.
- Click a query, matrix row or output to follow that token's path.
- Click Wq, Wk or Wv to edit shared projection weights. Each edit is committed
  when the field changes; invalid values leave the model unchanged.
- Hover connections for their meaning, or activate them for details.
- Click an output to inspect its weighted sum.
- Play inference to replay projection, comparison, softmax and value blending.
  Each stage plays once, with motion and stage timing kept together.
  The calculation already happened; animation duration is illustrative.
- Stop, reset or editing cancels playback. Reset restores Rust's example values.
- Keyboard users can Tab to controls and use Enter or Space on graph nodes.
  Reduced-motion settings hide moving particles.

Try zeroing Wq: every weight becomes one third and every output becomes the mean
value vector. Then change Wv to see why values and attention weights differ.

## Check

After building:

```sh
npm run check
npm test
npm run format:check
```

Tests load the built WebAssembly and the actual TypeScript boundary. They compare
against the native Rust example, an independent two-token answer, changed inputs,
invalid shapes/numbers and trace lifetime after releasing Rust allocations.

The browser interaction checks and their limits are in
[walkthrough 002](../walkthroughs/002-rust-browser.md). `public/walkthrough.html`
is a teaching template; the build embeds the current Rust source into the page.
