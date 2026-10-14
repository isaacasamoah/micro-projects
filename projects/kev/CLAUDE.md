# Kev conventions

- Read README.md for the implemented scope and commands. Keep Kev inside this
  repository's `projects/kev`; work on its working branch.
- Keep model arithmetic in Rust. `attention` owns the public head and trace;
  `math` is private. Borrow inputs, express dimensions with arrays and sequence
  lengths with vectors, and return recoverable errors with `Result`.
- `example` owns the synthetic inputs and hand-set parameters shared by native
  and browser examples. They are not trained weights or task data.
- `browser` is an optional WebAssembly adapter. Validate external array shapes;
  leave finite-value and numerical checks in the core. Generate TypeScript
  declarations from the Rust exports rather than writing duplicate bindings.
- Keep authored graph examples and their label rubric in `data/`; the browser
  copies that data at build time and validates it before display. Draft labels
  are not model predictions. Keep workspace navigation separate from inference.
- The browser uses strict TypeScript and native ES modules with explicit `.js`
  imports. Keep model-boundary code, SVG rendering and interaction separate.
  Narrow DOM values at runtime; do not silence uncertainty with `any` or casts.
- Visualize returned Rust values. Animation timing is illustrative. Do not add
  a JavaScript numerical fallback or display training before Rust implements it.
- Keep controls on the graph and detailed teaching in the walkthrough. Embed
  current source during the web build so shown code stays tied to the model.
- Keep generated WebAssembly, JavaScript, dependencies and private data out of
  Git. Commit lockfiles. Use Cargo formatting and Prettier for web source.
- Check native calculations, native/Wasm agreement, invalid boundary input and
  the actual browser interaction. Record observations and limits in walkthroughs.
