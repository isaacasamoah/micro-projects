# Working on micro-projects

- Keep each project self-contained under `projects/`; read its README first.
- Inspect dirty state and preserve work by others. Use an owned branch and
  separate worktree when an experiment changes an existing project.
- Keep slices small enough to explain and verify. No shared framework until
  an actual experiment needs it.
- Put setup and run instructions beside runnable code. State when code is only
  planned; do not claim demos, training or evaluation that did not happen.
- Visualisations must distinguish measured execution from illustrative animation.
- Check behavior with a meaningful example and a case that would expose a mistake.
- Keep credentials, private traces, datasets, model weights and generated build
  artifacts out of Git. Credit sources and check licenses before importing work.
- Document actual learning and limits. Use plain comments only where they explain
  a non-obvious reason; let readable code explain the mechanics.
