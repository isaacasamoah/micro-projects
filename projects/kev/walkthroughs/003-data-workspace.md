# A workspace for investigating decisions

## Before and after

Before, the browser showed one Rust self-attention computation. Now that same
model sits at the centre of four spatial views: Data to the left, Evaluation to
the right, and Capabilities above. The latter two describe planned experiments;
they do not report model results. Returning to Model preserves edited inputs.

Data presents 12 authored graph-decision examples. Each pairs an incoming event
with candidate nodes, a proposed knowledge type, relationships and a rationale.
These text examples do not yet pass through the numerical Rust model.

## Navigation: pseudocode beside the source

See [workspace.ts](../web/src/workspace.ts) for the implementation.

```text
assign each view a position relative to Model
when the pointer approaches an edge, reveal that direction's label
when its button is clicked:
    stop illustrative playback and close the model inspector
    translate the shared plane to the destination
    make only the destination interactive and move focus into it
    offer the route back to Model
```

The view positions and routes define the allowed navigation. CSS animates one
plane instead of rebuilding the model, so its state survives navigation.
Inactive panels are inert and hidden from the accessibility tree. Native buttons
remain keyboard reachable; touch devices show controls without requiring hover.
Reduced-motion settings disable the slide animation.

## Data: pseudocode beside the source

See [dataset.ts](../web/src/dataset.ts) for validation and
[data.ts](../web/src/data.ts) for the browser display.

```text
load the canonical authored JSON
validate its schema, labels, identifiers and cross-references
show an example selector
for the selected example:
    show the event and proposed knowledge type
    show each candidate and its proposed incoming-to-candidate relation
    distinguish unresolved candidates from no connection
    show the author's rationale
```

The [dataset rubric](../data/README.md) explains the choices. Membership in a
project or ticket is metadata. Semantic relations are supports, refines,
supersedes, contradicts and depends_on. A contradiction alone does not establish
which statement replaces the other. Different people or scopes can have different
preferences without a conflict.

## Why this shape

Navigation is separate from inference. Authored labels live in JSON with their
rubric, rather than in rendering code. Runtime validation catches malformed data;
it does not prove that our proposed labels are correct. We should debate those
labels before turning the draft into training examples.

## Verification and limits

`npm run build`, `npm run check`, `npm test` and `npm run format:check` passed
from `web/`. Eight tests cover the existing Rust/Wasm boundary and the new dataset
parser, including invalid references, duplicate IDs and conflicting assignments.

In Chrome, all four views and return routes were exercised. The observed plane
translations matched their directions. Leaving Model stopped playback; returning
preserved an edited input. The Data selector displayed the ambiguous-scope case
as needing clarification. Focus moved into the destination and a pointer near
the left edge revealed its control. The Data layout was visually inspected.

This is an interactive data discussion surface, not a training or evaluation
pipeline. No model weights changed. Mobile and screen-reader behaviour still
need direct user testing. All 12 examples have been inspected during development;
they must not later be described as a pristine held-out test set.

## Questions for the next step

- Are the knowledge types clear enough to label consistently?
- Is one semantic relation per event/candidate pair sufficient for the first task?
- Which situations should require clarification rather than a guessed link?
- How should we split whole scenario families before generating more examples?
