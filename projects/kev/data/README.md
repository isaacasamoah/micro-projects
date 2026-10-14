# First graph-decision dataset

[graph-decisions.seed.json](graph-decisions.seed.json) contains 12 original,
synthetic, author-labelled examples for discussing our first decision task.
They are **draft labels for review**, not real agent traces, a benchmark, or a
training run. Kev currently implements numerical self-attention; it cannot yet
encode this text or predict these decisions. The browser's Data view displays
these proposed answers without calling them model outputs.

## Three questions

1. What kind of knowledge is the incoming statement?
2. Which supplied candidates does it connect to, if any?
3. What is the directed relationship from the incoming statement to each candidate?

The JSON's `knowledge_types` and `relations` dictionaries are the label rubric's
one home. Hover a displayed type or relationship to read its definition.
An example contains one atomic incoming statement, candidate nodes, a ticket
anchor, and an expected answer with a rationale. IDs are local to an example.

The three knowledge types describe an assertion's purpose: domain facts and
contracts, operational instructions and procedures, or stated preferences.
A factual test result is domain evidence in this first rubric; an instruction
for running a test is operational. This boundary is a proposal to review.
Mixed events should first be separated into assertions; that extraction task
is outside this seed. Decisions and discussions can contain any of these types.

## Connection rules

All semantic edges point **from incoming statement to existing candidate**.
We propose at most one relationship per pair in this first task. If several fit,
choose explicit replacement first, then scoped refinement, then support; a
conflict without replacement evidence is contradiction. A prerequisite uses
`depends_on`. These conventions need review before we treat labels as truth.

Candidates absent from both `links` and `unresolved` have no supported connection
in the supplied context. `unresolved` means evidence is missing, not a negative
label. Multiple candidates can receive edges; an empty candidate list is valid.
There are no invented numerical confidence targets.

Ticket/project ownership, authorship, sources, event IDs and timestamps belong
in supplied metadata where known. Do not ask a model to guess them. The seed's
ticket anchors are synthetic; they are not real issues or evidence of authority.
Superseding requires replacement intent, matching scope and authority stated in
the example. Later timestamps alone must not delete or replace knowledge.

## How to develop this into training data

Review the labels together first. For each example, ask whether the answer is
supported, whether another answer is defensible, and what evidence is missing.
Keep disputed examples visible rather than forcing agreement.

Then construct more independent scenario families. Assign whole families to
training, development or held-out evaluation **before** generating paraphrases.
The `family` field groups related examples; all current examples have an
unassigned split and are visible during design, so they are not a pristine test
set. New evaluation families must remain separate from training and repeated
manual tuning.

Start with supervised labels and measure knowledge-type accuracy, selected-edge
precision/recall, relationship errors, appropriate deferral and calibration once
we have probabilistic predictions. An RL environment would additionally need
state transitions, an action space, rewards and episode-level evaluation; this
file does not provide those.

## Check

The browser validates labels, IDs, referenced candidates and conflicting target
assignments before rendering. Tests cover valid multi-link/no-link/uncertain
cases and invalid references. From `../web`, run `npm run build && npm test`.
Structural checks establish file consistency, not correctness of the labels.
