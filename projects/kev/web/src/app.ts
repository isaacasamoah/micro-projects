import { initialize, initialParameters, infer, at, row } from "./model.js";
import type { MatrixName, Parameters, Trace } from "./model.js";
import {
  format,
  inputMarkup,
  renderGraph,
  tokens,
  stageDurationMs,
} from "./graph.js";
import type { Stage, Token } from "./graph.js";

function element(id: string): HTMLElement {
  const found = document.getElementById(id);
  if (!(found instanceof HTMLElement))
    throw new Error(`Missing HTML element: ${id}`);
  return found;
}
const graph = document.getElementById("graph");
const play = element("play");
const reset = element("reset");
const status = element("status");
const inspector = element("inspector");
const details = element("details");
if (
  !(graph instanceof SVGSVGElement) ||
  !(play instanceof HTMLButtonElement) ||
  !(reset instanceof HTMLButtonElement) ||
  !(inspector instanceof HTMLDialogElement)
) {
  throw new Error("Missing graph controls");
}
const svg = graph;
const playButton = play;
const resetButton = reset;
const dialog = inspector;

type Playback =
  { kind: "idle" } | { kind: "playing"; stage: Stage; timer: number };
let playback: Playback = { kind: "idle" };
let selected: Token = 0;
let parameters: Parameters;
let trace: Trace;
const stageText: Record<Stage, string> = {
  projection: "Shared matrices project each input into Q, K and V.",
  comparison:
    "Every query meets every key, including its own. Scores are scaled by √2.",
  normalization: "Softmax turns each row of scores into attention weights.",
  blend: "Each query’s weights blend the values into its own output vector.",
};

function render(): void {
  const focusedId =
    document.activeElement instanceof SVGElement
      ? document.activeElement.id
      : null;
  const drawing = renderGraph(
    trace,
    selected,
    playback.kind === "playing" ? playback.stage : null,
  );
  const edges = svg.querySelector("#edges");
  const nodes = svg.querySelector("#nodes");
  if (!edges || !nodes) throw new Error("Missing graph layers");
  edges.innerHTML = drawing.edges;
  nodes.innerHTML = drawing.nodes;
  edges.querySelectorAll("animateMotion").forEach((animation) => {
    if (animation instanceof SVGAnimationElement) animation.beginElement();
  });
  const focused = focusedId ? document.getElementById(focusedId) : null;
  if (focused instanceof SVGElement) focused.focus();
  for (const token of tokens)
    svg
      .querySelector(`#input-${token}`)
      ?.classList.toggle("selected", token === selected);
  svg.querySelectorAll<HTMLInputElement>("[data-input]").forEach((input) => {
    const index = Number(input.dataset.input);
    input.value = String(at(parameters.inputs, index));
    const label = svg.querySelector(`[data-input-value="${index}"]`);
    if (label) label.textContent = format(at(parameters.inputs, index));
  });
}

function stop(): void {
  if (playback.kind === "playing") window.clearTimeout(playback.timer);
  playback = { kind: "idle" };
  playButton.textContent = "Play inference";
}

function recompute(): void {
  stop();
  trace = infer(parameters);
  status.textContent =
    "Rust computed all three outputs. Select a token or change an input.";
  render();
}

function updateParameter(
  kind: keyof Parameters,
  index: number,
  value: number,
): void {
  stop();
  try {
    if (!Number.isFinite(value)) throw new Error("Enter a finite value.");
    const candidate = { ...parameters, [kind]: parameters[kind].slice() };
    candidate[kind][index] = value;
    const nextTrace = infer(candidate);
    parameters = candidate;
    trace = nextTrace;
    status.textContent =
      "Rust recomputed all three outputs with the edited value.";
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : String(error);
  }
  render();
}

function showDetails(title: string, explanation: string, values: string): void {
  details.replaceChildren();
  const heading = document.createElement("h2");
  heading.id = "inspector-title";
  heading.textContent = title;
  const paragraph = document.createElement("p");
  paragraph.textContent = explanation;
  const pre = document.createElement("pre");
  pre.textContent = values;
  details.append(heading, paragraph, pre);
  if (!dialog.open) dialog.show();
}

function inspectMatrix(kind: MatrixName): void {
  stop();
  render();
  const width = kind === "value" ? 3 : 2;
  showDetails(
    `Shared W${kind[0]}`,
    `Two input coordinates map to ${width} ${kind} coordinates. Rows are input features; columns are output features. Every token uses this matrix. Editing it is a manual parameter change, not a training step.`,
    "",
  );
  const fields = document.createElement("div");
  fields.className = "matrix-fields";
  fields.style.gridTemplateColumns = `repeat(${width}, 1fr)`;
  parameters[kind].forEach((value, index) => {
    const label = document.createElement("label");
    label.textContent = `[${Math.floor(index / width)}, ${index % width}]`;
    const input = document.createElement("input");
    input.type = "number";
    input.step = "0.1";
    input.value = String(value);
    input.setAttribute(
      "aria-label",
      `${kind} weight ${Math.floor(index / width)}, ${index % width}`,
    );
    input.addEventListener("change", () => {
      updateParameter(kind, index, input.valueAsNumber);
      input.value = String(at(parameters[kind], index));
    });
    label.append(input);
    fields.append(label);
  });
  details.append(fields);
}

function follow(token: Token): void {
  stop();
  selected = token;
  dialog.close();
  status.textContent = `Following query ${token}: compare with every key, then blend every value.`;
  render();
}

function activate(event: Event): void {
  if (!(event.target instanceof Element)) return;
  const target = event.target.closest(
    "[data-matrix], [data-token], [data-output], [data-inspect]",
  );
  if (!target) return;
  const matrix = target.getAttribute("data-matrix");
  if (matrix === "query" || matrix === "key" || matrix === "value") {
    inspectMatrix(matrix);
    return;
  }
  if (target.hasAttribute("data-inspect")) {
    showDetails(
      "Connection",
      target.querySelector("title")?.textContent ?? "",
      "Numbers come from Rust; moving dots illustrate the computation order.",
    );
    return;
  }
  const token = Number(
    target.getAttribute("data-token") ?? target.getAttribute("data-output"),
  );
  if (token !== 0 && token !== 1 && token !== 2) return;
  const derived = target.getAttribute("data-derived");
  follow(token);
  if (derived === "key" || derived === "value") {
    const field = derived === "key" ? trace.keys : trace.values;
    const width = derived === "key" ? 2 : 3;
    showDetails(
      `${derived} ${token}`,
      derived === "key"
        ? "Every query compares with this key. It is projected from this token’s input using the shared key matrix."
        : "This vector carries the content that attention mixes. Scores compare keys; the output combines values.",
      JSON.stringify(row(field, token, width)),
    );
  } else if (target.hasAttribute("data-output")) {
    const terms = tokens.map(
      (v) =>
        `${format(at(trace.weights, token * 3 + v))} × [${row(trace.values, v, 3).map(format).join(", ")}]`,
    );
    showDetails(
      `Output ${token}`,
      "Sum these three weighted vectors coordinate by coordinate. This attention output is not yet a typed decision. There is no classifier or training loop in this Rust slice.",
      terms.join("\n+ ") +
        `\n= [${row(trace.output, token, 3).map(format).join(", ")}]`,
    );
  }
}

svg.addEventListener("click", activate);
svg.addEventListener("keydown", (event) => {
  if (
    (event.key === "Enter" || event.key === " ") &&
    event.target instanceof SVGElement &&
    event.target.matches('[role="button"]')
  ) {
    event.preventDefault();
    activate(event);
  }
});
svg.addEventListener("input", (event) => {
  if (
    !(event.target instanceof HTMLInputElement) ||
    !event.target.hasAttribute("data-input")
  )
    return;
  const index = Number(event.target.dataset.input);
  dialog.close();
  updateParameter("inputs", index, event.target.valueAsNumber);
});
element("close").addEventListener("click", () => dialog.close());
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") dialog.close();
});

playButton.addEventListener("click", () => {
  if (playback.kind === "playing") {
    stop();
    status.textContent = "Playback stopped. Parameters have not been trained.";
    render();
    return;
  }
  dialog.close();
  trace = infer(parameters);
  const stages: Stage[] = [
    "projection",
    "comparison",
    "normalization",
    "blend",
  ];
  function advance(index: number): void {
    const stage = stages[index];
    if (!stage) {
      stop();
      status.textContent =
        "Inference complete. Weights stayed fixed; all three output vectors were computed.";
    } else {
      playback = {
        kind: "playing",
        stage,
        timer: window.setTimeout(() => advance(index + 1), stageDurationMs),
      };
      playButton.textContent = "Stop";
      status.textContent = stageText[stage];
    }
    render();
  }
  advance(0);
});
resetButton.addEventListener("click", () => {
  parameters = initialParameters();
  selected = 0;
  dialog.close();
  recompute();
});

try {
  await initialize();
  parameters = initialParameters();
  svg.innerHTML = `<g id="edges"></g><g id="inputs">${inputMarkup(parameters)}</g><g id="nodes"></g>`;
  recompute();
  playButton.disabled = false;
  resetButton.disabled = false;
} catch (error) {
  status.textContent = `Could not load Kev: ${error instanceof Error ? error.message : String(error)}. Build the web app and serve it over HTTP.`;
}
