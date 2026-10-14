import { at, row } from "./model.js";
import type { Parameters, Trace } from "./model.js";

export const stageDurationMs = 1800;
export const tokens = [0, 1, 2] as const;
export type Token = (typeof tokens)[number];
export type Stage = "projection" | "comparison" | "normalization" | "blend";
const colors = {
  query: "#86b9ff",
  key: "#f2bc77",
  value: "#e8a3e6",
  output: "#81e2be",
};
export const format = (value: number) =>
  (Math.abs(value) < 0.0005 ? 0 : value).toFixed(3);
const vector = (values: number[]) => `[${values.map(format).join(", ")}]`;
const center = (token: number) => 220 + token * 205;
const cellY = (token: number) => 362 + token * 64;
const derivedY = (token: number, kind: "query" | "key" | "value") =>
  center(token) + { query: -60, key: 0, value: 60 }[kind];

function text(x: number, y: number, content: string, className = ""): string {
  return `<text x="${x}" y="${y}" class="${className}">${content}</text>`;
}

function edge(
  id: string,
  path: string,
  label: string,
  color: string,
  active: boolean,
  animate: boolean,
  width = 1.4,
): string {
  return `<g id="${id}-edge" class="edge ${active ? "active" : ""}" tabindex="0" role="button" data-inspect="edge" aria-label="${label}">
    <title>${label}</title><path id="${id}" class="wire" d="${path}" stroke="${color}" stroke-width="${width}"/>
    <path class="hit" d="${path}"/>
    ${animate ? `<circle r="3" fill="${color}" class="particle"><animateMotion begin="indefinite" dur="${stageDurationMs}ms" repeatCount="1" fill="freeze" path="${path}"/></circle>` : ""}
  </g>`;
}

export function inputMarkup(parameters: Parameters): string {
  return tokens
    .map(
      (
        token,
      ) => `<g id="input-${token}" class="input-node" transform="translate(28 ${center(token) - 64})">
    <rect class="shape" width="180" height="128" rx="13"/>
    <foreignObject width="180" height="128"><div xmlns="http://www.w3.org/1999/xhtml" class="input-card">
      <button class="token-button" data-token="${token}" aria-label="Follow token ${token}">Token ${token}</button>
      ${[0, 1].map((feature) => `<label>x${feature}<input aria-label="Token ${token} feature ${feature}" type="range" min="-2" max="2" step="0.05" data-input="${token * 2 + feature}" value="${at(parameters.inputs, token * 2 + feature)}"/><output data-input-value="${token * 2 + feature}">${format(at(parameters.inputs, token * 2 + feature))}</output></label>`).join("")}
    </div></foreignObject>
  </g>`,
    )
    .join("");
}

export function renderGraph(
  trace: Trace,
  selected: Token,
  stage: Stage | null,
): { edges: string; nodes: string } {
  let edges = "";
  let nodes =
    text(28, 44, "01 · SYNTHETIC INPUTS", "caption") +
    text(268, 44, "02 · SHARED PROJECTIONS", "caption") +
    text(595, 260, "03 · COMPARE Q WITH EVERY K", "caption") +
    text(856, 260, "04 · NORMALIZE EACH ROW", "caption") +
    text(1190, 44, "05 · MIX THE VALUES", "caption");

  (["query", "key", "value"] as const).forEach((kind, index) => {
    const x = 266 + index * 66;
    nodes += `<g id="matrix-${kind}" class="node matrix-button" tabindex="0" role="button" data-matrix="${kind}" aria-label="Edit shared ${kind} matrix" transform="translate(${x} 67)">
      <title>Shared W${kind[0]}: every token uses this same matrix. Click to edit parameters; this is not training.</title>
      <rect class="shape" width="59" height="38" rx="8" style="stroke:${colors[kind]}"/>
      ${text(12, 25, `W${kind[0]}`, "matrix-label")}
    </g>`;
  });
  nodes += text(
    266,
    126,
    "Click W to edit · click Q to follow a token",
    "tiny",
  );

  tokens.forEach((token) => {
    (["query", "key", "value"] as const).forEach((kind) => {
      const y = derivedY(token, kind);
      const traceKey = (
        { query: "queries", key: "keys", value: "values" } as const
      )[kind];
      const width = kind === "value" ? 3 : 2;
      const label = `${kind.toUpperCase()[0]}${token} = x${token} × W${kind[0]}`;
      const active = token === selected && stage === "projection";
      edges += edge(
        `project-${kind}-${token}`,
        `M208 ${center(token)} C234 ${center(token)} 235 ${y} 266 ${y}`,
        label,
        colors[kind],
        active,
        active,
      );
      nodes += `<g id="${kind}-${token}" class="node ${kind === "query" && token === selected ? "selected" : ""} ${active ? "lit" : ""}" tabindex="0" role="button" data-derived="${kind}" data-token="${token}" aria-label="${label}" transform="translate(266 ${y - 22})">
        <title>${label}. Derived from the input, not a separate editable query.</title>
        <rect class="shape" width="196" height="44" rx="9"/>
        ${text(12, 18, `${kind.toUpperCase()[0]}${token}`, `small ${kind}`)}
        ${text(12, 35, vector(row(trace[traceKey], token, width)), "numbers small")}
      </g>`;
    });
    const qy = derivedY(token, "query");
    edges += edge(
      `query-row-${token}`,
      `M462 ${qy} C525 ${qy} 528 ${cellY(token)} 592 ${cellY(token)}`,
      `Query ${token} is compared with keys 0, 1 and 2 to produce row ${token}.`,
      colors.query,
      token === selected,
      stage === "comparison" && token === selected,
    );
    const ky = derivedY(token, "key");
    const keyX = 628 + token * 63;
    edges += edge(
      `key-column-${token}`,
      `M462 ${ky} C508 ${ky} 502 278 ${keyX} 278 L${keyX} 309`,
      `Key ${token} contributes to column ${token}: every query compares against it.`,
      colors.key,
      stage === "comparison",
      stage === "comparison",
    );
    edges += edge(
      `normalize-${token}`,
      `M795 ${cellY(token)} L850 ${cellY(token)}`,
      `Softmax normalizes query ${token}'s three scaled scores together.`,
      colors.output,
      token === selected,
      stage === "normalization" && token === selected,
    );
    edges += edge(
      `attend-${token}`,
      `M1053 ${cellY(token)} C1110 ${cellY(token)} 1115 ${center(token)} 1186 ${center(token)}`,
      `Attention row ${token} weights the value vectors to produce output ${token}.`,
      colors.output,
      token === selected,
      stage === "blend" && token === selected,
    );
    const output = vector(row(trace.output, token, 3));
    nodes += `<g id="output-${token}" class="node ${token === selected ? "selected" : ""} ${stage === "blend" && token === selected ? "lit" : ""}" tabindex="0" role="button" data-output="${token}" aria-label="Output ${token}: ${output}" transform="translate(1186 ${center(token) - 46})">
      <title>Output ${token} is the weighted sum of V0, V1 and V2. These coordinates are not decision probabilities.</title>
      <rect class="shape" width="229" height="92" rx="13"/>
      ${text(14, 26, `Output ${token}`, "output")}
      ${text(14, 53, output, "numbers")}
      ${text(14, 76, "a vector, not a typed decision yet", "tiny")}
    </g>`;
  });

  (["scores", "weights"] as const).forEach((kind) => {
    const x = kind === "scores" ? 592 : 850;
    const active =
      stage === (kind === "scores" ? "comparison" : "normalization");
    nodes += `<g id="${kind}" class="${active ? "lit" : ""}"><rect class="shape" x="${x}" y="306" width="203" height="218" rx="12"/>
      ${text(x + 14, 296, kind === "scores" ? "QKᵀ / √2" : "softmax(scores)", "formula")}`;
    tokens.forEach((key) => {
      nodes += text(x + 20 + key * 63, 329, `k${key}`, "tiny");
    });
    tokens.forEach((query) => {
      nodes += `<g id="${kind}-row-${query}" class="node ${query === selected ? "selected" : ""}" tabindex="0" role="button" data-token="${query}" aria-label="Follow query ${query} through ${kind}"><title>Row ${query} belongs to query ${query}; columns are keys 0, 1, 2.</title>`;
      nodes += text(x - 25, cellY(query) + 5, `q${query}`, "tiny");
      tokens.forEach((key) => {
        const value = at(trace[kind], query * 3 + key);
        const cellX = x + 8 + key * 63;
        nodes += `<rect x="${cellX}" y="${cellY(query) - 19}" width="60" height="40" rx="6" fill="${query === selected ? "#284a4b" : "#1d2b3b"}" stroke="${query === selected ? "#81e2be" : "#33475e"}" fill-opacity="${kind === "weights" ? 0.25 + value * 0.75 : 1}"/>`;
        nodes += text(
          cellX + 5,
          cellY(query) + 6,
          format(value),
          "numbers small",
        );
      });
      nodes += "</g>";
    });
    nodes += "</g>";
  });
  nodes += text(850, 549, "Each row sums to 1", "tiny");

  tokens.forEach((valueToken) => {
    const weight = at(trace.weights, selected * 3 + valueToken);
    const vy = derivedY(valueToken, "value");
    const bottom = 718 + valueToken * 27;
    edges += edge(
      `mix-${valueToken}`,
      `M462 ${vy} C485 ${vy} 482 ${bottom} 548 ${bottom} L1100 ${bottom} C1160 ${bottom} 1138 ${center(selected) + 23} 1186 ${center(selected) + 23}`,
      `Output ${selected} receives ${(weight * 100).toFixed(1)}% of V${valueToken}: ${vector(row(trace.values, valueToken, 3))}.`,
      colors.value,
      true,
      stage === "blend",
      1 + weight * 4,
    );
  });
  nodes += text(
    595,
    678,
    `Following token ${selected} · all three queries are computed in Rust`,
    "hint",
  );
  nodes += text(
    650,
    806,
    "Purple paths carry values; thickness shows the selected query’s attention weights.",
    "tiny",
  );
  return { edges, nodes };
}
