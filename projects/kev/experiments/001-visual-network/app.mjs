import {
  DECISIONS,
  createModel,
  defaultInputs,
  forward,
  trainOutput,
  crossEntropy,
} from "./model.mjs";
const $ = (selector) => document.querySelector(selector);
const number = (value) => value.toFixed(3),
  vector = (values) => `[${values.map(number).join(", ")}]`;
let model = createModel(),
  inputs = defaultInputs(),
  trace = forward(model, inputs),
  target = 2,
  rate = 0.2;
let phase = -1,
  mode = null,
  timer = null,
  pending = null,
  lastUpdate = null,
  updates = 0;
const nodes = new Map(),
  edges = new Map(),
  parts = [];
const rows = [185, 365, 545],
  joinedRows = [145, 225, 305, 435, 515, 595];
function text(x, y, label, className = "", id = "") {
  return `<text x="${x}" y="${y}" text-anchor="middle" class="${className}" ${id ? `id="${id}"` : ""}>${label}</text>`;
}
function node(id, x, y, w, h, content, info, stage, circle = false) {
  nodes.set(id, { info, stage });
  const shape = circle
    ? `<circle class="node-shape" cx="${x}" cy="${y}" r="${w / 2}"/>`
    : `<rect class="node-shape" x="${x - w / 2}" y="${y - h / 2}" width="${w}" height="${h}" rx="18"/>`;
  parts.push(
    `<g id="node-${id}" data-node="${id}" class="node-group" tabindex="0" role="button" aria-label="${info.title}">${shape}${content}</g>`,
  );
}
function path(a, b) {
  return `M${a[0]},${a[1]} C${(a[0] + b[0]) / 2},${a[1]} ${(a[0] + b[0]) / 2},${b[1]} ${b[0]},${b[1]}`;
}
function edge(
  id,
  a,
  b,
  stage,
  info,
  weight = () => 1,
  color = "#617b97",
  parameter = null,
) {
  const d = path(a, b);
  edges.set(id, { a, b, stage, info, weight, parameter });
  parts.push(
    `<g id="edge-${id}" data-edge="${id}" class="edge-group" tabindex="0" role="button" aria-label="${info.title}"><path class="connection" d="${d}" style="stroke:${color}"/><path class="connection-hit" d="${d}"/></g>`,
  );
}
const prose = (title, description, detail = () => "") => ({
  title,
  description,
  detail,
});
const yHead = [240, 510];
rows.forEach((y, i) => {
  edge(
    `input-${i}`,
    [215, y],
    [269, y],
    1,
    prose(
      `${DECISIONS[i]} → keys and values`,
      "Multiply the five input coordinates by the key and value matrices.",
      () =>
        `input = ${vector(trace.tokens[i])}\nkey = ${vector(trace.keys[i])}\nvalue = ${vector(trace.values[i])}`,
    ),
  );
  for (let h = 0; h < 2; h++)
    edge(
      `attention-${h}-${i}`,
      [331, y],
      [385, yHead[h]],
      2,
      prose(
        `${DECISIONS[i]} → head ${h ? "B" : "A"}`,
        "This pathway carries a projected key and value. Thickness shows the attention weight this route receives.",
        () =>
          `query = ${vector(model.queries[h])}\nkey = ${vector(trace.keys[i])}\nscore = q·k / √2 = ${number(trace.heads[h].scores[i])}\nattention = ${number(trace.heads[h].weights[i])}\nvalue = ${vector(trace.values[i])}`,
      ),
      () => trace.heads[h].weights[i],
      h ? "#f2bc77" : "#86b9ff",
    );
});
for (let h = 0; h < 2; h++)
  for (let i = 0; i < 3; i++) {
    const j = h * 3 + i;
    edge(
      `head-${j}`,
      [615, yHead[h]],
      [736, joinedRows[j]],
      3,
      prose(
        `Head ${h ? "B" : "A"} → value ${i + 1}`,
        "Copy this head-output coordinate into the joined vector. There is no new weight and no averaging at this step.",
        () =>
          `joined[${j}] = head${h ? "B" : "A"}[${i}] = ${number(trace.joined[j])}`,
      ),
      () => trace.joined[j],
      h ? "#f2bc77" : "#86b9ff",
    );
  }
for (let i = 0; i < 6; i++)
  for (let j = 0; j < 3; j++)
    edge(
      `hidden-${i}-${j}`,
      [784, joinedRows[i]],
      [954, rows[j]],
      4,
      prose(
        `Joined ${i + 1} → hidden ${j + 1}`,
        "This connection multiplies one joined value by a hidden-layer weight. Click to edit that weight manually.",
        () =>
          `${number(trace.joined[i])} × ${number(model.hiddenWeights[i][j])} = ${number(trace.joined[i] * model.hiddenWeights[i][j])}\nThen sum all six contributions, add bias and apply ReLU.`,
      ),
      () => model.hiddenWeights[i][j],
      "#7891ab",
      { group: "hiddenWeights", i, j },
    );
for (let i = 0; i < 3; i++)
  for (let j = 0; j < 3; j++)
    edge(
      `output-${i}-${j}`,
      [1046, rows[i]],
      [1180, rows[j]],
      5,
      prose(
        `Hidden ${i + 1} → ${DECISIONS[j]} logit`,
        "One trainable output weight. Training changes this weight using hidden activation × output error.",
        () =>
          `weight = ${number(model.outputWeights[i][j])}\ncontribution = ${number(trace.hidden[i] * model.outputWeights[i][j])}${phase === 7 && pending ? `\ngradient = ${number(pending.gradients[i][j])}\nnext weight = ${number(pending.weightsAfter[i][j])}` : lastUpdate ? `\nlast gradient = ${number(lastUpdate.gradients[i][j])}\nlast update: ${number(lastUpdate.weightsBefore[i][j])} → ${number(lastUpdate.weightsAfter[i][j])}` : ""}`,
      ),
      () => model.outputWeights[i][j],
      "#7eafad",
      { group: "outputWeights", i, j },
    );
for (let j = 0; j < 3; j++)
  edge(
    `loss-${j}`,
    [1310, rows[j]],
    [1240, 704],
    6,
    prose(
      `${DECISIONS[j]} probability → loss`,
      "Cross-entropy compares the predicted distribution with the selected target label.",
      () =>
        `target = ${DECISIONS[target]}\nP(target) = ${number(trace.probabilities[target])}\nloss = −ln(P(target)) = ${number(crossEntropy(trace.logits, target))}`,
    ),
    () => 0.3,
    "#9c709c",
  );
parts.push('<g id="particles" aria-hidden="true"></g>');
[
  ["INPUTS", 120],
  ["PROJECT", 300],
  ["ATTENTION", 500],
  ["BOTH HEAD OUTPUTS", 760],
  ["HIDDEN", 1000],
  ["TYPED DECISION", 1245],
].forEach(([s, x]) => parts.push(text(x, 70, s, "caption")));
function sliderCard(
  prefix,
  i,
  title,
  labels,
  values,
  x,
  y,
  w,
  h,
  min,
  max,
  step,
  cls = "",
) {
  return `<foreignObject x="${x - w / 2}" y="${y - h / 2}" width="${w}" height="${h}"><div xmlns="http://www.w3.org/1999/xhtml" class="node-card ${cls}"><strong>${title}</strong>${labels.map((label, j) => `<label><span>${label}</span><input aria-label="${title} ${label}" id="${prefix}-${i}-${j}" type="range" min="${min}" max="${max}" step="${step}" value="${values[j]}"/><output id="value-${prefix}-${i}-${j}">${values[j].toFixed(2)}</output></label>`).join("")}</div></foreignObject>`;
}
rows.forEach((y, i) => {
  node(
    `input-${i}`,
    120,
    y,
    190,
    132,
    sliderCard(
      "input",
      i,
      DECISIONS[i],
      ["Clearance", "Progress"],
      inputs[i],
      120,
      y,
      190,
      132,
      0,
      1,
      0.01,
    ),
    prose(
      `${DECISIONS[i]} inputs`,
      "Two synthetic features on a 0–1 scale. Higher clearance means less obstruction; higher progress means more progress toward a goal. Neither is a probability.",
      () =>
        `token = ${vector(trace.tokens[i])}\nLast three coordinates encode route identity.`,
    ),
    0,
  );
  node(
    `project-${i}`,
    300,
    y,
    62,
    62,
    text(300, y - 3, "K / V") + text(300, y + 15, "5 → 2 / 3", "tiny"),
    prose(
      `${DECISIONS[i]} projection`,
      "Keys are compared with a query. Values are the information that attention blends. These shared projections stay fixed during the training demonstration.",
      () =>
        `key = ${vector(trace.keys[i])}\nvalue = ${vector(trace.values[i])}\n\nWk = ${JSON.stringify(model.keyWeights)}\nWv = ${JSON.stringify(model.valueWeights)}`,
    ),
    1,
    true,
  );
});
yHead.forEach((y, h) =>
  node(
    `head-${h}`,
    500,
    y,
    230,
    146,
    sliderCard(
      "query",
      h,
      `Head ${h ? "B" : "A"}`,
      ["Clearance", "Progress"],
      model.queries[h],
      500,
      y,
      230,
      146,
      -4,
      4,
      0.1,
      h ? "head-b" : "head-a",
    ),
    prose(
      `Head ${h ? "B" : "A"}`,
      "Edit the query to change what this head looks for. Both heads compare their query with every route key, then blend the values. Query values can be negative.",
      () =>
        `query = ${vector(model.queries[h])}\nscores = ${vector(trace.heads[h].scores)}\nattention = ${vector(trace.heads[h].weights)}\noutput = ${vector(trace.heads[h].output)}\n\nsoftmax(q·K / √2) × V`,
    ),
    2,
  ),
);
joinedRows.forEach((y, i) =>
  node(
    `joined-${i}`,
    760,
    y,
    48,
    48,
    text(760, y + 5, "", "value", `joined-value-${i}`) +
      text(710, y + 4, `${i < 3 ? "A" : "B"}${(i % 3) + 1}`, "tiny"),
    prose(
      `Joined value ${i + 1}: ${i < 3 ? "A" : "B"}${(i % 3) + 1}`,
      "Concatenation keeps the two head outputs separate: A1, A2, A3, B1, B2, B3. This node copies a number; it does not calculate an average.",
      () =>
        `[${trace.joined.map((x, j) => (j === i ? `→${number(x)}←` : number(x))).join(", ")}]`,
    ),
    3,
    true,
  ),
);
rows.forEach((y, i) => {
  node(
    `hidden-${i}`,
    1000,
    y,
    92,
    92,
    text(1000, y - 15, `h${i + 1}`) +
      text(1000, y + 9, "", "value", `hidden-value-${i}`) +
      text(1000, y + 28, "ReLU", "tiny"),
    prose(
      `Hidden neuron ${i + 1}`,
      "Combine all six incoming weighted values, add a bias and set negative results to zero. Click an incoming pathway to edit its weight.",
      () =>
        `pre-activation = ${number(trace.preHidden[i])}\nbias = ${number(model.hiddenBias[i])}\nReLU = max(0, pre-activation)\noutput = ${number(trace.hidden[i])}`,
    ),
    4,
    true,
  );
  node(
    `decision-${i}`,
    1245,
    y,
    130,
    130,
    text(1245, y - 22, DECISIONS[i]) +
      text(1245, y + 4, "", "value", `probability-${i}`) +
      text(1245, y + 27, "", "tiny", `target-${i}`) +
      `<circle class="winner-dot" cx="1290" cy="${y - 43}" r="5"/>`,
    prose(
      `${DECISIONS[i]} decision`,
      "Click to make this the training target. The highest probability is selected for inference. A fixed label set prevents malformed output, not incorrect decisions.",
      () =>
        `logit = ${number(trace.logits[i])}\nprobability = ${number(trace.probabilities[i])}\noutput bias = ${number(model.outputBias[i])}\nselected = ${trace.decision}\n\nThese probabilities are not calibrated. Ties pick the first label.`,
    ),
    5,
    true,
  );
});
node(
  "loss",
  1190,
  710,
  100,
  100,
  text(1190, 699, "Loss", "loss-label") +
    text(1190, 724, "", "value", "loss-value"),
  prose(
    "Training signal",
    "Click a decision node to set the label. Cross-entropy measures how little probability the model gives that label. Training updates the output layer only.",
    () =>
      `target = ${DECISIONS[target]}\nloss = ${number(crossEntropy(trace.logits, target))}\nlearning rate = ${rate.toFixed(2)}\nupdates = ${updates}${lastUpdate ? `\nlast loss: ${number(lastUpdate.lossBefore)} → ${number(lastUpdate.lossAfter)}` : ""}`,
  ),
  6,
  true,
);
parts.push(
  text(1000, 662, "", "frozen-note", "frozen-note"),
  text(1185, 798, "", "learning-note", "learning-note"),
);
$("#graph").innerHTML = parts.join("");
function stop() {
  clearTimeout(timer);
  timer = null;
  pending = null;
  mode = null;
  phase = -1;
  $("#inference").textContent = "▶ Play inference";
  $("#training").textContent = "↶ Play training";
}
function invalidate() {
  stop();
  lastUpdate = null;
  $("#status").textContent = "Manual edit · the network recomputed";
}
function update() {
  trace = forward(model, inputs);
  for (const [id, item] of nodes)
    $(`#node-${id}`).classList.toggle(
      "active",
      item.stage === phase || (phase === 8 && id.startsWith("decision")),
    );
  for (const [id, item] of edges) {
    const group = $(`#edge-${id}`),
      line = group.querySelector(".connection"),
      weight = item.weight();
    line.setAttribute(
      "stroke-width",
      String(
        0.7 +
          Math.min(
            4,
            Math.abs(weight) * (id.startsWith("attention") ? 6 : 1.4),
          ),
      ),
    );
    line.setAttribute("stroke-dasharray", Math.abs(weight) < 1e-9 ? "3 6" : "");
    line.style.opacity =
      phase >= 7 && id.startsWith("output")
        ? "1"
        : Math.abs(weight) < 1e-9
          ? ".14"
          : phase === item.stage
            ? ".95"
            : ".48";
    group.classList.toggle("active", phase === item.stage);
    group.classList.toggle("backward", phase >= 7 && id.startsWith("output"));
  }
  trace.joined.forEach(
    (v, i) => ($(`#joined-value-${i}`).textContent = number(v)),
  );
  trace.hidden.forEach(
    (v, i) => ($(`#hidden-value-${i}`).textContent = number(v)),
  );
  trace.probabilities.forEach((p, i) => {
    $(`#probability-${i}`).textContent = `${(p * 100).toFixed(1)}%`;
    $(`#target-${i}`).textContent = i === target ? "◎ training target" : "";
    $(`#node-decision-${i}`).classList.toggle("target", i === target);
    $(`#node-decision-${i}`).classList.toggle(
      "winner",
      trace.decision === DECISIONS[i],
    );
  });
  $("#loss-value").textContent = number(crossEntropy(trace.logits, target));
  $("#frozen-note").textContent =
    mode === "training" ? "↑ earlier layers stay frozen" : "";
  $("#learning-note").textContent =
    phase >= 7
      ? "only final weights + biases update"
      : `teach ${DECISIONS[target]} · η ${rate.toFixed(2)}`;
  for (const [prefix, values] of [
    ["input", inputs],
    ["query", model.queries],
  ])
    values.forEach((row, i) =>
      row.forEach((v, j) => {
        $(`#value-${prefix}-${i}-${j}`).textContent = v.toFixed(2);
        const input = $(`#${prefix}-${i}-${j}`);
        if (document.activeElement !== input) input.value = v;
      }),
    );
  animateEdges();
}
function animateEdges() {
  if (!mode || matchMedia("(prefers-reduced-motion: reduce)").matches) {
    $("#particles").innerHTML = "";
    return;
  }
  const reverse = phase === 7;
  $("#particles").innerHTML = [...edges.values()]
    .filter((e) =>
      reverse ? e.parameter?.group === "outputWeights" : e.stage === phase,
    )
    .map(
      (e) =>
        `<circle class="particle" r="4" fill="${reverse ? "#e8a3e6" : "#81e2be"}"><animateMotion dur="1.0s" repeatCount="indefinite" path="${reverse ? path(e.b, e.a) : path(e.a, e.b)}"/></circle>`,
    )
    .join("");
}
const phaseLabels = [
  "Read the scene",
  "Project inputs into keys and values",
  "Compare, normalize, blend",
  "Keep all six head outputs",
  "Mix inputs in the hidden neurons",
  "Score the three allowed decisions",
  "Compare with the target label",
  "Backpropagate error to final weights and biases",
  "Apply one update, then predict again",
];
function play(kind) {
  if (mode === kind) {
    stop();
    $("#status").textContent = `Stopped · ${updates} updates applied`;
    update();
    return;
  }
  stop();
  mode = kind;
  phase = 0;
  if (kind === "training") pending = trainOutput(model, inputs, target, rate);
  $(`#${kind}`).textContent = "■ Stop";
  function next() {
    if (phase === 8 && pending) {
      lastUpdate = pending;
      model = pending.model;
      pending = null;
      updates++;
    }
    $("#status").textContent =
      `${kind === "training" ? "Training" : "Inference"} · ${phaseLabels[phase]}${phase === 8 ? ` · loss ${number(lastUpdate.lossBefore)} → ${number(lastUpdate.lossAfter)}` : ""}`;
    update();
    const last = kind === "training" ? 8 : 5;
    timer = setTimeout(
      () => {
        if (phase === last) {
          const message =
            kind === "training"
              ? `Update ${updates} · loss ${number(lastUpdate.lossBefore)} → ${number(lastUpdate.lossAfter)} · ${trace.decision}`
              : `Inference → ${trace.decision} · weights unchanged`;
          stop();
          $("#status").textContent = message;
          update();
        } else {
          phase++;
          next();
        }
      },
      kind === "training" && phase >= 6 ? 1300 : 850,
    );
  }
  next();
}
function showTip(info, x, y) {
  const tip = $("#tooltip");
  tip.textContent = `${info.title}\n${info.description}\n${info.detail()}`;
  tip.hidden = false;
  tip.style.left = `${Math.max(8, Math.min(innerWidth - 325, x + 16))}px`;
  tip.style.top = `${Math.max(70, Math.min(innerHeight - tip.offsetHeight - 48, y + 16))}px`;
}
function inspect(kind, id) {
  const item = (kind === "edge" ? edges : nodes).get(id);
  if (!item) return;
  $("#tooltip").hidden = true;
  const info = item.info;
  $("#editor-content").innerHTML =
    '<h2></h2><p></p><pre></pre><div id="parameter"></div>';
  $("#editor h2").textContent = info.title;
  $("#editor p").textContent = info.description;
  $("#editor pre").textContent = info.detail();
  const parameter = $("#parameter");
  if (item.parameter) {
    const { group, i, j } = item.parameter;
    parameter.innerHTML = `<label>Weight <input id="weight-edit" type="number" min="-8" max="8" step="0.1" value="${model[group][i][j]}"/></label><p>${group === "outputWeights" ? "This weight is trainable." : "This weight stays frozen during training."} Editing it here is a manual change.</p>`;
    $("#weight-edit").oninput = (event) => {
      const input = event.target;
      if (!input.value || !input.checkValidity()) return;
      const v = Number(input.value);
      if (!Number.isFinite(v)) return;
      invalidate();
      model[group][i][j] = v;
      update();
      $("#editor pre").textContent = info.detail();
    };
  }
  if (kind === "node" && id === "loss") {
    parameter.innerHTML = `<label>Learning rate <input id="rate-edit" type="range" min="0.01" max="1" step="0.01" value="${rate}"/></label><p>Choose the target on a decision node. Training updates only the final layer. Repeatedly teaching one scene can overfit it.</p>`;
    $("#rate-edit").oninput = (event) => {
      invalidate();
      rate = Number(event.target.value);
      update();
      $("#editor pre").textContent = info.detail();
    };
  }
  if (kind === "node" && id.startsWith("hidden-")) {
    const i = Number(id.split("-")[1]);
    parameter.innerHTML = `<label>Bias <input id="bias-edit" type="number" min="-8" max="8" step="0.1" value="${model.hiddenBias[i]}"/></label>`;
    $("#bias-edit").oninput = (event) => {
      if (!event.target.value || !event.target.checkValidity()) return;
      invalidate();
      model.hiddenBias[i] = Number(event.target.value);
      update();
      $("#editor pre").textContent = info.detail();
    };
  }
  const editor = $("#editor");
  editor.hidden = false;
  const anchor = $(`#${kind}-${id}`).getBoundingClientRect();
  const x =
    anchor.right + 12 + editor.offsetWidth < innerWidth
      ? anchor.right + 12
      : anchor.left - editor.offsetWidth - 12;
  editor.style.left = `${Math.max(12, Math.min(innerWidth - editor.offsetWidth - 12, x))}px`;
  editor.style.top = `${Math.max(72, Math.min(innerHeight - editor.offsetHeight - 52, anchor.top))}px`;
}
function pick(element) {
  const nodeId = element.dataset.node,
    edgeId = element.dataset.edge;
  if (nodeId?.startsWith("decision-")) {
    invalidate();
    target = Number(nodeId.split("-")[1]);
    $("#status").textContent =
      `Teach ${DECISIONS[target]} · press Play training`;
    update();
  }
  inspect(nodeId ? "node" : "edge", nodeId || edgeId);
}
$("#graph").addEventListener("input", (event) => {
  if (!event.target.matches("input[type=range]")) return;
  const [prefix, i, j] = event.target.id.split("-");
  invalidate();
  (prefix === "input" ? inputs : model.queries)[Number(i)][Number(j)] = Number(
    event.target.value,
  );
  update();
  $("#editor").hidden = true;
});
$("#graph").addEventListener("pointerover", (event) => {
  if (event.target.closest("input") || !$("#editor").hidden) return;
  const element = event.target.closest("[data-node],[data-edge]");
  if (!element) return;
  const item = element.dataset.node
    ? nodes.get(element.dataset.node)
    : edges.get(element.dataset.edge);
  showTip(item.info, event.clientX, event.clientY);
});
$("#graph").addEventListener("pointerout", (event) => {
  const current = event.target.closest("[data-node],[data-edge]");
  if (!current?.contains(event.relatedTarget)) $("#tooltip").hidden = true;
});
$("#graph").addEventListener("click", (event) => {
  if (event.target.closest("input,label")) return;
  const element = event.target.closest("[data-node],[data-edge]");
  if (element) pick(element);
  else {
    $("#editor").hidden = true;
  }
});
$("#graph").addEventListener("keydown", (event) => {
  if (event.target.matches("input")) return;
  if (event.key === "Enter" || event.key === " ") {
    const element = event.target.closest("[data-node],[data-edge]");
    if (element) {
      event.preventDefault();
      pick(element);
    }
  }
});
$("#graph").addEventListener("focusin", (event) => {
  if (event.target.matches("input")) return;
  const e = event.target.closest("[data-node],[data-edge]");
  if (e) {
    const r = e.getBoundingClientRect(),
      item = e.dataset.node
        ? nodes.get(e.dataset.node)
        : edges.get(e.dataset.edge);
    showTip(item.info, r.x, r.y);
  }
});
$("#graph").addEventListener("focusout", () => ($("#tooltip").hidden = true));
$("#inference").onclick = () => {
  $("#editor").hidden = true;
  $("#tooltip").hidden = true;
  play("inference");
};
$("#training").onclick = () => {
  $("#editor").hidden = true;
  $("#tooltip").hidden = true;
  play("training");
};
$("#reset").onclick = () => {
  stop();
  model = createModel();
  inputs = defaultInputs();
  target = 2;
  rate = 0.2;
  lastUpdate = null;
  updates = 0;
  $("#editor").hidden = true;
  $("#tooltip").hidden = true;
  $("#status").textContent =
    "Reset · edit nodes, inspect paths, choose a target";
  update();
};
$("#close-editor").onclick = () => {
  $("#editor").hidden = true;
};
$("#about").onclick = () => $("#info").showModal();
$("#close-info").onclick = () => $("#info").close();
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    $("#editor").hidden = true;
    $("#tooltip").hidden = true;
  }
});
window.kev = Object.freeze({
  snapshot: () =>
    structuredClone({
      model,
      inputs,
      trace,
      target,
      rate,
      updates,
      lastUpdate,
      mode,
      phase,
    }),
});
update();
