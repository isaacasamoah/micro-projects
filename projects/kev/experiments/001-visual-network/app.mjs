import { DECISIONS, createModel, defaultInputs, forward, trainOutput, attend, softmax, project, crossEntropy } from './model.mjs';
const $ = selector => document.querySelector(selector);
const f = n => n.toFixed(3), vector = xs => `[${xs.map(f).join(', ')}]`;
let model = createModel(), inputs = defaultInputs(), stage = 0, timer = null, trainingTimer = null, training = null, updates = 0;
const stageNames = ['Inputs', 'Project K / V', 'Attend', 'Concatenate', 'Hidden layer', 'Decide'];
const stageDescriptions = [
  'Each route is five numbers: clearance, progress, and three route-identity bits. These are supplied features; no perception model is hidden here.',
  'Multiply each input by Wk and Wv. Keys describe what a route offers; values describe what information to carry onward. Inspect the exact matrices below.',
  'Each head compares its query with every key, divides scores by √2, applies softmax, and sums the values using those weights. Line thickness shows that attention weight.',
  'Join the two three-number head outputs into one six-number vector. Concatenation preserves both views; it does not average them.',
  'A 6 → 3 linear layer mixes the head outputs. Add a bias and apply ReLU: negative numbers become zero. These weights stay fixed in this first experiment.',
  'A 3 → 3 linear layer produces one logit per allowed decision. Softmax normalizes them; argmax chooses an enum member. No text decoding or token loop.'
];
const controls = (prefix, rows, labels, names, min, max, step) => rows.map((row, i) => `<div class="${prefix === 'input' ? 'route' : 'query'}"><strong>${names[i]}</strong>${row.map((value, j) => `<label class="feature" for="${prefix}-${i}-${j}"><span>${labels[j]}</span><input id="${prefix}-${i}-${j}" type="range" min="${min}" max="${max}" step="${step}" value="${value}"><output for="${prefix}-${i}-${j}">${value.toFixed(2)}</output></label>`).join('')}</div>`).join('');
$('#inputs').innerHTML = controls('input', inputs, ['Clearance', 'Progress'], DECISIONS, 0, 1, .01);
$('#queries').innerHTML = controls('query', model.queries, ['Clearance', 'Progress'], ['Head A · starts with clearance', 'Head B · starts with progress'], -4, 4, .1);
$('#stages').innerHTML = stageNames.map((name, i) => `<button data-stage="${i}">${i + 1}. ${name}</button>`).join('');
function stopPlayback() {
  if (timer) clearInterval(timer); timer = null;
  if (trainingTimer) clearTimeout(trainingTimer); trainingTimer = null;
  $('#play').textContent = '▶ Play inference';
}
function clearUpdate() { training = null; $('#phase-note').textContent = 'Inputs changed. Previous update detail cleared; learned output weights remain.'; }
function syncControls() {
  for (const [prefix, rows] of [['input', inputs], ['query', model.queries]]) rows.forEach((row, i) => row.forEach((value, j) => {
    const slider = $(`#${prefix}-${i}-${j}`); slider.value = value; slider.nextElementSibling.textContent = value.toFixed(2);
  }));
}
for (const prefix of ['input', 'query']) $(`#${prefix === 'input' ? 'inputs' : 'queries'}`).addEventListener('input', event => {
  const [, i, j] = event.target.id.split('-');
  if (i === undefined) return;
  (prefix === 'input' ? inputs : model.queries)[Number(i)][Number(j)] = Number(event.target.value);
  event.target.nextElementSibling.textContent = Number(event.target.value).toFixed(2);
  stopPlayback(); clearUpdate(); render();
});
function table(headers, rows, extra = '') {
  return `<div class="matrix-wrap"><table class="${extra}"><thead><tr>${headers.map(x => `<th>${x}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(x => `<td>${x}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}
function diagram(trace, backward = false) {
  const parts = [], y = [105, 205, 305];
  function line(x1, y1, x2, y2, phase, weight = 1, color = '#667e98') {
    const path = `M${x1},${y1} C${(x1+x2)/2},${y1} ${(x1+x2)/2},${y2} ${x2},${y2}`;
    const active = phase === stage;
    parts.push(`<path d="${path}" fill="none" stroke="${color}" stroke-width="${phase === 2 ? 1 + 6 * weight : 1.5}" opacity="${active ? .9 : .25}"/>`);
    if (active && timer && !matchMedia('(prefers-reduced-motion: reduce)').matches) parts.push(`<circle r="3.5" fill="${color}"><animateMotion dur="0.85s" repeatCount="indefinite" path="${path}"/></circle>`);
  }
  function node(x, cy, width, height, label, detail, phase, color = '#7be1be') {
    const active = phase === stage;
    parts.push(`<rect class="node ${active ? 'active' : ''}" x="${x}" y="${cy-height/2}" width="${width}" height="${height}" rx="10"/>`);
    parts.push(`<text x="${x+width/2}" y="${cy-4}" text-anchor="middle" style="fill:${active ? color : '#eef3fa'}">${label}</text><text class="small" x="${x+width/2}" y="${cy+15}" text-anchor="middle">${detail}</text>`);
  }
  const headings = [['ROUTE FEATURES', 73], ['KEYS + VALUES', 248], ['ATTENTION HEADS', 445], ['JOIN', 633], ['HIDDEN', 801], ['DECISION', 1030]];
  headings.forEach(([title, x]) => parts.push(`<text class="label" x="${x}" y="32" text-anchor="middle">${title}</text>`));
  y.forEach((cy, i) => {
    line(135, cy, 190, cy, 1);
    for (let h = 0; h < 2; h++) line(305, cy, 377, 140 + h * 140, 2, trace.heads[h].weights[i], h ? '#f2bb75' : '#83b8ff');
  });
  for (let h = 0; h < 2; h++) line(518, 140 + h * 140, 585, 205, 3);
  y.forEach(cy => { line(685, 205, 770, cy, 4); y.forEach(out => line(832, cy, 964, out, 5)); });
  y.forEach((cy, i) => {
    node(10, cy, 125, 62, DECISIONS[i], `${inputs[i][0].toFixed(2)} / ${inputs[i][1].toFixed(2)}`, 0);
    node(190, cy, 115, 62, `K ${vector(trace.keys[i])}`, `V [${trace.values[i].join(', ')}]`, 1);
    node(770, cy, 62, 58, `h${i + 1}`, f(trace.hidden[i]), 4);
    node(964, cy, 140, 62, DECISIONS[i], `${(trace.probabilities[i] * 100).toFixed(1)}%`, 5);
  });
  for (let h = 0; h < 2; h++) {
    const cy = 140 + h * 140, color = h ? '#f2bb75' : '#83b8ff';
    node(377, cy, 141, 100, `Head ${h ? 'B' : 'A'}`, `q ${vector(model.queries[h])}`, 2, color);
    parts.push(`<text class="small" x="447" y="${cy+37}" text-anchor="middle">${trace.heads[h].weights.map(v => (v*100).toFixed(0)+'%').join(' / ')}</text>`);
  }
  node(585, 205, 100, 188, '6 values', 'A then B', 3);
  trace.joined.forEach((v, i) => parts.push(`<rect x="${592+i*14}" y="${185-v*85}" width="9" height="${v*85}" rx="3" fill="${i < 3 ? '#83b8ff' : '#f2bb75'}"/>`));
  if (backward) {
    parts.push('<path d="M1070,369 L804,369" stroke="#e4a0e7" stroke-width="3" stroke-dasharray="7 5"/><text x="944" y="395" text-anchor="middle" style="fill:#e4a0e7">loss → gradients → final weights only</text>');
    if (!matchMedia('(prefers-reduced-motion: reduce)').matches) parts.push('<circle r="5" fill="#e4a0e7"><animateMotion dur="1.2s" repeatCount="indefinite" path="M1070,369 L804,369"/></circle>');
  } else parts.push('<text class="small" x="560" y="388" text-anchor="middle">Flow playback is illustrative. All displayed weights and activations are computed.</text>');
  $('#network').innerHTML = parts.join('');
}
function renderTraining() {
  if (!training) { $('#training-result').innerHTML = `<p class="subtle">${updates} output-layer updates so far. Change the target to explore a different training signal. Only the most recent update is shown.</p>`; return; }
  $('#training-result').innerHTML = `<div class="metric">Loss ${f(training.lossBefore)} → ${f(training.lossAfter)}</div><p>Update ${updates} · target ${DECISIONS[training.target]} · learning rate ${training.rate.toFixed(2)}</p><p class="subtle">Actual cross-entropy on this one scene. Lower loss here does not establish generalization or calibration.</p><details><summary>Inspect the gradients and parameter change</summary><pre>logit gradient = probabilities − one-hot target\n${vector(training.biasGradients)}\n\nweight gradient[i,j] = hidden[i] × logit gradient[j]\nnew weight = old weight − rate × gradient</pre>${table(['Weight','Before','Gradient','After'], training.gradients.flatMap((row, i) => row.map((g, j) => [`h${i+1} → ${DECISIONS[j]}`, f(training.weightsBefore[i][j]), f(g), f(training.weightsAfter[i][j])])), 'gradient')}</details>`;
}
function render(backward = false) {
  const trace = forward(model, inputs);
  document.querySelectorAll('[data-stage]').forEach(button => button.setAttribute('aria-current', Number(button.dataset.stage) === stage ? 'step' : 'false'));
  $('#inspector').innerHTML = `<strong>${stage + 1}. ${stageNames[stage]}</strong><p>${stageDescriptions[stage]}</p>`;
  diagram(trace, backward);
  $('#decision-title').textContent = `Decision: ${trace.decision}`;
  $('#probabilities').innerHTML = trace.probabilities.map((p, i) => `<div class="prob ${trace.decision === DECISIONS[i] ? 'selected' : ''}"><span>${DECISIONS[i]}</span><div class="bar"><div class="fill" style="width:${p*100}%"></div></div><b>${(p*100).toFixed(1)}%</b></div>`).join('');
  $('#typed').textContent = `enum Route { Left, Forward, Right }\n\nRoute::${trace.decision}`;
  $('#projection-values').innerHTML = table(['Route', 'Input x', 'Key xWk', 'Value xWv'], DECISIONS.map((name, i) => [name, vector(trace.tokens[i]), vector(trace.keys[i]), vector(trace.values[i])]));
  $('#attention-values').innerHTML = trace.heads.map((head, i) => `<h3>Head ${i ? 'B' : 'A'} · query ${vector(head.query)}</h3>${table(['Route','q·k / √2','Softmax weight'], DECISIONS.map((name, j) => [name, f(head.scores[j]), f(head.weights[j])]))}<pre>weighted sum of values = ${vector(head.output)}</pre>`).join('');
  $('#layer-values').innerHTML = `<pre>joined  = ${vector(trace.joined)}\npreReLU = ${vector(trace.preHidden)}\nReLU    = ${vector(trace.hidden)}\nlogits  = ${vector(trace.logits)}\nsoftmax = ${vector(trace.probabilities)}</pre>`;
  $('#matrices').textContent = JSON.stringify(model, null, 2);
  renderTraining();
}
$('#stages').onclick = event => { const button = event.target.closest('button'); if (!button) return; stopPlayback(); stage = Number(button.dataset.stage); render(); };
$('#step').onclick = () => { stopPlayback(); stage = (stage + 1) % stageNames.length; $('#phase-note').textContent = 'Stepping through one forward computation. Weights are unchanged.'; render(); };
$('#play').onclick = () => {
  if (timer) { stopPlayback(); render(); return; }
  stopPlayback(); stage = 0; $('#play').textContent = 'Pause'; $('#phase-note').textContent = 'Forward pass: read the input, mix evidence, choose a label.';
  timer = setInterval(() => { if (stage === stageNames.length - 1) { stopPlayback(); $('#phase-note').textContent = 'Inference complete. Model weights stayed fixed.'; } else stage++; render(); }, 1100);
  render();
};
$('#reset').onclick = () => { stopPlayback(); model = createModel(); training = null; updates = 0; stage = 0; syncControls(); $('#phase-note').textContent = 'Model reset to hand-set weights. Input scene preserved.'; render(); };
$('#equal').onclick = () => { stopPlayback(); inputs = DECISIONS.map(() => [.5, .5]); clearUpdate(); syncControls(); render(); };
$('#defaults').onclick = () => { stopPlayback(); inputs = defaultInputs(); clearUpdate(); syncControls(); render(); };
$('#rate').oninput = event => { $('#rate-value').textContent = Number(event.target.value).toFixed(2); };
$('#train').onclick = () => {
  stopPlayback(); training = trainOutput(model, inputs, Number($('#target').value), Number($('#rate').value)); model = training.model; updates++; stage = 5;
  $('#phase-note').textContent = 'Training updated the final layer. Everything upstream stayed fixed; the arrows replay that update.';
  render(true); trainingTimer = setTimeout(() => { trainingTimer = null; render(); }, 2600);
};
const walkthrough = [
  {name:'1. Linear projection', fn:project, pseudo:'For each output feature:\n  combine all input features\n  using one column of weights.\n\nThis same operation makes keys,\nvalues, hidden features and logits.', verdict:'Agree for this slice: explicit sums expose every multiply/add. Matrix libraries are faster, but would hide the first thing we want to understand.'},
  {name:'2. Stable softmax', fn:softmax, pseudo:'Shift all scores by the maximum.\nExponentiate each shifted score.\nDivide by their total.\n\nThe outputs are positive and sum to one.', verdict:'Agree: subtracting the maximum preserves the distribution and avoids overflow. Normalization alone is not evidence of calibrated probabilities.'},
  {name:'3. One attention head', fn:attend, pseudo:'Compare the query with each key.\nScale the scores by √key dimension.\nTurn scores into attention weights.\nUse them to average the value vectors.', verdict:'Agree with a limit: this is query pooling over route tokens. A full transformer would add token-wise queries, residuals, normalization and more blocks.'},
  {name:'4. The forward pass', fn:forward, pseudo:'Encode route features and identity.\nProject keys and values.\nRun both attention heads.\nJoin their results.\nMix through a hidden layer and ReLU.\nProduce three logits and probabilities.\nSelect one allowed route.', verdict:'Agree for exploration: a closed set of labels makes output structure explicit. Our enum is a JavaScript label list today; Rust will give us a compile-time enum.'},
  {name:'5. Cross-entropy', fn:crossEntropy, pseudo:'Measure how much probability\nthe model assigned to the target.\n\nLoss = −log(probability of target).\nCompute from logits stably.', verdict:'Agree for a classification lesson: the target is supplied by you. One example and a lower loss do not establish that the target rule was learned.'},
  {name:'6. One training update', fn:trainOutput, pseudo:'Run the current scene.\nCompare probabilities with its label.\nCompute final-layer gradients.\nUpdate final weights and biases.\nRun again and compare the loss.\n\nKeep earlier layers frozen.', verdict:'Agree with a deliberate boundary: the gradients are real and checked numerically. Full-network backpropagation and a dataset belong in later, individually explained slices.'}
];
$('#code-step').innerHTML = walkthrough.map((item, i) => `<option value="${i}">${item.name}</option>`).join('');
function showCode() { const item = walkthrough[Number($('#code-step').value)]; $('#pseudo').textContent = item.pseudo; $('#code').textContent = item.fn.toString(); $('#verdict').textContent = item.verdict; }
$('#code-step').onchange = showCode;
window.kev = Object.freeze({ snapshot: () => structuredClone({ model, inputs, trace: forward(model, inputs), updates, training }) });
showCode(); render();
