export const DECISIONS = Object.freeze(['Left', 'Forward', 'Right']);
export const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
export const project = (input, matrix) => matrix[0].map((_, j) => input.reduce((sum, x, i) => sum + x * matrix[i][j], 0));
export function softmax(scores) {
  const max = Math.max(...scores);
  const exp = scores.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / sum);
}
export function createModel() {
  return {
    // Sparse hand-set matrices make this first forward pass easy to inspect.
    keyWeights: [[1, 0], [0, 1], [0, 0], [0, 0], [0, 0]],
    valueWeights: [[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    queries: [[2, 0], [0, 2]],
    hiddenWeights: [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    hiddenBias: [-0.3, -0.3, -0.3],
    outputWeights: [[1.4, 0, 0], [0, 1.4, 0], [0, 0, 1.4]],
    outputBias: [0, 0, 0],
  };
}
export const defaultInputs = () => [[0.35, 0.9], [0.9, 0.55], [0.6, 0.3]];
export function attend(query, keys, values) {
  const scores = keys.map(key => dot(query, key) / Math.sqrt(query.length));
  const weights = softmax(scores);
  const output = values[0].map((_, j) => values.reduce((sum, value, i) => sum + weights[i] * value[j], 0));
  return { query: [...query], scores, weights, output };
}
export function forward(model, inputs) {
  const tokens = inputs.map((features, route) => [...features, ...DECISIONS.map((_, i) => Number(i === route))]);
  const keys = tokens.map(x => project(x, model.keyWeights));
  const values = tokens.map(x => project(x, model.valueWeights));
  const heads = model.queries.map(query => attend(query, keys, values));
  const joined = heads.flatMap(head => head.output);
  const preHidden = project(joined, model.hiddenWeights).map((x, i) => x + model.hiddenBias[i]);
  const hidden = preHidden.map(x => Math.max(0, x));
  const logits = project(hidden, model.outputWeights).map((x, i) => x + model.outputBias[i]);
  const probabilities = softmax(logits);
  const selected = probabilities.indexOf(Math.max(...probabilities));
  return { tokens, keys, values, heads, joined, preHidden, hidden, logits, probabilities, decision: DECISIONS[selected] };
}
export function crossEntropy(logits, target) {
  const max = Math.max(...logits);
  return max + Math.log(logits.reduce((sum, x) => sum + Math.exp(x - max), 0)) - logits[target];
}
export function trainOutput(model, inputs, target, rate) {
  if (!Number.isInteger(target) || target < 0 || target >= DECISIONS.length) throw new Error('Unknown target');
  if (!Number.isFinite(rate) || rate <= 0 || rate > 1) throw new Error('Learning rate must be in (0, 1]');
  const before = forward(model, inputs);
  const errors = before.probabilities.map((p, i) => p - Number(i === target));
  const gradients = before.hidden.map(h => errors.map(error => h * error));
  const next = structuredClone(model);
  next.outputWeights = model.outputWeights.map((row, i) => row.map((weight, j) => weight - rate * gradients[i][j]));
  next.outputBias = model.outputBias.map((bias, j) => bias - rate * errors[j]);
  const after = forward(next, inputs);
  return { model: next, before, after, gradients, biasGradients: errors, target, rate,
    lossBefore: crossEntropy(before.logits, target), lossAfter: crossEntropy(after.logits, target),
    weightsBefore: structuredClone(model.outputWeights), weightsAfter: structuredClone(next.outputWeights) };
}
