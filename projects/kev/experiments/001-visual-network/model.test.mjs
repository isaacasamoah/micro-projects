import test from 'node:test';
import assert from 'node:assert/strict';
import { softmax, attend, createModel, defaultInputs, forward, trainOutput, crossEntropy, DECISIONS } from './model.mjs';
const close = (a, b, eps = 1e-8) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
test('softmax stays normalized for large logits and invariant to a shared offset', () => {
  const a = softmax([1000, 1001, 999]); close(a.reduce((s, x) => s + x), 1);
  softmax([0, 1, -1]).forEach((x, i) => close(x, a[i]));
});
test('zero query gives uniform attention and the arithmetic mean of values', () => {
  const result = attend([0, 0], [[1, 0], [0, 2]], [[2, 0], [0, 4]]);
  assert.deepEqual(result.weights, [0.5, 0.5]); assert.deepEqual(result.output, [1, 2]);
});
test('each head reacts to its configured feature; route decision is closed-set', () => {
  const model = createModel(), inputs = defaultInputs(), first = forward(model, inputs);
  inputs[0][0] = 1; const changed = forward(model, inputs);
  assert.ok(changed.heads[0].weights[0] > first.heads[0].weights[0]);
  assert.deepEqual(changed.heads[1].weights, first.heads[1].weights);
  assert.ok(DECISIONS.includes(changed.decision));
});
test('one output-layer update lowers this example loss without changing the attention or input model', () => {
  const model = createModel(), copy = structuredClone(model), inputs = defaultInputs();
  const result = trainOutput(model, inputs, 2, 0.2);
  assert.ok(result.lossAfter < result.lossBefore); assert.deepEqual(model, copy);
  assert.deepEqual(result.before.heads, result.after.heads);
  assert.notDeepEqual(result.model.outputWeights, model.outputWeights);
  assert.throws(() => trainOutput(model, inputs, 5, 0.2));
});
test('analytic output gradients agree with independent finite differences', () => {
  const model = createModel(), inputs = defaultInputs(), target = 1, epsilon = 1e-5;
  const result = trainOutput(model, inputs, target, 0.2);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    const plus = structuredClone(model), minus = structuredClone(model);
    plus.outputWeights[i][j] += epsilon; minus.outputWeights[i][j] -= epsilon;
    const numeric = (crossEntropy(forward(plus, inputs).logits, target) - crossEntropy(forward(minus, inputs).logits, target)) / (2 * epsilon);
    close(numeric, result.gradients[i][j], 1e-6);
  }
});
