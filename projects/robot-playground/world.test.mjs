import test from 'node:test';
import assert from 'node:assert/strict';
import { initialState, step, success, distanceToGoal } from './world.mjs';
test('seed and replay reproduce the same world without mutating inputs', () => {
  const a = initialState(12), saved = structuredClone(a);
  assert.deepEqual(a, initialState(12)); assert.notDeepEqual(a, initialState(13));
  const actions = ['up', 'right', 'down', 'wait'];
  assert.deepEqual(actions.reduce(step, a), actions.reduce(step, initialState(12)));
  assert.deepEqual(a, saved); assert.throws(() => step(a, 'invalid'));
});
test('contact pushes the object toward the goal; waiting does not', () => {
  const s = { seed: 1, tick: 0, robot: { x: 74, y: 100 }, object: { x: 100, y: 100 }, goal: { x: 140, y: 100 } };
  const n = step(s, 'right'); assert.equal(n.object.x, 104); assert.equal(n.robot.x, 78);
  assert.deepEqual(step(s, 'wait').object, s.object);
  assert.ok(distanceToGoal(n) < distanceToGoal(s)); assert.equal(success(s), false);
  assert.equal(success(Array.from({length: 10}, () => 'right').reduce(step, s)), true);
});
test('wall blocks pushing and robot stays within bounds', () => {
  const s = { seed: 1, tick: 0, robot: { x: 214, y: 100 }, object: { x: 240, y: 100 }, goal: { x: 210, y: 100 } };
  const n = step(s, 'right'); assert.deepEqual(n.robot, s.robot); assert.deepEqual(n.object, s.object);
  let a = initialState(); for (let i = 0; i < 100; i++) a = step(a, 'left'); assert.equal(a.robot.x, 10);
});
