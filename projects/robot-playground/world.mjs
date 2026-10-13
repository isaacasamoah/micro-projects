export const SIZE = 256;
export const ROBOT_RADIUS = 10;
export const OBJECT_RADIUS = 16;
export const ACTIONS = { left: [-4, 0], right: [4, 0], up: [0, -4], down: [0, 4], wait: [0, 0] };
export function initialState(seed = 1) {
  if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) throw new Error('Seed must be a uint32');
  let value = seed;
  const random = () => ((value = (Math.imul(value, 1664525) + 1013904223) >>> 0) / 4294967296);
  return { seed, tick: 0, robot: { x: 45, y: 70 + random() * 116 }, object: { x: 120, y: 80 + random() * 96 }, goal: { x: 210, y: 80 + random() * 96 } };
}
const clamp = (x, r) => Math.max(r, Math.min(SIZE - r, x));
export const distanceToGoal = s => Math.hypot(s.object.x - s.goal.x, s.object.y - s.goal.y);
export const success = s => distanceToGoal(s) <= 8;
export function step(state, action) {
  if (!Object.hasOwn(ACTIONS, action)) throw new Error('Unknown action');
  const next = structuredClone(state);
  next.tick++;
  const [dx, dy] = ACTIONS[action];
  next.robot = { x: clamp(state.robot.x + dx, ROBOT_RADIUS), y: clamp(state.robot.y + dy, ROBOT_RADIUS) };
  const ox = state.object.x - next.robot.x, oy = state.object.y - next.robot.y;
  const distance = Math.hypot(ox, oy), contact = ROBOT_RADIUS + OBJECT_RADIUS;
  if (distance < contact) {
    if (!distance) return { ...structuredClone(state), tick: next.tick };
    next.object = { x: clamp(next.robot.x + ox / distance * contact, OBJECT_RADIUS), y: clamp(next.robot.y + oy / distance * contact, OBJECT_RADIUS) };
    // Reject the move when the wall prevents resolving contact without overlap.
    if (Math.hypot(next.object.x - next.robot.x, next.object.y - next.robot.y) < contact - 1e-8) {
      next.robot = structuredClone(state.robot);
      next.object = structuredClone(state.object);
    }
  }
  return next;
}
