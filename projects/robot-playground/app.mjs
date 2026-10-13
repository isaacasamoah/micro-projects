import { SIZE, ROBOT_RADIUS, OBJECT_RADIUS, ACTIONS, initialState, step, distanceToGoal, success } from './world.mjs';
const canvas = document.querySelector('#world'), status = document.querySelector('#status'), notice = document.querySelector('#notice');
const small = document.createElement('canvas'); small.width = small.height = 64;
let state = initialState(), recording = false, timer = null, activeEpisode = null, episodeCounter = 0;
const episodes = [];
const total = () => episodes.reduce((sum, episode) => sum + episode.transitions.length, 0);
function draw(target, s) {
  const ctx = target.getContext('2d'); ctx.save(); ctx.scale(target.width / SIZE, target.height / SIZE);
  ctx.fillStyle = '#172331'; ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.strokeStyle = '#253748'; ctx.lineWidth = 1;
  for (let i = 0; i <= SIZE; i += 32) { ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, SIZE); ctx.moveTo(0, i); ctx.lineTo(SIZE, i); ctx.stroke(); }
  for (const [position, radius, color, fill] of [[s.goal, 24, '#73dbc3', false], [s.object, OBJECT_RADIUS, '#f4a35a', true], [s.robot, ROBOT_RADIUS, '#80bfff', true]]) {
    ctx.beginPath(); ctx.arc(position.x, position.y, radius, 0, Math.PI * 2); ctx.fillStyle = color; ctx.strokeStyle = color; ctx.lineWidth = 2; fill ? ctx.fill() : ctx.stroke();
  }
  ctx.restore();
}
function frame(s) { draw(small, s); return small.toDataURL('image/png'); }
function refresh() {
  draw(canvas, state);
  status.textContent = `Step ${state.tick}\nGoal distance: ${distanceToGoal(state).toFixed(1)} pixels\n${success(state) ? 'Goal reached' : 'Goal not reached'}\nRecorded: ${total()} / 500 transitions in ${episodes.length} episodes`;
  document.querySelector('#record').textContent = recording ? 'Stop recording' : 'Record episode';
  document.querySelector('#record').setAttribute('aria-pressed', String(recording));
  document.querySelector('#random').textContent = timer ? 'Stop random policy' : 'Run random policy';
  document.querySelector('#random').setAttribute('aria-pressed', String(Boolean(timer)));
}
function act(action, source = 'external') {
  const before = state, after = step(before, action);
  if (recording) {
    activeEpisode.transitions.push({ source, action, before, after, observation: frame(before), nextObservation: frame(after), distanceBefore: distanceToGoal(before), distanceAfter: distanceToGoal(after), success: success(after) });
    if (total() >= 500) { recording = false; notice.textContent = 'Recording limit reached. Save your trace.'; }
  }
  state = after; refresh(); return structuredClone(state);
}
function stopRandom() { if (timer) clearInterval(timer); timer = null; }
function reset(seed) { const fresh = initialState(seed); stopRandom(); recording = false; activeEpisode = null; state = fresh; refresh(); return structuredClone(state); }
function exportTrace() { return structuredClone({ schema: 'little-robot-trace/v1', environment: { version: 1, size: SIZE, observationSize: 64, robotRadius: ROBOT_RADIUS, objectRadius: OBJECT_RADIUS, actionVectors: ACTIONS, successDistance: 8, dynamics: 'kinematic circle pushing; blocked wall moves rejected' }, episodes }); }
for (const button of document.querySelectorAll('[data-action]')) button.addEventListener('click', () => act(button.dataset.action, 'human'));
canvas.addEventListener('keydown', event => {
  const action = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right' }[event.key];
  if (action) { event.preventDefault(); act(action, 'human'); }
});
document.querySelector('#reset').onclick = () => { try { const input = document.querySelector('#seed'); if (!input.checkValidity() || !input.value) throw new Error('Enter a valid uint32 seed.'); reset(Number(input.value)); notice.textContent = 'World reset. Earlier recordings are still available to save.'; } catch (error) { notice.textContent = error.message; } };
document.querySelector('#record').onclick = () => {
  if (recording) recording = false;
  else if (total() >= 500) notice.textContent = 'Trace is full. Save it; reload the page to start a new trace.';
  else { recording = true; activeEpisode = { id: ++episodeCounter, initialState: structuredClone(state), transitions: [] }; episodes.push(activeEpisode); notice.textContent = 'Recording actual transitions, including image observations.'; }
  refresh();
};
document.querySelector('#random').onclick = () => {
  if (timer) stopRandom();
  else { const actions = Object.keys(ACTIONS); timer = setInterval(() => act(actions[Math.floor(Math.random() * actions.length)], 'random'), 100); notice.textContent = 'Random actions only. No learned controller.'; }
  refresh();
};
document.querySelector('#save').onclick = () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(exportTrace())], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url; link.download = 'robot-trace.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  notice.textContent = 'Trace download requested. Recording remains in memory.';
};
window.addEventListener('beforeunload', event => { if (total()) { event.preventDefault(); event.returnValue = ''; } });
window.robotLab = Object.freeze({ observe: () => ({ state: structuredClone(state), image: frame(state) }), step: act, reset, exportTrace });
refresh();
