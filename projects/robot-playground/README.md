# Little robot lab

Can we collect an understandable visual action/outcome dataset from a tiny robot world?

A standalone browser environment: move a blue robot to push an orange puck into a green goal. Manual control, a random policy, image observations and trace export are implemented. No learned controller, JEPA, language model training, or real robot is implemented.

## Run

From this directory, with Python 3 installed:

```sh
python3 -m http.server 8794 --bind 127.0.0.1
```

Open http://127.0.0.1:8794 in a browser. There are no JavaScript dependencies or build steps. Press Record episode, move with the buttons or focus the canvas and use arrow keys, then Save trace. Reset preserves previous recordings and ends the current episode. Reloading clears in-memory data; export first. The recording cap is 500 transitions per page session.

## Environment

The square world is 256 units wide. A radius-10 robot moves by four units in one of four directions or waits. Contact pushes a radius-16 puck. Goal success means puck-center distance at most eight units; the goal ring shows the extent that contains the puck on success. There is no friction, inertia, rotation or obstacle navigation. Wall-constrained moves that cannot resolve overlap are rejected. Success is a label, not an automatic terminal state; you can move out of the goal again.

World reset is seeded. The random baseline uses browser randomness, but all chosen actions are recorded and can be replayed deterministically with the pure `step` function. Seeds vary initial positions, not physics.

## Trace and controller interface

JSON exports use `little-robot-trace/v1`. Each episode has an initial state and transitions containing the action, source (human/random/external), before/after states, 64×64 PNG data URLs, distances and success. The export includes action vectors and environment constants. The state labels are ground truth from the simulator, not inferred by a model. Images include the goal ring, grid, robot and puck.

An agent or browser controller can use:

```js
robotLab.observe()         // cloned state and a PNG observation
robotLab.step('right')     // same real environment transition as the UI
robotLab.reset(42)         // resets world, stops recording and random control
robotLab.exportTrace()    // cloned current dataset
```

Only steps taken while recording are stored. The UI starts/stops recording. Keep human, random and future model runs identifiable; do not label random actions as demonstrations of successful behavior.

## Verify

Node 18+:

```sh
node --test world.test.mjs
```

Checks cover seed/replay determinism, input immutability, real pushing versus a wait control, goal detection, blocked wall contact and robot bounds. Browser acceptance also checks movement, actual PNG trace recording, reset preservation and export shape.

## Next experiment

Collect successful pushes and random interactions, splitting training and evaluation by whole episode and held-out seeds. First predict one-step state changes, then try a small image encoder with an action-conditioned latent predictor. A JEPA needs an explicit anti-collapse mechanism; this environment does not implement one. A language-model controller would instead select a discrete action from observations. Compare either learned policy with random and wait baselines on the same starts/goals and step budget.

Inspired by reading [LeWAM](https://le-wam.github.io/) by Minghao Fu, Tavis Siebert, Eryk Halicki and Randall Balestriero. This is original toy-environment code, not LeWAM code, its Push-T benchmark, or a reproduction of its results. The circles and simplified contact mechanics deliberately bound this first experiment.
