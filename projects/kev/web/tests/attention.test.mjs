import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { homedir } from "node:os";
import { join } from "node:path";
import { initialize, initialParameters, infer } from "../dist/app/model.js";

await initialize({
  module_or_path: await readFile(
    new URL("../dist/generated/kev_bg.wasm", import.meta.url),
  ),
});
const close = (a, b, tolerance = 1e-12) =>
  assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);

for (const uniform of [false, true]) {
  test(`browser trace matches native Rust terminal (${uniform ? "uniform queries" : "default"})`, () => {
    const parameters = initialParameters();
    if (uniform) parameters.query.fill(0);
    const trace = infer(parameters);
    const cargo = join(
      process.env.CARGO_HOME ?? join(homedir(), ".cargo"),
      "bin/cargo",
    );
    const output = execFileSync(
      cargo,
      [
        "run",
        "--quiet",
        "--locked",
        "--manifest-path",
        "../Cargo.toml",
        "--example",
        "attention",
        ...(uniform ? ["--", "--uniform"] : []),
      ],
      { encoding: "utf8" },
    );
    for (const field of ["scores", "weights", "output"]) {
      const rows = [
        ...output.matchAll(new RegExp(`  ${field}:\\s+(\\[[^\\]]+\\])`, "g")),
      ].flatMap((match) => JSON.parse(match[1]));
      assert.equal(rows.length, trace[field].length);
      rows.forEach((value, i) => close(value, trace[field][i], 0.00000051));
    }
    assert.equal(trace.tokenCount, 3);
  });
}

test("Wasm agrees with an independent two-token answer and preserves parameters", () => {
  const parameters = initialParameters();
  parameters.inputs = new Float64Array([1, 0, 0, 1]);
  const before = structuredClone(parameters);
  const trace = infer(parameters);
  const e = Math.exp(1 / Math.sqrt(2));
  const p = e / (e + 1);
  close(trace.weights[0], p);
  close(trace.output[0], p);
  close(trace.output[1], 1 - p);
  close(trace.output[2], 2 * p - 1);
  assert.deepEqual(parameters, before);
});

test("manual matrix and input changes actually reach Rust", () => {
  const parameters = initialParameters();
  const baseline = infer(parameters);
  parameters.query.fill(0);
  const uniform = infer(parameters);
  for (const weight of uniform.weights) close(weight, 1 / 3);
  assert.notDeepEqual(uniform.output, baseline.output);
  parameters.inputs[0] = 2;
  const changed = infer(parameters);
  close(changed.values[0], 2);
  close(changed.output[0], 1);
  close(changed.output[2], 1 / 3);
});

test("browser boundary rejects incomplete rows, wrong matrix sizes, nonfinite data and overflow", () => {
  for (const mutate of [
    (p) => {
      p.inputs = new Float64Array([1]);
    },
    (p) => {
      p.inputs = new Float64Array();
    },
    (p) => {
      p.query = new Float64Array(6);
    },
    (p) => {
      p.value = new Float64Array(4);
    },
    (p) => {
      p.inputs[0] = NaN;
    },
    (p) => {
      p.key[0] = Infinity;
    },
    (p) => {
      p.inputs[0] = Number.MAX_VALUE;
    },
  ]) {
    const parameters = initialParameters();
    mutate(parameters);
    assert.throws(() => infer(parameters));
  }
  assert.equal(infer(initialParameters()).tokenCount, 3);
});

test("copied trace remains valid after later calls release their Rust objects", () => {
  const first = infer(initialParameters());
  const copy = structuredClone(first);
  for (let i = 0; i < 100; i++) {
    const parameters = initialParameters();
    parameters.inputs[0] = i / 20;
    infer(parameters);
  }
  assert.deepEqual(first, copy);
});
