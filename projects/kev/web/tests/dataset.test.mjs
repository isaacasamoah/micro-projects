import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { parseDataset } from "../dist/app/dataset.js";
const seed = JSON.parse(
  await readFile(
    new URL("../../data/graph-decisions.seed.json", import.meta.url),
    "utf8",
  ),
);
test("seed preserves independent edges, no connection, and uncertainty", () => {
  const { cases } = parseDataset(seed);
  assert.equal(cases.length, 12);
  assert.equal(
    cases.find((c) => c.id === "multiple-evidence").expected.links.length,
    2,
  );
  assert.equal(cases.find((c) => c.id === "new-fact").candidates.length, 0);
  assert.deepEqual(
    cases.find((c) => c.id === "unknown-scope").expected.unresolved,
    ["n1", "n2"],
  );
  assert.equal(
    cases.find((c) => c.id === "preference-scope").expected.links.length,
    0,
  );
});
test("invalid references and contradictory target assignments are rejected", () => {
  for (const change of [
    (s) => (s.cases[0].expected.links[0].target = "missing"),
    (s) => s.cases[0].expected.unresolved.push("n1"),
    (s) => (s.cases[0].expected.links[0].relation = "looks_similar"),
    (s) => (s.cases[1].id = s.cases[0].id),
    (s) => s.cases[0].candidates.push(s.cases[0].candidates[0]),
    (s) => (s.status = "validated"),
  ]) {
    const broken = structuredClone(seed);
    change(broken);
    assert.throws(() => parseDataset(broken));
  }
});
