import {
  project,
  softmax,
  attend,
  forward,
  crossEntropy,
  trainOutput,
} from "./model.mjs";
const $ = (selector) => document.querySelector(selector);
const walkthrough = [
  {
    name: "1. Linear projection",
    fn: project,
    pseudo:
      "For each output feature:\n  combine all input features\n  using one column of weights.\n\nThis same operation makes keys,\nvalues, hidden features and logits.",
    verdict:
      "Agree for this slice: explicit sums expose every multiply/add. Matrix libraries are faster, but would hide the first thing we want to understand.",
  },
  {
    name: "2. Stable softmax",
    fn: softmax,
    pseudo:
      "Shift all scores by the maximum.\nExponentiate each shifted score.\nDivide by their total.\n\nThe outputs are positive and sum to one.",
    verdict:
      "Agree: subtracting the maximum preserves the distribution and avoids overflow. Normalization alone is not evidence of calibrated probabilities.",
  },
  {
    name: "3. One attention head",
    fn: attend,
    pseudo:
      "Compare the query with each key.\nScale the scores by √key dimension.\nTurn scores into attention weights.\nUse them to average the value vectors.",
    verdict:
      "Agree with a limit: this is query pooling over route tokens. A full transformer would add token-wise queries, residuals, normalization and more blocks.",
  },
  {
    name: "4. The forward pass",
    fn: forward,
    pseudo:
      "Encode route features and identity.\nProject keys and values.\nRun both attention heads.\nJoin their results.\nMix through a hidden layer and ReLU.\nProduce three logits and probabilities.\nSelect one allowed route.",
    verdict:
      "Agree for exploration: a closed set of labels makes output structure explicit. Our enum is a JavaScript label list today; Rust will give us a compile-time enum.",
  },
  {
    name: "5. Cross-entropy",
    fn: crossEntropy,
    pseudo:
      "Measure how much probability\nthe model assigned to the target.\n\nLoss = −log(probability of target).\nCompute from logits stably.",
    verdict:
      "Agree for a classification lesson: the target is supplied by you. One example and a lower loss do not establish that the target rule was learned.",
  },
  {
    name: "6. One training update",
    fn: trainOutput,
    pseudo:
      "Run the current scene.\nCompare probabilities with its label.\nCompute final-layer gradients.\nUpdate final weights and biases.\nRun again and compare the loss.\n\nKeep earlier layers frozen.",
    verdict:
      "Agree with a deliberate boundary: the gradients are real and checked numerically. Full-network backpropagation and a dataset belong in later, individually explained slices.",
  },
];
$("#code-step").innerHTML = walkthrough
  .map((item, i) => `<option value="${i}">${item.name}</option>`)
  .join("");
function showCode() {
  const item = walkthrough[Number($("#code-step").value)];
  $("#pseudo").textContent = item.pseudo;
  $("#code").textContent = item.fn.toString();
  $("#verdict").textContent = item.verdict;
}
$("#code-step").onchange = showCode;

showCode();
