import { parseDataset } from "./dataset.js";
import type { Dataset, GraphCase } from "./dataset.js";

function node<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  text = "",
  className = "",
): HTMLElementTagNameMap[K] {
  const result = document.createElement(tag);
  result.textContent = text;
  result.className = className;
  return result;
}
function showExample(
  container: HTMLElement,
  example: GraphCase,
  dataset: Dataset,
): void {
  container.replaceChildren();
  const incoming = node("article", "", "data-event");
  incoming.append(
    node(
      "small",
      `${example.anchor.kind} ${example.anchor.id} · ${example.id}`,
    ),
    node("h2", example.title),
    node("p", example.incoming, "data-statement"),
  );
  const label = node("span", example.expected.knowledge_type, "data-label");
  label.title = dataset.knowledge_types[example.expected.knowledge_type] ?? "";
  incoming.append(node("small", "Proposed knowledge type"), label);
  const targets = node("div", "", "data-targets");
  targets.append(node("h3", "What does it connect to?"));
  if (!example.candidates.length)
    targets.append(
      node(
        "p",
        "No candidates supplied. Keep the new fact without inventing a connection.",
      ),
    );
  for (const candidate of example.candidates) {
    const link = example.expected.links.find(
      (link) => link.target === candidate.id,
    );
    const relation =
      link?.relation ??
      (example.expected.unresolved.includes(candidate.id)
        ? "needs clarification"
        : "no connection");
    const card = node("article", "", "data-candidate");
    const badge = node(
      "strong",
      `Incoming → ${relation.replaceAll("_", " ")} → ${candidate.id}`,
    );
    badge.title = link ? (dataset.relations[link.relation] ?? "") : "";
    card.append(badge, node("p", candidate.text));
    targets.append(card);
  }
  const rationale = node("aside", "", "data-rationale");
  rationale.append(
    node("h3", "Why this label?"),
    node("p", example.expected.rationale),
    node(
      "small",
      "Author-proposed answer, not a model prediction. We still need to review whether this is the right decision.",
    ),
  );
  container.append(incoming, targets, rationale);
}
async function initializeData(): Promise<void> {
  const container = document.getElementById("data-content");
  if (!container) return;
  try {
    const response = await fetch("data/graph-decisions.seed.json");
    if (!response.ok)
      throw new Error(`Dataset request failed: ${response.status}`);
    const dataset = parseDataset(await response.json());
    const chooser = node("label", "Explore an example", "data-chooser");
    const select = node("select");
    select.id = "data-example";
    for (const example of dataset.cases) {
      const option = node("option", example.title);
      option.value = example.id;
      select.append(option);
    }
    chooser.append(select);
    const stage = node("div", "", "data-example-stage");
    const note = node(
      "p",
      `${dataset.cases.length} original synthetic examples · draft labels · no train/test split yet. These examples are not connected to the Rust attention demo.`,
      "data-note",
    );
    const download = node("a", "Read the dataset JSON ↗");
    download.href = "data/graph-decisions.seed.json";
    download.target = "_blank";
    download.rel = "noopener";
    container.replaceChildren(chooser, note, stage, download);
    const update = () => {
      const example = dataset.cases.find(
        (example) => example.id === select.value,
      );
      if (example) showExample(stage, example, dataset);
    };
    select.addEventListener("change", update);
    update();
  } catch (error) {
    container.textContent =
      error instanceof Error ? error.message : String(error);
  }
}
void initializeData();
