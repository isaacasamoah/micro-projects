export const knowledgeTypes = ["domain", "operational", "preference"] as const;
export const relations = [
  "supports",
  "refines",
  "supersedes",
  "contradicts",
  "depends_on",
] as const;
type KnowledgeType = (typeof knowledgeTypes)[number];
type Relation = (typeof relations)[number];
export interface GraphCase {
  id: string;
  family: string;
  title: string;
  anchor: { kind: string; id: string };
  incoming: string;
  candidates: { id: string; text: string }[];
  expected: {
    knowledge_type: KnowledgeType;
    links: { target: string; relation: Relation }[];
    unresolved: string[];
    rationale: string;
  };
}
export interface Dataset {
  cases: GraphCase[];
  knowledge_types: Record<string, string>;
  relations: Record<string, string>;
}
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("Expected an object in the seed dataset.");
  return Object.fromEntries(Object.entries(value));
}
function text(value: unknown): string {
  if (typeof value !== "string" || !value.trim())
    throw new Error("Expected nonempty dataset text.");
  return value;
}
function list(value: unknown): unknown[] {
  if (!Array.isArray(value)) throw new Error("Expected a dataset list.");
  return value;
}
function choice<T extends string>(value: unknown, options: readonly T[]): T {
  const found = options.find((option) => option === value);
  if (!found) throw new Error("Unknown decision label.");
  return found;
}
function glossary(
  value: unknown,
  keys: readonly string[],
): Record<string, string> {
  const source = object(value);
  return Object.fromEntries(keys.map((key) => [key, text(source[key])]));
}
export function parseDataset(value: unknown): Dataset {
  const source = object(value);
  if (
    source.schema !== "kev.graph-decisions.v0" ||
    source.status !== "authored-draft" ||
    source.split !== "unassigned"
  )
    throw new Error("Expected an unassigned, authored draft dataset.");
  const ids = new Set<string>();
  const cases = list(source.cases).map((item) => {
    const c = object(item),
      expected = object(c.expected),
      anchor = object(c.anchor);
    const id = text(c.id);
    if (ids.has(id)) throw new Error("Duplicate example ID.");
    ids.add(id);
    const candidates = list(c.candidates).map((item) => {
      const n = object(item);
      return { id: text(n.id), text: text(n.text) };
    });
    const nodeIds = new Set(candidates.map((n) => n.id));
    if (nodeIds.size !== candidates.length)
      throw new Error("Duplicate candidate ID.");
    const assigned = new Set<string>();
    const target = (value: unknown): string => {
      const id = text(value);
      if (!nodeIds.has(id) || assigned.has(id))
        throw new Error("Missing or multiply labelled candidate.");
      assigned.add(id);
      return id;
    };
    const links = list(expected.links).map((item) => {
      const l = object(item);
      return {
        target: target(l.target),
        relation: choice(l.relation, relations),
      };
    });
    const unresolved = list(expected.unresolved).map(target);
    return {
      id,
      family: text(c.family),
      title: text(c.title),
      anchor: { kind: text(anchor.kind), id: text(anchor.id) },
      incoming: text(c.incoming),
      candidates,
      expected: {
        knowledge_type: choice(expected.knowledge_type, knowledgeTypes),
        links,
        unresolved,
        rationale: text(expected.rationale),
      },
    };
  });
  if (!cases.length) throw new Error("The seed dataset has no examples.");
  return {
    cases,
    knowledge_types: glossary(source.knowledge_types, knowledgeTypes),
    relations: glossary(source.relations, relations),
  };
}
