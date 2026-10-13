import init, {
  example_inputs,
  example_query,
  example_key,
  example_value,
  forward,
} from "../generated/kev.js";

export interface Parameters {
  inputs: Float64Array;
  query: Float64Array;
  key: Float64Array;
  value: Float64Array;
}

export type MatrixName = "query" | "key" | "value";
export interface Trace {
  tokenCount: number;
  queries: Float64Array;
  keys: Float64Array;
  values: Float64Array;
  scores: Float64Array;
  weights: Float64Array;
  output: Float64Array;
}

export const initialize = init;

export function initialParameters(): Parameters {
  return {
    inputs: example_inputs(),
    query: example_query(),
    key: example_key(),
    value: example_value(),
  };
}

export function infer(parameters: Parameters): Trace {
  const result = forward(
    parameters.inputs,
    parameters.query,
    parameters.key,
    parameters.value,
  );
  try {
    return {
      tokenCount: result.token_count,
      queries: result.queries,
      keys: result.keys,
      values: result.values,
      scores: result.scores,
      weights: result.weights,
      output: result.output,
    };
  } finally {
    result.free();
  }
}

export function at(values: Float64Array, index: number): number {
  const value = values[index];
  if (value === undefined) throw new Error(`Missing trace coordinate ${index}`);
  return value;
}

export function row(
  values: Float64Array,
  index: number,
  width: number,
): number[] {
  return Array.from({ length: width }, (_, column) =>
    at(values, index * width + column),
  );
}
