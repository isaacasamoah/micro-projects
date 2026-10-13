use wasm_bindgen::prelude::*;

use crate::{
    attention::{AttentionHead, AttentionTrace},
    example,
};

#[wasm_bindgen]
pub struct BrowserTrace {
    trace: AttentionTrace<2, 3>,
}

#[wasm_bindgen]
impl BrowserTrace {
    #[wasm_bindgen(getter)]
    pub fn token_count(&self) -> usize {
        self.trace.queries.len()
    }

    #[wasm_bindgen(getter)]
    pub fn queries(&self) -> Vec<f64> {
        self.trace.queries.iter().flatten().copied().collect()
    }

    #[wasm_bindgen(getter)]
    pub fn keys(&self) -> Vec<f64> {
        self.trace.keys.iter().flatten().copied().collect()
    }

    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<f64> {
        self.trace.values.iter().flatten().copied().collect()
    }

    #[wasm_bindgen(getter)]
    pub fn scores(&self) -> Vec<f64> {
        self.trace.scores.iter().flatten().copied().collect()
    }

    #[wasm_bindgen(getter)]
    pub fn weights(&self) -> Vec<f64> {
        self.trace.weights.iter().flatten().copied().collect()
    }

    #[wasm_bindgen(getter)]
    pub fn output(&self) -> Vec<f64> {
        self.trace.output.iter().flatten().copied().collect()
    }
}

fn rows<const WIDTH: usize>(flat: &[f64]) -> Result<Vec<[f64; WIDTH]>, JsError> {
    let chunks = flat.chunks_exact(WIDTH);
    if !chunks.remainder().is_empty() {
        return Err(JsError::new("incomplete matrix row"));
    }
    chunks
        .map(|row| {
            row.try_into()
                .map_err(|_| JsError::new("invalid row width"))
        })
        .collect()
}

fn matrix<const ROWS: usize, const COLS: usize>(
    flat: &[f64],
) -> Result<[[f64; COLS]; ROWS], JsError> {
    rows(flat)?
        .try_into()
        .map_err(|_| JsError::new("incorrect matrix size"))
}

/// Fixed feature widths for this visual, with a variable number of input tokens.
/// Arrays cross the browser boundary in row-major order.
#[wasm_bindgen]
pub fn forward(
    inputs: &[f64],
    query: &[f64],
    key: &[f64],
    value: &[f64],
) -> Result<BrowserTrace, JsError> {
    let head = AttentionHead::<2, 2, 3> {
        query_weights: matrix(query)?,
        key_weights: matrix(key)?,
        value_weights: matrix(value)?,
    };
    let trace = head.forward(&rows::<2>(inputs)?)?;
    Ok(BrowserTrace { trace })
}

#[wasm_bindgen]
pub fn example_inputs() -> Vec<f64> {
    example::INPUTS.into_iter().flatten().collect()
}

#[wasm_bindgen]
pub fn example_query() -> Vec<f64> {
    example::head()
        .query_weights
        .into_iter()
        .flatten()
        .collect()
}

#[wasm_bindgen]
pub fn example_key() -> Vec<f64> {
    example::head().key_weights.into_iter().flatten().collect()
}

#[wasm_bindgen]
pub fn example_value() -> Vec<f64> {
    example::head()
        .value_weights
        .into_iter()
        .flatten()
        .collect()
}
