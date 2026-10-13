//! Synthetic two-feature tokens and hand-set parameters for the walkthrough.

use crate::attention::AttentionHead;

pub const INPUTS: [[f64; 2]; 3] = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

pub fn head() -> AttentionHead<2, 2, 3> {
    AttentionHead {
        query_weights: [[1.0, 0.0], [0.0, 1.0]],
        key_weights: [[1.0, 0.0], [0.0, 1.0]],
        value_weights: [[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]],
    }
}
