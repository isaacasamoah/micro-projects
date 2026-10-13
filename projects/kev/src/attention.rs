use std::{array, error::Error, fmt};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionError {
    EmptySequence,
    EmptyDimension,
    NonFinite,
}

impl fmt::Display for AttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::EmptySequence => "attention needs at least one input vector",
            Self::EmptyDimension => "feature dimensions must be greater than zero",
            Self::NonFinite => "attention inputs, parameters and results must be finite",
        })
    }
}

impl Error for AttentionError {}

/// Weight rows are input features; columns are output features.
pub fn project<const INPUT: usize, const OUTPUT: usize>(
    input: &[f64; INPUT],
    weights: &[[f64; OUTPUT]; INPUT],
) -> [f64; OUTPUT] {
    array::from_fn(|column| {
        input
            .iter()
            .zip(weights)
            .map(|(x, row)| x * row[column])
            .sum()
    })
}

pub fn softmax(scores: &[f64]) -> Result<Vec<f64>, AttentionError> {
    if scores.is_empty() {
        return Err(AttentionError::EmptySequence);
    }
    if scores.iter().any(|x| !x.is_finite()) {
        return Err(AttentionError::NonFinite);
    }
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials: Vec<_> = scores.iter().map(|score| (score - maximum).exp()).collect();
    let total: f64 = exponentials.iter().sum();
    Ok(exponentials.iter().map(|x| x / total).collect())
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionHead<const INPUT: usize, const KEY: usize, const VALUE: usize> {
    pub query_weights: [[f64; KEY]; INPUT],
    pub key_weights: [[f64; KEY]; INPUT],
    pub value_weights: [[f64; VALUE]; INPUT],
}

/// Rows in scores and weights are querying tokens; columns are attended tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionTrace<const KEY: usize, const VALUE: usize> {
    pub queries: Vec<[f64; KEY]>,
    pub keys: Vec<[f64; KEY]>,
    pub values: Vec<[f64; VALUE]>,
    pub scores: Vec<Vec<f64>>,
    pub weights: Vec<Vec<f64>>,
    pub output: Vec<[f64; VALUE]>,
}

impl<const INPUT: usize, const KEY: usize, const VALUE: usize> AttentionHead<INPUT, KEY, VALUE> {
    /// Unmasked self-attention. Every token may attend to every input token.
    ///
    /// The input width must match the projection matrices:
    /// ```compile_fail
    /// use kev::attention::AttentionHead;
    /// let head = AttentionHead {
    ///     query_weights: [[1.0, 0.0], [0.0, 1.0]],
    ///     key_weights: [[1.0, 0.0], [0.0, 1.0]],
    ///     value_weights: [[1.0], [1.0]],
    /// };
    /// head.forward(&[[1.0, 2.0, 3.0]]);
    /// ```
    pub fn forward(
        &self,
        inputs: &[[f64; INPUT]],
    ) -> Result<AttentionTrace<KEY, VALUE>, AttentionError> {
        if inputs.is_empty() {
            return Err(AttentionError::EmptySequence);
        }
        if INPUT == 0 || KEY == 0 || VALUE == 0 {
            return Err(AttentionError::EmptyDimension);
        }
        if inputs
            .iter()
            .flatten()
            .chain(self.query_weights.iter().flatten())
            .chain(self.key_weights.iter().flatten())
            .chain(self.value_weights.iter().flatten())
            .any(|x| !x.is_finite())
        {
            return Err(AttentionError::NonFinite);
        }

        let queries: Vec<_> = inputs
            .iter()
            .map(|x| project(x, &self.query_weights))
            .collect();
        let keys: Vec<_> = inputs
            .iter()
            .map(|x| project(x, &self.key_weights))
            .collect();
        let values: Vec<_> = inputs
            .iter()
            .map(|x| project(x, &self.value_weights))
            .collect();
        if queries
            .iter()
            .flatten()
            .chain(keys.iter().flatten())
            .chain(values.iter().flatten())
            .any(|x| !x.is_finite())
        {
            return Err(AttentionError::NonFinite);
        }

        let scale = (KEY as f64).sqrt();
        let scores: Vec<Vec<f64>> = queries
            .iter()
            .map(|query| {
                keys.iter()
                    .map(|key| query.iter().zip(key).map(|(q, k)| q * k).sum::<f64>() / scale)
                    .collect()
            })
            .collect();
        let weights: Vec<_> = scores
            .iter()
            .map(|row| softmax(row))
            .collect::<Result<_, _>>()?;
        let output: Vec<_> = weights
            .iter()
            .map(|row| {
                array::from_fn(|column| {
                    row.iter()
                        .zip(&values)
                        .map(|(weight, value)| weight * value[column])
                        .sum()
                })
            })
            .collect();
        if output.iter().flatten().any(|x: &f64| !x.is_finite()) {
            return Err(AttentionError::NonFinite);
        }

        Ok(AttentionTrace {
            queries,
            keys,
            values,
            scores,
            weights,
            output,
        })
    }
}
