use std::array;

use crate::attention::AttentionError;

/// Weight rows are input features; columns are output features.
pub(crate) fn project<const INPUT: usize, const OUTPUT: usize>(
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

pub(crate) fn softmax(scores: &[f64]) -> Result<Vec<f64>, AttentionError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_uses_columns_for_output_features() {
        assert_eq!(
            project(&[2.0, 3.0, -1.0], &[[1.0, 2.0], [4.0, -1.0], [2.0, 5.0]]),
            [12.0, -4.0]
        );
    }

    #[test]
    fn softmax_is_stable_and_invariant_to_a_shared_offset() {
        let probabilities = softmax(&[1000.0, 1001.0, 999.0]).unwrap();
        let shifted = softmax(&[0.0, 1.0, -1.0]).unwrap();
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        for (a, b) in probabilities.iter().zip(shifted) {
            assert!((a - b).abs() < 1e-12);
        }
        assert!(probabilities[1] > probabilities[0]);
        assert!(probabilities[0] > probabilities[2]);
        assert_eq!(softmax(&[]), Err(AttentionError::EmptySequence));
        assert_eq!(softmax(&[f64::INFINITY]), Err(AttentionError::NonFinite));
    }
}
