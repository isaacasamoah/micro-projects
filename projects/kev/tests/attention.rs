use kev::attention::{AttentionError, AttentionHead, project, softmax};

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

fn head() -> AttentionHead<2, 2, 3> {
    AttentionHead {
        query_weights: [[1.0, 0.0], [0.0, 1.0]],
        key_weights: [[1.0, 0.0], [0.0, 1.0]],
        value_weights: [[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]],
    }
}

#[test]
fn projection_uses_weight_columns_for_output_features() {
    assert_eq!(
        project(&[2.0, 3.0, -1.0], &[[1.0, 2.0], [4.0, -1.0], [2.0, 5.0]]),
        [12.0, -4.0]
    );
}

#[test]
fn softmax_is_stable_and_invariant_to_a_shared_offset() {
    let probabilities = softmax(&[1000.0, 1001.0, 999.0]).unwrap();
    let shifted = softmax(&[0.0, 1.0, -1.0]).unwrap();
    close(probabilities.iter().sum(), 1.0);
    for (a, b) in probabilities.iter().zip(shifted) {
        close(*a, b);
    }
    assert!(probabilities[1] > probabilities[0]);
    assert!(probabilities[0] > probabilities[2]);
}

#[test]
fn self_attention_matches_an_independent_two_token_calculation() {
    let model = head();
    let before = model.clone();
    let trace = model.forward(&[[1.0, 0.0], [0.0, 1.0]]).unwrap();
    let same = (1.0 / 2.0_f64.sqrt()).exp();
    let p = same / (same + 1.0);
    close(trace.scores[0][0], 1.0 / 2.0_f64.sqrt());
    close(trace.scores[0][1], 0.0);
    close(trace.weights[0][0], p);
    close(trace.weights[1][0], 1.0 - p);
    close(trace.output[0][0], p);
    close(trace.output[0][1], 1.0 - p);
    close(trace.output[0][2], 2.0 * p - 1.0);
    assert_eq!(model, before);
}

#[test]
fn zero_queries_give_uniform_attention_and_mean_values() {
    let mut model = head();
    model.query_weights = [[0.0; 2]; 2];
    let trace = model
        .forward(&[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        .unwrap();
    for row in &trace.weights {
        for weight in row {
            close(*weight, 1.0 / 3.0);
        }
    }
    for output in &trace.output {
        close(output[0], 2.0 / 3.0);
        close(output[1], 2.0 / 3.0);
        close(output[2], 0.0);
    }
}

#[test]
fn attention_without_positions_is_permutation_equivariant() {
    let model = head();
    let inputs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let order = [2, 0, 1];
    let original = model.forward(&inputs).unwrap();
    let permuted = model.forward(&order.map(|i| inputs[i])).unwrap();
    for (i, old_i) in order.into_iter().enumerate() {
        for (j, old_j) in order.into_iter().enumerate() {
            close(permuted.weights[i][j], original.weights[old_i][old_j]);
        }
        for j in 0..3 {
            close(permuted.output[i][j], original.output[old_i][j]);
        }
    }
}

#[test]
fn one_token_gets_its_own_projected_value() {
    let trace = head().forward(&[[2.0, 3.0]]).unwrap();
    assert_eq!(trace.weights, vec![vec![1.0]]);
    assert_eq!(trace.output, vec![[2.0, 3.0, -1.0]]);
}

#[test]
fn empty_or_non_finite_inputs_and_overflow_are_rejected() {
    assert_eq!(head().forward(&[]), Err(AttentionError::EmptySequence));
    assert_eq!(
        head().forward(&[[f64::NAN, 0.0]]),
        Err(AttentionError::NonFinite)
    );
    assert_eq!(
        head().forward(&[[f64::MAX, 1.0]]),
        Err(AttentionError::NonFinite)
    );
    assert_eq!(softmax(&[]), Err(AttentionError::EmptySequence));
    assert_eq!(softmax(&[f64::INFINITY]), Err(AttentionError::NonFinite));
    let empty: AttentionHead<2, 0, 1> = AttentionHead {
        query_weights: [[], []],
        key_weights: [[], []],
        value_weights: [[1.0], [1.0]],
    };
    assert_eq!(
        empty.forward(&[[1.0, 1.0]]),
        Err(AttentionError::EmptyDimension)
    );
    let mut model = head();
    model.value_weights[0][0] = f64::INFINITY;
    assert_eq!(model.forward(&[[1.0, 1.0]]), Err(AttentionError::NonFinite));
}
