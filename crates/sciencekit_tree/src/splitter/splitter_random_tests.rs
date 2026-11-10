//! `SKSplitter::Random` tests: seeded reproducibility (spec
//! `deterministic-randomness`).

use ndarray::{Array2, array};

use super::{SKScanContext, SKSplitter, sk_best_split};
use crate::criterion::SKTreeCriterion;
use crate::design::SKDesign;

fn fixture() -> (Array2<f64>, Vec<usize>) {
    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
    (features, vec![0, 0, 0, 1, 1, 1])
}

fn context<'a>(classes: &'a [usize], weights: &'a [f64]) -> SKScanContext<'a, f64> {
    SKScanContext::Classification {
        class_indices: classes,
        sample_weights: weights,
        number_of_classes: 2,
        criterion: SKTreeCriterion::Gini,
    }
}

/// The same node seed always yields the same random threshold.
#[test]
fn random_split_is_reproducible_for_a_fixed_seed() {
    let (features, classes) = fixture();
    let design = SKDesign::Dense(features.view());
    let weights = [1.0; 6];
    let first = sk_best_split(
        &design,
        &[0, 1, 2, 3, 4, 5],
        &[0],
        1,
        context(&classes, &weights),
        SKSplitter::Random,
        42,
    )
    .expect("valid boundaries exist");
    let second = sk_best_split(
        &design,
        &[0, 1, 2, 3, 4, 5],
        &[0],
        1,
        context(&classes, &weights),
        SKSplitter::Random,
        42,
    )
    .expect("valid boundaries exist");
    assert_eq!(first, second);
    assert!(
        [1.5, 2.5, 3.5, 4.5, 5.5].contains(&first.threshold()),
        "unexpected threshold {}",
        first.threshold()
    );
}

/// Different node seeds explore different boundaries, so the random splitter
/// is not secretly the best splitter.
#[test]
fn random_split_varies_with_the_seed() {
    let (features, classes) = fixture();
    let design = SKDesign::Dense(features.view());
    let weights = [1.0; 6];
    let thresholds: Vec<f64> = (0..16)
        .map(|seed| {
            sk_best_split(
                &design,
                &[0, 1, 2, 3, 4, 5],
                &[0],
                1,
                context(&classes, &weights),
                SKSplitter::Random,
                seed,
            )
            .expect("valid boundaries exist")
            .threshold()
        })
        .collect();
    let distinct: std::collections::BTreeSet<u64> = thresholds
        .iter()
        .map(|threshold| threshold.to_bits())
        .collect();
    assert!(distinct.len() > 1, "the random splitter never varied");
}
