//! Tests for the documented coverage matrix.

use std::collections::BTreeSet;

use super::{sk_onnx_coverage, sk_onnx_supported_families};

#[test]
fn coverage_rows_are_internally_consistent() {
    let mut families = BTreeSet::new();
    let mut estimators = BTreeSet::new();
    for row in sk_onnx_coverage() {
        assert!(
            families.insert(row.family),
            "the family `{}` appears twice",
            row.family
        );
        assert!(
            !row.estimators.is_empty(),
            "the family `{}` names no estimator",
            row.family
        );
        for estimator in row.estimators {
            assert!(
                estimators.insert(*estimator),
                "the estimator `{estimator}` appears twice"
            );
        }
        if row.supported {
            assert!(
                !row.operators.is_empty(),
                "the supported family `{}` names no operator",
                row.family
            );
        }
    }
}

#[test]
fn supported_families_match_the_matrix() {
    let expected: Vec<&str> = sk_onnx_coverage()
        .iter()
        .filter(|row| row.supported)
        .flat_map(|row| row.estimators.iter().copied())
        .collect();
    assert_eq!(sk_onnx_supported_families(), expected);
}
