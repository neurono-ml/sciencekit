//! Tests for the owned nominal block (spec `tabular-sources`).

use ndarray::array;
use sciencekit_common::{SK_CATEGORICAL_MISSING_CODE, SKError, sk_canonicalize_labels};

use super::SKNominalMatrixOwned;

/// A code that no label table entry can decode is rejected at construction.
#[test]
fn nominal_block_rejects_codes_outside_the_label_table() {
    let label_table = sk_canonicalize_labels(&["a"]).1;
    let error =
        SKNominalMatrixOwned::new(array![[0_i64], [2]], vec![Some(label_table)]).unwrap_err();
    match error {
        SKError::SchemaMismatch { expected, found } => {
            assert!(expected.contains("label table"));
            assert!(found.contains('2'));
        }
        other => panic!("expected SchemaMismatch, found {other:?}"),
    }
}

/// The missing sentinel is valid even when the table has no entry for it.
#[test]
fn nominal_block_accepts_the_missing_sentinel() {
    let label_table = sk_canonicalize_labels(&["a", "b"]).1;
    let block = SKNominalMatrixOwned::new(
        array![[0_i64], [SK_CATEGORICAL_MISSING_CODE]],
        vec![Some(label_table)],
    )
    .unwrap();
    assert_eq!(block.codes()[[1, 0]], SK_CATEGORICAL_MISSING_CODE);
}

/// A category-coded column carries no label table, so its codes are adopted.
#[test]
fn nominal_block_adopts_category_codes_without_a_label_table() {
    let block = SKNominalMatrixOwned::new(array![[3_i64], [7]], vec![None]).unwrap();
    assert_eq!(block.codes()[[1, 0]], 7);
}
