//! Dense, sparse and mixed horizontal concatenation tests (spec
//! `feature-union`, Task 1.3).

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKDataView, SKError, SKFeatureBlock};
use sciencekit_impute::missing_indicator::{SKMissingFeatureScope, SKMissingIndicator};
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_preprocessing::minmax_scaler::core_implementation::SKMinMaxScaler;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;
use sprs::{CsMat, TriMat};

use crate::{SKFeatureUnion, SKPipelineTransformer};

/// Test transformer that refuses dense inputs, proving the union never
/// densifies a sparse input block before handing it to sparse-capable members.
struct SKSparseOnlyShiftStage {
    shift_columns: usize,
    scale: f64,
}

impl SKPipelineTransformer<f64> for SKSparseOnlyShiftStage {
    type Fitted = ();

    fn fit_stage(&self, block: &SKFeatureBlock<f64>) -> Result<(), SKError> {
        if !block.is_sparse() {
            return Err(SKError::UnsupportedRepresentation {
                representation: "dense",
                suggestion: "pass a CSR sparse block",
            });
        }
        Ok(())
    }

    fn transform_stage(
        &self,
        _fitted: &(),
        block: &SKFeatureBlock<f64>,
    ) -> Result<SKFeatureBlock<f64>, SKError> {
        match block.view() {
            SKDataView::Sparse(sparse) => {
                let mut triplets = TriMat::new((sparse.rows(), sparse.cols() + self.shift_columns));
                sparse
                    .outer_iterator()
                    .enumerate()
                    .for_each(|(row_index, row)| {
                        row.iter().for_each(|(column, &value)| {
                            triplets.add_triplet(
                                row_index,
                                column + self.shift_columns,
                                value * self.scale,
                            );
                        });
                    });
                Ok(SKFeatureBlock::Sparse(triplets.to_csr()))
            }
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "dense",
                suggestion: "pass a CSR sparse block",
            }),
        }
    }
}

#[test]
fn all_dense_members_concatenate_to_a_dense_block() {
    let union = SKFeatureUnion::build()
        .stages((
            SKStandardScaler::build().build().unwrap(),
            SKMinMaxScaler::<f64>::build().build().unwrap(),
        ))
        .build()
        .unwrap();

    let input = SKFeatureBlock::from(array![[0.0_f64, 10.0], [2.0, 30.0]]);
    let output = union.fit(&input).unwrap().transform(&input).unwrap();
    assert!(!output.is_sparse());
    assert_eq!(
        (output.number_of_rows(), output.number_of_columns()),
        (2, 4)
    );
}

fn expect_csr(block: &SKFeatureBlock<f64>) -> sprs::CsMatView<'_, f64> {
    match block.view() {
        SKDataView::Sparse(csr) => {
            assert!(csr.is_csr());
            csr
        }
        _ => panic!("expected CSR output"),
    }
}

#[test]
fn all_sparse_members_concatenate_to_csr_with_shifted_column_offsets() {
    let mut triplets = TriMat::new((3, 2));
    triplets.add_triplet(0, 0, 2.0_f64);
    triplets.add_triplet(0, 1, 5.0);
    triplets.add_triplet(2, 1, 7.0);
    let sparse_input = SKFeatureBlock::Sparse(triplets.to_csr());

    let union = SKFeatureUnion::build()
        .stages((
            SKSimpleImputer::<f64>::build().build().unwrap(),
            SKSparseOnlyShiftStage {
                shift_columns: 1,
                scale: 10.0,
            },
        ))
        .build()
        .unwrap();

    let output = union
        .fit(&sparse_input)
        .unwrap()
        .transform(&sparse_input)
        .unwrap();
    let csr = expect_csr(&output);
    assert_eq!((csr.rows(), csr.cols()), (3, 5));
    let expected = array![
        [2.0_f64, 5.0, 0.0, 20.0, 50.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 7.0, 0.0, 0.0, 70.0]
    ];
    assert_eq!(csr.to_dense(), expected);
}

#[test]
fn mixed_dense_then_sparse_promotes_union_output_to_csr_with_exact_offsets() {
    let input = SKFeatureBlock::from(array![[1.0_f64, f64::NAN], [3.0, 4.0], [5.0, 6.0]]);
    let imputer = SKSimpleImputer::<f64>::build().build().unwrap();
    let indicator = SKMissingIndicator::<f64>::build()
        .sparse_output(true)
        .error_on_new(false)
        .build()
        .unwrap();

    let union = SKFeatureUnion::build()
        .stages((imputer, indicator))
        .build()
        .unwrap();
    let output = union.fit(&input).unwrap().transform(&input).unwrap();
    let csr = expect_csr(&output);
    assert_eq!((csr.rows(), csr.cols()), (3, 3));
    assert_eq!(
        csr.to_dense(),
        array![[1.0_f64, 5.0, 1.0], [3.0, 4.0, 0.0], [5.0, 6.0, 0.0]]
    );
}

#[test]
fn mixed_sparse_then_dense_then_sparse_accumulates_column_offsets_across_three_members() {
    let input = SKFeatureBlock::from(array![[0.0_f64, f64::NAN], [2.0, 4.0]]);
    let first_sparse = SKMissingIndicator::<f64>::build()
        .features(SKMissingFeatureScope::All)
        .sparse_output(true)
        .error_on_new(false)
        .build()
        .unwrap();
    let middle_dense = SKSimpleImputer::<f64>::build().build().unwrap();
    let third_sparse = SKMissingIndicator::<f64>::build()
        .features(SKMissingFeatureScope::MissingOnly)
        .sparse_output(true)
        .error_on_new(false)
        .build()
        .unwrap();

    let union = SKFeatureUnion::build()
        .stages((first_sparse, middle_dense, third_sparse))
        .build()
        .unwrap();
    let output = union.fit(&input).unwrap().transform(&input).unwrap();
    let csr = expect_csr(&output);
    assert_eq!((csr.rows(), csr.cols()), (2, 5));
    let expected: Array2<f64> = array![[0.0, 1.0, 0.0, 4.0, 1.0], [0.0, 0.0, 2.0, 4.0, 0.0]];
    assert_eq!(csr.to_dense(), expected);
}

#[test]
fn sparse_rows_with_no_nonzero_entries_preserve_valid_csr_row_pointers() {
    let input = SKFeatureBlock::Sparse(CsMat::new((3, 2), vec![0, 0, 0, 0], vec![], Vec::new()));
    let union = SKFeatureUnion::build()
        .stages((
            SKSparseOnlyShiftStage {
                shift_columns: 0,
                scale: 1.0,
            },
            SKSparseOnlyShiftStage {
                shift_columns: 1,
                scale: 2.0,
            },
        ))
        .build()
        .unwrap();

    let output = union.fit(&input).unwrap().transform(&input).unwrap();
    let csr = expect_csr(&output);
    assert_eq!((csr.rows(), csr.cols()), (3, 5));
    assert_eq!(csr.nnz(), 0);
}
