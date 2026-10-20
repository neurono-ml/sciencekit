//! Acceptance on lots of data: the out-of-core fit over a streamed matrix.

use ndarray::s;
use sciencekit_common::{SKBuilder, SKExecutionMode, SKFeatureTransformer, SKUnsupervisedFit};

use super::fixtures::{
    SKVectorSource, assert_statistics_agree, chunk_rows, dense_imputation, generated_features,
};
use crate::simple_imputer::SKSimpleImputer;

#[test]
fn a_large_generated_matrix_streams_like_the_concatenated_fit() {
    let features = generated_features(4_000, 8, 9);
    let dense_model = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&features)
        .unwrap();

    let batches = chunk_rows(&features, 512);
    assert_eq!(batches.len(), 8, "the source must really stream");
    let mut source = SKVectorSource::new(batches);
    let streamed_model = SKSimpleImputer::<f64>::build()
        .execution_mode(SKExecutionMode::OutOfCoreStreaming)
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap();

    assert_statistics_agree(streamed_model.statistics(), dense_model.statistics());
    assert_eq!(
        streamed_model.dropped_columns(),
        dense_model.dropped_columns()
    );

    // The two models must also impute the same rows identically.
    let probe = features.slice(s![..100, ..]).to_owned();
    assert_eq!(
        dense_imputation(streamed_model.transform(&probe).unwrap()),
        dense_imputation(dense_model.transform(&probe).unwrap())
    );
}
