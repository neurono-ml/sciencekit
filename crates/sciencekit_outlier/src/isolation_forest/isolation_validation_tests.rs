//! Companion validation tests for `SKIsolationForest` and `SKIsolationForestModel`.

use ndarray::{Array2, array};
use sciencekit_common::{SKError, SKExecutionMode, SKOutlierPredictor};
use sprs::CsMat;

use crate::{SKContamination, SKIsolationForest, SKIsolationMaximumSamples};

#[test]
fn isolation_validation_rejects_empty_inputs_across_operations() {
    let detector = SKIsolationForest::<f64>::build().build().unwrap();
    let empty_rows_dense = Array2::<f64>::zeros((0, 3));
    let empty_cols_dense = Array2::<f64>::zeros((3, 0));
    let empty_sparse = CsMat::<f64>::empty(sprs::CSR, 3);

    assert!(matches!(
        detector.fit(&empty_rows_dense),
        Err(SKError::EmptyInput { operation: "fit" })
    ));
    assert!(matches!(
        detector.fit(empty_sparse.view()),
        Err(SKError::EmptyInput { operation: "fit" })
    ));
    assert!(matches!(
        detector.fit(&empty_cols_dense),
        Err(SKError::EmptyInput { operation: "fit" })
    ));
    assert!(matches!(
        detector.fit_predict(&empty_rows_dense),
        Err(SKError::EmptyInput {
            operation: "fit_predict"
        })
    ));

    let train = array![[0.0_f64, 1.0], [1.0, 0.0], [2.0, 2.0]];
    let model = detector.fit(&train).unwrap();
    let empty_query = Array2::<f64>::zeros((0, 2));
    assert!(matches!(
        model.predict(&empty_query),
        Err(SKError::EmptyInput {
            operation: "predict"
        })
    ));
    assert!(matches!(
        model.decision_function(&empty_query),
        Err(SKError::EmptyInput {
            operation: "decision_function"
        })
    ));
    assert!(matches!(
        model.score_samples(&empty_query),
        Err(SKError::EmptyInput {
            operation: "score_samples"
        })
    ));
    assert!(matches!(
        model.anomaly_scores(&empty_query),
        Err(SKError::EmptyInput {
            operation: "anomaly_scores"
        })
    ));
}

#[test]
fn isolation_validation_rejects_zero_trees_sample_size_and_contamination_bounds() {
    assert!(matches!(
        SKIsolationForest::<f64>::build().number_of_trees(0).build(),
        Err(SKError::InvalidHyperparameter {
            name: "number_of_trees",
            ..
        })
    ));

    for invalid_samples in [
        SKIsolationMaximumSamples::Count(0),
        SKIsolationMaximumSamples::Fraction(0.0),
        SKIsolationMaximumSamples::Fraction(-0.25),
        SKIsolationMaximumSamples::Fraction(1.25),
        SKIsolationMaximumSamples::Fraction(f64::NAN),
        SKIsolationMaximumSamples::Fraction(f64::INFINITY),
    ] {
        assert!(matches!(
            SKIsolationForest::<f64>::build()
                .maximum_samples(invalid_samples)
                .build(),
            Err(SKError::InvalidHyperparameter {
                name: "maximum_samples",
                ..
            })
        ));
    }

    // Explicit Count exceeding training row count fails at fit time.
    let small_train = array![[0.0_f64, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
    let oversized_samples = SKIsolationForest::<f64>::build()
        .maximum_samples(SKIsolationMaximumSamples::Count(10))
        .build()
        .unwrap();
    assert!(matches!(
        oversized_samples.fit(&small_train),
        Err(SKError::InvalidHyperparameter {
            name: "maximum_samples",
            ..
        })
    ));

    for invalid_contamination in [0.0_f64, -0.1, 0.50001, 0.9, f64::NAN, f64::NEG_INFINITY] {
        assert!(matches!(
            SKIsolationForest::<f64>::build()
                .contamination(SKContamination::Fraction(invalid_contamination))
                .build(),
            Err(SKError::InvalidHyperparameter {
                name: "contamination",
                ..
            })
        ));
    }
}

#[test]
fn isolation_validation_rejects_feature_mismatch_non_finite_csc_and_streaming() {
    let detector = SKIsolationForest::<f64>::build()
        .number_of_trees(8)
        .build()
        .unwrap();
    let train = array![[0.0_f64, 1.0], [1.0, 2.0], [2.0, 3.0], [10.0, -10.0]];
    let model = detector.fit(&train).unwrap();

    let wrong_dense = Array2::<f64>::zeros((3, 5));
    let wrong_sparse = CsMat::<f64>::zero((2, 1));
    assert!(matches!(
        SKOutlierPredictor::predict(&model, &wrong_dense),
        Err(SKError::ShapeMismatch {
            ref expected,
            ref found,
        }) if expected == &[3, 2] && found == &[3, 5]
    ));
    assert!(matches!(
        model.score_samples(wrong_sparse.view()),
        Err(SKError::ShapeMismatch {
            ref expected,
            ref found,
        }) if expected == &[2, 2] && found == &[2, 1]
    ));

    // Non-finite in dense (standard and transposed layouts) and sparse CSR.
    let bad_dense = array![[0.0_f64, 1.0], [2.0, f64::NAN]];
    assert!(matches!(
        detector.fit(&bad_dense),
        Err(SKError::NonFiniteValue { position: 3 })
    ));
    let transposed = array![[0.0_f64, 2.0], [f64::INFINITY, 3.0]].reversed_axes();
    assert!(matches!(
        model.decision_function(transposed.view()),
        Err(SKError::NonFiniteValue { position: 1 })
    ));
    let bad_sparse = CsMat::new(
        (2, 3),
        vec![0, 1, 2],
        vec![0, 2],
        vec![1.0_f64, f64::NEG_INFINITY],
    );
    assert!(matches!(
        detector.fit(bad_sparse.view()),
        Err(SKError::NonFiniteValue { position: 5 })
    ));
    let bad_query_sparse = CsMat::new((1, 2), vec![0, 1], vec![1], vec![f64::NAN]);
    assert!(matches!(
        model.predict(bad_query_sparse.view()),
        Err(SKError::NonFiniteValue { position: 1 })
    ));

    // CSC sparse view and OutOfCoreStreaming mode are rejected cleanly.
    let csc = CsMat::new_csc((2, 2), vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 2.0]);
    assert!(matches!(
        detector.fit(csc.view()),
        Err(SKError::UnsupportedRepresentation {
            representation: "csc",
            ..
        })
    ));
    let streaming = SKIsolationForest::<f64>::build()
        .execution_mode(SKExecutionMode::OutOfCoreStreaming)
        .build()
        .unwrap();
    assert!(matches!(
        streaming.fit(&train),
        Err(SKError::ExecutionModeIncompatible {
            mode: "out-of-core-streaming",
            pattern: "random-access",
        })
    ));
}
