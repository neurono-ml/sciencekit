//! Companion tests for `SKIsolationForestBuilder` defaults and hyperparameter validation.

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode};

use crate::{SKContamination, SKIsolationForest, SKIsolationMaximumSamples};

#[test]
fn isolation_builder_defaults_match_specification() {
    let detector = SKIsolationForest::<f64>::build().build().unwrap();
    assert_eq!(detector.number_of_trees(), 100);
    assert_eq!(detector.maximum_samples(), SKIsolationMaximumSamples::Auto);
    assert_eq!(detector.contamination(), SKContamination::Auto);
    assert_eq!(detector.maximum_features(), 1.0);
    assert!(!detector.bootstrap());
    assert_eq!(detector.random_seed(), 0);
    assert_eq!(detector.backend(), SKBackendKind::Faer);
    assert_eq!(detector.execution_mode(), SKExecutionMode::Automatic);
}

#[test]
fn isolation_builder_custom_setters_and_trait_execution_mode_propagate() {
    let mut builder = SKIsolationForest::<f32>::build()
        .number_of_trees(50)
        .maximum_samples(SKIsolationMaximumSamples::Count(64))
        .contamination(SKContamination::Fraction(0.1))
        .maximum_features(0.75)
        .bootstrap(true)
        .random_seed(42)
        .backend(SKBackendKind::MatrixMultiply);
    SKBuilder::execution_mode(&mut builder, SKExecutionMode::InProcessSynchronous);
    let detector = builder.build().unwrap();

    assert_eq!(detector.number_of_trees(), 50);
    assert_eq!(
        detector.maximum_samples(),
        SKIsolationMaximumSamples::Count(64)
    );
    assert_eq!(detector.contamination(), SKContamination::Fraction(0.1));
    assert!((detector.maximum_features() - 0.75).abs() < 1e-6);
    assert!(detector.bootstrap());
    assert_eq!(detector.random_seed(), 42);
    assert_eq!(detector.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        detector.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );
}

#[test]
fn isolation_builder_rejects_invalid_hyperparameters() {
    let zero_trees = SKIsolationForest::<f64>::build()
        .number_of_trees(0)
        .build()
        .unwrap_err();
    assert!(matches!(
        zero_trees,
        SKError::InvalidHyperparameter {
            name: "number_of_trees",
            ..
        }
    ));

    for invalid_samples in [
        SKIsolationMaximumSamples::Count(0),
        SKIsolationMaximumSamples::Fraction(0.0),
        SKIsolationMaximumSamples::Fraction(-0.1),
        SKIsolationMaximumSamples::Fraction(1.1),
        SKIsolationMaximumSamples::Fraction(f64::NAN),
    ] {
        let error = SKIsolationForest::<f64>::build()
            .maximum_samples(invalid_samples)
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            SKError::InvalidHyperparameter {
                name: "maximum_samples",
                ..
            }
        ));
    }

    for invalid_contamination in [0.0_f64, -0.05, 0.5001, 1.0, f64::NAN, f64::INFINITY] {
        let error = SKIsolationForest::<f64>::build()
            .contamination(SKContamination::Fraction(invalid_contamination))
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            SKError::InvalidHyperparameter {
                name: "contamination",
                ..
            }
        ));
    }

    // Exact boundary 0.5 is accepted.
    assert!(
        SKIsolationForest::<f64>::build()
            .contamination(SKContamination::Fraction(0.5))
            .build()
            .is_ok()
    );

    for invalid_features in [0.0_f64, -0.2, 1.01, f64::NAN, f64::INFINITY] {
        let error = SKIsolationForest::<f64>::build()
            .maximum_features(invalid_features)
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            SKError::InvalidHyperparameter {
                name: "maximum_features",
                ..
            }
        ));
    }
}
