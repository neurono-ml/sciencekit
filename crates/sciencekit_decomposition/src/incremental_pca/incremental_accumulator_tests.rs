//! Companion tests for incremental PCA configuration and accumulator state.

use sciencekit_common::{SKBackendKind, SKExecutionMode};

use super::accumulator::SKCovarianceAccumulator;
use super::config::{
    DEFAULT_BATCH_SIZE, DEFAULT_MAXIMUM_NUMBER_OF_EPOCHS, SKIncrementalPCAConfiguration,
};

#[test]
fn incremental_configuration_and_accumulator_start_clean() {
    let configuration = SKIncrementalPCAConfiguration::default_configuration();
    assert_eq!(configuration.number_of_components, None);
    assert_eq!(configuration.batch_size, DEFAULT_BATCH_SIZE);
    assert_eq!(configuration.batch_size, 1024);
    assert!(!configuration.whiten);
    assert_eq!(
        configuration.maximum_number_of_epochs,
        DEFAULT_MAXIMUM_NUMBER_OF_EPOCHS
    );
    assert_eq!(configuration.maximum_number_of_epochs, 1);
    assert_eq!(configuration.backend, SKBackendKind::Faer);
    assert_eq!(configuration.execution_mode(), SKExecutionMode::Automatic);

    let accumulator = SKCovarianceAccumulator::<f64>::new(4);
    assert_eq!(accumulator.samples_seen(), 0);
    assert_eq!(accumulator.number_of_features(), 4);
    assert_eq!(accumulator.mean().dim(), 4);
    assert_eq!(accumulator.scatter().dim(), (4, 4));
    assert!(accumulator.mean().iter().all(|&value| value == 0.0));
    assert!(accumulator.scatter().iter().all(|&value| value == 0.0));
}
