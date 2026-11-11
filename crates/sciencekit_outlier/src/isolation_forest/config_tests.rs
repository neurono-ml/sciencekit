//! Companion tests for Isolation Forest configuration policies.

use crate::{
    SK_ISOLATION_AUTO_OFFSET, SK_ISOLATION_DEFAULT_MAXIMUM_SAMPLES, SKContamination,
    SKIsolationMaximumSamples,
};

#[test]
fn config_maximum_samples_resolves_auto_count_and_fraction() {
    assert_eq!(SK_ISOLATION_DEFAULT_MAXIMUM_SAMPLES, 256);
    assert_eq!(SK_ISOLATION_AUTO_OFFSET, -0.5);
    assert_eq!(
        SKIsolationMaximumSamples::<f64>::Auto.resolved_count(40),
        40
    );
    assert_eq!(
        SKIsolationMaximumSamples::<f64>::Auto.resolved_count(1000),
        256
    );
    assert_eq!(
        SKIsolationMaximumSamples::<f64>::Count(32).resolved_count(100),
        32
    );
    assert_eq!(
        SKIsolationMaximumSamples::<f64>::Fraction(0.25).resolved_count(80),
        20
    );
    assert_eq!(
        SKIsolationMaximumSamples::<f64>::Fraction(0.001).resolved_count(10),
        1
    );
    assert_eq!(SKContamination::<f64>::Auto, SKContamination::Auto);
}
