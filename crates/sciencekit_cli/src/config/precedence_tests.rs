//! Unit tests for configuration layering independent of the binary.

use super::{SKConfigurationOverrides, sk_default_configuration, sk_resolve_configuration};

#[test]
fn each_layer_overwrites_only_the_values_it_sets() {
    let defaults = sk_default_configuration();
    assert_eq!(defaults.alpha, 1.0);
    assert_eq!(defaults.estimator, "ridge");

    let environment = SKConfigurationOverrides {
        alpha: Some(3.0),
        ..SKConfigurationOverrides::default()
    };
    let flags = SKConfigurationOverrides {
        alpha: Some(4.0),
        estimator: Some("linear_regression".to_string()),
        ..SKConfigurationOverrides::default()
    };
    let resolved = sk_resolve_configuration(None, &environment, &flags);
    assert_eq!(resolved.alpha, 4.0);
    assert_eq!(resolved.estimator, "linear_regression");
    assert_eq!(resolved.metrics, vec!["r2".to_string()]);
}

#[test]
fn a_layer_merges_onto_another_with_its_own_values_winning() {
    let mut base = SKConfigurationOverrides {
        alpha: Some(1.0),
        model: Some("base.safetensors".to_string()),
        ..SKConfigurationOverrides::default()
    };
    base.merge(SKConfigurationOverrides {
        alpha: Some(5.0),
        ..SKConfigurationOverrides::default()
    });
    assert_eq!(base.alpha, Some(5.0));
    assert_eq!(base.model.as_deref(), Some("base.safetensors"));
}
