//! The stable tensor-name sets each covered family exports
//! (spec `native-model-export`, requirement "Safetensors native import").
//!
//! Exact-set validation is intentional: a missing tensor means a truncated
//! export, an extra tensor means a different estimator or a newer format.

pub(crate) const LINEAR_REGRESSION: &[&str] = &[
    "linear.coefficients",
    "linear.intercept",
    "linear.rank",
    "linear.singular_values",
];

pub(crate) const RIDGE: &[&str] = &[
    "ridge.alphas",
    "ridge.coefficients",
    "ridge.intercepts",
    "ridge.singular_values",
];

pub(crate) const SGD_REGRESSOR: &[&str] =
    &["sgd_regressor.coefficients", "sgd_regressor.intercept"];

pub(crate) const SGD_CLASSIFIER: &[&str] =
    &["sgd_classifier.coefficients", "sgd_classifier.intercepts"];

pub(crate) const LOGISTIC_REGRESSION: &[&str] = &["logistic.coefficients", "logistic.intercepts"];

pub(crate) const DECISION_TREE_REQUIRED: &[&str] = &["tree.feature_importances"];

pub(crate) const FOREST_IMPORTANCE: &str = "forest.feature_importances";

pub(crate) const STANDARD_SCALER: &[&str] = &[
    "standard_scaler.mean",
    "standard_scaler.scale",
    "standard_scaler.moments_mean",
    "standard_scaler.sum_squared_deviations",
];

pub(crate) const MINMAX_SCALER: &[&str] = &[
    "minmax_scaler.minimum",
    "minmax_scaler.maximum",
    "minmax_scaler.data_range",
    "minmax_scaler.scale",
];

pub(crate) const ROBUST_SCALER: &[&str] = &["robust_scaler.center", "robust_scaler.scale"];

pub(crate) const KMEANS: &[&str] = &["kmeans.cluster_centers", "kmeans.labels"];
