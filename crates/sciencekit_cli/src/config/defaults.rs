//! The documented default configuration layer.

/// The estimator used when no layer selects one.
pub const SK_DEFAULT_ESTIMATOR: &str = "ridge";
/// The ridge penalty used when no layer selects one.
pub const SK_DEFAULT_ALPHA: f64 = 1.0;
/// The prediction and matrix output format used when no layer selects one.
pub const SK_DEFAULT_OUTPUT_FORMAT: &str = "csv";
/// The math backend used when no layer selects one.
pub const SK_DEFAULT_BACKEND: &str = "faer";

/// The default metric selection.
pub fn sk_default_metrics() -> Vec<String> {
    vec!["r2".to_string()]
}
