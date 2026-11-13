//! The documented family/operator coverage matrix (spec `onnx-export`,
//! requirement "Per-family operators").
//!
//! The matrix is the single source of truth for what the exporter supports. It
//! is exposed as data so tests can assert that a family advertised here actually
//! exports, and so the crate documentation can be kept truthful.

/// One row of the coverage matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SKOnnxFamilyCoverage {
    /// The human-readable family name.
    pub family: &'static str,
    /// The estimator names this family covers.
    pub estimators: &'static [&'static str],
    /// The ONNX operators the family emits.
    pub operators: &'static [&'static str],
    /// Whether the exporter covers the family today.
    pub supported: bool,
    /// The documented limitation or the alternative path, when any.
    pub note: &'static str,
}

/// The family/operator coverage matrix.
pub fn sk_onnx_coverage() -> &'static [SKOnnxFamilyCoverage] {
    &[
        SKOnnxFamilyCoverage {
            family: "preprocessing.scalers",
            estimators: &["SKStandardScaler", "SKMinMaxScaler", "SKRobustScaler"],
            operators: &["Sub", "Mul", "Div", "Add", "Clip"],
            supported: true,
            note: "per-column statistics broadcast over the batch dimension",
        },
        SKOnnxFamilyCoverage {
            family: "preprocessing.encoders",
            estimators: &["SKOneHotEncoder"],
            operators: &["Gather", "Equal", "Cast", "Concat"],
            supported: true,
            note: "numeric integer categories and the ignore policy only; nominal string categories and the reject policy return UnsupportedOperation",
        },
    ]
}

/// The estimator names with a working ONNX exporter.
pub fn sk_onnx_supported_families() -> Vec<&'static str> {
    sk_onnx_coverage()
        .iter()
        .filter(|row| row.supported)
        .flat_map(|row| row.estimators.iter().copied())
        .collect()
}
