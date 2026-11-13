//! The documented native-import coverage matrix
//! (spec `native-model-import`, requirement "ONNX native import").
//!
//! This table is the single source of truth for what conversion supports. The
//! documentation and the tests both read it, so a family that is advertised
//! here is a family a test exercises.

/// One estimator the native importer reconstructs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SKNativeImportCoverage {
    /// The estimator's stable name, exactly as exported.
    pub estimator_name: &'static str,
    /// The native type the conversion produces.
    pub native_type: &'static str,
    /// Whether continued fitting is available after conversion.
    pub retrainable: bool,
    /// A limitation worth documenting, when any.
    pub note: &'static str,
}

/// The covered estimator matrix.
pub fn sk_native_import_coverage() -> &'static [SKNativeImportCoverage] {
    &[
        SKNativeImportCoverage {
            estimator_name: "SKLinearRegression",
            native_type: "SKLinearRegressionModel",
            retrainable: false,
            note: "an OLS state carries no accumulated design; use SKSGDRegressor for continued fitting",
        },
        SKNativeImportCoverage {
            estimator_name: "SKRidge",
            native_type: "SKRidgeModel",
            retrainable: false,
            note: "a ridge fit re-solves the full design",
        },
        SKNativeImportCoverage {
            estimator_name: "SKSGDRegressor",
            native_type: "SKSGDRegressorModel",
            retrainable: true,
            note: "partial_fit continues the schedule and the averaged iterate",
        },
        SKNativeImportCoverage {
            estimator_name: "SKDecisionTreeRegressor",
            native_type: "SKDecisionTreeRegressorModel",
            retrainable: false,
            note: "inference-only after conversion",
        },
        SKNativeImportCoverage {
            estimator_name: "SKRandomForestRegressor",
            native_type: "SKRandomForestRegressorModel",
            retrainable: false,
            note: "inference-only after conversion",
        },
        SKNativeImportCoverage {
            estimator_name: "SKLogisticRegression",
            native_type: "SKLogisticRegressionModel",
            retrainable: true,
            note: "partial_fit re-solves the accumulated design",
        },
        SKNativeImportCoverage {
            estimator_name: "SKSGDClassifier",
            native_type: "SKSGDClassifierModel",
            retrainable: true,
            note: "partial_fit continues the schedule and the averaged iterate",
        },
        SKNativeImportCoverage {
            estimator_name: "SKDecisionTreeClassifier",
            native_type: "SKDecisionTreeClassifierModel",
            retrainable: false,
            note: "inference-only after conversion",
        },
        SKNativeImportCoverage {
            estimator_name: "SKRandomForestClassifier",
            native_type: "SKRandomForestClassifierModel",
            retrainable: false,
            note: "inference-only after conversion",
        },
        SKNativeImportCoverage {
            estimator_name: "SKStandardScaler",
            native_type: "SKStandardScalerModel",
            retrainable: false,
            note: "per-column statistics only",
        },
        SKNativeImportCoverage {
            estimator_name: "SKMinMaxScaler",
            native_type: "SKMinMaxScalerModel",
            retrainable: false,
            note: "per-column statistics only",
        },
        SKNativeImportCoverage {
            estimator_name: "SKRobustScaler",
            native_type: "SKRobustScalerModel",
            retrainable: false,
            note: "per-column statistics only",
        },
        SKNativeImportCoverage {
            estimator_name: "SKKMeans",
            native_type: "SKKMeansModel",
            retrainable: false,
            note: "the centroid table is inference-only",
        },
    ]
}

/// Whether the matrix covers an estimator.
pub fn sk_native_import_supports(estimator_name: &str) -> bool {
    sk_native_import_coverage()
        .iter()
        .any(|row| row.estimator_name == estimator_name)
}
