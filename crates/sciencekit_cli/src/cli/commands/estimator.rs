//! Estimator fitting and transformer application shared by the commands.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sciencekit::linear_model::{SKLinearRegression, SKRidge};
use sciencekit::preprocessing::{SKMinMaxScaler, SKRobustScaler, SKStandardScaler};
use sciencekit::{
    SKBackendKind, SKBuilder, SKExportable, SKFeatureTransformer, SKModelState, SKSupervisedFit,
    SKUnsupervisedFit,
};

use crate::error::{SKCliError, sk_classify_library_error};
use crate::workflow::schema::SKWorkflowPipelineStep;

/// Whether an estimator name denotes a transformer rather than a predictor.
pub fn sk_is_transformer(name: &str) -> bool {
    matches!(name, "standard_scaler" | "minmax_scaler" | "robust_scaler")
}

fn sk_export<M: SKExportable<f64>>(model: &M) -> Result<SKModelState<f64>, SKCliError> {
    model
        .export_state()
        .map_err(|error| sk_classify_library_error("model export", error))
}

/// Fit a regression estimator and return its exportable state.
pub fn sk_fit_regressor(
    estimator: &str,
    alpha: f64,
    backend: SKBackendKind,
    features: ArrayView2<'_, f64>,
    targets: ArrayView1<'_, f64>,
) -> Result<SKModelState<f64>, SKCliError> {
    match estimator {
        "ridge" => {
            let model = SKRidge::build()
                .alpha(alpha)
                .backend(backend)
                .build()
                .map_err(|error| sk_classify_library_error("ridge build", error))?
                .fit(features, targets)
                .map_err(|error| sk_classify_library_error("ridge fit", error))?;
            sk_export(&model)
        }
        "linear_regression" => {
            let model = SKLinearRegression::build()
                .backend(backend)
                .build()
                .map_err(|error| sk_classify_library_error("linear regression build", error))?
                .fit(features, targets)
                .map_err(|error| sk_classify_library_error("linear regression fit", error))?;
            sk_export(&model)
        }
        other => Err(SKCliError::validation(
            "configuration.estimator",
            format!("`{other}` is not a regression estimator; use ridge or linear_regression"),
        )),
    }
}

fn sk_fit_scaler(
    transformer: &str,
    backend: SKBackendKind,
    features: ArrayView2<'_, f64>,
    with_mean: bool,
    with_std: bool,
) -> Result<Array2<f64>, SKCliError> {
    match transformer {
        "standard_scaler" => {
            let model = SKStandardScaler::build()
                .with_mean(with_mean)
                .with_std(with_std)
                .backend(backend)
                .build()
                .map_err(|error| sk_classify_library_error("standard scaler build", error))?
                .fit(features)
                .map_err(|error| sk_classify_library_error("standard scaler fit", error))?;
            model
                .transform(features)
                .map_err(|error| sk_classify_library_error("standard scaler transform", error))
        }
        "minmax_scaler" => {
            let model = SKMinMaxScaler::<f64>::build()
                .backend(backend)
                .build()
                .map_err(|error| sk_classify_library_error("minmax scaler build", error))?
                .fit(features)
                .map_err(|error| sk_classify_library_error("minmax scaler fit", error))?;
            model
                .transform(features)
                .map_err(|error| sk_classify_library_error("minmax scaler transform", error))
        }
        "robust_scaler" => {
            let model = SKRobustScaler::<f64>::build()
                .backend(backend)
                .build()
                .map_err(|error| sk_classify_library_error("robust scaler build", error))?
                .fit(features)
                .map_err(|error| sk_classify_library_error("robust scaler fit", error))?;
            model
                .transform(features)
                .map_err(|error| sk_classify_library_error("robust scaler transform", error))
        }
        other => Err(SKCliError::validation(
            "configuration.estimator",
            format!("`{other}` is not a supported transformer"),
        )),
    }
}

/// Fit a transformer and return its exportable state.
pub fn sk_fit_transformer(
    transformer: &str,
    backend: SKBackendKind,
    features: ArrayView2<'_, f64>,
) -> Result<SKModelState<f64>, SKCliError> {
    match transformer {
        "standard_scaler" => {
            let model = SKStandardScaler::build()
                .backend(backend)
                .build()
                .map_err(|error| sk_classify_library_error("standard scaler build", error))?
                .fit(features)
                .map_err(|error| sk_classify_library_error("standard scaler fit", error))?;
            sk_export(&model)
        }
        "minmax_scaler" => {
            let model = SKMinMaxScaler::<f64>::build()
                .backend(backend)
                .build()
                .map_err(|error| sk_classify_library_error("minmax scaler build", error))?
                .fit(features)
                .map_err(|error| sk_classify_library_error("minmax scaler fit", error))?;
            sk_export(&model)
        }
        "robust_scaler" => {
            let model = SKRobustScaler::<f64>::build()
                .backend(backend)
                .build()
                .map_err(|error| sk_classify_library_error("robust scaler build", error))?
                .fit(features)
                .map_err(|error| sk_classify_library_error("robust scaler fit", error))?;
            sk_export(&model)
        }
        other => Err(SKCliError::validation(
            "configuration.estimator",
            format!("`{other}` is not a supported transformer"),
        )),
    }
}

/// Apply one pipeline step, returning the transformed feature matrix.
pub fn sk_apply_pipeline_step(
    step: &SKWorkflowPipelineStep,
    backend: SKBackendKind,
    features: &Array2<f64>,
) -> Result<Array2<f64>, SKCliError> {
    sk_fit_scaler(
        &step.transformer,
        backend,
        features.view(),
        step.with_mean.unwrap_or(true),
        step.with_std.unwrap_or(true),
    )
}

/// Ensure a target vector exists for a supervised operation.
pub fn sk_require_targets(
    targets: Option<&Array1<f64>>,
) -> Result<ArrayView1<'_, f64>, SKCliError> {
    targets
        .map(|targets| targets.view())
        .ok_or_else(|| SKCliError::usage("a target column is required (`--target`)"))
}
