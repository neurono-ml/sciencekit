//! Dense fitting for the supported families (spec `ffi-abi`).

use ndarray::{ArrayView1, ArrayView2};
use sciencekit_common::{SKBuilder, SKFloat, SKSupervisedFit, SKUnsupervisedFit};
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_math::SKBackendSelection;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use crate::array::SKFfiDataType;
use crate::builder::{SKFfiBuilder, SKFfiEstimatorFamily};
use crate::error_kind::SKFfiError;
use crate::model::definition::{SKFfiFitted, SKFfiModel};

/// Fit the configured family on double-precision dense features.
pub fn sk_ffi_fit_dense_f64(
    builder: &SKFfiBuilder,
    features: ArrayView2<f64>,
    targets: Option<ArrayView1<f64>>,
) -> Result<SKFfiModel, SKFfiError> {
    let fitted = sk_ffi_fit_fitted::<f64>(builder, features, targets)?;
    Ok(SKFfiModel::from_f64(fitted))
}

/// Fit the configured family on single-precision dense features.
pub fn sk_ffi_fit_dense_f32(
    builder: &SKFfiBuilder,
    features: ArrayView2<f32>,
    targets: Option<ArrayView1<f32>>,
) -> Result<SKFfiModel, SKFfiError> {
    let fitted = sk_ffi_fit_fitted::<f32>(builder, features, targets)?;
    Ok(SKFfiModel::from_f32(fitted))
}

/// Dispatch one fit to the configured family.
pub fn sk_ffi_fit_fitted<F: SKFloat + SKBackendSelection>(
    builder: &SKFfiBuilder,
    features: ArrayView2<F>,
    targets: Option<ArrayView1<F>>,
) -> Result<SKFfiFitted<F>, SKFfiError> {
    match builder.family() {
        SKFfiEstimatorFamily::LinearRegression => {
            let targets = targets.ok_or_else(|| {
                SKFfiError::invalid_argument("linear regression requires a targets descriptor")
            })?;
            let estimator = SKLinearRegression::build()
                .include_intercept(builder.include_intercept())
                .backend(builder.backend())
                .build()?;
            Ok(SKFfiFitted::LinearRegression(
                estimator.fit(features, targets.view())?,
            ))
        }
        SKFfiEstimatorFamily::StandardScaler => {
            let estimator = SKStandardScaler::build()
                .with_mean(builder.with_mean())
                .with_std(builder.with_std())
                .backend(builder.backend())
                .build()?;
            Ok(SKFfiFitted::StandardScaler(estimator.fit(features)?))
        }
    }
}

/// The display name of a descriptor dtype.
pub fn dtype_name(dtype: SKFfiDataType) -> &'static str {
    match dtype {
        SKFfiDataType::F32 => "F32",
        SKFfiDataType::F64 => "F64",
        SKFfiDataType::I64 => "I64",
        SKFfiDataType::U8 => "U8",
    }
}
