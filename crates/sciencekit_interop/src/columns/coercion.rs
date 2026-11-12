//! Coercion of supported column types to the model scalar (spec
//! `dataset-interop-core`).

use ndarray::Array1;
use num_traits::cast;
use sciencekit_common::{SKError, SKFloat};

use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;

/// Whether an integer survives the trip to `F` and back unchanged.
///
/// Floating-point scalars represent integers exactly only up to their mantissa
/// width (`2^53` for `f64`, `2^24` for `f32`). Promoting any integer value above
/// that bound — in a complete column or in a nullable one — would truncate it
/// silently, so the value is rejected instead.
pub fn sk_integer_is_representable<F: SKFloat>(value: i64) -> bool {
    match cast::<i64, F>(value) {
        Some(converted) => converted.is_finite() && cast::<F, i64>(converted) == Some(value),
        None => false,
    }
}

/// Resolve a null on the numeric path according to the declared policy.
fn sk_scalar_for_null<F: SKFloat>(
    column: &str,
    policy: SKMissingValuePolicy,
) -> Result<F, SKError> {
    match policy {
        SKMissingValuePolicy::Reject => Err(SKError::NullValueInColumn {
            column: column.to_string(),
            policy: "reject",
        }),
        SKMissingValuePolicy::NumericNaN => Ok(F::nan()),
        SKMissingValuePolicy::ConstantFill(value) => {
            // The public enum variant can bypass `constant_fill`, so the value
            // is re-validated here — including after the cast, where a finite
            // `f64` can overflow to an infinite `f32`.
            let converted = cast::<f64, F>(value).filter(|converted| converted.is_finite());
            converted.ok_or_else(|| SKError::InvalidHyperparameter {
                name: "constant_fill_value",
                reason: format!(
                    "the fill value {value} is not representable as a finite model scalar"
                ),
            })
        }
        SKMissingValuePolicy::CategoricalMissingCode => Err(SKError::NullValueInColumn {
            column: column.to_string(),
            policy: "categorical_missing_code",
        }),
    }
}

/// Coerce one numeric column to an owned scalar vector under a policy.
///
/// Supports `f32`, `f64`, `i64` and `bool`. Integer values are checked for
/// exact representability in the scalar before conversion; an unsupported
/// column type is rejected with a suggestion naming the conversion path.
///
/// # Errors
/// Returns [`SKError::UnsupportedColumnType`] for nominal or category-coded
/// columns, [`SKError::NullValueInColumn`] when the policy rejects a null, and
/// [`SKError::Conversion`] for an integer the scalar cannot represent exactly.
pub fn sk_coerce_numeric_column<F: SKFloat>(
    column_name: &str,
    column: &SKColumnData,
    policy: SKMissingValuePolicy,
) -> Result<Array1<F>, SKError> {
    let values: Vec<F> = match column {
        SKColumnData::Float32(values) => values
            .iter()
            .map(|value| match value {
                Some(value) => cast::<f32, F>(*value).ok_or_else(|| {
                    SKError::Conversion(format!(
                        "column `{column_name}` carries a float32 value not representable in the model scalar"
                    ))
                }),
                None => sk_scalar_for_null(column_name, policy),
            })
            .collect::<Result<_, _>>()?,
        SKColumnData::Float64(values) => values
            .iter()
            .map(|value| match value {
                Some(value) => cast::<f64, F>(*value).ok_or_else(|| {
                    SKError::Conversion(format!(
                        "column `{column_name}` carries a float64 value not representable in the model scalar"
                    ))
                }),
                None => sk_scalar_for_null(column_name, policy),
            })
            .collect::<Result<_, _>>()?,
        SKColumnData::Integer64(values) => {
            // Representability is a property of the value, not of the column's
            // nullability: a complete integer column must not truncate silently
            // either.
            for value in values.iter().flatten() {
                if !sk_integer_is_representable::<F>(*value) {
                    return Err(SKError::Conversion(format!(
                        "column `{column_name}` carries the integer {value}, which the model scalar cannot represent exactly; cast the column to a floating type in the adapter before promoting it"
                    )));
                }
            }
            values
                .iter()
                .map(|value| match value {
                    Some(value) => cast::<i64, F>(*value).ok_or_else(|| {
                        SKError::Conversion(format!(
                            "column `{column_name}` carries the integer {value}, which the model scalar cannot represent exactly"
                        ))
                    }),
                    None => sk_scalar_for_null(column_name, policy),
                })
                .collect::<Result<_, _>>()?
        }
        SKColumnData::Boolean(values) => values
            .iter()
            .map(|value| match value {
                Some(true) => Ok(F::one()),
                Some(false) => Ok(F::zero()),
                None => sk_scalar_for_null(column_name, policy),
            })
            .collect::<Result<_, _>>()?,
        SKColumnData::Nominal(_) => {
            return Err(SKError::UnsupportedColumnType {
                column: column_name.to_string(),
                logical_type: "nominal".to_string(),
                suggestion:
                    "encode nominal columns as category codes in the adapter before numeric materialization"
                        .to_string(),
            });
        }
        SKColumnData::CategoryCode(_) => {
            return Err(SKError::UnsupportedColumnType {
                column: column_name.to_string(),
                logical_type: "category_code".to_string(),
                suggestion:
                    "select category-coded columns through the categorical view instead of the numeric path"
                        .to_string(),
            });
        }
    };

    Ok(Array1::from_vec(values))
}
