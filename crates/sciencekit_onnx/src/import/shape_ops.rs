//! Shape manipulation, indexing, selection and casting.

use ndarray::{ArrayD, Axis};

use super::SKOnnxImportError;
use super::reductions::{resolve_axes, resolve_axis};
use super::value::SKOnnxValue;

/// Concatenate tensors along `axis`.
pub(crate) fn concat(inputs: &[SKOnnxValue], axis: i64) -> Result<SKOnnxValue, SKOnnxImportError> {
    let first = inputs
        .first()
        .ok_or_else(|| SKOnnxImportError::InvalidOperator {
            operator: "Concat".to_string(),
            detail: "at least one input is required".to_string(),
        })?;
    let axis = resolve_axis(axis, first.shape().len())?;
    match first {
        SKOnnxValue::Float32(_) => {
            let arrays: Vec<ArrayD<f32>> = inputs
                .iter()
                .map(|input| match input {
                    SKOnnxValue::Float32(array) => Ok(array.clone()),
                    other => Err(dtype_error(other)),
                })
                .collect::<Result<_, _>>()?;
            Ok(SKOnnxValue::Float32(concatenate(axis, &arrays)?))
        }
        SKOnnxValue::Float64(_) => {
            let arrays: Vec<ArrayD<f64>> = inputs
                .iter()
                .map(|input| match input {
                    SKOnnxValue::Float64(array) => Ok(array.clone()),
                    other => Err(dtype_error(other)),
                })
                .collect::<Result<_, _>>()?;
            Ok(SKOnnxValue::Float64(concatenate(axis, &arrays)?))
        }
        SKOnnxValue::Int64(_) => {
            let arrays: Vec<ArrayD<i64>> = inputs
                .iter()
                .map(|input| match input {
                    SKOnnxValue::Int64(array) => Ok(array.clone()),
                    other => Err(dtype_error(other)),
                })
                .collect::<Result<_, _>>()?;
            Ok(SKOnnxValue::Int64(concatenate(axis, &arrays)?))
        }
        SKOnnxValue::Boolean(_) => {
            let arrays: Vec<ArrayD<bool>> = inputs
                .iter()
                .map(|input| match input {
                    SKOnnxValue::Boolean(array) => Ok(array.clone()),
                    other => Err(dtype_error(other)),
                })
                .collect::<Result<_, _>>()?;
            Ok(SKOnnxValue::Boolean(concatenate(axis, &arrays)?))
        }
    }
}

fn concatenate<T: Clone>(axis: Axis, arrays: &[ArrayD<T>]) -> Result<ArrayD<T>, SKOnnxImportError> {
    let views: Vec<_> = arrays.iter().map(|array| array.view()).collect();
    ndarray::concatenate(axis, &views).map_err(|error| SKOnnxImportError::InvalidOperator {
        operator: "Concat".to_string(),
        detail: error.to_string(),
    })
}

/// Insert size-one axes at the given positions.
pub(crate) fn unsqueeze(
    value: &SKOnnxValue,
    axes: &[i64],
) -> Result<SKOnnxValue, SKOnnxImportError> {
    let rank = value.shape().len() + axes.len();
    let mut positions = resolve_axes(axes, rank)?;
    if !has_no_duplicates(&positions) {
        return Err(SKOnnxImportError::InvalidOperator {
            operator: "Unsqueeze".to_string(),
            detail: format!("the axes {axes:?} repeat a position"),
        });
    }
    positions.sort_unstable();
    let mut result = value.clone();
    for position in positions {
        result = insert_axis(&result, position);
    }
    Ok(result)
}

/// Remove size-one axes, or the listed axes when provided.
pub(crate) fn squeeze(
    value: &SKOnnxValue,
    axes: Option<&[i64]>,
) -> Result<SKOnnxValue, SKOnnxImportError> {
    let rank = value.shape().len();
    let positions: Vec<usize> = match axes {
        Some(axes) => {
            let resolved = resolve_axes(axes, rank)?;
            if !has_no_duplicates(&resolved) {
                return Err(SKOnnxImportError::InvalidOperator {
                    operator: "Squeeze".to_string(),
                    detail: format!("the axes {axes:?} repeat a position"),
                });
            }
            if let Some(position) = resolved
                .iter()
                .find(|position| value.shape()[**position] != 1)
            {
                return Err(SKOnnxImportError::InvalidOperator {
                    operator: "Squeeze".to_string(),
                    detail: format!(
                        "axis {position} has extent {}; only size-one axes can be squeezed",
                        value.shape()[*position]
                    ),
                });
            }
            resolved
        }
        None => value
            .shape()
            .iter()
            .enumerate()
            .filter(|(_, dimension)| **dimension == 1)
            .map(|(position, _)| position)
            .collect(),
    };
    let mut result = value.clone();
    for position in positions.into_iter().rev() {
        result = remove_axis(&result, position);
    }
    Ok(result)
}

/// Flatten to `[rows, columns]`, preserving ONNX's `axis` semantics.
pub(crate) fn flatten(value: &SKOnnxValue, axis: i64) -> Result<SKOnnxValue, SKOnnxImportError> {
    let shape = value.shape();
    let axis = if axis < 0 {
        axis + shape.len() as i64
    } else {
        axis
    };
    let axis = axis.clamp(0, shape.len() as i64) as usize;
    let rows: usize = shape[..axis].iter().product();
    let columns: usize = shape[axis..].iter().product();
    super::matrix::reshape(value, &[rows as i64, columns as i64])
}

fn has_no_duplicates(positions: &[usize]) -> bool {
    let mut sorted = positions.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    sorted.len() == positions.len()
}

fn insert_axis(value: &SKOnnxValue, position: usize) -> SKOnnxValue {
    match value {
        SKOnnxValue::Float32(array) => {
            SKOnnxValue::Float32(array.clone().insert_axis(Axis(position)))
        }
        SKOnnxValue::Float64(array) => {
            SKOnnxValue::Float64(array.clone().insert_axis(Axis(position)))
        }
        SKOnnxValue::Int64(array) => SKOnnxValue::Int64(array.clone().insert_axis(Axis(position))),
        SKOnnxValue::Boolean(array) => {
            SKOnnxValue::Boolean(array.clone().insert_axis(Axis(position)))
        }
    }
}

fn remove_axis(value: &SKOnnxValue, position: usize) -> SKOnnxValue {
    match value {
        SKOnnxValue::Float32(array) => {
            SKOnnxValue::Float32(array.clone().index_axis_move(Axis(position), 0))
        }
        SKOnnxValue::Float64(array) => {
            SKOnnxValue::Float64(array.clone().index_axis_move(Axis(position), 0))
        }
        SKOnnxValue::Int64(array) => {
            SKOnnxValue::Int64(array.clone().index_axis_move(Axis(position), 0))
        }
        SKOnnxValue::Boolean(array) => {
            SKOnnxValue::Boolean(array.clone().index_axis_move(Axis(position), 0))
        }
    }
}

fn dtype_error(value: &SKOnnxValue) -> SKOnnxImportError {
    SKOnnxImportError::UnsupportedDtype {
        element_type: value.element_type().name().to_string(),
    }
}
