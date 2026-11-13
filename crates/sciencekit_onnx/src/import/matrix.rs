//! Matrix multiplication and transposition.

use ndarray::{Array2, IxDyn};

use super::SKOnnxImportError;
use super::value::SKOnnxValue;

/// Two-dimensional `MatMul`, the only rank the documented subset needs.
pub(crate) fn mat_mul(
    left: &SKOnnxValue,
    right: &SKOnnxValue,
) -> Result<SKOnnxValue, SKOnnxImportError> {
    if left.shape().len() != 2 || right.shape().len() != 2 {
        return Err(SKOnnxImportError::InvalidOperator {
            operator: "MatMul".to_string(),
            detail: format!(
                "only rank-2 operands are supported, got {:?} and {:?}",
                left.shape(),
                right.shape()
            ),
        });
    }
    match (left, right) {
        (SKOnnxValue::Float32(a), SKOnnxValue::Float32(b)) => Ok(SKOnnxValue::Float32(
            as_matrix(a)?.dot(&as_matrix(b)?).into_dyn(),
        )),
        _ => {
            let a = as_double_matrix(left)?;
            let b = as_double_matrix(right)?;
            Ok(SKOnnxValue::Float64(a.dot(&b).into_dyn()))
        }
    }
}

/// Permute axes; `permutation` defaults to the reverse order.
pub(crate) fn transpose(
    value: &SKOnnxValue,
    permutation: Option<&[i64]>,
) -> Result<SKOnnxValue, SKOnnxImportError> {
    let rank = value.shape().len();
    let permutation = match permutation {
        Some(axes) => super::reductions::resolve_axes(axes, rank)?,
        None => (0..rank).rev().collect(),
    };
    Ok(match value {
        SKOnnxValue::Float32(array) => {
            SKOnnxValue::Float32(array.clone().permuted_axes(permutation))
        }
        SKOnnxValue::Float64(array) => {
            SKOnnxValue::Float64(array.clone().permuted_axes(permutation))
        }
        SKOnnxValue::Int64(array) => SKOnnxValue::Int64(array.clone().permuted_axes(permutation)),
        SKOnnxValue::Boolean(array) => {
            SKOnnxValue::Boolean(array.clone().permuted_axes(permutation))
        }
    })
}

/// Rebuild a value in a new shape, preserving the row-major element order.
pub(crate) fn reshape(
    value: &SKOnnxValue,
    shape: &[i64],
) -> Result<SKOnnxValue, SKOnnxImportError> {
    let resolved = resolve_reshape(value.length(), shape)?;
    let dimension = IxDyn(&resolved);
    match value {
        SKOnnxValue::Float32(array) => Ok(SKOnnxValue::Float32(rebuild(
            dimension,
            array.iter().copied().collect(),
            &resolved,
            value.length(),
        )?)),
        SKOnnxValue::Float64(array) => Ok(SKOnnxValue::Float64(rebuild(
            dimension,
            array.iter().copied().collect(),
            &resolved,
            value.length(),
        )?)),
        SKOnnxValue::Int64(array) => Ok(SKOnnxValue::Int64(rebuild(
            dimension,
            array.iter().copied().collect(),
            &resolved,
            value.length(),
        )?)),
        SKOnnxValue::Boolean(array) => Ok(SKOnnxValue::Boolean(rebuild(
            dimension,
            array.iter().copied().collect(),
            &resolved,
            value.length(),
        )?)),
    }
}

fn rebuild<T>(
    dimension: IxDyn,
    values: Vec<T>,
    shape: &[usize],
    length: usize,
) -> Result<ndarray::ArrayD<T>, SKOnnxImportError> {
    ndarray::ArrayD::from_shape_vec(dimension, values).map_err(|_| {
        SKOnnxImportError::ShapeMismatch {
            expected: shape.to_vec(),
            found: vec![length],
        }
    })
}

fn resolve_reshape(length: usize, shape: &[i64]) -> Result<Vec<usize>, SKOnnxImportError> {
    let mut resolved = Vec::with_capacity(shape.len());
    let mut inferred = None;
    let mut known = 1usize;
    for (position, dimension) in shape.iter().enumerate() {
        if *dimension == -1 {
            if inferred.is_some() {
                return Err(SKOnnxImportError::InvalidOperator {
                    operator: "Reshape".to_string(),
                    detail: "the target shape declares two inferred dimensions".to_string(),
                });
            }
            inferred = Some(position);
            resolved.push(0);
        } else if *dimension == 0 {
            resolved.push(1);
        } else {
            let value = *dimension as usize;
            known *= value;
            resolved.push(value);
        }
    }
    if let Some(position) = inferred {
        if known == 0 || length % known != 0 {
            return Err(SKOnnxImportError::ShapeMismatch {
                expected: resolved,
                found: vec![length],
            });
        }
        resolved[position] = length / known;
    }
    Ok(resolved)
}

fn as_matrix(array: &ndarray::ArrayD<f32>) -> Result<Array2<f32>, SKOnnxImportError> {
    array
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|_| SKOnnxImportError::ShapeMismatch {
            expected: array.shape().to_vec(),
            found: array.shape().to_vec(),
        })
}

fn as_double_matrix(value: &SKOnnxValue) -> Result<Array2<f64>, SKOnnxImportError> {
    let array = match value {
        SKOnnxValue::Float32(a) => a.mapv(f64::from),
        SKOnnxValue::Float64(a) => a.clone(),
        SKOnnxValue::Int64(a) => a.mapv(|value| value as f64),
        SKOnnxValue::Boolean(a) => a.mapv(|value| if value { 1.0 } else { 0.0 }),
    };
    let shape = array.shape().to_vec();
    array
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|_| SKOnnxImportError::ShapeMismatch {
            expected: shape.clone(),
            found: shape,
        })
}
