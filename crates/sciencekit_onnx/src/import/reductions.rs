//! Reductions, arg-reductions and softmax along a declared axis.

use std::ops::Add;

use ndarray::{ArrayD, Axis, IxDyn};

use super::SKOnnxImportError;
use super::value::SKOnnxValue;

/// Sum over the given axes, keeping or dropping them.
pub(crate) fn reduce_sum(
    value: &SKOnnxValue,
    axes: &[i64],
    keep_dimensions: bool,
) -> Result<SKOnnxValue, SKOnnxImportError> {
    match value {
        SKOnnxValue::Float32(array) => Ok(SKOnnxValue::Float32(fold_axes(
            array,
            axes,
            keep_dimensions,
            |current, axis| current.sum_axis(axis),
        )?)),
        SKOnnxValue::Int64(array) => Ok(SKOnnxValue::Int64(fold_axes(
            array,
            axes,
            keep_dimensions,
            |current, axis| current.sum_axis(axis),
        )?)),
        _ => Ok(SKOnnxValue::Float64(fold_axes(
            &as_double(value),
            axes,
            keep_dimensions,
            |current, axis| current.sum_axis(axis),
        )?)),
    }
}

/// Mean over the given axes, keeping or dropping them.
pub(crate) fn reduce_mean(
    value: &SKOnnxValue,
    axes: &[i64],
    keep_dimensions: bool,
) -> Result<SKOnnxValue, SKOnnxImportError> {
    match value {
        SKOnnxValue::Float32(array) => Ok(SKOnnxValue::Float32(fold_axes(
            array,
            axes,
            keep_dimensions,
            |current, axis| {
                current.mean_axis(axis).unwrap_or_else(|| {
                    ArrayD::zeros(IxDyn(&reduced_shape(current.shape(), axis, false)))
                })
            },
        )?)),
        _ => Ok(SKOnnxValue::Float64(fold_axes(
            &as_double(value),
            axes,
            keep_dimensions,
            |current, axis| {
                current.mean_axis(axis).unwrap_or_else(|| {
                    ArrayD::zeros(IxDyn(&reduced_shape(current.shape(), axis, false)))
                })
            },
        )?)),
    }
}

/// Arg-max along `axis`, producing `i64` indices.
pub(crate) fn arg_max(
    value: &SKOnnxValue,
    axis: i64,
    keep_dimensions: bool,
) -> Result<SKOnnxValue, SKOnnxImportError> {
    arg_reduce(value, axis, keep_dimensions, true, "ArgMax")
}

/// Arg-min along `axis`, producing `i64` indices.
pub(crate) fn arg_min(
    value: &SKOnnxValue,
    axis: i64,
    keep_dimensions: bool,
) -> Result<SKOnnxValue, SKOnnxImportError> {
    arg_reduce(value, axis, keep_dimensions, false, "ArgMin")
}

/// Softmax along `axis`.
pub(crate) fn softmax(value: &SKOnnxValue, axis: i64) -> Result<SKOnnxValue, SKOnnxImportError> {
    let array = as_double(value);
    let axis = resolve_axis(axis, array.ndim())?;
    let maximum = array.fold_axis(axis, f64::NEG_INFINITY, |a, b| a.max(*b));
    let shifted = &array - &maximum.insert_axis(axis);
    let exponentiated = shifted.mapv(f64::exp);
    let total = exponentiated.fold_axis(axis, 0.0, |a, b| a + b);
    let result = &exponentiated / &total.insert_axis(axis);
    SKOnnxValue::from_f64_values(
        value.element_type(),
        value.shape(),
        result.iter().copied().collect(),
    )
}

fn arg_reduce(
    value: &SKOnnxValue,
    axis: i64,
    keep_dimensions: bool,
    maximum: bool,
    operator: &str,
) -> Result<SKOnnxValue, SKOnnxImportError> {
    let array = as_double(value);
    let axis = resolve_axis(axis, array.ndim())?;
    let mut indices = Vec::new();
    for lane in array.lanes(axis) {
        let mut best = 0usize;
        for (position, candidate) in lane.iter().enumerate() {
            let improved = if maximum {
                *candidate > lane[best]
            } else {
                *candidate < lane[best]
            };
            if improved {
                best = position;
            }
        }
        indices.push(best as i64);
    }
    let shape = reduced_shape(array.shape(), axis, keep_dimensions);
    let output = ArrayD::from_shape_vec(IxDyn(&shape), indices).map_err(|error| {
        SKOnnxImportError::InvalidOperator {
            operator: operator.to_string(),
            detail: error.to_string(),
        }
    })?;
    Ok(SKOnnxValue::Int64(output))
}

fn fold_axes<A>(
    array: &ArrayD<A>,
    axes: &[i64],
    keep_dimensions: bool,
    function: impl Fn(&ArrayD<A>, Axis) -> ArrayD<A>,
) -> Result<ArrayD<A>, SKOnnxImportError>
where
    A: Clone + Add<Output = A> + num_traits::Zero,
{
    let mut current = array.clone();
    let mut resolved = resolve_axes(axes, array.ndim())?;
    resolved.sort_unstable();
    resolved.dedup();
    for axis in resolved.into_iter().rev() {
        current = function(&current, Axis(axis));
        if keep_dimensions {
            current = current.insert_axis(Axis(axis));
        }
    }
    Ok(current)
}

fn reduced_shape(shape: &[usize], axis: Axis, keep_dimensions: bool) -> Vec<usize> {
    let mut result = shape.to_vec();
    if keep_dimensions {
        result[axis.index()] = 1;
    } else {
        result.remove(axis.index());
    }
    result
}

/// Resolve a possibly negative axis against a rank.
pub(crate) fn resolve_axis(axis: i64, rank: usize) -> Result<Axis, SKOnnxImportError> {
    let resolved = if axis < 0 { axis + rank as i64 } else { axis };
    if resolved < 0 || resolved >= rank as i64 {
        return Err(SKOnnxImportError::InvalidOperator {
            operator: "axis".to_string(),
            detail: format!("axis {axis} is outside rank {rank}"),
        });
    }
    Ok(Axis(resolved as usize))
}

/// Resolve a list of possibly negative axes against a rank.
pub(crate) fn resolve_axes(axes: &[i64], rank: usize) -> Result<Vec<usize>, SKOnnxImportError> {
    axes.iter()
        .map(|axis| resolve_axis(*axis, rank).map(|axis| axis.index()))
        .collect()
}

fn as_double(value: &SKOnnxValue) -> ArrayD<f64> {
    match value {
        SKOnnxValue::Float32(array) => array.mapv(f64::from),
        SKOnnxValue::Float64(array) => array.clone(),
        SKOnnxValue::Int64(array) => array.mapv(|value| value as f64),
        SKOnnxValue::Boolean(array) => array.mapv(|value| if value { 1.0 } else { 0.0 }),
    }
}
