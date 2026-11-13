//! Selection operators: casting and top-k.

use ndarray::{ArrayD, Axis, IxDyn};

use super::SKOnnxImportError;
use super::reductions::resolve_axis;
use super::value::SKOnnxValue;

/// Cast to another dense element type.
pub(crate) fn cast(
    value: &SKOnnxValue,
    element_type: serde_onnx::ir::ElemType,
) -> Result<SKOnnxValue, SKOnnxImportError> {
    value.cast_to(element_type)
}

/// The values and indices of the `count` largest entries along `axis`.
pub(crate) fn top_k(
    value: &SKOnnxValue,
    count: usize,
    axis: i64,
) -> Result<(SKOnnxValue, SKOnnxValue), SKOnnxImportError> {
    let array = as_double(value);
    let axis = resolve_axis(axis, array.ndim())?;
    let extent = array.shape()[axis.index()];
    if count > extent {
        return Err(SKOnnxImportError::InvalidOperator {
            operator: "TopK".to_string(),
            detail: format!("requested {count} entries from an extent of {extent}"),
        });
    }
    let mut picked: Vec<Vec<usize>> = Vec::new();
    for lane in array.lanes(axis) {
        let mut indices: Vec<usize> = (0..lane.len()).collect();
        indices.sort_by(|left, right| {
            lane[*right]
                .partial_cmp(&lane[*left])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(left.cmp(right))
        });
        indices.truncate(count);
        picked.push(indices);
    }
    let mut shape = array.shape().to_vec();
    shape[axis.index()] = count;
    let mut values = Vec::new();
    let mut indices = Vec::new();
    for (lane, picked_lane) in array.lanes(axis).into_iter().zip(picked.iter()) {
        for index in picked_lane {
            values.push(lane[*index]);
            indices.push(*index as i64);
        }
    }
    Ok((
        SKOnnxValue::from_f64_values(value.element_type(), shape.clone(), values)?,
        SKOnnxValue::Int64(
            ArrayD::from_shape_vec(IxDyn(&shape), indices).map_err(|error| {
                SKOnnxImportError::InvalidOperator {
                    operator: "TopK".to_string(),
                    detail: error.to_string(),
                }
            })?,
        ),
    ))
}

/// Gather rows (or slices) by integer indices along `axis`.
pub(crate) fn gather(
    data: &SKOnnxValue,
    indices: &SKOnnxValue,
    axis: i64,
) -> Result<SKOnnxValue, SKOnnxImportError> {
    let rank = data.shape().len();
    let axis = resolve_axis(axis, rank)?;
    let extent = data.shape()[axis.index()] as i64;
    let resolved: Vec<usize> = indices
        .to_i64_values()
        .into_iter()
        .map(|index| {
            let index = if index < 0 { index + extent } else { index };
            if index < 0 || index >= extent {
                return Err(SKOnnxImportError::InvalidOperator {
                    operator: "Gather".to_string(),
                    detail: format!("index {index} is outside extent {extent}"),
                });
            }
            Ok(index as usize)
        })
        .collect::<Result<_, _>>()?;
    let index_shape = indices.shape();
    match data {
        SKOnnxValue::Float32(array) => Ok(SKOnnxValue::Float32(gather_array(
            array,
            &resolved,
            axis,
            &index_shape,
        )?)),
        SKOnnxValue::Float64(array) => Ok(SKOnnxValue::Float64(gather_array(
            array,
            &resolved,
            axis,
            &index_shape,
        )?)),
        SKOnnxValue::Int64(array) => Ok(SKOnnxValue::Int64(gather_array(
            array,
            &resolved,
            axis,
            &index_shape,
        )?)),
        SKOnnxValue::Boolean(array) => Ok(SKOnnxValue::Boolean(gather_array(
            array,
            &resolved,
            axis,
            &index_shape,
        )?)),
    }
}

fn gather_array<T: Copy>(
    array: &ArrayD<T>,
    indices: &[usize],
    axis: Axis,
    index_shape: &[usize],
) -> Result<ArrayD<T>, SKOnnxImportError> {
    let mut shape = array.shape().to_vec();
    shape.splice(axis.index()..=axis.index(), index_shape.iter().copied());
    let mut values = Vec::new();
    for index in indices {
        values.extend(array.index_axis(axis, *index).iter().copied());
    }
    ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|error| {
        SKOnnxImportError::InvalidOperator {
            operator: "Gather".to_string(),
            detail: error.to_string(),
        }
    })
}

fn as_double(value: &SKOnnxValue) -> ArrayD<f64> {
    match value {
        SKOnnxValue::Float32(array) => array.mapv(f64::from),
        SKOnnxValue::Float64(array) => array.clone(),
        SKOnnxValue::Int64(array) => array.mapv(|value| value as f64),
        SKOnnxValue::Boolean(array) => array.mapv(|value| if value { 1.0 } else { 0.0 }),
    }
}
