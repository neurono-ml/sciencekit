//! Element-wise unary and binary operators with NumPy broadcasting.

use ndarray::{ArrayD, IxDyn, Zip};

use super::SKOnnxImportError;
use super::value::SKOnnxValue;

/// Apply one binary floating operator, promoting mixed dtypes to double.
pub(crate) fn binary(
    left: &SKOnnxValue,
    right: &SKOnnxValue,
    function: impl Fn(f64, f64) -> f64,
) -> Result<SKOnnxValue, SKOnnxImportError> {
    match (left, right) {
        (SKOnnxValue::Float32(a), SKOnnxValue::Float32(b)) => {
            Ok(SKOnnxValue::Float32(zip_binary(a, b, |x, y| {
                function(f64::from(x), f64::from(y)) as f32
            })?))
        }
        _ => {
            let a = to_double(left);
            let b = to_double(right);
            Ok(SKOnnxValue::Float64(zip_binary(&a, &b, function)?))
        }
    }
}

/// Apply one binary comparison, producing booleans.
pub(crate) fn compare(
    left: &SKOnnxValue,
    right: &SKOnnxValue,
    function: impl Fn(f64, f64) -> bool,
) -> Result<SKOnnxValue, SKOnnxImportError> {
    let a = to_double(left);
    let b = to_double(right);
    Ok(SKOnnxValue::Boolean(zip_binary(&a, &b, function)?))
}

/// Apply one unary floating operator in the operand's own dtype.
pub(crate) fn unary(
    left: &SKOnnxValue,
    function: impl Fn(f64) -> f64 + Copy,
) -> Result<SKOnnxValue, SKOnnxImportError> {
    match left {
        SKOnnxValue::Float32(array) => Ok(SKOnnxValue::Float32(
            array.mapv(|value| function(f64::from(value)) as f32),
        )),
        _ => Ok(SKOnnxValue::Float64(to_double(left).mapv(function))),
    }
}

/// Clamp every value into `[minimum, maximum]`.
pub(crate) fn clip(
    value: &SKOnnxValue,
    minimum: Option<f64>,
    maximum: Option<f64>,
) -> Result<SKOnnxValue, SKOnnxImportError> {
    let function = |input: f64| {
        let lower = minimum.map_or(input, |bound| input.max(bound));
        maximum.map_or(lower, |bound| lower.min(bound))
    };
    unary(value, function)
}

/// Compare two values for equality, tolerating integer operands.
pub(crate) fn equal(
    left: &SKOnnxValue,
    right: &SKOnnxValue,
) -> Result<SKOnnxValue, SKOnnxImportError> {
    match (left, right) {
        (SKOnnxValue::Int64(a), SKOnnxValue::Int64(b)) => {
            Ok(SKOnnxValue::Boolean(zip_binary(a, b, |x, y| x == y)?))
        }
        _ => compare(left, right, |x, y| x == y),
    }
}

/// Broadcast two arrays onto their common shape and combine every pair.
fn zip_binary<A, B, C>(
    left: &ArrayD<A>,
    right: &ArrayD<B>,
    function: impl Fn(A, B) -> C,
) -> Result<ArrayD<C>, SKOnnxImportError>
where
    A: Copy,
    B: Copy,
    C: Copy + Default,
{
    let shape = broadcast_shape(left.shape(), right.shape())?;
    let left_view = left
        .broadcast(IxDyn(&shape))
        .ok_or_else(|| shape_error(&shape, left.shape()))?;
    let right_view = right
        .broadcast(IxDyn(&shape))
        .ok_or_else(|| shape_error(&shape, right.shape()))?;
    let dimension = IxDyn(&shape);
    let mut output = ArrayD::from_elem(dimension, C::default());
    Zip::from(&mut output)
        .and(&left_view)
        .and(&right_view)
        .for_each(|slot, a, b| *slot = function(*a, *b));
    Ok(output)
}

/// Compute the NumPy broadcast shape of two shapes.
pub(crate) fn broadcast_shape(
    left: &[usize],
    right: &[usize],
) -> Result<Vec<usize>, SKOnnxImportError> {
    let rank = left.len().max(right.len());
    let mut shape = vec![1; rank];
    for (position, slot) in shape.iter_mut().enumerate() {
        let a = left
            .len()
            .checked_sub(rank - position)
            .map_or(1, |index| left[index]);
        let b = right
            .len()
            .checked_sub(rank - position)
            .map_or(1, |index| right[index]);
        if a != b && a != 1 && b != 1 {
            return Err(SKOnnxImportError::ShapeMismatch {
                expected: right.to_vec(),
                found: left.to_vec(),
            });
        }
        *slot = a.max(b);
    }
    Ok(shape)
}

fn to_double(value: &SKOnnxValue) -> ArrayD<f64> {
    match value {
        SKOnnxValue::Float32(array) => array.mapv(f64::from),
        SKOnnxValue::Float64(array) => array.clone(),
        SKOnnxValue::Int64(array) => array.mapv(|value| value as f64),
        SKOnnxValue::Boolean(array) => array.mapv(|value| if value { 1.0 } else { 0.0 }),
    }
}

fn shape_error(expected: &[usize], found: &[usize]) -> SKOnnxImportError {
    SKOnnxImportError::ShapeMismatch {
        expected: expected.to_vec(),
        found: found.to_vec(),
    }
}
