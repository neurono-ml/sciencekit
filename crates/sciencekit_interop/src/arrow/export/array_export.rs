//! Export of numeric vectors as Arrow arrays, with the conversion report
//! (spec `arrow-columnar-interop`).

use std::any::TypeId;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float32Array, Float64Array};
use arrow_schema::DataType;
use ndarray::ArrayView1;
use sciencekit_common::{SKError, SKFloat};

/// An exported Arrow array plus how faithfully it mirrors the source buffer.
#[derive(Debug, Clone)]
pub struct SKArrowExportedArray {
    array: ArrayRef,
    zero_copy: bool,
}

impl SKArrowExportedArray {
    /// The exported Arrow array.
    pub fn array(&self) -> &ArrayRef {
        &self.array
    }

    /// Whether the exported array aliases the source buffer.
    ///
    /// Export from an ndarray view always allocates an Arrow buffer, because
    /// the two libraries own separate allocations; the flag exists so callers
    /// can branch on the same report shape as the import path.
    pub fn is_zero_copy(&self) -> bool {
        self.zero_copy
    }
}

pub(crate) fn sk_float_data_type<F: SKFloat>() -> Result<DataType, SKError> {
    if TypeId::of::<F>() == TypeId::of::<f32>() {
        Ok(DataType::Float32)
    } else if TypeId::of::<F>() == TypeId::of::<f64>() {
        Ok(DataType::Float64)
    } else {
        Err(SKError::Conversion(
            "the model scalar is neither f32 nor f64, so it has no Arrow counterpart".to_string(),
        ))
    }
}

pub(crate) fn sk_primitive_array_from_slice<F: SKFloat>(values: &[F]) -> Result<ArrayRef, SKError> {
    if TypeId::of::<F>() == TypeId::of::<f32>() {
        let converted: Vec<f32> = values.iter().map(|value| value.to_f32().unwrap()).collect();
        Ok(Arc::new(Float32Array::from(converted)))
    } else if TypeId::of::<F>() == TypeId::of::<f64>() {
        let converted: Vec<f64> = values.iter().map(|value| value.to_f64().unwrap()).collect();
        Ok(Arc::new(Float64Array::from(converted)))
    } else {
        Err(SKError::Conversion(
            "the model scalar is neither f32 nor f64, so it has no Arrow counterpart".to_string(),
        ))
    }
}

/// Export a prediction or transformed vector as a floating-point Arrow array.
///
/// The result is a one-dimensional `f32`/`f64` array with the same values and
/// length as the input.
///
/// # Errors
/// Returns [`SKError::Conversion`] for a model scalar with no Arrow
/// counterpart.
pub fn sk_export_prediction<F: SKFloat>(
    values: ArrayView1<F>,
) -> Result<SKArrowExportedArray, SKError> {
    let values: Vec<F> = values.iter().copied().collect();
    let array = sk_primitive_array_from_slice(&values)?;
    Ok(SKArrowExportedArray {
        array,
        zero_copy: false,
    })
}
