//! The bounds-checked little-endian cursor used by the decoder.

use sciencekit_common::{SKTensor, SKTensorDataType, SKTensorValues};

use crate::error_kind::SKFfiError;

/// Build a malformed-file failure.
pub(crate) fn malformed(message: impl Into<String>) -> SKFfiError {
    SKFfiError::invalid_argument(format!("malformed model state: {}", message.into()))
}

/// Map a stored dtype byte onto its tensor dtype.
pub(crate) fn dtype_from_code(code: u8) -> Result<SKTensorDataType, SKFfiError> {
    match code {
        1 => Ok(SKTensorDataType::F64),
        2 => Ok(SKTensorDataType::I64),
        3 => Ok(SKTensorDataType::Bool),
        other => Err(malformed(format!("unsupported tensor dtype code {other}"))),
    }
}

/// A bounds-checked little-endian cursor.
pub(crate) struct SKModelStateReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> SKModelStateReader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> SKModelStateReader<'a> {
        SKModelStateReader { bytes, position: 0 }
    }

    pub(crate) fn take(&mut self, length: usize) -> Result<&'a [u8], SKFfiError> {
        let end = self
            .position
            .checked_add(length)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| malformed("unexpected end of file"))?;
        let slice = &self.bytes[self.position..end];
        self.position = end;
        Ok(slice)
    }

    pub(crate) fn u8(&mut self) -> Result<u8, SKFfiError> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn u16(&mut self) -> Result<u16, SKFfiError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    pub(crate) fn u32(&mut self) -> Result<u32, SKFfiError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, SKFfiError> {
        let bytes = self.take(8)?;
        let mut array = [0_u8; 8];
        array.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(array))
    }

    pub(crate) fn i64(&mut self) -> Result<i64, SKFfiError> {
        let bytes = self.take(8)?;
        let mut array = [0_u8; 8];
        array.copy_from_slice(bytes);
        Ok(i64::from_le_bytes(array))
    }

    pub(crate) fn f64(&mut self) -> Result<f64, SKFfiError> {
        let bytes = self.take(8)?;
        let mut array = [0_u8; 8];
        array.copy_from_slice(bytes);
        Ok(f64::from_le_bytes(array))
    }

    pub(crate) fn string(&mut self) -> Result<String, SKFfiError> {
        let length = self.u32()? as usize;
        let bytes = self.take(length)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| malformed("a string is not valid UTF-8"))
    }

    pub(crate) fn parameter(&mut self) -> Result<sciencekit_common::SKParameterValue, SKFfiError> {
        use sciencekit_common::SKParameterValue;

        match self.u8()? {
            0 => Ok(SKParameterValue::Bool(self.u8()? != 0)),
            1 => Ok(SKParameterValue::Integer(self.i64()?)),
            2 => Ok(SKParameterValue::Float(self.f64()?)),
            3 => Ok(SKParameterValue::String(self.string()?)),
            4 => {
                let count = self.u32()? as usize;
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    values.push(self.parameter()?);
                }
                Ok(SKParameterValue::List(values))
            }
            other => Err(malformed(format!("unknown parameter tag {other}"))),
        }
    }

    pub(crate) fn tensor(&mut self) -> Result<SKTensor<f64>, SKFfiError> {
        let name = self.string()?;
        let dtype = dtype_from_code(self.u8()?)?;
        let rank = self.u32()? as usize;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(self.u64()? as usize);
        }
        let count = self.u64()? as usize;
        if count != shape.iter().product::<usize>() {
            return Err(malformed("a tensor length disagrees with its shape"));
        }
        let values = match dtype {
            SKTensorDataType::F64 => {
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    values.push(self.f64()?);
                }
                SKTensorValues::Floating(values)
            }
            SKTensorDataType::I64 => {
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    values.push(self.i64()?);
                }
                SKTensorValues::Integer(values)
            }
            SKTensorDataType::Bool => {
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    values.push(self.u8()? != 0);
                }
                SKTensorValues::Boolean(values)
            }
            _ => return Err(malformed("unsupported tensor dtype")),
        };
        SKTensor::new(name, dtype, shape, values).map_err(SKFfiError::from)
    }
}
