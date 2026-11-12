//! The validated shape of a descriptor.

use super::error::SKFfiArrayError;

/// A validated, owned copy of the descriptor's shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SKFfiArrayShape {
    dimensions: Vec<usize>,
}

impl SKFfiArrayShape {
    /// Validate every dimension as a strictly positive `usize`.
    pub fn new(dimensions: &[i64]) -> Result<SKFfiArrayShape, SKFfiArrayError> {
        if dimensions.is_empty() || dimensions.len() > 2 {
            return Err(SKFfiArrayError::InvalidNumberOfDimensions(
                dimensions.len() as i32
            ));
        }
        let mut validated = Vec::with_capacity(dimensions.len());
        for &dimension in dimensions {
            if dimension < 0 {
                return Err(SKFfiArrayError::NegativeDimension(dimension));
            }
            let dimension = dimension as usize;
            if dimension == 0 {
                return Err(SKFfiArrayError::ZeroDimension(dimension));
            }
            validated.push(dimension);
        }
        Ok(SKFfiArrayShape {
            dimensions: validated,
        })
    }

    /// The dimensions in order.
    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    /// The rank (1 or 2).
    pub fn rank(&self) -> i32 {
        self.dimensions.len() as i32
    }

    /// The checked product of the dimensions.
    pub fn number_of_elements(&self) -> Result<usize, SKFfiArrayError> {
        self.dimensions
            .iter()
            .try_fold(1_usize, |product, dimension| {
                product.checked_mul(*dimension)
            })
            .filter(|count| *count <= isize::MAX as usize)
            .ok_or(SKFfiArrayError::LengthOutOfRange)
    }

    /// The `(rows, columns)` shape, when the rank is 2.
    pub fn matrix_shape(&self) -> Result<(usize, usize), SKFfiArrayError> {
        match self.dimensions.as_slice() {
            [rows, columns] => Ok((*rows, *columns)),
            _ => Err(SKFfiArrayError::WrongRank {
                expected: 2,
                found: self.rank(),
            }),
        }
    }

    /// The length, when the rank is 1.
    pub fn vector_length(&self) -> Result<usize, SKFfiArrayError> {
        match self.dimensions.as_slice() {
            [length] => Ok(*length),
            _ => Err(SKFfiArrayError::WrongRank {
                expected: 1,
                found: self.rank(),
            }),
        }
    }
}
