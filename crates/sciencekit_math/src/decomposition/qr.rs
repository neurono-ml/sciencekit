//! The thin orthonormalization of the randomized range finder.
//!
//! The backend QR decomposition returns the *full* `m × m` orthogonal factor,
//! which is prohibitive for tall matrices. The range finder only needs the
//! thin factor (`m × min(m, n)`), so the kernel orthonormalizes the projected
//! basis with a Householder QR that produces exactly that factor.

use ndarray::{Array2, ArrayView2};
use sciencekit_common::SKFloat;

/// Computes the thin QR factor `Q` of `matrix`.
///
/// The returned matrix has `rows × min(rows, columns)` orthonormal columns and
/// spans the same column space as `matrix`. A zero (rank-deficient) column is
/// completed by the corresponding canonical direction, so the factor always
/// has `min(rows, columns)` orthonormal columns.
pub(crate) fn sk_thin_qr<F: SKFloat>(matrix: &ArrayView2<'_, F>) -> Array2<F> {
    let (rows, columns) = matrix.dim();
    let steps = rows.min(columns);
    let mut working = matrix.to_owned();
    let mut reflectors: Vec<(usize, Vec<F>)> = Vec::with_capacity(steps);

    for step in 0..steps {
        let mut norm = F::zero();
        for row in step..rows {
            norm = norm + working[(row, step)] * working[(row, step)];
        }
        let norm = norm.sqrt();
        if norm == F::zero() {
            continue;
        }
        let head = working[(step, step)];
        let diagonal = if head >= F::zero() { -norm } else { norm };
        let mut reflector: Vec<F> = Vec::with_capacity(rows - step);
        for row in step..rows {
            reflector.push(working[(row, step)]);
        }
        reflector[0] = reflector[0] - diagonal;
        let mut squared = F::zero();
        for value in &reflector {
            squared = squared + (*value) * (*value);
        }
        if squared <= F::zero() {
            continue;
        }
        let factor = (F::one() + F::one()) / squared;
        for column in (step + 1)..columns {
            let mut projection = F::zero();
            for (offset, &value) in reflector.iter().enumerate() {
                projection = projection + value * working[(step + offset, column)];
            }
            let weight = factor * projection;
            for (offset, &value) in reflector.iter().enumerate() {
                working[(step + offset, column)] =
                    working[(step + offset, column)] - weight * value;
            }
        }
        reflectors.push((step, reflector));
    }

    let mut basis = Array2::<F>::zeros((rows, steps));
    for index in 0..steps {
        basis[(index, index)] = F::one();
    }
    for (step, reflector) in reflectors.iter().rev() {
        let mut squared = F::zero();
        for value in reflector {
            squared = squared + (*value) * (*value);
        }
        if squared <= F::zero() {
            continue;
        }
        let factor = (F::one() + F::one()) / squared;
        for column in 0..steps {
            let mut projection = F::zero();
            for (offset, &value) in reflector.iter().enumerate() {
                projection = projection + value * basis[(step + offset, column)];
            }
            let weight = factor * projection;
            for (offset, &value) in reflector.iter().enumerate() {
                basis[(step + offset, column)] = basis[(step + offset, column)] - weight * value;
            }
        }
    }
    basis
}
