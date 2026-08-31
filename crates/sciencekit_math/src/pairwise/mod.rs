//! Zero-copy pairwise-distance kernels (spec `pairwise-distances`).
//!
//! Distances are computed between **rows** of two feature matrices using the
//! SIMD-friendly norm-expansion for squared Euclidean:
//! `‖a−b‖² = ‖a‖² − 2 a·b + ‖b‖²`. This avoids a per-pair subtract loop and
//! lets the inner product run over contiguous row memory.
//!
//! All kernels take `&ArrayView2<F>` inputs and return owned `Array2<F>`
//! outputs; they are pure and thread-safe.

use ndarray::{Array, Array2, ArrayView2};
use sciencekit_common::SKFloat;
use wide::f64x4;

/// Squared Euclidean distance matrix between the rows of two matrices.
///
/// Computes `D[i, j] = ‖a_i‖² − 2 a_i·b_j + ‖b_j‖²` using the norm-expansion
/// formulation, which is symmetric and numerically stable for the diagonal.
pub fn sk_squared_euclidean_distance_matrix<F: SKFloat>(
    left: &ArrayView2<F>,
    right: &ArrayView2<F>,
) -> Array2<F> {
    assert_eq!(
        left.ncols(),
        right.ncols(),
        "pairwise distances require matching feature dimensions"
    );
    let (left_rows, right_rows) = (left.nrows(), right.nrows());
    let left_norms: Vec<F> = left.rows().into_iter().map(row_norm_squared).collect();
    let right_norms: Vec<F> = right.rows().into_iter().map(row_norm_squared).collect();

    let mut output = Array::zeros((left_rows, right_rows));
    for i in 0..left_rows {
        for j in 0..right_rows {
            let dot = dot_product_simd(&left.row(i), &right.row(j));
            output[[i, j]] = left_norms[i] + right_norms[j] - dot - dot;
        }
    }
    output
}

/// Euclidean distance matrix: the element-wise square root of the squared form.
pub fn sk_euclidean_distance_matrix<F: SKFloat>(
    left: &ArrayView2<F>,
    right: &ArrayView2<F>,
) -> Array2<F> {
    sk_squared_euclidean_distance_matrix(left, right).mapv(|value| value.sqrt())
}

/// Manhattan (L1) distance matrix: `Σ |a_k − b_k|`.
pub fn sk_manhattan_distance_matrix<F: SKFloat>(
    left: &ArrayView2<F>,
    right: &ArrayView2<F>,
) -> Array2<F> {
    assert_eq!(
        left.ncols(),
        right.ncols(),
        "pairwise distances require matching feature dimensions"
    );
    let mut output = Array::zeros((left.nrows(), right.nrows()));
    for (i, left_row) in left.rows().into_iter().enumerate() {
        for (j, right_row) in right.rows().into_iter().enumerate() {
            let mut sum = F::zero();
            for k in 0..left.ncols() {
                sum = sum + (left_row[k] - right_row[k]).abs();
            }
            output[[i, j]] = sum;
        }
    }
    output
}

/// Cosine distance matrix: `1 − cos θ` over row vectors.
///
/// Identical rows give `0`, orthogonal unit rows give `1`. Zero-norm rows yield
/// a cosine of `0` (distance `1`), matching scikit-learn's behaviour.
pub fn sk_cosine_distance_matrix<F: SKFloat>(
    left: &ArrayView2<F>,
    right: &ArrayView2<F>,
) -> Array2<F> {
    assert_eq!(
        left.ncols(),
        right.ncols(),
        "pairwise distances require matching feature dimensions"
    );
    let mut output = Array::zeros((left.nrows(), right.nrows()));
    for (i, left_row) in left.rows().into_iter().enumerate() {
        let left_norm = row_norm(left_row);
        for (j, right_row) in right.rows().into_iter().enumerate() {
            let right_norm = row_norm(right_row);
            let dot = dot_product_simd(&left_row, &right_row);
            let denominator = left_norm * right_norm;
            let cosine = if denominator == F::zero() {
                F::zero()
            } else {
                dot / denominator
            };
            output[[i, j]] = F::one() - cosine;
        }
    }
    output
}

/// Sum of the squares of a row (`‖x‖²`).
fn row_norm_squared<F: SKFloat>(row: ndarray::ArrayView1<F>) -> F {
    dot_product_simd(&row, &row)
}

/// Euclidean norm of a row.
fn row_norm<F: SKFloat>(row: ndarray::ArrayView1<F>) -> F {
    row_norm_squared(row).sqrt()
}

/// SIMD-friendly dot product over contiguous row data.
///
/// The `f64` hot path processes four lanes at a time with [`f64x4`] fused
/// multiply-add; shorter tails and non-`f64` types fall back to a plain
/// accumulation. Contiguity is handled by ndarray's row iterators.
fn dot_product_simd<F: SKFloat>(
    left: &ndarray::ArrayView1<F>,
    right: &ndarray::ArrayView1<F>,
) -> F {
    if core::any::TypeId::of::<F>() == core::any::TypeId::of::<f64>() {
        // Reinterpret as f64 for the wide path; both slices have identical layout.
        let left = cast_slice_f64(left.as_slice().expect("contiguous 1-D view"));
        let right = cast_slice_f64(right.as_slice().expect("contiguous 1-D view"));
        let mut accumulator = f64x4::splat(0.0);
        let mut index = 0;
        let length = left.len();
        while index + 4 <= length {
            let a = f64x4::from([
                left[index],
                left[index + 1],
                left[index + 2],
                left[index + 3],
            ]);
            let b = f64x4::from([
                right[index],
                right[index + 1],
                right[index + 2],
                right[index + 3],
            ]);
            // `a.mul_add(b, accumulator)` = `a*b + accumulator` (fused multiply-add).
            accumulator = a.mul_add(b, accumulator);
            index += 4;
        }
        let sum = accumulator.reduce_add();
        let mut tail = 0.0;
        while index < length {
            tail += left[index] * right[index];
            index += 1;
        }
        let total = sum + tail;
        // SAFETY: the branch guards `F == f64`; copying the 8 f64 bytes into `F`
        // is the identity for f64 and the types match in size.
        return unsafe { core::mem::transmute_copy::<f64, F>(&total) };
    }
    let mut sum = F::zero();
    for k in 0..left.len() {
        sum = sum + left[k] * right[k];
    }
    sum
}

/// Cast an `&[F]` slice whose element type is `f64` into `&[f64]`.
///
/// Only invoked after a [`TypeId`] guard confirms `F == f64`; the two slice
/// layouts are identical.
fn cast_slice_f64<F: SKFloat>(slice: &[F]) -> &[f64] {
    // SAFETY: guarded by the caller's `TypeId` check; `F` is `f64` here.
    unsafe { core::slice::from_raw_parts(slice.as_ptr() as *const f64, slice.len()) }
}

#[cfg(test)]
mod pairwise_tests;
