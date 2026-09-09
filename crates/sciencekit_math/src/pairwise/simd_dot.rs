//! SIMD-friendly dot-product and row-norm helpers shared by the distance
//! kernels. The `f64` hot path processes four lanes at a time with fused
//! multiply-add; shorter tails and non-`f64` types fall back to plain
//! accumulation.

use sciencekit_common::SKFloat;
use wide::f64x4;

/// SIMD-friendly dot product over contiguous row data.
///
/// The `f64` hot path processes four lanes at a time with [`f64x4`] fused
/// multiply-add; shorter tails and non-`f64` types fall back to a plain
/// accumulation. Contiguity is handled by ndarray's row iterators.
pub(crate) fn dot_product_simd<F: SKFloat>(
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

/// Sum of the squares of a row (`‖x‖²`).
pub(crate) fn row_norm_squared<F: SKFloat>(row: ndarray::ArrayView1<F>) -> F {
    dot_product_simd(&row, &row)
}

/// Euclidean norm of a row.
pub(crate) fn row_norm<F: SKFloat>(row: ndarray::ArrayView1<F>) -> F {
    row_norm_squared(row).sqrt()
}

/// Cast an `&[F]` slice whose element type is `f64` into `&[f64]`.
///
/// Only invoked after a [`TypeId`] guard confirms `F == f64`; the two slice
/// layouts are identical.
fn cast_slice_f64<F: SKFloat>(slice: &[F]) -> &[f64] {
    // SAFETY: guarded by the caller's `TypeId` check; `F` is `f64` here.
    unsafe { core::slice::from_raw_parts(slice.as_ptr() as *const f64, slice.len()) }
}
