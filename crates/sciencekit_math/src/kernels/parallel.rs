//! Row-parallel kernel evaluation and unified kernel dispatcher
//! (spec `kernel-functions`).

use ndarray::{Array2, ArrayViewMut1, Axis};
use rayon::prelude::*;
use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::config::{
    SKKernel, SKKernelGamma, SKKernelParameters, sk_validate_kernel_hyperparameters,
};
use super::linear::sk_linear_kernel_with_mode;
use super::polynomial::sk_integer_power;
use super::precomputed::sk_precomputed_kernel_matrix;
use super::radial_basis::sk_radial_basis_kernel_with_mode;
use super::validation::sk_kernel_view_dimensions;

/// Pair count at or above which kernel matrices evaluate rows in parallel.
pub const SK_KERNEL_PARALLEL_GRAIN: usize = 1024;

/// Fill an `(left_rows, right_rows)` matrix row by row, either sequentially or
/// across rayon threads when `allow_parallel` holds and the pair count reaches
/// [`SK_KERNEL_PARALLEL_GRAIN`].
pub(crate) fn sk_fill_kernel_rows<F, C>(
    left_rows: usize,
    right_rows: usize,
    allow_parallel: bool,
    compute_row: C,
) -> Array2<F>
where
    F: SKFloat,
    C: Fn(usize, ArrayViewMut1<'_, F>) + Sync + Send,
{
    let mut output = Array2::zeros((left_rows, right_rows));
    if !allow_parallel || left_rows * right_rows < SK_KERNEL_PARALLEL_GRAIN {
        output
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(row_index, row)| compute_row(row_index, row));
    } else {
        output
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(row_index, row)| compute_row(row_index, row));
    }
    output
}

/// Transform a matrix element-wise in-place, parallelizing across rows when
/// `allow_parallel` holds and the element count reaches
/// [`SK_KERNEL_PARALLEL_GRAIN`].
pub(crate) fn sk_transform_kernel_in_place<F, T>(
    matrix: &mut Array2<F>,
    allow_parallel: bool,
    transform: T,
) where
    F: SKFloat,
    T: Fn(F) -> F + Sync + Send,
{
    if !allow_parallel || matrix.len() < SK_KERNEL_PARALLEL_GRAIN {
        matrix.mapv_inplace(&transform);
    } else {
        matrix
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| row.iter_mut().for_each(|cell| *cell = transform(*cell)));
    }
}

fn sk_dispatch_kernel<F: SKFloat>(
    parameters: &SKKernelParameters<F>,
    left: SKDataView<'_, F>,
    right: SKDataView<'_, F>,
    allow_parallel: bool,
) -> Result<Array2<F>, SKError> {
    sk_validate_kernel_hyperparameters(
        parameters.kernel,
        SKKernelGamma::Value(parameters.gamma),
        parameters.degree,
        parameters.coef0,
    )?;
    match parameters.kernel {
        SKKernel::Linear => sk_linear_kernel_with_mode(left, right, allow_parallel),
        SKKernel::Polynomial => {
            let mut matrix = sk_linear_kernel_with_mode(left, right, allow_parallel)?;
            let gamma = parameters.gamma;
            let coef0 = parameters.coef0;
            let degree = parameters.degree;
            sk_transform_kernel_in_place(&mut matrix, allow_parallel, |dot_product| {
                sk_integer_power(gamma * dot_product + coef0, degree)
            });
            Ok(matrix)
        }
        SKKernel::RadialBasisFunction => {
            sk_radial_basis_kernel_with_mode(left, right, parameters.gamma, allow_parallel)
        }
        SKKernel::Sigmoid => {
            let mut matrix = sk_linear_kernel_with_mode(left, right, allow_parallel)?;
            let gamma = parameters.gamma;
            let coef0 = parameters.coef0;
            sk_transform_kernel_in_place(&mut matrix, allow_parallel, |dot_product| {
                (gamma * dot_product + coef0).tanh()
            });
            Ok(matrix)
        }
        SKKernel::Precomputed => {
            let (expected_rows, _) = sk_kernel_view_dimensions(&right)?;
            sk_precomputed_kernel_matrix(left, expected_rows)
        }
    }
}

/// Compute the kernel matrix sequentially for every row pair.
pub fn sk_kernel_matrix_sequential<F: SKFloat>(
    parameters: &SKKernelParameters<F>,
    left: SKDataView<'_, F>,
    right: SKDataView<'_, F>,
) -> Result<Array2<F>, SKError> {
    sk_dispatch_kernel(parameters, left, right, false)
}

/// Compute the kernel matrix, evaluating independent rows in parallel with
/// rayon when the pair count reaches [`SK_KERNEL_PARALLEL_GRAIN`].
pub fn sk_kernel_matrix<F: SKFloat>(
    parameters: &SKKernelParameters<F>,
    left: SKDataView<'_, F>,
    right: SKDataView<'_, F>,
) -> Result<Array2<F>, SKError> {
    sk_dispatch_kernel(parameters, left, right, true)
}
