//! Deterministic stratified five-fold cross-validated decision values for Platt
//! scaling (spec `support-vector-machines`).

use ndarray::Array1;
use sciencekit_common::{
    SK_RANDOM_DOMAIN_FOLD, SKDataView, SKError, SKFloat, SKRandomGenerator, sk_derive_seed,
};
use sciencekit_math::{SKKernel, SKKernelParameters, sk_kernel_matrix};

use super::platt::{SKPlattParameters, sk_fit_platt_sigmoid};
use crate::kernel::SKOwnedFeatureBlock;
use crate::smo::{SKBinarySvcSmoResult, SKSmoOptions, sk_solve_binary_c_svc};
use crate::svc::fit::sk_extract_subset_features;

fn sk_score_fold_rows<F: SKFloat>(
    pair_features: SKDataView<'_, F>,
    train_indices: &[usize],
    test_indices: &[usize],
    solved: &SKBinarySvcSmoResult<F>,
    kernel_parameters: SKKernelParameters<F>,
) -> Result<Vec<F>, SKError> {
    if solved.dual_coefficients.is_empty() {
        return Ok(vec![solved.bias; test_indices.len()]);
    }
    let dual = Array1::from(solved.dual_coefficients.clone());
    if kernel_parameters.kernel == SKKernel::Precomputed {
        let dense = pair_features.as_dense()?;
        return Ok(test_indices
            .iter()
            .map(|&test_row| {
                solved
                    .support_indices
                    .iter()
                    .zip(dual.iter())
                    .fold(solved.bias, |acc, (&sv_local, &coef)| {
                        acc + coef * dense[[test_row, train_indices[sv_local]]]
                    })
            })
            .collect());
    }
    let sv_rows: Vec<usize> = solved
        .support_indices
        .iter()
        .map(|&local| train_indices[local])
        .collect();
    let support_block = SKOwnedFeatureBlock::select_rows(pair_features, &sv_rows)?;
    let test_block = SKOwnedFeatureBlock::select_rows(pair_features, test_indices)?;
    let gram = sk_kernel_matrix(&kernel_parameters, test_block.view(), support_block.view())?;
    Ok((0..test_indices.len())
        .map(|row| gram.row(row).dot(&dual) + solved.bias)
        .collect())
}

/// Compute deterministic five-fold out-of-fold decision values (falling back to
/// in-sample margins when a class has fewer than two rows) and fit Platt scaling.
#[allow(clippy::too_many_arguments)]
pub fn sk_calibrate_binary_platt<F: SKFloat>(
    pair_features: SKDataView<'_, F>,
    signs: &[i8],
    upper_bounds: &[F],
    kernel_parameters: SKKernelParameters<F>,
    options: SKSmoOptions<F>,
    random_seed: u64,
    pair_index: u64,
) -> Result<SKPlattParameters<F>, SKError> {
    let mut positive_indices: Vec<usize> = signs
        .iter()
        .enumerate()
        .filter_map(|(i, &s)| (s > 0).then_some(i))
        .collect();
    let mut negative_indices: Vec<usize> = signs
        .iter()
        .enumerate()
        .filter_map(|(i, &s)| (s < 0).then_some(i))
        .collect();

    let mut generator =
        SKRandomGenerator::new(sk_derive_seed(random_seed, SK_RANDOM_DOMAIN_FOLD, pair_index));
    generator.shuffle(&mut positive_indices);
    generator.shuffle(&mut negative_indices);

    let fold_count = 5
        .min(positive_indices.len())
        .min(negative_indices.len());
    let total_rows = signs.len();

    if fold_count < 2 {
        let solved = sk_solve_binary_c_svc(
            signs,
            upper_bounds,
            pair_features,
            kernel_parameters,
            options,
        )?;
        let all_rows: Vec<usize> = (0..total_rows).collect();
        let margins = sk_score_fold_rows(
            pair_features,
            &all_rows,
            &all_rows,
            &solved,
            kernel_parameters,
        )?;
        return Ok(sk_fit_platt_sigmoid(&margins, signs));
    }

    let mut row_fold = vec![0usize; total_rows];
    positive_indices
        .iter()
        .enumerate()
        .for_each(|(rank, &row)| row_fold[row] = rank % fold_count);
    negative_indices
        .iter()
        .enumerate()
        .for_each(|(rank, &row)| row_fold[row] = rank % fold_count);

    let mut out_of_fold = vec![F::zero(); total_rows];
    for fold in 0..fold_count {
        let train_indices: Vec<usize> = (0..total_rows).filter(|&r| row_fold[r] != fold).collect();
        let test_indices: Vec<usize> = (0..total_rows).filter(|&r| row_fold[r] == fold).collect();
        let train_signs: Vec<i8> = train_indices.iter().map(|&r| signs[r]).collect();
        let train_bounds: Vec<F> = train_indices.iter().map(|&r| upper_bounds[r]).collect();
        let train_block =
            sk_extract_subset_features(pair_features, kernel_parameters.kernel, &train_indices)?;
        let solved = sk_solve_binary_c_svc(
            &train_signs,
            &train_bounds,
            train_block.view(),
            kernel_parameters,
            options,
        )?;
        let fold_margins = sk_score_fold_rows(
            pair_features,
            &train_indices,
            &test_indices,
            &solved,
            kernel_parameters,
        )?;
        test_indices
            .iter()
            .zip(fold_margins.into_iter())
            .for_each(|(&row, margin)| out_of_fold[row] = margin);
    }

    Ok(sk_fit_platt_sigmoid(&out_of_fold, signs))
}
