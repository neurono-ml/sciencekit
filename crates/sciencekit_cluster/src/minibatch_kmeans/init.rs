//! Bounded-sample centroid initialization and best-of-`n_init` selection
//! (spec `mini-batch-k-means`, decision 2).

use ndarray::Array2;
use sciencekit_common::{
    SK_RANDOM_DOMAIN_STAGE, SKDataView, SKError, SKFloat, SKRandomGenerator, sk_derive_seed,
};
use sprs::{CsMat, CsMatView};

use super::assign::sk_assign_batch_rows;
use super::config::SKMiniBatchKMeansConfiguration;
use crate::kmeans::distance::sk_view_dimensions;
use crate::kmeans::init::sk_initialize_centers;
use crate::kmeans::selection::sk_run_initializations;

/// Draws at most `initialization_batch_size` sorted row indices deterministically.
pub(crate) fn sk_sample_initialization_indices(
    number_of_rows: usize,
    initialization_batch_size: usize,
    random_seed: u64,
) -> Vec<usize> {
    if number_of_rows <= initialization_batch_size {
        return (0..number_of_rows).collect();
    }
    let sample_seed = sk_derive_seed(random_seed, SK_RANDOM_DOMAIN_STAGE, 0);
    let mut generator = SKRandomGenerator::new(sample_seed);
    let mut sampled =
        generator.sample_indices_without_replacement(number_of_rows, initialization_batch_size);
    sampled.sort_unstable();
    sampled
}

/// Initializes centers from at most `configuration.initialization_batch_size`
/// rows, running `configuration.number_of_initializations` independent seeded
/// trials and returning `(best_centers, sample_inertia)`.
pub(crate) fn sk_initialize_minibatch_centers<F: SKFloat>(
    features: SKDataView<'_, F>,
    configuration: &SKMiniBatchKMeansConfiguration<F>,
) -> Result<(Array2<F>, F), SKError> {
    let (number_of_rows, _) = sk_view_dimensions(&features);
    if number_of_rows <= configuration.initialization_batch_size {
        return run_trials_on_view(features, configuration);
    }

    let indices = sk_sample_initialization_indices(
        number_of_rows,
        configuration.initialization_batch_size,
        configuration.random_seed,
    );

    match features {
        SKDataView::Dense(dense) => {
            let mut sample = Array2::<F>::zeros((indices.len(), dense.ncols()));
            sample.rows_mut().into_iter().zip(indices.iter()).for_each(
                |(mut target_row, &source_index)| {
                    target_row.assign(&dense.row(source_index));
                },
            );
            run_trials_on_view(SKDataView::Dense(sample.view()), configuration)
        }
        SKDataView::Sparse(sparse) => {
            let sample = gather_sparse_rows(&sparse, &indices);
            run_trials_on_view(SKDataView::Sparse(sample.view()), configuration)
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: features.representation(),
            suggestion: "convert the input to a supported dense or CSR view",
        }),
    }
}

fn run_trials_on_view<F: SKFloat>(
    sample: SKDataView<'_, F>,
    configuration: &SKMiniBatchKMeansConfiguration<F>,
) -> Result<(Array2<F>, F), SKError> {
    let ((centers, inertia), _) = sk_run_initializations::<(Array2<F>, F), SKError>(
        configuration.number_of_initializations,
        configuration.random_seed,
        |_, seed| {
            let centers = sk_initialize_centers(
                sample,
                configuration.number_of_clusters,
                configuration.initialization,
                seed,
            )?;
            let (_, distances) = sk_assign_batch_rows(sample, &centers.view())?;
            let inertia = distances
                .iter()
                .fold(F::zero(), |accumulator, &value| accumulator + value);
            let inertia_f64 = inertia.to_f64().expect("finite scalar converts to f64");
            Ok(((centers, inertia), inertia_f64))
        },
    )?;
    Ok((centers, inertia))
}

fn gather_sparse_rows<F: SKFloat>(source: &CsMatView<'_, F>, indices: &[usize]) -> CsMat<F> {
    let mut indptr = Vec::with_capacity(indices.len() + 1);
    let mut columns = Vec::new();
    let mut values = Vec::new();
    indptr.push(0_usize);
    indices.iter().for_each(|&row_index| {
        if let Some(stored) = source.outer_view(row_index) {
            columns.extend_from_slice(stored.indices());
            values.extend_from_slice(stored.data());
        }
        indptr.push(values.len());
    });
    CsMat::new((indices.len(), source.cols()), indptr, columns, values)
}
