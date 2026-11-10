//! Seeded centroid initialization and best-of-initializations selection
//! (spec `k-means`, decisions 3 and 5).

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use sciencekit_common::{
    SK_RANDOM_DOMAIN_INITIALIZATION, SKDataView, SKError, SKFloat, SKRandomGenerator,
    sk_derive_seed,
};

use super::config::SKKMeansInitialization;
use super::distance::sk_squared_distances_to_center;

/// Derives the sub-seed of one initialization from the estimator seed.
pub(crate) fn sk_initialization_seed(random_seed: u64, initialization_index: usize) -> u64 {
    sk_derive_seed(
        random_seed,
        SK_RANDOM_DOMAIN_INITIALIZATION,
        initialization_index as u64,
    )
}

/// Builds the initial centers for one seeded initialization.
pub(crate) fn sk_initialize_centers<F: SKFloat>(
    features: SKDataView<'_, F>,
    number_of_clusters: usize,
    initialization: SKKMeansInitialization,
    seed: u64,
) -> Result<Array2<F>, SKError> {
    let number_of_rows = match features {
        SKDataView::Dense(dense) => dense.nrows(),
        SKDataView::Sparse(sparse) => sparse.rows(),
        _ => 0,
    };
    let mut generator = SKRandomGenerator::new(seed);
    let indices = match initialization {
        SKKMeansInitialization::Random => {
            generator.sample_indices_without_replacement(number_of_rows, number_of_clusters)
        }
        SKKMeansInitialization::KMeansPlusPlus => {
            sk_kmeans_plus_plus_indices(features, number_of_clusters, &mut generator)?
        }
    };
    Ok(sk_materialize_centers(features, &indices))
}

/// D² sampling over the squared distance to the nearest chosen center.
fn sk_kmeans_plus_plus_indices<F: SKFloat>(
    features: SKDataView<'_, F>,
    number_of_clusters: usize,
    generator: &mut SKRandomGenerator,
) -> Result<Vec<usize>, SKError> {
    let number_of_rows = match features {
        SKDataView::Dense(dense) => dense.nrows(),
        SKDataView::Sparse(sparse) => sparse.rows(),
        _ => 0,
    };
    let mut selected = vec![generator.next_index(number_of_rows)];
    let mut nearest = sk_squared_distances_to_center(features, sk_row_as_vector(features, selected[0]).view())?;

    while selected.len() < number_of_clusters {
        let total = nearest
            .iter()
            .fold(F::zero(), |accumulator, value| accumulator + *value);
        let candidate = if total > F::zero() {
            sk_sample_weighted_index(&nearest, total, generator)
        } else {
            (0..number_of_rows).find(|index| !selected.contains(index))
        };
        let candidate = candidate.expect("validated cluster count fits the rows");
        let candidate_row = sk_row_as_vector(features, candidate);
        let distances = sk_squared_distances_to_center(features, candidate_row.view())?;
        nearest = nearest
            .iter()
            .zip(distances.iter())
            .map(|(current, candidate_distance)| {
                if *candidate_distance < *current {
                    *candidate_distance
                } else {
                    *current
                }
            })
            .collect();
        selected.push(candidate);
    }
    Ok(selected)
}

/// Draws one index with probability proportional to its weight.
fn sk_sample_weighted_index<F: SKFloat>(
    weights: &Array1<F>,
    total: F,
    generator: &mut SKRandomGenerator,
) -> Option<usize> {
    let target = F::from(generator.next_f64()).expect("a unit draw fits the scalar") * total;
    let mut cumulative = F::zero();
    for (index, weight) in weights.iter().enumerate() {
        if *weight <= F::zero() {
            continue;
        }
        cumulative = cumulative + *weight;
        if cumulative > target {
            return Some(index);
        }
    }
    weights
        .iter()
        .enumerate()
        .rev()
        .find(|(_, weight)| **weight > F::zero())
        .map(|(index, _)| index)
}

/// Copies the selected training rows into a dense center matrix.
fn sk_materialize_centers<F: SKFloat>(
    features: SKDataView<'_, F>,
    indices: &[usize],
) -> Array2<F> {
    let number_of_features = match features {
        SKDataView::Dense(dense) => dense.ncols(),
        SKDataView::Sparse(sparse) => sparse.cols(),
        _ => 0,
    };
    let mut centers = Array2::zeros((indices.len(), number_of_features));
    match features {
        SKDataView::Dense(dense) => {
            for (center_index, &row_index) in indices.iter().enumerate() {
                centers.row_mut(center_index).assign(&dense.row(row_index));
            }
        }
        SKDataView::Sparse(sparse) => {
            for (center_index, &row_index) in indices.iter().enumerate() {
                let row = sparse
                    .outer_view(row_index)
                    .expect("row index comes from the validated shape");
                for (column, value) in row.iter() {
                    centers[[center_index, column]] = *value;
                }
            }
        }
        _ => {}
    }
    centers
}

/// Copies one training row into a dense vector.
fn sk_row_as_vector<F: SKFloat>(features: SKDataView<'_, F>, row_index: usize) -> Array1<F> {
    match features {
        SKDataView::Dense(dense) => dense.row(row_index).to_owned(),
        SKDataView::Sparse(sparse) => {
            let mut row = Array1::zeros(sparse.cols());
            if let Some(stored) = sparse.outer_view(row_index) {
                for (column, value) in stored.iter() {
                    row[column] = *value;
                }
            }
            row
        }
        _ => Array1::zeros(0),
    }
}

/// Runs every initialization in parallel and returns the lowest-inertia model,
/// breaking exact ties by the earliest initialization index.
pub(crate) fn sk_run_initializations<T, E>(
    number_of_initializations: usize,
    random_seed: u64,
    fit_initialization: impl Fn(usize, u64) -> Result<(T, f64), E> + Sync,
) -> Result<(T, usize), E>
where
    T: Send,
    E: Send,
{
    let results: Result<Vec<(T, f64)>, E> = (0..number_of_initializations)
        .into_par_iter()
        .map(|index| fit_initialization(index, sk_initialization_seed(random_seed, index)))
        .collect();
    let results = results?;
    let best_index = sk_choose_lowest_inertia_index(
        results
            .iter()
            .map(|(_, inertia)| *inertia)
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let (best_model, _) = results
        .into_iter()
        .nth(best_index)
        .expect("a validated initialization count is non-empty");
    Ok((best_model, best_index))
}

/// Index of the smallest inertia; exact ties keep the earliest index.
pub(crate) fn sk_choose_lowest_inertia_index(inertias: &[f64]) -> usize {
    let mut best_index = 0;
    let mut best_inertia = inertias[0];
    for (index, inertia) in inertias.iter().enumerate().skip(1) {
        if *inertia < best_inertia {
            best_index = index;
            best_inertia = *inertia;
        }
    }
    best_index
}
