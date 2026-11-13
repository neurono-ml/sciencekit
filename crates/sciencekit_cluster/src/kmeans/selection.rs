//! Best-of-initializations selection (spec `k-means`, decision 5).

use rayon::prelude::*;

pub(crate) use super::init::sk_initialization_seed;

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
