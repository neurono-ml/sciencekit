//! Round and resource scheduling for successive halving
//! (spec `halving-grid-search`, requirement `Successive halving loop`).

/// Computes the geometric resource budgets `minimum_resource * factor^round`
/// bounded by `maximum_resource`.
pub fn sk_halving_resource_schedule(
    minimum_resource: usize,
    maximum_resource: usize,
    factor: usize,
) -> Vec<usize> {
    if minimum_resource == 0 || minimum_resource > maximum_resource || factor < 2 {
        return Vec::new();
    }
    std::iter::successors(Some(minimum_resource), |&current| {
        current
            .checked_mul(factor)
            .filter(|&next| next <= maximum_resource)
    })
    .collect()
}

/// Computes the candidate count trajectory shrinking by `ceil(count / factor)`
/// until a single survivor remains.
pub fn sk_halving_candidate_schedule(number_of_candidates: usize, factor: usize) -> Vec<usize> {
    if number_of_candidates == 0 || factor < 2 {
        return Vec::new();
    }
    std::iter::successors(Some(number_of_candidates), |&current| {
        (current > 1).then(|| current.div_ceil(factor))
    })
    .collect()
}

/// Computes how many candidates survive a round given the remaining budget.
#[allow(dead_code)]
pub(crate) fn sk_surviving_candidate_count(
    alive_count: usize,
    factor: usize,
    _current_resource: usize,
    _maximum_resource: usize,
    _aggressive_elimination: bool,
) -> usize {
    if alive_count == 0 || factor < 2 {
        return 0;
    }
    alive_count.div_ceil(factor).max(1)
}
