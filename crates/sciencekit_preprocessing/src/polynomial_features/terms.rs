//! Exponent-term generation for polynomial expansion.
//!
//! A term is a sorted list of feature indices; its value is the product of the
//! input values at those indices (the empty term is the constant `1`). Terms
//! are laid out bias-first, then by increasing total degree, and within a degree
//! in lexicographic order of the index tuples — the documented, stable column
//! order (scikit-learn parity).

/// Build the expansion terms for `number_of_input_features`, `degree`,
/// `interaction_only` and `include_bias`.
///
/// `interaction_only` restricts a term to distinct features (no repeated index);
/// otherwise a feature may be repeated, producing power terms.
pub(super) fn build_terms(
    number_of_input_features: usize,
    degree: usize,
    interaction_only: bool,
    include_bias: bool,
) -> Vec<Vec<usize>> {
    let mut terms = Vec::new();
    if include_bias {
        terms.push(Vec::new());
    }
    for total_degree in 1..=degree {
        combinations(
            number_of_input_features,
            total_degree,
            interaction_only,
            &mut terms,
        );
    }
    terms
}

/// Append every combination of `size` features to `terms`, in lexicographic
/// order. Repetition is allowed unless `interaction_only`.
fn combinations(
    number_of_input_features: usize,
    size: usize,
    interaction_only: bool,
    terms: &mut Vec<Vec<usize>>,
) {
    let mut current = Vec::with_capacity(size);
    extend_combinations(
        number_of_input_features,
        size,
        interaction_only,
        &mut current,
        0,
        terms,
    );
}

/// Recursively extend `current` with the next feature index, starting the
/// search at `start` to keep the tuples non-decreasing (or strictly increasing
/// when `interaction_only`).
fn extend_combinations(
    number_of_input_features: usize,
    size: usize,
    interaction_only: bool,
    current: &mut Vec<usize>,
    start: usize,
    terms: &mut Vec<Vec<usize>>,
) {
    if current.len() == size {
        terms.push(current.clone());
        return;
    }
    let remaining = size - current.len();
    // With repetition the current slot may take any feature (non-decreasing
    // later slots stay >= it); strictly increasing needs room for the rest.
    let last_start = if interaction_only {
        number_of_input_features.saturating_sub(remaining)
    } else {
        number_of_input_features.saturating_sub(1)
    };
    for feature in start..=last_start {
        current.push(feature);
        let next_start = if interaction_only {
            feature + 1
        } else {
            feature
        };
        extend_combinations(
            number_of_input_features,
            size,
            interaction_only,
            current,
            next_start,
            terms,
        );
        current.pop();
    }
}
