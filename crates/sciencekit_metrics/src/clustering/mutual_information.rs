//! Normalized mutual information (spec `clustering-metrics`).

use ndarray::ArrayView1;
use sciencekit_common::SKError;

use super::contingency::{sk_contingency_table, sk_entropy, sk_mutual_information};

/// How the mutual information is normalized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SKMutualInformationNormalization {
    /// `2·MI / (H(U) + H(V))` — the arithmetic mean of the entropies.
    Arithmetic,
    /// `MI / sqrt(H(U)·H(V))` — the geometric mean of the entropies.
    Geometric,
    /// `MI / max(H(U), H(V))` — the larger entropy.
    Maximum,
}

/// Normalized mutual information between two labelings.
///
/// The score is `1` for identical labelings up to permutation and `0` for
/// independent labelings. When both labelings are trivial (one cluster each)
/// the score is `1`; when only one is trivial it is `0`. Entropies are in nats.
///
/// **Noise policy:** a label is just a label here; noise (`-1`) participates
/// as an ordinary label unless the caller filters it.
pub fn sk_normalized_mutual_information_score(
    labels_true: ArrayView1<'_, i64>,
    labels_pred: ArrayView1<'_, i64>,
    normalization: SKMutualInformationNormalization,
) -> Result<f64, SKError> {
    let table = sk_contingency_table(labels_true, labels_pred, "normalized_mutual_information")?;
    if table.true_counts.len() == 1 && table.predicted_counts.len() == 1 {
        return Ok(1.0);
    }
    let total = table.number_of_rows as f64;
    let true_entropy = sk_entropy(&table.true_counts, total);
    let predicted_entropy = sk_entropy(&table.predicted_counts, total);
    let mutual_information = sk_mutual_information(&table);
    let normalizer = match normalization {
        SKMutualInformationNormalization::Arithmetic => (true_entropy + predicted_entropy) / 2.0,
        SKMutualInformationNormalization::Geometric => (true_entropy * predicted_entropy).sqrt(),
        SKMutualInformationNormalization::Maximum => true_entropy.max(predicted_entropy),
    };
    if normalizer == 0.0 {
        return Ok(0.0);
    }
    Ok(mutual_information / normalizer)
}
