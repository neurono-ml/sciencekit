//! Homogeneity, completeness and V-measure (spec `clustering-metrics`).

use ndarray::ArrayView1;
use sciencekit_common::SKError;

use super::contingency::{
    SKContingencyTable, sk_contingency_table, sk_entropy, sk_mutual_information,
};

/// Homogeneity: each cluster contains only members of one class.
///
/// `1 - H(true | predicted) / H(true)`. A trivial true labeling (one class)
/// is perfectly homogeneous by convention. Values lie in `[0, 1]`; higher is
/// better.
///
/// **Noise policy:** a label is just a label; filter noise before calling if
/// it should not participate.
pub fn sk_homogeneity_score(
    labels_true: ArrayView1<'_, i64>,
    labels_pred: ArrayView1<'_, i64>,
) -> Result<f64, SKError> {
    let table = sk_contingency_table(labels_true, labels_pred, "homogeneity")?;
    Ok(sk_homogeneity_of(&table))
}

/// Completeness: all members of a class are assigned to the same cluster.
///
/// `1 - H(predicted | true) / H(predicted)`. A trivial predicted labeling is
/// perfectly complete by convention. Values lie in `[0, 1]`; higher is better.
///
/// **Noise policy:** a label is just a label; filter noise before calling.
pub fn sk_completeness_score(
    labels_true: ArrayView1<'_, i64>,
    labels_pred: ArrayView1<'_, i64>,
) -> Result<f64, SKError> {
    let table = sk_contingency_table(labels_true, labels_pred, "completeness")?;
    Ok(sk_completeness_of(&table))
}

/// V-measure: the harmonic mean of homogeneity and completeness.
///
/// Both trivial labelings score `1`; when both components are zero the score
/// is `0`. Values lie in `[0, 1]`; higher is better.
///
/// **Noise policy:** a label is just a label; filter noise before calling.
pub fn sk_v_measure_score(
    labels_true: ArrayView1<'_, i64>,
    labels_pred: ArrayView1<'_, i64>,
) -> Result<f64, SKError> {
    let table = sk_contingency_table(labels_true, labels_pred, "v_measure")?;
    let homogeneity = sk_homogeneity_of(&table);
    let completeness = sk_completeness_of(&table);
    if homogeneity + completeness == 0.0 {
        return Ok(0.0);
    }
    Ok(2.0 * homogeneity * completeness / (homogeneity + completeness))
}

/// Homogeneity of an already-built contingency table.
fn sk_homogeneity_of(table: &SKContingencyTable) -> f64 {
    if table.true_counts.len() == 1 && table.predicted_counts.len() == 1 {
        return 1.0;
    }
    let total = table.number_of_rows as f64;
    let true_entropy = sk_entropy(&table.true_counts, total);
    if true_entropy == 0.0 {
        return 1.0;
    }
    let conditional_entropy = true_entropy - sk_mutual_information(table);
    1.0 - conditional_entropy / true_entropy
}

/// Completeness of an already-built contingency table.
fn sk_completeness_of(table: &SKContingencyTable) -> f64 {
    if table.true_counts.len() == 1 && table.predicted_counts.len() == 1 {
        return 1.0;
    }
    let total = table.number_of_rows as f64;
    let predicted_entropy = sk_entropy(&table.predicted_counts, total);
    if predicted_entropy == 0.0 {
        return 1.0;
    }
    let conditional_entropy = predicted_entropy - sk_mutual_information(table);
    1.0 - conditional_entropy / predicted_entropy
}
