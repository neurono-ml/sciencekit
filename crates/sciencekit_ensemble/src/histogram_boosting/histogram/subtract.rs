//! Histogram subtraction from the parent node (spec
//! `histogram-gradient-boosting`, design decision 4).

use sciencekit_common::SKFloat;

use super::accumulate::SKNodeHistogram;

/// Derive the larger child histogram as `parent − smaller` without revisiting
/// the larger child's rows.
pub fn sk_subtract_histogram<F: SKFloat>(
    parent: &SKNodeHistogram<F>,
    smaller_child: &SKNodeHistogram<F>,
) -> SKNodeHistogram<F> {
    let features = parent
        .features
        .iter()
        .zip(smaller_child.features.iter())
        .map(|(parent_feature, smaller_feature)| {
            let mut difference = parent_feature.clone();
            difference.subtract_assign(smaller_feature);
            difference
        })
        .collect();
    SKNodeHistogram { features }
}
