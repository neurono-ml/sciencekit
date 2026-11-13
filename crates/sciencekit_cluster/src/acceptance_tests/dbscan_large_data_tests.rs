//! Large-data acceptance scenarios for `SKDBSCAN` (at least 20 000 rows).

use ndarray::{Axis, Slice};
use sciencekit_common::{SKBuilder, SKClusterLabel, SKClustererPredictor, SKUnsupervisedFit};

use super::dbscan_fixtures::planted_density_chains;
use crate::{SKDBSCAN, SKDBSCANSearchStrategy};

#[test]
fn dbscan_acceptance_large_twenty_thousand_row_fit_recovers_chains_and_noise() {
    // 20,000 density-chained rows across 4 non-convex chains (100 stations each)
    // plus 200 isolated noise rows = 20,200 total rows.
    let (data, truth) = planted_density_chains(20_000, 4, 100, 200);
    assert_eq!(data.nrows(), 20_200);

    let model = SKDBSCAN::<f64>::build()
        .eps(0.45)
        .minimum_samples(10)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    assert_eq!(model.search_strategy(), SKDBSCANSearchStrategy::KdTree);
    assert_eq!(model.number_of_clusters(), 4);
    assert_eq!(model.labels().len(), 20_200);
    assert_eq!(model.core_sample_indices().len(), 20_000);

    // Rows 0..4 belong to chains 0..4 in ascending order, so cluster ids 0..4
    // align 1:1 with truth[row], and rows 20_000..20_200 are all Noise (-1).
    for (row, (&expected, &assigned)) in truth.iter().zip(model.labels().iter()).enumerate() {
        if expected == -1 {
            assert_eq!(assigned, SKClusterLabel::Noise, "row {row} must be noise");
        } else {
            assert_eq!(
                assigned,
                SKClusterLabel::cluster(expected as u32),
                "row {row} cluster mismatch"
            );
        }
    }

    let range = Slice::from(19_990..20_010);
    let query_slice = data.slice_axis(Axis(0), range);
    let predicted = model.predict_clusters(query_slice).unwrap();
    assert_eq!(predicted.view(), model.labels().slice_axis(Axis(0), range));
}
