//! Inertia and metric production for KMeans.

use sciencekit_common::{SKBuilder, SKUnsupervisedFit};

use super::fixtures::{majority_accuracy, planted_design};
use crate::SKKMeans;

#[test]
fn acceptance_metrics_cluster_labels_recover_the_planted_groups() {
    let (data, truth) = planted_design(600, 3);
    let model = SKKMeans::<f64>::build()
        .number_of_clusters(3)
        .random_seed(12)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    assert_eq!(
        majority_accuracy(&truth, &model.labels().to_owned(), 3),
        1.0
    );
}

#[test]
fn acceptance_metrics_inertia_matches_a_direct_recomputation() {
    let (data, _) = planted_design(600, 3);
    let model = SKKMeans::<f64>::build()
        .number_of_clusters(3)
        .random_seed(12)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    let centers = model.cluster_centers();
    let mut expected = 0.0;
    for (row, label) in model.labels().iter().enumerate() {
        let cluster = label.cluster_index().unwrap() as usize;
        for feature in 0..data.ncols() {
            let difference = data[[row, feature]] - centers[[cluster, feature]];
            expected += difference * difference;
        }
    }
    assert!((model.inertia() - expected).abs() < 1e-6);
    assert_eq!(centers.nrows(), 3);
    assert_eq!(centers.ncols(), data.ncols());
}
