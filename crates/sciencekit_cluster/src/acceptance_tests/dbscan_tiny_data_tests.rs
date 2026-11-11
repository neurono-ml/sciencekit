//! Tiny-data acceptance scenarios for `SKDBSCAN`.

use ndarray::array;
use sciencekit_common::{SKBuilder, SKClusterLabel, SKClustererPredictor, SKUnsupervisedFit};

use crate::SKDBSCAN;

#[test]
fn dbscan_acceptance_tiny_single_row_with_minimum_samples_one_forms_one_cluster() {
    let data = array![[3.5_f64, -1.2]];
    let model = SKDBSCAN::<f64>::build()
        .eps(0.5)
        .minimum_samples(1)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    assert_eq!(model.number_of_clusters(), 1);
    assert_eq!(model.core_sample_indices().to_vec(), vec![0]);
    assert_eq!(model.labels().to_vec(), vec![SKClusterLabel::cluster(0)]);
    assert_eq!(
        model
            .predict_clusters(&array![[3.6_f64, -1.2], [9.0, 9.0]])
            .unwrap()
            .to_vec(),
        vec![SKClusterLabel::cluster(0), SKClusterLabel::Noise]
    );
}

#[test]
fn dbscan_acceptance_tiny_single_row_below_minimum_samples_is_noise() {
    let data = array![[3.5_f64, -1.2]];
    let model = SKDBSCAN::<f64>::build()
        .eps(0.5)
        .minimum_samples(2)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    assert_eq!(model.number_of_clusters(), 0);
    assert!(model.core_sample_indices().is_empty());
    assert_eq!(model.labels().to_vec(), vec![SKClusterLabel::Noise]);
}

#[test]
fn dbscan_acceptance_tiny_core_border_and_noise_partition() {
    let data = array![
        [0.0_f64, 0.0],
        [0.3, 0.0],
        [0.0, 0.3],
        [-0.85, 0.0],
        [10.0, 10.0],
    ];
    let model = SKDBSCAN::<f64>::build()
        .eps(0.9)
        .minimum_samples(3)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    assert_eq!(model.number_of_clusters(), 1);
    assert_eq!(model.core_sample_indices().to_vec(), vec![0, 1, 2]);
    assert_eq!(
        model.labels().to_vec(),
        vec![
            SKClusterLabel::cluster(0),
            SKClusterLabel::cluster(0),
            SKClusterLabel::cluster(0),
            SKClusterLabel::cluster(0),
            SKClusterLabel::Noise,
        ]
    );
}
