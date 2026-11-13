//! Large sparse acceptance: at least 20 000 rows and 10 000 features
//! (PRD §8.7, spec `linear-support-vector-classifier`).

use ndarray::Array1;
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKSupervisedFit};
use sciencekit_metrics::classification::sk_accuracy_score;
use sprs::TriMat;

use crate::SKLinearSVC;

const ROWS: usize = 20_000;
const COLUMNS: usize = 10_000;

fn large_sparse_fixture() -> (sprs::CsMat<f64>, Array1<i64>) {
    let mut triplets = TriMat::new((ROWS, COLUMNS));
    (0..ROWS).for_each(|row| {
        let class = row % 3;
        // A single strong column per class carries the signal; the remaining
        // entries are deterministic sparse noise.
        triplets.add_triplet(row, class, 2.0);
        triplets.add_triplet(row, 5_000 + (row * 7) % 500, 0.25);
        triplets.add_triplet(row, 7_000 + (row * 11) % 500, -0.25);
        triplets.add_triplet(row, 9_000 + (row * 13) % 900, 0.1);
    });
    let labels = Array1::from_shape_fn(ROWS, |row| (row % 3) as i64);
    (triplets.to_csr(), labels)
}

#[test]
fn acceptance_large_sparse_multiclass_fit_reaches_high_accuracy() {
    let (features, labels) = large_sparse_fixture();
    assert_eq!(features.rows(), ROWS);
    assert_eq!(features.cols(), COLUMNS);

    let model = SKLinearSVC::<f64>::build()
        .tolerance(1e-2)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let predicted = model.predict_labels(features.view()).unwrap();
    let accuracy: f64 = sk_accuracy_score(&labels.to_vec(), &predicted.to_vec()).unwrap();
    assert!(accuracy > 0.95, "accuracy was {accuracy}");
    assert_eq!(model.heads().len(), 3);
}
