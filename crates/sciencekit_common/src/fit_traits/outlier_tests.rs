//! Tests for the typed outlier prediction contract (spec `outlier-prediction`).

use ndarray::{Array1, Array2, array};

use crate::data_view::SKDataView;
use crate::sk_float::SKFloat;
use crate::{SKError, SKOutlierLabel, SKOutlierPredictor};

#[test]
fn outlier_label_size_is_one_byte_and_variants_report_roles() {
    assert_eq!(core::mem::size_of::<SKOutlierLabel>(), 1);
    assert!(SKOutlierLabel::Inlier.is_inlier());
    assert!(!SKOutlierLabel::Inlier.is_outlier());
    assert!(SKOutlierLabel::Outlier.is_outlier());
    assert!(!SKOutlierLabel::Outlier.is_inlier());
}

#[test]
fn outlier_label_i64_interop_round_trips_and_rejects_invalid_values() {
    assert_eq!(SKOutlierLabel::Inlier.to_i64(), 1);
    assert_eq!(SKOutlierLabel::Outlier.to_i64(), -1);
    assert_eq!(SKOutlierLabel::from_i64(1).unwrap(), SKOutlierLabel::Inlier);
    assert_eq!(
        SKOutlierLabel::from_i64(-1).unwrap(),
        SKOutlierLabel::Outlier
    );

    for invalid in [0_i64, 2, -2, i64::MIN, i64::MAX] {
        let error = SKOutlierLabel::from_i64(invalid).unwrap_err();
        assert!(matches!(error, SKError::Conversion(_)));
    }
}

#[test]
fn outlier_constant_detector_agrees_across_scalars_representations_and_threads() {
    let detector = ConstantRadiusDetector;
    let dense_f64 = array![[0.1_f64, 0.2], [0.5, 0.0], [3.0, 4.0]];
    let dense_f32 = array![[0.1_f32, 0.2], [0.5, 0.0], [3.0, 4.0]];
    let sparse_f64 = sprs::CsMat::new(
        (3, 2),
        vec![0, 2, 3, 5],
        vec![0, 1, 0, 0, 1],
        vec![0.1_f64, 0.2, 0.5, 3.0, 4.0],
    );

    let labels_f64 = SKOutlierPredictor::<f64>::predict(&detector, &dense_f64).unwrap();
    let labels_f32 = SKOutlierPredictor::<f32>::predict(&detector, &dense_f32).unwrap();
    let labels_sparse = SKOutlierPredictor::<f64>::predict(&detector, sparse_f64.view()).unwrap();

    assert_eq!(labels_f64.len(), 3);
    assert_eq!(labels_f64, labels_f32);
    assert_eq!(labels_f64, labels_sparse);
    assert_eq!(
        labels_f64,
        array![
            SKOutlierLabel::Inlier,
            SKOutlierLabel::Inlier,
            SKOutlierLabel::Outlier
        ]
    );

    let scores = SKOutlierPredictor::<f64>::score_samples(&detector, &dense_f64).unwrap();
    let decisions = SKOutlierPredictor::<f64>::decision_function(&detector, &dense_f64).unwrap();
    let offset = SKOutlierPredictor::<f64>::offset(&detector);
    assert_eq!(offset, -0.5);

    scores
        .iter()
        .zip(decisions.iter())
        .zip(labels_f64.iter())
        .for_each(|((&score, &decision), &label)| {
            assert!((decision - (score - offset)).abs() < 1e-12);
            if decision < 0.0 {
                assert_eq!(label, SKOutlierLabel::Outlier);
            } else {
                assert_eq!(label, SKOutlierLabel::Inlier);
            }
        });

    let concurrent = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| SKOutlierPredictor::<f64>::predict(&detector, &dense_f64).unwrap())
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    concurrent
        .into_iter()
        .for_each(|observed| assert_eq!(observed, labels_f64));
}

#[test]
fn outlier_detector_rejects_feature_mismatch() {
    let detector = ConstantRadiusDetector;
    let wrong = Array2::<f64>::zeros((2, 3));
    let error = SKOutlierPredictor::<f64>::predict(&detector, &wrong).unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// Stand-in fitted detector scoring `-sum(|x_j|) / 10` with fixed offset `-0.5`.
struct ConstantRadiusDetector;

impl<F: SKFloat> SKOutlierPredictor<F> for ConstantRadiusDetector {
    type Error = SKError;

    fn offset(&self) -> F {
        F::from(-0.5).expect("offset fits scalar")
    }

    fn score_samples<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let scale = F::from(0.1).expect("scale fits scalar");
        match view {
            SKDataView::Dense(dense) => {
                if dense.ncols() != 2 {
                    return Err(SKError::shape_mismatch_2d(
                        dense.nrows(),
                        2,
                        dense.nrows(),
                        dense.ncols(),
                    ));
                }
                Ok(Array1::from_iter(dense.rows().into_iter().map(|row| {
                    let norm_one = row
                        .iter()
                        .fold(F::zero(), |accumulator, &value| accumulator + value.abs());
                    -(norm_one * scale)
                })))
            }
            SKDataView::Sparse(sparse) => {
                if sparse.cols() != 2 {
                    return Err(SKError::shape_mismatch_2d(
                        sparse.rows(),
                        2,
                        sparse.rows(),
                        sparse.cols(),
                    ));
                }
                Ok(Array1::from_iter(sparse.outer_iterator().map(|row| {
                    let norm_one = row
                        .data()
                        .iter()
                        .fold(F::zero(), |accumulator, &value| accumulator + value.abs());
                    -(norm_one * scale)
                })))
            }
        }
    }

    fn decision_function<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let offset = <Self as SKOutlierPredictor<F>>::offset(self);
        let scores = self.score_samples(features)?;
        Ok(scores.mapv(|score| score - offset))
    }

    fn predict<'a, X>(&self, features: X) -> Result<Array1<SKOutlierLabel>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let decisions = self.decision_function(features)?;
        Ok(decisions.mapv(|decision| {
            if decision < F::zero() {
                SKOutlierLabel::Outlier
            } else {
                SKOutlierLabel::Inlier
            }
        }))
    }
}
