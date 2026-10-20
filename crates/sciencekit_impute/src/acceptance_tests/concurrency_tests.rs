//! Acceptance under concurrency: one estimator fitted from many threads and
//! one model transformed from many threads.

use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};

use super::fixtures::generated_features;
use crate::simple_imputer::SKSimpleImputer;

#[test]
fn the_shared_estimator_fits_concurrently() {
    let estimator = SKSimpleImputer::<f64>::build()
        .add_indicator(true)
        .build()
        .unwrap();
    let features = generated_features(400, 6, 10);
    let expected = estimator.fit(&features).unwrap().statistics().to_owned();

    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| scope.spawn(|| estimator.fit(&features).unwrap().statistics().to_owned()))
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), expected);
        }
    });
}

#[test]
fn one_fitted_model_transforms_concurrently() {
    let features = generated_features(300, 5, 12);
    let model = SKSimpleImputer::<f64>::build()
        .add_indicator(true)
        .build()
        .unwrap()
        .fit(&features)
        .unwrap();
    let expected = model.transform(&features).unwrap();

    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| scope.spawn(|| model.transform(&features).unwrap()))
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), expected);
        }
    });
}
