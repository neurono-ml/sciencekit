//! Minimal estimators used by the cross-validation tests.

use std::marker::PhantomData;

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKFloat, SKRegressorPredictor, SKSupervisedFit,
    SKTargetView,
};

fn sk_number_of_rows<F>(view: SKDataView<'_, F>) -> usize {
    match view {
        SKDataView::Dense(features) => features.nrows(),
        SKDataView::Sparse(features) => features.rows(),
        _ => 0,
    }
}

/// A regressor that predicts the mean of its training targets.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SKMockMeanRegressor<F: SKFloat> {
    marker: PhantomData<F>,
}

impl<F: SKFloat> SKMockMeanRegressor<F> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

/// The fitted mean predictor.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SKMockMeanRegressorModel<F: SKFloat> {
    mean: F,
}

impl<F: SKFloat> SKSupervisedFit<F> for SKMockMeanRegressor<F> {
    type Model = SKMockMeanRegressorModel<F>;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let targets = targets.try_into()?;
        if sk_number_of_rows(features) != targets.len() {
            return Err(SKError::ShapeMismatch {
                expected: vec![sk_number_of_rows(features)],
                found: vec![targets.len()],
            });
        }
        let values = targets.as_continuous()?;
        let sum = values
            .iter()
            .copied()
            .fold(F::zero(), |left, right| left + right);
        let count = F::from(values.len()).expect("the row count fits the scalar");
        Ok(SKMockMeanRegressorModel { mean: sum / count })
    }
}

impl<F: SKFloat> SKRegressorPredictor<F> for SKMockMeanRegressorModel<F> {
    type Error = SKError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        Ok(Array1::from_elem(sk_number_of_rows(features), self.mean))
    }
}

/// A classifier that predicts the most frequent training label.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SKMockMajorityClassifier<F: SKFloat> {
    marker: PhantomData<F>,
}

impl<F: SKFloat> SKMockMajorityClassifier<F> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

/// The fitted majority-label predictor (two binary classes).
#[derive(Debug, Clone, Copy)]
pub(crate) struct SKMockMajorityClassifierModel {
    majority: i64,
}

impl<F: SKFloat> SKSupervisedFit<F> for SKMockMajorityClassifier<F> {
    type Model = SKMockMajorityClassifierModel;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let targets = targets.try_into()?;
        if sk_number_of_rows(features) != targets.len() {
            return Err(SKError::ShapeMismatch {
                expected: vec![sk_number_of_rows(features)],
                found: vec![targets.len()],
            });
        }
        let SKTargetView::Integer(labels) = targets else {
            return Err(SKError::InvalidHyperparameter {
                name: "targets",
                reason: "the mock classifier reads integer labels".to_owned(),
            });
        };
        let zeros = labels.iter().filter(|&&label| label == 0).count();
        let ones = labels.iter().filter(|&&label| label == 1).count();
        Ok(SKMockMajorityClassifierModel {
            majority: i64::from(ones > zeros),
        })
    }
}

impl<F: SKFloat> SKClassifierPredictor<F> for SKMockMajorityClassifierModel {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        Ok(Array1::from_elem(
            sk_number_of_rows(features),
            self.majority,
        ))
    }

    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let number_of_rows = sk_number_of_rows(features);
        let mut probabilities = Array2::zeros((number_of_rows, 2));
        let majority_column = self.majority as usize;
        for mut row in probabilities.rows_mut() {
            row[majority_column] = F::one();
        }
        Ok(probabilities)
    }
}
