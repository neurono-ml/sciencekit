//! Example classifiers shared by the ranking scorer tests.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKClassifierPredictor, SKDataView, SKError, SKFloat};

/// A classifier returning a fixed probability matrix, ignoring the features.
pub(super) struct FixedProbabilityClassifier {
    pub(super) probabilities: Vec<Vec<f64>>,
}

impl<F: SKFloat> SKClassifierPredictor<F> for FixedProbabilityClassifier {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, _features: X) -> Result<Array1<i64>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        Ok(Array1::zeros(self.probabilities.len()))
    }

    fn predict_probabilities<'a, X>(&self, _features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let rows = self.probabilities.len();
        let columns = self.probabilities.first().map(Vec::len).unwrap_or(0);
        let mut output = Array2::zeros((rows, columns));
        for (row, values) in self.probabilities.iter().enumerate() {
            for (column, value) in values.iter().enumerate() {
                output[[row, column]] =
                    num_traits::cast::<f64, F>(*value).expect("values fit the scalar");
            }
        }
        Ok(output)
    }
}

/// A classifier whose inference must never run in the pure form.
pub(super) struct PanickingClassifier;

impl<F: SKFloat> SKClassifierPredictor<F> for PanickingClassifier {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, _features: X) -> Result<Array1<i64>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        panic!("the pure form must not run inference")
    }

    fn predict_probabilities<'a, X>(&self, _features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        panic!("the pure form must not run inference")
    }
}
