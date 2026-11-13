//! The two prediction trait surfaces of an imported model.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKClassifierPredictor, SKDataView, SKFloat, SKRegressorPredictor};

use super::SKOnnxImportError;
use super::interface::SKOnnxPredictionHead;
use super::predictor::SKOnnxPredictor;

impl<F: SKFloat> SKRegressorPredictor<F> for SKOnnxPredictor<F> {
    type Error = SKOnnxImportError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = sciencekit_common::SKError>,
    {
        self.run_observed(features, |rows, values| {
            self.predict_float_rows(rows, values)
        })
    }
}

impl<F: SKFloat> SKClassifierPredictor<F> for SKOnnxPredictor<F> {
    type Error = SKOnnxImportError;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = sciencekit_common::SKError>,
    {
        self.run_observed(features, |rows, values| {
            let outputs = self.run(rows, values)?;
            let labels = match &self.interface.head {
                SKOnnxPredictionHead::Classification { labels, .. }
                | SKOnnxPredictionHead::LabelsOnly { labels } => labels,
                SKOnnxPredictionHead::Regression { .. } => {
                    return Err(SKOnnxImportError::Central(
                        sciencekit_common::SKError::UnsupportedOperation {
                            operation: "onnx_classification",
                            reason: "the imported model does not declare a label output"
                                .to_string(),
                        },
                    ));
                }
            };
            let value = outputs
                .get(labels)
                .ok_or_else(|| SKOnnxImportError::UnsupportedOutput {
                    detail: format!("the label output `{labels}` was not produced"),
                })?;
            Ok(Array1::from_iter(value.to_i64_values()))
        })
    }

    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = sciencekit_common::SKError>,
    {
        self.run_observed(features, |rows, values| {
            let outputs = self.run(rows, values)?;
            let (name, class_count) = match &self.interface.head {
                SKOnnxPredictionHead::Classification {
                    probabilities,
                    class_count,
                    ..
                } => (probabilities, *class_count),
                _ => {
                    return Err(SKOnnxImportError::Central(
                        sciencekit_common::SKError::UnsupportedOperation {
                            operation: "onnx_classification",
                            reason: "the imported model does not declare a probability matrix"
                                .to_string(),
                        },
                    ));
                }
            };
            let value = outputs
                .get(name)
                .ok_or_else(|| SKOnnxImportError::UnsupportedOutput {
                    detail: format!("the probability output `{name}` was not produced"),
                })?;
            let matrix = Array2::from_shape_vec(
                (rows, class_count),
                value
                    .to_f64_values()
                    .into_iter()
                    .map(|value| num_traits::NumCast::from(value).unwrap_or_else(F::zero))
                    .collect(),
            )
            .map_err(|_| SKOnnxImportError::ShapeMismatch {
                expected: vec![rows, class_count],
                found: value.shape(),
            })?;
            Ok(matrix)
        })
    }
}
