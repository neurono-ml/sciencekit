//! Prediction surface (`decision_function`, `predict_labels`,
//! `predict_probabilities`) on [`SKLinearSVCModel`]
//! (spec `linear-support-vector-classifier`).

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    sk_run_operation,
};

use super::decision::{sk_linear_svc_decision_matrix, sk_linear_svc_labels_from_decisions};
use super::model::SKLinearSVCModel;
use crate::kernel::sk_svm_view_dimensions;

impl<F: SKFloat> SKLinearSVCModel<F> {
    /// One decision value per class for every row, shape `(rows, classes)`.
    ///
    /// Reads dense rows directly and scans the stored entries of a CSR input;
    /// a sparse query is never densified.
    pub fn decision_function<'a, X>(&self, features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let (rows_count, columns_count) = sk_svm_view_dimensions(&features).unwrap_or((0, 0));
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Predict,
            rows_count,
            columns_count,
            execution_mode: self.execution_mode(),
            backend: self.backend,
        };
        sk_run_operation(attributes, || sk_linear_svc_decision_matrix(self, features))
    }
}

impl<F: SKFloat> SKClassifierPredictor<F> for SKLinearSVCModel<F> {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let (rows_count, columns_count) = sk_svm_view_dimensions(&features).unwrap_or((0, 0));
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Predict,
            rows_count,
            columns_count,
            execution_mode: self.execution_mode(),
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            let decisions = sk_linear_svc_decision_matrix(self, features)?;
            Ok(sk_linear_svc_labels_from_decisions(&decisions))
        })
    }

    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let (rows_count, columns_count) = sk_svm_view_dimensions(&features).unwrap_or((0, 0));
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Predict,
            rows_count,
            columns_count,
            execution_mode: self.execution_mode(),
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            // A linear SVC has no calibrated probability model; fabricating a
            // distribution from margins would be misleading.
            Err(SKError::UnsupportedOperation {
                operation: "predict_probabilities",
                reason: "SKLinearSVC has no probability model; compose it with a calibrated \
                         classifier such as SKSVC(probability = true) or a logistic model"
                    .to_owned(),
            })
        })
    }
}
