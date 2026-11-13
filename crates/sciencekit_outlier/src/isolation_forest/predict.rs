//! Prediction, decision function, normality scoring and `fit_predict` for `SKIsolationForest`.

use ndarray::Array1;
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKOutlierLabel,
    SKOutlierPredictor, sk_run_operation,
};

use super::core_implementation::SKIsolationForest;
use super::fit::sk_fit_isolation_forest;
use super::model::SKIsolationForestModel;
use super::score::sk_compute_anomaly_scores;
use super::validation::{sk_validate_query_features, sk_view_dimensions};

impl<F: SKFloat> SKIsolationForest<F> {
    /// Fits the Isolation Forest on `features` and predicts typed outlier labels in one call.
    pub fn fit_predict<'a, X>(&self, features: X) -> Result<Array1<SKOutlierLabel>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let (rows_count, columns_count) = sk_view_dimensions(&features);
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::FitPredict,
            rows_count,
            columns_count,
            execution_mode: self.execution_mode(),
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            let model = sk_fit_isolation_forest(self, &features, "fit_predict")?;
            model.predict_unchecked(&features)
        })
    }
}

impl<F: SKFloat> SKIsolationForestModel<F> {
    /// Liu-Ting-Zhou anomaly score `2^(-E[h(x)] / c(n))` in `[0, 1]` per row (larger = more anomalous).
    pub fn anomaly_scores<'a, X>(&self, features: X) -> Result<Array1<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        self.observe_query(
            SKOperationKind::Score,
            &features,
            "anomaly_scores",
            |view| {
                let anomalies = sk_compute_anomaly_scores(
                    &self.trees,
                    self.subsample_size,
                    view,
                    self.execution_mode,
                )?;
                Ok(Array1::from_vec(anomalies))
            },
        )
    }

    /// Normality score (`-anomaly_score`) per row, where larger values indicate more normal points.
    pub fn score_samples<'a, X>(&self, features: X) -> Result<Array1<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        self.observe_query(SKOperationKind::Score, &features, "score_samples", |view| {
            self.score_samples_unchecked(view)
        })
    }

    /// Shifted normality score (`score_samples - offset`); negative values indicate outliers.
    pub fn decision_function<'a, X>(&self, features: X) -> Result<Array1<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        self.observe_query(
            SKOperationKind::Predict,
            &features,
            "decision_function",
            |view| self.decision_function_unchecked(view),
        )
    }

    /// Predicts one typed [`SKOutlierLabel`] per input row.
    pub fn predict<'a, X>(&self, features: X) -> Result<Array1<SKOutlierLabel>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        self.observe_query(SKOperationKind::Predict, &features, "predict", |view| {
            self.predict_unchecked(view)
        })
    }

    fn observe_query<T>(
        &self,
        operation_kind: SKOperationKind,
        features: &SKDataView<'_, F>,
        operation_name: &'static str,
        compute: impl FnOnce(&SKDataView<'_, F>) -> Result<T, SKError>,
    ) -> Result<T, SKError> {
        let (rows_count, columns_count) = sk_view_dimensions(features);
        let attributes = SKOperationAttributes {
            operation: operation_kind,
            rows_count,
            columns_count,
            execution_mode: self.execution_mode,
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            sk_validate_query_features(features, self.feature_count, operation_name)?;
            compute(features)
        })
    }

    fn score_samples_unchecked(&self, features: &SKDataView<'_, F>) -> Result<Array1<F>, SKError> {
        let anomalies = sk_compute_anomaly_scores(
            &self.trees,
            self.subsample_size,
            features,
            self.execution_mode,
        )?;
        Ok(Array1::from_iter(anomalies.into_iter().map(|score| -score)))
    }

    fn decision_function_unchecked(
        &self,
        features: &SKDataView<'_, F>,
    ) -> Result<Array1<F>, SKError> {
        let offset = self.offset;
        let scores = self.score_samples_unchecked(features)?;
        Ok(scores.mapv(|score| score - offset))
    }

    fn predict_unchecked(
        &self,
        features: &SKDataView<'_, F>,
    ) -> Result<Array1<SKOutlierLabel>, SKError> {
        let decisions = self.decision_function_unchecked(features)?;
        Ok(decisions.mapv(|decision| {
            if decision < F::zero() {
                SKOutlierLabel::Outlier
            } else {
                SKOutlierLabel::Inlier
            }
        }))
    }
}

impl<F: SKFloat> SKOutlierPredictor<F> for SKIsolationForestModel<F> {
    type Error = SKError;

    fn offset(&self) -> F {
        self.offset
    }

    fn score_samples<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        SKIsolationForestModel::score_samples(self, features)
    }

    fn decision_function<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        SKIsolationForestModel::decision_function(self, features)
    }

    fn predict<'a, X>(&self, features: X) -> Result<Array1<SKOutlierLabel>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        SKIsolationForestModel::predict(self, features)
    }
}
