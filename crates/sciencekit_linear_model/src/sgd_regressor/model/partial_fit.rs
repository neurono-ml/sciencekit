//! Incremental learning for the fitted SGD regressor (spec
//! `stochastic-gradient-regressor`).

use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKRandomGenerator,
    SKTargetView, sk_run_operation,
};

use super::core_implementation::SKSGDRegressorModel;
use super::prediction::dimensions;
use crate::sgd_regressor::fitting::{SKRegressorDesign, sk_reported_parts, sk_run_design_epoch};
use crate::sgd_regressor::targets::sk_loss_adapter;

impl<F: SKFloat> SKSGDRegressorModel<F> {
    /// Continue learning from one more batch, continuing the schedule counter
    /// and the averaged iterate.
    ///
    /// Returns the model itself for chaining. An empty batch is a no-op; a
    /// column or target count differing from the fitted design is rejected with
    /// `ShapeMismatch`, and nominal targets are rejected.
    pub fn partial_fit<'a, X, T>(&mut self, features: X, targets: T) -> Result<&mut Self, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let target = targets.try_into()?.as_continuous()?.into_owned();
        let (rows_count, columns_count) = dimensions(&data);
        if columns_count != self.feature_count() {
            return Err(SKError::ShapeMismatch {
                expected: vec![self.feature_count()],
                found: vec![columns_count],
            });
        }
        if rows_count == 0 {
            return Ok(self);
        }
        if target.len() != rows_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![rows_count],
                found: vec![target.len()],
            });
        }
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::PartialFit,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || -> Result<(), SKError> {
            let design = SKRegressorDesign::from_view(data)?;
            let loss = sk_loss_adapter(self.loss, self.epsilon);
            let sample_weights = vec![F::one(); rows_count];
            let mut generator = SKRandomGenerator::new(self.config.random_seed);
            let order = sk_batch_order(self.config.shuffle, rows_count, &mut generator);
            sk_run_design_epoch(
                &mut self.state,
                design,
                target.view(),
                &sample_weights,
                &loss,
                &self.config,
                &order,
            )?;
            let (coefficients, intercept) = sk_reported_parts(&self.state, self.feature_count());
            self.coefficients = coefficients;
            self.intercept = intercept;
            self.epochs_taken += 1;
            self.converged = false;
            Ok(())
        })?;
        Ok(self)
    }
}

/// Shuffle within the increment batch only.
fn sk_batch_order(shuffle: bool, rows: usize, generator: &mut SKRandomGenerator) -> Vec<usize> {
    if shuffle {
        generator.shuffled_indices(rows)
    } else {
        (0..rows).collect()
    }
}
