//! Incremental learning for the fitted SGD classifier (spec
//! `stochastic-gradient-classifier`).

use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKRandomGenerator,
    SKTargetView, sk_run_operation,
};

use super::super::core_implementation::sk_dimensions;
use super::core_implementation::SKSGDClassifierModel;
use crate::class_weight::sk_expand_class_weights;
use crate::sgd_classifier::fitting::SKFeatureDesign;
use crate::sgd_classifier::fitting::{sk_reported_parts, sk_run_head_epoch};
use crate::sgd_classifier::targets::{
    sk_canonicalize_against_classes, sk_head_targets, sk_positive_classes,
};

impl<F: SKFloat> SKSGDClassifierModel<F> {
    /// Continue learning from one more batch, continuing the schedule counter
    /// and the averaged iterate.
    ///
    /// Returns the model itself for chaining. An empty batch is a no-op; a
    /// column count differing from the fitted design is rejected with
    /// `ShapeMismatch`, and a label the fit never saw is rejected.
    pub fn partial_fit<'a, X, T>(&mut self, features: X, targets: T) -> Result<&mut Self, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let target = targets.try_into()?;
        let class_indices = sk_canonicalize_against_classes(&self.classes, target)?;
        let (rows_count, columns_count) = sk_dimensions(&data)?;
        if columns_count != self.feature_count() {
            return Err(SKError::ShapeMismatch {
                expected: vec![self.feature_count()],
                found: vec![columns_count],
            });
        }
        if rows_count == 0 {
            return Ok(self);
        }
        if class_indices.len() != rows_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![rows_count],
                found: vec![class_indices.len()],
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
            let design = SKFeatureDesign::from_view(data)?;
            let sample_weights = sk_expand_class_weights(
                &self.class_weight,
                &class_indices,
                self.number_of_classes(),
            )?;
            let positive_classes = sk_positive_classes(self.number_of_classes());
            let mut generator = SKRandomGenerator::new(self.config.random_seed);
            let order = sk_batch_order(self.config.shuffle, rows_count, &mut generator);
            for (head, &positive_class) in positive_classes.iter().enumerate() {
                let head_targets = sk_head_targets(&class_indices, positive_class);
                let head_targets = ndarray::Array1::from_vec(head_targets);
                sk_run_head_epoch(
                    &mut self.states[head],
                    design,
                    head_targets.view(),
                    &sample_weights,
                    self.loss,
                    &self.config,
                    &order,
                )?;
            }
            let (coefficients, intercepts) = sk_reported_parts(&self.states, self.feature_count());
            self.coefficients = coefficients;
            self.intercepts = intercepts;
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
