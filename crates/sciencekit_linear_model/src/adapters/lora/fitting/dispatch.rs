//! The public adapter fit and loss entry points, dispatching on the target
//! module exactly once per operation.

use sciencekit_common::{SKDataView, SKError, SKFloat, SKTargetView};

use super::linear::sk_linear_loss;
use super::logistic::{sk_integer_labels, sk_logistic_loss};
use super::validation::{sk_validate_finite_features, sk_validate_finite_targets};
use crate::adapters::lora::attached_model::SKLoraAttachedModel;
use crate::adapters::lora::config::SKLoraTargetModule;
use crate::adapters::lora::prediction::sk_feature_dimensions;

impl<F: SKFloat> SKLoraAttachedModel<F> {
    /// Fit the adapter on the supplied data.
    ///
    /// Only `A`/`B` move; the frozen base and its intercept stay untouched. A
    /// regression base consumes continuous targets, a classifier base consumes
    /// canonical integer class indices.
    ///
    /// # Errors
    ///
    /// Returns shape, non-finite, unknown-label and unsupported-family errors.
    pub fn fit<'a, X, T>(&mut self, features: X, targets: T) -> Result<&mut Self, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let targets = targets.try_into()?;
        match self.adapter().target() {
            SKLoraTargetModule::LinearCoefficients => self.fit_linear(&data, &targets)?,
            SKLoraTargetModule::LogisticCoefficients => self.fit_logistic(&data, &targets)?,
        }
        self.mark_fitted();
        Ok(self)
    }

    /// The mean training loss of the adapter at its current factors.
    ///
    /// Regression reports the mean squared error; classification reports the
    /// mean logistic loss (binary cross-entropy or softmax cross-entropy).
    ///
    /// # Errors
    ///
    /// Returns shape errors when the features or targets disagree with the
    /// base, and [`SKError::UnsupportedRepresentation`] when a classifier
    /// receives continuous targets.
    pub fn training_loss<'a, X, T>(&self, features: X, targets: T) -> Result<F, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let targets = targets.try_into()?;
        let (rows_count, columns_count) = sk_feature_dimensions(&data);
        if rows_count == 0 || columns_count == 0 {
            return Err(SKError::ShapeMismatch {
                expected: vec![rows_count.max(1), self.feature_count().max(1)],
                found: vec![rows_count, columns_count],
            });
        }
        if columns_count != self.feature_count() {
            return Err(SKError::shape_mismatch_2d(
                rows_count,
                self.feature_count(),
                rows_count,
                columns_count,
            ));
        }
        sk_validate_finite_features(&data)?;
        match self.adapter().target() {
            SKLoraTargetModule::LinearCoefficients => {
                let continuous = targets.as_continuous()?;
                if continuous.len() != rows_count {
                    return Err(SKError::ShapeMismatch {
                        expected: vec![rows_count],
                        found: vec![continuous.len()],
                    });
                }
                sk_validate_finite_targets(continuous.view())?;
                sk_linear_loss(
                    &data,
                    self.effective_coefficients()?.view(),
                    self.base().intercepts()?[0],
                    continuous.view(),
                )
            }
            SKLoraTargetModule::LogisticCoefficients => {
                let labels = sk_integer_labels(&targets)?;
                if labels.len() != rows_count {
                    return Err(SKError::ShapeMismatch {
                        expected: vec![rows_count],
                        found: vec![labels.len()],
                    });
                }
                sk_logistic_loss(self, &data, labels)
            }
        }
    }
}
