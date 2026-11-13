//! Fit-time input and resource-bound validation for successive halving.

use sciencekit_common::{SKError, SKFloat};

use crate::grid_search::observation::sk_view_dimensions;
use crate::split::validation::{sk_validate_finite_features, sk_validate_finite_targets};

use super::SKHalvingEvaluationRequest;

pub(super) fn sk_validate_fit_inputs<F: SKFloat, E>(
    request: &SKHalvingEvaluationRequest<'_, '_, F, E>,
) -> Result<usize, SKError> {
    if request.candidates.is_empty() {
        return Err(SKError::InvalidHyperparameter {
            name: "candidates",
            reason: "at least one configured candidate is required".to_owned(),
        });
    }
    let (total_rows, _) = sk_view_dimensions(request.features);
    if total_rows == 0 {
        return Err(SKError::EmptyInput {
            operation: "halving_grid_search",
        });
    }
    if request.targets.len() != total_rows {
        return Err(SKError::ShapeMismatch {
            expected: vec![total_rows],
            found: vec![request.targets.len()],
        });
    }
    sk_validate_finite_features(&request.features)?;
    sk_validate_finite_targets(Some(request.targets))?;

    if let Some(group_values) = request.groups {
        if group_values.len() != total_rows {
            return Err(SKError::ShapeMismatch {
                expected: vec![total_rows],
                found: vec![group_values.len()],
            });
        }
    }

    let effective_maximum_resource = match request.maximum_resource {
        Some(bound) if bound == 0 || bound > total_rows => {
            return Err(SKError::InvalidHyperparameter {
                name: "maximum_resource",
                reason: format!(
                    "maximum_resource ({bound}) must be within 1..={total_rows} for this dataset"
                ),
            });
        }
        Some(bound) => bound,
        None => total_rows,
    };

    if request.minimum_resource == 0 || request.minimum_resource > effective_maximum_resource {
        return Err(SKError::InvalidHyperparameter {
            name: "minimum_resource",
            reason: format!(
                "minimum_resource ({}) must lie within 1..={effective_maximum_resource}",
                request.minimum_resource
            ),
        });
    }
    Ok(effective_maximum_resource)
}
