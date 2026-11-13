//! Format-agnostic state export/import for [`SKLassoModel`].

use sciencekit_common::{SKError, SKExportable, SKFloat, SKImportable, SKModelState};

use super::SKLassoModel;
use crate::coordinate_descent::export_helpers::{
    export_coordinate_descent_state, import_coordinate_descent_core,
};

/// Stable estimator identifier for exported lasso states.
pub const SK_LASSO_ESTIMATOR_NAME: &str = "SKLasso";

impl<F: SKFloat> SKExportable<F> for SKLassoModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        export_coordinate_descent_state(&self.core, "lasso", SK_LASSO_ESTIMATOR_NAME)
    }
}

impl<F: SKFloat> SKImportable<F> for SKLassoModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let core = import_coordinate_descent_core(state, "lasso", SK_LASSO_ESTIMATOR_NAME)?;
        Ok(SKLassoModel::from_core(core))
    }
}
