//! Format-agnostic state export/import for [`SKElasticNetModel`].

use sciencekit_common::{SKError, SKExportable, SKFloat, SKImportable, SKModelState};

use super::SKElasticNetModel;
use crate::coordinate_descent::export_helpers::{
    export_coordinate_descent_state, import_coordinate_descent_core,
};

/// Stable estimator identifier for exported elastic_net states.
pub const SK_ELASTIC_NET_ESTIMATOR_NAME: &str = "SKElasticNet";

impl<F: SKFloat> SKExportable<F> for SKElasticNetModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        export_coordinate_descent_state(&self.core, "elastic_net", SK_ELASTIC_NET_ESTIMATOR_NAME)
    }
}

impl<F: SKFloat> SKImportable<F> for SKElasticNetModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let core =
            import_coordinate_descent_core(state, "elastic_net", SK_ELASTIC_NET_ESTIMATOR_NAME)?;
        Ok(SKElasticNetModel::from_core(core))
    }
}
