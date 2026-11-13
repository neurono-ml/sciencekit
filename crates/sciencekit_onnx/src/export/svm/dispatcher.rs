//! The estimator-name dispatcher of the support-vector family.

use sciencekit_common::{SKError, SKFloat, SKModelState};

use crate::graph::SKOnnxGraphBuilder;

use super::super::stage::SKOnnxStageOutput;
use super::{linear, multi_class, regression};

pub(crate) const SK_SVC_NAME: &str = "SKSVC";
pub(crate) const SK_SVR_NAME: &str = "SKSVR";
pub(crate) const SK_LINEAR_SVC_NAME: &str = "SKLinearSVC";

/// Translate one support-vector estimator state, when the name matches.
pub(crate) fn append_svm<F: SKFloat>(
    estimator_name: &str,
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<Option<SKOnnxStageOutput>, SKError> {
    match estimator_name {
        SK_SVR_NAME => regression::append_svr(state, builder, input, namespace).map(Some),
        SK_SVC_NAME => multi_class::append_svc(state, builder, input, namespace).map(Some),
        SK_LINEAR_SVC_NAME => linear::append_linear_svc(state, builder, input, namespace).map(Some),
        _ => Ok(None),
    }
}
