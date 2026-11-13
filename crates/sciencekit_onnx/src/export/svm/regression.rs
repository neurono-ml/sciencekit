//! `SKSVR` export: the kernel dual expansion produces one real prediction.

use sciencekit_common::{SKError, SKFloat, SKModelState};

use crate::graph::SKOnnxGraphBuilder;

use super::super::helpers::floating_tensor;
use super::super::stage::{SKOnnxStageKind, SKOnnxStageOutput};
use super::dispatcher::SK_SVR_NAME;
use super::expansion::{append_expansion, dense_support_vectors, read_bias};
use super::kernel::require_exportable_kernel;

pub(crate) fn append_svr<F: SKFloat>(
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<SKOnnxStageOutput, SKError> {
    require_exportable_kernel(state)?;
    let (shape, support_vectors) = dense_support_vectors(state, "svr", SK_SVR_NAME)?;
    builder.add_floating_initializer(
        format!("{namespace}svr.support_vectors"),
        shape,
        &support_vectors,
    );
    let (_, dual_coefficients) = floating_tensor(state, "svr.dual_coefficients")?;
    let bias = read_bias(state, "svr")?;
    let margin = append_expansion(
        state,
        builder,
        input,
        &format!("{namespace}svr.support_vectors"),
        &dual_coefficients,
        bias,
        namespace,
    )?;
    Ok(SKOnnxStageOutput {
        tensor: margin,
        width: 1,
        kind: SKOnnxStageKind::Regression,
    })
}
