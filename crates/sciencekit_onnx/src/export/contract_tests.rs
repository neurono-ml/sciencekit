//! Tests for the `SKToOnnx` blanket contract.

use sciencekit_common::{
    SKError, SKExportable, SKFloat, SKModelMetadata, SKModelState, SKTensor, SKTensorDataType,
    SKTensorValues,
};

use super::SKToOnnx;

/// A minimal exportable model whose state names a family with no exporter.
struct SKContractFixture<F: SKFloat> {
    values: Vec<F>,
}

impl<F: SKFloat> SKExportable<F> for SKContractFixture<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let metadata = SKModelMetadata::new(
            "SKContractFixture",
            1,
            SKTensorDataType::F32,
            self.values.len(),
        )?;
        let tensor = SKTensor::new(
            "fixture.values",
            SKTensorDataType::F32,
            vec![self.values.len()],
            SKTensorValues::Floating(self.values.clone()),
        )?;
        SKModelState::new(metadata, vec![tensor])
    }
}

#[test]
fn to_onnx_propagates_the_state_export_error() {
    let fixture = SKContractFixture::<f32> { values: vec![1.0, 2.0] };
    let error = fixture.to_onnx().expect_err("the fixture has no exporter");
    assert!(matches!(
        error,
        SKError::UnsupportedOperation {
            operation: "onnx_export",
            ..
        }
    ));
    let reason = match error {
        SKError::UnsupportedOperation { reason, .. } => reason,
        other => panic!("unexpected error: {other}"),
    };
    assert!(reason.contains("SKContractFixture"));
}
