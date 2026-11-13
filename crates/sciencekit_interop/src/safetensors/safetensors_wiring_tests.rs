//! Wiring tests for the `sciencekit_interop::safetensors` module (spec `safetensors-format`).

use sciencekit_common::{SKBuilder, SKExecutionMode};

use super::{
    SK_SAFETENSORS_FORMAT_VERSION, SK_SAFETENSORS_PADDING_TENSOR_NAME, SKSafetensorsReaderBuilder,
    SKSafetensorsWriterBuilder, SKShardPlanBuilder,
};

/// The Safetensors format module exposes its constants and implements `SKBuilder` on all builders.
#[test]
fn safetensors_wiring_exposes_format_constants_and_safetensors_dependency() {
    assert_eq!(SK_SAFETENSORS_FORMAT_VERSION, 1);
    assert_eq!(SK_SAFETENSORS_PADDING_TENSOR_NAME, "__padding__");
    assert_eq!(safetensors::Dtype::F64.bitsize(), 64);

    let mut writer_builder = SKSafetensorsWriterBuilder::new();
    SKBuilder::execution_mode(&mut writer_builder, SKExecutionMode::InProcessSynchronous);
    assert_eq!(
        SKBuilder::build(writer_builder).unwrap().execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );

    let mut reader_builder = SKSafetensorsReaderBuilder::new();
    SKBuilder::execution_mode(&mut reader_builder, SKExecutionMode::InProcessSynchronous);
    assert_eq!(
        SKBuilder::build(reader_builder).unwrap().execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );

    let mut shard_builder = SKShardPlanBuilder::new();
    SKBuilder::execution_mode(&mut shard_builder, SKExecutionMode::InProcessSynchronous);
    assert_eq!(
        SKBuilder::build(shard_builder).unwrap().execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );
}
