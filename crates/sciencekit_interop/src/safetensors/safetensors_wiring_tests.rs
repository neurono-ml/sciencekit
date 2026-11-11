//! Wiring tests for the `sciencekit_interop::safetensors` module (spec `safetensors-format`).

use super::{SK_SAFETENSORS_FORMAT_VERSION, SK_SAFETENSORS_PADDING_TENSOR_NAME};

/// The Safetensors format module exposes its format version and reserved padding tensor name.
#[test]
fn safetensors_wiring_exposes_format_constants_and_safetensors_dependency() {
    assert_eq!(SK_SAFETENSORS_FORMAT_VERSION, 1);
    assert_eq!(SK_SAFETENSORS_PADDING_TENSOR_NAME, "__padding__");
    assert_eq!(safetensors::Dtype::F64.bitsize(), 64);
}
