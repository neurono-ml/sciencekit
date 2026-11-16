//! Backend-name resolution from the configuration layer to the runtime kind.

use sciencekit::SKBackendKind;

use crate::error::SKCliError;

/// Resolve the configured backend name into the runtime [`SKBackendKind`].
pub fn sk_parse_backend(name: &str) -> Result<SKBackendKind, SKCliError> {
    match name {
        "faer" => Ok(SKBackendKind::Faer),
        "opencl" => Err(SKCliError::unsupported(
            "opencl",
            "configuration.backend = \"opencl\"",
        )),
        other => Err(SKCliError::validation(
            "configuration.backend",
            format!("`{other}` is not a known backend; this build supports `faer`"),
        )),
    }
}
