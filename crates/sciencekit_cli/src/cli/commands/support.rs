//! Small shared helpers for command argument resolution.

use std::path::Path;

use crate::config::SKEffectiveConfiguration;
use crate::error::SKCliError;
use crate::io::SKOutputFormat;

/// Resolve the output format.
///
/// A format chosen by a flag, the environment or the workflow file wins; only
/// when no layer has an opinion does the output extension decide.
pub fn sk_output_format(
    configuration: &SKEffectiveConfiguration,
    output_path: &Path,
) -> Result<SKOutputFormat, SKCliError> {
    if configuration.output_format_explicit {
        SKOutputFormat::from_name(&configuration.output_format)
    } else {
        Ok(SKOutputFormat::from_path(output_path))
    }
}

/// Confirm an output artifact's parent directory exists before any artifact is
/// written, so a multi-artifact command never leaves a partial record.
pub fn sk_preflight_output_path(path: &str, label: &str) -> Result<(), SKCliError> {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() && !parent.is_dir() {
            return Err(SKCliError::validation(
                label,
                format!("the output directory `{}` does not exist", parent.display()),
            ));
        }
    }
    Ok(())
}

/// Confirm the build can write prediction or matrix artifacts in `format`.
pub fn sk_ensure_output_format_supported(format: SKOutputFormat) -> Result<(), SKCliError> {
    match format {
        SKOutputFormat::Parquet => {
            #[cfg(feature = "parquet")]
            {
                Ok(())
            }
            #[cfg(not(feature = "parquet"))]
            {
                Err(SKCliError::unsupported(
                    "parquet",
                    "a Parquet output artifact",
                ))
            }
        }
        SKOutputFormat::Csv | SKOutputFormat::Json => Ok(()),
    }
}
