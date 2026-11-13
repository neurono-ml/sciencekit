//! Dry-run rendering and path validation.

use std::path::Path;

use crate::config::effective::SKEffectiveConfiguration;
use crate::error::SKCliError;

/// Render the effective configuration as pretty JSON for `--dry-run`.
pub fn sk_render_dry_run(configuration: &SKEffectiveConfiguration) -> Result<String, SKCliError> {
    serde_json::to_string_pretty(configuration)
        .map_err(|error| SKCliError::runtime(format!("configuration serialization: {error}")))
}

/// Validate the paths a resolved configuration names, without touching outputs.
pub fn sk_validate_dry_run_paths(
    configuration: &SKEffectiveConfiguration,
) -> Result<(), SKCliError> {
    if let Some(dataset) = &configuration.dataset {
        if !Path::new(dataset).exists() {
            return Err(SKCliError::validation(
                "configuration.dataset",
                format!("the dataset `{dataset}` does not exist"),
            ));
        }
    }
    for (path, label) in [
        (configuration.model.as_deref(), "configuration.model"),
        (
            configuration.predictions.as_deref(),
            "configuration.predictions",
        ),
    ] {
        if let Some(path) = path {
            if let Some(parent) = Path::new(path).parent() {
                if !parent.as_os_str().is_empty() && !parent.exists() {
                    return Err(SKCliError::validation(
                        label,
                        format!("the output directory `{}` does not exist", parent.display()),
                    ));
                }
            }
        }
    }
    Ok(())
}
