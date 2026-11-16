//! Configuration resolution: defaults, workflow file, environment and flags.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

pub(crate) mod backend;
pub(crate) mod defaults;
pub(crate) mod dry_run;
pub(crate) mod effective;
pub(crate) mod environment;
pub(crate) mod precedence;

#[allow(unused_imports)]
pub use backend::sk_parse_backend;
#[allow(unused_imports)]
pub use defaults::{
    SK_DEFAULT_ALPHA, SK_DEFAULT_BACKEND, SK_DEFAULT_ESTIMATOR, SK_DEFAULT_OUTPUT_FORMAT,
    sk_default_metrics,
};
#[allow(unused_imports)]
pub use dry_run::{sk_render_dry_run, sk_validate_dry_run_paths};
#[allow(unused_imports)]
pub use effective::{SKConfigurationOverrides, SKEffectiveConfiguration};
#[allow(unused_imports)]
pub use environment::{sk_environment_overrides, sk_split_metrics};
#[allow(unused_imports)]
pub use precedence::{
    sk_default_configuration, sk_file_overrides, sk_flag_overrides, sk_resolve_configuration,
    sk_validate_configuration,
};

#[cfg(test)]
mod precedence_tests;
