//! Configuration for path-based Arrow file sources (spec
//! `arrow-columnar-interop`).

use arrow_schema::Schema;

use crate::missing::SKMissingValuePolicy;
use crate::selection::SKColumnSelection;

/// The options every Arrow file source shares.
#[derive(Debug, Clone)]
pub struct SKArrowFileOptions {
    pub(crate) batch_size: usize,
    pub(crate) selection: Option<SKColumnSelection>,
    pub(crate) policy: SKMissingValuePolicy,
    pub(crate) explicit_schema: Option<Schema>,
    pub(crate) has_csv_header: bool,
    pub(crate) csv_delimiter: u8,
    pub(crate) infer_json_schema: bool,
}

impl Default for SKArrowFileOptions {
    fn default() -> Self {
        SKArrowFileOptions {
            batch_size: 1_024,
            selection: None,
            policy: SKMissingValuePolicy::Reject,
            explicit_schema: None,
            has_csv_header: true,
            csv_delimiter: b',',
            infer_json_schema: false,
        }
    }
}
