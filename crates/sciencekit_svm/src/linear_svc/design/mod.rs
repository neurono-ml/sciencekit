//! Design abstraction for the linear SVC dual coordinate-descent solver
//! (spec `linear-support-vector-classifier`).
//!
//! Pure dispatcher module: implementations live in dedicated files.

mod access;
mod dense;
mod dispatch;
mod sparse;

pub(crate) use access::SKLinearSvcDesignAccess;
pub(crate) use dispatch::{SKLinearSvcDesign, sk_linear_svc_design_access};
