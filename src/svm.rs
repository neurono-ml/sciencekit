//! The `svm` family: support-vector machines.
//!
//! Every item here resolves to `sciencekit_svm`. Its `SKClassWeight` policy and
//! its `SKKernel` re-export are the SVM-owned definitions, distinct from the
//! tree/ensemble and linear-model policies of the same names — the umbrella
//! keeps each inside its family module instead of flattening them.

pub use sciencekit_svm::linear_svc::{
    SKLinearSVC, SKLinearSVCBuilder, SKLinearSVCLoss, sk_linear_svc_loss_from_code,
};
pub use sciencekit_svm::svc::{SKBinarySVCModel, SKClassWeight, SKSVC, SKSVCBuilder, SKSVCModel};
pub use sciencekit_svm::svr::{SKSVR, SKSVRBuilder, SKSVRModel};
pub use sciencekit_svm::{SKKernel, SKKernelGamma, SKKernelParameters};
pub use sciencekit_svm::{kernel, linear_svc, probability, smo, svc, svr};
