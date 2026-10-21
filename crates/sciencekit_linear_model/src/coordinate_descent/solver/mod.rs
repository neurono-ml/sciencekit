//! Pure dispatcher: the coordinate-descent iteration, its screened outer loop,
//! the per-alpha parallel solve and the gap-safe screening. Carries no
//! implementation logic.

mod all_alphas;
mod gap_safe;
mod iteration;
mod orchestration;

pub(crate) use all_alphas::sk_solve_all_alphas;
pub(crate) use orchestration::sk_solve_alpha_with_initial_active;
