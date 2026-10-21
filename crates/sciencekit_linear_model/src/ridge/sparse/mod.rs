//! Pure dispatcher: the sparse ridge path (the augmented normal-equation
//! operator, the conjugate-gradient solver and its resolution). Carries no
//! implementation logic.

mod cg;
mod normal_equation_operator;
mod solve;

pub(super) use normal_equation_operator::{
    sk_csr_column_sums, sk_csr_transpose_vector_product, sk_sparse_gram,
};
pub(super) use solve::solve_sparse_ridge;
