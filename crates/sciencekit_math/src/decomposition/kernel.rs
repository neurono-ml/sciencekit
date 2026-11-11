//! The randomized range finder kernel: projection, power iterations and the
//! final small decomposition.

use ndarray::ArrayView1;
use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::operator::SKMatrixOperator;
use super::projection::sk_seeded_projection;
use super::result::{SKRandomizedSvdResult, sk_apply_sign_convention};
use crate::backend::{SKBackendSelection, SKMathBackend};

/// Rejects an invalid configuration before any computation happens.
pub(crate) fn sk_validate_parameters<F: SKFloat>(
    rows: usize,
    columns: usize,
    number_of_components: usize,
    number_of_power_iterations: usize,
    oversampling: usize,
    centering_length: Option<usize>,
) -> Result<(), SKError> {
    if rows == 0 || columns == 0 {
        return Err(SKError::EmptyInput {
            operation: "randomized_svd",
        });
    }
    if number_of_components == 0 {
        return Err(SKError::InvalidHyperparameter {
            name: "number_of_components",
            reason: "the number of components must be at least 1".into(),
        });
    }
    if number_of_components > rows.min(columns) {
        return Err(SKError::InvalidHyperparameter {
            name: "number_of_components",
            reason: format!(
                "the number of components must not exceed min(rows, columns) = {}",
                rows.min(columns)
            ),
        });
    }
    if number_of_power_iterations == 0 {
        return Err(SKError::InvalidHyperparameter {
            name: "number_of_power_iterations",
            reason: "at least one power iteration is required".into(),
        });
    }
    if oversampling == 0 {
        return Err(SKError::InvalidHyperparameter {
            name: "oversampling",
            reason: "the oversampling parameter must be at least 1".into(),
        });
    }
    if let Some(length) = centering_length {
        if length != columns {
            return Err(SKError::shape_mismatch_2d(columns, 1, length, 1));
        }
    }
    Ok(())
}

/// Runs the randomized range finder and returns the truncated factorization.
///
/// The steps are: draw the seeded test basis `Ω`, project `Y = XΩ`, apply the
/// requested power iterations (`Q = qr(Y)`, `Z = XᵀQ`, `Y = XZ`), orthonormalize
/// once more and decompose the small projected matrix `B = QᵀX`. The left
/// singular vectors are lifted back with `Q` and truncated to the requested
/// number of components.
pub(crate) fn sk_randomized_svd_kernel<F: SKBackendSelection>(
    backend: &dyn SKMathBackend<F>,
    matrix: SKDataView<'_, F>,
    number_of_components: usize,
    number_of_power_iterations: usize,
    oversampling: usize,
    random_seed: u64,
    centering: Option<ArrayView1<'_, F>>,
) -> Result<SKRandomizedSvdResult<F>, SKError> {
    let operator = SKMatrixOperator::new(matrix)?;
    let (rows, columns) = (operator.rows(), operator.columns());
    sk_validate_parameters::<F>(
        rows,
        columns,
        number_of_components,
        number_of_power_iterations,
        oversampling,
        centering.map(|vector| vector.len()),
    )?;

    let basis_size = number_of_components
        .saturating_add(oversampling)
        .min(rows.min(columns))
        .max(1);
    let projection = sk_seeded_projection::<F>(columns, basis_size, random_seed);
    let mut projected = operator.multiply(backend, &projection.view());

    for _ in 0..number_of_power_iterations {
        let basis = backend.qr(projected.view()).q;
        let transposed = operator.transpose_multiply(backend, &basis.view());
        projected = operator.multiply(backend, &transposed.view());
    }

    let basis = backend.qr(projected.view()).q;
    let transposed = operator.transpose_multiply(backend, &basis.view());
    let projected_matrix = transposed.t().to_owned();

    let decomposition = backend.svd(projected_matrix.view())?;
    let u = basis.dot(&decomposition.u);
    let v = decomposition.v;
    let components = number_of_components.min(u.ncols()).min(v.ncols());
    let singular_values: Vec<F> = decomposition
        .singular_values
        .iter()
        .copied()
        .take(components)
        .collect();

    let mut u = u.slice(ndarray::s![.., ..components]).to_owned();
    let mut v = v.slice(ndarray::s![.., ..components]).to_owned();
    sk_apply_sign_convention(&mut u, &mut v);

    Ok(SKRandomizedSvdResult {
        u,
        singular_values,
        v,
    })
}
