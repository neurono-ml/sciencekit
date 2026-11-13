//! The embedded OpenCL C sources (spec `opencl-backend`, requirement *OpenCL
//! kernels*).
//!
//! Sources are compiled at runtime on the selected device, so they are written
//! once in a precision-generic form: `SK_REAL` is replaced by `float` or
//! `double` before compilation. Keeping the sources in the library (instead of
//! loading files) makes a deployed binary self-contained and rules out a
//! missing-file failure mode.

use sciencekit_common::SKError;

/// The token an OpenCL C source uses for the scalar type.
pub(crate) const SCALAR_PLACEHOLDER: &str = "SK_REAL";

/// Pairwise distance between the rows of two matrices.
pub(crate) const PAIRWISE_DISTANCE_SOURCE: &str = r#"
__kernel void sk_pairwise_distance(
    __global const SK_REAL* left,
    __global const SK_REAL* right,
    __global SK_REAL* output,
    const unsigned int left_count,
    const unsigned int right_count,
    const unsigned int dimensions,
    const int metric)
{
    const unsigned int row = get_global_id(1);
    const unsigned int column = get_global_id(0);
    if (row >= left_count || column >= right_count) {
        return;
    }
    SK_REAL accumulator = (SK_REAL)0;
    for (unsigned int dimension = 0; dimension < dimensions; ++dimension) {
        const SK_REAL difference =
            left[row * dimensions + dimension] - right[column * dimensions + dimension];
        if (metric == 1) {
            accumulator += fabs(difference);
        } else if (metric == 2) {
            accumulator = fmax(accumulator, fabs(difference));
        } else {
            accumulator += difference * difference;
        }
    }
    if (metric == 3) {
        accumulator = sqrt(accumulator);
    }
    output[row * right_count + column] = accumulator;
}
"#;

/// Tiled dense matrix multiplication with 16 × 16 local-memory tiles.
pub(crate) const GEMM_SOURCE: &str = r#"
#define SK_TILE 16

__kernel void sk_gemm_tiled(
    __global const SK_REAL* left,
    __global const SK_REAL* right,
    __global SK_REAL* output,
    const unsigned int rows,
    const unsigned int shared_dimension,
    const unsigned int columns)
{
    __local SK_REAL left_tile[SK_TILE][SK_TILE];
    __local SK_REAL right_tile[SK_TILE][SK_TILE];

    const int local_row = get_local_id(1);
    const int local_column = get_local_id(0);
    const int global_row = get_group_id(1) * SK_TILE + local_row;
    const int global_column = get_group_id(0) * SK_TILE + local_column;
    const int tiles = (shared_dimension + SK_TILE - 1) / SK_TILE;

    SK_REAL accumulator = (SK_REAL)0;
    for (int tile = 0; tile < tiles; ++tile) {
        const int left_column = tile * SK_TILE + local_column;
        const int right_row = tile * SK_TILE + local_row;
        left_tile[local_row][local_column] =
            (global_row < (int)rows && left_column < (int)shared_dimension)
                ? left[global_row * shared_dimension + left_column]
                : (SK_REAL)0;
        right_tile[local_row][local_column] =
            (right_row < (int)shared_dimension && global_column < (int)columns)
                ? right[right_row * columns + global_column]
                : (SK_REAL)0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int step = 0; step < SK_TILE; ++step) {
            accumulator += left_tile[local_row][step] * right_tile[step][local_column];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (global_row < (int)rows && global_column < (int)columns) {
        output[global_row * columns + global_column] = accumulator;
    }
}
"#;

/// Elementwise affine transformation: `output = input * factor + offset`.
pub(crate) const ELEMENTWISE_TRANSFORM_SOURCE: &str = r#"
__kernel void sk_elementwise_transform(
    __global const SK_REAL* input,
    __global SK_REAL* output,
    const SK_REAL factor,
    const SK_REAL offset,
    const unsigned int element_count)
{
    const unsigned int index = get_global_id(0);
    if (index < element_count) {
        output[index] = input[index] * factor + offset;
    }
}
"#;

/// Elementwise combination: `output = left + right`.
pub(crate) const ELEMENTWISE_COMBINE_SOURCE: &str = r#"
__kernel void sk_elementwise_combine(
    __global const SK_REAL* left,
    __global const SK_REAL* right,
    __global SK_REAL* output,
    const unsigned int element_count)
{
    const unsigned int index = get_global_id(0);
    if (index < element_count) {
        output[index] = left[index] + right[index];
    }
}
"#;

/// Tree-ensemble prediction: one work item walks one row down the tree.
pub(crate) const TREE_PREDICT_SOURCE: &str = r#"
#define SK_TREE_COLUMNS 6
#define SK_MAXIMUM_DEPTH 64

__kernel void sk_tree_predict(
    __global const SK_REAL* features,
    __global const SK_REAL* nodes,
    __global SK_REAL* output,
    const unsigned int rows,
    const unsigned int feature_count,
    const unsigned int node_count)
{
    const unsigned int row = get_global_id(0);
    if (row >= rows) {
        return;
    }
    unsigned int node = 0;
    for (unsigned int depth = 0; depth < SK_MAXIMUM_DEPTH; ++depth) {
        if (node >= node_count) {
            output[row] = (SK_REAL)0;
            return;
        }
        const __global SK_REAL* node_row = nodes + node * SK_TREE_COLUMNS;
        if (node_row[5] > (SK_REAL)0) {
            output[row] = node_row[4];
            return;
        }
        const unsigned int feature = (unsigned int)node_row[0];
        const SK_REAL value = features[row * feature_count + feature];
        node = (value <= node_row[1]) ? (unsigned int)node_row[2] : (unsigned int)node_row[3];
    }
    output[row] = (SK_REAL)0;
}
"#;

/// The OpenCL C scalar token matching a Rust scalar name.
pub(crate) fn scalar_token(scalar_type: &str) -> Result<&'static str, SKError> {
    match scalar_type {
        "f32" => Ok("float"),
        "f64" => Ok("double"),
        other => Err(SKError::Conversion(format!(
            "the OpenCL backend does not support scalar type `{other}`"
        ))),
    }
}

/// Replace the scalar placeholder in `template` with the OpenCL C token.
pub(crate) fn scalar_source(template: &str, scalar_type: &str) -> Result<String, SKError> {
    Ok(template.replace(SCALAR_PLACEHOLDER, scalar_token(scalar_type)?))
}
