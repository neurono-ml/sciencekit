/*
 * sciencekit — stable C ABI (spec `ffi-abi`).
 *
 * Version: 1.0.0
 *
 * # Ownership rules
 *
 * - The descriptor struct pointer itself must be aligned for `SKFfiArray` and
 *   point to a readable struct. `data` must be aligned for the declared dtype;
 *   `shape` and `strides` must be aligned for `int64_t`; `len` values must be
 *   readable. A misaligned
 *   pointer is rejected with SK_FFI_STATUS_INVALID_ARGUMENT before any element
 *   is read.
 * - An array descriptor passed TO the library is borrowed for the duration of
 *   the call. The caller owns the buffer and may free it when the call
 *   returns. Input descriptors are never released with `sk_ffi_array_free`.
 * - An array descriptor RETURNED by the library owns its shape and data
 *   buffers. Release it exactly once with `sk_ffi_array_free`; a second call
 *   is a safe no-op. An output slot reused for another call must be released
 *   (or zero-initialised) first, or the previous buffers leak.
 * - Builders and models are opaque handles. Free each one once with its
 *   matching free function. Passing a freed handle to any function returns
 *   `SK_FFI_STATUS_INVALID_HANDLE` without dereferencing memory.
 * - `sk_ffi_last_error` returns a thread-local string valid until the next ABI
 *   call on that thread. Copy it before moving to another thread.
 *
 * # Thread safety
 *
 * Fitted model handles are immutable and safe to use from several threads at
 * once. Builder handles are mutable; serialize calls on one builder.
 */

#ifndef SCIENCEKIT_H
#define SCIENCEKIT_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles: never inspect or dereference these in the binding. */
typedef struct SKFfiBuilder SKFfiBuilder;
typedef struct SKFfiModel SKFfiModel;

/* Stable status codes. Never renumber them. */
typedef enum SKFfiStatus {
    SK_FFI_STATUS_OK = 0,
    SK_FFI_STATUS_INVALID_HANDLE = 1,
    SK_FFI_STATUS_SHAPE = 2,
    SK_FFI_STATUS_INVALID_ARGUMENT = 3,
    SK_FFI_STATUS_UNSUPPORTED = 4,
    SK_FFI_STATUS_NOT_FITTED = 5,
    SK_FFI_STATUS_IO = 6,
    SK_FFI_STATUS_INTERNAL = 7
} SKFfiStatus;

/* Element data types carried by a descriptor. */
typedef enum SKFfiDataType {
    SK_FFI_DTYPE_F32 = 0,
    SK_FFI_DTYPE_F64 = 1,
    SK_FFI_DTYPE_I64 = 2,
    SK_FFI_DTYPE_U8 = 3
} SKFfiDataType;

/* Estimator families accepted by `sk_ffi_builder_create`. */
typedef enum SKFfiEstimatorFamily {
    SK_FFI_ESTIMATOR_LINEAR_REGRESSION = 0,
    SK_FFI_ESTIMATOR_STANDARD_SCALER = 1
} SKFfiEstimatorFamily;

/* Math backend kinds accepted by `sk_ffi_builder_set_backend`. */
typedef enum SKFfiBackendKind {
    SK_FFI_BACKEND_FAER = 0,
    SK_FFI_BACKEND_MATRIX_MULTIPLY = 1,
    SK_FFI_BACKEND_NDARRAY_LINALG = 2
} SKFfiBackendKind;

/* Feature bits returned by `sk_ffi_features`. */
#define SK_FFI_FEATURE_LINEAR_REGRESSION (UINT64_C(1) << 0)
#define SK_FFI_FEATURE_STANDARD_SCALER (UINT64_C(1) << 1)
#define SK_FFI_FEATURE_MODEL_PERSISTENCE (UINT64_C(1) << 2)
#define SK_FFI_FEATURE_F32 (UINT64_C(1) << 3)
#define SK_FFI_FEATURE_F64 (UINT64_C(1) << 4)

/*
 * A borrowed or owned array descriptor.
 *
 * dtype:   one of SKFfiDataType.
 * ndim:    number of dimensions; every operation accepts 1 or 2.
 * shape:   `ndim` positive dimensions, never NULL.
 * strides: element strides, or NULL for C-contiguous (row-major) data.
 *          Non-NULL strides must describe the same row-major layout.
 * data:    the first element, never NULL.
 * len:     total element count; it must equal the product of `shape`.
 */
typedef struct SKFfiArray {
    int32_t dtype;
    int32_t ndim;
    const int64_t *shape;
    const int64_t *strides;
    const void *data;
    int64_t len;
} SKFfiArray;

/* Version and capabilities. */
SKFfiStatus sk_ffi_version(int32_t *major, int32_t *minor, int32_t *patch);
uint64_t sk_ffi_features(void);
const char *sk_ffi_status_name(int32_t status);
const char *sk_ffi_last_error(void);

/* Builder lifecycle and configuration. */
SKFfiBuilder *sk_ffi_builder_create(int32_t estimator_family, int32_t dtype);
SKFfiStatus sk_ffi_builder_set_include_intercept(SKFfiBuilder *builder, bool enabled);
SKFfiStatus sk_ffi_builder_set_with_mean(SKFfiBuilder *builder, bool enabled);
SKFfiStatus sk_ffi_builder_set_with_std(SKFfiBuilder *builder, bool enabled);
SKFfiStatus sk_ffi_builder_set_backend(SKFfiBuilder *builder, int32_t backend);
SKFfiStatus sk_ffi_builder_free(SKFfiBuilder *builder);

/* Fitting and inference. */
SKFfiModel *sk_ffi_fit(SKFfiBuilder *builder,
                       const SKFfiArray *features,
                       const SKFfiArray *targets);
SKFfiStatus sk_ffi_model_free(SKFfiModel *model);
SKFfiStatus sk_ffi_predict(SKFfiModel *model,
                           const SKFfiArray *features,
                           SKFfiArray *output);
SKFfiStatus sk_ffi_transform(SKFfiModel *model,
                             const SKFfiArray *features,
                             SKFfiArray *output);
SKFfiStatus sk_ffi_score(SKFfiModel *model,
                         const SKFfiArray *features,
                         const SKFfiArray *targets,
                         double *score);

/*
 * Persistence through the interop model-state contract.
 *
 * Scope: save/load currently support fitted linear-regression models in f64.
 * Saving an f32 model or a fitted scaler returns SK_FFI_STATUS_UNSUPPORTED;
 * loading a state exported by another estimator is rejected the same way.
 */
SKFfiStatus sk_ffi_model_save(SKFfiModel *model, const char *path);
SKFfiStatus sk_ffi_model_load(const char *path, SKFfiModel **output);

/* Release an output descriptor returned by predict or transform. */
SKFfiStatus sk_ffi_array_free(SKFfiArray *array);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SCIENCEKIT_H */
