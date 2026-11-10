/*
 * C smoke test for the sciencekit C ABI (spec `ffi-abi`).
 *
 * Build (from the crate directory, after `cargo build -p sciencekit_ffi`):
 *
 *   cc tests/c_smoke.c -Iinclude \
 *      -L$CARGO_TARGET_DIR/debug -lsciencekit_ffi -o /tmp/sciencekit_c_smoke
 *
 * Run with the shared library on the loader path:
 *
 *   LD_LIBRARY_PATH=$CARGO_TARGET_DIR/debug /tmp/sciencekit_c_smoke
 *
 * The test fits and predicts with a linear model, scores it, saves and loads
 * it, exercises the standard scaler and checks invalid-handle handling.
 */

#include "sciencekit.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s (last error: %s)\n", message,
                sk_ffi_last_error());
        failures += 1;
    }
}

static int close_enough(double left, double right) {
    return fabs(left - right) < 1.0e-9;
}

static SKFfiArray matrix(double *data, int64_t rows, int64_t columns,
                         int64_t *shape) {
    SKFfiArray array;
    array.dtype = SK_FFI_DTYPE_F64;
    array.ndim = 2;
    array.shape = shape;
    array.strides = NULL;
    array.data = data;
    array.len = rows * columns;
    (void)rows;
    (void)columns;
    return array;
}

static SKFfiArray vector(double *data, int64_t length, int64_t *shape) {
    SKFfiArray array;
    array.dtype = SK_FFI_DTYPE_F64;
    array.ndim = 1;
    array.shape = shape;
    array.strides = NULL;
    array.data = data;
    array.len = length;
    return array;
}

static SKFfiArray empty_array(void) {
    SKFfiArray array;
    memset(&array, 0, sizeof(array));
    return array;
}

int main(void) {
    int32_t major = -1, minor = -1, patch = -1;
    check(sk_ffi_version(&major, &minor, &patch) == SK_FFI_STATUS_OK,
          "version query succeeds");
    check(major == 1 && minor == 0 && patch == 0, "version is 1.0.0");
    check((sk_ffi_features() & SK_FFI_FEATURE_LINEAR_REGRESSION) != 0,
          "linear regression is advertised");

    double features_data[8] = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0};
    double target_data[4] = {1.0, 3.0, 4.0, 14.0};
    int64_t feature_shape[2] = {4, 2};
    int64_t target_shape[1] = {4};
    SKFfiArray features = matrix(features_data, 4, 2, feature_shape);
    SKFfiArray targets = vector(target_data, 4, target_shape);

    SKFfiBuilder *builder =
        sk_ffi_builder_create(SK_FFI_ESTIMATOR_LINEAR_REGRESSION,
                              SK_FFI_DTYPE_F64);
    check(builder != NULL, "builder creation succeeds");
    check(sk_ffi_builder_set_include_intercept(builder, true) ==
              SK_FFI_STATUS_OK,
          "include-intercept setter succeeds");
    check(sk_ffi_builder_set_backend(builder, SK_FFI_BACKEND_FAER) ==
              SK_FFI_STATUS_OK,
          "backend setter succeeds");

    SKFfiModel *model = sk_ffi_fit(builder, &features, &targets);
    check(model != NULL, "fit succeeds");

    SKFfiArray predictions = empty_array();
    check(sk_ffi_predict(model, &features, &predictions) == SK_FFI_STATUS_OK,
          "predict succeeds");
    check(predictions.len == 4, "predict returns four values");
    if (predictions.len == 4 && predictions.data != NULL) {
        const double *values = (const double *)predictions.data;
        for (int index = 0; index < 4; ++index) {
            check(close_enough(values[index], target_data[index]),
                  "prediction matches the target");
        }
    }
    check(sk_ffi_array_free(&predictions) == SK_FFI_STATUS_OK,
          "output array release succeeds");

    double score = 0.0;
    check(sk_ffi_score(model, &features, &targets, &score) ==
              SK_FFI_STATUS_OK,
          "score succeeds");
    check(close_enough(score, 1.0), "R2 is 1 on an exactly linear dataset");

    const char *path = "/tmp/sciencekit_c_smoke_model.bin";
    check(sk_ffi_model_save(model, path) == SK_FFI_STATUS_OK, "save succeeds");
    SKFfiModel *loaded = NULL;
    check(sk_ffi_model_load(path, &loaded) == SK_FFI_STATUS_OK,
          "load succeeds");
    check(loaded != NULL, "load returns a handle");
    SKFfiArray restored = empty_array();
    check(sk_ffi_predict(loaded, &features, &restored) == SK_FFI_STATUS_OK,
          "loaded model predicts");
    if (restored.len == 4 && restored.data != NULL) {
        const double *values = (const double *)restored.data;
        for (int index = 0; index < 4; ++index) {
            check(close_enough(values[index], target_data[index]),
                  "loaded prediction matches the target");
        }
    }
    check(sk_ffi_array_free(&restored) == SK_FFI_STATUS_OK,
          "restored array release succeeds");
    check(sk_ffi_model_free(loaded) == SK_FFI_STATUS_OK,
          "loaded model free succeeds");
    remove(path);

    check(sk_ffi_model_free(model) == SK_FFI_STATUS_OK, "model free succeeds");
    check(sk_ffi_model_free(model) == SK_FFI_STATUS_INVALID_HANDLE,
          "double free is rejected");

    SKFfiBuilder *scaler =
        sk_ffi_builder_create(SK_FFI_ESTIMATOR_STANDARD_SCALER,
                              SK_FFI_DTYPE_F64);
    check(scaler != NULL, "scaler builder creation succeeds");
    SKFfiModel *scaler_model = sk_ffi_fit(scaler, &features, NULL);
    check(scaler_model != NULL, "scaler fit succeeds");
    SKFfiArray transformed = empty_array();
    check(sk_ffi_transform(scaler_model, &features, &transformed) ==
              SK_FFI_STATUS_OK,
          "transform succeeds");
    check(transformed.ndim == 2 && transformed.len == 8,
          "transform returns a 4x2 matrix");
    check(sk_ffi_array_free(&transformed) == SK_FFI_STATUS_OK,
          "transform array release succeeds");
    check(sk_ffi_model_free(scaler_model) == SK_FFI_STATUS_OK,
          "scaler model free succeeds");
    check(sk_ffi_builder_free(scaler) == SK_FFI_STATUS_OK,
          "scaler builder free succeeds");

    check(sk_ffi_builder_free(builder) == SK_FFI_STATUS_OK,
          "builder free succeeds");
    check(sk_ffi_builder_free(NULL) == SK_FFI_STATUS_INVALID_HANDLE,
          "null handle is rejected");

    if (failures != 0) {
        fprintf(stderr, "%d C smoke-test check(s) failed\n", failures);
        return 1;
    }
    printf("sciencekit C smoke test passed\n");
    return 0;
}
