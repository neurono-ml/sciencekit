package sciencekit

/*
#cgo CFLAGS: -I${SRCDIR}/../crates/sciencekit_ffi/include
#cgo LDFLAGS: -L${SRCDIR}/../target/release -Wl,-rpath,${SRCDIR}/../target/release -lsciencekit_ffi
#include <stdlib.h>
#include "sciencekit.h"
*/
import "C"

import (
	"runtime"
	"unsafe"
)

// fit runs the configured estimator over borrowed descriptors.
//
// sk_ffi_fit reports failures only through the thread-local message, so the
// resulting error carries StatusUnknown.
func (builder *nativeBuilder) fit(features, targets *descriptor) (*nativeModel, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var targetPointer *C.SKFfiArray
	if targets != nil {
		targetPointer = targets.raw
	}
	pointer := C.sk_ffi_fit(builder.pointer, features.raw, targetPointer)
	if pointer == nil {
		return nil, newError(StatusUnknown, "fit", lastErrorMessage())
	}
	registerModel()
	return &nativeModel{pointer: pointer}, nil
}

// predictF64 predicts responses from double-precision features.
func (model *nativeModel) predictF64(features *descriptor) ([]float64, []int64, error) {
	return model.inferF64("predict", func(output *C.SKFfiArray) C.SKFfiStatus {
		return C.sk_ffi_predict(model.pointer, features.raw, output)
	})
}

// predictF32 predicts responses from single-precision features.
func (model *nativeModel) predictF32(features *descriptor) ([]float32, []int64, error) {
	return model.inferF32("predict", func(output *C.SKFfiArray) C.SKFfiStatus {
		return C.sk_ffi_predict(model.pointer, features.raw, output)
	})
}

// transformF64 standardizes double-precision features.
func (model *nativeModel) transformF64(features *descriptor) ([]float64, []int64, error) {
	return model.inferF64("transform", func(output *C.SKFfiArray) C.SKFfiStatus {
		return C.sk_ffi_transform(model.pointer, features.raw, output)
	})
}

// transformF32 standardizes single-precision features.
func (model *nativeModel) transformF32(features *descriptor) ([]float32, []int64, error) {
	return model.inferF32("transform", func(output *C.SKFfiArray) C.SKFfiStatus {
		return C.sk_ffi_transform(model.pointer, features.raw, output)
	})
}

// inferF64 runs one inference call and copies the owned output back to Go.
func (model *nativeModel) inferF64(operation string, invoke func(*C.SKFfiArray) C.SKFfiStatus) ([]float64, []int64, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	output, err := newOutputDescriptor()
	if err != nil {
		return nil, nil, newError(StatusInternal, operation, err.Error())
	}
	defer C.free(unsafe.Pointer(output))
	if err := statusError(operation, invoke(output)); err != nil {
		return nil, nil, err
	}
	values := copyF64Output(output)
	shape := copyShape(output)
	if err := statusError(operation, C.sk_ffi_array_free(output)); err != nil {
		return nil, nil, err
	}
	return values, shape, nil
}

// inferF32 runs one inference call and copies the owned output back to Go.
func (model *nativeModel) inferF32(operation string, invoke func(*C.SKFfiArray) C.SKFfiStatus) ([]float32, []int64, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	output, err := newOutputDescriptor()
	if err != nil {
		return nil, nil, newError(StatusInternal, operation, err.Error())
	}
	defer C.free(unsafe.Pointer(output))
	if err := statusError(operation, invoke(output)); err != nil {
		return nil, nil, err
	}
	values := copyF32Output(output)
	shape := copyShape(output)
	if err := statusError(operation, C.sk_ffi_array_free(output)); err != nil {
		return nil, nil, err
	}
	return values, shape, nil
}

// score computes the coefficient of determination.
func (model *nativeModel) score(features, targets *descriptor) (float64, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var value C.double
	status := C.sk_ffi_score(model.pointer, features.raw, targets.raw, &value)
	if err := statusError("score", status); err != nil {
		return 0, err
	}
	return float64(value), nil
}

// save writes the model state at path.
func (model *nativeModel) save(path string) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	cPath, release := cString(path)
	defer release()
	return statusError("model_save", C.sk_ffi_model_save(model.pointer, cPath))
}

// loadNativeModel reads a model state from path.
func loadNativeModel(path string) (*nativeModel, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	cPath, release := cString(path)
	defer release()

	// The out-pointer lives in C memory: C writes a C pointer into it, which
	// Go memory may not hold across a cgo boundary.
	slot := (**C.SKFfiModel)(C.malloc(C.size_t(unsafe.Sizeof(uintptr(0)))))
	if slot == nil {
		return nil, newError(StatusInternal, "model_load", "cannot allocate the output slot")
	}
	defer C.free(unsafe.Pointer(slot))
	*slot = nil

	if err := statusError("model_load", C.sk_ffi_model_load(cPath, slot)); err != nil {
		return nil, err
	}
	pointer := *slot
	if pointer == nil {
		return nil, newError(StatusUnknown, "model_load", "the library returned a null model")
	}
	registerModel()
	return &nativeModel{pointer: pointer}, nil
}

// newOutputDescriptor allocates a zeroed output descriptor in C memory.
func newOutputDescriptor() (*C.SKFfiArray, error) {
	output := (*C.SKFfiArray)(C.calloc(1, C.size_t(C.sizeof_SKFfiArray)))
	if output == nil {
		return nil, errOutOfMemory
	}
	return output, nil
}

// copyF64Output copies a native double-precision buffer into a Go slice.
func copyF64Output(array *C.SKFfiArray) []float64 {
	if array.data == nil || array.len <= 0 {
		return []float64{}
	}
	source := unsafe.Slice((*float64)(array.data), int(array.len))
	values := make([]float64, len(source))
	copy(values, source)
	return values
}

// copyF32Output copies a native single-precision buffer into a Go slice.
func copyF32Output(array *C.SKFfiArray) []float32 {
	if array.data == nil || array.len <= 0 {
		return []float32{}
	}
	source := unsafe.Slice((*float32)(array.data), int(array.len))
	values := make([]float32, len(source))
	copy(values, source)
	return values
}

// copyShape copies the descriptor dimensions into a Go slice.
func copyShape(array *C.SKFfiArray) []int64 {
	if array.shape == nil || array.ndim <= 0 {
		return nil
	}
	dimensions := unsafe.Slice((*int64)(unsafe.Pointer(array.shape)), int(array.ndim))
	shape := make([]int64, len(dimensions))
	copy(shape, dimensions)
	return shape
}
