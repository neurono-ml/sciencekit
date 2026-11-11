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

// lastErrorMessage copies the calling thread's last-error message. It must be
// called on the same OS thread as the failing ABI call.
func lastErrorMessage() string {
	return C.GoString(C.sk_ffi_last_error())
}

// nativeVersion reads the ABI version from the loaded library.
func nativeVersion() (int32, int32, int32, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var major, minor, patch C.int32_t
	status := C.sk_ffi_version(&major, &minor, &patch)
	if err := statusError("version", status); err != nil {
		return 0, 0, 0, err
	}
	return int32(major), int32(minor), int32(patch), nil
}

// nativeFeatures reads the capability bitmask.
func nativeFeatures() uint64 {
	return uint64(C.sk_ffi_features())
}

// nativeStatusName resolves a status code to its stable name.
func nativeStatusName(code int32) string {
	return C.GoString(C.sk_ffi_status_name(C.int32_t(code)))
}

// statusError converts a native status into a Go error, copying the ABI
// message before another call can overwrite it.
func statusError(operation string, status C.SKFfiStatus) error {
	if status == C.SK_FFI_STATUS_OK {
		return nil
	}
	return newError(Status(status), operation, lastErrorMessage())
}

// cString returns a NUL-terminated copy of path and its release function.
func cString(value string) (*C.char, func()) {
	pointer := C.CString(value)
	return pointer, func() { C.free(unsafe.Pointer(pointer)) }
}

// descriptor is a C-layout array descriptor whose shape and data buffers are
// copied into C-owned memory, so no Go pointer is ever stored in non-Go
// memory. Release it with freeDescriptor after the call.
type descriptor struct {
	raw         *C.SKFfiArray
	shape       []int64
	shapeMemory *C.int64_t
	dataMemory  unsafe.Pointer
}

// newDescriptor copies the shape and the values into C-owned buffers and
// builds a row-major descriptor over them.
func newDescriptor[T Float](dtype int32, shape []int64, values []T) (*descriptor, error) {
	raw := (*C.SKFfiArray)(C.calloc(1, C.size_t(C.sizeof_SKFfiArray)))
	if raw == nil {
		return nil, errOutOfMemory
	}

	shapeMemory := (*C.int64_t)(C.malloc(C.size_t(len(shape)) * C.size_t(unsafe.Sizeof(C.int64_t(0)))))
	if shapeMemory == nil {
		C.free(unsafe.Pointer(raw))
		return nil, errOutOfMemory
	}
	copy(unsafe.Slice((*int64)(unsafe.Pointer(shapeMemory)), len(shape)), shape)

	dataSize := unsafe.Sizeof(values[0]) * uintptr(len(values))
	dataMemory := C.malloc(C.size_t(dataSize))
	if dataMemory == nil {
		C.free(unsafe.Pointer(shapeMemory))
		C.free(unsafe.Pointer(raw))
		return nil, errOutOfMemory
	}
	copy(unsafe.Slice((*T)(dataMemory), len(values)), values)

	raw.dtype = C.int32_t(dtype)
	raw.ndim = C.int32_t(len(shape))
	raw.shape = shapeMemory
	raw.strides = nil
	raw.data = dataMemory
	product := int64(1)
	for _, dimension := range shape {
		product *= dimension
	}
	raw.len = C.int64_t(product)
	return &descriptor{raw: raw, shape: shape, shapeMemory: shapeMemory, dataMemory: dataMemory}, nil
}

// freeDescriptor releases the C-owned shape, data and descriptor buffers.
// Native-owned output buffers are freed through sk_ffi_array_free instead.
func (descriptor *descriptor) freeDescriptor() {
	if descriptor.raw == nil {
		return
	}
	C.free(unsafe.Pointer(descriptor.shapeMemory))
	C.free(descriptor.dataMemory)
	C.free(unsafe.Pointer(descriptor.raw))
	descriptor.raw = nil
	descriptor.shapeMemory = nil
	descriptor.dataMemory = nil
	descriptor.shape = nil
}

// nativeBuilder wraps one opaque SKFfiBuilder handle.
type nativeBuilder struct {
	pointer *C.SKFfiBuilder
}

// newNativeBuilder creates a configured-with-defaults builder handle.
func newNativeBuilder(estimatorFamily, dtype int32) (*nativeBuilder, error) {
	pointer := C.sk_ffi_builder_create(C.int32_t(estimatorFamily), C.int32_t(dtype))
	if pointer == nil {
		return nil, newError(StatusUnknown, "builder_create", "the library returned a null builder")
	}
	registerBuilder()
	return &nativeBuilder{pointer: pointer}, nil
}

// free releases the native builder.
func (builder *nativeBuilder) free() error {
	if builder.pointer == nil {
		return nil
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status := C.sk_ffi_builder_free(builder.pointer)
	builder.pointer = nil
	unregisterBuilder()
	return statusError("builder_free", status)
}

// callStatus runs one status-returning native call on a locked OS thread.
func (builder *nativeBuilder) callStatus(operation string, call func() C.SKFfiStatus) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	return statusError(operation, call())
}

// setIncludeIntercept configures the linear-regression intercept.
func (builder *nativeBuilder) setIncludeIntercept(enabled bool) error {
	return builder.callStatus("builder_set_include_intercept", func() C.SKFfiStatus {
		return C.sk_ffi_builder_set_include_intercept(builder.pointer, C.bool(enabled))
	})
}

// setWithMean configures scaler centering.
func (builder *nativeBuilder) setWithMean(enabled bool) error {
	return builder.callStatus("builder_set_with_mean", func() C.SKFfiStatus {
		return C.sk_ffi_builder_set_with_mean(builder.pointer, C.bool(enabled))
	})
}

// setWithStd configures scaler scaling.
func (builder *nativeBuilder) setWithStd(enabled bool) error {
	return builder.callStatus("builder_set_with_std", func() C.SKFfiStatus {
		return C.sk_ffi_builder_set_with_std(builder.pointer, C.bool(enabled))
	})
}

// setBackend configures the math backend.
func (builder *nativeBuilder) setBackend(kind BackendKind) error {
	return builder.callStatus("builder_set_backend", func() C.SKFfiStatus {
		return C.sk_ffi_builder_set_backend(builder.pointer, C.int32_t(kind))
	})
}

// nativeModel wraps one opaque SKFfiModel handle.
type nativeModel struct {
	pointer *C.SKFfiModel
}

// free releases the native model.
func (model *nativeModel) free() error {
	if model.pointer == nil {
		return nil
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status := C.sk_ffi_model_free(model.pointer)
	model.pointer = nil
	unregisterModel()
	return statusError("model_free", status)
}
