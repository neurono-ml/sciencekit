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
