package sciencekit

import (
	"errors"
	"fmt"
)

// Status is a stable sciencekit ABI status code.
type Status int32

// The status codes the C ABI defines, plus StatusUnknown for failures of the
// pointer-returning entry points (fit and model_load) that report their reason
// only through the thread-local message.
const (
	StatusUnknown         Status = -1
	StatusOK              Status = 0
	StatusInvalidHandle   Status = 1
	StatusShape           Status = 2
	StatusInvalidArgument Status = 3
	StatusUnsupported     Status = 4
	StatusNotFitted       Status = 5
	StatusIO              Status = 6
	StatusInternal        Status = 7
)

// Name returns the stable lowercase name of the status, as reported by the
// native library.
func (status Status) Name() string {
	switch status {
	case StatusUnknown:
		return "unknown"
	default:
		return nativeStatusName(int32(status))
	}
}

func (status Status) String() string {
	return status.Name()
}

// ScienceKitError is an error raised by the sciencekit Go binding. It carries
// the ABI status code, the operation that failed and the message the native
// library recorded.
type ScienceKitError struct {
	// Operation is the binding method that failed.
	Operation string
	// Code is the ABI status code.
	Code Status
	// Message is the native diagnostic, copied out of the thread-local slot.
	Message string
}

// Error implements the error interface.
func (errorValue *ScienceKitError) Error() string {
	name := errorValue.Code.Name()
	if errorValue.Message == "" {
		return fmt.Sprintf("sciencekit %s (%d): %s failed", name, int32(errorValue.Code), errorValue.Operation)
	}
	return fmt.Sprintf("sciencekit %s (%d): %s: %s", name, int32(errorValue.Code), errorValue.Operation, errorValue.Message)
}

// IsStatus reports whether err is a ScienceKitError with the given code.
func IsStatus(err error, status Status) bool {
	var failure *ScienceKitError
	if !errors.As(err, &failure) {
		return false
	}
	return failure.Code == status
}

// newError builds a ScienceKitError for one failed operation.
func newError(status Status, operation, message string) error {
	return &ScienceKitError{Operation: operation, Code: status, Message: message}
}

// closedError reports a call on a handle the caller already closed.
func closedError(operation string) error {
	return newError(StatusInvalidHandle, operation, "the handle is closed")
}

// allocationError reports a failed C-heap allocation in the binding.
func allocationError(operation string) error {
	return newError(StatusInternal, operation, "cannot allocate the C buffer")
}

// validationError reports a Go-side shape failure, raised before any native
// call.
func validationError(operation, message string) error {
	return newError(StatusInvalidArgument, operation, message)
}
