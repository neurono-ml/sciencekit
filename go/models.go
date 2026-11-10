package sciencekit

import (
	"runtime"
	"sync"
)

// modelHandle guards an immutable fitted model. Reads take the shared lock so
// several goroutines may predict at once; Close takes it exclusively.
type modelHandle struct {
	mu     sync.RWMutex
	native *nativeModel
	closed bool
}

// acquire locks the model for one operation and returns the release function.
func (handle *modelHandle) acquire(operation string) (*nativeModel, func(), error) {
	handle.mu.RLock()
	if handle.closed {
		handle.mu.RUnlock()
		return nil, nil, closedError(operation)
	}
	return handle.native, handle.mu.RUnlock, nil
}

// close releases the native model exactly once.
func (handle *modelHandle) close() error {
	handle.mu.Lock()
	defer handle.mu.Unlock()
	if handle.closed {
		return nil
	}
	handle.closed = true
	native := handle.native
	handle.native = nil
	return native.free()
}

// LinearRegressionModel is a fitted ordinary-least-squares model.
type LinearRegressionModel[T Float] struct {
	handle modelHandle
}

func newLinearRegressionModel[T Float](native *nativeModel) *LinearRegressionModel[T] {
	model := &LinearRegressionModel[T]{handle: modelHandle{native: native}}
	runtime.SetFinalizer(model, (*LinearRegressionModel[T]).finalize)
	return model
}

// Close releases the native model. It is safe to call more than once.
func (model *LinearRegressionModel[T]) Close() error {
	return model.handle.close()
}

func (model *LinearRegressionModel[T]) finalize() {
	_ = model.Close()
}

// StandardScalerModel is a fitted standard scaler.
type StandardScalerModel[T Float] struct {
	handle modelHandle
}

func newStandardScalerModel[T Float](native *nativeModel) *StandardScalerModel[T] {
	model := &StandardScalerModel[T]{handle: modelHandle{native: native}}
	runtime.SetFinalizer(model, (*StandardScalerModel[T]).finalize)
	return model
}

// Close releases the native model. It is safe to call more than once.
func (model *StandardScalerModel[T]) Close() error {
	return model.handle.close()
}

func (model *StandardScalerModel[T]) finalize() {
	_ = model.Close()
}
