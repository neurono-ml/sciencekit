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

// Predict returns one response per feature row.
func (model *LinearRegressionModel[T]) Predict(features Matrix[T]) ([]T, error) {
	if err := features.validate(); err != nil {
		return nil, err
	}
	native, release, err := model.handle.acquire("predict")
	if err != nil {
		return nil, err
	}
	defer release()

	featureDescriptor, err := descriptorFromMatrix(features)
	if err != nil {
		return nil, err
	}
	defer featureDescriptor.freeDescriptor()

	values, shape, err := nativePredict[T](native, featureDescriptor)
	if err != nil {
		return nil, err
	}
	return vectorFromValues(values, shape)
}

// Score returns the coefficient of determination on the given data.
func (model *LinearRegressionModel[T]) Score(features Matrix[T], targets []T) (float64, error) {
	if err := features.validate(); err != nil {
		return 0, err
	}
	if err := validateTargets(targets, features.Rows, "score"); err != nil {
		return 0, err
	}
	native, release, err := model.handle.acquire("score")
	if err != nil {
		return 0, err
	}
	defer release()

	featureDescriptor, err := descriptorFromMatrix(features)
	if err != nil {
		return 0, err
	}
	defer featureDescriptor.freeDescriptor()
	targetDescriptor, err := descriptorFromVector(targets)
	if err != nil {
		return 0, err
	}
	defer targetDescriptor.freeDescriptor()

	score, err := native.score(featureDescriptor, targetDescriptor)
	return score, err
}

// Save writes the fitted model state to path.
func (model *LinearRegressionModel[T]) Save(path string) error {
	if path == "" {
		return validationError("save", "the path is empty")
	}
	native, release, err := model.handle.acquire("save")
	if err != nil {
		return err
	}
	defer release()
	return native.save(path)
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

// Transform standardizes features with the fitted column statistics.
func (model *StandardScalerModel[T]) Transform(features Matrix[T]) (Matrix[T], error) {
	if err := features.validate(); err != nil {
		return Matrix[T]{}, err
	}
	native, release, err := model.handle.acquire("transform")
	if err != nil {
		return Matrix[T]{}, err
	}
	defer release()

	featureDescriptor, err := descriptorFromMatrix(features)
	if err != nil {
		return Matrix[T]{}, err
	}
	defer featureDescriptor.freeDescriptor()

	values, shape, err := nativeTransform[T](native, featureDescriptor)
	if err != nil {
		return Matrix[T]{}, err
	}
	return matrixFromValues(values, shape)
}

// Save writes the fitted model state to path. The native library currently
// rejects scaler persistence with StatusUnsupported.
func (model *StandardScalerModel[T]) Save(path string) error {
	if path == "" {
		return validationError("save", "the path is empty")
	}
	native, release, err := model.handle.acquire("save")
	if err != nil {
		return err
	}
	defer release()
	return native.save(path)
}

// Close releases the native model. It is safe to call more than once.
func (model *StandardScalerModel[T]) Close() error {
	return model.handle.close()
}

func (model *StandardScalerModel[T]) finalize() {
	_ = model.Close()
}

// nativePredict dispatches one prediction on the model's scalar type.
func nativePredict[T Float](model *nativeModel, features *descriptor) ([]T, []int64, error) {
	if dtypeOf[T]() == dtypeF32 {
		values, shape, err := model.predictF32(features)
		return convertFloatSlice[T](values), shape, err
	}
	values, shape, err := model.predictF64(features)
	return convertFloatSlice[T](values), shape, err
}

// nativeTransform dispatches one transform on the model's scalar type.
func nativeTransform[T Float](model *nativeModel, features *descriptor) ([]T, []int64, error) {
	if dtypeOf[T]() == dtypeF32 {
		values, shape, err := model.transformF32(features)
		return convertFloatSlice[T](values), shape, err
	}
	values, shape, err := model.transformF64(features)
	return convertFloatSlice[T](values), shape, err
}
