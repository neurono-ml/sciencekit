package sciencekit

import (
	"runtime"
	"sync"
)

// builderHandle serializes configuration and fit calls on one mutable native
// builder.
type builderHandle struct {
	mu     sync.Mutex
	native *nativeBuilder
	closed bool
}

// run executes action while holding the builder lock.
func (handle *builderHandle) run(operation string, action func(*nativeBuilder) error) error {
	handle.mu.Lock()
	defer handle.mu.Unlock()
	if handle.closed {
		return closedError(operation)
	}
	return action(handle.native)
}

// close releases the native builder exactly once.
func (handle *builderHandle) close() error {
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

// LinearRegressionBuilder configures ordinary least squares over T.
type LinearRegressionBuilder[T Float] struct {
	handle builderHandle
}

// NewLinearRegressionBuilder creates a linear-regression builder for T.
func NewLinearRegressionBuilder[T Float]() (*LinearRegressionBuilder[T], error) {
	native, err := newNativeBuilder(familyLinearRegression, dtypeOf[T]())
	if err != nil {
		return nil, err
	}
	builder := &LinearRegressionBuilder[T]{handle: builderHandle{native: native}}
	runtime.SetFinalizer(builder, (*LinearRegressionBuilder[T]).finalize)
	return builder, nil
}

// IncludeIntercept chooses whether the model solves for an intercept.
func (builder *LinearRegressionBuilder[T]) IncludeIntercept(enabled bool) error {
	return builder.handle.run("include_intercept", func(native *nativeBuilder) error {
		return native.setIncludeIntercept(enabled)
	})
}

// Backend selects the math backend.
func (builder *LinearRegressionBuilder[T]) Backend(kind BackendKind) error {
	if err := kind.validate(); err != nil {
		return err
	}
	return builder.handle.run("backend", func(native *nativeBuilder) error {
		return native.setBackend(kind)
	})
}

// Fit trains an ordinary-least-squares model on dense features and targets.
func (builder *LinearRegressionBuilder[T]) Fit(features Matrix[T], targets []T) (*LinearRegressionModel[T], error) {
	if err := features.validate(); err != nil {
		return nil, err
	}
	if err := validateTargets(targets, features.Rows, "fit"); err != nil {
		return nil, err
	}

	var trained *nativeModel
	err := builder.handle.run("fit", func(native *nativeBuilder) error {
		featureDescriptor, err := descriptorFromMatrix(features)
		if err != nil {
			return err
		}
		defer featureDescriptor.freeDescriptor()
		targetDescriptor, err := descriptorFromVector(targets)
		if err != nil {
			return err
		}
		defer targetDescriptor.freeDescriptor()

		trained, err = native.fit(featureDescriptor, targetDescriptor)
		return err
	})
	if err != nil {
		return nil, err
	}
	return newLinearRegressionModel[T](trained), nil
}

// Close releases the native builder. It is safe to call more than once.
func (builder *LinearRegressionBuilder[T]) Close() error {
	return builder.handle.close()
}

func (builder *LinearRegressionBuilder[T]) finalize() {
	_ = builder.Close()
}

// StandardScalerBuilder configures column standardization over T.
type StandardScalerBuilder[T Float] struct {
	handle builderHandle
}

// NewStandardScalerBuilder creates a standard-scaler builder for T.
func NewStandardScalerBuilder[T Float]() (*StandardScalerBuilder[T], error) {
	native, err := newNativeBuilder(familyStandardScaler, dtypeOf[T]())
	if err != nil {
		return nil, err
	}
	builder := &StandardScalerBuilder[T]{handle: builderHandle{native: native}}
	runtime.SetFinalizer(builder, (*StandardScalerBuilder[T]).finalize)
	return builder, nil
}

// WithMean chooses whether the scaler centers its columns.
func (builder *StandardScalerBuilder[T]) WithMean(enabled bool) error {
	return builder.handle.run("with_mean", func(native *nativeBuilder) error {
		return native.setWithMean(enabled)
	})
}

// WithStd chooses whether the scaler divides by column standard deviation.
func (builder *StandardScalerBuilder[T]) WithStd(enabled bool) error {
	return builder.handle.run("with_std", func(native *nativeBuilder) error {
		return native.setWithStd(enabled)
	})
}

// Backend selects the math backend.
func (builder *StandardScalerBuilder[T]) Backend(kind BackendKind) error {
	if err := kind.validate(); err != nil {
		return err
	}
	return builder.handle.run("backend", func(native *nativeBuilder) error {
		return native.setBackend(kind)
	})
}

// Fit trains a standard scaler on dense features.
func (builder *StandardScalerBuilder[T]) Fit(features Matrix[T]) (*StandardScalerModel[T], error) {
	if err := features.validate(); err != nil {
		return nil, err
	}

	var trained *nativeModel
	err := builder.handle.run("fit", func(native *nativeBuilder) error {
		featureDescriptor, err := descriptorFromMatrix(features)
		if err != nil {
			return err
		}
		defer featureDescriptor.freeDescriptor()

		trained, err = native.fit(featureDescriptor, nil)
		return err
	})
	if err != nil {
		return nil, err
	}
	return newStandardScalerModel[T](trained), nil
}

// Close releases the native builder. It is safe to call more than once.
func (builder *StandardScalerBuilder[T]) Close() error {
	return builder.handle.close()
}

func (builder *StandardScalerBuilder[T]) finalize() {
	_ = builder.Close()
}
