package sciencekit

import (
	"fmt"
	"unsafe"
)

// descriptorFromMatrix validates a dense matrix and builds a row-major
// descriptor over its backing slice.
func descriptorFromMatrix[T Float](matrix Matrix[T]) (*descriptor, error) {
	if err := matrix.validate(); err != nil {
		return nil, err
	}
	shape := []int64{int64(matrix.Rows), int64(matrix.Columns)}
	return newDescriptor(dtypeOf[T](), shape, unsafe.Pointer(&matrix.Data[0]), matrix.Data), nil
}

// descriptorFromVector validates a target vector and builds a one-dimensional
// descriptor over its backing slice.
func descriptorFromVector[T Float](values []T) (*descriptor, error) {
	if len(values) == 0 {
		return nil, validationError("array", "the vector is empty")
	}
	shape := []int64{int64(len(values))}
	return newDescriptor(dtypeOf[T](), shape, unsafe.Pointer(&values[0]), values), nil
}

// matrixFromValues copies a native row-major payload back into a Go matrix.
func matrixFromValues[T Float](values []T, shape []int64) (Matrix[T], error) {
	if len(shape) != 2 {
		return Matrix[T]{}, fmt.Errorf("sciencekit: expected a two-dimensional output, got %d dimensions", len(shape))
	}
	return NewMatrix(int(shape[0]), int(shape[1]), values)
}

// vectorFromValues copies a native one-dimensional payload back into a Go
// slice.
func vectorFromValues[T Float](values []T, shape []int64) ([]T, error) {
	if len(shape) != 1 {
		return nil, fmt.Errorf("sciencekit: expected a one-dimensional output, got %d dimensions", len(shape))
	}
	if int(shape[0]) != len(values) {
		return nil, fmt.Errorf("sciencekit: output shape %d does not match %d values", shape[0], len(values))
	}
	return values, nil
}
