package sciencekit

import (
	"fmt"
)

// descriptorFromMatrix validates a dense matrix and builds a descriptor whose
// values are copied into C-owned memory.
func descriptorFromMatrix[T Float](matrix Matrix[T]) (*descriptor, error) {
	if err := matrix.validate(); err != nil {
		return nil, err
	}
	shape := []int64{int64(matrix.Rows), int64(matrix.Columns)}
	return newDescriptor(dtypeOf[T](), shape, matrix.Data)
}

// descriptorFromVector validates a target vector and builds a descriptor whose
// values are copied into C-owned memory.
func descriptorFromVector[T Float](values []T) (*descriptor, error) {
	if len(values) == 0 {
		return nil, validationError("array", "the vector is empty")
	}
	shape := []int64{int64(len(values))}
	return newDescriptor(dtypeOf[T](), shape, values)
}

// matrixFromValues copies a native row-major payload back into a Go matrix.
func matrixFromValues[T Float](values []T, shape []int64) (Matrix[T], error) {
	if len(shape) != 2 {
		return Matrix[T]{}, newError(
			StatusInternal,
			"output_shape",
			fmt.Sprintf("expected a two-dimensional output, got %d dimensions", len(shape)),
		)
	}
	return NewMatrix(int(shape[0]), int(shape[1]), values)
}

// vectorFromValues copies a native one-dimensional payload back into a Go
// slice.
func vectorFromValues[T Float](values []T, shape []int64) ([]T, error) {
	if len(shape) != 1 {
		return nil, newError(
			StatusInternal,
			"output_shape",
			fmt.Sprintf("expected a one-dimensional output, got %d dimensions", len(shape)),
		)
	}
	if int(shape[0]) != len(values) {
		return nil, newError(
			StatusInternal,
			"output_shape",
			fmt.Sprintf("output shape %d does not match %d values", shape[0], len(values)),
		)
	}
	return values, nil
}
