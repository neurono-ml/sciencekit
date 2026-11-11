package sciencekit

import (
	"fmt"
	"reflect"
)

// BackendKind selects the math backend the native library uses for an
// operation.
type BackendKind int32

// The backend kinds the C ABI accepts.
const (
	BackendFaer           BackendKind = 0
	BackendMatrixMultiply BackendKind = 1
	BackendNdArrayLinalg  BackendKind = 2
)

// Native dtype codes from sciencekit.h.
const (
	dtypeF32 int32 = 0
	dtypeF64 int32 = 1
)

// Native estimator-family codes from sciencekit.h.
const (
	familyLinearRegression int32 = 0
	familyStandardScaler   int32 = 1
)

// Float is the scalar types the binding models.
type Float interface {
	~float32 | ~float64
}

// Matrix is a row-major dense matrix with Rows*Columns values in Data.
type Matrix[T Float] struct {
	Rows    int
	Columns int
	Data    []T
}

// NewMatrix validates the shape against the data length. The data slice is
// borrowed, never copied.
func NewMatrix[T Float](rows, columns int, data []T) (Matrix[T], error) {
	if rows <= 0 || columns <= 0 {
		return Matrix[T]{}, validationError(
			"new_matrix",
			fmt.Sprintf("rows and columns must be positive, got %dx%d", rows, columns),
		)
	}
	expected := rows * columns
	if len(data) != expected {
		return Matrix[T]{}, validationError(
			"new_matrix",
			fmt.Sprintf("shape %dx%d needs %d values, got %d", rows, columns, expected, len(data)),
		)
	}
	return Matrix[T]{Rows: rows, Columns: columns, Data: data}, nil
}

// validateTargets proves the target vector has one value per feature row.
func validateTargets[T Float](targets []T, rows int, operation string) error {
	if len(targets) != rows {
		return validationError(
			operation,
			fmt.Sprintf("the target vector has %d values, but the feature matrix has %d rows", len(targets), rows),
		)
	}
	return nil
}

// validate checks an already constructed matrix.
func (matrix Matrix[T]) validate() error {
	if matrix.Rows <= 0 || matrix.Columns <= 0 {
		return validationError(
			"matrix",
			fmt.Sprintf("rows and columns must be positive, got %dx%d", matrix.Rows, matrix.Columns),
		)
	}
	expected := matrix.Rows * matrix.Columns
	if len(matrix.Data) != expected {
		return validationError(
			"matrix",
			fmt.Sprintf("shape %dx%d needs %d values, got %d", matrix.Rows, matrix.Columns, expected, len(matrix.Data)),
		)
	}
	return nil
}

// dtypeOf maps a Go scalar type onto the ABI dtype code. Named types are
// accepted: the underlying kind decides the ABI dtype.
func dtypeOf[T Float]() int32 {
	var zero T
	if reflect.TypeOf(zero).Kind() == reflect.Float32 {
		return dtypeF32
	}
	return dtypeF64
}

// convertFloatSlice converts a native payload into the caller's scalar type,
// including named float types.
func convertFloatSlice[T Float, S Float](values []S) []T {
	converted := make([]T, len(values))
	for index, value := range values {
		converted[index] = T(value)
	}
	return converted
}

// validate rejects a backend code the ABI does not define.
func (kind BackendKind) validate() error {
	switch kind {
	case BackendFaer, BackendMatrixMultiply, BackendNdArrayLinalg:
		return nil
	default:
		return validationError(
			"backend",
			fmt.Sprintf("unknown backend kind %d; expected 0 (faer), 1 (matrixmultiply) or 2 (ndarray-linalg)", int32(kind)),
		)
	}
}
