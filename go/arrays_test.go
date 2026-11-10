package sciencekit

import (
	"testing"
)

// TestDescriptorCarriesMatrixShape proves a Go matrix becomes a two-dimensional
// descriptor with the right dtype, shape and element count.
func TestDescriptorCarriesMatrixShape(t *testing.T) {
	matrix, err := NewMatrix(2, 3, []float64{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("NewMatrix: %v", err)
	}
	descriptor, err := descriptorFromMatrix(matrix)
	if err != nil {
		t.Fatalf("descriptorFromMatrix: %v", err)
	}
	defer descriptor.free()

	if got := int32(descriptor.raw.dtype); got != dtypeF64 {
		t.Fatalf("expected dtype F64 (%d), got %d", dtypeF64, got)
	}
	if got := int32(descriptor.raw.ndim); got != 2 {
		t.Fatalf("expected ndim 2, got %d", got)
	}
	if descriptor.raw.len != 6 {
		t.Fatalf("expected len 6, got %d", int64(descriptor.raw.len))
	}
	if len(descriptor.shape) != 2 || descriptor.shape[0] != 2 || descriptor.shape[1] != 3 {
		t.Fatalf("expected shape [2 3], got %v", descriptor.shape)
	}
	if descriptor.raw.strides != nil {
		t.Fatal("expected null strides for a row-major matrix")
	}
}

// TestDescriptorFromFloat32Matrix proves the dtype follows the scalar type.
func TestDescriptorFromFloat32Matrix(t *testing.T) {
	matrix, err := NewMatrix(3, 1, []float32{1, 2, 3})
	if err != nil {
		t.Fatalf("NewMatrix: %v", err)
	}
	descriptor, err := descriptorFromMatrix(matrix)
	if err != nil {
		t.Fatalf("descriptorFromMatrix: %v", err)
	}
	defer descriptor.free()

	if got := int32(descriptor.raw.dtype); got != dtypeF32 {
		t.Fatalf("expected dtype F32 (%d), got %d", dtypeF32, got)
	}
	if descriptor.raw.len != 3 {
		t.Fatalf("expected len 3, got %d", int64(descriptor.raw.len))
	}
}

// TestDescriptorRejectsMalformedMatrix proves shape validation happens in Go,
// before any native call.
func TestDescriptorRejectsMalformedMatrix(t *testing.T) {
	_, err := descriptorFromMatrix(Matrix[float64]{Rows: 2, Columns: 2, Data: []float64{1, 2, 3}})
	if err == nil {
		t.Fatal("expected an error for a malformed matrix")
	}
	if !IsStatus(err, StatusInvalidArgument) {
		t.Fatalf("expected invalid-argument, got %v", err)
	}
}

// TestDescriptorFromVector proves targets become a one-dimensional descriptor.
func TestDescriptorFromVector(t *testing.T) {
	descriptor, err := descriptorFromVector([]float64{1, 2, 3})
	if err != nil {
		t.Fatalf("descriptorFromVector: %v", err)
	}
	defer descriptor.free()

	if got := int32(descriptor.raw.ndim); got != 1 {
		t.Fatalf("expected ndim 1, got %d", got)
	}
	if descriptor.raw.len != 3 {
		t.Fatalf("expected len 3, got %d", int64(descriptor.raw.len))
	}
	if len(descriptor.shape) != 1 || descriptor.shape[0] != 3 {
		t.Fatalf("expected shape [3], got %v", descriptor.shape)
	}
}

// TestDescriptorFromEmptyVector proves an empty vector is rejected in Go.
func TestDescriptorFromEmptyVector(t *testing.T) {
	if _, err := descriptorFromVector([]float64{}); err == nil {
		t.Fatal("expected an error for an empty vector")
	}
}

// TestNewMatrixRejectsBadShapes proves constructors validate rows, columns and
// the data length.
func TestNewMatrixRejectsBadShapes(t *testing.T) {
	if _, err := NewMatrix(0, 2, []float64{}); err == nil {
		t.Fatal("expected an error for zero rows")
	}
	if _, err := NewMatrix(2, 2, []float64{1, 2, 3}); err == nil {
		t.Fatal("expected an error for a length mismatch")
	}
}
