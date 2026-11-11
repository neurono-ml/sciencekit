package sciencekit

import (
	"runtime"
	"testing"
	"unsafe"
)

// TestDescriptorCopiesDataIntoCMemory proves the descriptor owns C buffers
// independent of the Go slice, so the Go garbage collector can move or reclaim
// the input.
func TestDescriptorCopiesDataIntoCMemory(t *testing.T) {
	values := []float64{1, 2, 3, 4, 5, 6}
	matrix, err := NewMatrix(2, 3, values)
	if err != nil {
		t.Fatalf("NewMatrix: %v", err)
	}
	descriptor, err := descriptorFromMatrix(matrix)
	if err != nil {
		t.Fatalf("descriptorFromMatrix: %v", err)
	}
	defer descriptor.freeDescriptor()

	runtime.GC()
	copied := unsafe.Slice((*float64)(descriptor.raw.data), len(values))
	for index, expected := range values {
		if copied[index] != expected {
			t.Fatalf("descriptor value %d: expected %v, got %v", index, expected, copied[index])
		}
	}

	values[0] = 99
	if copied[0] != 1 {
		t.Fatalf("expected the descriptor to own a copy, got %v", copied[0])
	}
}

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
	defer descriptor.freeDescriptor()

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
	defer descriptor.freeDescriptor()

	if got := int32(descriptor.raw.dtype); got != dtypeF32 {
		t.Fatalf("expected dtype F32 (%d), got %d", dtypeF32, got)
	}
	if descriptor.raw.len != 3 {
		t.Fatalf("expected len 3, got %d", int64(descriptor.raw.len))
	}
}

// TestOutputShapeMismatchIsStructured proves a native contract violation maps
// to a ScienceKitError instead of an unclassified fmt error.
func TestOutputShapeMismatchIsStructured(t *testing.T) {
	if _, err := vectorFromValues([]float64{1}, []int64{1, 1}); !IsStatus(err, StatusInternal) {
		t.Fatalf("expected a structured internal error, got %v", err)
	}
	if _, err := matrixFromValues([]float64{1}, []int64{1}); !IsStatus(err, StatusInternal) {
		t.Fatalf("expected a structured internal error, got %v", err)
	}
	if _, err := vectorFromValues([]float64{1}, []int64{2}); !IsStatus(err, StatusInternal) {
		t.Fatalf("expected a structured shape error, got %v", err)
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
	defer descriptor.freeDescriptor()

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
