package sciencekit

import (
	"runtime"
	"testing"
	"time"
)

// TestBuilderCloseIsIdempotent proves Close frees the handle once and that a
// second Close is a safe no-op.
func TestBuilderCloseIsIdempotent(t *testing.T) {
	builder, err := NewLinearRegressionBuilder[float64]()
	if err != nil {
		t.Fatalf("NewLinearRegressionBuilder: %v", err)
	}
	if err := builder.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := builder.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

// TestBuilderRejectsCallsAfterClose proves configuration on a closed handle
// fails before reaching the native library.
func TestBuilderRejectsCallsAfterClose(t *testing.T) {
	builder, err := NewStandardScalerBuilder[float64]()
	if err != nil {
		t.Fatalf("NewStandardScalerBuilder: %v", err)
	}
	if err := builder.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	err = builder.WithMean(true)
	if err == nil {
		t.Fatal("expected an error from a closed builder")
	}
	if !IsStatus(err, StatusInvalidHandle) {
		t.Fatalf("expected invalid-handle, got %v", err)
	}
}

// TestBuilderFinalizerReleasesHandle proves the finalizer frees builders the
// caller forgot to close, exactly once.
func TestBuilderFinalizerReleasesHandle(t *testing.T) {
	baselineBuilders, baselineModels := openHandleCounts()
	createAndForgetBuilder(t)
	waitForHandleCounts(t, baselineBuilders, baselineModels)
}

// TestModelFinalizerReleasesHandle proves the finalizer frees fitted models
// the caller forgot to close.
func TestModelFinalizerReleasesHandle(t *testing.T) {
	baselineBuilders, baselineModels := openHandleCounts()
	createAndForgetModel(t)
	waitForHandleCounts(t, baselineBuilders, baselineModels)
}

// createAndForgetBuilder builds a wrapper that becomes unreachable.
func createAndForgetBuilder(t *testing.T) {
	t.Helper()
	builder, err := NewStandardScalerBuilder[float64]()
	if err != nil {
		t.Fatalf("NewStandardScalerBuilder: %v", err)
	}
	runtime.KeepAlive(builder)
}

// createAndForgetModel fits a wrapper that becomes unreachable.
func createAndForgetModel(t *testing.T) {
	t.Helper()
	features, err := NewMatrix(4, 1, []float64{0, 1, 2, 3})
	if err != nil {
		t.Fatalf("NewMatrix: %v", err)
	}
	builder, err := NewLinearRegressionBuilder[float64]()
	if err != nil {
		t.Fatalf("NewLinearRegressionBuilder: %v", err)
	}
	model, err := builder.Fit(features, []float64{1, 3, 5, 7})
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	_ = builder.Close()
	runtime.KeepAlive(model)
}

// waitForHandleCounts waits for the finalizers to drain every handle.
func waitForHandleCounts(t *testing.T, builders, models int64) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		runtime.GC()
		currentBuilders, currentModels := openHandleCounts()
		if currentBuilders == builders && currentModels == models {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf(
				"finalizers did not release every handle: builders %d (want %d), models %d (want %d)",
				currentBuilders, builders, currentModels, models,
			)
		}
		time.Sleep(10 * time.Millisecond)
	}
}
