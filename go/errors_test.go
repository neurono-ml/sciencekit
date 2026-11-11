package sciencekit

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"
)

// TestNativeErrorCarriesStatusAndMessage proves a native shape failure maps to
// a Go error that carries the code, the operation and the ABI message.
func TestNativeErrorCarriesStatusAndMessage(t *testing.T) {
	features := mustMatrix(t, 3, 2, []float64{0, 0, 1, 0, 0, 1})
	targets := []float64{1, 2, 3}
	builder, err := NewLinearRegressionBuilder[float64]()
	if err != nil {
		t.Fatalf("NewLinearRegressionBuilder: %v", err)
	}
	defer builder.Close()
	model, err := builder.Fit(features, targets)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	defer model.Close()

	wrong := mustMatrix(t, 1, 3, []float64{1, 2, 3})
	_, err = model.Predict(wrong)
	if err == nil {
		t.Fatal("expected a native shape error")
	}
	if !IsStatus(err, StatusShape) {
		t.Fatalf("expected shape, got %v", err)
	}
	var failure *ScienceKitError
	if !errors.As(err, &failure) {
		t.Fatalf("expected a ScienceKitError, got %T", err)
	}
	if failure.Operation != "predict" {
		t.Fatalf("expected operation predict, got %q", failure.Operation)
	}
	if failure.Message == "" {
		t.Fatal("expected the native message on the error")
	}
	if !strings.Contains(strings.ToLower(failure.Message), "shape") {
		t.Fatalf("expected the native shape diagnosis, got %q", failure.Message)
	}
	if !strings.Contains(err.Error(), "sciencekit shape (2)") {
		t.Fatalf("expected the status name in the message, got %q", err.Error())
	}
}

// TestClosedBuilderFitFails proves fit on a closed builder returns the
// invalid-handle status instead of touching freed memory.
func TestClosedBuilderFitFails(t *testing.T) {
	builder, err := NewLinearRegressionBuilder[float64]()
	if err != nil {
		t.Fatalf("NewLinearRegressionBuilder: %v", err)
	}
	if err := builder.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	_, err = builder.Fit(mustMatrix(t, 2, 1, []float64{1, 2}), []float64{1, 2})
	if err == nil {
		t.Fatal("expected an error from a closed builder")
	}
	if !IsStatus(err, StatusInvalidHandle) {
		t.Fatalf("expected invalid-handle, got %v", err)
	}
}

// TestClosedModelOperationsFail proves every model operation rejects a closed
// handle.
func TestClosedModelOperationsFail(t *testing.T) {
	features := mustMatrix(t, 3, 1, []float64{1, 2, 3})
	builder, err := NewLinearRegressionBuilder[float64]()
	if err != nil {
		t.Fatalf("NewLinearRegressionBuilder: %v", err)
	}
	defer builder.Close()
	model, err := builder.Fit(features, []float64{2, 4, 6})
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if err := model.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := model.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}

	checks := map[string]func() error{
		"predict": func() error { _, err := model.Predict(features); return err },
		"score":   func() error { _, err := model.Score(features, []float64{2, 4, 6}); return err },
		"save":    func() error { return model.Save(filepath.Join(t.TempDir(), "closed.skstate")) },
	}
	for operation, check := range checks {
		err := check()
		if err == nil {
			t.Fatalf("%s on a closed model returned no error", operation)
		}
		if !IsStatus(err, StatusInvalidHandle) {
			t.Fatalf("%s: expected invalid-handle, got %v", operation, err)
		}
	}
}

// TestClosedScalerTransformFails proves the scaler model follows the same
// lifecycle contract.
func TestClosedScalerTransformFails(t *testing.T) {
	features := mustMatrix(t, 3, 1, []float64{1, 2, 3})
	builder, err := NewStandardScalerBuilder[float64]()
	if err != nil {
		t.Fatalf("NewStandardScalerBuilder: %v", err)
	}
	defer builder.Close()
	model, err := builder.Fit(features)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if err := model.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := model.Transform(features); !IsStatus(err, StatusInvalidHandle) {
		t.Fatalf("expected invalid-handle, got %v", err)
	}
}

// TestInvalidBackendIsRejectedInGo proves unknown backend codes never reach
// the native library.
func TestInvalidBackendIsRejectedInGo(t *testing.T) {
	builder, err := NewLinearRegressionBuilder[float64]()
	if err != nil {
		t.Fatalf("NewLinearRegressionBuilder: %v", err)
	}
	defer builder.Close()
	if err := builder.Backend(BackendKind(42)); !IsStatus(err, StatusInvalidArgument) {
		t.Fatalf("expected invalid-argument, got %v", err)
	}
}

// TestEmptyPathsAreRejected proves save and load validate the path in Go.
func TestEmptyPathsAreRejected(t *testing.T) {
	if _, err := LoadLinearRegression(""); !IsStatus(err, StatusInvalidArgument) {
		t.Fatalf("expected invalid-argument from load, got %v", err)
	}

	builder, err := NewLinearRegressionBuilder[float64]()
	if err != nil {
		t.Fatalf("NewLinearRegressionBuilder: %v", err)
	}
	defer builder.Close()
	model, err := builder.Fit(mustMatrix(t, 2, 1, []float64{1, 2}), []float64{1, 2})
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	defer model.Close()
	if err := model.Save(""); !IsStatus(err, StatusInvalidArgument) {
		t.Fatalf("expected invalid-argument from save, got %v", err)
	}
}

// TestIsStatusIgnoresUnrelatedErrors proves the helper does not match plain
// Go errors.
func TestIsStatusIgnoresUnrelatedErrors(t *testing.T) {
	if IsStatus(errors.New("unrelated"), StatusShape) {
		t.Fatal("expected an unrelated error not to match a status")
	}
	if IsStatus(nil, StatusOK) {
		t.Fatal("expected nil not to match status ok")
	}
}
