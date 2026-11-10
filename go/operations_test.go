package sciencekit

import (
	"math"
	"testing"
)

// TestLinearRegressionFitPredictAndScore proves the full double-precision
// regression path returns the fitted values and a perfect R² on a separable
// fixture.
func TestLinearRegressionFitPredictAndScore(t *testing.T) {
	features := mustMatrix(t, 4, 2, []float64{0, 0, 1, 0, 0, 1, 2, 3})
	targets := []float64{1, 3, 4, 14}

	builder, err := NewLinearRegressionBuilder[float64]()
	if err != nil {
		t.Fatalf("NewLinearRegressionBuilder: %v", err)
	}
	defer builder.Close()
	if err := builder.IncludeIntercept(true); err != nil {
		t.Fatalf("IncludeIntercept: %v", err)
	}
	if err := builder.Backend(BackendFaer); err != nil {
		t.Fatalf("Backend: %v", err)
	}

	model, err := builder.Fit(features, targets)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	defer model.Close()

	predictions, err := model.Predict(features)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	assertNear(t, predictions, targets, 1e-8, "predictions")

	score, err := model.Score(features, targets)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if math.Abs(score-1) > 1e-8 {
		t.Fatalf("expected R² 1, got %v", score)
	}
}

// TestLinearRegressionFloat32 proves single-precision descriptors survive the
// round trip.
func TestLinearRegressionFloat32(t *testing.T) {
	features := mustMatrix(t, 4, 1, []float32{0, 1, 2, 3})
	targets := []float32{1, 3, 5, 7}

	builder, err := NewLinearRegressionBuilder[float32]()
	if err != nil {
		t.Fatalf("NewLinearRegressionBuilder: %v", err)
	}
	defer builder.Close()
	model, err := builder.Fit(features, targets)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	defer model.Close()

	predictions, err := model.Predict(features)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	for index, expected := range targets {
		if math.Abs(float64(predictions[index]-expected)) > 1e-4 {
			t.Fatalf("prediction %d: expected %v, got %v", index, expected, predictions[index])
		}
	}
}

// TestStandardScalerTransform proves centering and scaling, and that a
// center-only scaler leaves the scale untouched.
func TestStandardScalerTransform(t *testing.T) {
	features := mustMatrix(t, 4, 2, []float64{0, 0, 2, 2, 4, 4, 6, 6})

	builder, err := NewStandardScalerBuilder[float64]()
	if err != nil {
		t.Fatalf("NewStandardScalerBuilder: %v", err)
	}
	defer builder.Close()
	if err := builder.WithMean(true); err != nil {
		t.Fatalf("WithMean: %v", err)
	}
	if err := builder.WithStd(true); err != nil {
		t.Fatalf("WithStd: %v", err)
	}
	model, err := builder.Fit(features)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	defer model.Close()

	transformed, err := model.Transform(features)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}
	if transformed.Rows != 4 || transformed.Columns != 2 {
		t.Fatalf("expected a 4x2 transform, got %dx%d", transformed.Rows, transformed.Columns)
	}
	if math.Abs(transformed.Data[0]+transformed.Data[6]) > 1e-8 {
		t.Fatalf("expected a centered first column, got %v", transformed.Data)
	}
	if math.Abs(transformed.Data[0]-transformed.Data[1]) > 1e-8 {
		t.Fatalf("expected both columns to share a scale, got %v", transformed.Data[:2])
	}

	centerOnly, err := NewStandardScalerBuilder[float64]()
	if err != nil {
		t.Fatalf("NewStandardScalerBuilder: %v", err)
	}
	defer centerOnly.Close()
	if err := centerOnly.WithMean(true); err != nil {
		t.Fatalf("WithMean: %v", err)
	}
	if err := centerOnly.WithStd(false); err != nil {
		t.Fatalf("WithStd: %v", err)
	}
	offsetModel, err := centerOnly.Fit(features)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	defer offsetModel.Close()

	offset, err := offsetModel.Transform(features)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}
	if math.Abs(offset.Data[0]-(-3)) > 1e-8 {
		t.Fatalf("expected a centered first value of -3, got %v", offset.Data[0])
	}
}

// TestFitRejectsMismatchedTargets proves the wrapper validates target length
// before reaching the library.
func TestFitRejectsMismatchedTargets(t *testing.T) {
	features := mustMatrix(t, 3, 1, []float64{1, 2, 3})
	builder, err := NewLinearRegressionBuilder[float64]()
	if err != nil {
		t.Fatalf("NewLinearRegressionBuilder: %v", err)
	}
	defer builder.Close()

	if _, err := builder.Fit(features, []float64{1, 2}); err == nil {
		t.Fatal("expected an error for mismatched targets")
	} else if !IsStatus(err, StatusInvalidArgument) {
		t.Fatalf("expected invalid-argument, got %v", err)
	}
}

// mustMatrix builds a matrix or fails the test.
func mustMatrix[T Float](t *testing.T, rows, columns int, data []T) Matrix[T] {
	t.Helper()
	matrix, err := NewMatrix(rows, columns, data)
	if err != nil {
		t.Fatalf("NewMatrix: %v", err)
	}
	return matrix
}

// assertNear compares two slices within a tolerance.
func assertNear(t *testing.T, actual, expected []float64, tolerance float64, label string) {
	t.Helper()
	if len(actual) != len(expected) {
		t.Fatalf("%s: expected %d values, got %d", label, len(expected), len(actual))
	}
	for index := range expected {
		if math.Abs(actual[index]-expected[index]) > tolerance {
			t.Fatalf("%s[%d]: expected %v, got %v", label, index, expected[index], actual[index])
		}
	}
}
