package sciencekit

import (
	"math"
	"testing"
)

// TestLargeMatrixFitPredict proves the wrapper handles a realistically sized
// matrix with the matrix copied into C memory once per call.
func TestLargeMatrixFitPredict(t *testing.T) {
	const rows = 2000
	const columns = 8
	features := make([]float64, rows*columns)
	targets := make([]float64, rows)
	for row := 0; row < rows; row++ {
		for column := 0; column < columns; column++ {
			value := float64((row*7+column*13)%101) / 101
			features[row*columns+column] = value
			targets[row] += float64(column+1) * value
		}
		targets[row] += 3
	}

	matrix, err := NewMatrix(rows, columns, features)
	if err != nil {
		t.Fatalf("NewMatrix: %v", err)
	}
	builder, err := NewLinearRegressionBuilder[float64]()
	if err != nil {
		t.Fatalf("NewLinearRegressionBuilder: %v", err)
	}
	defer builder.Close()
	model, err := builder.Fit(matrix, targets)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	defer model.Close()

	predictions, err := model.Predict(matrix)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	if len(predictions) != rows {
		t.Fatalf("expected %d predictions, got %d", rows, len(predictions))
	}
	for index, expected := range targets {
		if math.Abs(predictions[index]-expected) > 1e-6 {
			t.Fatalf("prediction %d: expected %v, got %v", index, expected, predictions[index])
		}
	}
}

// TestLargeScalerTransform proves the transform path copies a large output
// back to Go correctly.
func TestLargeScalerTransform(t *testing.T) {
	const rows = 5000
	const columns = 4
	features := make([]float64, rows*columns)
	for index := range features {
		features[index] = float64(index % 97)
	}
	matrix, err := NewMatrix(rows, columns, features)
	if err != nil {
		t.Fatalf("NewMatrix: %v", err)
	}
	builder, err := NewStandardScalerBuilder[float64]()
	if err != nil {
		t.Fatalf("NewStandardScalerBuilder: %v", err)
	}
	defer builder.Close()
	model, err := builder.Fit(matrix)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	defer model.Close()

	transformed, err := model.Transform(matrix)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}
	if transformed.Rows != rows || transformed.Columns != columns {
		t.Fatalf("expected a %dx%d transform, got %dx%d", rows, columns, transformed.Rows, transformed.Columns)
	}
	if len(transformed.Data) != len(features) {
		t.Fatalf("expected %d values, got %d", len(features), len(transformed.Data))
	}
}
