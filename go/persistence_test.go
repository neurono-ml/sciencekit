package sciencekit

import (
	"math"
	"path/filepath"
	"testing"
)

// TestPersistenceRoundTrip proves a fitted double-precision regression saves
// and loads through the interop state container.
func TestPersistenceRoundTrip(t *testing.T) {
	features := mustMatrix(t, 4, 2, []float64{0, 0, 1, 0, 0, 1, 2, 3})
	targets := []float64{1, 3, 4, 14}
	path := filepath.Join(t.TempDir(), "linear-regression.skstate")

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

	if err := model.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := LoadLinearRegression(path)
	if err != nil {
		t.Fatalf("LoadLinearRegression: %v", err)
	}
	defer loaded.Close()

	single := mustMatrix(t, 1, 2, []float64{2, 3})
	prediction, err := loaded.Predict(single)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	if len(prediction) != 1 || math.Abs(prediction[0]-14) > 1e-8 {
		t.Fatalf("expected the loaded model to predict 14, got %v", prediction)
	}
}

// TestSaveRejectsUnsupportedModels proves the wrapper surfaces the native
// unsupported status for a scaler and for single precision.
func TestSaveRejectsUnsupportedModels(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.skstate")

	features := mustMatrix(t, 3, 1, []float64{1, 2, 3})
	scalerBuilder, err := NewStandardScalerBuilder[float64]()
	if err != nil {
		t.Fatalf("NewStandardScalerBuilder: %v", err)
	}
	defer scalerBuilder.Close()
	scaler, err := scalerBuilder.Fit(features)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	defer scaler.Close()
	if err := scaler.Save(path); err == nil {
		t.Fatal("expected an error saving a scaler")
	} else if !IsStatus(err, StatusUnsupported) {
		t.Fatalf("expected unsupported, got %v", err)
	}

	floatBuilder, err := NewLinearRegressionBuilder[float32]()
	if err != nil {
		t.Fatalf("NewLinearRegressionBuilder: %v", err)
	}
	defer floatBuilder.Close()
	floatModel, err := floatBuilder.Fit(mustMatrix(t, 3, 1, []float32{1, 2, 3}), []float32{2, 4, 6})
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	defer floatModel.Close()
	if err := floatModel.Save(path); err == nil {
		t.Fatal("expected an error saving an f32 model")
	} else if !IsStatus(err, StatusUnsupported) {
		t.Fatalf("expected unsupported, got %v", err)
	}
}

// TestLoadMissingFile proves a missing state file yields an IO error carrying
// the native message.
func TestLoadMissingFile(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "absent.skstate")
	model, err := LoadLinearRegression(missing)
	if err == nil {
		_ = model.Close()
		t.Fatal("expected an error loading a missing file")
	}
	if model != nil {
		t.Fatalf("expected a nil model on failure, got %v", model)
	}
	if !IsStatus(err, StatusUnknown) && !IsStatus(err, StatusIO) {
		t.Fatalf("expected an IO failure, got %v", err)
	}
	if len(err.Error()) == 0 {
		t.Fatal("expected the native message on the error")
	}
}
