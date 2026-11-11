package sciencekit

import (
	"math"
	"sync"
	"testing"
)

// TestConcurrentPredict proves a fitted model serves several goroutines at
// once.
func TestConcurrentPredict(t *testing.T) {
	features := mustMatrix(t, 4, 2, []float64{0, 0, 1, 0, 0, 1, 2, 3})
	targets := []float64{1, 3, 4, 14}
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

	var waitGroup sync.WaitGroup
	failures := make(chan error, 16)
	for worker := 0; worker < 16; worker++ {
		waitGroup.Add(1)
		go func() {
			defer waitGroup.Done()
			for iteration := 0; iteration < 25; iteration++ {
				predictions, err := model.Predict(features)
				if err != nil {
					failures <- err
					return
				}
				for index := range targets {
					if math.Abs(predictions[index]-targets[index]) > 1e-8 {
						failures <- &concurrencyError{"unexpected prediction"}
						return
					}
				}
			}
		}()
	}
	waitGroup.Wait()
	close(failures)
	for err := range failures {
		t.Fatalf("concurrent predict: %v", err)
	}
}

// TestConcurrentFitSerializesOneBuilder proves a single mutable builder may be
// shared by fitting goroutines because the wrapper serializes the calls.
func TestConcurrentFitSerializesOneBuilder(t *testing.T) {
	features := mustMatrix(t, 4, 2, []float64{0, 0, 1, 0, 0, 1, 2, 3})
	targets := []float64{1, 3, 4, 14}
	builder, err := NewLinearRegressionBuilder[float64]()
	if err != nil {
		t.Fatalf("NewLinearRegressionBuilder: %v", err)
	}
	defer builder.Close()

	var waitGroup sync.WaitGroup
	models := make(chan *LinearRegressionModel[float64], 8)
	failures := make(chan error, 8)
	for worker := 0; worker < 8; worker++ {
		waitGroup.Add(1)
		go func() {
			defer waitGroup.Done()
			model, err := builder.Fit(features, targets)
			if err != nil {
				failures <- err
				return
			}
			models <- model
		}()
	}
	waitGroup.Wait()
	close(failures)
	close(models)
	for err := range failures {
		t.Fatalf("concurrent fit: %v", err)
	}
	count := 0
	for model := range models {
		count++
		if _, err := model.Predict(features); err != nil {
			t.Fatalf("predict after concurrent fit: %v", err)
		}
		if err := model.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
	}
	if count != 8 {
		t.Fatalf("expected 8 fitted models, got %d", count)
	}
}

// TestConcurrentCloseAndPredict proves Close never races with an in-flight
// prediction: each call either succeeds or reports a closed handle.
func TestConcurrentCloseAndPredict(t *testing.T) {
	features := mustMatrix(t, 4, 1, []float64{0, 1, 2, 3})
	builder, err := NewLinearRegressionBuilder[float64]()
	if err != nil {
		t.Fatalf("NewLinearRegressionBuilder: %v", err)
	}
	defer builder.Close()
	model, err := builder.Fit(features, []float64{1, 3, 5, 7})
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	var waitGroup sync.WaitGroup
	waitGroup.Add(2)
	go func() {
		defer waitGroup.Done()
		for iteration := 0; iteration < 50; iteration++ {
			_, err := model.Predict(features)
			if err != nil && !IsStatus(err, StatusInvalidHandle) {
				t.Errorf("unexpected predict error: %v", err)
				return
			}
		}
	}()
	go func() {
		defer waitGroup.Done()
		if err := model.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	}()
	waitGroup.Wait()

	if _, err := model.Predict(features); !IsStatus(err, StatusInvalidHandle) {
		t.Fatalf("expected invalid-handle after close, got %v", err)
	}
}

// concurrencyError is a tiny error used by the concurrent prediction test.
type concurrencyError struct{ message string }

func (failure *concurrencyError) Error() string { return failure.message }
