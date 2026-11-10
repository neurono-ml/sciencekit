package sciencekit_test

import (
	"fmt"

	sciencekit "github.com/neurono-ml/sciencekit-go"
)

// ExampleNewLinearRegressionBuilder fits a line through three collinear points
// and predicts the next value.
func ExampleNewLinearRegressionBuilder() {
	features, err := sciencekit.NewMatrix(3, 1, []float64{0, 1, 2})
	if err != nil {
		panic(err)
	}
	builder, err := sciencekit.NewLinearRegressionBuilder[float64]()
	if err != nil {
		panic(err)
	}
	defer builder.Close()

	model, err := builder.Fit(features, []float64{1, 3, 5})
	if err != nil {
		panic(err)
	}
	defer model.Close()

	query, err := sciencekit.NewMatrix(1, 1, []float64{4})
	if err != nil {
		panic(err)
	}
	predictions, err := model.Predict(query)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%.0f\n", predictions[0])
	// Output: 9
}

// ExampleNewStandardScalerBuilder centers a column and skips scaling.
func ExampleNewStandardScalerBuilder() {
	features, err := sciencekit.NewMatrix(3, 1, []float64{1, 2, 3})
	if err != nil {
		panic(err)
	}
	builder, err := sciencekit.NewStandardScalerBuilder[float64]()
	if err != nil {
		panic(err)
	}
	defer builder.Close()
	if err := builder.WithMean(true); err != nil {
		panic(err)
	}
	if err := builder.WithStd(false); err != nil {
		panic(err)
	}

	model, err := builder.Fit(features)
	if err != nil {
		panic(err)
	}
	defer model.Close()

	transformed, err := model.Transform(features)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%.1f\n", transformed.Data[0])
	// Output: -1.0
}
