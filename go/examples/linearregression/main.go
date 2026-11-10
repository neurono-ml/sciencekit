// Command linearregression fits a line and prints one prediction.
package main

import (
	"fmt"
	"log"

	sciencekit "github.com/neurono-ml/sciencekit-go"
)

func main() {
	features, err := sciencekit.NewMatrix(4, 2, []float64{
		0, 0,
		1, 0,
		0, 1,
		2, 3,
	})
	if err != nil {
		log.Fatal(err)
	}
	builder, err := sciencekit.NewLinearRegressionBuilder[float64]()
	if err != nil {
		log.Fatal(err)
	}
	defer builder.Close()

	model, err := builder.Fit(features, []float64{1, 3, 4, 14})
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	predictions, err := model.Predict(features)
	if err != nil {
		log.Fatal(err)
	}
	score, err := model.Score(features, []float64{1, 3, 4, 14})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("predictions: %.4f\nR²: %.4f\n", predictions, score)
}
