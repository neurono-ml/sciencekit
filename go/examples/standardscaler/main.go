// Command standardscaler centers and scales a small matrix.
package main

import (
	"fmt"
	"log"

	sciencekit "github.com/neurono-ml/sciencekit-go"
)

func main() {
	features, err := sciencekit.NewMatrix(4, 2, []float64{
		0, 0,
		2, 2,
		4, 4,
		6, 6,
	})
	if err != nil {
		log.Fatal(err)
	}
	builder, err := sciencekit.NewStandardScalerBuilder[float64]()
	if err != nil {
		log.Fatal(err)
	}
	defer builder.Close()

	model, err := builder.Fit(features)
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	transformed, err := model.Transform(features)
	if err != nil {
		log.Fatal(err)
	}
	for row := 0; row < transformed.Rows; row++ {
		offset := row * transformed.Columns
		fmt.Printf("%.4f\n", transformed.Data[offset:offset+transformed.Columns])
	}
}
