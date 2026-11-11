package sciencekit

// LoadLinearRegression reads a fitted double-precision linear-regression model
// from an interop state file.
func LoadLinearRegression(path string) (*LinearRegressionModel[float64], error) {
	if path == "" {
		return nil, validationError("load", "the path is empty")
	}
	native, err := loadNativeModel(path)
	if err != nil {
		return nil, err
	}
	return newLinearRegressionModel[float64](native), nil
}
