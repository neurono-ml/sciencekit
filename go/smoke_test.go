package sciencekit

import "testing"

// TestVersionReportsABI proves the module links against the native library and
// that the ABI version is readable.
func TestVersionReportsABI(t *testing.T) {
	version, err := Version()
	if err != nil {
		t.Fatalf("Version returned an error: %v", err)
	}
	if version.Major != 1 {
		t.Fatalf("expected ABI major 1, got %d", version.Major)
	}
}

// TestFeaturesIncludeEveryFamily proves the capability bitmask is readable.
func TestFeaturesIncludeEveryFamily(t *testing.T) {
	features := Features()
	for _, flag := range []uint64{
		FeatureLinearRegression,
		FeatureStandardScaler,
		FeatureModelPersistence,
		FeatureF32,
		FeatureF64,
	} {
		if features&flag == 0 {
			t.Fatalf("feature bit 0x%x is missing from 0x%x", flag, features)
		}
	}
}

// TestStatusName proves status codes map to their stable lowercase names.
func TestStatusName(t *testing.T) {
	if got := StatusShape.Name(); got != "shape" {
		t.Fatalf("expected shape, got %q", got)
	}
	if got := Status(99).Name(); got != "unknown" {
		t.Fatalf("expected unknown, got %q", got)
	}
}
