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

// TestABIMajorIsEnforced proves the binding rejects a library with a
// different ABI major and accepts the compatible one.
func TestABIMajorIsEnforced(t *testing.T) {
	version, err := Version()
	if err != nil {
		t.Fatalf("Version: %v", err)
	}
	if version.Major != expectedABIMajor {
		t.Fatalf("expected ABI major %d, got %d", expectedABIMajor, version.Major)
	}
	if err := checkABICompatibility(version); err != nil {
		t.Fatalf("expected the loaded library to be compatible: %v", err)
	}
	incompatible := VersionInfo{Major: expectedABIMajor + 1}
	if err := checkABICompatibility(incompatible); !IsStatus(err, StatusUnsupported) {
		t.Fatalf("expected unsupported for an incompatible major, got %v", err)
	}
	if err := checkABICompatibility(VersionInfo{Major: expectedABIMajor - 1}); !IsStatus(err, StatusUnsupported) {
		t.Fatalf("expected unsupported for an older major, got %v", err)
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
