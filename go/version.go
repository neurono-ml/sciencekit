package sciencekit

import "fmt"

// expectedABIMajor is the ABI major version this binding implements. A library
// that reports another major is rejected at package initialization.
const expectedABIMajor int32 = 1

// VersionInfo is the semantic version of the C ABI.
type VersionInfo struct {
	Major int32
	Minor int32
	Patch int32
}

// Feature bits reported by Features.
const (
	FeatureLinearRegression uint64 = 1 << 0
	FeatureStandardScaler   uint64 = 1 << 1
	FeatureModelPersistence uint64 = 1 << 2
	FeatureF32              uint64 = 1 << 3
	FeatureF64              uint64 = 1 << 4
)

// cachedVersion is captured at package initialization so an incompatible
// library surfaces when the program starts, not on first use (design decision
// 4: version query at init).
var (
	cachedVersion VersionInfo
	cachedError   error
)

func init() {
	major, minor, patch, err := nativeVersion()
	cachedVersion = VersionInfo{Major: major, Minor: minor, Patch: patch}
	if err != nil {
		cachedError = err
		return
	}
	cachedError = checkABICompatibility(cachedVersion)
}

// checkABICompatibility rejects a library whose ABI major differs from the one
// this binding implements.
func checkABICompatibility(version VersionInfo) error {
	if version.Major != expectedABIMajor {
		return newError(
			StatusUnsupported,
			"abi_version",
			fmt.Sprintf(
				"the loaded library reports ABI major %d, but this binding requires %d",
				version.Major,
				expectedABIMajor,
			),
		)
	}
	return nil
}

// requireCompatibleABI returns the initialization failure, if the loaded
// library is incompatible or unreadable.
func requireCompatibleABI() error {
	return cachedError
}

// Version reads the ABI version of the loaded library, captured at package
// initialization.
func Version() (VersionInfo, error) {
	return cachedVersion, cachedError
}

// Features reads the capability bitmask of the loaded library.
func Features() uint64 {
	return nativeFeatures()
}

// HasFeature reports whether every bit in flag is present in the loaded
// library.
func HasFeature(flag uint64) bool {
	return Features()&flag == flag
}
