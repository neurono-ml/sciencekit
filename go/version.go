package sciencekit

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

// Version reads the ABI version of the loaded library.
func Version() (VersionInfo, error) {
	major, minor, patch, err := nativeVersion()
	if err != nil {
		return VersionInfo{}, err
	}
	return VersionInfo{Major: major, Minor: minor, Patch: patch}, nil
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
