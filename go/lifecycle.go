package sciencekit

import "sync/atomic"

// Live-handle diagnostics. They drive the finalizer tests and let callers spot
// a leak in their own programs.
var (
	openBuilders atomic.Int64
	openModels   atomic.Int64
)

func registerBuilder()   { openBuilders.Add(1) }
func unregisterBuilder() { openBuilders.Add(-1) }
func registerModel()     { openModels.Add(1) }
func unregisterModel()   { openModels.Add(-1) }

// openHandleCounts returns the wrappers that have not released their native
// handle yet.
func openHandleCounts() (int64, int64) {
	return openBuilders.Load(), openModels.Load()
}
