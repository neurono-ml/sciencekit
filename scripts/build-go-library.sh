#!/usr/bin/env bash
# Build the sciencekit shared library used by the Go bindings and print the
# cgo/loader flags for the current shell.
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

cargo build -p sciencekit_ffi --release

target_directory="${CARGO_TARGET_DIR:-$repository_root/target}"
library_directory="$target_directory/release"
header_directory="$repository_root/crates/sciencekit_ffi/include"

cat <<EOF
The native library is at $library_directory.

Export these variables before building or running Go code outside the default
in-tree layout:

  export CGO_CFLAGS="-I$header_directory"
  export CGO_LDFLAGS="-L$library_directory -lsciencekit_ffi"
  export LD_LIBRARY_PATH="$library_directory:\$LD_LIBRARY_PATH"   # DYLD_LIBRARY_PATH on macOS

Then run: go test ./go/...
EOF
