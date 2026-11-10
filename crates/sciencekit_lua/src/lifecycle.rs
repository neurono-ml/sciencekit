//! Per-thread count of live opaque handles.
//!
//! The counter backs the `sciencekit.live_handles()` diagnostic and the
//! lifecycle tests: a userdata that becomes unreachable is collected by Lua,
//! its Rust payload drops and the counter returns to its baseline. It is
//! thread-local because mlua userdata never crosses threads in module mode;
//! a handle created on one thread is freed on the same thread.

use std::cell::Cell;

thread_local! {
    static LIVE_HANDLES: Cell<usize> = const { Cell::new(0) };
}

/// Record that one handle was created on this thread.
pub(crate) fn sk_lua_handle_created() {
    LIVE_HANDLES.with(|count| count.set(count.get() + 1));
}

/// Record that one handle was freed on this thread.
pub(crate) fn sk_lua_handle_freed() {
    LIVE_HANDLES.with(|count| count.set(count.get().saturating_sub(1)));
}

/// How many handles are live on the calling thread.
pub(crate) fn sk_lua_live_handles() -> usize {
    LIVE_HANDLES.with(Cell::get)
}
