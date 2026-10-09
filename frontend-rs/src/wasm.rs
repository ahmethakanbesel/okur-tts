//! WebAssembly exports with a plain C ABI (no bindings generator): the host writes UTF-8 text into memory from
//! `alloc`, calls `letters`, and reads back `[len: u32 LE][utf-8 bytes]` from the returned pointer, then frees both.
//! Kept tiny so any host (JS, Swift, Kotlin via wasm runtimes) can call it.

use std::sync::OnceLock;

use crate::Frontend;

static FRONTEND: OnceLock<Frontend> = OnceLock::new();

#[unsafe(no_mangle)]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len.max(1));
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// # Safety
/// `ptr` must come from `alloc(capacity)` with the same `capacity`, and not be freed twice.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, capacity: usize) {
    drop(unsafe { Vec::from_raw_parts(ptr, 0, capacity.max(1)) });
}

/// # Safety
/// `ptr..ptr+len` must be a live allocation holding UTF-8 (invalid sequences are replaced).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn letters(ptr: *const u8, len: usize) -> *mut u8 {
    let input = unsafe { std::slice::from_raw_parts(ptr, len) };
    let text = String::from_utf8_lossy(input);
    let out = FRONTEND.get_or_init(Frontend::new).letters(&text);
    let bytes = out.as_bytes();
    let total = 4 + bytes.len();
    let buf = alloc(total);
    unsafe {
        std::ptr::copy_nonoverlapping((bytes.len() as u32).to_le_bytes().as_ptr(), buf, 4);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.add(4), bytes.len());
    }
    buf
}
