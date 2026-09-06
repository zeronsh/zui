// Run on the process's main thread, as required by display-link registry
// mutations. A normal libtest worker thread would violate that requirement.
// Include the private implementation so this exercises the real CoreVideo and
// dispatch-source path without exporting test-only APIs from gpui_macos.
#[cfg(target_os = "macos")]
#[path = "../src/display_link.rs"]
mod display_link;

fn main() {
    #[cfg(target_os = "macos")]
    {
        use core_graphics::display::kCGNullDirectDisplayID;
        use display_link::WindowFrameSource;
        use std::{
            ffi::c_void,
            sync::{
                Arc,
                atomic::{AtomicBool, Ordering},
            },
        };

        extern "C" fn unused_callback(_: *mut c_void) {}
        let requested = Arc::new(AtomicBool::new(true));
        let mut source =
            WindowFrameSource::new(std::ptr::null_mut(), unused_callback, requested.clone());

        source.stop();
        assert!(!source.is_running());
        assert!(
            !requested.swap(true, Ordering::AcqRel),
            "stopping a source must allow invalidation to queue another restart"
        );

        // Force a real failed subscription without depending on the runner
        // having an active screen. The null ID never identifies display hardware.
        for _ in 0..2 {
            let error = source
                .start(kCGNullDirectDisplayID)
                .expect_err("invalid display must fail");
            assert!(!source.is_running());
            assert!(
                !requested.swap(true, Ordering::AcqRel),
                "a failed start must allow the next invalidation to retry: {error}"
            );
        }
        println!("PASS: stopped and failed frame sources allow redraw requests to retry");
    }
}
