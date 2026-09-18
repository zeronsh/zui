# `gpui_web` runtime tests

Run the native Unicode policy tests with:

```sh
rustup run nightly-2026-09-08 cargo test --locked -p gpui_web --test input_policy
```

Run the WASM-only pointer, touch, keyboard, and dispatcher tests in Node with
`wasm-bindgen-cli` matching the workspace's `wasm-bindgen` version:

```sh
cargo install wasm-bindgen-cli --version 0.2.120 --locked
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
  rustup run nightly-2026-09-08 cargo test --locked -p gpui_web \
  --target wasm32-unknown-unknown --lib -- --nocapture
```


The hosted gate runs these checks in [`web-runtime.yml`](../../../.github/workflows/web-runtime.yml).

## Touch keyboard regressions

```sh
cargo test --locked -p gpui_web --test touch_keyboard
cargo test --locked -p gpui --lib touch_focus
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
  cargo test --locked -p gpui_web --target wasm32-unknown-unknown --test touch_keyboard_dom
```

Use the same nightly and matching wasm-bindgen runner as above. The DOM test
requires a browser and WebDriver (or `NO_HEADLESS=1` to open the test server
manually). It checks real DOM focus synchronously, not the OS keyboard.

The bridge uses the focus request made by the current press/release and the
mounted input targets registered with `Window::handle_input`, including unfocused
editors and cached paint. It never consults the previous frame's active platform
input handler. Inputs should register during paint even when unfocused. A newly
created input that has not been painted yet is not a touch target; tap it once it
is visible. Mouse/pen retain the existing unconditional DOM focus behavior.

On real iOS Safari and Android Chrome, verify blank transcript/header/sidebar
and backdrop taps do not open the keyboard; first editor taps and switching
editors do; OS-dismiss followed by a blank tap does not reopen it; scrolling and
cancelled gestures do not open it. Check hardware keyboard shortcuts after a
blank touch tap, and desktop mouse selection, typing, clipboard and IME/emoji.
Browser emulation does not validate iOS user activation or soft-keyboard dismissal.