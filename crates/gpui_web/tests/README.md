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
