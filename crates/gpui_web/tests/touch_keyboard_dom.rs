#![cfg(target_family = "wasm")]

#[path = "../src/input_policy.rs"]
#[allow(dead_code)]
mod input_policy;

use input_policy::focus_after_touch;
use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn touch_focus_changes_real_dom_target_synchronously() {
    let document = web_sys::window().unwrap().document().unwrap();
    let input: web_sys::HtmlInputElement =
        document.create_element("input").unwrap().unchecked_into();
    let canvas: web_sys::HtmlCanvasElement =
        document.create_element("canvas").unwrap().unchecked_into();
    let body = document.body().unwrap();
    body.append_child(&input).unwrap();
    body.append_child(&canvas).unwrap();
    let assert_input = || {
        assert_eq!(
            document.active_element(),
            Some(input.clone().unchecked_into())
        )
    };
    let assert_canvas = || {
        assert_eq!(
            document.active_element(),
            Some(canvas.clone().unchecked_into())
        )
    };

    // Blank tap, first editor tap, switching editors (shared DOM input), then
    // click-away. All assertions run before a frame or async callback can run.
    focus_after_touch(&input, &canvas, None, None);
    assert_canvas();
    focus_after_touch(&input, &canvas, Some(true), None);
    assert_input();
    focus_after_touch(&input, &canvas, Some(false), Some(true));
    assert_input();
    focus_after_touch(&input, &canvas, Some(false), None);
    assert_canvas();

    // OS keyboard dismissal may leave the hidden input focused. A subsequent
    // blank tap must park focus on a non-editable element rather than reopen.
    input.focus().unwrap();
    focus_after_touch(&input, &canvas, None, None);
    assert_canvas();
    focus_after_touch(&input, &canvas, Some(true), Some(false));
    assert_canvas();
    focus_after_touch(&input, &canvas, Some(true), None);
    assert_input();

    input.remove();
    canvas.remove();
}
