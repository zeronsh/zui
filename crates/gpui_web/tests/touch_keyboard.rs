#[path = "../src/input_policy.rs"]
#[allow(dead_code)]
mod input_policy;

use input_policy::touch_requests_keyboard;

#[test]
fn blank_tap_does_not_request_keyboard() {
    assert!(!touch_requests_keyboard(None, None));
    assert!(!touch_requests_keyboard(Some(false), None));
}

#[test]
fn editor_tap_requests_keyboard_without_waiting_for_paint() {
    assert!(touch_requests_keyboard(Some(true), None));
    assert!(touch_requests_keyboard(None, Some(true)));
    assert!(touch_requests_keyboard(Some(false), Some(true)));
}

#[test]
fn release_blur_overrides_press_focus() {
    assert!(!touch_requests_keyboard(Some(true), Some(false)));
}
