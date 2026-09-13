use super::*;
use crate::{
    InteractiveElement, MouseDownEvent, ParentElement, Styled, TestAppContext, canvas, div,
};

// A minimal editor exercises the real paint-time registration and mouse dispatch,
// rather than substituting a platform input handler left over from the last frame.
struct Editor;
impl InputHandler for Editor {
    fn selected_text_range(
        &mut self,
        _: bool,
        _: &mut Window,
        _: &mut App,
    ) -> Option<crate::UTF16Selection> {
        None
    }
    fn marked_text_range(&mut self, _: &mut Window, _: &mut App) -> Option<Range<usize>> {
        None
    }
    fn text_for_range(
        &mut self,
        _: Range<usize>,
        _: &mut Option<Range<usize>>,
        _: &mut Window,
        _: &mut App,
    ) -> Option<String> {
        None
    }
    fn replace_text_in_range(
        &mut self,
        _: Option<Range<usize>>,
        _: &str,
        _: &mut Window,
        _: &mut App,
    ) {
    }
    fn replace_and_mark_text_in_range(
        &mut self,
        _: Option<Range<usize>>,
        _: &str,
        _: Option<Range<usize>>,
        _: &mut Window,
        _: &mut App,
    ) {
    }
    fn unmark_text(&mut self, _: &mut Window, _: &mut App) {}
    fn bounds_for_range(
        &mut self,
        _: Range<usize>,
        _: &mut Window,
        _: &mut App,
    ) -> Option<Bounds<Pixels>> {
        None
    }
    fn character_index_for_point(
        &mut self,
        _: Point<Pixels>,
        _: &mut Window,
        _: &mut App,
    ) -> Option<usize> {
        None
    }
}

struct FocusProbe {
    editors: [FocusHandle; 2],
    neutral: FocusHandle,
    show_editors: bool,
}

impl Render for FocusProbe {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let mut root = div().flex().flex_col().size_full();
        if self.show_editors {
            for (i, focus) in self.editors.iter().enumerate() {
                let pressed = focus.clone();
                let painted = focus.clone();
                root = root.child(
                    div()
                        .id(i)
                        .track_focus(focus)
                        .w(px(100.))
                        .h(px(40.))
                        .flex_none()
                        .on_mouse_down(MouseButton::Left, move |_, window, cx| {
                            window.focus(&pressed, cx)
                        })
                        .child(
                            canvas(
                                |_, _, _| (),
                                move |_, _, window, cx| {
                                    window.handle_input(&painted, Editor, cx);
                                },
                            )
                            .size_full(),
                        ),
                );
            }
        }
        let neutral = self.neutral.clone();
        root.child(
            div()
                .id("blur")
                .w(px(100.))
                .h(px(40.))
                .flex_none()
                .on_mouse_down(MouseButton::Left, |_, window, _| window.blur()),
        )
        .child(
            div()
                .id("neutral")
                .w(px(100.))
                .h(px(40.))
                .flex_none()
                .on_mouse_down(MouseButton::Left, move |_, window, cx| {
                    window.focus(&neutral, cx)
                }),
        )
    }
}

fn press(window: &mut Window, cx: &mut App, x: f32, y: f32) -> DispatchEventResult {
    window.dispatch_event(
        PlatformInput::MouseDown(MouseDownEvent {
            button: MouseButton::Left,
            position: point(px(x), px(y)),
            modifiers: Modifiers::default(),
            click_count: 1,
            first_mouse: false,
        }),
        cx,
    )
}

#[test]
fn touch_focus_uses_this_press_not_the_previous_painted_editor() {
    let mut cx = TestAppContext::single();
    let host = cx.add_window(|_, cx| FocusProbe {
        editors: [cx.focus_handle(), cx.focus_handle()],
        neutral: cx.focus_handle(),
        show_editors: true,
    });
    cx.update_window(host.into(), |_, window, cx| window.draw(cx).clear())
        .unwrap();
    host.update(&mut cx, |view, window, cx| {
        assert_eq!(window.rendered_frame.input_focus_handles.len(), 2);
        assert!(window.platform_window.take_input_handler().is_none());
        // Blank, first editor tap, editor switching, then same editor again:
        // there is intentionally NO repaint between any of these presses.
        assert_eq!(press(window, cx, 150., 200.).text_input_focus, None);
        assert_eq!(press(window, cx, 20., 20.).text_input_focus, Some(true));
        assert!(view.editors[0].is_focused(window));
        assert_eq!(press(window, cx, 20., 60.).text_input_focus, Some(true));
        assert!(view.editors[1].is_focused(window));
        assert_eq!(press(window, cx, 20., 60.).text_input_focus, Some(true));
        // OS keyboard dismissal does not necessarily change GPUI focus. A
        // blank press must not mistake that retained editor for fresh intent.
        assert_eq!(press(window, cx, 150., 200.).text_input_focus, None);
        assert_eq!(press(window, cx, 20., 100.).text_input_focus, Some(false));
        assert!(window.focus.is_none());
        assert_eq!(press(window, cx, 20., 140.).text_input_focus, Some(false));
        assert!(view.neutral.is_focused(window));
    })
    .unwrap();
}

#[test]
fn touch_focus_registration_survives_cached_paint_and_clears_on_unmount() {
    let mut cx = TestAppContext::single();
    let host = cx.add_window(|_, cx| FocusProbe {
        editors: [cx.focus_handle(), cx.focus_handle()],
        neutral: cx.focus_handle(),
        show_editors: true,
    });
    cx.update_window(host.into(), |_, window, cx| {
        window.draw(cx).clear();
        let start = window.paint_index();
        window
            .next_frame
            .input_focus_handles
            .push(window.rendered_frame.input_focus_handles[0]);
        let end = window.paint_index();
        // Exercise the same range-copy operation used by cached views.
        std::mem::swap(&mut window.next_frame, &mut window.rendered_frame);
        window.reuse_paint(start..end);
        assert_eq!(window.next_frame.input_focus_handles.len(), 3);
        std::mem::swap(&mut window.next_frame, &mut window.rendered_frame);
        window.next_frame.clear();
    })
    .unwrap();
    host.update(&mut cx, |view, _, cx| {
        view.show_editors = false;
        cx.notify();
    })
    .unwrap();
    cx.update_window(host.into(), |_, window, cx| {
        window.draw(cx).clear();
        assert!(window.rendered_frame.input_focus_handles.is_empty());
    })
    .unwrap();
}
