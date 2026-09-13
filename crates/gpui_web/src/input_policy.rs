use std::ops::Range;

use unicode_segmentation::UnicodeSegmentation;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DeleteDirection {
    Backward,
    Forward,
}

/// Whether a completed touch tap requested an editable focus target.
pub(crate) fn touch_requests_keyboard(down: Option<bool>, up: Option<bool>) -> bool {
    // A release handler can override the press (e.g. closing a picker).
    up.or(down).unwrap_or(false)
}

#[cfg(target_family = "wasm")]
pub(crate) fn focus_after_touch(
    input: &web_sys::HtmlInputElement,
    canvas: &web_sys::HtmlCanvasElement,
    down: Option<bool>,
    up: Option<bool>,
) {
    if touch_requests_keyboard(down, up) {
        input.focus().ok();
    } else {
        // A non-editable keyboard target dismisses the soft keyboard without
        // preventing hardware keyboard shortcuts. Never defer this to a frame.
        canvas.set_tab_index(-1);
        canvas.focus().ok();
    }
}

/// Computes a deletion range without splitting a Unicode grapheme cluster.
///
/// `text_range_utf16` describes the UTF-16 document range represented by
/// `text`. A collapsed selection is only changed when its caret is a valid
/// grapheme boundary in that range. Non-collapsed selections are returned
/// unchanged because the editor already owns their exact semantics.
pub(crate) fn deletion_range(
    selection: &Range<usize>,
    direction: DeleteDirection,
    text: &str,
    text_range_utf16: Range<usize>,
) -> Option<Range<usize>> {
    if !selection.is_empty() {
        return Some(selection.clone());
    }

    let caret = selection.start;
    let relative_caret = caret.checked_sub(text_range_utf16.start)?;
    let text_range_length = text_range_utf16.end.checked_sub(text_range_utf16.start)?;
    if relative_caret > text_range_length || text.encode_utf16().count() != text_range_length {
        return None;
    }

    let caret_byte = byte_offset_from_utf16(text, relative_caret)?;
    match direction {
        DeleteDirection::Backward => {
            let (start_byte, grapheme) = text
                .grapheme_indices(true)
                .rev()
                .find(|(start_byte, _)| *start_byte < caret_byte)?;
            if start_byte + grapheme.len() != caret_byte {
                return None;
            }

            let start = text_range_utf16.start + text[..start_byte].encode_utf16().count();
            Some(start..caret)
        }
        DeleteDirection::Forward => {
            let (_, grapheme) = text
                .grapheme_indices(true)
                .find(|(start_byte, _)| *start_byte == caret_byte)?;
            let end = caret.checked_add(grapheme.encode_utf16().count())?;
            (end <= text_range_utf16.end).then_some(caret..end)
        }
    }
}

fn byte_offset_from_utf16(text: &str, target: usize) -> Option<usize> {
    if target == 0 {
        return Some(0);
    }

    let mut utf16_offset = 0;
    for (byte_offset, character) in text.char_indices() {
        if utf16_offset == target {
            return Some(byte_offset);
        }
        utf16_offset += character.len_utf16();
        if utf16_offset > target {
            return None;
        }
    }

    (utf16_offset == target).then_some(text.len())
}
