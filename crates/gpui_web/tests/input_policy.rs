#[path = "../src/input_policy.rs"]
mod input_policy;

use input_policy::{DeleteDirection, deletion_range};
use std::ops::Range;

fn utf16_len(text: &str) -> usize {
    text.encode_utf16().count()
}

fn document_range(text: &str) -> Range<usize> {
    0..utf16_len(text)
}

#[test]
fn deletes_an_emoji_as_one_grapheme_in_both_directions() {
    let text = "a😀b";
    let full_range = document_range(text);

    assert_eq!(
        deletion_range(&(3..3), DeleteDirection::Backward, text, full_range.clone()),
        Some(1..3)
    );
    assert_eq!(
        deletion_range(&(1..1), DeleteDirection::Forward, text, full_range),
        Some(1..3)
    );
}

#[test]
fn deletes_a_combining_sequence_as_one_grapheme_in_both_directions() {
    let text = "ae\u{301}b";
    let full_range = document_range(text);

    assert_eq!(
        deletion_range(&(3..3), DeleteDirection::Backward, text, full_range.clone()),
        Some(1..3)
    );
    assert_eq!(
        deletion_range(&(1..1), DeleteDirection::Forward, text, full_range),
        Some(1..3)
    );
}

#[test]
fn deletes_a_zwj_sequence_as_one_grapheme_in_both_directions() {
    let text = "x👩\u{200d}💻y";
    let full_range = document_range(text);

    assert_eq!(
        deletion_range(&(6..6), DeleteDirection::Backward, text, full_range.clone()),
        Some(1..6)
    );
    assert_eq!(
        deletion_range(&(1..1), DeleteDirection::Forward, text, full_range),
        Some(1..6)
    );
}

#[test]
fn preserves_explicit_selection_and_rejects_unaligned_caret() {
    let text = "😀";
    let full_range = document_range(text);

    assert_eq!(
        deletion_range(&(0..2), DeleteDirection::Backward, text, full_range.clone()),
        Some(0..2)
    );
    assert_eq!(
        deletion_range(&(1..1), DeleteDirection::Backward, text, full_range.clone()),
        None
    );
    assert_eq!(
        deletion_range(&(1..1), DeleteDirection::Forward, text, full_range),
        None
    );
}
