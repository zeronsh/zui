//! Continuous corners: gpui's rounded quads against Core Animation's `cornerCurve = .continuous`.
//! Shapes here are symmetric, so the bitmap's row order (Core Graphics is y-up) doesn't matter.
use super::*;
use cocoa::base::{id, nil};
use cocoa::foundation::{NSPoint, NSRect, NSSize, NSString};
use core_graphics::{
    base::kCGImageAlphaNone, color::CGColor, color_space::CGColorSpace, context::CGContext,
};
use gpui::{BorderStyle, Corners, Edges, Hsla, solid_background, transparent_black};

pub(super) fn bounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
    Bounds {
        origin: point(ScaledPixels(x), ScaledPixels(y)),
        size: size(ScaledPixels(w), ScaledPixels(h)),
    }
}

pub(super) fn radii(r: f32) -> Corners<ScaledPixels> {
    Corners::all(ScaledPixels(r))
}

fn gray(l: f32) -> Hsla {
    Hsla {
        h: 0.,
        s: 0.,
        l,
        a: 1.,
    }
}

pub(super) fn white_quad(b: Bounds<ScaledPixels>, corner_radii: Corners<ScaledPixels>) -> Quad {
    Quad {
        order: 0,
        border_style: BorderStyle::default(),
        bounds: b,
        content_mask: ContentMask { bounds: b },
        background: solid_background(gray(1.)),
        border_color: transparent_black(),
        corner_radii,
        border_widths: Edges::default(),
        fade: gpui::EdgeFadeParams::default(),
    }
}

/// What `draw` paints in white over a black backdrop, as coverage: the red channel over 255.
pub(super) fn gpui_mask(w: i32, h: i32, draw: impl FnOnce(&mut Scene)) -> Vec<f32> {
    let mut scene = Scene::default();
    let mut backdrop = white_quad(bounds(0., 0., w as f32, h as f32), Corners::default());
    backdrop.background = solid_background(gray(0.));
    scene.insert_primitive(backdrop);
    draw(&mut scene);
    scene.finish();
    let mut renderer =
        MetalRenderer::new_headless(Arc::new(Mutex::new(InstanceBufferPool::default())));
    let image = renderer
        .render_scene_to_image(&scene, size(DevicePixels(w), DevicePixels(h)))
        .expect("render_scene_to_image failed");
    image.pixels().map(|p| p[0] as f32 / 255.).collect()
}

/// Core Animation's coverage of a white `w × h` rect with `cornerRadius = r` and the given
/// `cornerCurve` (`"continuous"` or `"circular"`), at 1×.
pub(super) fn ca_mask(w: usize, h: usize, r: f64, curve: &str) -> Vec<f32> {
    let mut ctx = CGContext::create_bitmap_context(
        None,
        w,
        h,
        8,
        w,
        &CGColorSpace::create_device_gray(),
        kCGImageAlphaNone,
    );
    unsafe {
        let layer: id = msg_send![class!(CALayer), layer];
        let frame = NSRect::new(NSPoint::new(0., 0.), NSSize::new(w as f64, h as f64));
        let _: () = msg_send![layer, setFrame: frame];
        let white = CGColor::rgb(1., 1., 1., 1.);
        let _: () = msg_send![layer, setBackgroundColor: white.as_concrete_TypeRef()];
        let _: () = msg_send![layer, setCornerRadius: r];
        let _: () = msg_send![layer, setCornerCurve: NSString::alloc(nil).init_str(curve)];
        let _: () = msg_send![layer, renderInContext: ctx.as_ptr()];
    }
    ctx.data().iter().map(|&v| v as f32 / 255.).collect()
}

/// Summed absolute coverage difference, in px².
pub(super) fn diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(a, b)| (a - b).abs()).sum()
}

#[test]
fn continuous_corner_matches_core_animation() {
    for r in [8.0f32, 20.0, 40.0] {
        let ours = gpui_mask(200, 200, |s| {
            s.insert_primitive(white_quad(bounds(0., 0., 200., 200.), radii(r)));
        });
        let apple = ca_mask(200, 200, r as f64, "continuous");
        let circle = ca_mask(200, 200, r as f64, "circular");
        let (e, base) = (diff(&ours, &apple), diff(&circle, &apple));
        assert!(
            e < 0.5 * base,
            "r = {r}: {e:.2} px² off Apple's curve, the circle is {base:.2}"
        );
    }
}

#[test]
fn pill_follows_apple_continuous_curve() {
    // Where a side has no room for the full curve, Core Animation compresses the curve into that side
    // and keeps easing into the other: a pill is not a circle.
    for (w, h, r) in [(200, 40, 20.), (200, 40, 18.), (88, 64, 32.)] {
        let ours = gpui_mask(w, h, |s| {
            s.insert_primitive(white_quad(bounds(0., 0., w as f32, h as f32), radii(r)));
        });
        let apple = ca_mask(w as usize, h as usize, r as f64, "continuous");
        let circle = ca_mask(w as usize, h as usize, r as f64, "circular");
        let (e, base) = (diff(&ours, &apple), diff(&circle, &apple));
        assert!(
            e < 0.6 * base,
            "{w} × {h} at r = {r}: {e:.2} px² off Apple's curve, the circle is {base:.2}"
        );
    }
}

#[test]
fn rounded_clip_takes_the_same_curve() {
    for (w, h, r) in [(100, 100, 20.), (200, 40, 20.)] {
        let b = bounds(0., 0., w as f32, h as f32);
        let quad = gpui_mask(w, h, |s| s.insert_primitive(white_quad(b, radii(r))));
        let clipped = gpui_mask(w, h, |s| {
            let mut quad = white_quad(b, Corners::default());
            quad.fade = gpui::EdgeFadeParams {
                clip_left: 0.,
                clip_top: 0.,
                clip_right: w as f32,
                clip_bottom: h as f32,
                clip_radius: r,
                ..Default::default()
            };
            s.insert_primitive(quad)
        });
        let e = diff(&quad, &clipped);
        assert!(
            e < 2.0,
            "{w} × {h} at r = {r}: clip and quad corners differ by {e:.2} px²"
        );
    }
}

#[test]
fn blurred_shadow_takes_the_same_curve() {
    // A blur of a pixel keeps the half-coverage contour on the shape's edge, so the thresholded shadow
    // should match the thresholded quad, not a circular corner.
    let b = bounds(0., 0., 200., 200.);
    let r = 30.;
    let threshold = |m: Vec<f32>| -> Vec<f32> {
        m.into_iter()
            .map(|v| if v >= 0.5 { 1. } else { 0. })
            .collect()
    };
    let quad = threshold(gpui_mask(200, 200, |s| {
        s.insert_primitive(white_quad(b, radii(r)))
    }));
    let shadow = threshold(gpui_mask(200, 200, |s| {
        s.insert_primitive(Shadow {
            order: 0,
            blur_radius: ScaledPixels(1.),
            bounds: b,
            corner_radii: radii(r),
            content_mask: ContentMask { bounds: b },
            color: gray(1.),
            element_bounds: b,
            element_corner_radii: radii(r),
            inset: 0,
            pad: 0,
        })
    }));
    let circle = threshold(ca_mask(200, 200, r as f64, "circular"));
    let (e, base) = (diff(&shadow, &quad), diff(&circle, &quad));
    assert!(
        e < 0.3 * base,
        "blurred shadow: {e:.0} px off the quad's curve, the circle is {base:.0}"
    );
}

#[test]
fn each_corner_takes_its_own_radius() {
    let mut top_left_only = Corners::default();
    top_left_only.top_left = ScaledPixels(20.);
    let ours = gpui_mask(200, 200, |s| {
        s.insert_primitive(white_quad(bounds(0., 0., 200., 200.), top_left_only));
    });
    let apple = ca_mask(200, 200, 20., "continuous");
    let circle = ca_mask(200, 200, 20., "circular");
    // Only the top-left 50 × 50 is rounded; Apple's mask is symmetric, so its top-left is a reference.
    let corner = |m: &[f32]| -> Vec<f32> {
        (0..50)
            .flat_map(|y| m[y * 200..y * 200 + 50].to_vec())
            .collect()
    };
    let (e, base) = (
        diff(&corner(&ours), &corner(&apple)),
        diff(&corner(&circle), &corner(&apple)),
    );
    assert!(
        e < 0.5 * base,
        "top-left: {e:.2} px² off Apple's curve, the circle is {base:.2}"
    );
    // The other corners are square: every pixel outside the top-left 50 × 50 is covered.
    for y in 0..200 {
        for x in 0..200 {
            if x >= 50 || y >= 50 {
                assert!(
                    ours[y * 200 + x] > 0.99,
                    "({x}, {y}) uncovered: {}",
                    ours[y * 200 + x]
                );
            }
        }
    }
}

#[test]
fn small_quad_corner_stays_a_circle() {
    let ours = gpui_mask(12, 12, |s| {
        s.insert_primitive(white_quad(bounds(0., 0., 12., 12.), radii(6.)))
    });
    let circle = ca_mask(12, 12, 6., "circular");
    let e = diff(&ours, &circle);
    assert!(e < 1.5, "12 × 12 at r = 6: {e:.2} px² off a circle");
}

#[test]
fn half_pixel_radius_stays_filled() {
    let ours = gpui_mask(40, 40, |s| {
        s.insert_primitive(white_quad(bounds(0., 0., 40., 40.), radii(0.5)))
    });
    assert!(ours.iter().all(|c| c.is_finite()));
    for y in 1..39 {
        for x in 1..39 {
            assert!(
                ours[y * 40 + x] > 0.99,
                "({x}, {y}) uncovered: {}",
                ours[y * 40 + x]
            );
        }
    }
}

#[test]
fn unblurred_shadow_takes_the_same_curve() {
    let b = bounds(0., 0., 200., 200.);
    let quad = gpui_mask(200, 200, |s| s.insert_primitive(white_quad(b, radii(20.))));
    let shadow = gpui_mask(200, 200, |s| {
        s.insert_primitive(Shadow {
            order: 0,
            blur_radius: ScaledPixels(0.),
            bounds: b,
            corner_radii: radii(20.),
            content_mask: ContentMask { bounds: b },
            color: gray(1.),
            element_bounds: b,
            element_corner_radii: radii(20.),
            inset: 0,
            pad: 0,
        })
    });
    let e = diff(&quad, &shadow);
    assert!(e < 2.0, "shadow and quad corners differ by {e:.2} px²");
}

pub(super) fn bordered_quad(
    b: Bounds<ScaledPixels>,
    r: f32,
    widths: Edges<ScaledPixels>,
    fill: f32,
) -> Quad {
    let mut quad = white_quad(b, radii(r));
    quad.background = solid_background(gray(fill));
    quad.border_color = gray(1.);
    quad.border_widths = widths;
    quad
}

#[test]
fn uniform_border_follows_the_curve() {
    let b = bounds(0., 0., 200., 200.);
    let ours = gpui_mask(200, 200, |s| {
        s.insert_primitive(bordered_quad(b, 20., Edges::all(ScaledPixels(1.)), 0.))
    });
    // The border's area in the 40 × 40 top-left square: the curve (L(2.7) · span) and the straight rest.
    let span = 1.28 * 20.;
    let expected = 1.6582 * span + 2. * (40. - span);
    let area: f32 = (0..40)
        .flat_map(|y| ours[y * 200..y * 200 + 40].to_vec())
        .sum();
    assert!(
        (area - expected).abs() < 0.1 * expected,
        "corner border area {area:.1} px², expected {expected:.1}"
    );
}

#[test]
fn thick_border_leaves_no_holes() {
    let b = bounds(0., 0., 60., 60.);
    let filled = gpui_mask(60, 60, |s| s.insert_primitive(white_quad(b, radii(4.))));
    let bordered = gpui_mask(60, 60, |s| {
        s.insert_primitive(bordered_quad(b, 4., Edges::all(ScaledPixels(6.)), 1.))
    });
    let e = diff(&filled, &bordered);
    assert!(
        e < 0.5,
        "a white border on a white fill differs from the fill by {e:.2} px²"
    );
}

#[test]
fn uneven_border_leaves_no_holes() {
    let widths = Edges {
        top: ScaledPixels(1.),
        right: ScaledPixels(4.),
        bottom: ScaledPixels(2.),
        left: ScaledPixels(8.),
    };
    // A square's corners, and a pill's, whose curve differs along each side
    for (w, h, r) in [(120, 120, 20.), (200, 40, 20.)] {
        let b = bounds(0., 0., w as f32, h as f32);
        let filled = gpui_mask(w, h, |s| s.insert_primitive(white_quad(b, radii(r))));
        let bordered = gpui_mask(w, h, |s| {
            s.insert_primitive(bordered_quad(b, r, widths, 1.))
        });
        let e = diff(&filled, &bordered);
        assert!(
            e < 0.5,
            "{w} × {h}: a white uneven border on a white fill differs from the fill by {e:.2} px²"
        );
    }
}

/// Area of the border in the top-left 40 × 40 square, supersampled: inside the outer curve
/// (radii `span`, exponent 2.7), outside the inner one (radii `span - width`, exponent `inner_n`).
fn border_corner_area(span: f32, left: f32, top: f32, inner_n: f32) -> f32 {
    let inside = |x: f32, y: f32, rx: f32, ry: f32, n: f32| {
        let (qx, qy) = (
            (span - x).max(0.) / rx.max(1e-4),
            (span - y).max(0.) / ry.max(1e-4),
        );
        qx.powf(n) + qy.powf(n) <= 1.
    };
    const S: usize = 8;
    let mut area = 0.;
    for py in 0..40 * S {
        for px in 0..40 * S {
            let (x, y) = ((px as f32 + 0.5) / S as f32, (py as f32 + 0.5) / S as f32);
            let in_outer = inside(x, y, span, span, 2.7);
            let in_inner = x >= left && y >= top && inside(x, y, span - left, span - top, inner_n);
            if in_outer && !in_inner {
                area += 1. / (S * S) as f32;
            }
        }
    }
    area
}

#[test]
fn uneven_border_inner_edge_follows_the_curve() {
    let b = bounds(0., 0., 200., 200.);
    let widths = Edges {
        top: ScaledPixels(1.),
        right: ScaledPixels(4.),
        bottom: ScaledPixels(2.),
        left: ScaledPixels(8.),
    };
    let ours = gpui_mask(200, 200, |s| {
        s.insert_primitive(bordered_quad(b, 20., widths, 0.))
    });
    let area: f32 = (0..40)
        .flat_map(|y| ours[y * 200..y * 200 + 40].to_vec())
        .sum();
    let superellipse = border_corner_area(25.6, 8., 1., 2.7);
    let ellipse = border_corner_area(25.6, 8., 1., 2.0);
    assert!(
        (area - superellipse).abs() < 0.1 * superellipse
            && (area - superellipse).abs() < (area - ellipse).abs(),
        "corner border area {area:.1} px², expected {superellipse:.1} (n = 2.7), not {ellipse:.1} (n = 2)"
    );
}

#[test]
fn dashed_border_keeps_its_dashes_around_the_corner() {
    let b = bounds(0., 0., 200., 200.);
    let solid = gpui_mask(200, 200, |s| {
        s.insert_primitive(bordered_quad(b, 20., Edges::all(ScaledPixels(2.)), 0.))
    });
    let dashed = gpui_mask(200, 200, |s| {
        let mut quad = bordered_quad(b, 20., Edges::all(ScaledPixels(2.)), 0.);
        quad.border_style = BorderStyle::Dashed;
        s.insert_primitive(quad)
    });
    // Dashes are 2 widths on, 1 off: about 2/3 of the solid border, around the corners too.
    let (all_solid, all_dashed): (f32, f32) = (solid.iter().sum(), dashed.iter().sum());
    let ratio = all_dashed / all_solid;
    let corner = |m: &[f32]| -> f32 {
        (0..26)
            .flat_map(|y| m[y * 200..y * 200 + 26].to_vec())
            .sum()
    };
    let corner_ratio = corner(&dashed) / corner(&solid);
    assert!((0.55..0.78).contains(&ratio), "dashed / solid = {ratio:.2}");
    assert!(
        (0.3..0.95).contains(&corner_ratio),
        "corner dashed / solid = {corner_ratio:.2}"
    );
    assert!(dashed.iter().all(|c| c.is_finite()));

    // A pill's dashes run around its uneven corners too
    let b = bounds(0., 0., 200., 40.);
    let solid = gpui_mask(200, 40, |s| {
        s.insert_primitive(bordered_quad(b, 20., Edges::all(ScaledPixels(2.)), 0.))
    });
    let dashed = gpui_mask(200, 40, |s| {
        let mut quad = bordered_quad(b, 20., Edges::all(ScaledPixels(2.)), 0.);
        quad.border_style = BorderStyle::Dashed;
        s.insert_primitive(quad)
    });
    let ratio = dashed.iter().sum::<f32>() / solid.iter().sum::<f32>();
    assert!(
        (0.55..0.78).contains(&ratio),
        "pill dashed / solid = {ratio:.2}"
    );
    assert!(dashed.iter().all(|c| c.is_finite()));
}
