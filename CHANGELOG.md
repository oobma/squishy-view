# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-25

First public release.

### Added

- Anisotropic (non-proportional) viewport projection with X/Y factors and live
  preview, rendered off-screen and composited over the viewport (visual only,
  geometry is never touched).
- Live refresh on camera moves, mesh edits, selections, X-Ray / shading /
  overlay changes and Quad View.
- Remapped mouse selection: click, Shift+click and box select pick exactly what
  is shown.
  - Custom edit-mode box select implemented with bmesh + BVH occlusion
    (safe from the crashes of calling `view3d.select_box` from Python).
  - X-Ray select-through.
  - `Alt`+click loop select: edge loops, vertex loops and face strips.
- Squeeze gestures: `Alt`+drag adjusts the factors live (left/right = X,
  up/down = Y), `Ctrl+Alt+MMB` resets to 1:1, `ESC` cancels a gesture.
- Auto-pause: while Loop Cut, Knife or the Circle/Lasso selectors are active,
  the squeeze suspends (real projection) and resumes when the tool ends.
  The list of pause tools is configurable in the panel.
- Squeezed annotations: native annotations are redrawn with the squeezed
  projection and stay aligned with the model.
- Quad View support: each quadrant renders with its own matrices.
- Version-aware color pipeline for Blender 4.2 and 5.x.

### License

- GPL-3.0-or-later.

[0.1.0]: https://github.com/oobma/squishy-view/releases/tag/v0.1.0
