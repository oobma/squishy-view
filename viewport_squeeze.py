# SPDX-FileCopyrightText: 2026 Viewport Squeeze Project
# SPDX-License-Identifier: GPL-2.0-or-later

bl_info = {
    "name": "Viewport Squeeze (Non-Proportional View)",
    "author": "Viewport Squeeze Project",
    "maintainer": "oobma",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Squeeze  |  Shift+Alt+Q",
    "description": (
        "Squeezes or stretches the viewport projection along X/Y without "
        "altering the geometry. Includes selection sync: click, Shift+click "
        "and box select pick exactly what is shown in the squeezed view."
    ),
    "support": "COMMUNITY",
    "category": "3D View",
}

import bpy
import bmesh
import gpu
import functools
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from bpy_extras.view3d_utils import region_2d_to_origin_3d, region_2d_to_vector_3d
from gpu_extras.presets import draw_texture_2d
from gpu_extras.batch import batch_for_shader

# ----------------------------------------------------------------------------
# Drawing engine state
# ----------------------------------------------------------------------------
_state = {
    "busy": False,
    "tex": {},          # region pointer -> {"off", "w", "h", "vm", "wm", "fx", "fy", "warmed"}
    "timer_on": False,   # main-loop timer running
    "paused": False,     # suspended while a pause tool (loop cut...) is active
    "last_error": None,  # diagnostic: last drawing exception
}

_handles = []


def _free_offscreens():
    for item in _state["tex"].values():
        try:
            item["off"].free()
        except Exception:
            pass
    _state["tex"].clear()


def _tag_redraw_3d_views(context):
    try:
        for win in context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except Exception:
        pass


def _invalidate_render(key=None):
    """Marks cached renders as stale (the timer re-renders on its next tick).

    Needed when something changes without moving the view: selection, geometry
    edits, mode switches... otherwise the cached offscreen image would keep
    showing the old state until the user pans or orbits."""
    try:
        if key is None:
            for item in _state["tex"].values():
                item["vm"] = None
        else:
            item = _state["tex"].get(key)
            if item is not None:
                item["vm"] = None
    except Exception:
        pass


def _viewport_ui_key(space):
    """Display settings that change what the offscreen render shows without
    moving the view or evaluating the depsgraph (X-Ray, shading mode,
    overlays...). The timer re-renders when this key changes."""
    sh = getattr(space, "shading", None)
    ov = getattr(space, "overlay", None)
    return (
        getattr(sh, "type", None),
        bool(getattr(sh, "show_xray", False)),
        bool(getattr(sh, "show_xray_wireframe", False)),
        round(float(getattr(sh, "xray_alpha", 0.5)), 3),
        bool(getattr(ov, "show_overlays", True)),
        bool(getattr(ov, "show_floor", True)),
    )


def _ensure_depsgraph_handler():
    """Keeps the invalidation handler registered. Without it, geometry edits
    would not refresh the squeezed image until the view moved."""
    try:
        if _depsgraph_dirty_cb not in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.append(_depsgraph_dirty_cb)
    except Exception:
        pass


def _c_idname(py_idname):
    """Python operator id ("mesh.loopcut_slide") to its C form
    ("MESH_OT_loopcut_slide"), which is the form window.modal_operators
    exposes for active modal operators."""
    if "." in py_idname:
        head, tail = py_idname.split(".", 1)
        return head.upper() + "_OT_" + tail
    return py_idname


def _modal_matches(bid, label, names):
    for n in names:
        if bid == n or label == n or bid.startswith(n):
            return True
        c = _c_idname(n)
        if bid == c or label == c or bid.startswith(c):
            return True
    return False


def _modal_pause_requested(scene):
    """True while one of the configured pause tools (modal operators) is
    active. The squeeze suspends so the tool's preview and its mouse input
    stay in the real projection (loop cut, knife...)."""
    try:
        raw = scene.viewport_squeeze.pause_tools
    except Exception:
        return False
    names = [n.strip() for n in raw.split(",") if n.strip()]
    if not names:
        return False
    try:
        for win in bpy.context.window_manager.windows:
            for op in win.modal_operators:
                try:
                    bid = op.bl_idname or ""
                    label = op.name or ""
                except Exception:
                    continue
                if _modal_matches(bid, label, names):
                    return True
    except Exception:
        pass
    return False


def _depsgraph_dirty_cb(scene, depsgraph=None):
    """Blender evaluates the depsgraph on any real scene change (selection,
    edits, mode switches, new objects...), even when the view stays still.
    Invalidate the cached render so the next timer tick shows the change."""
    try:
        props = getattr(scene, "viewport_squeeze", None)
    except Exception:
        props = None
    if props is None or not props.enabled:
        return
    if props.factor_x == 1.0 and props.factor_y == 1.0:
        return
    _invalidate_render()


# --- Squeezed annotations -------------------------------------------------
# Blender draws native annotations (3D world strokes) with the real projection
# and after all POST_PIXEL handlers, so while the squeeze is active they stay
# at their real (unsqueezed) positions. To keep them aligned with the squeezed
# model we hide the originals and redraw them projected through the same S
# matrix used for the offscreen render.

_ANN_MARKER = "_viewport_squeeze_ann_hidden"


def _annotation_datablock(scene):
    ann = getattr(scene, "annotation", None)
    if ann is None:
        try:
            ann = bpy.data.annotations[0] if len(bpy.data.annotations) else None
        except Exception:
            ann = None
    return ann


def _annotation_hiding_needed(scene, props):
    if not (props.enabled and props.draw_annotations):
        return False
    if _state.get("paused"):
        return False
    if props.factor_x == 1.0 and props.factor_y == 1.0:
        return False
    return _annotation_datablock(scene) is not None


def _apply_annotation_hiding(context):
    """Hides the original annotations while the squeeze redraws them aligned.

    The hidden state is saved with the file (workspace overlays), so a marker
    in the scene records that WE hid it: this lets the addon restore the
    annotations even after a crash/reload (the in-memory bookkeeping is gone
    but the marker persists). If the user had them hidden themselves, the
    marker is never set and their choice is respected."""
    try:
        scene = context.scene
        props = scene.viewport_squeeze
    except Exception:
        return
    hide = _annotation_hiding_needed(scene, props)
    try:
        for win in context.window_manager.windows:
            for area in win.screen.areas:
                if area.type != 'VIEW_3D':
                    continue
                for space in area.spaces:
                    if getattr(space, "type", None) != 'VIEW_3D':
                        continue
                    ov = getattr(space, "overlay", None)
                    if ov is None:
                        continue
                    if hide:
                        if ov.show_annotation:
                            ov.show_annotation = False
                            scene[_ANN_MARKER] = True
                    elif scene.get(_ANN_MARKER):
                        ov.show_annotation = True
        if not hide and scene.get(_ANN_MARKER):
            del scene[_ANN_MARKER]
    except Exception:
        pass


def _restore_annotation_hiding():
    try:
        scene = bpy.context.scene
    except Exception:
        scene = None
    try:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type != 'VIEW_3D':
                    continue
                for space in area.spaces:
                    if getattr(space, "type", None) != 'VIEW_3D':
                        continue
                    ov = getattr(space, "overlay", None)
                    if ov is not None and scene is not None and scene.get(_ANN_MARKER):
                        ov.show_annotation = True
    except Exception:
        pass
    try:
        if scene is not None and scene.get(_ANN_MARKER):
            del scene[_ANN_MARKER]
    except Exception:
        pass


def _props_changed(context):
    _tag_redraw_3d_views(context)
    _apply_annotation_hiding(context)
    try:
        props = context.scene.viewport_squeeze
    except Exception:
        return
    if props.enabled and not (props.factor_x == 1.0 and props.factor_y == 1.0):
        _start_timer()
    else:
        _stop_timer()


def _remap_point(x, y, w, h, fx, fy):
    """Squeezed-view screen point -> real-view screen point (inverse of the
    transform S applied to the projection).

    The squeezed view draws content at NDC c at (f*c+1)/2*size;
    undoing it requires x_real = x_vis/f + size/2*(1 - 1/f)."""
    if fx > 0.0:
        x = x / fx + (w / 2.0) * (1.0 - 1.0 / fx)
    if fy > 0.0:
        y = y / fy + (h / 2.0) * (1.0 - 1.0 / fy)
    return x, y


# Remapped selection modal state (mini-modal for click/box)
_modal_state = {"op": None, "draw_handle": None}


def _draw_box_cb():
    """Draws the box-select rectangle on screen (region coordinates)."""
    op = _modal_state.get("op")
    if op is None or not op.dragging:
        return
    try:
        x0, y0 = op.start
        x1, y1 = op.current
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": pts})
        shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
        shader.uniform_float("lineWidth", 1.0)
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.7))
        try:
            blend_prev = gpu.state.blend_get()
        except Exception:
            blend_prev = 'NONE'
        gpu.state.blend_set('ALPHA')
        try:
            batch.draw(shader)
        finally:
            try:
                gpu.state.blend_set(blend_prev)
            except Exception:
                pass
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Squeezed render (main-loop timer) and draw handlers
# ----------------------------------------------------------------------------
def _squeeze_timer_cb():
    """Renders the squeezed image per viewport, outside all draw phases.

    draw_view3d performs an off-screen render: inside the viewport draw it is
    rejected (PRE_VIEW/POST_VIEW: "Nested off-screen drawing not supported")
    and from POST_PIXEL it can leave the shared GPU viewport state in a bad
    shape (it broke edit-mode box select: crash in
    edbm_backbuf_check_and_select_verts). Running it from a main-loop timer
    uses the same clean context as the reference capture code.
    Returns the next interval, or None to stop when idle."""
    st = _state
    try:
        scene = bpy.context.scene
        view_layer = bpy.context.view_layer
        props = scene.viewport_squeeze
    except Exception:
        return None
    if not props.enabled or (props.factor_x == 1.0 and props.factor_y == 1.0):
        st["timer_on"] = False
        return None
    try:
        _ensure_depsgraph_handler()
        paused = _modal_pause_requested(scene)
        if paused != st.get("paused", False):
            st["paused"] = paused
            # show/hide the native annotations for the new state and repaint
            _apply_annotation_hiding(bpy.context)
            for win in bpy.context.window_manager.windows:
                for area in win.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
        if paused:
            # suspended: the real viewport shows the tool's preview
            return 1.0 / 60.0
        rendered_any = False
        seen = set()
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type != 'VIEW_3D':
                    continue
                space = area.spaces.active
                rv3d = getattr(space, "region_3d", None)
                region = next((r for r in area.regions if r.type == 'WINDOW'), None)
                if region is None or rv3d is None or getattr(space, "type", None) != 'VIEW_3D':
                    continue
                key = region.as_pointer()
                seen.add(key)
                rw, rh = int(region.width), int(region.height)
                if rw < 4 or rh < 4:
                    continue
                vm = rv3d.view_matrix.copy()
                wm = rv3d.window_matrix.copy()
                scale = max(0.25, min(1.0, props.res_scale))
                w = max(16, int(rw * scale))
                h = max(16, int(rh * scale))
                cached = st["tex"].get(key)
                if cached is None or cached["w"] != w or cached["h"] != h:
                    if cached is not None:
                        try:
                            cached["off"].free()
                        except Exception:
                            pass
                    cached = {"off": gpu.types.GPUOffScreen(w, h), "w": w, "h": h,
                              "vm": None, "wm": None, "fx": None, "fy": None,
                              "ui": None, "warmed": False}
                    st["tex"][key] = cached
                ui_key = _viewport_ui_key(space)
                if (cached["vm"] == vm and cached["wm"] == wm
                        and cached["fx"] == props.factor_x
                        and cached["fy"] == props.factor_y
                        and cached["ui"] == ui_key):
                    continue  # nothing changed: no GPU work while idle
                off = cached["off"]
                # The first render warms up the object GPU batches (avoids
                # blank captures on the first frame after enabling).
                S = Matrix.Diagonal((props.factor_x, props.factor_y, 1.0, 1.0))
                for _ in range(2 if not cached["warmed"] else 1):
                    off.draw_view3d(scene, view_layer, space, region, vm, S @ wm,
                                    draw_background=True, do_color_management=True)
                cached["warmed"] = True
                cached["vm"] = vm
                cached["wm"] = wm
                cached["fx"] = props.factor_x
                cached["fy"] = props.factor_y
                cached["ui"] = ui_key
                rendered_any = True
        for key in [k for k in st["tex"] if k not in seen]:
            try:
                st["tex"][key]["off"].free()
            except Exception:
                pass
            st["tex"].pop(key, None)
        if rendered_any:
            for win in bpy.context.window_manager.windows:
                for area in win.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
    except Exception:
        import traceback
        st["last_error"] = traceback.format_exc(limit=4)
    return 1.0 / 60.0


def _start_timer():
    _ensure_depsgraph_handler()
    if _state.get("timer_on"):
        return
    try:
        bpy.app.timers.register(_squeeze_timer_cb, first_interval=1.0 / 60.0)
        _state["timer_on"] = True
    except Exception:
        pass


def _stop_timer():
    if not _state.get("timer_on"):
        return
    _state["timer_on"] = False
    try:
        bpy.app.timers.unregister(_squeeze_timer_cb)
    except Exception:
        pass


def _vert_occluded(region, rv3d, inv, inv3, bvh, p_world, x, y, w, h, fx, fy):
    """True when a vertex is hidden behind the mesh's own visible faces.

    Casts the real-viewport ray through the vertex's un-squeezed screen
    position and compares the first hit distance with the vertex distance."""
    rx, ry = _remap_point(x, y, w, h, fx, fy)
    origin = inv @ region_2d_to_origin_3d(region, rv3d, (rx, ry))
    direction = inv3 @ region_2d_to_vector_3d(region, rv3d, (rx, ry))
    if direction.length < 1e-12:
        return False
    direction.normalize()
    hit = bvh.ray_cast(origin, direction)
    if hit is None or hit[0] is None:
        return False
    p_local = inv @ p_world
    d0 = (p_local - origin).length
    return hit[3] < d0 - max(1e-5, 1e-4 * d0)


def _edit_box_select_mesh(obj, base, region, rv3d, w, h, fx, fy,
                          xmin, xmax, ymin, ymax, extend, xray):
    """Selects the vertices of one edit mesh that fall inside the box.

    With X-Ray off only visible vertices are selectable (occlusion test);
    with X-Ray on selection goes through the mesh, like in Blender."""
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bm.verts.index_update()
    to_world = obj.matrix_world
    # occlusion geometry: only the visible faces block the selection
    bvh = None
    if not xray:
        polys = []
        for f in bm.faces:
            if f.hide or len(f.verts) < 3:
                continue
            polys.append([l.vert.index for l in f.loops])
        if polys:
            try:
                coords = [v.co.copy() for v in bm.verts]
                bvh = BVHTree.FromPolygons(coords, polys,
                                           all_triangles=False, epsilon=0.0)
            except Exception:
                bvh = None
    if not extend:
        for v in bm.verts:
            v.select = False
        for e in bm.edges:
            e.select = False
        for f in bm.faces:
            f.select = False
    inv = to_world.inverted()
    inv3 = inv.to_3x3()
    found = False
    for v in bm.verts:
        if v.hide:
            continue
        p = to_world @ v.co
        clip = base @ Vector((p.x, p.y, p.z, 1.0))
        if clip.w <= 1e-9:
            continue
        x = (clip.x / clip.w * 0.5 + 0.5) * w
        y = (clip.y / clip.w * 0.5 + 0.5) * h
        if x < xmin or x > xmax or y < ymin or y > ymax:
            continue
        if bvh is not None and _vert_occluded(region, rv3d, inv, inv3, bvh,
                                              p, x, y, w, h, fx, fy):
            continue
        v.select = True
        found = True
    if found:
        bm.select_flush(True)
    if found or not extend:
        bmesh.update_edit_mesh(me)


def _edit_box_select(context, xmin, xmax, ymin, ymax, extend):
    """Box selection evaluated directly on the edit mesh (bmesh).

    bpy.ops.view3d.select_box relies on internal GPU selection buffers that
    are not reliably available when the operator runs from Python: it crashed
    Blender in edbm_backbuf_check_and_select_verts. Here every vertex is
    projected with the same squeezed matrix used for the offscreen render,
    tested against the dragged box and occlusion-checked against the visible
    faces of the edit mesh (only visible vertices are selectable)."""
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return
    props = context.scene.viewport_squeeze
    fx, fy = props.factor_x, props.factor_y
    if fx == 1.0 and fy == 1.0:
        return
    w, h = region.width, region.height
    base = (Matrix.Diagonal((fx, fy, 1.0, 1.0))
            @ rv3d.window_matrix @ rv3d.view_matrix)
    space = context.space_data
    xray = bool(getattr(getattr(space, "shading", None), "show_xray", False))
    targets = []
    try:
        targets = [o for o in context.objects_in_mode
                   if o.type == 'MESH' and o.mode == 'EDIT']
    except Exception:
        targets = []
    if not targets:
        obj = context.edit_object
        if obj is not None and obj.type == 'MESH':
            targets = [obj]
    for obj in targets:
        try:
            _edit_box_select_mesh(obj, base, region, rv3d, w, h, fx, fy,
                                  xmin, xmax, ymin, ymax, extend, xray)
        except Exception:
            pass
    _invalidate_render()
    _tag_redraw_3d_views(context)


def _deferred_select(ctx_refs, rx, ry, extend):
    """Runs view3d.select from a timer: a clean context outside the modal
    event handler, where nested operator calls can crash edit-mode selection
    (selection backbuf)."""
    try:
        win, area, region = ctx_refs
        with bpy.context.temp_override(window=win, area=area, region=region):
            bpy.ops.view3d.select(location=(rx, ry), extend=extend,
                                  deselect_all=not extend)
    except Exception:
        pass
    # the selection changed: refresh the squeezed image without a view nudge
    _invalidate_render()
    _tag_redraw_3d_views(bpy.context)
    return None


def _deferred_select_box(ctx_refs, xmin, xmax, ymin, ymax, mode):
    """Runs view3d.select_box from a timer (same reason as above)."""
    try:
        win, area, region = ctx_refs
        with bpy.context.temp_override(window=win, area=area, region=region):
            bpy.ops.view3d.select_box(wait_for_input=False,
                                      xmin=xmin, xmax=xmax,
                                      ymin=ymin, ymax=ymax, mode=mode)
    except Exception:
        pass
    # the selection changed: refresh the squeezed image without a view nudge
    _invalidate_render()
    _tag_redraw_3d_views(bpy.context)
    return None


def _post_view_squeeze_cb():
    """Draws the squeezed image (rendered by the main-loop timer) stretched
    over the whole viewport, covering the scene below (non-destructive).

    Runs in POST_VIEW: everything drawn after this phase (POST_PIXEL operator
    previews, the squeezed annotations) stays visible on top of the image.
    Native gizmos are drawn before this phase, so they end up covered.
    Suspended while a pause tool (loop cut, knife...) is active."""
    st = _state
    try:
        scene = bpy.context.scene
        props = scene.viewport_squeeze
    except Exception:
        return
    if not props.enabled or (props.factor_x == 1.0 and props.factor_y == 1.0):
        return
    if st.get("paused"):
        return
    if st["busy"]:
        return
    st["busy"] = True
    try:
        space = bpy.context.space_data
        region = bpy.context.region
        if not isinstance(space, bpy.types.SpaceView3D) or region is None:
            return
        rw, rh = int(region.width), int(region.height)
        if rw < 4 or rh < 4:
            return
        key = region.as_pointer()
        cached = st["tex"].get(key)
        if cached is None:
            return
        # In POST_VIEW the matrix stack holds the 3D view projection, so the
        # 2D texture draw needs its own pixel-space matrix to fill the region.
        with gpu.matrix.push_pop():
            gpu.matrix.load_identity()
            gpu.matrix.load_projection_matrix(Matrix(((2.0 / rw, 0.0, 0.0, -1.0),
                                                      (0.0, 2.0 / rh, 0.0, -1.0),
                                                      (0.0, 0.0, -2.0, 0.0),
                                                      (0.0, 0.0, 0.0, 1.0))))
            draw_texture_2d(cached["off"].texture_color, (0, 0), rw, rh)
    except Exception:
        import traceback
        st["last_error"] = traceback.format_exc(limit=4)
    finally:
        st["busy"] = False


def _post_view_annotations_cb():
    """Redraws the native annotations squeezed, aligned with the squeezed
    model. The originals are hidden (see _apply_annotation_hiding) because
    Blender draws them with the real projection at unsqueezed positions."""
    st = _state
    try:
        scene = bpy.context.scene
        props = scene.viewport_squeeze
    except Exception:
        return
    if not _annotation_hiding_needed(scene, props):
        return
    ann = _annotation_datablock(scene)
    if ann is None or st["busy"]:
        return
    st["busy"] = True
    try:
        region = bpy.context.region
        space = bpy.context.space_data
        if region is None or not isinstance(space, bpy.types.SpaceView3D):
            return
        cached = st["tex"].get(region.as_pointer())
        if cached is None:
            return
        # same matrices as the drawn texture: the composition stays consistent
        vm, wm = cached["vm"], cached["wm"]
        # robustness: a newly created viewport may still show the originals
        ov = getattr(space, "overlay", None)
        if ov is not None and ov.show_annotation:
            ov.show_annotation = False
            scene[_ANN_MARKER] = True
        w, h = float(region.width), float(region.height)
        if w < 4 or h < 4:
            return
        M = Matrix.Diagonal((props.factor_x, props.factor_y, 1.0, 1.0)) @ wm @ vm
        try:
            default_thickness = float(scene.tool_settings.annotation_thickness)
        except Exception:
            default_thickness = 3.0
        polylines = []
        for layer in ann.layers:
            try:
                color = tuple(layer.color)
            except Exception:
                color = (0.38, 0.61, 0.78)
            line_width = default_thickness
            try:
                line_width = float(layer.thickness)
            except Exception:
                pass
            for frame in layer.frames:
                for stroke in frame.strokes:
                    pts = []
                    visible = True
                    for point in stroke.points:
                        co = point.co
                        v = M @ Vector((co[0], co[1], co[2], 1.0))
                        if v.w <= 1e-6:
                            visible = False
                            break
                        pts.append(((v.x / v.w + 1.0) * 0.5 * w,
                                    (v.y / v.w + 1.0) * 0.5 * h))
                    if visible and len(pts) > 1:
                        polylines.append((pts, color, line_width))
        if not polylines:
            return
        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        try:
            blend_prev = gpu.state.blend_get()
        except Exception:
            blend_prev = 'NONE'
        gpu.state.blend_set('ALPHA')
        with gpu.matrix.push_pop():
            gpu.matrix.load_identity()
            gpu.matrix.load_projection_matrix(Matrix(((2.0 / w, 0.0, 0.0, -1.0),
                                                      (0.0, 2.0 / h, 0.0, -1.0),
                                                      (0.0, 0.0, -2.0, 0.0),
                                                      (0.0, 0.0, 0.0, 1.0))))
            shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
            for pts, color, line_width in polylines:
                batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": pts})
                shader.uniform_float("lineWidth", line_width)
                shader.uniform_float("color", (color[0], color[1], color[2], 1.0))
                batch.draw(shader)
        try:
            gpu.state.blend_set(blend_prev)
        except Exception:
            pass
    except Exception:
        pass
    finally:
        st["busy"] = False


# ----------------------------------------------------------------------------
# Scene properties
# ----------------------------------------------------------------------------
class ViewportSqueezeProps(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(
        name="Non-Proportional View",
        description="Enables the non-proportional viewport projection "
                    "(visual only, does not alter the geometry)",
        default=False,
        update=lambda self, context: _props_changed(context),
    )
    factor_x: bpy.props.FloatProperty(
        name="Squeeze X",
        description="Horizontal compression/stretch factor. "
                    "1.0 = normal, <1.0 compresses, >1.0 stretches",
        default=1.0,
        min=0.1,
        max=4.0,
        step=1,
        update=lambda self, context: _props_changed(context),
    )
    factor_y: bpy.props.FloatProperty(
        name="Squeeze Y",
        description="Vertical compression/stretch factor. "
                    "1.0 = normal, <1.0 compresses, >1.0 stretches",
        default=1.0,
        min=0.1,
        max=4.0,
        step=1,
        update=lambda self, context: _props_changed(context),
    )
    res_scale: bpy.props.FloatProperty(
        name="Preview Resolution",
        description="Resolution of the squeezed render relative to the "
                    "viewport. Lower it for heavy scenes",
        default=1.0,
        min=0.25,
        max=1.0,
        subtype='FACTOR',
        update=lambda self, context: _tag_redraw_3d_views(context),
    )
    remap_selection: bpy.props.BoolProperty(
        name="Mouse Selection Sync",
        description="Remaps viewport clicks so you select exactly what you see "
                    "(click, Shift+click and box). Disable it to use tools "
                    "that need direct click-drag input",
        default=True,
    )
    draw_annotations: bpy.props.BoolProperty(
        name="Squeezed Annotations",
        description="Redraws the native scene annotations squeezed and aligned "
                    "with the model (the originals are hidden while active)",
        default=True,
        update=lambda self, context: _props_changed(context),
    )
    pause_tools: bpy.props.StringProperty(
        name="Pause While Tools",
        description="Comma-separated modal operator ids (for example "
                    "mesh.loopcut_slide, mesh.knife_tool): while one of them is "
                    "active the squeeze suspends, so its on-screen preview and "
                    "its mouse input stay in the real projection",
        default="mesh.loopcut_slide, mesh.loopcut, mesh.knife_tool, mesh.knife, mesh.knife_project",
    )


# ----------------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------------
class VIEWPORT_SQUEEZE_OT_toggle(bpy.types.Operator):
    bl_idname = "view3d.viewport_squeeze_toggle"
    bl_label = "Toggle Non-Proportional View"
    bl_description = "Toggles the non-proportional viewport projection"

    @classmethod
    def poll(cls, context):
        return context.scene is not None

    def execute(self, context):
        context.scene.viewport_squeeze.enabled = not context.scene.viewport_squeeze.enabled
        _tag_redraw_3d_views(context)
        return {'FINISHED'}


class VIEWPORT_SQUEEZE_OT_reset(bpy.types.Operator):
    bl_idname = "view3d.viewport_squeeze_reset"
    bl_label = "Reset Squeeze 1:1"
    bl_description = "Resets the X and Y factors to 1.0 (normal proportions)"

    @classmethod
    def poll(cls, context):
        return context.scene is not None

    def execute(self, context):
        props = context.scene.viewport_squeeze
        props.factor_x = 1.0
        props.factor_y = 1.0
        return {'FINISHED'}


class VIEWPORT_SQUEEZE_OT_select(bpy.types.Operator):
    bl_idname = "view3d.viewport_squeeze_select"
    bl_label = "Viewport Squeeze: Select Remapped"
    bl_description = ("Selects at the real position corresponding to the point "
                      "clicked in the squeezed view (click, Shift+click and box)")
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        try:
            props = context.scene.viewport_squeeze
        except Exception:
            return False
        if not (props.enabled and props.remap_selection):
            return False
        if props.factor_x == 1.0 and props.factor_y == 1.0:
            return False
        space = context.space_data
        return (context.region is not None and space is not None
                and getattr(space, "type", None) == 'VIEW_3D')

    def invoke(self, context, event):
        self.start = (float(event.mouse_region_x), float(event.mouse_region_y))
        self.current = self.start
        self.dragging = False
        self.extend = bool(event.shift)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            self.current = (float(event.mouse_region_x), float(event.mouse_region_y))
            dx = self.current[0] - self.start[0]
            dy = self.current[1] - self.start[1]
            if not self.dragging and (dx * dx + dy * dy) > 25.0:
                self.dragging = True
                _modal_state["op"] = self
                if _modal_state["draw_handle"] is None:
                    _modal_state["draw_handle"] = bpy.types.SpaceView3D.draw_handler_add(
                        _draw_box_cb, (), 'WINDOW', 'POST_PIXEL')
            if self.dragging:
                try:
                    context.area.tag_redraw()
                except Exception:
                    pass
            return {'RUNNING_MODAL'}
        if event.type == 'ESC':
            self._cleanup(context)
            return {'CANCELLED'}
        if event.type == 'LEFTMOUSE':
            if event.value == 'RELEASE':
                self._execute(context)
                self._cleanup(context)
                return {'FINISHED'}
            return {'RUNNING_MODAL'}
        if event.type == 'WINDOW_DEACTIVATE':
            self._cleanup(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        self._cleanup(context)

    def _cleanup(self, context):
        if _modal_state.get("draw_handle") is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(
                    _modal_state["draw_handle"], 'WINDOW')
            except Exception:
                pass
            _modal_state["draw_handle"] = None
        if _modal_state.get("op") is self:
            _modal_state["op"] = None
        try:
            context.area.tag_redraw()
        except Exception:
            pass

    def _execute(self, context):
        try:
            props = context.scene.viewport_squeeze
            region = context.region
            if region is None:
                return
            w, h = region.width, region.height
            fx, fy = props.factor_x, props.factor_y
            x0, y0 = self.start
            ctx_refs = (context.window, context.area, context.region)
            if self.dragging:
                x1, y1 = self.current
                if context.mode == 'EDIT_MESH':
                    # custom bmesh selection: view3d.select_box is not safe
                    # when called from Python in edit mode (Blender crash in
                    # edbm_backbuf_check_and_select_verts)
                    xmin, xmax = sorted((x0, x1))
                    ymin, ymax = sorted((y0, y1))
                    _edit_box_select(context, xmin, xmax, ymin, ymax,
                                     self.extend)
                    return
                rx0, ry0 = _remap_point(x0, y0, w, h, fx, fy)
                rx1, ry1 = _remap_point(x1, y1, w, h, fx, fy)
                xmin, xmax = sorted((rx0, rx1))
                ymin, ymax = sorted((ry0, ry1))
                bpy.app.timers.register(
                    functools.partial(_deferred_select_box, ctx_refs,
                                      int(round(xmin)), int(round(xmax)),
                                      int(round(ymin)), int(round(ymax)),
                                      'ADD' if self.extend else 'SET'),
                    first_interval=0.0)
            else:
                rx, ry = _remap_point(x0, y0, w, h, fx, fy)
                bpy.app.timers.register(
                    functools.partial(_deferred_select, ctx_refs,
                                      int(round(rx)), int(round(ry)),
                                      self.extend),
                    first_interval=0.0)
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Panel (N-Panel / Sidebar)
# ----------------------------------------------------------------------------
class VIEW3D_PT_viewport_squeeze(bpy.types.Panel):
    bl_label = "Viewport Squeeze"
    bl_idname = "VIEW3D_PT_viewport_squeeze"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Squeeze"

    def draw(self, context):
        layout = self.layout
        props = context.scene.viewport_squeeze
        layout.prop(props, "enabled")
        if not props.enabled:
            layout.label(text="Visual only, it does not alter geometry.", icon='INFO')
            return
        col = layout.column(align=True)
        col.prop(props, "factor_x", slider=True)
        col.prop(props, "factor_y", slider=True)
        layout.operator("view3d.viewport_squeeze_reset", icon='LOOP_BACK')
        layout.separator()
        layout.prop(props, "res_scale", slider=True)
        layout.label(text="Lower the resolution for heavy scenes.", icon='MEMORY')
        layout.separator()
        layout.prop(props, "remap_selection")
        if props.remap_selection:
            layout.label(text="Click, Shift+click and box select what you see.", icon='MOUSE_LMB')
            layout.label(text="For click-drag tools: disable it.", icon='INFO')
        else:
            layout.label(text="Mouse desynced: select with squeeze off.", icon='ERROR')
        layout.separator()
        layout.prop(props, "draw_annotations")
        if props.draw_annotations:
            layout.label(text="Annotations are redrawn squeezed and aligned.", icon='INFO')
        else:
            layout.label(text="Annotations stay at their real positions.", icon='ERROR')
        layout.separator()
        layout.prop(props, "pause_tools")
        layout.label(text="Squeeze pauses while these tools are active.", icon='TOOL_SETTINGS')


# ----------------------------------------------------------------------------
# Keymap
# ----------------------------------------------------------------------------
_addon_keymaps = []


def _register_keymap():
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc is None:
        return
    # defensive cleanup (re-registrations) and get-or-create of the "3D View" keymap
    for km in list(kc.keymaps):
        if km.name == "3D View":
            for kmi in list(km.keymap_items):
                if kmi.idname.startswith("view3d.viewport_squeeze"):
                    km.keymap_items.remove(kmi)
    km = kc.keymaps.get("3D View")
    if km is None:
        km = kc.keymaps.new(name="3D View", space_type='VIEW_3D')
    kmi = km.keymap_items.new(
        "view3d.viewport_squeeze_toggle", 'Q', 'PRESS', shift=True, alt=True)
    _addon_keymaps.append((km, kmi))
    kmi = km.keymap_items.new(
        "view3d.viewport_squeeze_select", 'LEFTMOUSE', 'PRESS',
        ctrl=False, shift=False, alt=False)
    _addon_keymaps.append((km, kmi))
    kmi = km.keymap_items.new(
        "view3d.viewport_squeeze_select", 'LEFTMOUSE', 'PRESS',
        ctrl=False, shift=True, alt=False)
    _addon_keymaps.append((km, kmi))


def _unregister_keymap():
    for km, kmi in _addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    _addon_keymaps.clear()


# ----------------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------------
_classes = (
    ViewportSqueezeProps,
    VIEWPORT_SQUEEZE_OT_toggle,
    VIEWPORT_SQUEEZE_OT_reset,
    VIEWPORT_SQUEEZE_OT_select,
    VIEW3D_PT_viewport_squeeze,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.viewport_squeeze = bpy.props.PointerProperty(
        type=ViewportSqueezeProps)
    h1 = bpy.types.SpaceView3D.draw_handler_add(_post_view_squeeze_cb, (), 'WINDOW', 'POST_VIEW')
    h2 = bpy.types.SpaceView3D.draw_handler_add(_post_view_annotations_cb, (), 'WINDOW', 'POST_VIEW')
    _handles.extend((h1, h2))
    _ensure_depsgraph_handler()
    _register_keymap()
    # if a saved file had the squeeze enabled, restore its working state
    try:
        _props_changed(bpy.context)
    except Exception:
        pass


def unregister():
    _stop_timer()
    try:
        bpy.app.handlers.depsgraph_update_post.remove(_depsgraph_dirty_cb)
    except Exception:
        pass
    _unregister_keymap()
    for h in _handles:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(h, 'WINDOW')
        except Exception:
            pass
    _handles.clear()
    _free_offscreens()
    _restore_annotation_hiding()
    if hasattr(bpy.types.Scene, "viewport_squeeze"):
        del bpy.types.Scene.viewport_squeeze
    for cls in reversed(_classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass


if __name__ == "__main__":
    register()
