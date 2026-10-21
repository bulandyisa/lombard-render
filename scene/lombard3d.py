# «Ломбард» в 3D (10 с): ночная улица под дождём → внутри ломбарда. Окружение — настоящая 3D-геометрия
# (EEVEE, отражения в мокрой брусчатке, дымка, глубина резкости), персонажи — нарисованные «картонные фигурки» (HD-2D).
# Сборка: Blender -b --python lombard/3d/lombard3d.py   → lombard/3d/lombard3d.blend
import json
import math
import os
import random

import bpy
import bmesh
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "tex")
FONT = os.environ.get("LOMBARD_FONT", "/System/Library/Fonts/Supplemental/Arial.ttf")
FPS, A_END, B_START, END = 24, 108, 109, 240
rnd = random.Random(7)

for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob)
sc = bpy.context.scene
sc.frame_start, sc.frame_end = 1, END
sc.render.fps = FPS


def srgb(h):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in c) + (1.0,)


def img(name):
    return bpy.data.images.load(os.path.join(TEX, name), check_existing=True)


# ---------- материалы ----------
def mat_basic(name, color, rough=0.5, metal=0.0, emit=None, emit_strength=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = srgb(color) if isinstance(color, str) else color
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    if emit:
        p.inputs["Emission Color"].default_value = srgb(emit)
        p.inputs["Emission Strength"].default_value = emit_strength
    return m


def coords(nt, mode="xz", scale=1.0):
    """Координаты объекта по выбранной плоскости (для стен — x,z; для пола — x,y)."""
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    a, b = mode
    nt.links.new(sep.outputs[a.upper()], comb.inputs[0])
    nt.links.new(sep.outputs[b.upper()], comb.inputs[1])
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (scale, scale, scale)
    nt.links.new(comb.outputs[0], mp.inputs[0])
    return mp.outputs[0]


def mat_brick():
    m = bpy.data.materials.new("brick")
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    br = nt.nodes.new("ShaderNodeTexBrick")
    br.inputs["Color1"].default_value = srgb("#3A1A12")
    br.inputs["Color2"].default_value = srgb("#2A120D")
    br.inputs["Mortar"].default_value = srgb("#15100E")
    br.inputs["Scale"].default_value = 1.0
    br.inputs["Mortar Size"].default_value = 0.012
    br.inputs["Brick Width"].default_value = 0.25
    br.inputs["Row Height"].default_value = 0.075
    v = coords(nt, "xz", 1.0)
    nt.links.new(v, br.inputs["Vector"])
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 3.0
    nt.links.new(v, nz.inputs["Vector"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 0.5
    nt.links.new(br.outputs["Color"], mix.inputs[6])
    nt.links.new(nz.outputs["Color"], mix.inputs[7])
    nt.links.new(mix.outputs[2], p.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    nt.links.new(br.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], p.inputs["Normal"])
    p.inputs["Roughness"].default_value = 0.55                      # мокрый кирпич
    return m


def mat_cobble():
    m = bpy.data.materials.new("cobble")
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    v = coords(nt, "xy", 1.0)
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.feature = "DISTANCE_TO_EDGE"
    vor.inputs["Scale"].default_value = 5.0
    nt.links.new(v, vor.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[1].position = 0.08
    nt.links.new(vor.outputs["Distance"], ramp.inputs[0])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.45
    nt.links.new(ramp.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], p.inputs["Normal"])
    # лужи: шум → маска; в лужах почти зеркально и темнее
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 0.35
    nz.inputs["Detail"].default_value = 3.0
    nt.links.new(v, nz.inputs["Vector"])
    pr = nt.nodes.new("ShaderNodeValToRGB")
    pr.color_ramp.elements[0].position = 0.46
    pr.color_ramp.elements[1].position = 0.52
    nt.links.new(nz.outputs["Fac"], pr.inputs[0])
    rough = nt.nodes.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value = 0.32
    rough.inputs["To Max"].default_value = 0.02
    nt.links.new(pr.outputs[0], rough.inputs["Value"])
    nt.links.new(rough.outputs[0], p.inputs["Roughness"])
    col = nt.nodes.new("ShaderNodeMix")
    col.data_type = "RGBA"
    col.inputs[6].default_value = srgb("#1C1A1A")
    col.inputs[7].default_value = srgb("#08080A")
    nt.links.new(pr.outputs[0], col.inputs["Factor"])
    nt.links.new(col.outputs[2], p.inputs["Base Color"])
    # в луже рельеф камней пропадает
    inv = nt.nodes.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    nt.links.new(pr.outputs[0], inv.inputs[1])
    nt.links.new(inv.outputs[0], bump.inputs["Strength"])
    return m


def mat_planks(name, c1, c2, rough=0.35, scale=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    br = nt.nodes.new("ShaderNodeTexBrick")
    br.inputs["Color1"].default_value = srgb(c1)
    br.inputs["Color2"].default_value = srgb(c2)
    br.inputs["Mortar"].default_value = srgb("#0A0605")
    br.inputs["Mortar Size"].default_value = 0.004
    br.inputs["Brick Width"].default_value = 1.6
    br.inputs["Row Height"].default_value = 0.14
    br.offset = 0.37
    v = coords(nt, "yx", scale)
    nt.links.new(v, br.inputs["Vector"])
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.inputs["Scale"].default_value = 6.0
    wave.inputs["Distortion"].default_value = 8.0
    nt.links.new(v, wave.inputs["Vector"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 0.35
    nt.links.new(br.outputs["Color"], mix.inputs[6])
    nt.links.new(wave.outputs["Color"], mix.inputs[7])
    nt.links.new(mix.outputs[2], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = rough
    return m


def mat_image(name, image, emission=1.0, alpha=True, circle=False, flicker=None, mult=None):
    """Картинка без освещения (Emission) с прозрачностью; circle — мягкий круг (сцена в банке)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.surface_render_method = "DITHERED"
    nt = m.node_tree
    for nd in list(nt.nodes):
        if nd.type == "BSDF_PRINCIPLED":
            nt.nodes.remove(nd)
    out = nt.nodes["Material Output"]
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.extension = "CLIP"
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    strength = em.inputs["Strength"]
    strength.default_value = emission
    if flicker is not None:                      # мерцание: (сила, частота) от кадра и случайного числа объекта
        info = nt.nodes.new("ShaderNodeObjectInfo")
        fr = nt.nodes.new("ShaderNodeValue")
        d = fr.outputs[0].driver_add("default_value")
        d.driver.expression = "frame"
        mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = flicker[1]
        nt.links.new(fr.outputs[0], mul.inputs[0])
        ph = nt.nodes.new("ShaderNodeMath"); ph.operation = "MULTIPLY"; ph.inputs[1].default_value = 40.0
        nt.links.new(info.outputs["Random"], ph.inputs[0])
        add = nt.nodes.new("ShaderNodeMath"); add.operation = "ADD"
        nt.links.new(mul.outputs[0], add.inputs[0]); nt.links.new(ph.outputs[0], add.inputs[1])
        sn = nt.nodes.new("ShaderNodeMath"); sn.operation = "SINE"
        nt.links.new(add.outputs[0], sn.inputs[0])
        sc_ = nt.nodes.new("ShaderNodeMath"); sc_.operation = "MULTIPLY_ADD"
        sc_.inputs[1].default_value = flicker[0] * emission; sc_.inputs[2].default_value = emission
        nt.links.new(sn.outputs[0], sc_.inputs[0])
        nt.links.new(sc_.outputs[0], strength)
    a_out = tex.outputs["Alpha"]
    if circle:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        grad = nt.nodes.new("ShaderNodeTexGradient"); grad.gradient_type = "SPHERICAL"
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Location"].default_value = (-0.5, -0.5, 0)
        mp.inputs["Scale"].default_value = (2.0, 2.0, 1)
        nt.links.new(tc.outputs["UV"], mp.inputs[0])
        mp2 = nt.nodes.new("ShaderNodeVectorMath"); mp2.operation = "SUBTRACT"; mp2.inputs[1].default_value = (0.5, 0.5, 0)
        nt.links.new(tc.outputs["UV"], mp2.inputs[0])
        ln = nt.nodes.new("ShaderNodeVectorMath"); ln.operation = "LENGTH"
        nt.links.new(mp2.outputs[0], ln.inputs[0])
        rp = nt.nodes.new("ShaderNodeMapRange")
        rp.inputs["From Min"].default_value = 0.5; rp.inputs["From Max"].default_value = 0.38
        nt.links.new(ln.outputs["Value"], rp.inputs["Value"])
        a_out = rp.outputs[0]
    if alpha:
        mix = nt.nodes.new("ShaderNodeMixShader")
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        gt = nt.nodes.new("ShaderNodeMath"); gt.operation = "GREATER_THAN"; gt.inputs[1].default_value = 0.5
        if circle:
            nt.links.new(a_out, mix.inputs[0])            # мягкий край круга — без порога
        else:
            nt.links.new(a_out, gt.inputs[0]); nt.links.new(gt.outputs[0], mix.inputs[0])
        nt.links.new(tr.outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    else:
        nt.links.new(em.outputs[0], out.inputs["Surface"])
    return m


def mat_glass():
    m = bpy.data.materials.new("glass")
    m.use_nodes = True
    m.surface_render_method = "BLENDED"
    m.use_backface_culling = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = srgb("#7F8A80")
    p.inputs["Roughness"].default_value = 0.04
    p.inputs["Alpha"].default_value = 0.05
    p.inputs["Specular IOR Level"].default_value = 0.55
    return m


def mat_neon(name, color, t_on, strength=12.0):
    """Неон: горит с кадра t_on (свойство объекта), первый раз мигает; общий «дребезг» — ключи power."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for nd in list(nt.nodes):
        if nd.type == "BSDF_PRINCIPLED":
            nt.nodes.remove(nd)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = srgb(color)
    at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_type = "OBJECT"; at.attribute_name = "t_on"
    fr = nt.nodes.new("ShaderNodeValue"); fr.label = "frame"
    d = fr.outputs[0].driver_add("default_value"); d.driver.expression = "frame"
    dt = nt.nodes.new("ShaderNodeMath"); dt.operation = "SUBTRACT"
    nt.links.new(fr.outputs[0], dt.inputs[0]); nt.links.new(at.outputs["Fac"], dt.inputs[1])
    on = nt.nodes.new("ShaderNodeMath"); on.operation = "GREATER_THAN"; on.inputs[1].default_value = 0.0
    nt.links.new(dt.outputs[0], on.inputs[0])
    b1 = nt.nodes.new("ShaderNodeMath"); b1.operation = "GREATER_THAN"; b1.inputs[1].default_value = 1.0
    b2 = nt.nodes.new("ShaderNodeMath"); b2.operation = "LESS_THAN"; b2.inputs[1].default_value = 2.0
    nt.links.new(dt.outputs[0], b1.inputs[0]); nt.links.new(dt.outputs[0], b2.inputs[0])
    blink = nt.nodes.new("ShaderNodeMath"); blink.operation = "MULTIPLY"
    nt.links.new(b1.outputs[0], blink.inputs[0]); nt.links.new(b2.outputs[0], blink.inputs[1])
    notb = nt.nodes.new("ShaderNodeMath"); notb.operation = "SUBTRACT"; notb.inputs[0].default_value = 1.0
    nt.links.new(blink.outputs[0], notb.inputs[1])
    lit = nt.nodes.new("ShaderNodeMath"); lit.operation = "MULTIPLY"
    nt.links.new(on.outputs[0], lit.inputs[0]); nt.links.new(notb.outputs[0], lit.inputs[1])
    pw = nt.nodes.new("ShaderNodeValue"); pw.name = "power"; pw.outputs[0].default_value = 1.0
    for f, v in ((1, 1.0), (57, 1.0), (58, 0.0), (60, 0.0), (61, 1.0), (65, 1.0), (66, 0.35), (67, 1.0)):
        pw.outputs[0].default_value = v
        pw.outputs[0].keyframe_insert("default_value", frame=f)
    l2 = nt.nodes.new("ShaderNodeMath"); l2.operation = "MULTIPLY"
    nt.links.new(lit.outputs[0], l2.inputs[0]); nt.links.new(pw.outputs[0], l2.inputs[1])
    st = nt.nodes.new("ShaderNodeMath"); st.operation = "MULTIPLY"; st.inputs[1].default_value = strength
    nt.links.new(l2.outputs[0], st.inputs[0])
    off = nt.nodes.new("ShaderNodeMath"); off.operation = "MAXIMUM"; off.inputs[1].default_value = 0.15  # погасшая трубка чуть видна
    nt.links.new(st.outputs[0], off.inputs[0])
    nt.links.new(off.outputs[0], em.inputs["Strength"])
    nt.links.new(em.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    return m


# ---------- примитивы ----------
def link(ob, col=None):
    (col or sc.collection).objects.link(ob)
    return ob


def box(name, x0, x1, y0, y1, z0, z1, mat):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(me)
    bm.free()
    ob = link(bpy.data.objects.new(name, me))
    ob.scale = (x1 - x0, y1 - y0, z1 - z0)
    ob.location = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    me.materials.append(mat)
    return ob


def cyl(name, r, h, loc, mat, verts=24, rot=(0, 0, 0), r2=None):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=verts, radius1=r, radius2=r if r2 is None else r2, depth=h)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    ob = link(bpy.data.objects.new(name, me))
    ob.location = loc
    ob.rotation_euler = rot
    me.materials.append(mat)
    return ob


def plane(name, w, h, loc, mat, rot=(math.radians(90), 0, 0), origin="center"):
    """Плоскость w×h; по умолчанию стоит вертикально лицом к −Y. origin='bottom' — опора внизу."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    oy = 0 if origin == "bottom" else -h / 2
    vs = [bm.verts.new((x, y, 0)) for x, y in ((-w / 2, oy), (w / 2, oy), (w / 2, oy + h), (-w / 2, oy + h))]
    f = bm.faces.new(vs)
    for lp, (u, v) in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
        lp[uv].uv = (u, v)
    bm.to_mesh(me)
    bm.free()
    ob = link(bpy.data.objects.new(name, me))
    ob.location = loc
    ob.rotation_euler = rot
    me.materials.append(mat)
    return ob


ANCHORS = json.load(open(os.path.join(TEX, "anchors.json"))) if os.path.exists(os.path.join(TEX, "anchors.json")) else {}


def card(name, image_name, height, loc, track_cam, emission=1.0):
    """Нарисованный персонаж: вертикальная карточка с опорой внизу (под центром ног), всегда лицом к камере (Locked Track)."""
    im = img(image_name)
    w = height * im.size[0] / im.size[1]
    ox = (0.5 - ANCHORS.get(image_name, 0.5)) * w                 # сдвиг, чтобы ноги стояли в точке опоры
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    vs = [bm.verts.new((x + ox, 0, z)) for x, z in ((-w / 2, 0), (w / 2, 0), (w / 2, height), (-w / 2, height))]
    f = bm.faces.new(vs)
    for lp, (u, v) in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
        lp[uv].uv = (u, v)
    bm.to_mesh(me)
    bm.free()
    ob = link(bpy.data.objects.new(name, me))
    ob.location = loc
    me.materials.append(mat_image(name, im, emission))
    c = ob.constraints.new("LOCKED_TRACK")
    c.target = track_cam
    c.track_axis = "TRACK_NEGATIVE_Y"
    c.lock_axis = "LOCK_Z"
    ob.visible_shadow = False
    return ob


def pose_set(prefix, files, height, cam, emission=1.0, mirror=()):
    """Набор рисунков-поз одного персонажа на общем корне. Высота каждой — по высоте фигуры относительно первой позы;
    mirror — позы, для которых нужна ещё и зеркальная копия (шаг другой ногой), имя копии = файл + «~»."""
    root = link(bpy.data.objects.new(prefix + "_root", None))
    h0 = img(files[0]).size[1]
    cards = {}
    for f in files:
        for flip in ((False, True) if f in mirror else (False,)):
            c = card(prefix + "_" + f + ("~" if flip else ""), f, height * img(f).size[1] / h0, (0, 0, 0), cam, emission)
            c.parent = root
            if flip:
                c.scale = (-1, 1, 1)
            cards[f + ("~" if flip else "")] = c
    return {"root": root, "cards": cards}


def show_schedule(ps, schedule):
    """schedule = [(кадр, поза)]: в каждый момент видна одна поза (смена — шагом, как в ограниченной анимации)."""
    for name, c in ps["cards"].items():
        for f, cur in schedule:
            c.hide_render = cur != name
            c.keyframe_insert("hide_render", frame=f)
            c.hide_viewport = cur != name
            c.keyframe_insert("hide_viewport", frame=f)


def breathe(ob, f0, f1, amp, period):
    """Дыхание корня персонажа: по высоте сильнее, чем по ширине, вдох медленнее выдоха."""
    for f in range(f0, f1 + 1, 4):
        u = ((f - f0) / FPS / period) % 1.0
        b = math.sin(math.pi * u) ** 2
        ob.scale = (1 + amp * 0.35 * b, 1, 1 + amp * b)
        ob.keyframe_insert("scale", frame=f)


def light(name, kind, loc, energy, color, size=0.2, rot=(0, 0, 0), spot=None):
    ld = bpy.data.lights.new(name, kind)
    ld.energy = energy
    ld.color = srgb(color)[:3]
    if kind in ("POINT", "SPOT"):
        ld.shadow_soft_size = size
    if kind == "AREA":
        ld.size = size
    if spot:
        ld.spot_size = math.radians(spot)
        ld.spot_blend = 0.6
    ob = link(bpy.data.objects.new(name, ld))
    ob.location = loc
    ob.rotation_euler = rot
    return ob


# ---------- мир и рендер ----------
w = bpy.data.worlds.new("night")
sc.world = w
w.use_nodes = True
w.node_tree.nodes["Background"].inputs["Color"].default_value = srgb("#18202E")
w.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0

sc.render.engine = "BLENDER_EEVEE"
sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
sc.view_settings.view_transform = "Standard"
sc.render.filter_size = 1.1
ee = sc.eevee
for attr, val in (("taa_render_samples", 64), ("use_raytracing", True), ("use_shadows", True), ("shadow_ray_count", 2),
                  ("use_fast_gi", True), ("volumetric_tile_size", "8"), ("volumetric_samples", 48), ("volumetric_end", 14.0)):
    try:
        setattr(ee, attr, val)
    except Exception as e:
        print("skip", attr, e)
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGB"

# ---------- камеры ----------
def camera(name, lens, fstop):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.dof.use_dof = True
    cd.dof.aperture_fstop = fstop
    cd.clip_start = 0.05
    cam = link(bpy.data.objects.new(name, cd))
    tgt = link(bpy.data.objects.new(name + "_target", None))
    c = cam.constraints.new("TRACK_TO")
    c.target = tgt
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"
    return cam, tgt


def keys(ob, path, frames_vals):
    for f, v in frames_vals:
        setattr(ob, path, v)
        ob.keyframe_insert(path, frame=f)


camA, tgtA = camera("camA", 28, 2.8)
keys(camA, "location", [(1, Vector((-1.7, -8.2, 0.32))), (A_END, Vector((0.35, -4.9, 1.15)))])
keys(tgtA, "location", [(1, Vector((0.3, 0.0, 2.2))), (A_END, Vector((1.25, 0.0, 1.75)))])
keys(camA.data, "lens", [(1, 30.0), (A_END, 36.0)])
camB, tgtB = camera("camB", 26, 2.2)
keys(camB, "location", [(B_START, Vector((0.25, 0.9, 1.55))), (END, Vector((-0.1, 4.8, 1.48)))])
keys(tgtB, "location", [(B_START, Vector((0.1, 7.6, 1.4))), (END, Vector((0.05, 7.6, 1.55)))])
sc.camera = camA
for name, f, cam in (("A", 1, camA), ("B", B_START, camB)):
    mk = sc.timeline_markers.new(name, frame=f)
    mk.camera = cam

# ---------- улица ----------
brick = mat_brick()
dark_wood = mat_planks("dark_wood", "#2A1A10", "#22150D", rough=0.45, scale=1.0)
frame_wood = mat_basic("frame_wood", "#1B120C", rough=0.4)
brass = mat_basic("brass", "#B08D3A", rough=0.25, metal=1.0)
iron = mat_basic("iron", "#141414", rough=0.4, metal=0.6)
glass = mat_glass()

ground = plane("street", 30, 22, (0, -10.0, 0), mat_cobble(), rot=(0, 0, 0))
for nm, x0, x1, z0, z1 in (("wall_L", -14, -2.8, 0, 9), ("wall_R", 2.6, 14, 0, 9), ("wall_T", -2.8, 2.6, 3.05, 9),
                          ("wall_sill", -2.8, 0.7, 0, 0.72), ("wall_mid", 0.5, 0.8, 0.72, 3.05)):
    box(nm, x0, x1, 0.0, 0.4, z0, z1, brick)
# витрина и дверь
win = plane("window_glass", 3.1, 2.0, (-1.05, 0.05, 1.74), glass)
for nm, x0, x1, z0, z1 in (("wf_b", -2.75, 0.55, 0.68, 0.78), ("wf_t", -2.75, 0.55, 2.72, 2.82), ("wf_l", -2.75, -2.6, 0.7, 2.8),
                          ("wf_r", 0.42, 0.55, 0.7, 2.8), ("wf_m", -1.11, -0.99, 0.7, 2.8)):
    box(nm, x0, x1, -0.02, 0.08, z0, z1, frame_wood)
for nm, x0, x1, z0, z1 in (("df_l", 0.78, 0.86, 0, 2.65), ("df_r", 2.22, 2.3, 0, 2.65), ("df_t", 0.78, 2.3, 2.58, 2.68)):
    box(nm, x0, x1, -0.03, 0.1, z0, z1, frame_wood)
hinge = link(bpy.data.objects.new("door_hinge", None))
hinge.location = (2.22, 0.05, 0)
door_parts = [box("door_low", 0.86, 2.22, 0.02, 0.08, 0.0, 1.05, frame_wood), box("door_top", 0.86, 2.22, 0.02, 0.08, 2.35, 2.58, frame_wood),
              box("door_side1", 0.86, 0.98, 0.02, 0.08, 1.05, 2.35, frame_wood), box("door_side2", 2.1, 2.22, 0.02, 0.08, 1.05, 2.35, frame_wood),
              plane("door_glass", 1.12, 1.3, (1.54, 0.05, 1.7), glass), cyl("door_handle", 0.025, 0.32, (1.0, -0.02, 1.15), brass)]
for p_ in door_parts:
    p_.parent = hinge
    p_.location = p_.location - hinge.location
hinge.rotation_euler = (0, 0, 0)
for f, a in ((1, 0.0), (80, 0.0), (82, -3.0), (94, -38.0), (108, -52.0)):     # «−» — внутрь, от героя
    hinge.rotation_euler = (0, 0, math.radians(a))
    hinge.keyframe_insert("rotation_euler", frame=f)
# колокольчик над дверью — качается, когда дверь задевает его
bell_pivot = link(bpy.data.objects.new("bell_pivot", None))
bell_pivot.location = (1.2, -0.12, 2.95)
box("bell_bracket", 1.18, 1.22, -0.12, 0.02, 2.93, 2.97, iron)
bell = cyl("bell", 0.07, 0.11, (0, 0, -0.1), brass, r2=0.025)
bell.parent = bell_pivot
for f, a in ((1, 0), (83, 0), (86, 24), (90, -16), (94, 10), (98, -6), (102, 3), (106, 0)):
    bell_pivot.rotation_euler = (math.radians(a), 0, 0)
    bell_pivot.keyframe_insert("rotation_euler", frame=f)
# козырёк
awn_mat = mat_basic("awning", "#3B2B1E", rough=0.7)
awn = plane("awning", 5.6, 1.0, (-0.1, -0.45, 2.98), awn_mat, rot=(math.radians(-14), 0, 0))
# вывеска: тёмная подложка, неоновая рамка, буквы по одной
box("sign_board", -2.45, 2.25, -0.06, 0.0, 3.25, 4.35, iron)
cu = bpy.data.curves.new("neon_frame", "CURVE")
cu.dimensions = "3D"
cu.bevel_depth = 0.022
cu.bevel_resolution = 4
sp = cu.splines.new("POLY")
pts = []
x0, x1, z0, z1, r = -2.3, 2.1, 3.35, 4.25, 0.12
for cx, cz, a0 in ((x1 - r, z1 - r, 0), (x0 + r, z1 - r, 90), (x0 + r, z0 + r, 180), (x1 - r, z0 + r, 270)):
    for k in range(7):
        a = math.radians(a0 + k * 15)
        pts.append((cx + r * math.cos(a), -0.09, cz + r * math.sin(a)))
sp.points.add(len(pts) - 1)
for p_, (x, y, z) in zip(sp.points, pts):
    p_.co = (x, y, z, 1)
sp.use_cyclic_u = True
frame_ob = link(bpy.data.objects.new("neon_frame", cu))
frame_ob["t_on"] = 3.0
cu.materials.append(mat_neon("neon_frame_mat", "#FFB46A", 0, strength=9.0))
font = bpy.data.fonts.load(FONT, check_existing=True)
neon_mat = mat_neon("neon_text_mat", "#FFC27A", 0, strength=11.0)
dg = bpy.context.evaluated_depsgraph_get()
letter_t = 6.0
for line, zc, size in (("СКУПКА ВОСПОМИНАНИЙ", 3.92, 0.36), ("· НАЛИЧНЫЕ ·", 3.52, 0.3)):
    tc = bpy.data.curves.new("txt", "FONT")
    tc.body = line
    tc.font = font
    tc.size = size
    tc.align_x = "CENTER"
    tc.align_y = "CENTER"
    tc.extrude = 0.008
    tc.bevel_depth = 0.004
    tob = link(bpy.data.objects.new("txt", tc))
    tob.rotation_euler = (math.radians(90), 0, 0)
    tob.location = (-0.1, -0.09, zc)
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(tob.evaluated_get(dg))
    me.transform(tob.matrix_world)
    bpy.data.objects.remove(tob)
    # буквы = связные куски меша; кусок, лежащий над/под соседним (краткая у «Й»), — в ту же букву
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    seen, parts = set(), []
    for v in bm.verts:
        if v.index in seen:
            continue
        stack, comp = [v], []
        seen.add(v.index)
        while stack:
            u = stack.pop()
            comp.append(u.index)
            for e in u.link_edges:
                o = e.other_vert(u)
                if o.index not in seen:
                    seen.add(o.index)
                    stack.append(o)
        xs = [bm.verts[i].co.x for i in comp]
        parts.append((min(xs), max(xs), comp))
    bm.free()
    parts.sort(key=lambda p: (p[0] + p[1]) / 2)
    letters = []
    for p_ in parts:
        cx = (p_[0] + p_[1]) / 2
        if letters and letters[-1][0] - 0.01 <= cx <= letters[-1][1] + 0.01:
            letters[-1] = (min(letters[-1][0], p_[0]), max(letters[-1][1], p_[1]), letters[-1][2] + p_[2])
        else:
            letters.append(p_)
    for lx0, lx1, comp in letters:
        keep = set(comp)
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.index not in keep], context="VERTS")
        lm = bpy.data.meshes.new("letter")
        bm.to_mesh(lm)
        bm.free()
        lm.materials.append(neon_mat)
        lob = link(bpy.data.objects.new("letter", lm))
        lob["t_on"] = letter_t
        letter_t += 1.3
    bpy.data.meshes.remove(me)
print("letters until frame", letter_t)
# свет вывески — на брусчатку и стену
light("neon_light", "AREA", (-0.1, -0.7, 3.7), 900, "#FFB060", size=4.0, rot=(math.radians(-60), 0, 0))
# фонарь
cyl("lamp_pole", 0.055, 3.7, (-3.9, -1.9, 1.85), iron)
cyl("lamp_head", 0.16, 0.42, (-3.9, -1.9, 3.85), mat_basic("lamp_glass", "#FFD9A0", emit="#FFC880", emit_strength=8.0), verts=6).visible_shadow = False
light("lamp_light", "POINT", (-3.9, -1.9, 3.75), 700, "#FFC27A", size=0.15)
light("street_fill", "AREA", (1.0, -6.0, 5.0), 400, "#5A6E96", size=8.0, rot=(math.radians(-35), 0, 0))
# окна верхних этажей
for i, (x, z) in enumerate(()):
    lit = rnd.random() < 0.6
    plane(f"upper_win{i}", 1.1, 1.4, (x, -0.01, z), mat_basic(f"uw{i}", "#0E0D0C", rough=0.2,
          emit="#E8A35C" if lit else "#1A2230", emit_strength=0.8 if lit else 0.2))
# дальний план улицы (нарисованная картинка) слева
plane("street_far", 14, 9.3, (-13.5, -6, 4.3), mat_image("street_far", img("street_l04.png"), emission=0.55, alpha=False),
      rot=(math.radians(90), 0, math.radians(-90)))
# герой со спины у двери
HERO_POSES = pose_set("hero", ["hero_p1.png", "hero_p2.png", "hero_p3.png"], 1.86, camA, emission=0.72,
                      mirror={"hero_p2.png"})
hero = HERO_POSES["root"]
# стоит под дождём и смотрит на вывеску → два шага к двери (16 к на шаг, вверх на проходе) → тянется и толкает дверь → входит
show_schedule(HERO_POSES, [(1, "hero_p1.png"), (40, "hero_p2.png"), (48, "hero_p1.png"), (56, "hero_p2.png~"), (64, "hero_p1.png"),
                           (74, "hero_p3.png"), (98, "hero_p2.png"), (106, "hero_p1.png")])
for f, y, z in ((1, -1.4, 0.0), (40, -1.4, 0.0), (44, -1.26, -0.03), (48, -1.12, 0.012), (52, -0.98, 0.0), (56, -0.86, -0.03),
                (60, -0.74, 0.012), (64, -0.64, 0.0), (72, -0.58, 0.0), (80, -0.54, 0.0), (94, -0.36, 0.0), (102, -0.05, -0.02),
                (108, 0.22, 0.0)):
    hero.location = (1.35, y, z)
    hero.keyframe_insert("location", frame=f)
for f, ang in ((1, 0.0), (74, 0.0), (80, 2.5), (90, 4.0), (98, 1.0), (108, 0.0)):        # наклон вперёд при толчке
    hero.rotation_euler = (math.radians(ang), 0, 0)
    hero.keyframe_insert("rotation_euler", frame=f)
breathe(hero, 1, 40, 0.008, 3.4)

# ---------- ломбард ----------
IN = dict(x0=-3.0, x1=3.0, y0=0.4, y1=9.0, h=4.0)
plane("floor", 4.8, 8.6, (0, 4.7, 0.001), mat_planks("floor", "#3A2516", "#2C1B10", rough=0.3), rot=(0, 0, 0))
plane("ceiling", 4.8, 8.6, (0, 4.7, IN["h"]), mat_basic("ceil", "#0D0B0A", rough=0.9), rot=(math.radians(180), 0, 0))
wallp = mat_basic("wallpaper", "#141A26", rough=0.8)
plane("back_wall", 4.8, 4.0, (0, 9.0, 2.0), wallp)
plane("side_L", 8.6, 4.0, (-2.4, 4.7, 2.0), wallp, rot=(math.radians(90), 0, math.radians(-90)))
plane("side_R", 8.6, 4.0, (2.4, 4.7, 2.0), wallp, rot=(math.radians(90), 0, math.radians(90)))
plane("carpet", 2.2, 3.4, (0, 4.6, 0.004), mat_image("carpet", img("carpet.png"), emission=0.25, alpha=False), rot=(0, 0, math.radians(90)))
# стеллажи с банками
mem_imgs = ["mem_subway.png", "mem_sea.png", "mem_star.png", "mem_wedding.png", "mem_snow.png", "mem_kid.png", "mem_cake.png"]
mem_mats = [mat_image(f"mem{i}", img(n), emission=2.4, circle=True, flicker=(0.18, 0.21 + 0.03 * i)) for i, n in enumerate(mem_imgs)]
lid_mat = mat_basic("lid", "#8C6A2A", rough=0.3, metal=1.0)
camB_target = camB
n_jars = 0
for side in (-1, 1):
    xs = side * 2.08
    for y in (1.0, 2.2, 3.4, 4.6, 5.8):
        box(f"upright_{side}_{y}", xs - 0.25, xs + 0.25, y - 0.03, y + 0.03, 0, 3.75, dark_wood)
    for zi, z in enumerate((0.45, 1.1, 1.75, 2.4, 3.05)):
        box(f"shelf_{side}_{zi}", xs - 0.25, xs + 0.25, 1.0, 5.8, z - 0.035, z, dark_wood)
        y = 1.15
        while y < 5.65:
            r = rnd.uniform(0.075, 0.13)
            h = rnd.uniform(0.17, 0.3)
            if y + r > 5.75:
                break
            cx = xs - side * rnd.uniform(0.0, 0.06)
            cyl("jar", r, h, (cx, y, z + h / 2), glass, verts=20)
            cyl("lid", r * 0.86, 0.035, (cx, y, z + h + 0.017), lid_mat, verts=20)
            if rnd.random() < 0.82:
                s = min(2 * r, h) * 0.9
                mc = plane("memory", s, s, (cx, y, z + h * 0.48), mem_mats[rnd.randrange(len(mem_mats))])
                c = mc.constraints.new("TRACK_TO")
                c.target = camB
                c.track_axis = "TRACK_Z"
                c.up_axis = "UP_Y"
                mc.visible_shadow = False
                n_jars += 1
            y += 2 * r + rnd.uniform(0.04, 0.12)
    # тёплый свет от банок на проход
    light(f"shelf_glow_{side}", "AREA", (side * 1.25, 3.4, 1.8), 260, "#FFA855", size=4.5, rot=(0, math.radians(-90 * side), 0))
print("jars with memories", n_jars)
# прилавок, лампа, весы-камертон
box("counter", -1.4, 1.4, 6.6, 7.2, 0, 1.0, dark_wood)
box("counter_top", -1.5, 1.5, 6.55, 7.25, 1.0, 1.06, mat_planks("counter_top", "#3B2414", "#311D10", rough=0.18))
cyl("fork_base", 0.07, 0.04, (0.9, 6.85, 1.08), brass)
cyl("fork_stem", 0.012, 0.18, (0.9, 6.85, 1.19), brass)
for dx in (-0.035, 0.035):
    cyl("fork_prong", 0.009, 0.22, (0.9 + dx, 6.85, 1.38), brass)
cyl("lamp_shade", 0.32, 0.22, (0.0, 6.9, 2.85), mat_basic("shade", "#2A3A2A", rough=0.4, metal=0.3), r2=0.07)
cyl("lamp_bulb", 0.06, 0.08, (0.0, 6.9, 2.73), mat_basic("bulb", "#FFE2B0", emit="#FFD08A", emit_strength=30.0)).visible_shadow = False
cyl("lamp_cord", 0.006, 1.1, (0.0, 6.9, 3.5), iron)
light("counter_spot", "SPOT", (0.0, 6.9, 2.66), 520, "#FFC788", size=0.08, rot=(0, 0, 0), spot=62)
light("old_key", "AREA", (0.0, 5.6, 2.2), 120, "#FFC08A", size=1.2, rot=(math.radians(-60), 0, 0))
# задняя стена: часы, маятник, зеркало, рисунок
clock_mat = mat_image("clock", img("clock.png"), emission=0.6, alpha=False)
for i, (x, z, r) in enumerate(((-1.75, 3.1, 0.22), (-1.25, 2.55, 0.16), (1.15, 3.25, 0.2), (1.9, 2.6, 0.24), (1.75, 1.75, 0.14), (-1.95, 2.15, 0.13))):
    cyl(f"clock_rim{i}", r * 1.08, 0.05, (x, 8.97, z), brass, rot=(math.radians(90), 0, 0))
    plane(f"clock_face{i}", 2 * r, 2 * r, (x, 8.94, z), clock_mat)
box("pendulum_case", 1.05, 1.55, 8.75, 8.98, 0.6, 3.0, frame_wood)
pend = link(bpy.data.objects.new("pendulum", None))
pend.location = (1.3, 8.72, 2.6)
pr = cyl("pend_rod", 0.012, 1.1, (0, 0, -0.55), brass)
pd = cyl("pend_disc", 0.09, 0.02, (0, 0, -1.1), brass, rot=(math.radians(90), 0, 0))
for o in (pr, pd):
    o.parent = pend
dr = pend.driver_add("rotation_euler", 1)
dr.driver.expression = "sin(frame*0.1309)*0.22"
mirror = cyl("mirror", 0.45, 0.02, (-0.6, 8.96, 2.1), mat_basic("mirror", "#5A5E64", rough=0.12, metal=1.0), verts=48,
             rot=(math.radians(90), 0, 0))
mirror.scale = (1.0, 1.45, 1.0)
cyl("mirror_frame", 0.5, 0.03, (-0.6, 8.985, 2.1), brass, verts=48, rot=(math.radians(90), 0, 0)).scale = (1.0, 1.45, 1.0)
plane("drawing", 0.42, 0.54, (0.55, 8.96, 2.2), mat_image("drawing", img("drawing.png"), emission=0.7, alpha=False),
      rot=(math.radians(90), math.radians(-3), 0))
# старик за прилавком
OLD_POSES = pose_set("old", ["old_p1.png", "old_p2.png", "old_p3.png"], 1.74, camB, emission=0.9)
old = OLD_POSES["root"]
old.location = (0.05, 7.55, 0.08)                     # руки — на уровне прилавка
show_schedule(OLD_POSES, [(B_START, "old_p1.png"), (150, "old_p2.png"), (192, "old_p3.png")])
for f, z in ((B_START, 0.08), (146, 0.08), (149, 0.065), (153, 0.09), (158, 0.08), (188, 0.08), (191, 0.068), (195, 0.088), (200, 0.08)):
    old.location.z = z                                                       # антиципация вниз → подъём при смене позы
    old.keyframe_insert("location", index=2, frame=f)
breathe(old, B_START, END, 0.009, 3.8)
# дымка в ломбарде
vol = box("haze", -2.35, 2.35, 0.45, 8.95, 0.02, 3.95, bpy.data.materials.new("haze"))
vm = vol.data.materials[0]
vm.use_nodes = True
nt = vm.node_tree
for nd in list(nt.nodes):
    if nd.type == "BSDF_PRINCIPLED":
        nt.nodes.remove(nd)
pv = nt.nodes.new("ShaderNodeVolumePrincipled")
pv.inputs["Density"].default_value = 0.0035
pv.inputs["Anisotropy"].default_value = 0.55
pv.inputs["Color"].default_value = srgb("#D8C0A0")
nt.links.new(pv.outputs[0], nt.nodes["Material Output"].inputs["Volume"])
vol.visible_shadow = False
light("fill_in", "AREA", (0, 3.0, 3.8), 90, "#6F86B0", size=5.0, rot=(math.radians(0), 0, 0))

out = os.path.join(HERE, "lombard3d.blend")
bpy.ops.wm.save_as_mainfile(filepath=out)
print("SAVED", out)
