# QC 3D-сцены без рендера: по всем кадрам проверяет пересечения подвижных объектов с карточками персонажей
# и друг с другом, выход персонажей за пол, попадание в кадр. Запуск:
# Blender -b сцена.blend --python lombard/3d/qc3d.py -- [кадр_с кадр_по]
import sys

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
sc = bpy.context.scene
f0, f1 = (int(args[0]), int(args[1])) if len(args) >= 2 else (sc.frame_start, sc.frame_end)


def animated(ob):
    """Объект движется сам или через родителя/констрейнт."""
    o = ob
    while o:
        if (o.animation_data and o.animation_data.action) or o.constraints:
            return True
        o = o.parent
    return False


def bvh(ob, dg):
    """BVH в МИРОВЫХ координатах (FromObject строит в локальных — для детей и констрейнтов это неверно)."""
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    if me is None or not me.polygons:
        if me is not None:
            ev.to_mesh_clear()
        return None
    mw = ev.matrix_world
    verts = [mw @ v.co for v in me.vertices]
    polys = [tuple(p.vertices) for p in me.polygons]
    ev.to_mesh_clear()
    return BVHTree.FromPolygons(verts, polys)


def bbox(ob):
    return [ob.matrix_world @ Vector(c) for c in ob.bound_box]


meshes = [o for o in sc.objects if o.type == "MESH" and not o.hide_render]
cards = [o for o in meshes if any(c.type == "LOCKED_TRACK" for c in o.constraints)]
movers = [o for o in meshes if animated(o) and o not in cards and not o.name.startswith(("haze", "memory")) and not o.name.endswith("_shadow")]
static_big = [o for o in meshes if not animated(o) and o.name.startswith(("wall", "door", "counter", "shelf", "upright", "floor"))]
print(f"QC3D: карточек {len(cards)}, подвижных {len(movers)}, статики {len(static_big)}, кадры {f0}–{f1}")

problems = []
for f in range(f0, f1 + 1):
    sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    card_trees = {c.name: bvh(c, dg) for c in cards if not c.hide_render}
    for m in movers:
        tm = bvh(m, dg)
        if tm is None:
            continue
        for cn, ct in card_trees.items():
            if ct and tm.overlap(ct):
                problems.append((f, f"{m.name} пересекает {cn}"))
        for s in static_big:
            ts = bvh(s, dg)
            if ts and tm.overlap(ts) and not s.name.startswith("door"):
                problems.append((f, f"{m.name} пересекает {s.name}"))
    for c in cards:
        if c.hide_render:
            continue
        zs = [v.z for v in bbox(c)]
        if min(zs) < -0.02:
            problems.append((f, f"{c.name} ниже пола на {-min(zs):.2f} м"))
        # карточка внутри стены? центр карточки vs. ограничивающие коробки крупной статики
        cen = c.matrix_world.translation
        for s in static_big:
            bb = bbox(s)
            xs, ys, zs2 = [v.x for v in bb], [v.y for v in bb], [v.z for v in bb]
            if min(xs) < cen.x < max(xs) and min(ys) < cen.y < max(ys) and min(zs2) < cen.z + 0.9 < max(zs2) and s.name.startswith(("wall", "counter")):
                problems.append((f, f"{c.name} внутри {s.name}"))

# сжимаем в диапазоны
report = {}
for f, msg in problems:
    report.setdefault(msg, []).append(f)
if not report:
    print("QC3D: пересечений нет")
for msg, fr in report.items():
    runs, a, prev = [], fr[0], fr[0]
    for x in fr[1:]:
        if x != prev + 1:
            runs.append((a, prev))
            a = x
        prev = x
    runs.append((a, prev))
    print("QC3D ПРОБЛЕМА:", msg, "кадры", ", ".join(f"{a}–{b}" if a != b else str(a) for a, b in runs))
print("QC3D DONE")
