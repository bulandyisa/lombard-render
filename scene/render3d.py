# Рендер кадров сцены «Ломбард 3D» в PNG:
# Blender -b lombard/3d/lombard3d.blend --python lombard/3d/render3d.py -- <папка> <ширина> <сэмплы> <кадр> [кадр…|с-по]
import os
import sys
import time

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
out, width, samples = args[0], int(args[1]), int(args[2])
frames = []
for a in args[3:]:
    if "-" in a:
        f0, f1 = map(int, a.split("-"))
        frames += list(range(f0, f1 + 1))
    else:
        frames.append(int(a))
sc = bpy.context.scene
sc.render.resolution_x, sc.render.resolution_y = width, width * 9 // 16
sc.eevee.taa_render_samples = samples
if os.environ.get("LOMBARD_ENGINE", "").upper() == "CYCLES":     # облако без видеокарты: Cycles на процессоре
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.volume_step_rate = 2.0
    sc.cycles.max_bounces = 6
print("ENGINE", sc.render.engine, flush=True)
os.makedirs(out, exist_ok=True)
for f in frames:
    dst = os.path.join(out, f"f_{f:04d}.png")
    if os.path.exists(dst):
        continue
    t = time.time()
    sc.frame_set(f)
    sc.render.filepath = dst
    bpy.ops.render.render(write_still=True)
    print(f"FRAME {f} {time.time() - t:.1f}s", flush=True)
print("RENDER DONE")
