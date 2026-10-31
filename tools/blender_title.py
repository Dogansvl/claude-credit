"""Titre 3D « C'EST QUI LUI ? » : scène Blender (Cycles CPU), exécutée en mode headless.

Lancé par tools/render_title.py (ne pas appeler à la main sauf pour déboguer) :
    blender -b --factory-startup -P tools/blender_title.py -- --font F.ttf --out DIR --pass surf --frames 1-150

Rendu en DEUX passes (budget CPU : le brouillard volumétrique pleine définition coûterait ~10x trop cher) :
  --pass surf : surfaces seules (sans volume), pleine définition, peu d'échantillons. EXR multicouche :
                groupes de lumière (head / red / amb), Denoising Albedo/Normal (débruitage OIDN en post),
                Depth (guide du suréchantillonnage), IndexOB (masque du titre).
  --pass fog  : brouillard seul, basse définition, beaucoup d'échantillons. Toutes les surfaces deviennent
                des occultants émissifs blancs d'intensité 1 : la passe Emit = transmittance caméra->surface T,
                les groupes de lumière = diffusion du volume S par source (phares / feux rouges).
Recomposition (render_title.py) : image = Σ surf_g·T + Σ S_g + (1−T)·brume ; T et S sont suréchantillonnés
par un filtre bilatéral joint guidé par la profondeur pleine définition -> bords du titre nets.

Scène (mètres, Z vers le haut, la caméra regarde vers +Y) :
  - texte extrudé + biseau, matériau charbon / métal vieilli, debout à y=0 ;
  - sol d'asphalte mouillé (flaques miroir, lignes de places), mur de béton, piliers et poutres de parking ;
  - brouillard (Principled Volume, anisotropie 0,6, couche plus dense au sol, bruit qui dérive) ;
  - deux phares (spots) d'une voiture hors champ à gauche qui tourne : le faisceau balaie le titre de gauche
    à droite, puis la lueur rouge de feux arrière monte dans le brouillard, derrière le titre ;
  - lent travelling avant de la caméra (léger arc latéral pour la parallaxe).
--mode cover : image fixe de couverture (titre en haut, phares de face au fond = contre-jour pour la silhouette).
--layout OUT.json : aucun rendu ; écrit la boîte projetée du titre pour chaque frame demandée.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Vector

FPS = 30
N_FRAMES = 150
TEXT_BODY = "C’EST QUI\nLUI ?"   # apostrophe typographique, espace fine avant « ? »
HEAD_COLOR = (0.90, 0.95, 1.00)          # blanc chaud légèrement bleuté (LED / xénon)
RED = (1.0, 0.035, 0.015)
FOG_DENSITY = 0.05


# ----------------------------------------------------------------------------- arguments
def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="blender_title.py")
    p.add_argument("--font", required=True)
    p.add_argument("--out", help="dossier de sortie des EXR")
    p.add_argument("--pass", dest="pass_", choices=["surf", "text", "fog", "beauty"], default="surf")
    p.add_argument("--border-margin", type=int, default=30, help="passe text : marge (px) autour du titre")
    p.add_argument("--frames", default="1-150", help="'a-b' ou liste '1,40,80' (frames Blender, 1..150)")
    p.add_argument("--mode", choices=["seq", "cover"], default="seq")
    p.add_argument("--width", type=int, default=1080)
    p.add_argument("--height", type=int, default=1920)
    p.add_argument("--scale", type=int, default=100, help="pourcentage de résolution")
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--fog-mode", choices=["hetero", "homog"], default="hetero")
    p.add_argument("--step-rate", type=float, default=1.0, help="pas du ray-marching volumique (grand = rapide)")
    p.add_argument("--bounces", type=int, default=3)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--title-cx", type=float, default=480 / 1080, help="centre horizontal visé du titre (fraction)")
    p.add_argument("--title-cy", type=float, default=0.43, help="centre vertical visé du titre (fraction, depuis le haut)")
    p.add_argument("--layout", help="écrit la boîte projetée du titre par frame (JSON) sans rendre")
    p.add_argument("--save-blend", help="sauvegarde la scène (.blend) pour inspection")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args(argv)


def parse_frames(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


# ----------------------------------------------------------------------------- maths
def smoothstep(e0: float, e1: float, x: float) -> float:
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def pchip(xs: list[float], ys: list[float], x: float) -> float:
    """Interpolation cubique monotone (Fritsch-Carlson) : aucun dépassement entre les clés."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    n = len(xs)
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    d = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
    m = [0.0] * n
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    i = max(j for j in range(n - 1) if xs[j] <= x)
    t = (x - xs[i]) / h[i]
    h00, h10 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t
    h01, h11 = -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    return h00 * ys[i] + h10 * h[i] * m[i] + h01 * ys[i + 1] + h11 * h[i] * m[i + 1]


# ----------------------------------------------------------------------------- nœuds
class NodeBuilder:
    """Petit utilitaire pour câbler des nœuds de shader lisiblement."""

    def __init__(self, tree: bpy.types.NodeTree):
        self.tree, self.nodes, self.links = tree, tree.nodes, tree.links
        self.x = 0

    def add(self, kind: str, **inputs):
        n = self.nodes.new(kind)
        n.location = (self.x, 0)
        self.x += 220
        for k, v in inputs.items():
            if k.startswith("_"):
                setattr(n, k[1:], v)
            else:
                n.inputs[k].default_value = v
        return n

    def link(self, a, b):
        self.links.new(a, b)

    def math(self, op: str, a, b=None, c=None, clamp=False):
        n = self.add("ShaderNodeMath")
        n.operation = op
        n.use_clamp = clamp
        for idx, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[idx].default_value = v
            else:
                self.link(v, n.inputs[idx])
        return n.outputs[0]

    def ramp(self, fac, p0: float, p1: float):
        r = self.add("ShaderNodeValToRGB")
        r.color_ramp.elements[0].position = p0
        r.color_ramp.elements[1].position = p1
        self.link(fac, r.inputs["Fac"])
        return r.outputs["Color"]


def new_material(name: str):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    return mat, NodeBuilder(mat.node_tree)


def emission_material(name: str, color, strength: float, sampling: str = "NONE"):
    mat, nb = new_material(name)
    em = nb.add("ShaderNodeEmission", Strength=strength)
    em.inputs["Color"].default_value = (*color, 1)
    out = nb.add("ShaderNodeOutputMaterial")
    nb.link(em.outputs["Emission"], out.inputs["Surface"])
    mat.cycles.emission_sampling = sampling
    return mat, em


# ----------------------------------------------------------------------------- matériaux (shaders volontairement légers)
def mat_aged_metal():
    """Charbon / métal vieilli : base très sombre, plaques métalliques usées, rugosité marbrée, piqûres."""
    mat, nb = new_material("AgedCharcoalMetal")
    tc = nb.add("ShaderNodeTexCoord")
    big = nb.add("ShaderNodeTexNoise", Scale=2.4, Detail=2.0, Roughness=0.6)
    nb.link(tc.outputs["Object"], big.inputs["Vector"])
    fine = nb.add("ShaderNodeTexNoise", Scale=42.0, Detail=1.0, Roughness=0.6)
    nb.link(tc.outputs["Object"], fine.inputs["Vector"])
    patina = nb.ramp(big.outputs["Fac"], 0.40, 0.64)
    base = nb.add("ShaderNodeMix", _data_type="RGBA")
    base.inputs["A"].default_value = (0.018, 0.018, 0.020, 1)   # charbon
    base.inputs["B"].default_value = (0.080, 0.075, 0.070, 1)   # métal usé
    nb.link(patina, base.inputs["Factor"])
    metal = nb.math("MULTIPLY", patina, 0.9)
    rough = nb.math("MULTIPLY_ADD", fine.outputs["Fac"], 0.30, 0.24)    # 0.24..0.54
    bump = nb.add("ShaderNodeBump", Strength=0.28, Distance=0.004)
    nb.link(fine.outputs["Fac"], bump.inputs["Height"])
    bsdf = nb.add("ShaderNodeBsdfPrincipled")
    nb.link(base.outputs["Result"], bsdf.inputs["Base Color"])
    nb.link(metal, bsdf.inputs["Metallic"])
    nb.link(rough, bsdf.inputs["Roughness"])
    nb.link(bump.outputs["Normal"], bsdf.inputs["Normal"])
    out = nb.add("ShaderNodeOutputMaterial")
    nb.link(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def mat_wet_asphalt():
    """Asphalte mouillé : granulat fin + flaques quasi miroir qui reflètent phares, titre et feux rouges."""
    mat, nb = new_material("WetAsphalt")
    tc = nb.add("ShaderNodeTexCoord")
    puddle = nb.add("ShaderNodeTexNoise", Scale=0.75, Detail=2.0, Roughness=0.6)
    nb.link(tc.outputs["Object"], puddle.inputs["Vector"])
    water = nb.ramp(puddle.outputs["Fac"], 0.53, 0.60)                 # 1 = flaque (~30 %)
    dry = nb.math("SUBTRACT", 1.0, water)
    grain = nb.add("ShaderNodeTexNoise", Scale=160.0, Detail=1.0, Roughness=0.6)
    nb.link(tc.outputs["Object"], grain.inputs["Vector"])
    col = nb.add("ShaderNodeMix", _data_type="RGBA")
    col.inputs["A"].default_value = (0.030, 0.030, 0.032, 1)
    col.inputs["B"].default_value = (0.010, 0.010, 0.012, 1)
    nb.link(water, col.inputs["Factor"])
    rough = nb.math("MULTIPLY_ADD", dry, 0.24, 0.05)
    rough = nb.math("MULTIPLY_ADD", grain.outputs["Fac"], nb.math("MULTIPLY", dry, 0.16), rough)
    bump = nb.add("ShaderNodeBump", Strength=0.30, Distance=0.002)
    nb.link(nb.math("MULTIPLY", grain.outputs["Fac"], dry), bump.inputs["Height"])
    bsdf = nb.add("ShaderNodeBsdfPrincipled")
    nb.link(col.outputs["Result"], bsdf.inputs["Base Color"])
    nb.link(rough, bsdf.inputs["Roughness"])
    nb.link(bump.outputs["Normal"], bsdf.inputs["Normal"])
    out = nb.add("ShaderNodeOutputMaterial")
    nb.link(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def mat_paint():
    """Peinture de marquage usée (lignes de places)."""
    mat, nb = new_material("ParkingPaint")
    tc = nb.add("ShaderNodeTexCoord")
    wear = nb.add("ShaderNodeTexNoise", Scale=3.0, Detail=2.0, Roughness=0.7)
    nb.link(tc.outputs["Object"], wear.inputs["Vector"])
    w = nb.ramp(wear.outputs["Fac"], 0.42, 0.58)
    col = nb.add("ShaderNodeMix", _data_type="RGBA")
    col.inputs["A"].default_value = (0.03, 0.03, 0.03, 1)
    col.inputs["B"].default_value = (0.42, 0.41, 0.38, 1)
    nb.link(w, col.inputs["Factor"])
    bsdf = nb.add("ShaderNodeBsdfPrincipled", Roughness=0.35)
    nb.link(col.outputs["Result"], bsdf.inputs["Base Color"])
    out = nb.add("ShaderNodeOutputMaterial")
    nb.link(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def mat_concrete(name: str, base: float):
    """Béton sale : coulures verticales et taches."""
    mat, nb = new_material(name)
    tc = nb.add("ShaderNodeTexCoord")
    mp = nb.add("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0, 1.0, 0.15)            # coulures verticales (axe local Z)
    nb.link(tc.outputs["Object"], mp.inputs["Vector"])
    streak = nb.add("ShaderNodeTexNoise", Scale=3.0, Detail=2.0, Roughness=0.6)
    nb.link(mp.outputs["Vector"], streak.inputs["Vector"])
    dirt = nb.ramp(streak.outputs["Fac"], 0.25, 0.80)
    col = nb.add("ShaderNodeMix", _data_type="RGBA")
    col.inputs["A"].default_value = (base * 0.45, base * 0.44, base * 0.42, 1)
    col.inputs["B"].default_value = (base, base * 0.98, base * 0.94, 1)
    nb.link(dirt, col.inputs["Factor"])
    bsdf = nb.add("ShaderNodeBsdfPrincipled", Roughness=0.85)
    nb.link(col.outputs["Result"], bsdf.inputs["Base Color"])
    out = nb.add("ShaderNodeOutputMaterial")
    nb.link(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def mat_fog(density: float, hetero: bool, step_rate: float):
    """Brouillard : bruit 3D lent qui dérive (vent de gauche à droite) + couche plus dense près du sol."""
    mat, nb = new_material("Fog")
    vol = nb.add("ShaderNodeVolumePrincipled")
    vol.inputs["Color"].default_value = (0.80, 0.84, 0.90, 1)
    vol.inputs["Anisotropy"].default_value = 0.6
    vol.inputs["Absorption Color"].default_value = (0.0, 0.0, 0.0, 1)
    mp = None
    if hetero:
        tc = nb.add("ShaderNodeTexCoord")
        mp = nb.add("ShaderNodeMapping")
        nb.link(tc.outputs["Object"], mp.inputs["Vector"])
        noise = nb.add("ShaderNodeTexNoise", Scale=0.33, Detail=1.5, Roughness=0.55)
        nb.link(mp.outputs["Vector"], noise.inputs["Vector"])
        sep = nb.add("ShaderNodeSeparateXYZ")
        nb.link(tc.outputs["Object"], sep.inputs["Vector"])
        zw = nb.math("ADD", sep.outputs["Z"], 5.9)                     # z monde (centre du cube à 5,9 m)
        layer = nb.math("EXPONENT", nb.math("MULTIPLY", zw, -0.55))
        layer = nb.math("MULTIPLY_ADD", layer, 1.2, 0.40)             # 1,6 au sol -> 0,4 en hauteur
        wisp = nb.math("MULTIPLY_ADD", noise.outputs["Fac"], 3.2, -1.15)
        wisp = nb.math("MAXIMUM", wisp, 0.05)
        dens = nb.math("MULTIPLY", nb.math("MULTIPLY", wisp, layer), density)
        nb.link(dens, vol.inputs["Density"])
    else:
        vol.inputs["Density"].default_value = density * 0.55
    out = nb.add("ShaderNodeOutputMaterial")
    nb.link(vol.outputs["Volume"], out.inputs["Volume"])
    mat.cycles.volume_sampling = "MULTIPLE_IMPORTANCE"
    mat.cycles.volume_interpolation = "LINEAR"
    mat.cycles.homogeneous_volume = not hetero
    mat.cycles.volume_step_rate = step_rate
    return mat, mp


# ----------------------------------------------------------------------------- géométrie
def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def build_text(font_path: str, size: float) -> bpy.types.Object:
    font = bpy.data.fonts.load(font_path)
    cu = bpy.data.curves.new("TitleCurve", type="FONT")
    cu.body = TEXT_BODY
    cu.font = font
    cu.size = size
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    cu.space_line = 0.86
    cu.space_word = 0.82
    cu.extrude = 0.075
    cu.bevel_depth = 0.011
    cu.bevel_resolution = 3
    cu.resolution_u = 10
    tmp = link(bpy.data.objects.new("TitleTmp", cu))
    dg = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
    bpy.data.objects.remove(tmp)
    mesh.use_auto_smooth = True
    mesh.auto_smooth_angle = math.radians(35)
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    obj = link(bpy.data.objects.new("Title", mesh))
    obj.rotation_euler = (math.radians(90), 0, 0)   # debout, face à -Y
    obj.pass_index = 1
    obj.data.materials.append(mat_aged_metal())
    return obj


def add_plane(name, size, loc, rot, mat):
    sx, sy = size[0] / 2, size[1] / 2
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(-sx, -sy, 0), (sx, -sy, 0), (sx, sy, 0), (-sx, sy, 0)], [], [(0, 1, 2, 3)])
    obj = link(bpy.data.objects.new(name, mesh))
    obj.location, obj.rotation_euler = loc, rot
    obj.data.materials.append(mat)
    return obj


def add_box(name, dims, loc, mat):
    dx, dy, dz = (d / 2 for d in dims)
    verts = [(-dx, -dy, -dz), (dx, -dy, -dz), (dx, dy, -dz), (-dx, dy, -dz),
             (-dx, -dy, dz), (dx, -dy, dz), (dx, dy, dz), (-dx, dy, dz)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    obj = link(bpy.data.objects.new(name, mesh))
    obj.location = loc
    obj.data.materials.append(mat)
    return obj


def add_light(name, kind, color, radius, group, size_deg=None, blend=None):
    ld = bpy.data.lights.new(name, type=kind)
    ld.color = color
    if kind == "SUN":
        ld.angle = radius
    else:
        ld.shadow_soft_size = radius
    if kind == "SPOT":
        ld.spot_size = math.radians(size_deg)
        ld.spot_blend = blend
    ob = link(bpy.data.objects.new(name, ld))
    ob.lightgroup = group
    return ob


def aim(obj, target: Vector):
    """Oriente un spot / une caméra (axe -Z) vers target."""
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


# ----------------------------------------------------------------------------- animation (temps en s)
# La voiture entre sur le parking DERRIÈRE le titre et tourne sur sa gauche (vue de dessus : sens trigo),
# sur un arc de rayon CAR_R autour de CAR_C. Cap θ (degrés, 0 = +X) : d'abord vers la gauche (titre dans le
# noir), puis face caméra (faisceaux à travers les lettres = révélation), puis vers la droite, enfin de dos :
# il ne reste que la lueur rouge des feux arrière derrière le titre et la tache des phares sur le mur du fond.
CAR_C = (-2.0, 9.0)
CAR_R = 2.5
HEAD_T = [0.0, 0.40, 0.75, 1.20, 2.20, 3.50, 5.00]
HEAD_DEG = [-172.0, -150.0, -100.0, -42.0, 8.0, 62.0, 88.0]


def seq_state(t: float) -> dict:
    th = math.radians(pchip(HEAD_T, HEAD_DEG, t))
    px = CAR_C[0] + CAR_R * math.cos(th - math.pi / 2)
    py = CAR_C[1] + CAR_R * math.sin(th - math.pi / 2)
    return dict(car=(px, py, th),
                head=smoothstep(0.0, 0.35, t),
                red=0.55 + 0.45 * smoothstep(3.0, 4.2, t),      # feux de position puis freinage
                cam_u=smoothstep(-0.6, 5.6, t))


def build_car():
    """Voiture stylisée (masse sombre dans le brouillard) : 2 phares (spots + optiques émissives),
    2 feux arrière (points rouges + optiques). Repère local : avant = +Y, sol = z 0."""
    car = link(bpy.data.objects.new("Car", None))
    paint, nb = new_material("CarPaint")
    bsdf = nb.add("ShaderNodeBsdfPrincipled", Roughness=0.28)
    bsdf.inputs["Base Color"].default_value = (0.012, 0.012, 0.014, 1)
    bsdf.inputs["Coat Weight"].default_value = 0.6
    o = nb.add("ShaderNodeOutputMaterial")
    nb.link(bsdf.outputs["BSDF"], o.inputs["Surface"])
    for name, dims, loc in (("CarBody", (1.82, 4.40, 0.72), (0, 0, 0.62)),
                            ("CarCabin", (1.50, 2.20, 0.52), (0, -0.25, 1.24))):
        b = add_box(name, dims, loc, paint)
        b.parent = car
    heads, head_em, tails, tail_em = [], [], [], []
    for sx in (-1, 1):
        sp = add_light(f"Head{sx}", "SPOT", HEAD_COLOR, 0.06, "head", 40, 0.5)
        sp.parent = car
        sp.location = (0.62 * sx, 2.27, 0.72)
        sp.rotation_euler = (math.radians(86.0), 0.0, math.radians(-2.0 * sx))
        heads.append(sp)
        m, em = emission_material(f"HeadLens{sx}", HEAD_COLOR, 0.0)
        lens = add_box(f"HeadLens{sx}", (0.34, 0.05, 0.13), (0.62 * sx, 2.215, 0.72), m)
        lens.parent, lens.visible_shadow, lens.lightgroup = car, False, "head"
        head_em.append(em)
        tl = add_light(f"Tail{sx}", "POINT", RED, 0.05, "red")
        tl.parent = car
        tl.location = (0.70 * sx, -2.30, 0.86)
        tails.append(tl)
        m, em = emission_material(f"TailLens{sx}", RED, 0.0)
        lens = add_box(f"TailLens{sx}", (0.32, 0.05, 0.10), (0.70 * sx, -2.215, 0.86), m)
        lens.parent, lens.visible_shadow, lens.lightgroup = car, False, "red"
        tail_em.append(em)
    car["head_em"], car["tail_em"] = len(head_em), len(tail_em)
    build_car.head_em, build_car.tail_em = head_em, tail_em
    return car, heads, tails


# ----------------------------------------------------------------------------- scène
def build_scene(args) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    scn.cycles.device = "CPU"
    scn.render.fps = FPS
    scn.frame_start, scn.frame_end = 1, N_FRAMES
    scn.render.resolution_x, scn.render.resolution_y = args.width, args.height
    scn.render.resolution_percentage = args.scale
    scn.view_settings.view_transform = "Standard"   # EXR linéaire ; étalonnage en post

    cyc = scn.cycles
    cyc.samples = args.samples
    cyc.use_adaptive_sampling = False
    cyc.use_denoising = False                    # OIDN appliqué en post (render_title.py)
    cyc.seed = 7
    cyc.use_animated_seed = False                # bruit fixe à l'écran -> débruitage stable dans le temps
    cyc.max_bounces = args.bounces
    cyc.diffuse_bounces = 1
    cyc.glossy_bounces = 1
    cyc.transmission_bounces = 0
    cyc.volume_bounces = 0
    cyc.transparent_max_bounces = 2
    cyc.caustics_reflective = False
    cyc.caustics_refractive = False
    cyc.blur_glossy = 1.0
    cyc.sample_clamp_indirect = 4.0
    cyc.volume_step_rate = args.step_rate
    cyc.volume_max_steps = 192
    cyc.filter_width = 1.5
    if args.threads > 0:
        scn.render.threads_mode = "FIXED"
        scn.render.threads = args.threads
    scn.render.use_persistent_data = True

    vl = scn.view_layers[0]
    for g in ("head", "red", "amb"):
        vl.lightgroups.add(name=g)

    world = bpy.data.worlds.new("Night")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.0016, 0.0021, 0.0034, 1)
    world.lightgroup = "amb"
    scn.world = world

    cover = args.mode == "cover"
    title = build_text(args.font, size=1.05)
    title.location = (0.0, 0.0, 2.45 if cover else 1.08)

    # --- décor : sol, marquage, mur, piliers, poutres
    add_plane("Ground", (90, 90), (0, 10, 0), (0, 0, 0), mat_wet_asphalt())
    paint = mat_paint()
    for i, x in enumerate((-6.5, -3.9, -1.3, 1.3, 3.9, 6.5)):
        add_plane(f"Line{i}", (0.13, 5.6), (x, 6.9, 0.004), (0, 0, 0), paint)
    # mur du fond (rappel du mur tagué de 8346), loin dans le brouillard ; lampadaires éteints
    add_plane("Wall", (80, 14), (0, 19.0, 7.0), (math.radians(90), 0, 0), mat_concrete("Concrete", 0.34))
    pole = mat_concrete("PoleMetal", 0.10)
    for i, (px, py) in enumerate([(-4.6, 6.5), (6.2, 11.0), (-9.5, 13.0)]):
        add_box(f"Pole{i}", (0.14, 0.14, 6.5), (px, py, 3.25), pole)
        add_box(f"PoleArm{i}", (1.3, 0.12, 0.10), (px + 0.6, py, 6.4), pole)

    # --- lumières
    moon = add_light("Moon", "SUN", (0.55, 0.66, 1.0), math.radians(10), "amb")
    moon.data.energy = 0.12
    moon.rotation_euler = Vector((0.35, 0.80, -0.50)).to_track_quat("-Z", "Y").to_euler()  # avant-gauche, haut
    moon.visible_volume_scatter = False          # la brume ambiante est ajoutée analytiquement en post
    car, heads, tails = build_car()

    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 35.0
    cam_data.sensor_fit = "AUTO"
    cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.1
    cam_data.clip_end = 200
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = 2.8
    cam = link(bpy.data.objects.new("Cam", cam_data))
    scn.camera = cam

    rig = dict(scene=scn, title=title, car=car, heads=heads, tails=tails, cam=cam,
               cover=cover, fog_map=None, pass_=args.pass_, margin=args.border_margin,
               head_em=build_car.head_em, tail_em=build_car.tail_em)
    if cover:
        setup_cover(rig)
    else:
        calibrate_shift(rig, args.title_cx, args.title_cy, ref_frame=N_FRAMES)

    # --- passe
    if args.pass_ in ("fog", "beauty"):
        fog_mat, rig["fog_map"] = mat_fog(FOG_DENSITY, args.fog_mode == "hetero", args.step_rate)
        add_box("FogVolume", (64, 32, 12), (0, 9.0, 5.9), fog_mat)    # commence 3,6 m devant la caméra
    if args.pass_ == "fog":
        make_occluders(scn)
    configure_passes(scn, args.pass_)
    return rig


def make_occluders(scn):
    """Passe brouillard : toute surface = émetteur blanc d'intensité 1, sans éclairer la scène.
    La passe Emit vaut alors exactement la transmittance du brouillard entre la caméra et la surface."""
    occ, _ = emission_material("Occluder", (1.0, 1.0, 1.0), 1.0, sampling="NONE")
    for ob in scn.objects:
        if ob.type == "MESH" and ob.name != "FogVolume":
            ob.data.materials.clear()
            ob.data.materials.append(occ)
            ob.lightgroup = ""
    scn.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0


def configure_passes(scn, pass_: str):
    vl = scn.view_layers[0]
    vl.use_pass_combined = True
    vl.use_pass_z = True
    if pass_ in ("surf", "text"):
        vl.use_pass_object_index = True
        vl.cycles.denoising_store_passes = True
    elif pass_ == "fog":
        vl.use_pass_emit = True
    s = scn.render.image_settings
    s.file_format = "OPEN_EXR_MULTILAYER"
    s.color_depth = "16"
    s.exr_codec = "ZIP"
    scn.render.use_compositing = False
    scn.render.use_sequencer = False


def apply_seq_frame(rig, f: int):
    """Place voiture / caméra / brouillard pour la frame f (1..150)."""
    t = (f - 1) / FPS
    s = seq_state(t)
    px, py, th = s["car"]
    rig["car"].location = (px, py, 0.0)
    rig["car"].rotation_euler = (0.0, 0.0, th - math.pi / 2)    # avant de la voiture = +Y local
    for sp in rig["heads"]:
        sp.data.energy = 3200.0 * s["head"]
    for tl in rig["tails"]:
        tl.data.energy = 60.0 * s["red"]
    for em in rig["tail_em"]:
        em.inputs["Strength"].default_value = 30.0 * s["red"]
    for em in rig["head_em"]:
        em.inputs["Strength"].default_value = 160.0 * s["head"]
    u = s["cam_u"]
    cam = rig["cam"]
    cam.location = Vector((-0.30 + 0.42 * u, -10.6 + 1.05 * u, 1.62 - 0.06 * u))
    look = Vector((0.10 * u, 0.0, 1.05))
    aim(cam, look)
    cam.data.dof.focus_distance = (cam.location - look).length
    if rig["fog_map"] is not None:      # dérive lente du brouillard
        rig["fog_map"].inputs["Location"].default_value = (-0.11 * t, 0.0, 0.025 * t)


def setup_cover(rig):
    """Couverture : titre en haut, éclairé de face ; voiture face caméra au fond, phares dans le brouillard
    (contre-jour au centre/bas : la silhouette incrustée plus tard s'y découpera en noir)."""
    cam = rig["cam"]
    cam.location = Vector((0.0, -10.4, 1.30))
    aim(cam, Vector((0.0, 0.0, 1.75)))
    cam.data.lens = 30.0
    cam.data.dof.focus_distance = 10.45
    rig["car"].location = (0.0, 10.5, 0.0)
    rig["car"].rotation_euler = (0.0, 0.0, math.radians(180.0))   # face caméra
    for sp in rig["heads"]:
        sp.data.energy = 2600.0
    for em in rig["head_em"]:
        em.inputs["Strength"].default_value = 120.0
    for tl in rig["tails"]:
        tl.data.energy = 0.0
    key = add_light("CoverKey", "SPOT", HEAD_COLOR, 0.25, "head", 30, 0.6)
    key.location = Vector((-4.5, -7.5, 4.2))
    aim(key, Vector((0.4, 0.0, 2.45)))
    key.data.energy = 2600.0
    red = add_light("CoverRed", "POINT", RED, 0.1, "red")
    red.location = Vector((4.2, 4.0, 0.9))
    red.data.energy = 90.0
    calibrate_shift(rig, 0.5, 0.25, ref_frame=None)


# ----------------------------------------------------------------------------- cadrage
def title_box_px(rig) -> tuple[float, float, float, float]:
    """Boîte englobante du titre projeté, en pixels pleine définition (origine en haut à gauche)."""
    scn = rig["scene"]
    W, H = scn.render.resolution_x, scn.render.resolution_y
    mw = rig["title"].matrix_world
    xs, ys = [], []
    for v in rig["title"].data.vertices:
        co = world_to_camera_view(scn, rig["cam"], mw @ v.co)
        xs.append(co.x * W)
        ys.append((1.0 - co.y) * H)
    return min(xs), min(ys), max(xs), max(ys)


def set_title_border(rig):
    """Passe text : ne rend que le cadre du titre (+ marge), en pleine définition."""
    scn = rig["scene"]
    bpy.context.view_layer.update()
    x0, y0, x1, y1 = title_box_px(rig)
    m = rig["margin"]
    W, H = scn.render.resolution_x, scn.render.resolution_y
    scn.render.use_border = True
    scn.render.use_crop_to_border = False
    scn.render.border_min_x = max(0.0, (x0 - m) / W)
    scn.render.border_max_x = min(1.0, (x1 + m) / W)
    scn.render.border_min_y = max(0.0, 1.0 - (y1 + m) / H)
    scn.render.border_max_y = min(1.0, 1.0 - (y0 - m) / H)


def calibrate_shift(rig, cx: float, cy: float, ref_frame):
    """Décentrement optique (shift) pour placer le centre du titre au point visé (pas de recadrage 2D)."""
    scn, cam = rig["scene"], rig["cam"]
    if ref_frame is not None:
        apply_seq_frame(rig, ref_frame)
    bpy.context.view_layer.update()
    big = max(scn.render.resolution_x, scn.render.resolution_y)
    for _ in range(4):
        x0, y0, x1, y1 = title_box_px(rig)
        cam.data.shift_x += ((x0 + x1) / 2 - cx * scn.render.resolution_x) / big
        cam.data.shift_y -= ((y0 + y1) / 2 - cy * scn.render.resolution_y) / big
        bpy.context.view_layer.update()


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    rig = build_scene(args)
    scn = rig["scene"]
    frames = [1] if rig["cover"] else parse_frames(args.frames)

    if args.layout:
        boxes = {}
        for f in frames:
            if not rig["cover"]:
                apply_seq_frame(rig, f)
            bpy.context.view_layer.update()
            boxes[str(f)] = [round(v, 1) for v in title_box_px(rig)]
            if not rig["cover"]:
                lamps = [(tuple(round(c, 2) for c in sp.matrix_world.translation),
                          tuple(round(c, 2) for c in (sp.matrix_world.to_3x3() @ Vector((0, 0, -1)))))
                         for sp in rig["heads"]]
                boxes[str(f)] = boxes[str(f)] + [lamps]
        Path(args.layout).write_text(json.dumps({"mode": args.mode, "boxes": boxes}, indent=1))
        print(f"[title] layout -> {args.layout}")
        return
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.save_blend).resolve()))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in frames:
        dst = out_dir / ("cover.exr" if rig["cover"] else f"f_{f:04d}.exr")
        if args.skip_existing and dst.exists() and dst.stat().st_size > 0:
            continue
        if not rig["cover"]:
            apply_seq_frame(rig, f)
        if rig["pass_"] == "text":
            set_title_border(rig)
        scn.frame_set(f)
        tmp = dst.with_name(dst.stem + ".partial.exr")  # écriture atomique (reprise sûre)
        scn.render.filepath = str(tmp)
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        tmp.replace(dst)
        print(f"[title] {args.pass_} frame {f} -> {dst.name} en {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
