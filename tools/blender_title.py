"""Titre 3D « C'EST QUI LUI ? » : scène Blender (Cycles CPU), exécutée en mode headless.

Lancé par tools/render_title.py (appel direct seulement pour déboguer) :
    blender -b --factory-startup -noaudio -P tools/blender_title.py -- --font F.ttf --out DIR --pass surf --frames 1-150

Rendu en PASSES séparées (budget CPU : le brouillard volumétrique pleine définition coûterait ~10x trop cher) :
  --pass surf : surfaces seules (sans volume). EXR multicouche : groupes de lumière (head / red / amb),
                Denoising Albedo/Normal (débruitage OIDN en post), Depth (guide du suréchantillonnage), IndexOB.
  --pass text : comme surf, mais seulement dans un cadre autour du titre (render border) : bords nets en pleine déf.
  --pass fog  : brouillard seul, basse définition. Toutes les surfaces deviennent des occultants émissifs blancs
                d'intensité 1 : la passe Emit = transmittance caméra->surface T, les groupes de lumière = diffusion
                du volume S par source (phares / feux rouges), avec les ombres volumétriques des lettres.
Recomposition (render_title.py) : image = Σ surf_g·T + Σ S_g + (1−T)·brume.

Scène (mètres, Z vers le haut, la caméra regarde vers +Y) :
  - titre extrudé + biseau (Bodoni Moda Black), métal vieilli / charbon, posé sur le sol à y=0 ;
  - asphalte mouillé (rugosité variable, sans flaques à bord dur), lignes de places, mur de béton au fond, poteaux ;
  - brouillard Principled Volume (anisotropie 0,6), plus dense au sol, volutes qui dérivent ;
  - une voiture hors champ à l'avant-gauche tourne à droite : ses phares balaient la face du titre de gauche à
    droite (révélation), passent devant l'objectif puis s'éloignent ; ses feux arrière rouges restent ensuite
    allumés au fond à droite, derrière le titre, dans le brouillard ;
  - lent travelling avant de la caméra avec un léger arc latéral (parallaxe).
--mode cover : image fixe (titre haut, bien éclairé ; phares de face au fond = contre-jour pour la silhouette).
--layout OUT.json : aucun rendu ; écrit la boîte projetée du titre pour les frames demandées.
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
from mathutils import Vector

FPS = 30
N_FRAMES = 150
TEXT_BODY = "C’EST QUI\nLUI ?"   # apostrophe typographique, espace fine avant « ? »
TITLE_W = 3.95                              # largeur de la ligne « C'EST QUI » (m)
HEAD_COLOR = (0.93, 0.96, 1.00)             # blanc chaud légèrement bleuté (xénon / LED)
RED = (1.0, 0.030, 0.012)
FOG_DENSITY = 0.085
NEAR_FOG = 0.15                             # densité relative du brouillard près de la caméra
CAM_Z = 0.55                                # caméra basse : l'asphalte mouillé reflète presque comme un miroir
HEAD_W = 2400.0                             # W par phare (spot)
TAIL_W = 30.0                               # W par feu arrière (point)


# ----------------------------------------------------------------------------- arguments
def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="blender_title.py")
    p.add_argument("--font", required=True)
    p.add_argument("--out", help="dossier de sortie des EXR")
    p.add_argument("--pass", dest="pass_", choices=["surf", "text", "fog", "beauty"], default="surf")
    p.add_argument("--border-margin", type=int, default=34, help="passe text : marge (px) autour du titre")
    p.add_argument("--frames", default="1-150", help="'a-b' ou liste '1,40,80' (frames Blender, 1..150)")
    p.add_argument("--mode", choices=["seq", "cover"], default="seq")
    p.add_argument("--width", type=int, default=1080)
    p.add_argument("--height", type=int, default=1920)
    p.add_argument("--scale", type=int, default=100, help="pourcentage de résolution")
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--step-rate", type=float, default=1.0, help="pas du ray-marching volumique (grand = rapide)")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--title-cx", type=float, default=480 / 1080, help="centre horizontal visé du titre (fraction)")
    p.add_argument("--title-cy", type=float, default=0.405, help="centre vertical visé (fraction depuis le haut)")
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

    def noise(self, vec, scale: float, detail: float = 1.0, rough: float = 0.55):
        n = self.add("ShaderNodeTexNoise", Scale=scale, Detail=detail, Roughness=rough)
        self.link(vec, n.inputs["Vector"])
        return n.outputs["Fac"]


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


# ----------------------------------------------------------------------------- matériaux
def mat_aged_metal():
    """Charbon / métal vieilli : base très sombre, plaques de métal usé, rugosité marbrée, rayures fines."""
    mat, nb = new_material("AgedCharcoalMetal")
    tc = nb.add("ShaderNodeTexCoord")
    big = nb.noise(tc.outputs["Object"], 1.6, 2.0, 0.6)
    patina = nb.ramp(big, 0.38, 0.66)                                    # 0 = charbon, 1 = métal usé
    mp = nb.add("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (30.0, 1.2, 30.0)                 # rayures horizontales (brossé)
    nb.link(tc.outputs["Object"], mp.inputs["Vector"])
    brushed = nb.noise(mp.outputs["Vector"], 6.0, 1.0, 0.5)
    fine = nb.noise(tc.outputs["Object"], 55.0, 1.0, 0.6)
    base = nb.add("ShaderNodeMix", _data_type="RGBA")
    base.inputs["A"].default_value = (0.030, 0.030, 0.033, 1)           # charbon
    base.inputs["B"].default_value = (0.20, 0.19, 0.18, 1)              # acier usé (F0 sombre)
    nb.link(patina, base.inputs["Factor"])
    metal = nb.math("MULTIPLY_ADD", patina, 0.55, 0.40)                  # 0.40..0.95
    rough = nb.math("MULTIPLY_ADD", brushed, 0.22, 0.20)                 # 0.20..0.42
    rough = nb.math("MULTIPLY_ADD", fine, 0.12, rough)
    bump = nb.add("ShaderNodeBump", Strength=0.22, Distance=0.003)
    nb.link(nb.math("ADD", fine, nb.math("MULTIPLY", brushed, 0.5)), bump.inputs["Height"])
    bsdf = nb.add("ShaderNodeBsdfPrincipled")
    nb.link(base.outputs["Result"], bsdf.inputs["Base Color"])
    nb.link(metal, bsdf.inputs["Metallic"])
    nb.link(rough, bsdf.inputs["Roughness"])
    nb.link(bump.outputs["Normal"], bsdf.inputs["Normal"])
    out = nb.add("ShaderNodeOutputMaterial")
    nb.link(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def mat_wet_asphalt():
    """Asphalte mouillé : film d'eau continu (rugosité 0,06..0,30 en variation douce), granulat fin."""
    mat, nb = new_material("WetAsphalt")
    tc = nb.add("ShaderNodeTexCoord")
    wet = nb.ramp(nb.noise(tc.outputs["Object"], 0.35, 2.0, 0.55), 0.35, 0.70)     # 1 = plus mouillé
    grain = nb.noise(tc.outputs["Object"], 140.0, 1.0, 0.6)
    col = nb.add("ShaderNodeMix", _data_type="RGBA")
    col.inputs["A"].default_value = (0.040, 0.040, 0.042, 1)
    col.inputs["B"].default_value = (0.016, 0.016, 0.018, 1)
    nb.link(wet, col.inputs["Factor"])
    rough = nb.math("MULTIPLY_ADD", wet, -0.24, 0.30)                    # 0.30 sec -> 0.06 flaque
    bump = nb.add("ShaderNodeBump", Strength=0.18, Distance=0.002)
    nb.link(nb.math("MULTIPLY", grain, nb.math("SUBTRACT", 1.0, wet)), bump.inputs["Height"])
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
    w = nb.ramp(nb.noise(tc.outputs["Object"], 3.0, 2.0, 0.7), 0.40, 0.60)
    col = nb.add("ShaderNodeMix", _data_type="RGBA")
    col.inputs["A"].default_value = (0.03, 0.03, 0.03, 1)
    col.inputs["B"].default_value = (0.36, 0.35, 0.32, 1)
    nb.link(w, col.inputs["Factor"])
    bsdf = nb.add("ShaderNodeBsdfPrincipled", Roughness=0.25)
    nb.link(col.outputs["Result"], bsdf.inputs["Base Color"])
    out = nb.add("ShaderNodeOutputMaterial")
    nb.link(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def mat_concrete(name: str, base: float):
    """Béton sale : coulures verticales et taches."""
    mat, nb = new_material(name)
    tc = nb.add("ShaderNodeTexCoord")
    mp = nb.add("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0, 1.0, 0.15)
    nb.link(tc.outputs["Object"], mp.inputs["Vector"])
    dirt = nb.ramp(nb.noise(mp.outputs["Vector"], 3.0, 2.0, 0.6), 0.25, 0.80)
    col = nb.add("ShaderNodeMix", _data_type="RGBA")
    col.inputs["A"].default_value = (base * 0.45, base * 0.44, base * 0.42, 1)
    col.inputs["B"].default_value = (base, base * 0.98, base * 0.94, 1)
    nb.link(dirt, col.inputs["Factor"])
    bsdf = nb.add("ShaderNodeBsdfPrincipled", Roughness=0.85)
    nb.link(col.outputs["Result"], bsdf.inputs["Base Color"])
    out = nb.add("ShaderNodeOutputMaterial")
    nb.link(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def mat_fog(density: float, step_rate: float):
    """Brouillard : volutes (2 échelles de bruit) qui dérivent lentement + couche plus dense près du sol."""
    mat, nb = new_material("Fog")
    vol = nb.add("ShaderNodeVolumePrincipled")
    vol.inputs["Color"].default_value = (0.86, 0.89, 0.93, 1)
    vol.inputs["Anisotropy"].default_value = 0.6
    vol.inputs["Absorption Color"].default_value = (0.0, 0.0, 0.0, 1)
    geo = nb.add("ShaderNodeNewGeometry")
    mp = nb.add("ShaderNodeMapping")                                     # dérive (animée par frame)
    nb.link(geo.outputs["Position"], mp.inputs["Vector"])
    big = nb.noise(mp.outputs["Vector"], 0.16, 1.0, 0.5)
    mid = nb.noise(mp.outputs["Vector"], 0.55, 1.0, 0.5)
    wisp = nb.math("ADD", nb.math("MULTIPLY", big, 0.65), nb.math("MULTIPLY", mid, 0.35))
    wisp = nb.math("MULTIPLY_ADD", wisp, 3.4, -1.25)                     # contraste des volutes
    wisp = nb.math("MAXIMUM", wisp, 0.10)
    sep = nb.add("ShaderNodeSeparateXYZ")
    nb.link(geo.outputs["Position"], sep.inputs["Vector"])
    layer = nb.math("EXPONENT", nb.math("MULTIPLY", sep.outputs["Z"], -0.62))
    layer = nb.math("MULTIPLY_ADD", layer, 1.55, 0.20)                   # 1,75 au sol -> 0,2 en hauteur
    # brouillard plus léger entre la caméra et le titre (voile limité : silhouette nette devant le banc de brume)
    near = nb.math("MULTIPLY_ADD", nb.math("MINIMUM", nb.math("MAXIMUM", nb.math("MULTIPLY_ADD", sep.outputs["Y"],
                   1 / 5.0, 3 / 5.0), 0.0), 1.0), 1.0 - NEAR_FOG, NEAR_FOG)    # NEAR_FOG si y <= -3 m -> 1 si y >= 2 m
    dens = nb.math("MULTIPLY", nb.math("MULTIPLY", nb.math("MULTIPLY", wisp, layer), near), density)
    nb.link(dens, vol.inputs["Density"])
    out = nb.add("ShaderNodeOutputMaterial")
    nb.link(vol.outputs["Volume"], out.inputs["Volume"])
    mat.cycles.volume_sampling = "MULTIPLE_IMPORTANCE"
    mat.cycles.volume_interpolation = "LINEAR"
    mat.cycles.homogeneous_volume = False
    mat.cycles.volume_step_rate = step_rate
    return mat, mp


# ----------------------------------------------------------------------------- géométrie
def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def build_text(font_path: str) -> bpy.types.Object:
    """Texte extrudé + biseau, converti en maillage, mis à l'échelle pour que la 1re ligne fasse TITLE_W."""
    font = bpy.data.fonts.load(font_path)
    cu = bpy.data.curves.new("TitleCurve", type="FONT")
    cu.body = TEXT_BODY
    cu.font = font
    cu.size = 1.0
    cu.align_x = "CENTER"
    cu.align_y = "BOTTOM_BASELINE"
    cu.space_line = 0.84
    cu.space_character = 1.02
    cu.extrude = 0.085
    cu.bevel_depth = 0.014
    cu.bevel_resolution = 3
    cu.resolution_u = 12
    tmp = link(bpy.data.objects.new("TitleTmp", cu))
    dg = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
    bpy.data.objects.remove(tmp)
    xs = [v.co.x for v in mesh.vertices]
    ys = [v.co.y for v in mesh.vertices]
    k = TITLE_W / (max(xs) - min(xs))
    cx, ymin = (max(xs) + min(xs)) / 2, min(ys)
    for v in mesh.vertices:                                  # origine : centre-bas du titre
        v.co = Vector(((v.co.x - cx) * k, (v.co.y - ymin) * k, v.co.z * k))
    mesh.use_auto_smooth = True
    mesh.auto_smooth_angle = math.radians(35)
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    obj = link(bpy.data.objects.new("Title", mesh))
    obj.rotation_euler = (math.radians(90), 0, 0)             # debout, face à -Y
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
    elif kind == "AREA":
        ld.size = radius
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


# ----------------------------------------------------------------------------- voitures (lumières seules)
def build_lamp_pair(prefix: str, kind: str, color, group: str, sep: float, lens_dims, size_deg=None, blend=None):
    """Paire de feux (phares = spots, feux arrière = points) + optiques émissives, sous un pivot (empty).
    Repère local du pivot : avant de la voiture = +Y, sol = z 0."""
    piv = link(bpy.data.objects.new(prefix, None))
    lamps, ems = [], []
    for sx in (-1, 1):
        lp = add_light(f"{prefix}{sx}", kind, color, 0.07 if kind == "SPOT" else 0.06, group, size_deg, blend)
        lp.parent = piv
        lp.location = (sep / 2 * sx, 0.0, 0.0)
        if kind == "SPOT":
            lp.rotation_euler = (math.radians(87.0), 0.0, math.radians(-3.0 * sx))   # légèrement plongeants
        lamps.append(lp)
        m, em = emission_material(f"{prefix}Lens{sx}", color, 0.0)
        lens = add_box(f"{prefix}Lens{sx}", lens_dims, (sep / 2 * sx, -0.03 if kind == "SPOT" else 0.03, 0.0), m)
        lens.parent, lens.visible_shadow, lens.lightgroup = piv, False, group
        ems.append(em)
    return piv, lamps, ems


# La voiture (lumières seules : dans le brouillard on ne voit que ses feux) arrive au fond, derrière le titre, et
# fait un demi-tour par la gauche (cap θ en degrés, 0 = +X, 90 = +Y = s'éloigne de la caméra) :
#   phares d'abord tournés vers la gauche (hors champ) -> vers l'objectif (faisceaux à travers les lettres,
#   éblouissement, révélation ~1,1 s) -> vers la droite -> elle s'arrête dos à nous, derrière le titre à droite :
#   il ne reste que ses feux arrière rouges dans le brouillard (coup de frein vers 3,4 s : « il attend »).
POS_T = [0.00, 0.40, 1.10, 2.20, 3.00, 3.55, 5.00]
POS_X = [-6.4, -5.3, -3.0, -0.3, 1.8, 2.5, 2.5]
POS_Y = [11.4, 10.8, 10.0, 9.7, 10.6, 11.4, 11.4]
HEAD_T = [0.00, 0.40, 0.80, 1.15, 1.60, 2.20, 2.90, 3.55, 5.00]
HEAD_DEG = [-178.0, -168.0, -132.0, -92.0, -52.0, -12.0, 48.0, 94.0, 96.0]
CAM_REF = (0.0, -10.0)                    # position approximative de la caméra (orientation des feux)


def car_path(t: float) -> tuple[float, float, float]:
    """Position (x, y) au sol et cap (rad) de l'avant de la voiture à l'instant t (clés monotones par morceaux)."""
    return pchip(POS_T, POS_X, t), pchip(POS_T, POS_Y, t), math.radians(pchip(HEAD_T, HEAD_DEG, t))


def facing(x: float, y: float, th: float, sign: float) -> float:
    """0..1 : à quel point l'avant (sign=+1) ou l'arrière (sign=-1) de la voiture est tourné vers la caméra."""
    dx, dy = CAM_REF[0] - x, CAM_REF[1] - y
    c = sign * (math.cos(th) * dx + math.sin(th) * dy) / max(1e-6, math.hypot(dx, dy))
    return smoothstep(0.0, 0.85, c)


def seq_state(t: float) -> dict:
    x, y, th = car_path(t)
    on = smoothstep(0.10, 0.45, t)                                       # la voiture débouche dans le champ
    front, rear = facing(x, y, th, 1.0), facing(x, y, th, -1.0)
    brake = 1.0 + 1.1 * math.exp(-((t - 3.45) / 0.40) ** 2)
    glare = smoothstep(0.55, 0.97, (math.cos(th) * (CAM_REF[0] - x) + math.sin(th) * (CAM_REF[1] - y))
                       / math.hypot(CAM_REF[0] - x, CAM_REF[1] - y))
    red_on = smoothstep(1.25, 2.40, t)
    return dict(car=(x, y, th), head=on, head_lens=on * front,
                red=red_on * (0.08 + 0.92 * rear) * brake, red_lens=red_on * rear * brake,
                glare=on * glare, cam_u=t / (N_FRAMES / FPS))


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
    cyc.max_bounces = 3
    cyc.diffuse_bounces = 1
    cyc.glossy_bounces = 1
    cyc.transmission_bounces = 0
    cyc.volume_bounces = 0
    cyc.transparent_max_bounces = 2
    cyc.caustics_reflective = False
    cyc.caustics_refractive = False
    cyc.blur_glossy = 1.0
    cyc.sample_clamp_indirect = 3.0
    cyc.volume_step_rate = args.step_rate
    cyc.volume_max_steps = 160
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
    title = build_text(args.font)
    title.location = (0.0, 0.0, 2.45 if cover else 0.0)

    # --- décor : sol, marquage, mur, poteaux
    add_plane("Ground", (120, 120), (0, 20, 0), (0, 0, 0), mat_wet_asphalt())
    paint = mat_paint()
    for i, x in enumerate((-7.5, -5.0, 5.0, 7.5)):
        add_plane(f"Line{i}", (0.12, 5.0), (x, 3.0, 0.003), (0, 0, 0), paint)
    add_plane("Stop", (16.0, 0.14), (0, 5.6, 0.003), (0, 0, 0), paint)
    # mur du fond (rappel du mur tagué de 8346), loin dans le brouillard ; lampadaires éteints
    add_plane("Wall", (110, 16), (0, 26.0, 8.0), (math.radians(90), 0, 0), mat_concrete("Concrete", 0.30))
    pole = mat_concrete("PoleMetal", 0.08)
    for i, (px, py, hgt) in enumerate([(-6.0, 5.0, 7.0), (6.8, 11.0, 7.5), (-9.5, 16.0, 7.5), (2.2, 19.0, 7.5)]):
        add_box(f"Pole{i}", (0.16, 0.16, hgt), (px, py, hgt / 2), pole)
        add_box(f"PoleArm{i}", (1.4, 0.12, 0.10), (px + 0.65, py, hgt - 0.1), pole)

    # --- lumières
    # lune voilée en contre-jour haut (lueur froide du brouillard derrière le titre, liseré sur les biseaux)
    moon = add_light("Moon", "SUN", (0.55, 0.66, 1.0), math.radians(12), "amb")
    moon.data.energy = 0.22
    moon.rotation_euler = Vector((0.35, -0.75, -0.55)).to_track_quat("-Z", "Y").to_euler()
    # lampadaire lointain, avant-gauche, froid : faible lumière de face (le titre reste « presque invisible »)
    lamp = add_light("StreetLamp", "SPOT", (0.70, 0.82, 1.0), 0.4, "amb", 70.0, 1.0)
    lamp.location = Vector((-7.5, -6.0, 7.5))
    aim(lamp, Vector((0.0, 0.0, 0.6)))
    lamp.data.energy = 900.0
    car, heads, head_em = build_lamp_pair("Head", "SPOT", HEAD_COLOR, "head", 1.30, (0.34, 0.05, 0.13), 38.0, 0.55)
    tail, tails, tail_em = build_lamp_pair("Tail", "POINT", RED, "red", 1.40, (0.30, 0.05, 0.09))
    tail.parent = car
    tail.location = (0.0, -4.35, 0.14)                              # arrière de la voiture, feux tournés vers -Y
    tail.rotation_euler = (0.0, 0.0, math.radians(180.0))

    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 35.0
    cam_data.sensor_fit = "AUTO"
    cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.1
    cam_data.clip_end = 250
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = 2.8
    cam = link(bpy.data.objects.new("Cam", cam_data))
    scn.camera = cam

    rig = dict(scene=scn, title=title, car=car, heads=heads, head_em=head_em, tail=tail, tails=tails,
               tail_em=tail_em, cam=cam, cover=cover, fog_map=None, pass_=args.pass_, margin=args.border_margin)
    if cover:
        setup_cover(rig)
    else:
        calibrate_shift(rig, args.title_cx, args.title_cy, ref_frame=N_FRAMES)

    if args.pass_ in ("fog", "beauty"):
        fog_mat, rig["fog_map"] = mat_fog(FOG_DENSITY, args.step_rate)
        add_box("FogVolume", (80, 48, 9), (0, 9.0, 4.5), fog_mat)     # englobe la caméra (pas de bord visible)
    if args.pass_ == "fog":
        make_occluders(scn)
    configure_passes(scn, args.pass_)
    return rig


def make_occluders(scn):
    """Passe brouillard. Chaque surface garde son matériau + une émission blanche d'intensité 1 (non échantillonnée
    comme lumière) : la passe Emit (émission vue directement) vaut la transmittance T caméra->surface.
    Les lampes n'éclairent plus directement les surfaces (déjà dans la passe surf) : les groupes de lumière ne
    contiennent que la diffusion du volume, vue directement OU via un rebond (brouillard reflété par l'asphalte
    mouillé et le métal, lueur du brouillard qui éclaire la face des lettres)."""
    done = set()
    for ob in scn.objects:
        if ob.type == "MESH" and ob.name != "FogVolume":
            ob.lightgroup = ""
            ob.visible_shadow = "Lens" not in ob.name
            for slot in ob.material_slots:
                mat = slot.material
                if mat is None or mat.name in done:
                    continue
                done.add(mat.name)
                nt = mat.node_tree
                out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
                src = out.inputs["Surface"].links[0].from_socket if out.inputs["Surface"].links else None
                em = nt.nodes.new("ShaderNodeEmission")
                em.inputs["Color"].default_value = (1, 1, 1, 1)
                em.inputs["Strength"].default_value = 1.0
                if src is None or src.node.type == "EMISSION":
                    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
                else:
                    add = nt.nodes.new("ShaderNodeAddShader")
                    nt.links.new(src, add.inputs[0])
                    nt.links.new(em.outputs["Emission"], add.inputs[1])
                    nt.links.new(add.outputs["Shader"], out.inputs["Surface"])
                mat.cycles.emission_sampling = "NONE"
        elif ob.type == "LIGHT":
            ob.visible_diffuse = False
            ob.visible_glossy = False
            ob.visible_transmission = False
    scn.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    scn.cycles.max_bounces = 1
    scn.cycles.diffuse_bounces = 0                 # coût : seul le reflet spéculaire (asphalte, métal) est gardé
    scn.cycles.glossy_bounces = 1
    scn.cycles.volume_bounces = 0


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


def place_camera(rig, u: float):
    """Travelling avant lent (ease-in-out doux) avec léger arc latéral ; u = 0..1 sur les 5 s."""
    e = u * u * (3 - 2 * u) * 0.35 + u * 0.65
    cam = rig["cam"]
    cam.location = Vector((-0.38 + 0.50 * e, -10.6 + 1.35 * e, CAM_Z - 0.04 * e))
    look = Vector((0.06 * e, 0.0, CAM_Z + 0.25))
    aim(cam, look)
    cam.data.dof.focus_distance = (Vector((cam.location.x, 0.0, cam.location.z)) - cam.location).length


def apply_seq_frame(rig, f: int):
    """Place voiture / caméra / brouillard pour la frame f (1..150)."""
    t = (f - 1) / FPS
    s = seq_state(t)
    px, py, th = s["car"]
    rig["car"].location = (px, py, 0.70)
    rig["car"].rotation_euler = (0.0, 0.0, th - math.pi / 2)     # avant (+Y local) dans la direction du cap
    for sp in rig["heads"]:
        sp.data.energy = HEAD_W * s["head"]
    for em in rig["head_em"]:
        em.inputs["Strength"].default_value = 160.0 * s["head_lens"]
    for tl in rig["tails"]:
        tl.data.energy = TAIL_W * s["red"]
    for em in rig["tail_em"]:
        em.inputs["Strength"].default_value = 45.0 * s["red_lens"]
    place_camera(rig, s["cam_u"])
    if rig["fog_map"] is not None:      # dérive lente du brouillard (vent de gauche à droite)
        rig["fog_map"].inputs["Location"].default_value = (-0.16 * t, -0.05 * t, 0.03 * t)


def setup_cover(rig):
    """Couverture : titre en haut, bien éclairé (clé froide de face + liseré rouge par l'arrière) ; voiture au fond,
    face caméra : la lueur des phares dans le brouillard, au centre, sert de contre-jour à la silhouette qui sera
    incrustée au centre/bas (pieds sur les reflets de l'asphalte mouillé)."""
    cam = rig["cam"]
    cam.location = Vector((0.0, -11.0, 1.05))
    aim(cam, Vector((0.0, 0.0, 1.05)))
    cam.data.lens = 41.0
    cam.data.dof.focus_distance = 11.0
    cam.data.dof.aperture_fstop = 4.0
    rig["car"].location = (0.35, 17.5, 0.70)
    rig["car"].rotation_euler = (0.0, 0.0, math.radians(180.0))    # face caméra
    for sp in rig["heads"]:
        sp.data.energy = 1500.0
    for em in rig["head_em"]:
        em.inputs["Strength"].default_value = 120.0
    # accent rouge : feux arrière d'une voiture garée au fond à gauche
    tail = rig["tail"]
    tail.parent = None
    tail.location = (-3.3, 10.0, 0.84)
    tail.rotation_euler = (0.0, 0.0, math.radians(172.0))
    for tl in rig["tails"]:
        tl.data.energy = 22.0
    for em in rig["tail_em"]:
        em.inputs["Strength"].default_value = 40.0
    key = add_light("CoverKey", "SPOT", HEAD_COLOR, 0.35, "head", 30, 0.7)
    key.location = Vector((-4.0, -8.5, 6.5))
    aim(key, Vector((0.2, 0.0, rig["title"].location.z + 0.75)))
    key.data.energy = 1500.0
    key.visible_volume_scatter = False                              # pas de cône parasite devant le titre
    calibrate_shift(rig, 0.47, 0.235, ref_frame=None)


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
            if not rig["cover"]:                 # contrôle de composition : feux projetés (px) et état
                W, H = scn.render.resolution_x, scn.render.resolution_y
                proj = {}
                for key in ("car", "tail"):
                    co = world_to_camera_view(scn, rig["cam"], rig[key].matrix_world.translation)
                    proj[key] = [round(co.x * W), round((1 - co.y) * H), round(co.z, 1)]
                st = seq_state((f - 1) / FPS)
                proj["state"] = {k: (round(v, 2) if isinstance(v, float) else [round(c, 2) for c in v])
                                 for k, v in st.items()}
                boxes[str(f)] = boxes[str(f)] + [proj]
        Path(args.layout).write_text(json.dumps({"mode": args.mode, "boxes": boxes,
                                                 "shift": [rig["cam"].data.shift_x, rig["cam"].data.shift_y]},
                                                indent=1))
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
