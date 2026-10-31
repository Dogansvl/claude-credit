"""Spécification créative du montage (temps RELATIFS).

`edit/build_edl.py` résout cette spec en `edit/edl.json` (temps ABSOLUS) en utilisant :
  - les transcriptions mot à mot (analysis/IMG_xxxx.transcript.json) pour les ancres A(...)
  - le manifeste des voix (assets/audio/voice/manifest.json) pour la durée des voix off
  - edit/decisions.json pour les choix validés visuellement (identités, positions d'incrustation...)

Conventions
-----------
* Toutes les durées/temps sont en secondes. `t` dans une shot = temps relatif au début de la shot.
* src_in / src_out : temps dans le PROXY (30 fps CFR, rotation appliquée). Float ou ancre A(...).
* Si src_out est défini (et speed constante), dur = (src_out - src_in) / speed.
* kind :
    clip   : plan vidéo depuis le proxy
    freeze : image figée ; `freeze_src` = ('clip', IMG, t | A(...)) ou 'prev_last' (dernière image de la shot précédente)
    black  : noir
    card   : carton sur noir
    photo  : photo (parallaxe 2,5D)
    title  : séquence titre 3D pré-rendue (assets/title/title_seq.mp4)
    montage: suite de micro-plans (`items`)
* camera.zoom = (début, fin) ; camera.center = 'person' | 'face' | 'decision:<clé>' | [x, y] normalisés ;
  camera.punch = durée d'un punch-in rapide au début (ease-out) puis glissement.
* grade.preset ∈ cold | night_red | blue | warm | warm_to_cold | bleach ; grade.mix pour warm_to_cold = (début, fin).
* fx : liste d'effets avec t0/t1 relatifs à la shot (t1=None → fin de shot) :
    black_eyes, stretch_smile, melt, widen_eyes, demon_face, head_tic, negative (flash), glitch,
    subliminal (images insérées), portrait_eyes, speed_ramp, privacy_blur, fade_to_black, fade_from_black,
    darken, rgb_split, shake
* overlays : timecode, card, name, hud_faces, red_circles, subtitle (auto), title_sub
* silhouette : incrustation de la silhouette détourée de 8346 (slot = clé dans decisions.json["silhouette_slots"])
* audio.src_gain_db : gain du son source ; audio.mute_src : couper le son source (consignes) ;
  audio.src_fx : effets sur le son source (pitch_drift, slow_down, reverse_tail...)
* cues : SFX ponctuels {t | at: A(...)} ; sfx = clé de la bibliothèque (pipeline/audio/synth.py CATALOG)
* voice : voix off {id: N01|E_IDENT..., t: relatif shot (début de PAROLE, pas du fichier)}
* subs : sous-titres de vraies répliques {text, kind: 'dialogue', at: A(...) (début), until: A(...) (fin)}
  Les sous-titres de voix off sont générés automatiquement depuis le manifeste.
"""
from __future__ import annotations


def A(phrase: str, edge: str = "start", offset: float = 0.0, fallback: float = 0.0,
      occ: int = 0, window: tuple | None = None) -> dict:
    """Ancre sur la transcription mot à mot du clip de la shot.

    phrase   : texte normalisé recherché (accents/ponctuation ignorés, recherche floue)
    edge     : 'start' (début du 1er mot) | 'end' (fin du dernier mot)
    offset   : décalage ajouté (s)
    fallback : valeur utilisée si la phrase est introuvable (timecodes approximatifs du brief)
    occ      : 0 = première occurrence, -1 = dernière
    window   : (tmin, tmax) restreint la recherche
    """
    return {"anchor": phrase, "edge": edge, "offset": offset, "fallback": fallback, "occ": occ, "window": window}


def ref(shot: str, edge: str = "start", offset: float = 0.0) -> dict:
    """Référence temporelle absolue à une shot (pour la musique)."""
    return {"ref": shot, "edge": edge, "offset": offset}


TC_DATE = "30.10.2026"

SHOTS: list[dict] = [
    # =====================================================================
    # 1. ACCROCHE
    # =====================================================================
    dict(id="S1a", section=1, label="accroche — silhouette dans les phares", kind="clip", clip="IMG_8346",
         src_in=A("c est qui lui", "start", -0.85, fallback=5.6),
         src_out=A("c est qui lui", "end", 0.12, fallback=7.9),
         camera=dict(zoom=(1.0, 1.07), center="decision:silhouette_center_8346"),
         grade=dict(preset="cold"),
         audio=dict(src_gain_db=1.0),
         subs=[dict(text="C'est qui lui ?", at=A("c est qui lui", "start", fallback=6.4),
                    until=A("c est qui lui", "end", 0.2, fallback=7.8))]),
    dict(id="S1b", section=1, label="arrêt sur image + punch-in", kind="freeze", freeze_src="prev_last", dur=0.95,
         camera=dict(zoom=(1.07, 1.42), punch=0.22, center="decision:silhouette_center_8346"),
         grade=dict(preset="bleach"),
         fx=[dict(type="glitch", t0=0.0, t1=0.07, amount=1.0), dict(type="rgb_split", t0=0.0, t1=0.95, amount=0.4)],
         cues=[dict(t=0.0, sfx="impact_big", gain_db=-1), dict(t=0.0, sfx="braam", gain_db=-5),
               dict(t=0.0, sfx="freeze_thump", gain_db=-6), dict(t=0.15, sfx="tinnitus", dur=0.8, gain_db=-24)]),
    dict(id="S1c", section=1, label="noir", kind="black", dur=0.45,
         cues=[dict(t=0.0, sfx="sub_drop", gain_db=-6)]),

    # =====================================================================
    # 2. LA ROUTE
    # =====================================================================
    dict(id="S2a", section=2, label="carton faits réels", kind="card", dur=2.3,
         overlays=[dict(type="card", text="INSPIRÉ DE FAITS RÉELS", t0=0.25, t1=2.15, style="card")],
         cues=[dict(t=0.0, sfx="reverse_swell", dur=1.2, gain_db=-18)]),
    dict(id="S2b", section=2, label="route brouillard feux rouges", kind="clip", clip="IMG_8336",
         src_in=1.0, dur=2.6,
         camera=dict(zoom=(1.0, 1.05), center=[0.5, 0.5]),
         grade=dict(preset="cold"),
         audio=dict(src_gain_db=-14),
         overlays=[dict(type="card", text="30 OCTOBRE 2026", t0=0.35, t1=2.45, style="card_over"),
                   dict(type="timecode", text="20:40", date=TC_DATE)],
         fx=[dict(type="fade_from_black", t0=0.0, t1=0.35)],
         voice=[dict(id="N01", t=0.3)]),
    dict(id="S2c", section=2, label="voie rapide gyrophares", kind="clip", clip="IMG_8342",
         src_in=1.0, dur=2.4,
         camera=dict(zoom=(1.03, 1.0), center=[0.5, 0.5]),
         grade=dict(preset="cold", keep_blue=True),
         audio=dict(src_gain_db=-14),
         overlays=[dict(type="timecode", text="20:43", date=TC_DATE)]),
    dict(id="S2d", section=2, label="banquette arrière", kind="clip", clip="IMG_8343",
         src_in=1.0, dur=2.4,
         camera=dict(zoom=(1.0, 1.04), center=[0.5, 0.45]),
         grade=dict(preset="cold"),
         audio=dict(src_gain_db=-16),
         overlays=[dict(type="timecode", text="20:43", date=TC_DATE)],
         voice=[dict(id="N02", t=0.35)]),
    dict(id="S2e", section=2, label="banquette arrière 2", kind="clip", clip="IMG_8345",
         src_in=1.0, dur=1.9,
         camera=dict(zoom=(1.0, 1.04), center=[0.5, 0.45]),
         grade=dict(preset="cold"),
         audio=dict(src_gain_db=-16),
         overlays=[dict(type="timecode", text="20:46", date=TC_DATE)],
         voice=[dict(id="N03", t=0.1)],
         cues=[dict(t=1.75, sfx="vhs_click", gain_db=-10)]),
    # --- présentation des prénoms (arrêts sur image) -------------------
    dict(id="S2f", section=2, label="prénom GABRIEL", kind="clip", clip="IMG_8341",
         src_in="decision:name_gabriel.t", dur=0.9, freeze_after=0.2,
         camera=dict(zoom=(1.0, 1.06), center="face"),
         grade=dict(preset="cold"),
         audio=dict(mute_src=True),
         overlays=[dict(type="name", name="GABRIEL", confirm="name_gabriel", t0=0.2)],
         cues=[dict(t=0.2, sfx="freeze_thump", gain_db=-8), dict(t=0.0, sfx="whoosh_rev", dur=0.25, gain_db=-16)]),
    dict(id="S2g", section=2, label="prénom HUGO", kind="clip", drop_if_unconfirmed="name_hugo", clip="decision:name_hugo.clip",
         src_in="decision:name_hugo.t", dur=0.9, freeze_after=0.2,
         camera=dict(zoom=(1.0, 1.06), center="face"),
         grade=dict(preset="cold"),
         audio=dict(mute_src=True),
         overlays=[dict(type="name", name="HUGO", confirm="name_hugo", t0=0.2)],
         cues=[dict(t=0.2, sfx="freeze_thump", gain_db=-8), dict(t=0.0, sfx="whoosh_rev", dur=0.25, gain_db=-16)]),
    dict(id="S2h", section=2, label="prénom ARSÈNE", kind="clip", drop_if_unconfirmed="name_arsene", clip="decision:name_arsene.clip",
         src_in="decision:name_arsene.t", dur=0.9, freeze_after=0.2,
         camera=dict(zoom=(1.0, 1.06), center="face"),
         grade=dict(preset="cold"),
         audio=dict(mute_src=True),
         overlays=[dict(type="name", name="ARSÈNE", confirm="name_arsene", t0=0.2)],
         cues=[dict(t=0.2, sfx="freeze_thump", gain_db=-8), dict(t=0.0, sfx="whoosh_rev", dur=0.25, gain_db=-16)]),
    dict(id="S2i", section=2, label="prénom DOGAN (gel qui repart)", kind="freeze",
         freeze_src=("clip", "IMG_8338", A("tu sais quand il vient", "start", -0.12, fallback=0.4)), dur=0.9,
         camera=dict(zoom=(1.06, 1.0), center="face"),
         grade=dict(preset="night_red"),
         overlays=[dict(type="name", name="DOGAN", confirm="name_dogan", t0=0.0)],
         cues=[dict(t=0.0, sfx="freeze_thump", gain_db=-8)]),
    dict(id="S2j", section=2, label="« Tu sais quand il vient ? »", kind="clip", clip="IMG_8338",
         src_in=A("tu sais quand il vient", "start", -0.12, fallback=0.4),
         src_out=A("tu sais quand il vient", "end", 0.28, fallback=1.6),
         camera=dict(zoom=(1.0, 1.03), center="face"),
         grade=dict(preset="night_red"),
         audio=dict(src_gain_db=2.0),
         subs=[dict(text="Tu sais quand il vient ?", at=A("tu sais quand il vient", "start", fallback=0.5),
                    until=A("tu sais quand il vient", "end", 0.15, fallback=1.5))]),
    dict(id="S2k", section=2, label="« Il vient directement, lui. »", kind="clip", clip="IMG_8338",
         src_in=A("il vient directement lui", "start", -0.07, fallback=2.3, occ=-1),
         src_out=A("il vient directement lui", "end", 0.3, fallback=3.75, occ=-1),
         camera=dict(zoom=(1.03, 1.08), center="face"),
         grade=dict(preset="night_red"),
         audio=dict(src_gain_db=2.0, fade_out=0.08),
         subs=[dict(text="Il vient directement, lui.", at=A("il vient directement lui", "start", fallback=2.35, occ=-1),
                    until=A("il vient directement lui", "end", 0.3, fallback=3.8, occ=-1))]),
    dict(id="S2l", section=2, label="Dogan hurle, on l'agrippe", kind="clip", clip="IMG_8338",
         src_in="decision:scream_8338_t", dur=0.9,
         camera=dict(zoom=(1.08, 1.16), center="face"),
         grade=dict(preset="night_red"),
         audio=dict(src_gain_db=3.0, fade_out=0.05),
         fx=[dict(type="head_tic", t0=0.0, t1=0.9, amount=0.6), dict(type="glitch", t0=0.78, t1=0.9, amount=0.8)],
         cues=[dict(t=0.0, sfx="impact_hit", gain_db=-4), dict(t=0.0, sfx="glitch_short", gain_db=-14)]),
    dict(id="S2m", section=2, label="noir", kind="black", dur=0.35),

    # =====================================================================
    # 3. LE PARKING
    # =====================================================================
    dict(id="S3a", section=3, label="8346 parking — LE plan", kind="clip", clip="IMG_8346",
         src_in=A("t as entendu le bruit", "start", -0.35, fallback=0.0),
         src_out=A("oh putain", "end", 0.35, fallback=12.3),
         camera=dict(zoom=(1.0, 1.16), center="decision:silhouette_center_8346"),
         grade=dict(preset="cold"),
         audio=dict(src_gain_db=1.5),
         overlays=[dict(type="timecode", text="20:48", date=TC_DATE)],
         subs=[dict(text="T'as entendu le bruit ?", at=A("t as entendu le bruit", "start", fallback=0.3),
                    until=A("t as entendu le bruit", "end", 0.25, fallback=2.4)),
               dict(text="Je crois qu'il y a un gars là.", at=A("je crois qu il y a un gars", "start", fallback=3.2),
                    until=A("je crois qu il y a un gars la", "end", 0.25, fallback=5.4)),
               dict(text="C'est qui lui ?", at=A("c est qui lui", "start", fallback=6.4),
                    until=A("c est qui lui", "end", 0.3, fallback=7.8)),
               dict(text="Oh putain !", at=A("oh putain", "start", fallback=10.6),
                    until=A("oh putain", "end", 0.35, fallback=11.9))],
         cues=[dict(at=A("c est qui lui", "start", -0.9, fallback=5.5), sfx="reverse_swell", dur=0.9, gain_db=-12),
               dict(at=A("oh putain", "start", -0.45, fallback=10.15), sfx="whoosh_rev", dur=0.45, gain_db=-14),
               # impact + braam sur la FIN de la réplique (sinon « Oh putain ! » est masqué)
               dict(at=A("oh putain", "end", 0.02, fallback=11.9), sfx="impact_big", gain_db=-2),
               dict(at=A("oh putain", "end", 0.02, fallback=11.9), sfx="braam_long", gain_db=-5)]),
    dict(id="S3b", section=3, label="mur vide, il a disparu", kind="freeze",
         freeze_src=("clip", "IMG_8346", "decision:wall_empty_8346_t"), dur=2.0,
         camera=dict(zoom=(1.16, 1.24), center="decision:silhouette_center_8346"),
         grade=dict(preset="cold"),
         fx=[dict(type="fade_to_black", t0=1.4, t1=2.0)],
         overlays=[dict(type="timecode", text="20:48", date=TC_DATE)],
         voice=[dict(id="N04", t=0.2)]),
    dict(id="S3c", section=3, label="noir — lui ne les a plus quittés", kind="black", dur=2.7,
         voice=[dict(id="N05", t=0.45)],
         fx=[dict(type="subliminal", t0="voice_end:N05:-0.35", frames=3, what="silhouette_negative")],
         cues=[dict(t="voice_end:N05:-0.35", sfx="impact_sub", gain_db=-3),
               dict(t="voice_end:N05:-0.35", sfx="sub_drop", gain_db=-8)]),

    # =====================================================================
    # 4. L'ESCAPE
    # =====================================================================
    dict(id="S4a", section=4, label="accueil bleu — Identification terminée", kind="clip", clip="IMG_8347",
         src_in=A("identification terminee", "start", -0.75, fallback=3.4),
         src_out=A("identification terminee", "end", 0.35, fallback=7.9),
         camera=dict(zoom=(1.0, 1.05), center="faces"),
         grade=dict(preset="blue"),
         audio=dict(src_gain_db=2.0),
         overlays=[dict(type="timecode", text="20:55", date=TC_DATE),
                   dict(type="hud_faces", mode="identify", t0=0.45, step=0.42, order="decision:hud_order_8347")],
         subs=[dict(text="Identification terminée.", at=A("identification terminee", "start", fallback=4.2),
                    until=A("identification terminee", "end", 0.2, fallback=7.6))],
         cues=[dict(t=0.25, sfx="scan_sweep", gain_db=-14)]),  # hud_lock blips auto-placés par le builder
    dict(id="S4b", section=4, label="gel — l'entité a choisi", kind="freeze", freeze_src="prev_last", dur=1.55,
         camera=dict(zoom=(1.08, 1.3), center="faces"),
         grade=dict(preset="blue"),
         fx=[dict(type="glitch", t0=0.0, t1=0.1, amount=1.0), dict(type="black_eyes", t0=0.15, t1=None, amount=1.0, ramp=0.5),
             dict(type="rgb_split", t0=0.0, t1=None, amount=0.25)],
         overlays=[dict(type="hud_faces", mode="victim", t0=0.0, order="decision:hud_order_8347")],
         voice=[dict(id="E_IDENT", t=0.15)],
         cues=[dict(t=0.0, sfx="glitch_long", gain_db=-12), dict(t=0.0, sfx="impact_sub", gain_db=-10)]),
    dict(id="S4c", section=4, label="cri face caméra", kind="clip", clip="IMG_8347",
         src_in="decision:scream_8347_t", dur=0.7,
         camera=dict(zoom=(1.1, 1.18), center="face"),
         grade=dict(preset="blue"),
         audio=dict(src_gain_db=2.0, fade_out=0.05),
         fx=[dict(type="negative", t0=0.0, t1=0.067), dict(type="head_tic", t0=0.0, t1=0.7, amount=0.8)],
         cues=[dict(t=0.0, sfx="impact_hit", gain_db=-4)]),
    dict(id="S4d", section=4, label="toilettes sang — C'est quoi ça ?", kind="clip", clip="IMG_8348",
         src_in=A("c est quoi ca", "start", -1.0, fallback=6.8),
         src_out=A("c est quoi ca", "end", 0.4, fallback=9.0),
         camera=dict(zoom=(1.0, 1.06), center=[0.5, 0.55]),
         grade=dict(preset="cold", red_accent=1.0),
         audio=dict(src_gain_db=2.0),
         overlays=[dict(type="timecode", text="20:57", date=TC_DATE)],
         subs=[dict(text="C'est quoi ça ?", at=A("c est quoi ca", "start", fallback=7.6),
                    until=A("c est quoi ca", "end", 0.25, fallback=8.8))],
         cues=[dict(t=0.0, sfx="drip", dur=3.0, gain_db=-20)]),
    dict(id="S4e", section=4, label="sac mortuaire — narrateur organisateurs", kind="clip", clip="IMG_8348",
         src_in="decision:bodybag_8348_t", dur=4.0,
         camera=dict(zoom=(1.0, 1.13), center="decision:bodybag_center_8348"),
         grade=dict(preset="cold", red_accent=1.0),
         audio=dict(src_gain_db=-12),
         overlays=[dict(type="timecode", text="20:57", date=TC_DATE)],
         fx=[dict(type="subliminal", t0=3.0, frames=2, what="silhouette")],
         voice=[dict(id="N06", t=0.2)]),
    dict(id="S4f", section=4, label="salle — portraits, silhouette dans le coin", kind="clip", clip="IMG_8349",
         src_in="decision:room_wide_8349_t", dur=1.7,
         camera=dict(zoom=(1.0, 1.04), center=[0.5, 0.45]),
         grade=dict(preset="cold"),
         audio=dict(src_gain_db=-10),
         overlays=[dict(type="timecode", text="21:00", date=TC_DATE)],
         fx=[dict(type="portrait_eyes", t0=0.0, t1=None, amount=1.0)],
         silhouette=dict(slot="room_corner_8349", t0=0.25, t1=1.45, opacity=0.6),
         cues=[dict(t=0.0, sfx="whoosh_rev", dur=0.5, gain_db=-16), dict(t=0.3, sfx="creak", gain_db=-16)]),
    dict(id="S4g", section=4, label="Gabriel terrifié", kind="clip", clip="IMG_8349",
         src_in="decision:gabriel_terror_8349_t", dur=1.8,
         camera=dict(zoom=(1.0, 1.08), center="face"),
         grade=dict(preset="cold"),
         audio=dict(src_gain_db=-6),
         fx=[dict(type="head_tic", t0=0.3, t1=1.8, amount=0.5), dict(type="widen_eyes", t0=0.6, t1=None, amount=0.6, ramp=0.6)],
         overlays=[dict(type="timecode", text="21:00", date=TC_DATE)]),
    dict(id="S4h", section=4, label="piano — Dogan haut-de-forme", kind="clip", clip="IMG_8354",
         src_in="decision:piano_8354_t", dur=5.0,
         camera=dict(zoom=(1.0, 1.1), center="face"),
         grade=dict(preset="cold", warm_highlights=0.3),
         audio=dict(src_gain_db=0.0, src_fx=[dict(type="pitch_drift", t0=2.8, t1=5.0, semitones=-0.8),
                                              dict(type="slow_down", t0=3.8, t1=5.0, to=0.7)]),
         overlays=[dict(type="timecode", text="22:12", date=TC_DATE)],
         fx=[dict(type="speed_ramp", t0=3.8, t1=5.0, to=0.6), dict(type="black_eyes", t0=4.3, t1=None, amount=1.0, ramp=0.5)],
         voice=[dict(id="N07a", t=0.3), dict(id="N07b", t="after:N07a:0.35")]),
    # --- flashs de possession -----------------------------------------
    dict(id="S4i", section=4, label="possession 1 (8340 yeux fixes)", kind="clip", clip="decision:possess_1.clip",
         src_in="decision:possess_1.t", dur=0.6,
         camera=dict(zoom=(1.1, 1.2), center="face"),
         grade=dict(preset="cold"),
         audio=dict(mute_src=True),
         fx=[dict(type="black_eyes", amount=1.0), dict(type="head_tic", amount=0.7)],
         voice=[dict(id="decision:possess_1.whisper", t=0.05)],
         cues=[dict(t=0.0, sfx="glitch_short", gain_db=-12)]),
    dict(id="S4j", section=4, label="possession 2 (8344 sourire Gabriel)", kind="clip", clip="decision:possess_2.clip",
         src_in="decision:possess_2.t", dur=0.6,
         camera=dict(zoom=(1.1, 1.2), center="face"),
         grade=dict(preset="cold"),
         audio=dict(mute_src=True),
         fx=[dict(type="stretch_smile", amount=1.0, ramp=0.3), dict(type="black_eyes", amount=1.0)],
         voice=[dict(id="E_GABRIEL", t=0.05)],
         cues=[dict(t=0.0, sfx="glitch_short", gain_db=-12)]),
    dict(id="S4k", section=4, label="possession 3 (8339 selfie figé)", kind="clip", clip="decision:possess_3.clip",
         src_in="decision:possess_3.t", dur=0.6,
         camera=dict(zoom=(1.1, 1.2), center="face"),
         grade=dict(preset="cold"),
         audio=dict(mute_src=True),
         fx=[dict(type="melt", amount=0.7, ramp=0.5), dict(type="black_eyes", amount=1.0),
             dict(type="negative", t0=0.4, t1=0.467)],
         voice=[dict(id="decision:possess_3.whisper", t=0.05)],
         cues=[dict(t=0.0, sfx="glitch_short", gain_db=-12)]),
    dict(id="S4l", section=4, label="possession 4 (8337 passager / 8341)", kind="clip", clip="decision:possess_4.clip",
         src_in="decision:possess_4.t", dur=0.6,
         camera=dict(zoom=(1.1, 1.18), center="face"),
         grade=dict(preset="cold"),
         audio=dict(mute_src=True),
         fx=[dict(type="demon_face", amount=0.8), dict(type="head_tic", amount=0.9)],
         voice=[dict(id="decision:possess_4.whisper", t=0.05)],
         cues=[dict(t=0.0, sfx="glitch_short", gain_db=-12), dict(t=0.5, sfx="whoosh", dur=0.4, gain_db=-14)]),
    # --- photo de groupe ----------------------------------------------
    dict(id="S4m", section=4, label="photo de groupe — regardez derrière eux", kind="photo", photo="decision:group_photo",
         dur=5.6,
         camera=dict(zoom=(1.0, 1.16), center="decision:group_photo_center", parallax=1.0),
         grade=dict(preset="cold", red_accent=1.0),
         overlays=[dict(type="red_circles", targets="faces", t0=0.5, step=0.38, draw=0.32),
                   dict(type="red_circles", targets="silhouette", t0=4.95, step=0.0, draw=0.28, frantic=True)],
         silhouette=dict(slot="group_photo_bg", t0=3.95, t1=None, opacity=0.9, fade_in=0.7),
         voice=[dict(id="N08a", t=0.35), dict(id="N08b", t="after:N08a:0.45")],
         cues=[dict(t=0.5, sfx="marker_scribble", gain_db=-14), dict(t=0.88, sfx="marker_scribble", gain_db=-14),
               dict(t=1.26, sfx="marker_scribble", gain_db=-14), dict(t=1.64, sfx="marker_scribble", gain_db=-14),
               dict(t=4.55, sfx="braam", gain_db=-4), dict(t=4.55, sfx="impact_sub", gain_db=-8),
               dict(t=4.95, sfx="marker_scribble", gain_db=-10)]),
    # --- montage qui accélère ------------------------------------------
    dict(id="S4n", section=4, label="montage qui accélère", kind="montage", dur=1.65,
         items=[  # (clip|photo|'silhouette', src_t | decision key, durée)
             dict(clip="IMG_8346", src_in="decision:silhouette_peak_8346_t", dur=0.30, fx=["head_tic"]),
             dict(clip="IMG_8348", src_in="decision:bodybag_8348_t", dur=0.26),
             dict(clip="IMG_8341", src_in="decision:name_gabriel.t", dur=0.22, fx=["black_eyes"]),
             dict(clip="IMG_8347", src_in="decision:scream_8347_t", dur=0.19),
             dict(clip="IMG_8344", src_in="decision:possess_2.t", dur=0.16, fx=["stretch_smile", "black_eyes"]),
             dict(clip="IMG_8340", src_in="decision:possess_1.t", dur=0.14, fx=["black_eyes"]),
             dict(photo="decision:group_photo", dur=0.12, fx=["silhouette"]),
             dict(clip="IMG_8336", src_in=2.0, dur=0.10),
             dict(clip="IMG_8346", src_in="decision:silhouette_peak_8346_t", dur=0.16, fx=["rgb_split"]),
         ],
         grade=dict(preset="cold"),
         audio=dict(mute_src=True),
         cues=[dict(t=0.0, sfx="riser", dur=1.65, gain_db=-8)]),  # + impact_hit léger à chaque coupe (auto)
    dict(id="S4o", section=4, label="noir sec", kind="black", dur=0.35),

    # =====================================================================
    # 5. FAUX HAPPY END
    # =====================================================================
    dict(id="S5a", section=5, label="Burger King chaud — tout était fini", kind="clip", clip="IMG_8355",
         src_in="decision:bk_8355_t", dur=4.4,
         camera=dict(zoom=(1.0, 1.04), center=[0.5, 0.45]),
         grade=dict(preset="warm"),
         audio=dict(src_gain_db=-3), keep_words=True,
         fx=[dict(type="privacy_blur")],
         overlays=[dict(type="timecode", text="23:18", date=TC_DATE)],
         voice=[dict(id="N09a", t=0.9), dict(id="N09b", t="after:N09a:0.45")]),
    dict(id="S5b", section=5, label="rires qui se déforment", kind="clip", clip="IMG_8356",
         src_in="decision:bk_8356_t", dur=3.2,
         camera=dict(zoom=(1.0, 1.1), center="face"),
         grade=dict(preset="warm_to_cold", mix=(0.0, 0.75)),
         keep_words=True,
         audio=dict(src_gain_db=0.0, src_fx=[dict(type="slow_down", t0=1.2, t1=3.2, to=0.55),
                                              dict(type="pitch_drift", t0=1.2, t1=3.2, semitones=-7)]),
         fx=[dict(type="privacy_blur"), dict(type="speed_ramp", t0=1.2, t1=3.2, to=0.55),
             dict(type="stretch_smile", t0=1.8, t1=None, amount=0.8, ramp=1.2)],
         silhouette=dict(slot="bk_window_8356", t0=0.9, t1=None, opacity=0.55),
         overlays=[dict(type="timecode", text="23:24", date=TC_DATE)]),
    dict(id="S5c", section=5, label="gel sourire étiré yeux noirs", kind="freeze", freeze_src="prev_last", dur=1.4,
         camera=dict(zoom=(1.1, 1.2), center="face"),
         grade=dict(preset="warm_to_cold", mix=(0.75, 1.0)),
         fx=[dict(type="privacy_blur"), dict(type="stretch_smile", amount=1.0), dict(type="black_eyes", amount=1.0, ramp=0.6),
             dict(type="melt", t0=0.6, t1=None, amount=0.5, ramp=0.8)],
         silhouette=dict(slot="bk_window_8356", opacity=0.6),
         cues=[dict(t=0.0, sfx="reverse_swell", dur=1.4, gain_db=-8), dict(t=1.38, sfx="sub_drop", gain_db=-8)]),
    dict(id="S5d", section=5, label="noir", kind="black", dur=0.3),

    # =====================================================================
    # 6. FIN
    # =====================================================================
    dict(id="S6a", section=6, label="parking 23:42 — Hugo, silhouette loin", kind="clip", clip="IMG_8357",
         src_in="decision:hugo_8357_t", dur=1.3,
         camera=dict(zoom=(1.0, 1.02), center="face"),
         grade=dict(preset="cold"),
         audio=dict(src_gain_db=-8),
         fx=[dict(type="privacy_blur")],
         silhouette=dict(slot="parking_8357_far", opacity=0.85),
         overlays=[dict(type="timecode", text="23:42", date=TC_DATE)]),
    dict(id="S6b", section=6, label="silhouette plus proche", kind="clip", clip="IMG_8357",
         src_in="decision:hugo_8357_t+1.3", dur=1.1,
         camera=dict(zoom=(1.03, 1.05), center="face"),
         grade=dict(preset="cold"),
         audio=dict(src_gain_db=-10),
         fx=[dict(type="privacy_blur")],
         silhouette=dict(slot="parking_8357_mid", opacity=0.9),
         overlays=[dict(type="timecode", text="23:42", date=TC_DATE)],
         cues=[dict(t=0.0, sfx="tinnitus", dur=2.0, gain_db=-30, ramp_to_db=-17)]),
    dict(id="S6c", section=6, label="silhouette juste derrière", kind="clip", clip="IMG_8357",
         src_in="decision:hugo_8357_t+2.4", dur=0.9,
         camera=dict(zoom=(1.06, 1.08), center="face"),
         grade=dict(preset="cold"),
         audio=dict(src_gain_db=-12),
         fx=[dict(type="privacy_blur")],
         silhouette=dict(slot="parking_8357_near", opacity=0.95),
         overlays=[dict(type="timecode", text="23:42", date=TC_DATE)]),
    dict(id="S6d", section=6, label="silence total puis chuchotement", kind="freeze", freeze_src="prev_last", dur=2.5,
         camera=dict(zoom=(1.08, 1.13), center="face"),
         grade=dict(preset="cold"),
         fx=[dict(type="privacy_blur"), dict(type="darken", t0=1.2, t1=None, amount=0.35)],
         silhouette=dict(slot="parking_8357_near", opacity=0.95),
         overlays=[dict(type="timecode", text="23:42", date=TC_DATE)],
         silence=True,  # coupe TOUT (musique, sfx, source) sauf la voix E_CQL
         voice=[dict(id="E_CQL", t=1.5)]),
    dict(id="S6e", section=6, label="SCREAMER", kind="clip", clip="IMG_8349",
         src_in="decision:screamer_8349_t", dur=0.95,
         camera=dict(zoom=(1.0, 1.18), punch=0.12, center="face"),
         grade=dict(preset="bleach"),
         audio=dict(src_gain_db=6.0, src_fx=[dict(type="scream_layer")]),
         fx=[dict(type="demon_face", amount=1.0), dict(type="head_tic", amount=1.0),
             dict(type="subliminal", t0=0.27, frames=2, what="silhouette"),
             dict(type="subliminal", t0=0.63, frames=2, what="silhouette_negative")],
         cues=[dict(t=0.0, sfx="screamer_stack", gain_db=0.0), dict(t=0.0, sfx="impact_big", gain_db=0.0)]),
    dict(id="S6f", section=6, label="noir", kind="black", dur=0.6),

    # =====================================================================
    # 7. TITRE
    # =====================================================================
    dict(id="S7a", section=7, label="titre 3D brouillard + phares", kind="title", dur=5.0,
         overlays=[dict(type="title_sub", text="INSPIRÉ DE FAITS RÉELS — 30.10.2026", t0=2.6, t1=5.0)],
         cues=[dict(t=0.35, sfx="car_pass", gain_db=-10), dict(t="title:reveal", sfx="impact_sub", gain_db=-6),
               dict(t=2.75, sfx="piano_note", gain_db=-6)],
         voice=[dict(id="E_IDENT", t=3.25)]),
    dict(id="S7b", section=7, label="noir final (traîne du chuchotement)", kind="black", dur=1.25),
]

# ---------------------------------------------------------------------------
# Lits musicaux / ambiances (rendus à la bonne longueur par pipeline/audio/synth.py)
# gain_db = niveau de base ; automation = [(temps relatif au début du lit, dB)] ; duck = atténuation sous les voix
# ---------------------------------------------------------------------------
MUSIC: list[dict] = [
    dict(kind="drone_low", start=ref("S1a", "start"), end=ref("S1b", "end", 0.4), gain_db=-24, fade_in=0.4, fade_out=0.3),
    dict(kind="drone_low", start=ref("S2a", "start"), end=ref("S2j", "start"), gain_db=-17, fade_in=1.8, fade_out=0.4,
         duck=7, params=dict(seed=2)),
    dict(kind="music_box_leitmotif", start=ref("S2b", "start", 0.2), end=ref("S2i", "end"), gain_db=-27, fade_in=1.0,
         fade_out=0.3, duck=5),
    dict(kind="drone_low", start=ref("S2j", "start"), end=ref("S2k", "end", -0.55), gain_db=-27, fade_out=0.25,
         params=dict(seed=3)),
    dict(kind="drone_tension", start=ref("S3a", "start"), end=ref("S3a", "end"), gain_db=-30,
         automation=[(0.0, -30), ("rel_end", -13)], duck=4),
    dict(kind="heartbeat", start=ref("S3a", "start", 2.0), end=ref("S3a", "end", -0.6), gain_db=-15,
         params=dict(bpm_start=62, bpm_end=118)),
    dict(kind="riser", start=ref("S3a", "end", -4.35), end=ref("S3a", "end", -0.33), gain_db=-15),  # finit sur l'impact
    dict(kind="drone_low", start=ref("S3a", "end", -0.6), end=ref("S3c", "end"), gain_db=-16, fade_out=0.6,
         params=dict(seed=5, darker=True), duck=6),
    dict(kind="clock_tick", start=ref("S4a", "start"), end=ref("S4h", "start"), gain_db=-24, params=dict(bpm=60)),
    dict(kind="drone_low", start=ref("S4a", "start"), end=ref("S4m", "start", 3.9), gain_db=-18, fade_in=0.6,
         fade_out=0.05, duck=6, params=dict(seed=7)),
    dict(kind="drone_tension", start=ref("S4d", "start"), end=ref("S4g", "end"), gain_db=-24, fade_in=0.5, fade_out=0.4, duck=5),
    dict(kind="drone_tension", start=ref("S4i", "start"), end=ref("S4l", "end"), gain_db=-17, params=dict(seed=9)),
    dict(kind="drone_low", start=ref("S4m", "start", 4.5), end=ref("S4n", "end"), gain_db=-15, fade_out=0.02,
         params=dict(seed=11, darker=True)),
    dict(kind="warm_pad", start=ref("S5a", "start"), end=ref("S5b", "start", 1.6), gain_db=-22, fade_in=0.6, fade_out=0.8,
         duck=4),
    dict(kind="drone_low", start=ref("S5b", "start", 1.2), end=ref("S5c", "end"), gain_db=-18, fade_in=1.5, fade_out=0.02,
         params=dict(seed=13, darker=True)),
    dict(kind="piano_reverse", start=ref("S5b", "start", 1.2), end=ref("S5c", "end"), gain_db=-20, fade_in=0.8,
         fade_out=0.05, duck=4),
    dict(kind="wind", start=ref("S6a", "start"), end=ref("S6c", "end"), gain_db=-26, fade_in=0.3, fade_out=0.02),
    dict(kind="heartbeat", start=ref("S6a", "start"), end=ref("S6c", "end"), gain_db=-14,
         params=dict(bpm_start=58, bpm_end=40)),
    dict(kind="wind", start=ref("S7a", "start"), end=ref("S7b", "end"), gain_db=-22, fade_in=0.2, fade_out=0.6),
    dict(kind="drone_low", start=ref("S7a", "start"), end=ref("S7b", "end"), gain_db=-19, fade_in=0.6, fade_out=0.6,
         params=dict(seed=17)),
]

# Durée cible (le builder vérifie la fenêtre 85–95 s et signale l'écart)
TARGET_DURATION = 92.0
