"""Son du film « C'EST QUI LUI ? » : DSP, synthèse sonore, leitmotiv piano, mixage.

Modules (importés à la demande, pour garder l'import du paquet léger) :
  dsp    : briques de traitement (filtres, réverbe, pitch/stretch, loudness, limiteur, master)
  synth  : bibliothèque de sons 100 % synthétisés (CATALOG, render)
  piano  : extraction du leitmotiv du vrai piano (IMG_8354) et variantes
  mixer  : mix(edl, quality) -> render/mix_{quality}.wav

Convention : audio = numpy float32 (N, 2), 48 kHz.
"""
__all__ = ["dsp", "synth", "piano", "mixer"]
