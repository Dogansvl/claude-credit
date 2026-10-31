# Voix off — rapport de génération

Généré par `python tools/make_voices.py` le 2026-10-31 15:40 — backend : **piper:tom**.

Backends écartés automatiquement : `edge: edge-tts indisponible : ClientConnectorDNSError: Cannot connect to host speech.platform.bing.com:443 ssl:<ssl.SSLContext object at 0x7faf3e99f9d0> [Name or service not known]`

## Voix choisie et pourquoi

* **Premier choix automatique : edge-tts `fr-FR-HenriNeural`** (rate −12 %, pitch −8 Hz), voix neurale masculine
  grave et posée. Dans ce conteneur le service est bloqué (`speech.platform.bing.com`) ; sur le PC du client,
  `python tools/make_voices.py` l'utilise automatiquement (sonde réseau réelle, repli piper sinon).
* **Narrateur hors ligne : piper `fr_FR-tom-medium`** (44,1 kHz). Banc d'essai ci-dessous (12 phrases × 3 prises,
  chaîne réglée pour chaque voix) : de loin la meilleure articulation (WER ≈ 2,4 × plus bas que pierre), prosodie
  posée et régulière (σ F0 ≈ 2,2 demi-tons dans la phrase, hauteur médiane la plus stable d'une phrase à l'autre),
  pleine bande. Son défaut — une voix très soufflée (HNR ≈ 5 dB, aigus 4–9 kHz très présents) — est dompté par la
  chaîne (étagère −6 dB au-dessus de 4,2 kHz, de-esser ferme) ; en voix off d'horreur ce grain voilé, proche,
  sert le propos.
* `upmc-pierre` a le timbre le plus propre (meilleur DNSMOS, HNR ≈ 10 dB, prosodie plus mobile) mais des défauts
  d'articulation systématiques qu'un francophone entend : [ʁ] mal réalisé (« Ils riaient » compris « Il viait »,
  « derrière eux » → « d'ailleurs peu »), « s'est mis » → « semis », et une bande limitée à ~10 kHz.
  `gilles-low` (16 kHz) : le plus grave mais grain de vocodeur, nasales absentes du modèle, pire WER, hauteur
  irrégulière d'une phrase à l'autre. `jessica` : féminine (référence).
* **Sélection de prises** : le bruit interne de VITS est à **graine explicite** (attribut `seed` des nœuds ONNX
  `RandomNormalLike`, rendu reproductible) ; 6 prises par phrase, chacune traitée puis réécoutée par l'ASR ; la
  meilleure (intelligibilité, prosodie, tenue de la fenêtre du montage) est gardée.
* **Entité : base tom** également — après chuchotement seule l'enveloppe spectrale (l'articulation) compte, et tom
  prononce nettement mieux les prénoms isolés (« Gabriel » reconnu 3/3 contre 0/3 pour pierre).
* Modèles sherpa-onnx (supertonic, kokoro multi-lang, matcha, zipvoice) : **non évalués** — l'accès au dépôt
  GitHub `k2-fsa/sherpa-onnx` (API et releases) est refusé dans cette session (proxy : « GitHub access to this
  repository is not enabled »). D'après leur documentation : kokoro multi-lang n'a qu'une voix française féminine
  (ff_siwis), matcha et zipvoice sont zh/en ; supertonic 2 annonce le français : à tester sur le PC du client.

### Graphies testées pour « escape game » (ASR aller-retour, voix retenue)

| graphie TTS | phonèmes (espeak) | reconnu « escape game » | exemple ASR |
|---|---|---|---|
| `escape game` | `ɛskˈap ɡˈeɪm` | 3/4 | 4 amis ont pris la route pour un escape game. |
| `escape guème` | `ɛskˈap ɡˈɛm` | 1/4 | Chathrami ont pris la route pour un escape game. |
| `eskèpe guème` | `ɛskˈɛp ɡˈɛm` | 1/4 | 4 amis ont pris la route pour un escape d'AIM. |
| `èscaipe gaime` | `ɛskˈɛp ɡˈɛm` | 1/4 | 4 amis ont pris la route pour un escape d'AIM. |
| `esquèpe guéme` | `ɛskˈɛp ɡˈem` | 2/4 | 4 amis ont pris la route pour un escape d'Ain. |
| `eskeïpe gueïme` | `ɛskˈɛjp ɡˈɛjm` | 4/4 | 4 amis ont pris la route pour un escape game. |

Graphie retenue : `eskeïpe gueïme` → `ɛskˈɛjp ɡˈɛjm` : la prononciation « à l'anglaise avec l'accent français » courante en France ; [ɛskɛp ɡɛm] (sans diphtongue) est souvent entendu « escape d'AIM » par l'ASR, et la graphie brute fait lire « escape » [ɛskap] à espeak.

## Chaînes de traitement

**Narrateur** (`pipeline/audio/voicefx.narrator_chain`) : TTS → 48 kHz → transposition **avec préservation des
formants** (rubberband R3 `-F`) : −2,5 demi-tons pour tom (−1 pour edge-Henri, déjà abaissé de 8 Hz) → EQ (passe-haut
65 Hz, chaleur +3 dB à 150 Hz, −2 dB à 320 Hz, +1 dB à 2,4 kHz, étagère −6 dB au-dessus de 4,2 kHz pour le souffle
de tom ; exciteur discret au-dessus de 9 kHz pour les voix à bande étroite) → de-esser (bande > 4,5 kHz) →
compression douce (2,4:1, genou 8 dB, 12/140 ms) → réverbe type plate synthétique 1,2 s, pré-délai 14 ms, 12 % wet →
−20 LUFS intégrés, crête vraie ≤ −3 dBTP (limiteur à anticipation suréchantillonné ×4) → tête 60 ms avant la parole,
queue de réverbe ≤ 0,55 s fondue. Débit : piper `length_scale` 1,10 (plus lent que nature) ; si la fenêtre du montage
l'exige, compression temporelle rubberband ≤ +10 % (formants préservés) ou débit TTS accéléré, la meilleure prise
gagne.

**Entité** (`voicefx.entity_chain`) : TTS → 48 kHz → transposition −3,5 à −5,5 dt **formants vers le bas**
(rubberband sans `-F`) → **chuchotement** : dans les trames voisées, enveloppe spectrale (liftrage cepstral 2,2 ms)
× bruit blanc de phase aléatoire, énergie de trame conservée ; trames non voisées (fricatives, explosions) gardées
telles quelles → consonnes nettes → couches désaccordées (0 / +15…+22 / −22…−27 cents, décalages 8–19 ms,
panoramiques ±0,3–0,5) + couche sous-grave (−7 à −9 dt, passe-bas 2,5 kHz) + **couche sèche « collée à
l'oreille »** (−1 à −2 dt, dominante pour E_IDENT et E_CQL, décalée à gauche avec retard interaural 0,35 ms) → EQ de
proximité (+4,5 dB à 150 Hz, souffle +3 dB à 4 kHz, −3 dB à 7,5 kHz) → de-esser, compression → **pré-swell en
réverbe inversée** `reverse(reverb(reverse(x)))` (IR sombre synthétique) qui monte jusqu'au mot (prénoms 0,6 s,
E_IDENT 1,0 s ; E_CQL : aucun, remplacé par une inspiration très proche) → queue sombre courte → −20 LUFS, crête
≤ −3 dBTP. `onset_s` = début réel de la parole (après pré-swell / inspiration).

## Fichiers

| ID | type | texte affiché | texte TTS | onset (s) | parole (s) | durée (s) | fenêtre parole max (s) | LUFS | crête (dBTP) | réglage | ASR (fichier final) | WER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N01 | narrator | Le 30 octobre 2026, quatre amis ont pris la route pour un escape game. | Le trente octobre deux mille vingt-six, quatre amis ont pris la route pour un eskeïpe gueïme. | 0.061 | 4.88 | 5.50 | 4.90 | -20.0 | -3.12 | {'length_scale': 0.94, 'seed': 53073, 'tempo': 1.1} | Le 30 octobre 2026, 4 amis ont pris la route pour un escape game. | 0.0 |
| N02 | narrator | Ils se sont filmés toute la soirée. | Ils se sont filmés toute la soirée. | 0.060 | 1.77 | 2.38 | 2.00 | -20.0 | -8.25 | {'length_scale': 1.1, 'seed': 77019} | Ils se sont filmés toute la soirée. | 0.0 |
| N03 | narrator | Voici ce qu'ils ont filmé. | Voici ce qu'ils ont filmé. | 0.060 | 1.71 | 2.32 | 1.95 | -20.0 | -8.38 | {'length_scale': 1.1, 'seed': 22397} | Voici ce qu'ils ont filmé. | 0.0 |
| N04 | narrator | Ils ne l'ont jamais revu. | Ils ne l'ont jamais revu. | 0.060 | 1.26 | 1.87 | 1.75 | -20.0 | -7.46 | {'length_scale': 1.1, 'seed': 40686} | Ils ne l'ont jamais revu. | 0.0 |
| N05 | narrator | Lui ne les a plus jamais quittés. | Lui ne les a plus jamais quittés. | 0.059 | 1.78 | 2.39 | 2.00 | -20.0 | -5.62 | {'length_scale': 1.1, 'seed': 96520} | Lui ne les a plus jamais quittés. | 0.0 |
| N06 | narrator | Les organisateurs jurent que cette pièce ne fait pas partie du jeu. | Les organisateurs jurent que cette pièce ne fait pas partie du jeu. | 0.060 | 3.89 | 4.50 | 4.15 | -20.0 | -6.04 | {'length_scale': 1.1, 'seed': 87330} | Les organisateurs jurent que cette pièce ne fait pas partie du jeu. | 0.0 |
| N07a | narrator | À 22h12, Dogan s'est mis au piano. | À vingt-deux heures douze, Dogane s'est mis au piano. | 0.060 | 2.69 ⚠ | 3.30 | 2.64 | -20.0 | -6.12 | {'length_scale': 0.94, 'seed': 64376, 'tempo': 1.1} | A 22h12, Dogen se nige au piano. | 0.273 |
| N07b | narrator | Il ne se souvient pas d'avoir joué. | Il ne se souvient pas d'avoir joué. | 0.060 | 1.60 ⚠ | 2.21 | 1.56 | -20.0 | -5.81 | {'length_scale': 0.94, 'seed': 20082, 'tempo': 1.1} | Il ne se souvient pas d'avoir joué. | 0.0 |
| N08a | narrator | Sur la photo, ils sont quatre. | Sur la photo, ils sont quatre. | 0.060 | 1.92 | 2.53 | 1.93 | -20.0 | -6.11 | {'length_scale': 1.1, 'seed': 73015, 'tempo': 1.032} | sur la photo I-104. | 0.333 |
| N08b | narrator | Regardez derrière eux. | Regardez derrière eux. | 0.062 | 1.56 | 2.18 | 1.57 | -20.0 | -5.23 | {'length_scale': 1.1, 'seed': 82573, 'tempo': 1.032} | Regardez derrière eux. | 0.0 |
| N09a | narrator | À 23h, tout était fini. | À vingt-trois heures, tout était fini. | 0.060 | 2.34 | 2.95 | 2.58 | -20.0 | -3.59 | {'length_scale': 1.1, 'seed': 25854} | A 23 heures, tout était fini. | 0.0 |
| N09b | narrator | Ils riaient. | Ils riaient. | 0.060 | 0.61 | 1.22 | 0.92 | -20.0 | -6.64 | {'length_scale': 1.1, 'seed': 51484} | Il riait. | 0.0 |
| E_DOGAN | entity | Dogan… | Dogane. | 0.653 | 0.55 | 2.30 | 0.55 | -20.08 | -3.14 | {'length_scale': 0.89, 'seed': 64662, 'tempo': 1.1} | Ducan. | 1.0 |
| E_GABRIEL | entity | Gabriel… | Gabriel. | 0.654 | 0.55 | 2.25 | 0.55 | -20.0 | -3.56 | {'length_scale': 0.85, 'seed': 85646, 'tempo': 1.257} | D'abriens. | 2.0 |
| E_HUGO | entity | Hugo… | Hugo. | 0.653 | 0.41 | 2.16 | 0.55 | -20.0 | -4.33 | {'length_scale': 1.0, 'seed': 95316} | Et toi ! | 2.0 |
| E_ARSENE | entity | Arsène… | Arsène. | 0.653 | 0.75 | 2.41 | 0.85 | -20.0 | -5.39 | {'length_scale': 1.0, 'seed': 16719} | Affronter. | 1.0 |
| E_IDENT | entity | Identification terminée. | Identification terminée. | 1.052 | 1.41 | 3.90 | 1.42 | -20.0 | -3.1 | {'length_scale': 0.893, 'seed': 21189, 'tempo': 1.036} | Identification terminée. | 0.0 |
| E_CQL | entity | C'est qui lui ? | C'est qui lui ? | 0.484 | 0.76 | 1.56 | 0.97 | -20.0 | -7.58 | {'length_scale': 1.0, 'seed': 74375} | C'est qui lui ? | 0.0 |

### Entité : ASR par couche

| ID | ASR couche « oreille » (chuchotée, −1,5 à −2,5 dt, formants préservés) | ASR mix final | ASR TTS brut |
|---|---|---|---|
| E_DOGAN | Ducando | Ducan. | Dugan |
| E_GABRIEL | Gabriel | D'abriens. | Gabriel |
| E_HUGO | Indo | Et toi ! | et deux |
| E_ARSENE | Arsène | Affronter. | Arsène |
| E_IDENT | Identification terminée. | Identification terminée. | Identification terminée. |
| E_CQL | C'est qui lui ? | C'est qui lui ? | C'est qui lui ? |

## Fenêtres du montage (edit/edl_spec.py)

Fenêtre = du début de parole jusqu'à la voix suivante (ou fin de shot `silence`, ou fin du film).

| shot | voix | début abs. approx. (s) | fenêtre (s) | parole + pauses livrées (s) | marge (s) | au débit préféré (s) | il manquerait (s) |
|---|---|---|---|---|---|---|---|
| S2b | N01 | 6.30 | 5.05 | 4.88 | +0.17 | 6.17 | 1.27 |
| S2d | N02 | 11.35 | 2.15 | 1.77 | +0.38 | 1.80 |  |
| S2e | N03 | 13.50 | 2.10 | 1.71 | +0.39 | 1.70 |  |
| S3b | N04 | 35.60 | 1.90 | 1.26 | +0.64 | 1.23 |  |
| S3c | N05 | 37.50 | 2.15 | 1.78 | +0.36 | 1.76 |  |
| S4b | E_IDENT | 44.60 | 4.50 | 1.41 | +3.09 | 1.49 |  |
| S4e | N06 | 49.10 | 4.30 | 3.89 | +0.41 | 3.61 |  |
| S4h | N07a + N07b | 56.70 | 4.75 | 4.64 | +0.11 | 5.84 | 1.24 |
| S4i | E_DOGAN | 61.45 | 0.55 | 0.55 | +0.00 | 0.61 | 0.06 |
| S4j | E_GABRIEL | 62.00 | 0.55 | 0.55 | +0.00 | 0.74 | 0.19 |
| S4k | E_HUGO | 62.55 | 0.55 | 0.41 | +0.14 | 0.35 |  |
| S4l | E_ARSENE | 63.10 | 0.85 | 0.75 | +0.10 | 0.68 |  |
| S4m | N08a + N08b | 63.95 | 4.10 | 3.94 | +0.16 | 4.03 | 0.08 |
| S5a | N09a + N09b | 71.65 | 3.70 | 3.35 | +0.35 | 3.21 |  |
| S6d | E_CQL | 84.25 | 1.05 | 0.76 | +0.29 | 0.72 |  |
| S7a | E_IDENT | 90.80 | 1.50 | 1.41 | +0.09 | 1.49 | 0.07 |

« il manquerait » : allongement de la fenêtre (shot ou position de la voix suivante) qui permettrait de garder le débit lent préféré sans compression ; relancer alors `tools/make_voices.py`, qui s'adapte automatiquement à `edit/edl_spec.py`.

Durée totale estimée de la spec : 92.3 s (approximative : ancres non résolues).

## Spectrogrammes

`spectrograms/narrator.png`, `spectrograms/entity.png` (traits cyan : début et fin de parole mesurés).

## Limites connues
* Les durées de parole sont contraintes par les fenêtres de `edit/edl_spec.py` (voir tableau) : le débit
  « lent » est conservé partout où la fenêtre le permet ; les phrases N01 et N07a+N07b sont au débit naturel.
  Allonger S2b/S2c (≈ +0,4 s) et S4h (≈ +0,5 s) permettrait un débit plus lent.
* Les prénoms de l'entité doivent tenir dans les flashs de 0,55 s (S4i–S4l) : murmures courts ; leurs queues
  et pré-swells débordent volontairement sur les flashs voisins.
* Voix piper = repli hors ligne ; sur le PC du client, relancer `python tools/make_voices.py` produit la version
  edge-tts (même format, même manifeste), puis relancer `edit/build_edl.py`.
