# Architecture — pipeline « C'EST QUI LUI ? »

Python 3.12 (venv `/home/user/venv` ici ; Windows : `py -3.12 -m venv .venv`). ffmpeg ≥ 6, Blender ≥ 4.0 (titre 3D uniquement).
Tous les chemins viennent de `config.py` (surchargeables par variables d'environnement). **Jamais** de chemin en dur, utiliser `pathlib`.
Le code doit tourner sous Linux ET Windows (pas de commandes shell propres à Linux ; ffmpeg appelé via `subprocess` avec une liste d'arguments).

## Flux
```
rushs (lecture seule)
  └─ pipeline/ingest.py      → proxies 1080x1920 30 fps CFR (H.264 CRF 14, AAC 48 kHz stéréo), photos PNG orientées
  └─ pipeline/analyze.py     → analysis/: transcriptions mot à mot, planches contact 1 i/s, visages, personnes, pics audio, report.md
edit/edl_spec.py + edit/decisions.json + analysis + assets/audio/voice/manifest.json
  └─ edit/build_edl.py       → edit/edl.json (temps absolus, toutes ancres résolues, sous-titres, cues, musique)
  └─ pipeline/render.py      → rendu vidéo (par shot, avec cache) → render/video_{preview|final}.mp4 (sans son)
  └─ pipeline/audio/mixer.py → render/mix_{preview|final}.wav (master −14 LUFS, −1 dBTP)
  └─ pipeline/deliver.py     → OUT_DIR/CEST_QUI_LUI_bande-annonce.mp4, couverture.jpg, LISEZMOI.txt
  └─ pipeline/qc.py          → render/qc/: rapport, planche 1 i/s du film, mesures son, ASR liste noire, flashs, noirs
run_all.py : enchaîne tout (`python run_all.py --stage all --quality preview|final`).
```

## Conventions communes
- Image = `numpy.ndarray` float32 RGB dans [0,1], forme (H, W, 3), H=1920, W=1080, sauf mention contraire.
- Audio = `numpy.ndarray` float32 (N, 2) stéréo, 48 kHz, dans [-1,1].
- Temps en secondes (float). Index d'image = round(t * 30).
- Tout ce qui est aléatoire est déterministe (graine explicite, ou dérivée de l'index d'image).
- Caches dans `config.CACHE_DIR/<module>/...` ; une clé de cache = hash des paramètres.

## Modules et interfaces (contrat entre les modules)

### pipeline/ingest.py
- `probe(path) -> dict` : durée, fps moyen / r_frame_rate, codec, rotation, dimensions, flux audio.
- `make_proxy(clip: str, force=False) -> Path` : 1080x1920 (rotation appliquée, `scale` + `pad`/`crop` si le ratio n'est pas exactement 9:16), `fps=30` CFR (`-vf fps=30` : resampling temporel correct pour 120 fps et ~16 fps VFR), `-pix_fmt yuv420p`, H.264 CRF 14 preset medium, AAC 48 kHz stéréo 256k (silence ajouté si pas d'audio), durée vidéo = durée audio.
- `make_photo_proxy(photo: str) -> Path` : PNG orienté selon l'EXIF (iPhone), hauteur max 2560, sRGB.
- `run_ingest() -> dict` : inventaire `analysis/inventory.json`.

### pipeline/analyze.py
- `transcribe(clip) -> dict` écrit `analysis/<clip>.transcript.json` :
  `{"clip","backend","language":"fr","words":[{"w":"tu","start":0.41,"end":0.55,"p":0.9}],"segments":[{"start","end","text"}]}`
  Backends dans l'ordre : faster-whisper (large-v3/medium, `word_timestamps=True`) → openai-whisper → sherpa-onnx (modèles GitHub, ici) ; le meilleur disponible.
- `find_phrase(clip, phrase, edge='start', occ=0, window=None) -> float | None` : recherche floue (normalisation accents/ponctuation/apostrophes, difflib) dans les mots.
- `contact_sheet(clip) -> Path` : `analysis/contact/<clip>.jpg`, 1 image/s, horodatée.
- `detect_faces(clip, step_s=0.2) -> dict` : `analysis/<clip>.faces.json` `{"t":[...], "faces":[[{"bbox":[x0,y0,x1,y1] (px proxy),"score"}...]...]}`.
- `detect_persons(clip, step_s=0.2) -> dict` : idem `analysis/<clip>.persons.json` (efficientdet / segmentation).
- `audio_peaks(clip) -> dict` : enveloppe RMS 10 ms + pics, `analysis/<clip>.audio.json`.
- `flash_frames(clip)` : luminance moyenne par image (pour trouver les éclairs de 8341).
- `run_analysis() -> Path` : tout + `analysis/report.md` (transcriptions, timecodes des répliques du brief, consignes repérées, visages par clip, planches).

### edit/build_edl.py
- `build(spec_module, decisions: dict, analysis_dir, voice_manifest) -> dict` ; écrit `edit/edl.json`.
- Résout : ancres `A(...)` via `find_phrase` (sinon fallback, signalé), `decision:<clé>` (y compris `<clé>.t`, `<clé>.clip`, `<clé>+1.3`), valeurs `{"auto": ...}` (`person_peak`, `person_center`, `flash_peak`, `audio_peak`, `face_largest`), `after:N07a:0.35`, `voice_end:N05:-0.35`, `title:reveal` (assets/title/title_meta.json).
- Calcule start/dur absolus de chaque shot, place voix (début de PAROLE = `t` ; le fichier commence à `t - onset_s`), sous-titres (voix off : depuis le manifeste, découpés en blocs de ≤ 2 lignes ; répliques : depuis les ancres), cues SFX (absolus), lits musicaux (absolus), blips HUD auto, impacts légers à chaque coupe du montage, masque de silence (`silence=True`).
- Masques de son source : coupe tout mot de `config.INSTRUCTION_BLACKLIST` trouvé dans la plage utilisée (en plus de `mute_src`).
- Vérifie la durée totale (85–95 s), les chevauchements voix/voix et voix/réplique, et écrit `edit/edl_report.md`.

Format `edl.json` :
```json
{"meta":{"fps":30,"w":1080,"h":1920,"duration":92.1,"built_with":{"transcripts":true,"voice_manifest":true}},
 "shots":[{"id":"S1a","section":1,"kind":"clip","start":0.0,"dur":2.1,"clip":"IMG_8346","src_in":5.62,"speed":1.0,
           "freeze_after":null,"camera":{...},"grade":{...},"fx":[{"type":"glitch","t0":0.0,"t1":0.07,...}],
           "overlays":[...],"silhouette":null,"audio":{"src_gain_db":1.0,"mute_src":false,"src_fx":[],"mute_ranges":[]}}],
 "voices":[{"id":"N01","file":"assets/audio/voice/N01.wav","t_file":5.1,"t_speech":5.35,"dur":6.0,"gain_db":0}],
 "subtitles":[{"start":5.35,"end":8.1,"text":"Le 30 octobre 2026, quatre amis","kind":"narrator"}],
 "cues":[{"t":2.1,"sfx":"impact_big","gain_db":-1,"dur":null,"params":{}}],
 "music":[{"kind":"drone_low","t0":0.0,"t1":3.0,"gain_db":-24,"fade_in":0.4,"fade_out":0.3,"duck":0,"automation":[],"params":{}}],
 "silence":[[85.1,86.1]],
 "warnings":[]}
```

### pipeline/gfx.py (motion design + étalonnage)
- `font(name: str, size: int, weight: int | None = None) -> PIL.ImageFont` — familles : `"inter"`, `"archivo"`, `"bodoni"`, `"cormorant"`, `"plexmono"`, `"oswald"`… (fichiers dans assets/fonts, axes variables gérés).
- `grade(img, preset: str, mix: float = 0.0, **kw) -> img` — presets `cold` (look principal : froid, désaturé, noirs écrasés, rouges préservés), `night_red` (habitacle rouge, rouges gardés), `blue` (accueil bleu), `warm` (Burger King chaud saturé), `warm_to_cold` (mix 0→1), `bleach` (screamer/arrêts sur image : contraste dur, désaturé). kw : `red_accent`, `keep_blue`, `warm_highlights`.
- `film_finish(img, frame_idx: int, grain=0.055, vignette=0.38, halation=0.22) -> img` — grain argentique temporel, vignettage, halos.
- `glitch(img, frame_idx, amount)`, `rgb_split(img, amount, frame_idx)`, `negative(img)`, `darken(img, amount)`.
- `draw_card(img|None, text, t, dur, style='card'|'card_over') -> img` — carton de cinéma sobre (A24/Blumhouse) : capitales, approche large, apparition/disparition douces + léger scintillement.
- `draw_name(img, name, t) -> img` — prénom en arrêt sur image.
- `draw_timecode(img, hhmm: str, date: str, t: float) -> img` — horodatage caméscope sobre (`● 20:48  30.10.2026`), en haut à gauche, dans la zone sûre.
- `draw_subtitle(img, text, kind='dialogue'|'narrator'|'entity', alpha=1.0) -> img` — zone sûre TikTok : texte centré dans x∈[60, 1080−180], base ≤ 0,78·H (jamais dans les 20 % du bas), ≤ 2 lignes, ≤ 34 caractères/ligne, contour + ombre douce.
- `draw_hud_face(img, bbox, t_since_lock: float, label: str, sublabel: str, frame_idx, mode='identify'|'victim', landmarks=None) -> img` — HUD de reconnaissance faciale (coins, balayage, points du maillage, étiquette `SUJET 01 — DOGAN — IDENTIFIÉ` qui s'écrit, en rouge en mode `victim`).
- `draw_red_circle(img, center_xy, axes_xy, progress: float, seed: int, frantic=False) -> img` — cercle rouge tracé à la main (feutre), animé par `progress` 0→1.
- `draw_title_sub(img, text, t, dur) -> img` — ligne sous le titre final.
- `safe_zone() -> (x0, y0, x1, y1)`.

### pipeline/facefx.py (VFX visages, MediaPipe FaceLandmarker 478 points)
- `class FaceTrack` : `FaceTrack.compute(video_path, t0, t1, fps=30) -> FaceTrack` (mode VIDEO, lissage one-euro, cache npz) ; `.at(i) -> list[Face]` (i = index d'image relatif à t0) ; `FaceTrack.from_image(img_uint8)`.
  `Face` : `.lm` (478,2) px float32, `.bbox` (x0,y0,x1,y1), `.score`.
- Effets (img float32 → img) : `black_eyes(img, face, amount)`, `stretch_smile(img, face, amount)`, `melt(img, face, amount, t)`, `widen_eyes(img, face, amount)`, `demon_face(img, face, amount, t)`, `head_tic(img, face|None, frame_idx, amount)`, `portrait_eyes(img, faces, t, amount)`.
- Les effets doivent rester crédibles (bords adoucis, grain conservé) et ne jamais planter si aucun visage n'est trouvé (retourner l'image inchangée).

### pipeline/silhouette.py
- `extract(video_path, t_range, out_dir, hint_center=None) -> dict` : détoure la silhouette de 8346 (rembg u2net_human_seg / isnet, efficientdet pour la boîte, MediaPipe en secours) ; écrit `still.png` RGBA (meilleure image) + `frames/%03d.png` ; renvoie `{"still": path, "bbox": [...], "foot": [x,y], "height_px": h, "method": ...}`.
  Secours si échec : silhouette procédurale crédible (tête, épaules, manteau long), marquée `method="procedural"`.
- `composite(img, sil_rgba, x, foot_y, height, opacity=0.9, blur=1.5, darkness=0.9, blend='normal'|'multiply', frame_idx=0, occluder_mask=None) -> img` : incrustation raccord (assombrie, désaturée, flou de profondeur, grain, légère respiration/oscillation) ; `occluder_mask` (H,W) pour la passer DERRIÈRE les personnes.
- `person_mask(img) -> np.ndarray` (H,W) float32 : masque des personnes au premier plan (pour l'occlusion).

### pipeline/parallax.py
- `depth(img) -> np.ndarray` (H,W) float32, 1 = proche (MiDaS small ONNX, cache).
- `render_parallax(img, depth, t: float, dur: float, zoom=(1.0,1.16), center=(0.5,0.45), amount=1.0, out_size=(1080,1920)) -> img` : une image à l'instant t (poussée 2,5D avec décalage par couches selon la profondeur, bords comblés).

### pipeline/audio/dsp.py
`load(path) -> (N,2)` (rééchantillonné 48 kHz), `save(path, x)`, `db(x)`, `gain(x, db)`, `fade(x, fi, fo)`, `pad_to(x, n)`, `mix_at(dst, src, t)`,
`lowpass/highpass/bandpass(x, hz, order)`, `peaking(x, hz, db, q)`, `shelf(x, hz, db, kind)`, `compress(x, thr_db, ratio, attack, release, makeup)`,
`make_ir(decay_s, predelay=0.02, damping=0.5, seed=0) -> (N,2)`, `reverb(x, ir, wet)`, `reverse_reverb(x, ir, wet)`, `pitch(x, semitones, formant=False)` (rubberband CLI / ffmpeg rubberband / repli librosa-like), `stretch(x, ratio)`, `whisperize(x)`, `detune_double(x, cents, delay_ms)`, `distort(x, drive)`,
`duck(x, key, depth_db, attack, release)`, `lufs(x) -> float`, `true_peak_db(x) -> float`, `limit(x, ceiling_db=-1.0, lookahead_ms=5, release_ms=80)` (true-peak suréchantillonné ×4), `master(x, lufs=-14, tp=-1) -> x`.

### pipeline/audio/synth.py (sound design 100 % synthétisé, libre de droits)
`CATALOG: dict[str, Callable[..., np.ndarray]]` ; `render(name, dur=None, seed=0, **params) -> (N,2)`.
Clés exigées par edl_spec : `impact_big, impact_hit, impact_sub, sub_drop, braam, braam_long, riser, reverse_swell, whoosh, whoosh_rev, freeze_thump, glitch_short, glitch_long, vhs_click, tinnitus, heartbeat, clock_tick, scan_sweep, hud_lock, marker_scribble, drip, creak, car_pass, screamer_stack, piano_note, drone_low, drone_tension, music_box_leitmotif, warm_pad, wind`.
`piano_note` et `music_box_leitmotif` utilisent le vrai piano de 8354 si `pipeline/audio/piano.py` a extrait un motif (sinon motif de secours composé).

### pipeline/audio/piano.py
`extract_leitmotif(proxy_8354) -> dict` (onsets + hauteurs, 4–7 notes, cache JSON) ; `piano_variants(...)` → `slow`, `reverse`, `detuned`, `note` (une note isolée avec longue réverbe), `music_box` (re-synthèse du motif en boîte à musique).

### pipeline/audio/mixer.py
`mix(edl: dict, quality='final') -> Path` : son source des shots (extrait du proxy aux bonnes plages, `src_gain_db`, `mute_src`, `mute_ranges`, `src_fx`, fondus 10 ms aux coupes), voix (manifeste), cues, lits musicaux (avec automation + ducking sous voix/répliques), silences francs (`edl.silence` : TOUT à zéro), puis `master()` : −14 LUFS intégrés, −1 dBTP, le screamer doit être le moment le plus fort (court terme).

### pipeline/render.py
`render(edl: dict, quality='preview'|'final') -> Path` : pour chaque shot, décode le proxy (ffmpeg pipe rawvideo, seek précis), applique vitesse/gel, caméra (zoom/centre/punch, rééchantillonnage Lanczos), effets visages, silhouette, étalonnage, `film_finish`, overlays (timecode, cartons, noms, HUD, cercles, sous-titres), puis encode. Preview = 540x960, final = 1080x1920. Cache par shot (hash de la spec de la shot + versions des modules). Concat sans perte de synchro (nombre d'images exact = round(dur*30)).

### pipeline/deliver.py
`deliver(video, mix) -> dict` : mux final H.264 High, yuv420p, 1080x1920, 30 fps, CRF 16, `-movflags +faststart`, AAC 320 kb/s 48 kHz ; couverture JPG (silhouette + titre) ; LISEZMOI.txt (≤ 5 lignes).

### pipeline/qc.py
`qc(final_mp4) -> dict` : ffprobe (specs exactes), durée 85–95 s, blackdetect (aucun noir involontaire : seuls les noirs prévus dans l'EDL sont tolérés), ebur128 (−14 ±1 LUFS, TP ≤ −1), position du pic court terme (= screamer), ASR du mix final → aucun mot de la liste noire, flashs (> 3 transitions de luminance fortes par seconde = échec), planche 1 i/s du film `render/qc/contact_final.jpg`, sous-titres dans la zone sûre. Rapport `render/qc/qc_report.md`.

## Assets pré-générés (versionnés)
- `assets/audio/voice/*.wav` + `manifest.json` : `{"backend":"...","items":{"N01":{"file":"N01.wav","text":"…(affiché)","tts_text":"…","onset_s":0.05,"speech_dur":5.9,"dur":6.6,"kind":"narrator"}, "E_IDENT":{...,"kind":"entity"}}}`.
- `assets/audio/sfx/*.wav` : aperçus de la bibliothèque (le mixer re-synthétise à la bonne durée).
- `assets/title/title_seq.mp4` (1080x1920, 30 fps, 5,0 s) + `title_meta.json` (`{"reveal": 1.2, ...}`) + `cover_plate.png`.
- `assets/fonts/` (OFL).

## Détails de `edl.json` (contrat builder → render/mixer)
- Une shot de `edl.json` = la shot de `edl_spec.py` avec TOUTES les valeurs résolues en nombres/chaînes concrets :
  `decision:*`, `{"auto":...}`, ancres `A(...)`, `"decision:x_t+1.3"`. `start`/`dur` absolus ajoutés. Pour `kind=clip` : `clip`, `src_in`, `speed` (float ou [v0,v1]), `freeze_after` (s, ou null).
- `fx[*].t0/t1` et `overlays[*].t0/t1` : relatifs à la shot, `t1=None` remplacé par `dur`. Les chaînes `voice_end:…`, `after:…`, `title:reveal` sont résolues.
- `freeze_src` → `{"clip": "IMG_8338", "t": 0.31}` (pour `prev_last` : clip et instant de la dernière image de la shot précédente, en tenant compte de la vitesse et du freeze_after).
- `photo` → id (`"IMG_8350"`) ; `montage.items[*]` → `{"clip"|"photo", "src_in", "dur", "fx":[...]}` avec `start` relatif.
- Overlays `name` : `show` booléen (= identité confirmée dans decisions.json) ; si non confirmé → `show:false` (le gel reste, sans prénom).
- Overlays `hud_faces` : `order` → liste de prénoms gauche→droite ou `null` (étiquettes sans prénom : `SUJET 01 — IDENTIFIÉ`). Les boîtes viennent de `FaceTrack` au rendu.
- Overlays `red_circles` : `targets` = `faces` (visages détectés sur la photo au rendu) ou `silhouette` (position du slot).
- `silhouette` : `{"slot": "...", "x","foot_y","height","blur","darkness","blend","occlude_persons","opacity","t0","t1","fade_in","space"}` (slot de decisions.json fusionné).
- `voices` (niveau racine) : `{"id","file","t_file","t_speech","dur","speech_dur","gain_db","kind","shot","allow_in_silence"}`. Les `voice` des shots sont retirées des shots.
- `subtitles` (racine) : `{"start","end","text","kind":"dialogue"|"narrator"|"entity"}` absolus, ≤ 2 lignes de ≤ 34 caractères (retour à la ligne `\n` déjà inséré).
- `cues` (racine) : absolus `{"t","sfx","gain_db","dur","params","shot"}` ; inclut les `hud_lock` auto (à chaque verrouillage HUD) et les `impact_hit` légers (−14 dB) à chaque coupe du montage.
- `music` (racine) : absolus `{"kind","t0","t1","gain_db","fade_in","fade_out","duck","automation":[[t_abs, dB]...],"params"}`.
- `silence` (racine) : plages absolues où TOUT est coupé (musique, SFX, son source), sauf les voix `allow_in_silence=true` (E_CQL de S6d).
- `src_audio` (racine) : liste à plat des segments de son source `{"clip","src_in","src_out","t","speed","gain_db","fade_in","fade_out","src_fx":[...],"mute_ranges":[[t0,t1]...] (absolus timeline)}` déjà prêts pour le mixer.
