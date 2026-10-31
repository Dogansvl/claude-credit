# C'EST QUI LUI ? — la bande-annonce, montée par une chaîne automatique

Fausse bande-annonce d'horreur verticale (9:16, 1080×1920, 30 i/s, ~92 s) fabriquée à partir des rushs
de l'escape game du 30/10/2026. Le brief d'origine est dans `docs/BRIEF.md`.

## Ce que fait la chaîne

`run_all.py` enchaîne tout, dans l'ordre (les étapes facultatives `voices` et `title` passent AVANT `edl`,
qui lit les durées réelles des voix et l'instant de révélation du titre) :

| étape | ce qu'elle fait | où ça sort (dossier de travail) |
|---|---|---|
| `models` | télécharge les modèles d'IA manquants (~650 Mo, une seule fois) | `edit_escape_tmp\models\` |
| `ingest` | copies de travail des rushs : 1080×1920, 30 i/s constants, rotation appliquée. **Les originaux ne sont jamais modifiés.** | `proxies\` |
| `analyze` | transcription mot à mot, planches contact 1 image/s, visages, personnes, pics audio, rapport | `analysis\` (`report.md`, `contact\*.jpg`) |
| `edl` | liste de montage complète (coupes calées sur les répliques, voix off, sous-titres, sons) | `edl.json`, `edl_report.md` |
| `render` | rendu image par image : VFX sur les visages, silhouette incrustée, caméra, étalonnage, grain, cartons, HUD, sous-titres | `render\video_preview.mp4` / `video_final.mp4` |
| `mix` | mixage et master du son (−14 LUFS, −1 dBTP, screamer le plus fort) | `render\mix_*.wav` |
| `deliver` | fichier final H.264 + AAC, couverture, LISEZMOI | voir « Où sortent les fichiers » |
| `qc` | contrôle qualité automatique : format, durée, noirs, loudness, flashs, sous-titres, prénoms, transcription (aucune consigne de tournage audible) | `render\qc\qc_report.md`, `render\qc\contact_*.jpg` |

Étapes facultatives (à lancer seulement si besoin) : `voices` (régénère les voix off, déjà fournies dans
`assets\audio\voice`) et `title` (re-rend le titre 3D avec Blender, déjà fourni dans `assets\title`, ~1 h).

## Installation sous Windows (une seule fois)

1. **Python 3.12** : `winget install Python.Python.3.12` (ou python.org en cochant « Add python.exe to PATH »).
2. **ffmpeg** : `winget install Gyan.FFmpeg`, puis **ferme et rouvre** le terminal. Vérifie avec `ffmpeg -version`.
3. **Blender** : facultatif, le titre 3D est déjà rendu. Seulement pour l'étape `title` :
   `winget install BlenderFoundation.Blender`.
4. Récupère le dépôt (branche `claude/cest-qui-lui-trailer`) **HORS de OneDrive**, par exemple dans
   `C:\Users\dogan\edit_escape_tmp\claude-credit` (OneDrive synchroniserait des milliers de petits fichiers de
   travail et peut verrouiller des fichiers pendant le rendu). Puis, dans un terminal ouvert dans ce dossier :

   ```bat
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   À chaque nouvelle session, refais seulement `.venv\Scripts\activate`.
5. **Premier passage sur le PC** : régénère les voix off avec la voix neuronale edge-tts (plus naturelle que les
   voix fournies, faites sans accès à edge-tts) puis la liste de montage :

   ```bat
   python run_all.py --stage voices,edl
   ```

**Rushs OneDrive « disponibles en ligne uniquement »** (icône nuage) : la chaîne les télécharge elle-même en les
lisant (sans les modifier). Pour aller plus vite / travailler hors connexion : clic droit sur le dossier des rushs >
« Toujours conserver sur cet appareil ».

Les chemins par défaut sont déjà ceux de ton PC (`config.py`) : rushs dans
`C:\Users\dogan\OneDrive\Bureau\Documents\Photos, Souvenirs et Voyages\Montages\Escape game horreur`,
travail dans `C:\Users\dogan\edit_escape_tmp`. Rien à régler.

## Fabriquer la bande-annonce

### 1. Préversion (basse définition, 540×960)

```bat
python run_all.py --quality preview
```

La préversion sort dans `C:\Users\dogan\edit_escape_tmp\render\preview\CEST_QUI_LUI_preview.mp4`
(regarde-la sur ton téléphone) avec son contrôle qualité dans `render\qc\qc_report.md`.

### 2. Revue OBLIGATOIRE avant le final

Certaines décisions doivent être validées en regardant les vraies images (qui est qui, à quel instant,
où placer la silhouette…) :

1. Ouvre `C:\Users\dogan\edit_escape_tmp\analysis\report.md` : il liste les clés à vérifier et les valeurs proposées.
2. Regarde les planches `C:\Users\dogan\edit_escape_tmp\analysis\contact\*.jpg` (une image par seconde, horodatée)
   et la planche du film `render\qc\contact_preview.jpg`.
3. Complète `edit\decisions.json` (identités confirmées avec `"confirmed": true`, ordre gauche→droite du HUD,
   instants, positions de la silhouette…), puis mets `"_reviewed": true`.
   Un prénom ne s'affiche à l'écran que si l'identité est confirmée.

Tu peux aussi demander à Claude Code, lancé en local dans le dossier du dépôt (avec accès à
`C:\Users\dogan\edit_escape_tmp`) : « Lis analysis/report.md, regarde les planches analysis/contact/*.jpg et
render/qc/contact_preview.jpg, complète edit/decisions.json et mets _reviewed à true. »

Tant que `_reviewed` vaut `false`, le rendu final refuse de démarrer (pour passer outre en connaissance de
cause : `--ignore-review`).

### 3. Version finale (1080×1920)

```bat
python run_all.py --quality final
```

Les étapes déjà faites (proxies, analyse, shots déjà rendues) sont réutilisées grâce au cache : seul ce qui a
changé est recalculé. Le résumé final indique chaque étape (OK / ÉCHEC / À CORRIGER) et le chemin du journal.

**Durée et place** (mesures sur une machine à 4 cœurs) : première ingestion des 18 rushs 4K ≈ 10–20 min
(HEVC 4K = décodage lourd), analyse ≈ 8 min, préversion ≈ 10–15 min, version finale ≈ 25 min sur 4 cœurs libres
(≈ 40 min si la machine est occupée par ailleurs) : rendu image par image ≈ 0,75 s de calcul par image, puis
encodage H.264 « slow » plafonné à 18 Mb/s (≈ 200 Mo pour 92 s). Un PC à 8 cœurs ou plus va nettement plus vite.
Prévois environ 8 Go libres sur `C:` pour `edit_escape_tmp` (modèles, proxies, cache des shots, rendus).

**Rushs HDR** : si l'iPhone a filmé en HDR (HLG / Dolby Vision, réglage par défaut des iPhone récents), les
copies de travail sont converties en SDR BT.709 (méthode BT.2408 : blanc de référence 203 nits → blanc SDR,
hautes lumières adoucies, teintes conservées). Le format et la conversion sont notés dans
`analysis\inventory.json` (champ `notes` de chaque clip).

### Lancer seulement certaines étapes

```bat
python run_all.py --stage render,mix,deliver,qc --quality preview
python run_all.py --from edl --quality final
python run_all.py --quality preview --force        (recalcule tout, sans le cache)
python run_all.py --quality final --no-asr         (contrôle qualité sans transcription, plus rapide)
python run_all.py --stage voices                   (régénère les voix off)
python run_all.py --stage title                    (re-rend le titre 3D, Blender requis)
```

## Où sortent les fichiers

**Livrables** (version finale uniquement), dans
`...\Escape game horreur\MONTAGE_CEST_QUI_LUI\` :

- `CEST_QUI_LUI_bande-annonce.mp4` : 1080×1920, 30 i/s, H.264 High, yuv420p, AAC 320 kb/s 48 kHz, faststart ;
- `CEST_QUI_LUI_couverture.jpg` : couverture TikTok (la silhouette et le titre) ;
- `LISEZMOI.txt` : 5 lignes sur ce qui a été fait, dont le verdict du contrôle qualité.

**Dossier de travail** `C:\Users\dogan\edit_escape_tmp\` (hors OneDrive, supprimable une fois le film validé) :

- `proxies\` copies de travail ; `analysis\` transcriptions, planches, `report.md` ;
- `edl.json`, `edl_report.md` : la liste de montage (avec `script.md`) ;
- `cache\` : visages suivis, silhouette détourée, cartes de profondeur, shots déjà rendues ;
- `render\` : `video_*.mp4` (image seule), `mix_*.wav` (son seul), `preview\`, `qc\qc_report.md`, `qc\contact_*.jpg` ;
- `logs\` : un journal complet par lancement.

## Variables d'environnement (facultatives)

| variable | rôle | défaut |
|---|---|---|
| `ESCAPE_RUSH_DIR` | dossier des rushs originaux (lecture seule) | le dossier OneDrive ci-dessus |
| `ESCAPE_WORK_DIR` | dossier de travail | `C:\Users\dogan\edit_escape_tmp` |
| `ESCAPE_OUT_DIR` | dossier des livrables | `<rushs>\MONTAGE_CEST_QUI_LUI` (ou `<travail>\MONTAGE_CEST_QUI_LUI` si les rushs sont dans le dépôt) |
| `ESCAPE_MODELS_DIR` | modèles d'IA | `<travail>\models` |
| `ESCAPE_RENDER_JOBS` | shots rendues en parallèle | nombre de cœurs − 1 (entre 1 et 4) |
| `ESCAPE_FFMPEG`, `ESCAPE_FFPROBE` | chemins de ffmpeg / ffprobe s'ils ne sont pas dans le PATH | `ffmpeg`, `ffprobe` |
| `ESCAPE_ASR_THREADS` | threads de la transcription | `2` |

Exemple (PowerShell) : `$env:ESCAPE_RENDER_JOBS = "3"` puis `python run_all.py --quality final`.

## Option cloud

Si tu préfères que Claude fasse le montage dans le cloud plutôt que sur ton PC :

1. Double-clique `tools\preparer_rushs_pour_claude.bat`. Il crée des copies légères (1080×1920, 30 i/s, débit
   plafonné selon la durée : moins de 25 Mo chacune) de tes rushs dans `C:\Users\dogan\edit_escape_tmp\rushs_legers`
   (les originaux ne sont pas touchés ; les rushs HDR restent en HDR 10 bits, la chaîne les convertit ensuite).
2. Sur github.com/Dogansvl/claude-credit, passe sur la branche `claude/cest-qui-lui-trailer`, puis
   **Add file > Upload files** et dépose ces 22 fichiers dans un dossier `rushs/`.
3. Préviens Claude (« les rushs sont en ligne »).

Ce que fait alors Claude (pour mémoire) :

```bash
git pull origin claude/cest-qui-lui-trailer          # récupère rushs/IMG_83xx.mp4 + IMG_835x.jpeg
export ESCAPE_RUSH_DIR="$PWD/rushs" ESCAPE_WORK_DIR=/home/user/edit_escape_tmp
python run_all.py --quality preview                  # ingest (passage direct des .mp4 1080x1920) + analyse + préversion
# revue : analysis/report.md, analysis/contact/*.jpg → edit/decisions.json ("_reviewed": true)
python run_all.py --quality final
```

`config.rush_path` trouve indifféremment `IMG_xxxx.MOV` ou `IMG_xxxx.mp4` (et `.jpeg/.jpg/.HEIC` pour les photos).
Quand les rushs sont dans le dépôt, les livrables vont par défaut dans `<travail>/MONTAGE_CEST_QUI_LUI` (jamais
dans le dépôt : la vidéo dépasse la taille autorisée par GitHub) ; Claude te les transmet ensuite.

## En cas de problème

- **« ffmpeg introuvable »** : refais `winget install Gyan.FFmpeg` puis rouvre le terminal (ou règle `ESCAPE_FFMPEG`).
- **Une étape échoue** : le résumé indique laquelle ; le détail est dans `edit_escape_tmp\logs\`. Relance à partir
  de cette étape avec `--from <étape>`.
- **Contrôle qualité « À CORRIGER »** : ouvre `render\qc\qc_report.md`, chaque ligne en ÉCHEC dit ce qui ne va pas
  (noir non prévu, loudness, mot de consigne entendu, sous-titre hors zone…).
- **Un rush manque** (ex. une photo) : l'ingestion le signale (`MANQUANT`), le montage le remplace (autre photo
  du groupe, ou noir signalé dans `edl_report.md`) au lieu de planter. Dépose le fichier et relance `--from ingest`.
- **Tests techniques** : `python -m unittest tests.test_render tests.test_robustness -v`.
