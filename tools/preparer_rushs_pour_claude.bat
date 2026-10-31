@echo off
REM ==========================================================================
REM  C'EST QUI LUI ? - prepare des copies LEGERES des rushs pour Claude (cloud)
REM  Double-clique ce fichier. Les originaux ne sont JAMAIS modifies.
REM  Resultat : C:\Users\dogan\edit_escape_tmp\rushs_legers  (22 fichiers < 25 Mo)
REM  Ensuite : github.com/Dogansvl/claude-credit > branche claude/cest-qui-lui-trailer
REM            > Add file > Upload files > glisse les 22 fichiers dans un dossier "rushs/"
REM
REM  Videos : 1080x1920, 30 i/s, rotation appliquee, debit plafonne selon la duree
REM  (taille visee ~22 Mo maxi). Rushs HDR (HLG/Dolby Vision) : gardes en HDR 10 bits
REM  (HEVC) ; la conversion en SDR est faite ensuite par la chaine (meme reglage partout).
REM  Photos : copiees telles quelles.
REM ==========================================================================
chcp 65001 >nul
setlocal EnableDelayedExpansion

set "SRC=C:\Users\dogan\OneDrive\Bureau\Documents\Photos, Souvenirs et Voyages\Montages\Escape game horreur"
set "DST=C:\Users\dogan\edit_escape_tmp\rushs_legers"

where ffmpeg >nul 2>nul
if errorlevel 1 goto :noffmpeg
where ffprobe >nul 2>nul
if errorlevel 1 goto :noffmpeg

if not exist "%SRC%\" (
  echo [ERREUR] Dossier des rushs introuvable :
  echo   %SRC%
  pause
  exit /b 1
)
if not exist "%DST%" mkdir "%DST%"

set "GEOM=fps=30,scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1920"

REM Un seul motif : sous Windows *.MOV trouve aussi les .mov (sinon chaque fichier serait traite deux fois)
REM ffprobe en sortie "default" : la sortie csv ajoute une virgule (matrice de rotation) -> HDR non reconnu
for %%F in ("%SRC%\IMG_83*.MOV") do (
  set "TR="
  set "DUR=1"
  for /f "usebackq delims=" %%T in (`ffprobe -v error -select_streams v:0 -show_entries stream^=color_transfer -of default^=nw^=1:nk^=1 "%%~fF"`) do set "TR=%%T"
  for /f "usebackq tokens=1 delims=." %%D in (`ffprobe -v error -show_entries format^=duration -of default^=nw^=1:nk^=1 "%%~fF"`) do set /a DUR=%%D+1
  REM debit video maxi (kb/s) pour rester sous ~22 Mo : 176000 kbit / duree - son 192k ; plafond 10000
  set /a MR=176000/!DUR!-200
  if !MR! GTR 10000 set "MR=10000"
  if !MR! LSS 1500 set "MR=1500"
  set /a BUF=2*!MR!
  echo.
  echo === %%~nF  [transfert: !TR!, !DUR! s, max !MR! kb/s]
  set "HDR=0"
  if /i "!TR!"=="arib-std-b67" set "HDR=1"
  if /i "!TR!"=="smpte2084" set "HDR=1"
  if "!HDR!"=="1" (
    ffmpeg -y -v error -stats -i "%%~fF" -map 0:v:0 -map 0:a:0? -vf "!GEOM!,format=yuv420p10le" -c:v libx265 -preset medium -crf 20 -pix_fmt yuv420p10le -tag:v hvc1 -x265-params "vbv-maxrate=!MR!:vbv-bufsize=!BUF!:colorprim=bt2020:transfer=!TR!:colormatrix=bt2020nc:range=limited:log-level=error" -color_primaries bt2020 -color_trc !TR! -colorspace bt2020nc -color_range tv -c:a aac -b:a 192k -ar 48000 -ac 2 -movflags +faststart "%DST%\%%~nF.mp4"
  ) else (
    ffmpeg -y -v error -stats -i "%%~fF" -map 0:v:0 -map 0:a:0? -vf "!GEOM!,format=yuv420p" -c:v libx264 -preset slow -crf 19 -maxrate !MR!k -bufsize !BUF!k -c:a aac -b:a 192k -ar 48000 -ac 2 -movflags +faststart "%DST%\%%~nF.mp4"
  )
  if errorlevel 1 echo [ERREUR] conversion de %%~nxF echouee
)

for %%P in ("%SRC%\IMG_835*.jpeg" "%SRC%\IMG_835*.jpg" "%SRC%\IMG_835*.HEIC") do (
  if exist "%%~fP" copy /y "%%~fP" "%DST%\" >nul
)

echo.
echo Termine. Fichiers prets dans :
echo   %DST%
dir /b "%DST%"
for %%S in ("%DST%\*.*") do (
  if %%~zS GTR 25000000 echo [ATTENTION] %%~nxS depasse 25 Mo : GitHub ^(envoi par le site^) le refusera.
)
echo.
echo Etape suivante : envoie ces fichiers dans le dossier "rushs/" de la branche
echo claude/cest-qui-lui-trailer sur github.com/Dogansvl/claude-credit (Add file ^> Upload files),
echo puis dis a Claude : "les rushs sont en ligne".
explorer "%DST%"
pause
exit /b 0

:noffmpeg
echo [ERREUR] ffmpeg / ffprobe introuvables.
echo Installe-les en tapant dans un terminal :  winget install Gyan.FFmpeg
echo puis ferme et relance ce script.
pause
exit /b 1
