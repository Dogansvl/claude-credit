@echo off
REM ==========================================================================
REM  C'EST QUI LUI ? - prepare des copies LEGERES des rushs pour Claude (cloud)
REM  Double-clique ce fichier. Les originaux ne sont JAMAIS modifies.
REM  Resultat : C:\Users\dogan\edit_escape_tmp\rushs_legers  (22 fichiers < 25 Mo)
REM  Ensuite : github.com/Dogansvl/claude-credit > branche claude/cest-qui-lui-trailer
REM            > Add file > Upload files > glisse les 22 fichiers dans un dossier "rushs/"
REM ==========================================================================
chcp 65001 >nul
setlocal EnableDelayedExpansion

set "SRC=C:\Users\dogan\OneDrive\Bureau\Documents\Photos, Souvenirs et Voyages\Montages\Escape game horreur"
set "DST=C:\Users\dogan\edit_escape_tmp\rushs_legers"

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [ERREUR] ffmpeg est introuvable.
  echo Installe-le en tapant dans un terminal :  winget install Gyan.FFmpeg
  echo puis ferme et relance ce script.
  pause
  exit /b 1
)

if not exist "%DST%" mkdir "%DST%"

set "SDR=fps=30,scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1920,format=yuv420p"
set "HDR=fps=30,zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1920,format=yuv420p"

for %%F in ("%SRC%\IMG_83*.MOV" "%SRC%\IMG_83*.mov") do (
  if exist "%%~fF" (
    set "TR="
    for /f "usebackq delims=" %%T in (`ffprobe -v error -select_streams v:0 -show_entries stream^=color_transfer -of csv^=p^=0 "%%~fF"`) do set "TR=%%T"
    set "VF=!SDR!"
    if /i "!TR!"=="arib-std-b67" set "VF=!HDR!"
    if /i "!TR!"=="smpte2084" set "VF=!HDR!"
    echo.
    echo === %%~nF  [transfert: !TR!]
    ffmpeg -y -v error -stats -i "%%~fF" -map 0:v:0 -map 0:a:0? -vf "!VF!" -c:v libx264 -preset slow -crf 19 -maxrate 10M -bufsize 20M -c:a aac -b:a 192k -ar 48000 -ac 2 -movflags +faststart "%DST%\%%~nF.mp4"
  )
)

for %%P in ("%SRC%\IMG_835*.jpeg" "%SRC%\IMG_835*.jpg" "%SRC%\IMG_835*.JPG" "%SRC%\IMG_835*.HEIC") do (
  if exist "%%~fP" copy /y "%%~fP" "%DST%\" >nul
)

echo.
echo Termine. Fichiers prets dans :
echo   %DST%
dir /b "%DST%"
echo.
echo Etape suivante : envoie ces fichiers dans le dossier "rushs/" de la branche
echo claude/cest-qui-lui-trailer sur github.com/Dogansvl/claude-credit (Add file ^> Upload files),
echo puis dis a Claude : "les rushs sont en ligne".
explorer "%DST%"
pause
