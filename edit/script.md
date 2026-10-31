# C'EST QUI LUI ? — script de la bande-annonce

Format : 9:16, 1080×1920, 30 i/s, cible ≈ 92 s (fenêtre 85–95 s).
Timecodes indicatifs : `edit/edl.json` fait foi. Il est généré par `edit/build_edl.py`, qui recale chaque coupe sur la transcription mot à mot.

Légende : **[N]** narrateur (grave, lent, posé, légère réverbe) · **[E]** l'entité (chuchotement très proche, grave, doublé, désaccordé, réverbe inversée) · **[R]** vraie réplique du rush · `SFX` sound design · `TC` timecode incrusté.

---

## 1. ACCROCHE (0 → ~3,2 s)
- **8346** : la silhouette immobile dans les phares. Poussée lente vers elle.
- **[R]** « C'est qui lui ? »
- Arrêt sur image à la fin de la réplique : punch-in brutal sur la silhouette, glitch de 2 images, `SFX` impact + braam + acouphène.
- Coupe au noir, sub-drop.

## 2. LA ROUTE (~3,2 → ~24 s)
- Carton sur noir : **INSPIRÉ DE FAITS RÉELS**. `SFX` le drone grave démarre.
- **8336** route dans le brouillard, feux arrière rouges. Carton **30 OCTOBRE 2026**. `TC 20:40`
  **[N]** « Le 30 octobre 2026, quatre amis ont pris la route pour un escape game. »
- **8342** voie rapide, gyrophares bleus au loin. `TC 20:43`
- **8343** banquette arrière, conducteur et passager de dos.
  **[N]** « Ils se sont filmés toute la soirée. »
- **8345** même axe, la route défile. **[N]** « Voici ce qu'ils ont filmé. » `SFX` clic de cassette.
- Présentation en arrêts sur image (thump + souffle inversé à chaque gel) :
  **GABRIEL** (8341, dans un éclair) · **HUGO** · **ARSÈNE** · **DOGAN** (8338, lumière rouge).
  Un prénom ne s'affiche que sur un plan où l'identité est confirmée (`edit/decisions.json`).
- Le gel de Dogan repart : **[R]** « Tu sais quand il vient ? » (coupe franche) **[R]** « Il vient directement, lui. »
  La musique se retire avant « lui » : silence franc.
- **8338** il hurle, on l'agrippe (0,9 s), impact. Noir.

## 3. LE PARKING (~24 → ~40 s)
- **8346** presque en entier, `TC 20:48`. Poussée lente vers l'ombre du mur tagué.
  **[R]** « T'as entendu le bruit ? » · « Je crois qu'il y a un gars là. » · « C'est qui lui ? » · « Oh putain ! »
  `SFX` drone de tension croissant, cœur de 62 à 118 bpm, riser calé sur « Oh putain ! » → impact + braam long.
- La silhouette recule et disparaît derrière le mur. Gel sur le mur vide, lente poussée, fondu au noir.
  **[N]** « Ils ne l'ont jamais revu. »
- Noir. **[N]** « Lui ne les a plus jamais quittés. » Boom sub sur « quittés » et silhouette subliminale en négatif (3 images).

## 4. L'ESCAPE (~40 → ~73 s)
- **8347** accueil bleu, les quatre devant la porte aux toiles d'araignée. `TC 20:55`
  **[R]** voix robot « Identification terminée. » Le HUD de reconnaissance faciale verrouille chaque visage, un par un :
  `SUJET 01 — DOGAN — IDENTIFIÉ`, etc. (ordre gauche→droite fixé dans `decisions.json`).
- Gel : les cadres virent au rouge, `VICTIME 01…`, les yeux noircissent.
  **[E]** « Identification terminée. » Il vient de choisir ses quatre victimes.
- **8347** le cri face caméra (0,7 s), flash négatif, impact.
- **8348** toilettes maculées de sang. `TC 20:57` **[R]** « C'est quoi ça ? »
- **8348** le corps dans le sac-poubelle ficelé, poussée lente.
  **[N]** « Les organisateurs jurent que cette pièce ne fait pas partie du jeu. »
- **8349** la salle (portraits, armoire, lampe à franges). `TC 21:00` Les yeux des portraits suivent la caméra. Dans le coin sombre, **la silhouette** (incrustée, ~1 s).
- **8349** Gabriel assis, terrifié : tics de tête saccadés.
- **8354** Dogan en haut-de-forme au piano, sous le candélabre. `TC 22:12` Vrai son de piano, qui se désaccorde peu à peu.
  **[N]** « À 22h12, Dogan s'est mis au piano. » … « Il ne se souvient pas d'avoir joué. »
- Flashs de possession, les visages filmés dans la voiture devenus les preuves : yeux noirs, sourire étiré, visage qui fond.
  **[E]** « Dogan… » « Gabriel… » « Hugo… » « Arsène… »
- **Photo de groupe** (8350–8353), poussée 2,5D sur carte de profondeur. Quatre cercles rouges tracés à la main, un par visage.
  **[N]** « Sur la photo, ils sont quatre. » … « Regardez derrière eux. »
  Silence. La silhouette apparaît au fond, dans l'ombre. Braam. Un cinquième cercle, frénétique, autour d'elle.
- Montage qui accélère (coupes de plus en plus courtes), riser, coupe sèche au noir et au silence.

## 5. FAUX HAPPY END (~73 → ~81 s)
- **8355** Burger King décoré pour Halloween, lumière chaude et saturée, fous rires. `TC 23:18`
  **[N]** « À 23h, tout était fini. » … « Ils riaient. »
- **8356** les rires ralentissent et se déforment, l'image se refroidit. Dans le reflet de la vitre, la silhouette.
- Gel sur un visage hilare : le sourire s'étire, les yeux noircissent. Les rires deviennent un drone. Noir.

## 6. FIN (~81 → ~88 s)
- **8357** parking du Burger King, `TC 23:42`. Hugo immobile face caméra. Derrière lui, entre les voitures, la silhouette, plus proche à chaque coupe (3 coupes). Battements de cœur qui ralentissent, acouphène qui monte.
- Silence total d'une seconde. **[E]** chuchoté, collé à l'oreille : « C'est qui lui ? »
- **SCREAMER** : le cri de Dogan collé à l'objectif (8349), yeux noirs, bouche étirée, deux éclairs de la silhouette, impact maximal (le moment le plus fort du film). Noir.

## 7. TITRE (~88 → ~92 s)
- Titre en 3D **C'EST QUI LUI ?** dans le brouillard, balayé par des phares.
- **INSPIRÉ DE FAITS RÉELS — 30.10.2026**
- Dernier sursaut : une note du vrai piano, et **[E]** « Identification terminée. »

---

### Textes du narrateur (orthographe des sous-titres)
| id | texte affiché | texte prononcé (TTS) |
|---|---|---|
| N01 | Le 30 octobre 2026, quatre amis ont pris la route pour un escape game. | Le trente octobre deux mille vingt-six, quatre amis ont pris la route pour un escape game. |
| N02 | Ils se sont filmés toute la soirée. | idem |
| N03 | Voici ce qu'ils ont filmé. | idem |
| N04 | Ils ne l'ont jamais revu. | idem |
| N05 | Lui ne les a plus jamais quittés. | idem |
| N06 | Les organisateurs jurent que cette pièce ne fait pas partie du jeu. | idem |
| N07a | À 22h12, Dogan s'est mis au piano. | À vingt-deux heures douze, Dogane s'est mis au piano. |
| N07b | Il ne se souvient pas d'avoir joué. | idem |
| N08a | Sur la photo, ils sont quatre. | idem |
| N08b | Regardez derrière eux. | idem |
| N09a | À 23h, tout était fini. | À vingt-trois heures, tout était fini. |
| N09b | Ils riaient. | idem |

### L'entité
| id | texte |
|---|---|
| E_DOGAN | Dogan… |
| E_GABRIEL | Gabriel… |
| E_HUGO | Hugo… |
| E_ARSENE | Arsène… |
| E_IDENT | Identification terminée. |
| E_CQL | C'est qui lui ? |

### Règles
- Consignes de tournage (« fais genre… », 8340 et 8344 surtout) : son source coupé sur ces plans, liste noire vérifiée par ASR sur le mix final.
- Prénoms exacts : Dogan, Gabriel, Hugo, Arsène. Pas de prénom à l'écran si l'identité n'est pas confirmée.
- Sous-titres dans la zone sûre TikTok : ni dans les 20 % du bas, ni dans la marge droite.
- Flashs : jamais plus de 3 par seconde.
