# `agent-kiki` — Design

**Date :** 2026-05-04
**Statut :** Design validé, prêt pour plan d'implémentation
**Auteur :** Clément Saillant
**Brainstorming :** session interactive (8 sections validées)
**Spec parent :** `docs/specs/2026-04-26-eu-kiki-design.md`

---

## 0. Sommaire

`agent-kiki` (alias `aki`) est une **CLI agentique** qui transforme une tâche en langage naturel en code écrit/édité sur disque, en pilotant l'infrastructure `eu-kiki` (gateway FastAPI + 3 workers MLX + adapters LoRA domaine-spécifiques).

L'agent est **full-autonomous** (boucle ReAct sans humain à chaque tour, sauf approbations sécurité), **adaptatif** (s'ajuste au mode SCRATCH dossier vide ou EDIT codebase existant), et **multi-modèle interne** : Apertus 70B planifie, Devstral 24B 4-bit code, EuroLLM 22B compresse le contexte long. Il logue chaque tour en JSONL pour permettre, à terme, le fine-tune d'un adapter `agent-react` sur les traces réelles (V1).

Périmètre V0 : outil interne, MacStudio M3 Ultra, single-shot, 7 tools, 3 zones de sécurité, traces complètes.

---

## 1. Identité, périmètre, emplacement

| Champ | Valeur |
|---|---|
| Nom | `agent-kiki` (binaire principal) |
| Alias | `aki` (lien symbolique vers le même binaire) |
| Repo GitHub | `L-electron-Rare/agent-kiki` (**privé**, séparé d'eu-kiki) |
| Path local | `/Users/electron/Documents/Projets/agent-kiki/` |
| Couplage à eu-kiki | **Aucune dépendance Python** — communication via HTTP OpenAI-compatible (gateway `:9200`). Repo, CI, secrets et pyproject.toml indépendants. |
| Statut produit | Outil interne (lab + Clément), pas de release publique au MVP |
| Cible matérielle MVP | MacStudio M3 Ultra (workers MLX) ; client local sur GrosMac/macM1 via SSH/Tailscale |
| Langage | Python 3.13+, géré par `uv`, layout `src/agent_kiki/`, `pyproject.toml` autonome, `hatchling` build (cohérence stack maison sans partage de lockfile) |

**Hors-scope explicite (V0) :**
- Pas d'UI graphique, pas de TUI riche (juste CLI structuré + couleurs sobres via `rich`)
- Pas de codebase RAG/embeddings — exploration via `list_dir` + `read_file` + `search` ripgrep
- Pas de fine-tuning d'adapter `agent-react` (déféré V1, alimenté par traces collectées)
- Pas de génération multi-langue cross-fichier dans un même run (un domaine dominant par run)
- Pas de mode watch/daemon — invocation atomique
- Pas de REPL au MVP — single-shot uniquement (REPL pollue le dataset V1)
- Pas de portage Linux/Windows

---

## 2. Architecture haut-niveau & flux de données

```
┌────────────────────────────────────┐
│  agent-kiki CLI (local)            │
│  ────────────────────────────      │
│  • orchestrator (boucle ReAct)     │
│  • tool executor (local)           │
│  • mode detector (SCRATCH/EDIT)    │
│  • approval gate (3 zones)         │
│  • trace logger (JSONL)            │
└──────────────┬─────────────────────┘
               │ HTTPS POST /v1/chat/completions
               │ Headers : X-Eu-Kiki-Hint, X-Eu-Kiki-Role
               ▼
┌────────────────────────────────────────────────┐
│  eu-kiki gateway  studio:9200                  │
│  ────────────────────────                      │
│  • Jina v3 router (auto-domain)                │
│  • domain hint override (X-Eu-Kiki-Hint)       │
│  • role-based routing :                        │
│    - role=plan  → Apertus 70B  :9301           │
│    - role=code  → Devstral 24B :9302           │
│    - role=lang  → EuroLLM 22B  :9303           │
└────────────────────────────────────────────────┘
```

**Flux d'un tour :**

1. CLI reçoit `task` + détecte `mode` (cwd vide, repo git, projet, mixed).
2. **Phase plan** → POST gateway avec `role=plan`, hint domaine → Apertus retourne `<thought>` + `<tool_call>{...}</tool_call>`.
3. Parser strict côté client (regex sur balises XML + `json.loads` + `jsonschema`).
   - Format invalide → 1 retry avec correction prompt.
   - 2e échec → abort run avec exit 2 et trace complète.
4. Approval gate selon `--safe`/`--auto`/`--yolo` et zone du tool.
5. **Exécution tool** :
   - `read_file`, `list_dir`, `search`, `run_cmd`, `finish` → exécution locale, pas d'appel LLM.
   - `write_file`, `edit_file` → 2nd appel gateway avec `role=code` + hint domaine → Devstral retourne le code/edit, le wrapper l'écrit/applique.
6. **Trace JSONL** flushée (1 ligne par tour dans `trace.jsonl`, 1 ligne dans `codegen.jsonl` si délégation, 1 ligne dans `approvals.jsonl` si user prompt).
7. Boucle vers étape 2 jusqu'à `finish` ou dépassement budget.

**Décisions structurantes :**

- **Apertus ne génère jamais le code lui-même.** Il décide *quoi* faire (intentions), Devstral décide *comment* l'écrire (contenu). Apertus reste aveugle au code complet sauf si un futur `read_file` le lui montre — son contexte tient sur le plan.
- **Routage par headers HTTP**, pas de modif du gateway eu-kiki au MVP. `extra_headers` du protocole OpenAI-compatible est utilisé. Patch trivial dans le gateway si les headers ne sont pas forwardés au routeur (à valider en intégration).
- **Pas de streaming** au MVP. Le format ReAct se parse complet, streaming = piège côté parsing partiel.
- **Trois usages distincts du multi-modèle** : Apertus planifie, Devstral code, EuroLLM compresse les anciens tours quand le contexte d'Apertus sature. C'est la justification fonctionnelle de l'archi multi-worker d'eu-kiki.

---

## 3. Boucle agentique : format ReAct

**Format imposé à Apertus** (system prompt strict, few-shot exemples) :

```xml
<thought>
J'ai besoin de voir la structure du projet d'abord.
</thought>
<tool_call>
{"name": "list_dir", "arguments": {"path": "."}}
</tool_call>
```

**Choix XML enveloppe + JSON args (option D du brainstorming) :**

- ✅ Robustesse XML pour l'enveloppe : parser regex tolérant `<thought>(.+?)</thought>` + `<tool_call>(.+?)</tool_call>`, récupère même balises mal fermées.
- ✅ Rigueur JSON pour les args : validation `jsonschema` strict, types préservés.
- ✅ JSON ne contient **jamais le code lui-même** (Apertus n'émet que les intentions ; c'est Devstral qui produit le code dans une 2e requête sans wrapper). Donc le risque d'échappement multi-lignes JSON disparaît.
- ✅ Dataset V1 réutilisable : la partie `tool_call` JSON est un format standard ré-injectable dans un trainer LoRA.

**Markdown réservé aux artefacts humains :**
- Plan haut-niveau initial → `## Plan` Markdown rendu coloré + écrit dans `plan.md`.
- Run report final → `## Run Report` Markdown + écrit dans `report.md`.

**Retry sur format invalide :**
- Tour 1 : Apertus produit du contenu non-parseable.
- Tour 1bis : retry avec message *"Ton précédent message n'a pas pu être parsé : `<erreur>`. Reformule en respectant strictement le format `<thought>...</thought><tool_call>{...}</tool_call>`."*
- Tour 1ter : si échec encore, abort run avec exit code 2 et trace complète. Cas typique à analyser pour V1 (signal de fragilité du prompt).

**Pas d'auto-correction agressive** : on préfère échouer proprement et logger plutôt que de tordre le format à la volée et masquer un défaut sous-jacent.

---

## 4. Catalogue des 7 tools (V0 figé)

| Tool | Args | Effet | Coût LLM |
|---|---|---|---|
| `list_dir` | `path` | Arbre du dossier (max 200 entrées, troncation explicite : ajout d'une ligne `… N entries truncated` à la fin ; ignore `.git/`, `node_modules/`, `__pycache__/`, `.venv/`, `target/`, `dist/`) | aucun (local) |
| `read_file` | `path`, `offset?`, `limit?` | Contenu fichier (limite : 2000 lignes OU 64 KB, le plus restrictif gagne ; troncation explicite avec marker `… file truncated at L<N>`) | aucun (local) |
| `search` | `pattern`, `path?`, `glob?` | Grep récursif, ripgrep si dispo, fallback Python | aucun (local) |
| `write_file` | `path`, `spec` | Crée nouveau fichier (refuse si existe sauf `overwrite=true`). `spec` est une description NL ; le wrapper appelle Devstral pour générer `content`. | **1 appel `role=code`** |
| `edit_file` | `path`, `old_string`, `new_string` ou `path`, `spec` | Remplacement chirurgical. `old_string` doit être unique dans le fichier (sinon erreur, l'agent retry avec plus de contexte). Variante `spec` : décrit l'edit, le wrapper appelle Devstral pour produire `old_string` + `new_string`. | **1 appel `role=code`** (variante `spec`) |
| `run_cmd` | `cmd`, `cwd?`, `timeout?` | Exécute commande shell. Whitelist + 3 zones (cf §6). Retourne stdout+stderr+exitcode. | aucun (local) |
| `finish` | `summary` | Termine le run avec succès. Écrit `report.md`. | aucun (local) |

**Schémas pydantic** stockés dans `agent_kiki/tools/base.py` ; documentés dans `docs/trace-schema.md` ; exposés à Apertus dans le system prompt sous forme JSON Schema synthétique.

**Contrainte délégation Apertus → Devstral :**
- `write_file` : Apertus émet `{"name": "write_file", "arguments": {"path": "Cargo.toml", "spec": "Manifest Cargo pour parser TOML, dépendance serde, edition 2021, version 0.1.0"}}`. Le wrapper appelle Devstral avec un prompt court : *"Génère le contenu pour `path` selon `spec`. Contraintes du projet : `<résumé>`."* Devstral retourne le contenu complet, le wrapper écrit.
- `edit_file` variante `spec` : même pattern, Devstral retourne `{"old_string": "...", "new_string": "..."}`.
- `edit_file` variante `old_string`/`new_string` directe : Apertus l'a déterminé seul (cas où il connaît exactement le bout à remplacer parce qu'il vient de le lire).

---

## 5. Mode adaptatif

**Heuristique de détection** déclenchée au démarrage, avant le tour 1 :

```python
def detect_mode(cwd: Path) -> Mode:
    if not cwd.exists() or list(cwd.iterdir()) == []:
        return Mode.SCRATCH
    if (cwd / ".git").is_dir():
        return Mode.EDIT_REPO
    if any((cwd / f).exists() for f in PROJECT_MARKERS):
        return Mode.EDIT_PROJECT
    return Mode.MIXED

PROJECT_MARKERS = [
    "pyproject.toml", "package.json", "Cargo.toml",
    "go.mod", "CMakeLists.txt", "Makefile",
]
```

**Profil SCRATCH (dossier vide) :**
- System prompt : *"Tu démarres un projet from scratch. Ton 1er message DOIT être un `## Plan` Markdown listant les fichiers à créer + raison. Puis exécute via les tools."*
- Pas de `read_file` au tour 0 (rien à lire).
- Premier tour quasi-obligé : `write_file` du fichier racine du langage cible.
- Hint domaine inféré du prompt user (mots-clés : "rust" → `rust`, "kicad" → `kicad`, défaut `python`).

**Profil EDIT_REPO / EDIT_PROJECT (codebase existant) :**
- System prompt : *"Tu interviens dans un projet existant. AVANT toute écriture : lis README + manifest + au moins 2 fichiers clés liés à la tâche. Émets ensuite un `## Plan` Markdown qui référence les patterns observés. Puis édite chirurgicalement."*
- Garde-fou côté wrapper : refus du 1er `write_file`/`edit_file` si aucun `read_file` du README ni du manifest n'a eu lieu dans ce run.
- Hint domaine inféré du manifest (pyproject → `python`, Cargo → `rust`, package.json → `typescript`/`javascript`, etc.) ; override possible par `--hint domain=X`.

**Profil MIXED (fichiers en vrac sans marker) :**
- L'agent demande **confirmation interactive** au tour 0 :
  *"Le dossier contient X fichiers sans marker de projet. Tu veux que je (1) traite comme SCRATCH et ignore l'existant, (2) traite comme EDIT et lis tout, (3) annule ?"*
- Seul cas où l'agent stop pour clarification.

**Gestion du contexte long d'Apertus :**
- Fenêtre glissante : maximum **6 derniers tour-pairs** (thought + tool_result) gardés en clair dans le contexte d'Apertus.
- Au-delà, les anciens tours sont résumés en 1 ligne via un appel `role=lang` (EuroLLM 22B → résumé court multilingue).
- Le `## Plan` initial reste constant en système, jamais résumé.
- Stratégie de résumé loggée dans `summaries.jsonl`.

---

## 6. Garde-fous & sandboxing

**Limites dures (configurables via flags) :**

| Limite | Défaut | Flag |
|---|---|---|
| `max_steps` | 30 tours | `--max-steps N` |
| `max_tokens_total` | 200 000 | `--max-tokens N` |
| `max_wallclock_seconds` | 900 (15 min) | `--timeout N` |
| `max_file_writes` | 50 | `--max-writes N` |

Au dépassement → arrêt clean + flush trace + exit code spécifique :
- `4` = max steps
- `5` = max tokens
- `6` = max wallclock
- `7` = max writes

**Filesystem jail :**
- `write_file` / `edit_file` : `path` doit être strictement sous le `cwd` au démarrage. Path traversal refusé. Symlinks résolus avant validation.
- `read_file` : même règle ; flag `--read-outside PATH` pour autoriser un dossier précis hors cwd (le path autorisé est gelé au démarrage et logué).
- Aucune écriture hors cwd, jamais.

**`run_cmd` — 3 zones :**

| Zone | Comportement | Contenu |
|---|---|---|
| **AUTO-OK** | passe sans demander | `pytest`, `uv`, `cargo`, `npm`, `pnpm`, `yarn`, `make`, `cmake`, `go`, `rustc`, `git status`, `git diff`, `git log`, `git show`, `ls`, `cat`, `head`, `tail`, `find`, `wc`, `file`, `black`, `ruff`, `prettier`, `rustfmt`, `gofmt`, `clang-format`, `ctest`, `cargo test`, `npm test`, `go test` |
| **CONFIRM** | y/n interactif | tout le reste : `curl`, `wget`, `git push`, `pip install`, `npm install`, `kicad-cli`, `freecad-cli`, `bun`, `deno`, `make-cad`, etc. |
| **HARD-DENY** | refus systématique sauf `--allow-destructive` | `rm -rf` (avec ou sans `/` ou `~`), `dd of=/dev/`, `mkfs.*`, `shutdown`, `reboot`, fork bomb (`:(){:|:&};:`), `chmod -R 777 /`, `chown -R … /`, `sudo …`, `> /dev/sd*`, `mv … /dev/null`, écriture dans `/etc/`, `/System/`, `/usr/` |

Mécanique : la première chaîne du `cmd` est matchée contre la whitelist AUTO-OK. Pour `git`, sub-command whitelist (status/diff/log/show OK ; push/reset/clean en CONFIRM). HARD-DENY est une regex compilée sur la commande complète.

**Bypass HARD-DENY :** flag `--allow-destructive` (off par défaut) rétrograde HARD-DENY → CONFIRM. La résolution finale dépend ensuite du mode d'approbation actif :
- `--safe` + `--allow-destructive` → confirmation interactive sur chaque commande destructive (cas d'usage : nettoyage manuel intentionnel).
- `--auto` + `--allow-destructive` → confirmation 1re fois par tool, auto ensuite.
- `--yolo` + `--allow-destructive` → **tout auto, y compris destructif**. Combo réservé sandbox jetable. Le warning rouge au démarrage est explicite : *"⚠ DESTRUCTIVE COMMANDS WILL BE AUTO-APPROVED"*.

**Modes d'approbation (3 niveaux) :**

| Mode | Comportement |
|---|---|
| `--safe` (DÉFAUT) | Confirmation interactive Y/n/abort avant **chaque** `write_file`, `edit_file`, `run_cmd` (sauf AUTO-OK pour run_cmd). |
| `--auto` | Auto-approuve `read_file`/`list_dir`/`search`. Demande approbation **uniquement la 1re fois par nom de tool** dans un run (1× pour `write_file`, 1× pour `edit_file`, 1× pour `run_cmd` zone CONFIRM), puis auto pour les occurrences suivantes du même tool. Le tool name est l'unité de mémoire — pas la commande exacte ni les args. |
| `--yolo` | Tout auto-approuvé sauf HARD-DENY. Warning rouge au démarrage. Réservé aux runs en sandbox isolée (Docker/VM/dossier jetable). |

**Kill switch :**
- 1er `Ctrl-C` → termine le tour en cours, flush trace, écrit `RUN_INTERRUPTED.md` avec état + todo restante.
- 2e `Ctrl-C` rapproché (<2s) → arrêt immédiat (sortie d'urgence).

**Trace systématiquement :**
- Toutes les approbations user (Y/n/abort) → `approvals.jsonl`.
- Tous les `run_cmd` refusés par whitelist + raison.
- Tout dépassement budget + ce qui restait à faire.

---

## 7. CLI UX & flags

**Surface principale :**

```bash
agent-kiki [OPTIONS] TASK
aki [OPTIONS] TASK              # alias court
```

`TASK` = positionnel obligatoire, langage naturel (FR ou EN, l'agent route via EuroLLM si nécessaire).

**Catalogue de flags (V0) :**

```
Configuration générale:
  -C, --cwd PATH               Dossier de travail (défaut: cwd actuel)
  --hint domain=DOM            Force le routing eu-kiki (python|rust|kicad|...)
  --gateway URL                URL du gateway eu-kiki (défaut: http://studio:9200)
  --no-emoji                   Sortie ASCII pure
  --no-color                   Désactive les couleurs

Modes d'approbation:
  --safe                       Confirmation à chaque write/edit/run_cmd CONFIRM (DÉFAUT)
  --auto                       Auto sauf 1re occurrence par type
  --yolo                       Tout auto (warning rouge au démarrage)
  --allow-destructive          Rétrograde HARD-DENY en CONFIRM

Limites:
  --max-steps N                Max tours ReAct (défaut: 30)
  --max-tokens N               Max tokens cumulés (défaut: 200000)
  --timeout SECONDS            Wallclock max (défaut: 900)
  --max-writes N               Max écritures fichier (défaut: 50)
  --read-outside PATH          Autorise read_file hors cwd, dans PATH

Reprise:
  --resume RUN_ID              Reprend depuis le contexte d'un run précédent
  --replay RUN_ID              Re-exécute la trace, en mode pas-à-pas (debug)

Sortie & logging:
  --trace-dir PATH             Où écrire les traces (défaut: .agent-kiki/runs/)
  --quiet                      Pas d'output sauf erreurs
  -v, --verbose                Affiche thoughts complets
  --json                       Output NDJSON machine-readable

Modes recherche [research]:
  --orchestrator MODEL         Pilote alternatif: claude|gpt-4|local (défaut: local)
                               Documente l'écart sovereign vs frontier
  --no-codegen-delegate        Le planner code lui-même (Apertus monolithe)
                               Mode debug pour mesurer l'apport de Devstral

Aide:
  --help                       Cette aide
  --version                    Versions CLI + workers eu-kiki
```

**Sortie terminale (mode défaut) :**

```
🪁 agent-kiki — run a3f2bc91-2026-05-04-185402
📂 mode: SCRATCH (cwd vide)
🎯 hint domain: rust (inféré: "Rust" dans la tâche)

## Plan
1. Cargo.toml — manifest avec serde + nom + version
2. src/lib.rs — exports publics
3. src/parser.rs — logique de parsing
4. tests/integration.rs — cas de test

[1/4] write_file Cargo.toml ........................... ✓ (Devstral, 1.2s)
[2/4] write_file src/lib.rs ........................... ✓ (Devstral, 0.9s)
[3/4] write_file src/parser.rs ........................ ✓ (Devstral, 3.4s)
[4/4] write_file tests/integration.rs ................. ✓ (Devstral, 1.6s)
[5/?] run_cmd "cargo build"  → confirm? [Y/n] y
       ↳ ✓ exit 0 (4.2s)
[6/?] run_cmd "cargo test"   → confirm? [Y/n] y
       ↳ ✗ exit 101 (test_parse_array failed at line 47)
[7/?] read_file src/parser.rs ......................... ✓
[8/?] edit_file src/parser.rs ......................... ✓ (Devstral, 1.1s)
[9/?] run_cmd "cargo test"   → confirm? [Y/n] y
       ↳ ✓ exit 0 (3.8s)
[10/?] finish

## Run Report
✓ Run terminé en 9 tours (28s wallclock, 14k tokens)
4 fichiers créés, 1 fichier édité, 0 erreur résiduelle
Trace: .agent-kiki/runs/a3f2bc91-2026-05-04-185402/
```

`--verbose` ajoute le `<thought>` complet d'Apertus avant chaque action (encadré gris).
`--json` produit sur stdout un stream NDJSON, 1 objet par tour + 1 objet stats final.

**Configuration persistante :** `~/.config/agent-kiki/config.toml` pour les défauts perso.
Précédence : CLI flags > env vars (`AGENT_KIKI_GATEWAY`, etc.) > config file > defaults.

**Pourquoi pas de REPL au MVP :** full-autonomous + collecte de traces V1 = chaque run doit être atomique et reproductible. Multi-tour passe par `aki --resume <run-id> "ajoute un test"` qui démarre un nouveau run avec contexte du précédent (plus structuré, plus loggable). REPL en V1+ si le besoin se confirme.

---

## 8. Logging & schéma JSONL

**Layout par run :**

```
.agent-kiki/runs/<run-id>/
├── meta.json              # Métadonnées run
├── plan.md                # Plan initial Markdown (rendu humain)
├── trace.jsonl            # 1 ligne par tour ReAct, append-only
├── codegen.jsonl          # 1 ligne par appel role=code (Devstral)
├── summaries.jsonl        # 1 ligne par appel role=lang (EuroLLM résumés)
├── approvals.jsonl        # 1 ligne par décision user (Y/n/abort)
├── report.md              # Run report final Markdown
└── files/
    ├── pre/               # Snapshots fichiers AVANT édition
    └── post/              # Snapshots APRÈS — diffs reconstructibles offline
```

`run-id` format : `<8-hex-random>-<YYYY-MM-DD>-<HHMMSS>` (court, daté, unique).

**Schéma `meta.json` :**

```json
{
  "run_id": "a3f2bc91-2026-05-04-185402",
  "schema_version": "1.0.0",
  "started_at": "2026-05-04T18:54:02+02:00",
  "ended_at": "2026-05-04T18:54:30+02:00",
  "exit_code": 0,
  "exit_reason": "finish",
  "task": "Crée un parser TOML en Rust avec tests unitaires",
  "cwd": "/Users/electron/Documents/Projets/euk-test/toml-parser",
  "mode": "SCRATCH",
  "hint_domain": "rust",
  "approval_mode": "safe",
  "agent_kiki_version": "0.1.0",
  "gateway_url": "http://studio:9200",
  "workers": {
    "planner": {"model": "apertus-70b-base", "adapter": null, "endpoint": ":9301"},
    "coder":   {"model": "devstral-24b-mlx-4bit", "adapter": "rust", "endpoint": ":9302"},
    "lang":    {"model": "eurollm-22b", "adapter": null, "endpoint": ":9303"}
  },
  "stats": {
    "turns": 9,
    "files_created": 4, "files_edited": 1,
    "tokens_total": 14210,
    "tokens_planner": 8420, "tokens_coder": 5510, "tokens_lang": 280,
    "wallclock_seconds": 28.4
  },
  "limits_hit": []
}
```

**Schéma `trace.jsonl` (1 ligne / tour) :**

```json
{
  "schema_version": "1.0.0",
  "run_id": "a3f2bc91-...",
  "turn": 3,
  "timestamp": "2026-05-04T18:54:09.123+02:00",
  "phase": "plan",
  "context_window": {"total_tokens": 2840, "messages_count": 7, "summarized_turns": 0},
  "planner_request": {
    "model": "apertus-70b-base",
    "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
    "headers": {"X-Eu-Kiki-Hint": "domain=rust", "X-Eu-Kiki-Role": "plan"}
  },
  "planner_response": {
    "raw": "<thought>...</thought>\n<tool_call>{\"name\": \"write_file\", ...}</tool_call>",
    "parsed_thought": "Je dois créer le manifest Cargo.",
    "parsed_tool_call": {"name": "write_file", "arguments": {"path": "Cargo.toml", "spec": "..."}},
    "parse_status": "ok",
    "parse_retries": 0,
    "tokens_in": 320, "tokens_out": 95, "latency_ms": 1240
  },
  "tool_execution": {
    "name": "write_file",
    "arguments": {"path": "Cargo.toml", "spec": "..."},
    "approved_by_user": true,
    "approval_latency_ms": 0,
    "result_status": "ok",
    "result_summary": "wrote 18 lines to Cargo.toml",
    "result_truncated": false,
    "delegated_codegen": "codegen.jsonl#3",
    "snapshot_pre": null,
    "snapshot_post": "files/post/Cargo.toml"
  },
  "errors": []
}
```

**Schéma `codegen.jsonl` :**

```json
{
  "run_id": "a3f2bc91-...",
  "turn_ref": 3,
  "model": "devstral-24b-mlx-4bit",
  "adapter": "rust",
  "prompt": "Génère le contenu pour Cargo.toml selon spec: ...",
  "response_raw": "[package]\nname = \"toml-parser\"\n...",
  "tokens_in": 410, "tokens_out": 132, "latency_ms": 1180,
  "validation": {"syntax_check": "skipped", "linter": "skipped"}
}
```

**Schéma `approvals.jsonl` :**

```json
{
  "run_id": "a3f2bc91-...",
  "turn_ref": 5,
  "tool": "run_cmd",
  "preview": "cargo build",
  "decision": "yes",
  "decision_latency_ms": 1240,
  "auto_approved": false
}
```

**Versionning :**
- `schema_version` SemVer dans **chaque** ligne (résiste aux concaténations futures de runs hétérogènes).
- Bump majeur si format incompatible. MVP = `1.0.0`.
- Outils de lecture vérifient le major et plantent explicitement sur mismatch.
- Migration script `scripts/migrate_traces.py` livré dès V0 (vide initialement, cadre la pratique).

**Outillage offline (livré V0) :**
- `aki-stats <run-id>` → résumé human-readable (turns, tokens, errors, files touched).
- `aki-replay <run-id>` → re-joue la trace tour par tour, output coloré, debug pur (n'appelle pas le LLM).
- `aki-export-dataset --since DATE --to OUT.jsonl` → exporte les **runs réussis non-interrompus** au format ShareGPT/messages, prêt à passer dans un trainer LoRA. C'est le pipeline V1.

**Privacy :**
- Pas de prompts envoyés dehors par défaut (tout reste sur disque local).
- `meta.json` n'inclut pas le contenu absolu des fichiers — c'est dans `files/post/`. Export dataset = opt-in explicite par run pour inclure ces snapshots.
- Variables d'environnement, `.env` du cwd → filtrés du contexte avant logging (regex `.*_TOKEN`, `.*_KEY`, `.*_SECRET`, `PASSWORD.*`, `LITELLM_MASTER_KEY`, plus patterns workspace `SAILLANT_.*`, `KXKM_.*` à étendre selon découvertes).

---

## 9. Stack technique

| Brique | Choix | Raison |
|---|---|---|
| Runtime | Python 3.13+ | Cohérence pyproject.toml partagé eu-kiki |
| Package mgr | `uv` | Standard maison |
| HTTP client | `httpx` | Robuste, déjà dans eu-kiki |
| Validation | `pydantic v2` + `jsonschema` | Schémas tools / traces |
| CLI parser | `typer` | Commandes propres, auto-help, type hints |
| Output coloré | `rich` | Tables, spinners, panels |
| Parsing ReAct | regex + `json.loads` | XML enveloppe trivialement parseable |
| Logging | `structlog` → JSONL | Performant, format-stable |
| Tests | `pytest` + `pytest-asyncio` + `respx` | Mock httpx sans toucher gateway |
| Lint/format | `ruff` | Cohérence eu-kiki |

**Pas de dépendances lourdes :** pas de LangChain/LangGraph/DSPy/smolagents.
- Boucle ~300 lignes Python, écrite à la main.
- Format XML+JSON exact contrôlé par nous.
- Routing par header HTTP (frameworks supposent un seul LLM client).
- Évite ~50+ packages transitifs pour 0 gain net.

---

## 10. Layout fichiers

**Repo autonome** `L-electron-Rare/agent-kiki` cloné en `/Users/electron/Documents/Projets/agent-kiki/` :

```
agent-kiki/                            # repo autonome, GitHub privé
├── README.md
├── CLAUDE.md                          # guide local Claude Code
├── LICENSE                            # Apache-2.0 par défaut (à ajuster pour interne)
├── .gitignore
├── pyproject.toml                     # autonome, hatchling, scripts agent-kiki+aki
├── uv.lock
├── src/
│   └── agent_kiki/
│       ├── __init__.py
│       ├── __main__.py                # python -m agent_kiki
│       ├── cli.py                     # typer commands
│       ├── config.py                  # pydantic settings
│       ├── modes.py                   # detect_mode SCRATCH/EDIT/MIXED
│       ├── orchestrator/
│       │   ├── loop.py                # boucle ReAct principale
│       │   ├── planner.py             # client gateway role=plan
│       │   ├── coder.py               # client gateway role=code
│       │   ├── lang.py                # client gateway role=lang (résumés)
│       │   ├── parser.py              # XML+JSON parser, retry
│       │   ├── prompts.py             # system prompts par mode
│       │   └── budget.py              # max_steps, tokens, time, writes
│       ├── tools/
│       │   ├── base.py                # protocol Tool, schémas pydantic
│       │   ├── filesystem.py          # read/write/edit/list/search
│       │   ├── shell.py               # run_cmd + 3 zones
│       │   ├── finish.py
│       │   └── jail.py                # cwd jail, path validation
│       ├── approvals.py               # interactive y/n, modes safe/auto/yolo
│       ├── tracing/
│       │   ├── logger.py              # JSONL writer
│       │   ├── schemas.py             # pydantic models
│       │   ├── snapshots.py           # files/pre, files/post
│       │   └── secrets_filter.py      # regex secrets
│       ├── ui/
│       │   ├── terminal.py            # rich layout, spinners
│       │   └── json_stream.py         # mode --json
│       └── orchestrators/
│           ├── local_eu_kiki.py       # défaut: Apertus + Devstral + EuroLLM
│           └── claude.py              # mode recherche --orchestrator claude
├── tests/
│   ├── unit/
│   ├── integration/                   # gateway mocké via respx
│   └── golden/                        # 5-10 traces référence
├── scripts/
│   ├── aki-stats
│   ├── aki-replay
│   └── aki-export-dataset
└── docs/
    ├── architecture.md
    ├── prompts.md                     # snapshots system prompts
    ├── trace-schema.md
    └── specs/
        ├── 2026-05-04-agent-kiki-design.md  # copie canonique côté agent
        └── 2026-05-04-agent-kiki-plan.md    # plan d'implémentation
```

**Note de coexistence avec eu-kiki :** ce design vit aussi dans `eu-kiki/docs/specs/` parce qu'il a été pensé là (cohérent avec la convention "le spec vit où la conversation a eu lieu"). Le repo agent-kiki en a une **copie canonique** dans son propre `docs/specs/` qui devient la source de vérité. Tout amendement futur se fait côté agent-kiki ; le miroir dans eu-kiki devient un archive du design initial.

---

## 11. Stratégie de tests V0

| Niveau | Quoi | Cible |
|---|---|---|
| Unit | parser XML+JSON (cas valides, invalides, edge cases unicode), jail filesystem, whitelist shell 3 zones, budget enforcement | ~60 tests |
| Integration | boucle ReAct contre gateway mocké (`respx`). 6 scénarios : SCRATCH-Rust, SCRATCH-Python, EDIT-add-feature, mode MIXED interactif, dépassement de budget, format invalide retry | ~10 tests |
| Golden | 5 runs de référence sur tâches fixées, trace attendue snapshot. Test de non-régression sur le format de sortie (pas le contenu LLM, qui varie). | 5 traces |
| Contract | Le gateway eu-kiki retourne bien des headers `X-Eu-Kiki-Role` et le routing correspond. | 3 tests |

**Pas de tests E2E contre vrai gateway au CI** (besoin du MacStudio). E2E manuel via `make e2e-studio` quand opportun.

---

## 12. Roadmap

| Version | Période visée | Contenu |
|---|---|---|
| **V0 — MVP** | 2-3 semaines à partir du plan d'impl | Tout ce que ce spec décrit : 7 tools, modes SCRATCH/EDIT/MIXED, 3 zones safety, traces JSONL complètes, single-shot CLI. |
| **V1 — Adapter `agent-react`** | Après ~200-500 traces collectées (~1 mois d'usage perso) | Export dataset via `aki-export-dataset`, fine-tune LoRA Apertus 70B sur tes propres traces ReAct, déploiement worker `:9304`, basculement par défaut. |
| **V2 — Édition codebase RAG** | +1-2 mois après V1 | Indexation `tree-sitter` + embeddings (sentence-transformers déjà dans eu-kiki), tool `semantic_search`, mode EDIT dopé sur gros repos. **Variant possible** : s'en tenir à `search` ripgrep si V0+V1 marchent assez bien — décision data-driven. |
| **V3 — AST edits** | Si V2 prouve qu'on touche le plafond textuel | Tools `replace_function`, `add_import`, `rename_symbol` via tree-sitter (Python, Rust, TS d'abord). Alternative : abandon si Devstral fait suffisamment bien des edits textuels chirurgicaux. |
| **V4 — REPL & multi-agent collab** | Si V0-V3 sont stables | Mode interactif, agent qui peut spawner des sous-agents (Apertus principal délègue à un Apertus spécialisé "review"). Risqué, à n'envisager qu'avec preuves de stabilité. |

---

## 13. Critères "MVP fini" (V0 mergeable)

1. ✅ 5/5 golden tests passent (boucle ReAct, parsing, jail, whitelist, budget).
2. ✅ 3 tâches manuelles réussissent end-to-end sur MacStudio :
   - `aki "Crée un parser TOML en Rust avec tests"` (mode SCRATCH).
   - `aki "Ajoute un endpoint /healthz à life-core"` (mode EDIT, repo réel).
   - `aki "Génère un schéma KiCad pour LM358 + 4 résistances"` (mode SCRATCH, domaine `kicad`).
3. ✅ Trace JSONL d'un run validée par `pydantic` parsing strict (toutes les `schema_version` présentes, tous les schémas conformes).
4. ✅ `aki-export-dataset` génère un fichier non-vide depuis ≥10 runs réussis.
5. ✅ Hard-deny vérifié : `aki "supprime tout"` → refus à `rm -rf` même en `--yolo`.

---

## 14. Risques identifiés

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| Apertus 70B base diverge en tool-use sur runs longs (>10 tours) | Élevée | Boucle infinie / format invalide | `max_steps` strict + retry parsing limité à 1 + collecte traces pour V1 |
| Headers `X-Eu-Kiki-Role` non-forwardés au routeur eu-kiki | Moyenne | Routing ne marche pas → tout passe par le routeur Jina auto | Patch trivial dans gateway (tâche dépendante du plan d'impl) |
| Devstral 4-bit produit du code syntaxiquement invalide | Moyenne | Tests échouent, agent re-itère | Boucle agentique le détecte via `run_cmd cargo build`, repair tour suivant |
| Contexte Apertus sature avant fenêtre glissante (mode EDIT large repo) | Moyenne | Erreur HTTP du worker | Fenêtre glissante + résumé EuroLLM + fallback abort propre |
| User clique Y par fatigue sur un `run_cmd` destructeur | Moyenne | Perte de fichiers locaux | Zone HARD-DENY étroite (jamais bypassable sans `--allow-destructive`) |
| Adapter `agent-react` V1 pas assez bon avec 200-500 traces | Moyenne | V1 ne dépasse pas V0 | Continuer collecte jusqu'à 1000+, ajouter curation manuelle des bonnes traces |
| Gateway eu-kiki indispo (MacStudio offline) | Faible | Agent inutilisable | Affichage erreur claire au démarrage + suggestion `--orchestrator claude` |

---

## 15. Hors-scope explicites pour mémoire

- Multi-utilisateur (auth, quotas par user) — pas pertinent en outil interne.
- Chiffrement des traces — local-first, pas de besoin avant V1+ si publication dataset.
- Mode "GUI desktop" — pas envisagé.
- Intégration IDE (VS Code, Zed) — V5+ éventuellement.
- Génération de tests AVANT le code (TDD strict imposé) — l'agent peut le faire si demandé, mais pas un mode forcé.
- Auto-commit / auto-PR git — V2+ (l'agent peut faire `git add` mais pas commit/push au MVP).

---

## 16. Références

- Spec parent eu-kiki : `docs/specs/2026-04-26-eu-kiki-design.md`
- Plan eu-kiki : `docs/specs/2026-04-26-eu-kiki-plan.md`
- Précédents pertinents : Aider (search/replace blocks), Claude Code (edit_file pattern), ReAct paper (Yao et al. 2022), ToolBench / BFCL (benchmarks tool-use OSS).
- HumanEval+ baseline Devstral 24B 4-bit : 97.1 % pass@1 (validation intrinsèque de la qualité codegen, terrain où le design capitalise).
