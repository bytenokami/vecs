# Workflow Profile: vecs v0.1

Fills the slots declared in `docs/workflow-framework-v0.1.md` for the vecs repository (POC scale, single contributor, no CI, no per-module context docs).

**Module definition:** a "module" in vecs is a top-level Python file under `src/vecs/` (e.g., `indexer.py`, `searcher.py`, `codex_routing.py`, `mcp_server.py`).

## Base-level

| Slot | Value |
|---|---|
| `framework_version_pin.mode` | `pin` |
| `framework_version_pin.version` | `0.1.0` |

The pin is documentation only; no automated profile-loader enforces it. Base bumps must be reviewed manually.

## Phase 0 — Bootstrap

`enabled: true`, `cadence: one-time`.

| Slot | Value |
|---|---|
| `bootstrap_done_check` | `mkdir -p docs/features docs/templates .claude/prompts && uv sync && uv run pytest -q` |
| `roster_definition_path` | `"inline"` (declared in Phase 6 below) |
| `pipeline_setup_runbook` | `README.md` (Install section, lines 7–32) |
| `embedding_index_setup` | `scripts/register_self.py` (idempotent; registers vecs as a vecs project, then run `vecs index -p vecs`) |
| `rerun_trigger` | `manual` |
| `rerun_in_flight_policy` | `manual` |

Cites: `pyproject.toml:19-30` (pytest, pytest-cov, ruff declared as dev deps; uv is the package manager that consumes the file), `README.md:7-32` (MCP install).

## Phase 1 — Acceptance Definition

`enabled: true`, `cadence: per-feature`.

| Slot | Value |
|---|---|
| `acceptance_source` | `docs/features/<feature-name>/acceptance.md` |
| `acceptance_format` | `checklist` |
| `pass_threshold` | `all-pass` |

Precedent: `docs/superpowers/specs/2026-05-05-codex-sessions-design.md` uses an explicit Q1–Q7 locked-decision table.

## Phase 2 — Project Context

`enabled: true`, `cadence: scaffold+per-feature`.

| Slot | Value |
|---|---|
| `context_tree_root` | `src/vecs/` |
| `per_module_doc_name` | `CLAUDE.md` (root context doc at `src/vecs/CLAUDE.md`; per-file docs added as features touch them) |
| `context_coverage_rule` | `touched-modules` |
| `context_update_trigger` | `per-feature-acceptance` |
| `staleness_check` | `commit-sha-tag` |
| `staleness_repair_owner` | `"self"` |
| `staleness_repair_trigger` | `on-detect` |
| `bloat_ceiling` | `"none"` (POC scale; revisit if any single context doc exceeds ~10k tokens) |
| `context_write_coordination` | `none` |

`src/vecs/CLAUDE.md` is the root context doc; `staleness_check: commit-sha-tag` baselines from the commit that introduces it. Per-file docs accrete only when a feature touches the file (matches `context_coverage_rule: touched-modules`).

## Phase 3 — Validation Pipeline

`enabled: true`, `cadence: scaffold+per-feature`.

| Slot | Value |
|---|---|
| `pipeline_runner` | `uv run pytest -q` |
| `ci_config_path` | `"none"` |
| `pre_merge_gate` | `advisory` |
| `acceptance_input_adapter` | `scripts/check_acceptance.py` |

`scripts/check_acceptance.py <feature-name>` parses `docs/features/<feature-name>/acceptance.md`, walks each `- [ ]` / `- [x]` item, asks the operator y/n per item, exits 0/1. `--non-interactive` flag accepts pre-checked `[x]` items as pass. Cross-phase rule check: `pre_merge_gate = advisory`, so Phase 5 enablement constraint does not bind.

## Phase 4 — Multi-Agent Review Loop

`enabled: true`, `cadence: per-feature`.

| Slot | Value |
|---|---|
| `loop_participants` | `[architect, critical-sinker, reviewer]` |
| `loop_kill_criteria` | `reviewer-or-no-progress` |
| `progress_detector.strategy` | `manual` |
| `progress_detector.signal_source` | `"none"` |
| `escalation_trigger` | `no-progress-N-cycles` |
| `escalation_target` | `"human"` |
| `escalation_human_route_evidence` | not required (target is `"human"` directly) |

## Phase 5 — Test Substrate

`enabled: true`, `cadence: scaffold+per-feature`.

| Slot | Value |
|---|---|
| `test_runner` | `uv run pytest` |
| `test_locations` | `[tests/]` |
| `new_test_required_per_feature` | `true` |
| `test_required_exemptions` | `none` |
| `coverage_target` | unset (`pytest-cov>=7.1.0` per `pyproject.toml:29` but no target enforced) |

Cite: `tests/` contains 13 modules including `test_searcher.py`, `test_indexer.py`, `test_codex_routing.py`, `test_bm25.py`, `test_ast_chunker.py`.

## Phase 6 — Specialist Roster

`enabled: true`, `cadence: scaffold+per-feature`.
`routing_authority: manager-dispatch`.
`specialist_scope_rule:` "Narrow each specialist to one concern. Read-only roles deny `Edit`/`Write`. Builder roles split by file count: 1–2 files → `builder-small`; 3+ files → `builder-large` (worktree-isolated). Reviewer must emit one verdict line per loop iteration; that line is the falsifiable signal consumed by Phase 7's `review-loop-satisfied` pass criterion. **Manager prompt-load rule:** when a roster entry's `prompt-path` is a file (not `"inline"`), the manager reads the file at dispatch time and uses its contents verbatim as the agent prompt prefix. Inline prompts are constructed per-call from the `role_shim` and per-task context; the manager must echo the shim verbatim to keep behavior consistent across invocations."

### Roster

| role-name | subagent-type | prompt-path | role_shim | backing_fallback | output_shape | tool_policy | max_file_scope | mutates | isolation | instance_policy |
|---|---|---|---|---|---|---|---|---|---|---|
| `architect` | `Plan` | `"inline"` | Design feature plan; consume Phase 1 acceptance | `native` | `prose` | `inherit-from-type` | 0 | false | inherit | per-feature |
| `critical-sinker` | `general-purpose` | `.claude/prompts/critical-sinker.md` | Standing skeptic; numbered gap list; no praise | `prompt-on-generic` | `one-line-findings` | `{deny: [Edit, Write]}` | 0 | false | inherit | per-feature |
| `reviewer` | `caveman:cavecrew-reviewer` | `"inline"` | Diff/branch review; severity-tagged one-line findings + per-iteration verdict line | `native` | `one-line-findings` | `inherit-from-type` | unbounded | false | inherit | per-feature |
| `investigator` | `caveman:cavecrew-investigator` | `"inline"` | Read-only code locator; file:line table | `native` | `one-line-findings` | `inherit-from-type` | unbounded | false | inherit | per-feature |
| `builder-small` | `caveman:cavecrew-builder` | `"inline"` | 1–2 file edits only; refuses 3+ | `native` | `one-line-findings` | `inherit-from-type` | 2 | true | inherit | per-feature |
| `builder-large` | `general-purpose` | `"inline"` | Multi-file implementation; cites tests written | `prompt-on-generic` | `prose` | `inherit-from-type` | unbounded | true | worktree | per-feature |
| `explorer` | `Explore` | `"inline"` | Broad read-only codebase exploration | `native` | `prose` | `inherit-from-type` | unbounded | false | inherit | per-feature |

`required_roles: [architect, critical-sinker, reviewer]`.

### role_definitions

| Role | Definition |
|---|---|
| `architect` | Produces architecture and stepwise plan for the feature. Folds the base glossary's Planner/Lead — no separate planner exists in vecs. |
| `critical-sinker` | Standing skeptic. Reviews every architect or coder output for gaps, contradictions, missing slots. Output is a numbered gap list with no praise or balance. |
| `reviewer` | Diff and branch reviewer. Emits severity-tagged one-line findings plus a per-iteration verdict line; the verdict feeds Phase 7's `review-loop-satisfied`. |
| `investigator` | Read-only locator. Returns file:line tables for "where is X" / "what calls Y". Refuses to suggest fixes. |
| `builder-small` | Surgical 1–2 file edit. Refuses 3+ scope. Caveman house style (compressed output). |
| `builder-large` | Multi-file implementation. Runs in worktree isolation. |
| `explorer` | Broad read-only exploration; wider breadth than `investigator`, less targeted. |

### Role aliases

The base glossary names Planner/Lead as a distinct role; vecs folds it into `architect`. No separate planner role exists at this scale.

## Phase 7 — Dry-Run

`enabled: true`, `cadence: per-feature`.

| Slot | Value |
|---|---|
| `dryrun_selection` | `smallest-real-subtask` |
| `dryrun_acceptance` | `"inline"` (declared in each feature design doc, Phase 7 section) |
| `dryrun_pass_criteria` | `[pipeline-pass, review-loop-satisfied]` |
| `iterate_before_real` | `true` |
| `abort_policy` | `branch-drop` |

Vecs uses git branches per feature (`.worktrees/` directory exists at repo root). `review-loop-satisfied` is falsifiable only if the reviewer emits a verdict line per iteration; see `specialist_scope_rule`.

## Phase 8 — Retrospective

`enabled: true`, `cadence: per-feature`.

| Slot | Value |
|---|---|
| `retro_template` | `docs/templates/retro.md` |
| `gap_log_destination` | `docs/features/<feature-name>/gaps.md` |
| `feedback_targets` | `[0, 2]` |
| `feedback_artifact` | `context-doc-edit` |
| `feedback_apply_owner` | `"self"` |
| `rollup.cadence` | `"none"` |

The `retro_template` file exists at `docs/templates/retro.md` (stub; phase-by-phase notes + gap log section + feedback-applied table).

---

## Gaps surfaced (0 open)

All gaps from prior revisions are now closed by either an artifact in the repo or a slot encoding the design choice.

| Prior gap | Resolution |
|---|---|
| No per-module context docs | `src/vecs/CLAUDE.md` written (root context doc; per-file docs accrete as features touch modules per `context_coverage_rule: touched-modules`) |
| `acceptance_input_adapter` human-only | `scripts/check_acceptance.py` (interactive + non-interactive modes) |
| vecs not in `~/.vecs/config.yaml` | `scripts/register_self.py` (idempotent registration); operator runs once |
| `general-purpose` prompt drift | `specialist_scope_rule` now declares the manager prompt-load rule; `.claude/prompts/critical-sinker.md` stubbed |
| `retro_template` file missing | `docs/templates/retro.md` stubbed |
| `critical-sinker` prompt missing | `.claude/prompts/critical-sinker.md` stubbed |
| Architect can't edit files | Encoded as `max_file_scope: 0` — correct by design |
| Planner/Lead fold | Encoded in Role aliases section |
| Reviewer verdict line | Encoded in `specialist_scope_rule` |
| `framework_version_pin` non-enforcement | Documented at base-level — POC reality |

Open work that is not a profile gap (operator follow-up, not schema violation):
- `scripts/check_acceptance.py` has not been exercised by a real feature acceptance run; first real use may surface bugs.
- `scripts/register_self.py` has not been run; semantic search over vecs itself is dormant until it is.
- `src/vecs/CLAUDE.md` will need updates as `searcher.py`, `indexer.py`, or `codex_routing.py` evolves; `staleness_check: commit-sha-tag` flags drift on next staleness scan.
