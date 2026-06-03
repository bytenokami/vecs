Authored by Claude

# Workflow Framework v0.1

A thin template for AI-driven feature development. The base declares phases and slots; project profiles fill the slots. Composition, not inheritance.

## Conventions

- **Phase** — numbered stage with one required output and a slot list.
- **Slot** — named extension point a profile must (or may) fill.
- **Profile** — per-project markdown document filling every required slot of this base. Slot keys match base names verbatim. Recommended location: `docs/workflow-<project>-profile-v<n>.md`.
- **Open enum** — value list includes `"custom:<id>"`; profile may extend with a paired prose definition named `*_custom_definition: prose`.
- **Required** without a default forces the profile to declare a value.
- **Disabled phase** — when `enabled: false`, the phase is skipped, and cross-phase rules referencing its slots are skipped too. Profile must declare any stand-in.
- All paths and IDs use kebab-case.

Primitive types used in slot definitions:

| Type | Meaning |
|---|---|
| `command` | shell-invokable string run from repo root; exit 0 = success |
| `file-path`, `dir-path`, `filename` | POSIX, repo-relative |
| `glob` | gitignore-style syntax |
| `token-count` | non-negative integer; tokenizer matches host agent (cl100k by default) |
| `percentage` | integer 0–100 |
| `role-name` | kebab-case identifier; uniqueness scoped to the profile's `roster` |
| `phase-id` | integer 0–8 referencing a phase below |
| `semver` | semantic-version string |

**Subagent type registry.** `subagent-type` values, their `invocation_args` schemas, and default derivations for `max_file_scope` and `mutates` come from the host agent registry of the system running the framework (e.g., Claude Code's subagent list). The profile cites which registry and version it relies on.

Every phase declares a header:

| Header slot | Type | Default |
|---|---|---|
| `enabled` | `bool` | `true` |
| `cadence` | `enum[one-time, per-feature, per-loop, scaffold+per-feature] \| "custom:<id>"` | per phase below |
| `cadence_custom_definition` | prose | required-if-custom |

Base-level slots (apply framework-wide):

| Slot | Type | Req/Default |
|---|---|---|
| `framework_version_pin.mode` | `enum[track-latest, pin] \| "custom:<id>"` | `track-latest` |
| `framework_version_pin.version` | `semver \| "n/a"` | required-if `framework_version_pin.mode = pin` |
| `framework_version_pin_custom_definition` | prose | required-if-custom |

---

## Phase 0 — Bootstrap

**Output:** working agent infrastructure exists and is verified.
**Cadence default:** `one-time` (re-run on major restructure).

| Slot | Type | Req/Default |
|---|---|---|
| `bootstrap_done_check` | `command \| file-path` | required |
| `roster_definition_path` | `file-path \| "inline" \| "none"` | required |
| `pipeline_setup_runbook` | `file-path \| "inline" \| "none"` | required |
| `embedding_index_setup` | `file-path \| "none"` | optional |
| `rerun_trigger` | `enum[new-module, new-test-framework, manual] \| "custom:<id>"` | required |
| `rerun_in_flight_policy` | `enum[block-new, drain, snapshot-fork, manual] \| "custom:<id>"` | `manual` |

---

## Phase 1 — Acceptance Definition

**Output:** per-feature acceptance criteria are written and locatable.
**Cadence default:** `per-feature`.

| Slot | Type | Req/Default |
|---|---|---|
| `acceptance_source` | `file-path \| "inline" \| "pr-description" \| "none"` | required |
| `acceptance_format` | `enum[checklist, gherkin, prose] \| "custom:<id>"` | required |
| `acceptance_format_custom_definition` | prose | required-if-custom |
| `pass_threshold` | `enum[all-pass, percentage, weighted] \| "custom:<id>"` | `all-pass` |
| `pass_threshold_custom_definition` | prose | required-if-custom |

---

## Phase 2 — Project Context

**Output:** per-module context docs exist, stay fresh, stay under bloat ceiling.
**Cadence default:** `scaffold+per-feature`.

| Slot | Type | Req/Default |
|---|---|---|
| `context_tree_root` | `dir-path` | required |
| `per_module_doc_name` | filename | optional |
| `context_coverage_rule` | `enum[every-module, touched-modules, none] \| "custom:<id>"` | required |
| `context_update_trigger` | `enum[per-feature-acceptance, periodic-audit, both] \| "custom:<id>"` | required |
| `staleness_check` | `enum[commit-sha-tag, date-tag, none] \| "custom:<id>"` | required |
| `staleness_repair_owner` | `role-name \| "self"` | required |
| `staleness_repair_trigger` | `enum[on-detect, batched, pre-feature] \| "custom:<id>"` | required |
| `bloat_ceiling` | `token-count \| "none"` | required |
| `project_bloat_ceiling` | `token-count \| "none"` | optional |
| `context_write_coordination` | `enum[none, last-write, lease, queue, section-scoped-merge] \| "custom:<id>"` | `none` |

---

## Phase 3 — Validation Pipeline

**Output:** runner consumes acceptance criteria and emits pass/fail.
**Cadence default:** `scaffold+per-feature`.

| Slot | Type | Req/Default |
|---|---|---|
| `pipeline_runner` | command | required |
| `ci_config_path` | `file-path \| "none"` | required |
| `pre_merge_gate` | `enum[required, advisory, none] \| "custom:<id>"` | required |
| `acceptance_input_adapter` | `command \| file-path` | required |

**Cross-phase rules:**
- `acceptance_input_adapter` must support the `acceptance_format` chosen in Phase 1.
- If `pre_merge_gate = required`, Phase 5 must be `enabled: true` (gate has nothing to gate on otherwise).
- Profile validates both at load.

---

## Phase 4 — Multi-Agent Review Loop

**Output:** architect output passes critical-sinker review without iteration cap.
**Cadence default:** `per-feature`.

| Slot | Type | Req/Default |
|---|---|---|
| `loop_participants` | `list[role-name]` (refs Phase 6 roster) | required |
| `loop_kill_criteria` | `enum[reviewer-satisfied, reviewer-or-no-progress] \| "custom:<id>"` | required |
| `progress_detector.strategy` | `enum[overlap-heuristic, manual] \| "custom:<id>"` | required |
| `progress_detector.signal_source` | `enum[diff-overlap, comment-overlap, score-delta] \| "none"` | required |
| `progress_detector.threshold` | `number` | required-if `progress_detector.strategy = overlap-heuristic` |
| `progress_detector.window` | `int` | required-if `progress_detector.strategy = overlap-heuristic` |
| `escalation_trigger` | `enum[reviewer-deadlock, oscillation-detected, no-progress-N-cycles, manual] \| "custom:<id>"` | required |
| `escalation_target` | `role-name \| "human"` | required |
| `escalation_human_route_evidence` | `file-path \| "inline"` | required-if `escalation_target != "human"` |

**Validation rule:** `escalation_target` must be either `"human"` or a role (refs Phase 6 roster entry) whose declared behavior surfaces to a human (e.g., via `prompt-path` or a `tool_policy` requiring human approval). The profile records the proof in `escalation_human_route_evidence`.

---

## Phase 5 — Test Substrate

**Output:** concrete test artifacts exist and run under the validation pipeline.
**Cadence default:** `scaffold+per-feature`.

| Slot | Type | Req/Default |
|---|---|---|
| `test_runner` | command | required |
| `test_locations` | `list[dir-path]` | required |
| `new_test_required_per_feature` | `bool` | required |
| `test_required_exemptions` | `enum[none, refactor-only, docs-only, generated] \| "custom:<id>"` | `none` |
| `coverage_target` | `percentage \| "none"` | optional |

---

## Phase 6 — Specialist Roster

**Output:** named roles mapped to subagent types with prompts, contracts, and routing rule.
**Cadence default:** `scaffold+per-feature`.

### Top-level slots

| Slot | Type | Req/Default |
|---|---|---|
| `roster` | `map[role-name -> entry]` (entry shape below) | required |
| `required_roles` | `list[role-name]` (may be empty) | required |
| `role_definitions` | `map[role-name -> prose]` | required |
| `routing_authority` | `enum[static-map, lead-agent, manager-dispatch] \| "custom:<id>"` | `manager-dispatch` |
| `specialist_scope_rule` | prose | required |

### Roster entry shape (per role-name)

| Field | Type | Req/Default |
|---|---|---|
| `subagent-type` | string (name of host agent type; see Subagent type registry) | required |
| `prompt-path` | `file-path \| "inline"` | required |
| `role_shim` | prose (how to make the host agent behave as this role) | required |
| `invocation_args` | map (discriminated by `subagent-type`; profile validates) | required |
| `backing_fallback` | `enum[native, prompt-on-generic, none]` | required |
| `output_shape` | `enum[one-line-findings, prose, structured-json] \| "custom:<id>"` | required |
| `tool_policy` | `{allow: list, deny: list} \| "inherit-from-type"` | `inherit-from-type` |
| `max_file_scope` | `int \| "unbounded"` | derived from `subagent-type` (see registry) |
| `mutates` | `bool` | derived from `subagent-type` (see registry) |
| `isolation` | `enum[inherit, worktree, inline]` | `inherit` |
| `instance_policy` | `enum[singleton, per-feature, pool] \| "custom:<id>"` | `per-feature` |

**Validation rules:** every `required_roles` entry must appear in `roster`; `role_definitions` is keyed exactly by `roster.keys()`; both may be empty only if `required_roles` is empty.

---

## Phase 7 — Dry-Run

**Output:** the smallest real subtask runs end-to-end through all enabled phases once before real work begins.
**Cadence default:** `per-feature`.

| Slot | Type | Req/Default |
|---|---|---|
| `dryrun_selection` | `enum[smallest-real-subtask, adjacent-feature, synthetic-toy] \| "custom:<id>"` | required |
| `dryrun_acceptance` | `file-path \| "inline"` | required |
| `dryrun_pass_criteria` | `list[enum[pipeline-pass, review-loop-satisfied, test-substrate-green, acceptance-met]]` | required |
| `iterate_before_real` | `bool` | `true` |
| `abort_policy` | `enum[branch-drop, branch-drop-with-context-revert, runbook, manager-call] \| "custom:<id>"` | `branch-drop` |

**Cross-phase rule:** every entry in `dryrun_pass_criteria` must reference a phase with `enabled: true` (`pipeline-pass` → Phase 3, `review-loop-satisfied` → Phase 4, `test-substrate-green` → Phase 5, `acceptance-met` → Phase 1).

---

## Phase 8 — Retrospective

**Output:** gap log produced and routed to feedback targets.
**Cadence default:** `per-feature`.

| Slot | Type | Req/Default |
|---|---|---|
| `retro_template` | `file-path` | required |
| `gap_log_destination` | `file-path` | required |
| `feedback_targets` | `list[phase-id]` | `[0, 2]` |
| `feedback_artifact` | `enum[profile-diff, runbook-patch, schema-proposal, context-doc-edit] \| "custom:<id>"` | required |
| `feedback_apply_owner` | `role-name \| "self"` | required |
| `rollup.cadence` | `enum[weekly, per-release, manual] \| "custom:<id>" \| "none"` | `none` |
| `rollup.aggregator_role` | `role-name` | required-if `rollup.cadence != "none"` |
| `rollup.destination` | `file-path` | required-if `rollup.cadence != "none"` |
| `rollup.rollup_inputs` | `glob` | required-if `rollup.cadence != "none"` |

**Validation rule:** every entry in `feedback_targets` must reference a phase with `enabled: true`.

---

## Glossary

- **Phase** — Numbered stage with one required output and a slot list.
- **Slot / Extension point** — Named injection site in a phase; required or optional.
- **Cadence** — How often a phase runs (one-time, per-feature, per-loop, scaffold+per-feature).
- **Profile** — Per-project document filling every required base slot.
- **Base** — This document; project-agnostic shape.
- **Critical Sinker** — Standing skeptic role; challenges other roles' output per operation. Implementation lives in profile, not base.
- **Roster** — Map of role-name to backing subagent + contract.
- **Loop Kill Criteria** — Condition that terminates a multi-agent loop without iteration cap.
- **Dry-Run** — Single end-to-end pass on a small real subtask before committing to the full feature.
- **Staleness** — Drift between a context doc and the code it describes; detected by `staleness_check`.
- **Module** — Unit of decomposition chosen by the profile (e.g., top-level package, top-level source directory). The profile states its module definition in Phase 2 prose.
- **Context tree** — Rooted directory of per-module context docs; root declared by `context_tree_root`.
- **Embedding index** — Optional vector store backing semantic search over the context tree; setup declared by `embedding_index_setup`.
- **Manual** — Across enums, `manual` always means "human-driven": a human triggers, decides, or executes the case; no automation runs it unattended.

Role *definitions* (Architect, Reviewer, Planner/Lead, Specialist, etc.) live in `Phase 6.role_definitions`. Role *names* may appear in prose anywhere in the doc.

---

## Why composition, not inheritance

1. Workflow phases vary too much project-to-project for substitution to hold across profiles.
2. Override-heavy subclasses turn the base into a liability — base changes risk breaking children silently.
3. Composition keeps the base thin: named slots, no behavior, no project-specific values.
4. Profile fills slots per project; phases stay swappable, disable-able (`enabled: bool`), and extensible (`"custom:<id>"`).
5. Tension: closed enums and required slots still encode some base-owned shape. Mitigation is the per-phase enable flag, the `"custom:<id>"` extension on enums likely to need it, and the rule that defaults exist only where every project is plausibly served by the same value.
