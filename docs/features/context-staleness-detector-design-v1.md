# Feature Design: context-staleness-detector v1

Targets the vecs workflow profile at `docs/workflow-vecs-profile-v0.1.md`. Walks Phases 1–8 in profile order. Each phase produced by the role the profile's `agent_roster` designates.

**Status.** Pre-implementation design. This document specifies what to build and how to validate it. The source files it references (`src/vecs/staleness.py`, `tests/test_staleness.py`) and the `ProjectConfig.context_tree_root` field DO NOT YET EXIST; their construction is the work this document plans. Phase 5 tests are specifications, not yet authored code. Phase 1 acceptance criteria are forward-looking contracts the implementation must satisfy.

**Feature statement.** Add a CLI subcommand `vecs staleness -p <project>` and an MCP tool `mcp__vecs__staleness(project)` that walks every per-module context doc (`CLAUDE.md` under each project's `context_tree_root`), reads each doc's baseline commit SHA (the commit that last modified the doc itself), and compares it against the last commit of each file the doc references. Emits a list of stale docs with the drifted files. Implements the profile's Phase 2 `staleness_check: commit-sha-tag` so the workflow can detect when context docs lie.

**Filesystem layout for this feature** (per profile + brief):
- Design doc (this file): `docs/features/context-staleness-detector-design-v1.md`
- Acceptance: `docs/features/context-staleness-detector/acceptance.md`
- Gap log: `docs/features/context-staleness-detector/gaps.md`

---

## Phase 1 — Acceptance criteria

Produced by `architect` (`Plan`). Source: profile slot `acceptance_source` + `acceptance_format: checklist`.

- [ ] `vecs staleness -p <project>` exits 0 and prints `no stale docs` when every `CLAUDE.md` under the project's `context_tree_root` has a baseline commit SHA that is equal to OR an ancestor of every referenced file's last-modifying commit. Ancestry is determined via `git merge-base --is-ancestor <baseline> <file-head>` (exit 0 = is-ancestor).
- [ ] `vecs staleness -p <project>` exits 1 and prints one line per stale doc in form `<doc-relpath>: <baseline-sha7> < <file-relpath>@<head-sha7>` for each drifted file.
- [ ] `mcp__vecs__staleness(project)` returns dict `{"stale": [{"doc": <relpath>, "baseline": <sha>, "drifted": [{"path": <relpath>, "head": <sha>}, ...]}, ...], "docs_scanned": <int>}`; empty `stale` list when fresh.
- [ ] Baseline SHA per doc resolved via `git log -1 --format=%H -- <doc-path>`.
- [ ] When `context_tree_root` contains zero `CLAUDE.md` files: CLI exits 0; MCP returns `{"stale": [], "docs_scanned": 0}`.
- [ ] Unparseable baseline (doc not yet committed) reported on stderr as `unparseable baseline: <doc-relpath>`, excluded from stale list. Exit precedence: drift wins. Exit 1 when drift exists (whether or not unparseable docs coexist); exit 2 only when unparseable docs exist AND no drift. Unparseable always logs to stderr regardless of exit code.
- [ ] A "reference" in `CLAUDE.md` is an inline-code span matching regex `` `[A-Za-z0-9_./-]+\.[A-Za-z0-9]+(?::\d+)?` `` AND existing as a file under HEAD; the optional `:NN` line suffix is stripped before lookup. Spans that do not satisfy both predicates are ignored.
- [ ] Referenced path missing from HEAD reported as `<doc>: <baseline-sha7> < <path>@MISSING`, counts as drift.
- [ ] Unknown project (`-p` not in `config.projects`) exits 2; stderr contains `unknown project: <name>`.
- [ ] Known project with `context_tree_root` unset (`None`) exits 2; stderr contains `context_tree_root not set for project <name>`. Distinguished from unknown-project case by the stderr message; both share exit 2.
- [ ] When `context_tree_root` is not inside a git working tree, exits 2 with stderr `not a git repository: <root>`. No git calls attempted.
- [ ] MCP signature matches existing tool style: `mcp__vecs__staleness(project: str | None = None)`. When `project=None`, scans every project that declares `context_tree_root`; result dict keys by project name.

## Phase 2 — Required context

Produced by `investigator` (`caveman:cavecrew-investigator`) + `explorer` (`Explore`).

**Exists in vecs:**
- One context doc: `src/vecs/CLAUDE.md` (only `CLAUDE.md` in repo).
- `Manifest` class at `src/vecs/indexer.py:109` — tracks SHA256 file-content hashes per project. Atomic temp+rename under fcntl lock (`indexer.py:229`).
- CLI subcommand pattern: `@main.command()` at `src/vecs/cli.py:12, 31, 63`.
- MCP tool pattern: `@mcp.tool()` at `src/vecs/mcp_server.py:72, 135, 153`.
- Path-walking via `rglob` at `src/vecs/indexer.py:692, 697`; exclude-dir filter at `:703`.
- Project config: `ProjectConfig` dataclass at `src/vecs/config.py:43` — fields `name`, `code_dirs`, `sessions_dirs`, `docs_dir`, `codex_cwds`. **No `context_tree_root` field** (verified).
- Codebase is git-agnostic: zero `subprocess`, `pygit2`, or `git` imports under `src/vecs/`. **This feature breaks that invariant** — the first git subprocess and the first `subprocess` import in `src/vecs/` both land in `staleness.py`. Treat as an explicit invariant change, not an incidental detail.

**Must add:**
- `context_tree_root: Path | None = None` field on `ProjectConfig`. Type is `Path | None` (with default `None`) rather than the profile's `dir-path, required` to keep existing configs loading (they predate this field). Loader contract: `load_config()` treats a missing YAML key as `None`. Feature-runtime contract: `vecs staleness -p <project>` exits 2 with `context_tree_root not set for project <project>` when the field is `None`; `mcp__vecs__staleness(project=None)` silently skips projects where the field is `None` (does not raise). See Phase 8 gap #1 for the profile-schema deviation.
- New module `src/vecs/staleness.py` housing: `resolve_baseline_sha(doc_path: Path, repo_root: Path) -> str | None`, `last_commit(file_path: Path, repo_root: Path) -> str | None`, `is_ancestor(ancestor_sha: str, descendant_sha: str, repo_root: Path) -> bool` (wraps `git merge-base --is-ancestor`), `extract_references(text: str, repo_root: Path) -> list[Path]` (regex `` r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]+(?::\d+)?)`" `` then strips `:NN` suffix then filters to paths existing in HEAD), `walk_context_tree(root: Path) -> Iterator[Path]`, `is_git_repo(root: Path) -> bool`, top-level `find_stale(project_config) -> StalenessReport`. `repo_root` for all helpers is `project.context_tree_root`'s nearest enclosing git working tree (resolved once per project).
- New CLI subcommand registered in `cli.py` mirroring `index` decorator style.
- New MCP tool registered in `mcp_server.py` mirroring `index_status` decorator style.
- New test module `tests/test_staleness.py`.
- Per-module CLAUDE.md updates as feature lands (Phase 2 of profile: `context_coverage_rule: touched-modules`).

**Implementation order:** add `context_tree_root` field to `ProjectConfig` first; CLI command second; reference extractor + git helpers third; MCP tool last. The MCP tool's `project=None` mode enumerates projects via the new field, so the field must exist before the tool can be smoke-tested.

**Git subprocess pattern** (first in codebase): `subprocess.run(["git", "log", "-1", "--format=%H", "--", path], capture_output=True, text=True, cwd=repo_root)`; non-zero rc or empty stdout returns `None`. Matches the atomic-write discipline already established in `codex_routing.py:137-182` and `indexer.py:229-248`.

## Phase 3 — Validation pipeline

Inherited from profile:
- `pipeline_runner: uv run pytest -q`
- `acceptance_input_adapter: scripts/check_acceptance.py context-staleness-detector`
- `pre_merge_gate: advisory`
- `ci_config_path: "none"`

Profile's Phase 3 cross-phase rule does not bind (gate is advisory; Phase 5 is enabled anyway).

## Phase 4 — Review loop

Inherited from profile:
- `loop_participants: [architect, critical-sinker, reviewer]`
- `loop_kill_criteria: reviewer-or-no-progress`
- `progress_detector.strategy: manual`
- `escalation_target: "human"`

Reviewer must emit one verdict line per iteration (profile's `specialist_scope_rule`).

## Phase 5 — Test substrate

Produced by `architect`. Lives at `tests/test_staleness.py` (matches per-module test convention in `src/vecs/CLAUDE.md`). Uses `pytest tmp_path` + `git init` fixture for repo state.

| Test | Assertion |
|---|---|
| `test_staleness_fresh_doc_no_drift` | baseline equals HEAD of referenced file → empty stale list |
| `test_staleness_one_stale_one_fresh` | two docs, only one reported stale |
| `test_staleness_doc_with_no_references` | zero inline-code path tokens → no contribution |
| `test_staleness_baseline_older_no_referenced_drift` | baseline N commits behind, referenced files unchanged → not stale |
| `test_staleness_baseline_ancestor_of_head_with_drift` | `git merge-base --is-ancestor baseline HEAD` true AND referenced file commit ≠ baseline → stale |
| `test_staleness_head_ancestor_of_baseline_no_drift` | doc edited after referenced file's last commit → not stale |
| `test_staleness_deleted_referenced_file` | referenced path missing from HEAD → `@MISSING`, exit 1 |
| `test_staleness_unparseable_baseline` | uncommitted doc + no drift → stderr + exit 2 |
| `test_staleness_unparseable_plus_drift` | uncommitted doc AND drift coexist → exit 1 (drift wins), unparseable on stderr |
| `test_staleness_unknown_project` | bad `-p` → exit 2 |
| `test_staleness_non_git_root` | `context_tree_root` outside git tree → exit 2, stderr `not a git repository` |
| `test_staleness_line_suffix_stripped` | `` `path:42` `` resolves like `` `path` `` |
| `test_staleness_reference_regex_filters_nonpaths` | inline-code spans like `` `True` `` or `` `dict` `` ignored (no extension OR not in HEAD) |
| `test_mcp_staleness_returns_dict` | MCP variant returns dict matching CLI semantics |
| `test_mcp_staleness_docs_scanned_count` | MCP return includes `docs_scanned` count |
| `test_mcp_staleness_no_project_scans_all` | `mcp__vecs__staleness()` with no args scans every project declaring `context_tree_root` |
| `test_mcp_staleness_skips_projects_without_context_tree_root` | mixed config (some projects with `context_tree_root`, some without) + `project=None` → result dict keys include only the projects with the field set; no error raised for the others |
| `test_cli_staleness_project_missing_context_tree_root` | known project, `context_tree_root` is `None` → CLI exits 2 with `context_tree_root not set for project <name>` on stderr |

## Phase 6 — Specialist roster picks

From profile's roster:

| Feature-phase activity | Profile role | Subagent-type |
|---|---|---|
| Acceptance design (Phase 1) | `architect` | `Plan` |
| Context survey (Phase 2) | `investigator` | `caveman:cavecrew-investigator` |
| Broader survey (Phase 2) | `explorer` | `Explore` |
| Test design (Phase 5) | `architect` | `Plan` |
| Dry-run plan (Phase 7) | `architect` | `Plan` |
| Implementation, small (≤2 files) | `builder-small` | `caveman:cavecrew-builder` |
| Implementation, multi-file | `builder-large` | `general-purpose` (worktree) |
| Gap-finding | `critical-sinker` | `general-purpose` + `.claude/prompts/critical-sinker.md` |
| Diff review | `reviewer` | `caveman:cavecrew-reviewer` |

**Roles engaged during this design pass:**
- `architect` (`Plan`) — produced Phase 1 acceptance, Phase 5 test plan, Phase 7 dry-run plan (this document).
- `investigator` (`caveman:cavecrew-investigator`) — produced Phase 2 context survey (cited file:line table).
- `explorer` (`Explore`) — produced Phase 2 git-pattern survey.
- `critical-sinker` (`general-purpose` + `.claude/prompts/critical-sinker.md`) — produced 9 findings. Seven integrated as direct edits (Phase 1: exit precedence, reference-token regex, non-git-root case, MCP signature; Phase 2: regex spec + `is_git_repo` helper; Phase 5: 5 new test cases; Phase 7: dry-run note correction). Two retained as Phase 8 entries (profile-schema deviation; reviewer-absence trace).
- `reviewer` (`caveman:cavecrew-reviewer`) — ran six passes per profile `loop_kill_criteria: reviewer-or-no-progress`. Pass 1 `revise` (8 findings; 5 integrated). Pass 2 `revise` (5 findings, all NEW; 4 integrated). Pass 3 `revise` (2 MINOR, both NEW; integrated). Pass 4 `revise` w/ scope drift; resolved by pre-implementation status preamble. Pass 5 `revise` (1 finding: signature mismatch); integrated by harmonizing `resolve_baseline_sha(doc_path: Path, repo_root: Path) -> str | None`. **Pass 6 `verdict: ship`** — zero BLOCKER, zero MAJOR, no overlap with prior passes; loop terminated.

## Phase 7 — Dry-run

`dryrun_selection: smallest-real-subtask` (per profile).

**Subtask.** Implement only `resolve_baseline_sha(doc_path: Path, repo_root: Path) -> str | None` (signature matches Phase 2 spec). New module `src/vecs/staleness.py` with one function plus one test file. ~30 lines total. No CLI, no MCP, no reference extraction, no drift comparison.

**Dry-run acceptance** (4 items, `dryrun_acceptance: "inline"`):
- [ ] Helper returns 40-char SHA for `src/vecs/CLAUDE.md` against live repo.
- [ ] Helper returns `None` for uncommitted file inside a `tmp_path` git fixture.
- [ ] `tests/test_staleness.py` exists and passes under `uv run pytest -q`.
- [ ] No writes outside `src/vecs/` and `tests/`.

**Roles.** `architect` (this plan), `builder-small` (1–2 files), `critical-sinker` (gap-find on helper signature + fixture approach), `reviewer` (verdict).

**Pass criteria** (per profile `dryrun_pass_criteria`): `[pipeline-pass, review-loop-satisfied]`.

**Expected learning.** Validates git-subprocess approach inside pytest fixtures; surfaces worktree-vs-main-checkout SHA quirks. (`scripts/check_acceptance.py` is not exercised during dry-run because `dryrun_acceptance: "inline"` — the inline checklist above is operator-verified. The script's first real run happens against Phase 1 acceptance at `docs/features/context-staleness-detector/acceptance.md` once the full feature lands.)

**Sequencing.** Phase 7 dry-run runs FIRST in implementation time: implements `resolve_baseline_sha` plus one pytest test using `tmp_path` + `subprocess.run(["git", "init", ...])` fixture. That fixture pattern is the calibration artifact. The full Phase 5 test matrix (15 tests) is authored against the same fixture pattern AFTER the dry-run validates it. Phase 3 pipeline (`uv run pytest -q`) runs the substrate only after dry-run and full-test-authoring both complete. This resolves the apparent circular dependency between Phase 3 (runs tests), Phase 5 (declares tests), and Phase 7 (validates the fixture pattern the tests depend on).

## Phase 8 — Gaps surfaced

1. **Profile-schema deviation on `context_tree_root` type.** Profile (`workflow-framework-v0.1.md`) declares `context_tree_root: dir-path, required`. This feature ships it as `Path | None` on `ProjectConfig` so existing configs keep loading. Schema-strictly this is a profile violation. → Resolve by either tightening to `dir-path, required` once every project sets it, or adding an `optional` annotation to the base slot in v0.2.

2. **Performance: N git calls per scan.** One `git log` per doc + one `git merge-base --is-ancestor` per referenced file. POC scale acceptable; not benchmarked at larger trees. → Flag for benchmark before applying to projects with >50 docs; explore `git log --name-only --since=<baseline-sha>` batching.

3. **Profile roster has no "infra-pattern setter" role.** This feature introduces the first git subprocess in vecs. `builder-large` (worktree-isolated `general-purpose`) handles it adequately; no roster change needed.

4. **Review-loop trace.** Reviewer ran six passes (see Phase 6 attribution). Pass 6 returned `verdict: ship`; loop terminated per `loop_kill_criteria: reviewer-or-no-progress`. → Not a profile gap; trace for kill-criteria audit.
