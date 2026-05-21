# Feature Design: prose-staleness-detector v1 (V+)

Targets the vecs workflow profile at `docs/workflow-vecs-profile-v0.1.md`. Walks Phases 1–8 in profile order. Each phase produced by the role the profile's `roster` designates. This is the **V+ rewrite**: the prior Mem0-based draft was reopened when six sub-agent passes surfaced (a) Mem0 v2.0.2/v3 is officially ADD-only per `docs.mem0.ai/migration/oss-v2-to-v3` (the UPDATE event the prior design depended on is deprecated), and (b) bi-temporal storage (Graphiti/Zep; XTDB/Datomic) is the mature industry pattern. V+ is a vecs-only, bi-temporal, INSERT/NOOP/SUPERSEDE state machine sitting directly on the existing Chroma + Voyage stack. No Mem0 dependency.

**Status.** Pre-implementation design. The source files this document references (`src/vecs/prose_drift.py`, `tests/test_prose_drift.py`), the `ProjectConfig.prose_drift_enabled` field, the Anthropic-client wiring, the new Chroma collection `<project>-prose-facts`, and the SQLite cache at `~/.vecs/prose_drift_cache/<project>.db` DO NOT YET EXIST; their construction is the work this document plans. Phase 5 tests are specifications, not yet authored code. Phase 1 acceptance criteria are forward-looking contracts the implementation must satisfy. Research synthesis at `docs/research/code-vs-prose-vecs-2026-05.md` §5 was the original prototype shape; the locked decision to ship V+ replaces §4's Mem0 tie-break.

**Feature statement.** Add a CLI subcommand `vecs prose-drift -p <project>` and an MCP tool `mcp__vecs__prose_drift(project)` that surfaces contradictions between (a) `(subject, predicate, object)` triples extracted from indexed chat sessions and (b) triples extracted from indexed repo-prose docs. Session-side triples are persisted in a new per-project Chroma collection `<project>-prose-facts` under an INSERT/NOOP/SUPERSEDE state machine with bi-temporal columns (`valid_from`, `invalid_at`); no row is ever deleted, so the chain is queryable as of any historical timestamp. Doc-side triples are extracted on-demand at query time from already-indexed doc-chunks, cached by `(source_relpath, sha256(text))`, and NOT persisted into the facts collection. The extraction LLM is Anthropic Claude Opus 4.7 (`claude-opus-4-7`); embeddings stay on Voyage (`voyage-3`); the new facts collection is a sibling of `-code`, `-sessions`, `-docs`. Per-project scoping is the collection name. Implements the profile's Phase 2 `staleness_check` extension by promoting that profile slot to a list (`[commit-sha-tag, custom:prose-vplus]`), allowed by base v0.1.0's open-enum rule.

**Filesystem layout for this feature** (per profile + brief):
- Design doc (this file): `docs/features/prose-staleness-detector-design-v1.md`
- Acceptance: `docs/features/prose-staleness-detector/acceptance.md`
- Gap log: `docs/features/prose-staleness-detector/gaps.md`

---

## Phase 1 — Acceptance criteria

Produced by `architect` (`Plan`). Source: profile slot `acceptance_source` + `acceptance_format: checklist`.

- [ ] `vecs prose-drift -p <project>` exits 0 and prints `no prose drift` when the doc-side triple set (extracted fresh per call from indexed doc-chunks) contains zero entries whose `(subject, predicate)` collide with a session-side current-state triple (a row in `<project>-prose-facts` with `is_current=True`) whose `object` differs. (v1 writes only session-sourced rows; `source_type` field removed — see Fix 4 below.)
- [ ] `vecs prose-drift -p <project>` exits 1 and prints one line per drift in form `<subject> | <predicate> | doc="<object-from-doc>" @ <project>/<docs-dir-relpath> ≠ chat="<object-from-chat>" @ session=<session-id> (chat_history_versions=<int>)` for each collision. The `<docs-dir-relpath>` segment is the value of the Chroma doc-chunk metadata key `file_path` (cite `src/vecs/doc_chunker.py:103`), which is computed as `f.relative_to(project.docs_dir)` — relative to `project.docs_dir`, NOT relative to the repo root. Prefixing with `<project>/` makes the printed location unambiguous even when multiple projects share the same `team.md` filename. Order: stable, sorted by `(subject, predicate)`.
- [ ] `mcp__vecs__prose_drift(project: str | None = None)` returns a dict `{"drift": [{"subject": <str>, "predicate": <str>, "doc": {"object": <str>, "source": <relpath>}, "chat": {"object": <str>, "session_id": <str>}, "chat_history_versions": <int>}, ...], "facts_scanned": <int>, "facts_scanned_docs": <int>, "project": <str>}`. With `project=None`, scans every project where `prose_drift_enabled = true`; returns a dict keyed by project name. If no projects are enabled, returns `{}` (empty dict, NOT `{"projects": {}}`). Worked shape examples:

```python
# project=None, two enabled projects, one with drift:
{
  "vecs": {"drift": [...], "facts_scanned": 12, "facts_scanned_docs": 5, "project": "vecs"},
  "client-uk": {"drift": [], "facts_scanned": 8, "facts_scanned_docs": 3, "project": "client-uk"},
}
# project=None, zero enabled projects:
{}
# project="vecs", one project queried:
{"drift": [...], "facts_scanned": 12, "facts_scanned_docs": 5, "project": "vecs"}
```

- [ ] **BE-dev contradiction scenario** (executable per `--non-interactive` mode, see Phase 3): given the fixture at `tests/fixtures/prose_drift/` (a `docs/team.md` asserting "the team has no backend developer" and `sessions/be_dev_announce.jsonl` asserting "we hired Sasha as our BE dev"), the CLI exits 1 and the printed drift line contains `team`, `has_role`, both objects, and both source labels. Equivalent MCP call returns a `drift` list of length ≥ 1 with the same payload.
- [ ] Unknown project (`-p` not in `config.projects`) exits 2; stderr contains `unknown project: <name>`.
- [ ] Known project with `prose_drift_enabled = false` (or unset) exits 2; stderr contains `prose drift not enabled for project <name>`. Distinguished from unknown-project by stderr message; both share exit 2.
- [ ] Known project with `prose_drift_enabled = true` but zero indexed chat sessions exits 0; stdout contains `no chat sessions for project <name>`; MCP variant returns `{"drift": [], "facts_scanned": 0, "facts_scanned_docs": <int>, "project": <name>}` — doc-chunks may still exist for the project even when sessions are empty.
- [ ] When `ANTHROPIC_API_KEY` is unset, CLI exits 3 with stderr `ANTHROPIC_API_KEY not set`. MCP tool returns `{"error": "anthropic_key_missing"}` and does not raise. Check runs before any extraction call so no partial work occurs.
- [ ] When `anthropic` SDK is not importable, CLI exits 3 with stderr `anthropic not installed: pip install anthropic`. MCP tool returns `{"error": "anthropic_unavailable", "detail": "<import-error-str>"}` and does not raise. (Anthropic is the only new runtime dep; if absent at runtime, graceful degrade.)
- [ ] Extraction LLM model id is exactly `claude-opus-4-7`. Sourced from module-level constant `PROSE_EXTRACTION_MODEL` in `src/vecs/prose_drift.py`. The constant value is asserted in a unit test.
- [ ] Extraction calls do NOT pass a `temperature` parameter. `claude-opus-4-7` rejects `temperature` (including `temperature=0`) with `400 invalid_request_error`; verified empirically via spike. The SDK call payload is asserted in unit tests to contain no `temperature` key. Determinism is instead achieved via the verdict cache (see below) keyed on `(sha256(text), model)`.
- [ ] Embeddings for session-side facts and the doc-side query path reuse the existing Voyage client singleton via `get_voyage_client().embed(...)` (`src/vecs/clients.py:12`); model id `voyage-3` (matches `SESSIONS_MODEL` per `src/vecs/config.py:20`). No new Voyage account or key.
- [ ] Chroma collection name for facts is `<project>-prose-facts` (sibling to existing `<project>-code`, `<project>-sessions`, `<project>-docs` per `src/vecs/CLAUDE.md` "Invariants"). Code-vecs collections are never read or written by the prose-drift path.
- [ ] Per-project scoping is by collection name (`<project>-prose-facts`); no cross-project reads. A unit test asserts that calling the state machine for project `A` then querying project `B`'s collection returns zero of A's rows.
- [ ] **Row schema.** Each row in `<project>-prose-facts` carries:
  - `id`: UUID4 (per row; allows multiple rows in one logical chain).
  - `embedding`: `voyage(f"{subject} {predicate} {object}")`.
  - `document`: `f"{subject} {predicate} {object}"` (for `chroma.get(...)` debugging).
  - `metadata`:
    - `subject: str`
    - `predicate: str`
    - `object: str`
    - `chain_key: str = f"{subject}|{predicate}"` (composite; non-unique across rows; used to locate the current state and all historical rows for a (subject, predicate) chain)
    - `valid_from: int` (unix epoch ms; the moment this state became current)
    - `invalid_at: int | None` (unix epoch ms; the moment this state was superseded; `None` for the one current row in a chain)
    - `is_current: bool` (companion field; `True` for the one current row in a chain, `False` on superseded rows. Schema field name + bool-vs-int representation is pinned post-Phase-7-dry-run per Fix 1: if Chroma normalizes bool literals to int, swap to `is_current_int: 0|1` and update the where-clause accordingly. v1 ships whichever form the dry-run validates.)
    - `source_id: str` (session_id; all v1 rows are session-sourced)
    - `version: int` (starts at 1 on INSERT; bumps by 1 on each SUPERSEDE in the chain)
  - **Schema note**: the prior `source_type: "session" | "doc"` field has been REMOVED for v1. Every row is implicitly session-sourced; `source_id` already captures provenance. If symmetric doc-side persistence ships in v2, `source_type` is reintroduced with a migration.
- [ ] **Row-id strategy.** UUID4 per row + `chain_key` metadata column (chosen over the alternative `subject|predicate#vN` suffix scheme). Rationale: Chroma cannot enforce uniqueness on supersession, and version-suffix ids encode mutable state into the primary key (forcing id rewrites on SUPERSEDE). UUIDs are write-once; chain lookup uses `collection.get(where={"chain_key": ..., "invalid_at": None})` which Chroma supports natively via metadata filters. Documented constraint: Chroma's `where` clause for `None`-valued fields requires storing the absence as a sentinel; v1 uses `{"$ne": True}` on a companion boolean `is_current` metadata key set to `True` for the current row only, and `False` (or sentinel-equivalent) on superseded rows. (Implementation detail; the abstract semantics in this design remain `invalid_at IS NULL ⇔ current`.)
- [ ] **State machine (write path).** For each extracted session-side triple `(subject, predicate, object)`:
  - Lookup: `collection.get(where={"chain_key": "<subject>|<predicate>", "is_current": True})`. If multiple rows return with `is_current=True` for the same `chain_key` (transient SUPERSEDE-recovery state per Fix 2), the row with the highest `version` is treated as the operative current row, and lower-version rows are repaired in-line by flipping their `is_current` to `False` and setting `invalid_at=now_ms`.
  - If lookup empty (after repair) → **INSERT**: add a new row with `valid_from=now_ms, invalid_at=None, is_current=True, version=1`.
  - If lookup returns a single row with `object == new_object` → **NOOP**: no write.
  - If lookup returns a single row with `object != new_object` → **SUPERSEDE (reordered, crash-safe).** Write order is reversed from the naive form to make the transition crash-recoverable: (a) `collection.add(...)` a new row with `valid_from=now_ms, invalid_at=None, is_current=True, version=old_version+1` FIRST; (b) `collection.update(ids=[old_id], metadatas=[{..., invalid_at: now_ms, is_current: False}])` SECOND. Failure mode: if (a) succeeds and (b) fails, two rows briefly carry `is_current=True` for the same `chain_key`. Next lookup detects this via "len(current) > 1", picks the highest-version row as operative, and updates the lower-version row to `is_current=False, invalid_at=now_ms`. Failure mode: if (a) itself fails, no write lands; chain is unchanged; next run retries cleanly.
  - Event names INSERT/NOOP/SUPERSEDE are module-level constants `EVENT_INSERT = "INSERT"`, `EVENT_NOOP = "NOOP"`, `EVENT_SUPERSEDE = "SUPERSEDE"` in `src/vecs/prose_drift.py`.
- [ ] **No deletes.** No code path issues `collection.delete(...)` against `<project>-prose-facts`. Both SUPERSEDE and version bumps are append + flag-flip operations only. A grep test asserts this. History is queryable forever via `valid_from`/`invalid_at` filters.
- [ ] **Session-indexing wire-in.** After the success-path `manifest.save()` at `src/vecs/indexer.py:964`, only when `project.prose_drift_enabled = true`, **per fully-succeeded file** in the `fully_succeeded` set returned by `_track_embed_success` at `src/vecs/indexer.py:956`: extract triples via `extract_facts(user_messages, project)`, then for each triple call `add_fact_with_state_machine(triple, source_id=session_id, project=project.name) -> Event`. Failures inside extraction or the state machine are caught and logged via `_log(...)` per file; they do not abort the indexer, do not break the per-file loop, and do not roll back the Chroma write of the underlying session chunks. The new per-file block is inserted as the lines immediately following the success-path `manifest.save()` (i.e., becomes new lines starting at `:965` after the edit; line numbers shift on insertion). The early-return `manifest.save()` at `:951` is unrelated and untouched.
- [ ] **Per-file scoping (no double-faceting).** Only files in the `fully_succeeded` set (the set returned by `_track_embed_success` at `src/vecs/indexer.py:956`) get state-machine calls on this run. Files that partially failed are NOT faceted on this run; they will be re-indexed on the next run (because `manifest.mark_session_indexed` was skipped for them at `:959-962`), at which point the state machine runs for them once their chunks fully succeed. This prevents duplicate INSERT events when a partial-success file is re-fed.
- [ ] **Per-message stash.** Implementation captures the per-file `messages` list during the first loop at `src/vecs/indexer.py:904` (where each file's `messages = parser_fn(content)` is already in scope at `:912`) into a dict `file_messages: dict[Path, list[dict]]`. The post-success state-machine loop reads `messages = file_messages[f]` rather than re-parsing the file. No additional file I/O.
- [ ] **Role filter.** Session-side triples are extracted from `role == "user"` messages only. Assistant and system turns are excluded from `extract_facts` to prevent hallucinated/speculative model output from polluting the drift baseline. Filter is applied at the preprocessed-message level (output of `preprocess_session` at `src/vecs/chunkers.py:46`) before the messages reach `extract_facts`.
- [ ] `index_docs` at `src/vecs/indexer.py:1033` is NOT modified. Zero state-machine writes happen on the doc-indexing path. A grep test asserts this.
- [ ] Doc-side triples are extracted at query time via `extract_facts_from_doc(text: str, source_relpath: str) -> list[Triple]` in `src/vecs/prose_drift.py`. Results are cached in the verdict cache (see below). Doc triples are NEVER written into `<project>-prose-facts` in v1.
- [ ] **Verdict cache.** A per-project SQLite database at `~/.vecs/prose_drift_cache/<project>.db` (WAL mode) houses two tables that coexist and serve different access patterns:

```sql
CREATE TABLE IF NOT EXISTS doc_facts (
    source_relpath TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    triples_json TEXT NOT NULL,
    extracted_at INTEGER NOT NULL,
    PRIMARY KEY (source_relpath, sha256)
);

CREATE TABLE IF NOT EXISTS extraction_cache (
    text_sha TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    triples_json TEXT NOT NULL,
    extracted_at INTEGER NOT NULL,
    PRIMARY KEY (text_sha, model, prompt_version)
);

PRAGMA journal_mode = WAL;
```

- `doc_facts(source_relpath, sha256, triples_json, extracted_at)` — fast lookup for the doc-side path; `sha256` is `sha256(text.encode("utf-8")).hexdigest()` over raw doc-chunk text (no canonicalization needed; doc text has no role/timestamp metadata to strip).
- `extraction_cache(text_sha, model, prompt_version, triples_json, extracted_at)` — keyed by canonical hash (per Fix 3 for session-side; raw `sha256(text.encode("utf-8"))` for doc-side).
- **Invariant**: for any doc-chunk processed via `extract_facts_from_doc(text, source_relpath)`, `doc_facts.sha256 == extraction_cache.text_sha` (when both rows exist). `doc_facts` is the per-`source_relpath` lookup; `extraction_cache` is the content-keyed lookup. The two tables coexist but serve different access patterns; the 'join' (when needed) is on this equality. Session-side text passes through `_extract_cache_key_text` first; doc-side text does NOT (no role/timestamp fields to strip).

Both `extract_facts(messages, project)` and `extract_facts_from_doc(text, source_relpath)` MUST consult `extraction_cache` (keyed on `(text_sha, PROSE_EXTRACTION_MODEL, EXTRACTION_PROMPT_VERSION)`) before issuing an Anthropic call. Cache hit → return parsed triples; cache miss → call Anthropic, persist result, return. For `extract_facts_from_doc`, the doc-side write path persists to BOTH `doc_facts` (per-relpath index) and `extraction_cache` (content-keyed cache), with `doc_facts.sha256 == extraction_cache.text_sha` by construction.

- [ ] **Cache canonicalization** (Fix 3). `text_sha` for `extract_facts(messages: list[dict], ...)` is computed via a canonical serialization that excludes per-message timestamps and metadata; only `role` and `text` participate (matches `preprocess_session` output keys at `src/vecs/chunkers.py:46`). Helper definitions in `src/vecs/prose_drift.py`:

```python
def _extract_cache_key_text(messages: list[dict]) -> str:
    """Canonical text for cache hashing. Excludes per-message timestamps and metadata."""
    minimal = [{"role": m["role"], "text": m["text"]} for m in messages]
    return json.dumps(minimal, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def _extract_cache_key(messages: list[dict], model: str, prompt_version: str) -> tuple[str, str, str]:
    text = _extract_cache_key_text(messages)
    return (hashlib.sha256(text.encode("utf-8")).hexdigest(), model, prompt_version)
```

Cache PK is `(text_sha, model, prompt_version)`. The key is stable across dict-key ordering and per-message timestamp variation; bumping `EXTRACTION_PROMPT_VERSION` invalidates all prior entries with the same `(text_sha, model)`.

- [ ] **Prompt versioning** (Fix 7). Module constant `EXTRACTION_PROMPT_VERSION: str = "v1"` in `src/vecs/prose_drift.py`. The constant is bumped (e.g., to `"v2"`) on every prompt-template edit. The `extraction_cache` PK includes `prompt_version`, so prompt revisions automatically invalidate stale extractions — bug-fix prompt changes cannot silently bypass the cache.

- [ ] **Empty-triple-list extraction** (Fix 6). `extract_facts(...)` returning `[]` is an explicit success path (chat may contain only "thanks" / "rerun that" with no extractable facts). The empty result is **positive-cached** as `triples_json = "[]"` so identical re-extractions skip the Anthropic call. The per-file log line emits `triples=0` for that file. The code path has no special branch beyond storing and returning the empty list; the absence of triples is normal.

- [ ] Code-vecs path untouched: zero new imports of `anthropic` or `prose_drift` inside `indexer.index_code`, `chunkers`, `ast_chunker`, `bm25_index`, or `searcher`. Verified by grep test (see Phase 5).
- [ ] Re-running `vecs prose-drift -p <project>` twice in succession with no new sessions or doc changes returns the same drift list (deterministic over the facts collection state + verdict cache; Opus 4.7 non-determinism is neutralized by the cache lookup on the second run).
- [ ] CLI `--limit N` flag caps the number of drift lines printed (default 50); exit code is still 1 when truncated and stderr notes `drift truncated: showing N of M`. Truncation preserves the `(subject, predicate)` sort order — printed lines are the first N of the sorted set.
- [ ] When the new YAML key `prose_drift_enabled` is absent from a project's config block, `load_config` parses it as `False` (no exception). Existing configs continue loading. YAML schema (under each project block in `~/.vecs/config.yaml`):

```yaml
projects:
  - name: vecs
    code_dirs: [src/]
    sessions_dirs: [~/.claude/projects/]
    docs_dir: docs/
    prose_drift_enabled: true   # new field; default False if absent
```

- [ ] After this feature lands, `src/vecs/CLAUDE.md` "Staleness baseline" section reads `staleness_check: [commit-sha-tag, custom:prose-vplus]` (list-valued; baseline migration).
- [ ] After this feature lands, `src/vecs/CLAUDE.md` contains a new bullet under the existing "Tests" section reading verbatim: `Opt-in integration tests gated by VECS_TEST_REAL_LLM=1 — real LLM calls; default-skipped in CI. See tests/test_prose_drift.py::test_integration_real_anthropic.`
- [ ] `facts_scanned` is precisely defined as the count of rows in `<project>-prose-facts` with `is_current=True` at scan time. (No `source_type` filter; v1 only writes session-sourced rows by construction per Fix 4.) `facts_scanned_docs` is `len(list(iterate_indexed_docs(project)))` at scan time — the doc-chunk cardinality iterated for the call. Both are added to the MCP return payload as `int`.
- [ ] **No `--as-of` flag in v1** (Fix 6/MAJOR 5). The time-travel query (`--as-of <ISO-date>`) is explicitly NOT part of the v1 CLI surface. argparse's default behavior for unknown flags (exit 2 with usage error) is sufficient; no custom stub message or stub test is required. The time-travel idea is parked in `docs/features/prose-staleness-detector/v2-roadmap.md` for future work.

## Phase 2 — Required context

Produced by `investigator` (`caveman:cavecrew-investigator`) + `explorer` (`Explore`).

**Exists in vecs:** (file:line cites verified by direct file read 2026-05-21)

- Session indexing pipeline `_index_session_files(...)` defined at `src/vecs/indexer.py:855` (def). `_embed_and_store(...)` invocation inside the loop at `src/vecs/indexer.py:954` (call site). `_embed_and_store` itself defined at `:385` with its `collection.upsert(...)` at `:449`. The success-path `manifest.save()` lands at `:964`; an early-return `manifest.save()` (no chunks) lands at `:951`. **The wire-in point for state-machine writes is AFTER the success-path `manifest.save()` at `:964`** (see "Failure-mode ordering" below for rationale).
- `index_sessions(...)` at `src/vecs/indexer.py:972` (Claude Code entry) and the Codex equivalent both delegate to `_index_session_files`; one wire-in covers both agents.
- `index_docs(...)` at `src/vecs/indexer.py:1033` — **untouched by this feature**. No state-machine call lands here. Doc-fact extraction is query-time only.
- Chunker entry points: `preprocess_session(raw_jsonl)` at `src/vecs/chunkers.py:46` returns `list[{"role": str, "text": str, "timestamp": str}]` (per-message records, role preserved). `chunk_session(messages, session_id, ...)` at `src/vecs/chunkers.py:99` groups N messages into a chunk dict `{"text": <concatenated f"[{role}]: {text}" string>, "metadata": {...}}` (see `src/vecs/chunkers.py:117-128`). Chunk dicts do NOT have a per-chunk `role` key — roles are inlined into the `text` blob (`src/vecs/chunkers.py:113`), so role-filtering must happen at the `preprocess_session` output stage, before `chunk_session` is called.
- Per-doc-chunk source metadata is stored in the Chroma `metadata` dict per chunk under the existing `docs_collection` schema under the key `file_path` (cite `src/vecs/doc_chunker.py:103`). This is the relpath/abspath the chunker wrote when emitting the chunk; `iterate_indexed_docs` reads it back unchanged.
- Codex routing: `discover_codex_sessions(config)` defined at `src/vecs/codex_routing.py:257`; per-session id derivation at `:116`.
- Voyage client wrapper: `get_voyage_client() -> voyageai.Client` at `src/vecs/clients.py:12`; singleton init `_vo_client = voyageai.Client()` inside the function body. Uses `VOYAGE_API_KEY` env.
- ChromaDB client: `get_chromadb_client() -> chromadb.ClientAPI` at `src/vecs/clients.py:20`; persistent path resolves via `src/vecs/config.py` `CHROMADB_DIR`.
- Per-project collection naming: `ProjectConfig.code_collection` at `src/vecs/config.py:56`; `sessions_collection` at `:60`; `docs_collection` at `:64`. The new `-prose-facts` collection is a sibling under the same pattern.
- `ProjectConfig` dataclass and `load_config(...)` at `src/vecs/config.py:159` already tolerate missing fields — a new optional `prose_drift_enabled: bool = False` follows the same pattern.
- `SESSIONS_MODEL = "voyage-3"` declared at `src/vecs/config.py:20`.
- CLI subcommand pattern: `@main.command()` in `src/vecs/cli.py`.
- MCP tool pattern: `@mcp.tool()` in `src/vecs/mcp_server.py` (see registered tools such as `semantic_search`, `reindex`).
- Test fixture pattern: `@pytest.fixture def cfg(tmp_path)` at `tests/test_codex_routing.py:31`.
- `pyproject.toml` dependency list: no `anthropic` present today.
- `src/vecs/CLAUDE.md` "Staleness baseline" section currently reads `staleness_check: commit-sha-tag` (scalar) — to be promoted to list under this feature.

**Must add:**
- `anthropic` to `pyproject.toml` dependencies, pinned to one known-good version (exact pin chosen at implementation time against the published Anthropic SDK release at that moment). This is the only new runtime dep — V+ does not depend on `mem0ai`, `graphiti`, or any third-party memory store. Stays minimal.
- No new dev dependencies for testing. Mocking uses pytest's `monkeypatch` fixture (already available via `pytest>=7` declared in `pyproject.toml`).
- `prose_drift_enabled: bool = False` on `ProjectConfig` (default `False` keeps existing configs and flows untouched). YAML key: `prose_drift_enabled` (nested under each project block — see Phase 1 example).
- New `prose_facts_collection` property on `ProjectConfig` returning `f"{self.name}-prose-facts"`.
- New module `src/vecs/prose_drift.py` housing:
  - `PROSE_EXTRACTION_MODEL = "claude-opus-4-7"` (module constant)
  - `EXTRACTION_PROMPT_VERSION: str = "v1"` (module constant; bumped on every prompt-template edit per Fix 7)
  - Event constants `EVENT_INSERT = "INSERT"`, `EVENT_NOOP = "NOOP"`, `EVENT_SUPERSEDE = "SUPERSEDE"`
  - `@dataclass class Triple: subject: str; predicate: str; object: str; source: str`
  - `extract_facts(messages: list[dict], project: str) -> list[Triple]` — session-side. Reads `extraction_cache` (PK `(text_sha, model, prompt_version)`) first; on miss calls Anthropic. Empty-result extractions are positive-cached as `triples_json = "[]"` (Fix 6). Called inside the per-file post-success block (see wire-in). The `messages` argument is the output of `preprocess_session(...)` at `src/vecs/chunkers.py:46` (or `preprocess_codex_session(...)` at `src/vecs/codex_chunker.py:63`) — each dict has keys `{"role": str, "text": str, "timestamp": str}`. `_extract_cache_key_text` reads `m["role"]` and `m["text"]` only; `m["timestamp"]` is excluded from the cache key by design (per Fix 3 cache-canonicalization). Function never accesses `m["content"]` — that key does not exist in this schema.
  - `extract_facts_from_doc(text: str, source_relpath: str) -> list[Triple]` — doc-side, query-time. Reads `extraction_cache` first (keyed on `(sha256(text.encode("utf-8")).hexdigest(), PROSE_EXTRACTION_MODEL, EXTRACTION_PROMPT_VERSION)` — doc text has no role/timestamp metadata, so the raw sha256 IS the canonical hash; no `_extract_cache_key_text` pass needed); on miss calls Anthropic; writes back to BOTH `extraction_cache` (with `text_sha = sha256(text.encode("utf-8")).hexdigest()`) AND `doc_facts` (with `sha256 = sha256(text.encode("utf-8")).hexdigest()` — same value, by construction). The join invariant `doc_facts.sha256 == extraction_cache.text_sha` holds for every row pair on the doc-side path (Pass-3 Fix 2). Doc triples are returned in-memory; never written to Chroma in v1.
  - `add_fact_with_state_machine(triple: Triple, source_id: str, project: str) -> str` — returns one of `EVENT_INSERT`, `EVENT_NOOP`, `EVENT_SUPERSEDE`. Implements the state machine spec from Phase 1 against `<project>-prose-facts`. (Note: the `source_type` parameter has been REMOVED per Fix 4; all v1 rows are session-sourced.)
  - `_preflight_global(config) -> Result` and `_preflight_project(config, project_name) -> Result` — see "Entry-point pre-flight" subsection below for the two-stage split (Pass-3 Fix 3). Both return `Ok` or `Err(code, [detail])`; CLI maps codes to exit 2/3, MCP maps codes to single-error dicts. The MCP `project=None` flow consults only the global stage and silently skips disabled/unknown projects.
  - `iterate_indexed_docs(project: str) -> Iterator[tuple[str, str]]` — iterates the Chroma `<project>-docs` collection via `get_chromadb_client().get_collection(f"{project}-docs")`, yields `(chunk_text, source_relpath)` tuples. Uses Chroma's `.get(include=["documents", "metadatas"])` API; `source_relpath` comes from each chunk's metadata key `file_path` (cite `src/vecs/doc_chunker.py:103`). Reads `metadatas[i]["file_path"]`, no fallback key.
  - `find_prose_drift(project_config) -> DriftReport` — iterates doc chunks, extracts triples per chunk, queries `<project>-prose-facts` for the current-state row per `chain_key`, yields drift entries when objects differ.
  - `ProseDriftError` exception hierarchy: `AnthropicUnavailable`, `AnthropicKeyMissing`, `ProseDriftDisabled`.
- The SQLite cache at `~/.vecs/prose_drift_cache/<project>.db` uses the DDL above (two tables: `doc_facts`, `extraction_cache`). Single-writer assumption: vecs is solo/POC scale; concurrent `vecs prose-drift` invocations on the same project are out-of-scope for v1 (documented in Phase 8 gap candidates). Reads use a connection-per-call; writes use IMMEDIATE transactions.
- New CLI subcommand `prose-drift` in `cli.py` (lazy import of `prose_drift` inside the function body).
- New MCP tool `prose_drift` in `mcp_server.py` (lazy import inside the function body). The MCP tool is registered as the Python function `prose_drift` in `src/vecs/mcp_server.py` via `@mcp.tool()`; the resulting fully-qualified MCP tool name exposed to clients is `mcp__vecs__prose_drift` (matching the existing convention for `mcp__vecs__semantic_search`, `mcp__vecs__reindex`, etc.). Acceptance refers to the client-facing name; the design refers to the function name.
- New test module `tests/test_prose_drift.py`.
- New test fixture directory `tests/fixtures/prose_drift/` containing `docs/team.md` ("the team has no backend developer") and `sessions/be_dev_announce.jsonl` (minimal valid session asserting "we hired Sasha as our BE dev"). Used by Phase 5 (`test_be_dev_contradiction_surfaces`) and Phase 7 dry-run.
- `src/vecs/CLAUDE.md` "Staleness baseline" updated from scalar to list: `staleness_check: [commit-sha-tag, custom:prose-vplus]`. Also adds a new bullet under the existing "Tests" section: `Opt-in integration tests gated by VECS_TEST_REAL_LLM=1 — real LLM calls; default-skipped in CI. See tests/test_prose_drift.py::test_integration_real_anthropic.`
- Per-module CLAUDE.md updates for `cli.py`, `mcp_server.py`, `indexer.py`, `config.py`, and the new `prose_drift.py` (profile Phase 2 `context_coverage_rule: touched-modules`).

**Cost decision.** Extraction LLM is Anthropic Claude Opus 4.7 (`claude-opus-4-7`) per locked decision. Estimated cost at vecs scale: ~$0.80/month at ~1500 extraction calls/month (single-developer chat volume) using the verdict cache to deduplicate. Two orders of magnitude below the prior design's ~$15-25/mo Mem0+Opus estimate, because (a) the verdict cache eliminates re-extraction on identical text and (b) V+ skips Mem0's internal LLM consolidation calls (Mem0 was making at least 2 calls per `add`; V+ makes one). If a cost ceiling later triggers, the model swap to Sonnet is a single constant change (`PROSE_EXTRACTION_MODEL = "claude-sonnet-4-6"`). Documented here, not relitigated.

**Temperature constraint (NEW invariant).** `claude-opus-4-7` rejects requests carrying a `temperature` parameter with `400 invalid_request_error: temperature parameter is deprecated for this model`. Verified empirically via spike on 2026-05-21. The Anthropic SDK call MUST NOT include `temperature` in its kwargs. Determinism is achieved via the `extraction_cache` SQLite table keyed on `(sha256(text), model)` — once a text-shape is extracted once, all subsequent calls hit the cache and return identical triples. This trades request-level determinism for cache-level determinism; under steady state (cache warm), behavior is equivalent.

**Failure-mode ordering** (concurrency / partial-write rationale). The wire-in sits **AFTER** `manifest.save()` at `:964`, not before. Ordering: `_embed_and_store(...) → _track_embed_success(...) → manifest.save() → for f in fully_succeeded: try: extract + state-machine except: _log(...)`. Per-file scoping is load-bearing: `_track_embed_success` at `:956` separates `fully_succeeded` files from partial-success files; only the former had `manifest.mark_session_indexed` called at `:959-962`. If the state machine runs only for `fully_succeeded` files, re-indexing on the next run will not re-run for those files (their manifest entries are up to date). Partial-success files are intentionally NOT faceted on this run; they re-enter the loop on the next run with the same chunks, and if all chunks succeed that time, the state machine runs for them then — exactly once per file across runs. This prevents the facts collection from receiving the same triples twice and generating spurious SUPERSEDE events. The trade: a state-machine failure leaves the corresponding file un-faceted in the facts collection until a manual retrigger or a future per-fact reconciliation pass lands (since the manifest already records the file as indexed and won't replay it). Accepted for v1; tracked as future work.

**Invariant changes (explicit).** Treat these as breaks, not incidentals:
1. **First `anthropic` dependency in `pyproject.toml`.** No `mem0ai` is added; this is V+'s only new runtime dep.
2. **First Anthropic API call in `src/vecs/`.** Prior LLM-adjacent calls were embedding-only (Voyage). Outbound traffic to `api.anthropic.com` is a new invariant.
3. **First `ANTHROPIC_API_KEY` environment-variable dependency** in `src/vecs/`.
4. **New Chroma collection `<project>-prose-facts` owned by vecs (NOT by a third-party library).** Schema is documented in Phase 1; vecs owns reads and writes end-to-end.
5. **First feature flag (`prose_drift_enabled`) gating a code path inside `_index_session_files`.**
6. **`staleness_check` profile slot becomes list-valued** in `src/vecs/CLAUDE.md`. Open-enum semantics in base v0.1.0 allow it.
7. **New on-disk cache** at `~/.vecs/prose_drift_cache/<project>.db` for the verdict cache (`doc_facts` + `extraction_cache` tables). Sibling to existing `~/.vecs/chromadb/`, `~/.vecs/manifests/`, BM25 sidecar `.db` files.
8. **First bi-temporal storage pattern in vecs.** `valid_from`/`invalid_at` columns mirror the Graphiti/Zep and XTDB/Datomic pattern. No row is ever deleted from `<project>-prose-facts`.

Each invariant is recorded as a gap candidate for Phase 8 if it surfaces friction during dry-run.

**Per-file wire-in pseudo-code** (success-path block inserted after `manifest.save()` at `src/vecs/indexer.py:964`):

```python
# new — runs ONLY when project.prose_drift_enabled is True
if project.prose_drift_enabled:
    try:
        from vecs.prose_drift import extract_facts, add_fact_with_state_machine
        for f in fully_succeeded:
            session_id = f.stem  # matches indexer.py:910
            messages = file_messages.get(f)  # captured during the :904 loop
            if not messages:
                continue
            user_messages = [m for m in messages if m.get("role") == "user"]
            if not user_messages:
                continue
            try:
                triples = extract_facts(user_messages, project.name)
            except Exception as e:
                _log(f"[{project.name}] prose extract failed for {f.name}: {e}")
                continue
            if not triples:
                # Fix 6 — empty-triple-list is a success path. Cache stores "[]".
                _log(f"[{project.name}] prose-drift {f.name}: triples=0")
                continue
            counts = {"INSERT": 0, "NOOP": 0, "SUPERSEDE": 0}
            for t in triples:
                try:
                    event = add_fact_with_state_machine(
                        t, source_id=session_id, project=project.name,
                    )
                    counts[event] = counts.get(event, 0) + 1
                except Exception as e:
                    _log(f"[{project.name}] prose state-machine failed for {f.name} triple {t}: {e}")
                    continue
            _log(
                f"[{project.name}] prose-drift {f.name}: "
                f"INSERT={counts['INSERT']} NOOP={counts['NOOP']} SUPERSEDE={counts['SUPERSEDE']}"
            )
    except ImportError as e:
        _log(f"[{project.name}] anthropic not installed; skipping prose-drift facet: {e}")
```

Notes on the pseudo-code:
- `file_messages: dict[Path, list[dict]]` is populated inside the first loop at `:904` (where `messages = parser_fn(content)` already exists at `:912`). Implementation adds one line `file_messages[f] = messages` inside that loop.
- The per-file try/except for extract and per-triple try/except for state-machine ensures one failing file or one bad triple does not block the remainder. The outer `ImportError` catch covers the case where `anthropic` is absent at indexer call time despite `prose_drift_enabled=true` (degrades gracefully; indexer still completes).
- Per-file log line surfaces INSERT/NOOP/SUPERSEDE counts so an operator running `vecs reindex` sees state-machine activity inline.

**State-machine pseudo-code** (lives in `src/vecs/prose_drift.py`). Note: SUPERSEDE writes are reordered (add-first, update-second) to be crash-safe per Fix 2; the lookup branch repairs transient duplicates per the same fix; `source_type` parameter is removed per Fix 4.

```python
def add_fact_with_state_machine(triple, source_id, project) -> str:
    db = get_chromadb_client()
    collection = db.get_or_create_collection(f"{project}-prose-facts")
    chain_key = f"{triple.subject}|{triple.predicate}"
    now_ms = int(time.time() * 1000)
    current = collection.get(where={"chain_key": chain_key, "is_current": True})

    # Fix 2 — recovery: if more than one row reports is_current=True for this
    # chain_key (transient state after a crashed SUPERSEDE), the highest-version
    # row wins; lower-version rows are demoted to is_current=False.
    if len(current["ids"]) > 1:
        rows = sorted(
            zip(current["ids"], current["metadatas"]),
            key=lambda pair: pair[1].get("version", 0),
            reverse=True,
        )
        operative_id, operative_meta = rows[0]
        for stale_id, stale_meta in rows[1:]:
            repaired = dict(stale_meta)
            repaired["invalid_at"] = now_ms
            repaired["is_current"] = False
            collection.update(ids=[stale_id], metadatas=[repaired])
        current = {"ids": [operative_id], "metadatas": [operative_meta]}

    if not current["ids"]:
        # INSERT
        new_id = str(uuid.uuid4())
        doc = f"{triple.subject} {triple.predicate} {triple.object}"
        emb = get_voyage_client().embed([doc], model=SESSIONS_MODEL).embeddings[0]
        collection.add(
            ids=[new_id], embeddings=[emb], documents=[doc],
            metadatas=[{
                "subject": triple.subject, "predicate": triple.predicate, "object": triple.object,
                "chain_key": chain_key,
                "valid_from": now_ms, "invalid_at": None, "is_current": True,
                "source_id": source_id,
                "version": 1,
            }],
        )
        return EVENT_INSERT

    cur_meta = current["metadatas"][0]
    if cur_meta["object"] == triple.object:
        return EVENT_NOOP

    # SUPERSEDE — crash-safe ordering (Fix 2): add NEW first, then flip OLD.
    old_id = current["ids"][0]
    new_id = str(uuid.uuid4())
    doc = f"{triple.subject} {triple.predicate} {triple.object}"
    emb = get_voyage_client().embed([doc], model=SESSIONS_MODEL).embeddings[0]

    # Step 1: durably write the new current row BEFORE touching the old one.
    collection.add(
        ids=[new_id], embeddings=[emb], documents=[doc],
        metadatas=[{
            "subject": triple.subject, "predicate": triple.predicate, "object": triple.object,
            "chain_key": chain_key,
            "valid_from": now_ms, "invalid_at": None, "is_current": True,
            "source_id": source_id,
            "version": cur_meta["version"] + 1,
        }],
    )
    # Step 2: flip the old row. If this fails, the chain has two is_current=True
    # rows; the next lookup's repair branch (above) restores the invariant.
    old_meta = dict(cur_meta)
    old_meta["invalid_at"] = now_ms
    old_meta["is_current"] = False
    collection.update(ids=[old_id], metadatas=[old_meta])
    return EVENT_SUPERSEDE
```

**State-machine invariants** (Fix 10).

1. For every `chain_key` X in `<project>-prose-facts`, AT MOST one row has `is_current=True`. After Fix 2's reordered-write recovery, transient violations may briefly exist (between Step 1's `add` and Step 2's `update` in SUPERSEDE); the next-lookup repair branch restores the invariant by selecting the highest-version row.
2. For every row with `is_current=False`, `invalid_at IS NOT NULL`.
3. For every row with `is_current=True`, `invalid_at IS NULL`.
4. `version` is monotonically increasing per `chain_key` in supersession order.
5. Rows are never DELETEd by v1 code. v2 compaction is a future feature.

**Entry-point pre-flight** (Fix 9 + Pass-3 Fix 3 — split into global and per-project stages so MCP `project=None` is well-defined).

Preflight is split into two stages so `mcp__vecs__prose_drift(project=None)` (which scans many projects) has unambiguous semantics:

```python
def _preflight_global(config) -> Result:
    """Checks invariants that apply to any prose-drift invocation regardless of project."""
    if not anthropic_importable():
        return Err("anthropic_unavailable", import_error_str)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return Err("anthropic_key_missing")
    return Ok()

def _preflight_project(config, project_name) -> Result:
    """Checks invariants specific to one project."""
    if project_name not in config.projects:
        return Err("project_unknown", project_name)
    if not config.projects[project_name].prose_drift_enabled:
        return Err("prose_drift_disabled", project_name)
    return Ok()
```

Both stages live in a tiny utility module (or inline in `cli.py` / `mcp_server.py`) so the `find_prose_drift` import — and therefore the `anthropic` import chain — stays lazy.

**CLI `vecs prose-drift -p <project>` flow:**
1. `_preflight_global(config)` → on `Err`, CLI exit 3 with documented stderr (`anthropic not installed: ...` or `ANTHROPIC_API_KEY not set`).
2. On `Ok`: `_preflight_project(config, project)` → on `Err`, CLI exit 2 with documented stderr (`unknown project: <name>` or `prose drift not enabled for project <name>`).
3. On `Ok`: import + run `find_prose_drift(project)` + emit drift report.

**MCP `mcp__vecs__prose_drift(project=None)` flow:**
1. `_preflight_global(config)` → on `Err`, return `{"error": <code>, "detail": <...>}` (single-error dict, NOT a per-project dict). MCP caller gets unambiguous failure.
2. On `Ok`: iterate `config.projects.values()` filtering `prose_drift_enabled == True`. SKIP (do not fail) disabled or unknown projects silently. For each enabled project: run `find_prose_drift(project)` and add the result under that project's name in the return dict.
3. Zero enabled projects → return `{}` (exact empty dict).

**MCP `mcp__vecs__prose_drift(project=<name>)` flow:**
1. `_preflight_global(config)` → on `Err`, return single-error dict.
2. On `Ok`: `_preflight_project(config, project)` → on `Err`, return single-error dict (`{"error": "project_unknown", "detail": <name>}` or `{"error": "prose_drift_disabled", "detail": <name>}`).
3. On `Ok`: import + run `find_prose_drift(project)` and return its result.

Preflight runs BEFORE `find_prose_drift` is imported. Missing `anthropic` is caught early without crashing the entry point; the unit test `test_preflight_runs_before_find_prose_drift_import` asserts the ordering.

## Phase 3 — Validation pipeline

Inherited from profile:
- `pipeline_runner: uv run pytest -q`
- `acceptance_input_adapter: scripts/check_acceptance.py prose-staleness-detector`
- `pre_merge_gate: advisory`
- `ci_config_path: "none"`

`--non-interactive` mode for the BE-dev scenario: the acceptance file at `docs/features/prose-staleness-detector/acceptance.md` lists the BE-dev contradiction as a checklist item marked `[ ]`. The adapter in `--non-interactive` mode treats pre-checked `[x]` items as pass; BE-dev remains `[ ]` until the implementor manually runs the fixture end-to-end. The unit-test version of the same scenario (Phase 5 `test_be_dev_contradiction_surfaces`) is what the pipeline (`uv run pytest -q`) actually exercises automatically; the acceptance checkbox is the operator-attested live-LLM version.

**`scripts/check_acceptance.py` section-boundary behavior** (Fix 8/MAJOR 8). The acceptance file is split into two sections: an executable checklist and a separately-headed `## Operator sign-off (out-of-band; not gated by --non-interactive)` section. In `--non-interactive` mode, items under `## Operator sign-off` are SKIPPED — they are not counted in pass/fail and do not block the gate. Items above that heading are gated normally (pre-checked `[x]` = pass; `[ ]` = fail). Operator-attested items (live-LLM run, env-var setup confirmation, on-disk cache acknowledgement, cost-band acknowledgement) live under the Operator section so CI can pass without manual ticks.

Profile's Phase 3 cross-phase rule does not bind (gate is advisory; Phase 5 is enabled anyway).

## Phase 4 — Review loop

Inherited from profile:
- `loop_participants: [architect, critical-sinker, reviewer]`
- `loop_kill_criteria: reviewer-or-no-progress`
- `progress_detector.strategy: manual`
- `escalation_target: "human"`

**No-progress definition for this feature.** Two consecutive reviewer passes that emit the same set of BLOCKER+MAJOR findings (set equality on finding-id, not full text) constitute no progress. The architect-or-builder must change at least one BLOCKER/MAJOR or escalate to human. MINOR findings carrying across passes do not trigger no-progress. Reviewer must emit one verdict line per iteration (profile's `specialist_scope_rule`); that line is the falsifiable signal. Finding-ids: critical-sinker numbers gaps sequentially per pass (G-1, G-2, ...); reviewer cites BLOCKER/MAJOR findings as R-1, R-2, ...; the no-progress detector compares the set of `(pass-N+1 BLOCKER/MAJOR ids, deduped)` against pass-N. Gap-log `docs/features/prose-staleness-detector/gaps.md` archives finding-ids per pass and the decision taken (`addressed`, `accepted-risk`, `deferred-to-future`).

## Phase 5 — Test substrate

Produced by `architect`. Lives at `tests/test_prose_drift.py`. Fixture data lives at `tests/fixtures/prose_drift/` (`docs/team.md` + `sessions/be_dev_announce.jsonl`). Mocks Anthropic in unit tests; one opt-in integration test (gated by `VECS_TEST_REAL_LLM=1`) exercises a real Anthropic call against a real Chroma collection.

| Test | Assertion |
|---|---|
| `test_extract_facts_calls_anthropic_with_opus47` | Mock anthropic client; `extract_facts(messages, project)` issues one call with `model="claude-opus-4-7"` |
| `test_extraction_does_not_pass_temperature` | The Anthropic SDK call payload does NOT contain a `temperature` key. Asserts wire-level kwargs absence (deprecated parameter for `claude-opus-4-7`). |
| `test_extract_facts_returns_subject_predicate_object` | Mocked LLM response → parsed into `list[Triple]` with `subject, predicate, object, source` fields |
| `test_extraction_cache_hits_on_repeat_text` | Call `extract_facts` twice with identical `messages` → second call hits `extraction_cache`, zero new Anthropic calls |
| `test_extraction_cache_misses_on_text_change` | Same input identifier, mutated text → cache miss, fresh Anthropic call issued |
| `test_extraction_cache_keys_on_model` | Same text + different `model` constant → cache miss; cache is keyed on `(text_sha, model, prompt_version)` |
| `test_cache_invalidates_on_prompt_version_bump` (Fix 7) | Same `(text_sha, model)` but different `prompt_version` → cache miss; bumping `EXTRACTION_PROMPT_VERSION` invalidates stale extractions |
| `test_extract_cache_key_stable_across_dict_order` (Fix 3) | Same messages serialized with different dict-key ordering → same cache key (canonical `sort_keys=True`) |
| `test_extract_cache_key_ignores_timestamps` (Fix 3) | Same `role`+`text`, different per-message timestamps → same cache key |
| `test_extract_cache_key_serialization_round_trip` (Fix 3) | `_extract_cache_key_text(messages)` is deterministic across repeated invocations on identical inputs |
| `test_empty_triple_list_logs_zero_caches_and_does_not_error` (Fix 6) | `extract_facts` returns `[]` → per-file log `triples=0`; cache entry stored with `triples_json="[]"`; second call hits cache and issues zero new Anthropic calls |
| `test_doc_fact_cache_invalidates_on_text_change` | Same `source_relpath`, mutated `text` → `doc_facts` cache miss, fresh extraction |
| `test_extraction_cache_ddl_initializes_on_first_use` | Calling `extract_facts` against an empty cache dir creates the `.db` with both tables (`doc_facts`, `extraction_cache`) and WAL mode |
| `test_insert_event_on_new_chain` | `add_fact_with_state_machine(triple, ...)` on an empty collection → returns `EVENT_INSERT`; one row exists; `version=1`, `valid_from` is recent, `invalid_at is None`, `is_current=True` |
| `test_noop_event_on_identical_triple` | Pre-seed one row for `(team, has_role, "no BE dev")`. Call state machine with the same triple → returns `EVENT_NOOP`; row count unchanged; original row's `valid_from` unchanged |
| `test_supersede_event_on_object_change` | Pre-seed one row for `(team, has_role, "no BE dev")`. Call state machine with `(team, has_role, "Sasha")` → returns `EVENT_SUPERSEDE`; collection now has 2 rows: old row has `invalid_at` set + `is_current=False`; new row has `valid_from=now`, `invalid_at=None`, `is_current=True`, `version=2` |
| `test_history_preserved_on_supersede` | After 3 sequential SUPERSEDEs on the same `(subject, predicate)`, the collection contains 4 rows total (1 INSERT + 3 SUPERSEDE chain); exactly 1 has `is_current=True`; the others all have `invalid_at` set; `version` values are `[1, 2, 3, 4]` |
| `test_supersede_crash_recovery_preserves_version` (Fix 2) | Simulate add-success + update-failure mid-SUPERSEDE (mock the second `collection.update` to raise). Next state-machine call on the same chain detects two `is_current=True` rows, picks the higher-version row as operative, demotes the lower-version row to `is_current=False`. Version stays monotonic; no reset to `version=1`. |
| `test_supersede_no_data_loss_under_crash` (Fix 2) | Mid-crash state has both rows present; lookup picks highest-version; old row's `object` value is still retrievable for historical query |
| `test_state_machine_invariants_hold_after_full_cycle` (Fix 10) | Fixture exercises INSERT, NOOP, SUPERSEDE in sequence; asserts all 5 state-machine invariants (Phase 2): at most one `is_current=True` per `chain_key`, every `is_current=False` row has `invalid_at IS NOT NULL`, every `is_current=True` row has `invalid_at IS NULL`, `version` is monotonic, zero deletes |
| `test_preflight_runs_before_find_prose_drift_import` (Fix 9) | Monkeypatch the `find_prose_drift` import to raise on access. `_preflight_global` with a missing `ANTHROPIC_API_KEY` returns `Err("anthropic_key_missing")` WITHOUT triggering the import — verified by absence of the raise |
| `test_cache_key_text_uses_preprocess_session_schema` (Pass-3 Fix 1) | Fixture is the literal output of `preprocess_session("user: hello\nassistant: hi\n")` (or an equivalent dict list with `{"role","text","timestamp"}` keys). Call `_extract_cache_key_text(messages)` → no `KeyError` raised; returned JSON includes the user-message `text` value; returned JSON contains neither the substring `"content"` nor the per-message timestamp |
| `test_doc_extract_writes_extraction_cache_with_matching_text_sha` (Pass-3 Fix 2) | Call `extract_facts_from_doc("Our team has no backend developer.", "docs/team.md")`. After the call, both `doc_facts` and `extraction_cache` rows exist in the per-project SQLite file. Assert `doc_facts.sha256 == extraction_cache.text_sha` for the row pair (the join invariant from Phase 2). Doc-side `sha256` is `sha256(text.encode("utf-8")).hexdigest()` over raw doc text — no canonicalization |
| `test_v1_never_writes_doc_rows` (Fix 4) | After exercising INSERT, NOOP, SUPERSEDE plus a full `find_prose_drift` pass over a fixture project with multiple doc-chunks, zero rows exist in `<project>-prose-facts` whose `source_id` references a doc relpath; all rows carry session-id provenance. (Replaces the prior `source_type == "session"` shape assertion.) |
| `test_no_deletes_on_state_machine` | Spy `collection.delete` → after a sequence of INSERT/NOOP/SUPERSEDE, `delete` is called zero times |
| `test_chain_key_format` | After INSERT for triple `(team, has_role, X)`, the stored metadata `chain_key == "team\|has_role"` (pipe-delimited subject and predicate) |
| `test_concurrent_chain_isolation` | INSERT `(team, has_role, X)` then INSERT `(team, headcount, Y)` → two separate chains; each has its own current row; `chain_key`s are distinct |
| `test_per_project_collection_isolation` | INSERT for project `A` then query `<B>-prose-facts` → zero rows of A's triples in B's collection |
| `test_chromadb_collection_name` | First state-machine call against project `vecs` creates `vecs-prose-facts` (verified via `db.get_collection`) |
| `test_voyage_embedder_used_for_facts` | `add_fact_with_state_machine` invocation issues one call to `get_voyage_client().embed(...)` with `model="voyage-3"` |
| `test_iterate_indexed_docs_reads_correct_metadata_key` | Fixture Chroma collection where chunk metadata has `file_path` key; `iterate_indexed_docs` yields `(text, file_path)` tuples reading from the `file_path` key specifically. Negative variant: a chunk with metadata key `path` (not `file_path`) is NOT yielded with that value (asserts no fallback). |
| `test_find_prose_drift_iterates_all_doc_chunks` | Fixture project with 3 doc-chunks; `find_prose_drift` invokes `extract_facts_from_doc` exactly 3 times (once per chunk) |
| `test_index_docs_does_not_write_facts` | Run `index_docs` against the fixture project → zero state-machine calls on the doc path; `<project>-prose-facts` row count is unchanged |
| `test_state_machine_runs_after_manifest_save_not_before` | Spy verifies the new per-file loop strictly follows `manifest.save()` in source order at `_index_session_files`'s success path |
| `test_state_machine_only_for_fully_succeeded_files` | Fixture with 2 files; file A fully succeeds, file B partially fails (mock `_track_embed_success` to return only A in `fully_succeeded`). State machine runs only for A's triples; never for B on this run. |
| `test_state_machine_skips_partial_file_runs_next_time` | Partial-fail file B from the previous test: on a second indexer run where B's chunks now all succeed, state machine runs for B exactly once. Across the two runs, B's INSERT events sum to the count of its triples (no duplicate INSERTs). |
| `test_state_machine_not_called_when_disabled` | Same fixture with `prose_drift_enabled=false` → zero state-machine calls; `<project>-prose-facts` not created |
| `test_state_machine_failure_does_not_abort_indexer` | `extract_facts` raises on file A → indexer completes; file B's extract + state machine still attempts; chunk count unchanged; manifest already saved; error logged via `_log` |
| `test_assistant_turns_excluded_from_extract` | Fixture session with 2 user + 3 assistant turns; `extract_facts` receives `messages` of length 2 (user-only); the 3 assistant turns are not present in the call payload |
| `test_be_dev_contradiction_surfaces` | End-to-end (mocked Anthropic) against `tests/fixtures/prose_drift/`: drift list contains exactly one entry with `(team, has_role, ...)`, both source labels, and `chat_history_versions >= 1` |
| `test_drift_entry_carries_session_id` | Pre-seed `<project>-prose-facts` with a session-sourced row where `source_id="be_dev_announce"`; `find_prose_drift` surfaces it as the drift entry's `chat.session_id == "be_dev_announce"` |
| `test_drift_entry_carries_chat_history_versions` | Pre-seed a chain where the current row has `version=3`; `find_prose_drift` surfaces `chat_history_versions=3` on the drift entry |
| `test_code_vecs_path_untouched` | AST-walk `indexer.index_code`, `chunkers`, `ast_chunker`, `bm25_index`, `searcher` → zero references to `anthropic` or `prose_drift` |
| `test_cli_prose_drift_no_drift_exit_0` | Mock returns empty list; CLI exits 0; stdout `no prose drift` |
| `test_cli_prose_drift_with_drift_exit_1` | Mock returns 2-entry list; CLI exits 1; two lines matching Phase 1 format |
| `test_cli_unknown_project_exit_2` (Pass-3 Fix 3) | bad `-p` → exit 2, stderr `unknown project`. Assert this is decided by `_preflight_project` returning `Err("project_unknown", ...)` (NOT by `_preflight_global`); `_preflight_global` returned `Ok` upstream |
| `test_cli_disabled_project_exit_2` (Pass-3 Fix 3) | known project, `prose_drift_enabled=false` → exit 2, stderr `prose drift not enabled`. Assert this is decided by `_preflight_project` returning `Err("prose_drift_disabled", ...)` (NOT by `_preflight_global`); `_preflight_global` returned `Ok` upstream |
| `test_cli_no_chat_sessions_exit_0` | enabled project with empty sessions → exit 0, stdout `no chat sessions` |
| `test_cli_anthropic_import_missing_exit_3` | Simulated `ImportError` on `anthropic` → exit 3, stderr `anthropic not installed` |
| `test_cli_anthropic_key_missing_exit_3` | Unset `ANTHROPIC_API_KEY` → exit 3, stderr `ANTHROPIC_API_KEY not set` |
| `test_mcp_prose_drift_returns_dict` | MCP variant returns dict matching CLI semantics |
| `test_mcp_prose_drift_none_scans_all_enabled` | `prose_drift(project=None)` → keys are exactly the projects with `prose_drift_enabled=true` |
| `test_mcp_project_none_value_shape` | With 2 enabled projects, outer dict has 2 keys; each value matches the single-project shape `{"drift", "facts_scanned", "facts_scanned_docs", "project"}` |
| `test_mcp_project_none_no_enabled_projects_returns_empty_dict` | All projects `prose_drift_enabled=false`, `project=None` → returns `{}` (exact: empty dict) |
| `test_mcp_project_none_global_preflight_fails_returns_error_dict` (Pass-3 Fix 3) | `ANTHROPIC_API_KEY` unset; call `mcp__vecs__prose_drift(project=None)` → returns single-error dict `{"error": "anthropic_key_missing"}` (NOT a per-project dict, NOT `{}`). Verifies the MCP `project=None` flow treats global-preflight failure as a top-level error rather than fanning out across projects |
| `test_mcp_project_none_skips_disabled_projects_silently` (Pass-3 Fix 3) | Three projects in config: two with `prose_drift_enabled=true`, one with `prose_drift_enabled=false`. `mcp__vecs__prose_drift(project=None)` returns a dict with exactly the 2 enabled project names as keys; the disabled project is silently absent (no `error` key, no raise). Asserts that `_preflight_project` is NOT consulted in the `project=None` flow — enabled-filter happens by `config.projects.values()` filter, not by preflight Err |
| `test_mcp_anthropic_missing_returns_error_dict` | Simulated `ImportError` → returns `{"error": "anthropic_unavailable", ...}`, does not raise |
| `test_cli_limit_flag_truncates` | Mock returns 100 entries, `--limit 10` → 10 lines printed, stderr `drift truncated: showing 10 of 100`, exit 1 |
| `test_limit_preserves_sort_order` | 100 mock entries shuffled, `--limit 10` → printed lines are the first 10 of the `(subject, predicate)`-sorted list |
| `test_extraction_model_constant` | `PROSE_EXTRACTION_MODEL == "claude-opus-4-7"` |
| `test_config_yaml_missing_field_loads` | YAML without `prose_drift_enabled` → `ProjectConfig.prose_drift_enabled is False` |
| `test_config_yaml_with_prose_drift_enabled_true_loads` | YAML with `prose_drift_enabled: true` under a project block → `ProjectConfig.prose_drift_enabled is True` |
| `test_anthropic_version_pinned` | `pyproject.toml` declares `anthropic==<X>` (exact-pin form, not `>=` or `~=`) |
| `test_facts_scanned_equals_current_row_count` | `facts_scanned` returned by `find_prose_drift` equals the count of rows in `<project>-prose-facts` with `is_current=True` at scan time. (Fix 4: no `source_type` filter; v1 only writes session-sourced rows by construction.) |
| `test_facts_scanned_docs_equals_chroma_docs_cardinality` | `facts_scanned_docs` returned by `find_prose_drift` equals `len(list(iterate_indexed_docs(project)))` at scan time |
| `test_facts_scanned_is_int` | The returned `facts_scanned` value `isinstance(int)` |
| `test_integration_real_anthropic` (gated by `VECS_TEST_REAL_LLM=1`) | Real Anthropic Opus 4.7 + tmp_path Chroma + BE-dev fixture → INSERT then SUPERSEDE fires, drift surfaces. Skipped by default. |

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
| Implementation, multi-file (state machine across `indexer.py` + `prose_drift.py` + `cli.py` + `mcp_server.py` + `config.py`) | `builder-large` | `general-purpose` (worktree) |
| Gap-finding | `critical-sinker` | `general-purpose` + `.claude/prompts/critical-sinker.md` |
| Diff review | `reviewer` | `caveman:cavecrew-reviewer` |

Full-feature implementation crosses 5 files → `builder-large` (worktree isolation per profile roster). Phase 7 dry-run subtask is sized for `builder-small`.

## Phase 7 — Dry-run

`dryrun_selection: smallest-real-subtask` (per profile).

**Subtask.** Implement `add_fact_with_state_machine(triple, source_id, project) -> str` as the smallest real subtask plus one end-to-end test that exercises INSERT, then NOOP, then SUPERSEDE in sequence on the BE-dev fixture. The test uses a real Anthropic call (gated by `ANTHROPIC_API_KEY`) to extract triples from the BE-dev session fixture, then feeds them through the state machine against a `tmp_path`-pinned Chroma instance. New module `src/vecs/prose_drift.py` with the state-machine function plus `extract_facts` plus the verdict cache. One new test file. ~180-220 lines total. No CLI, no MCP, no doc-fact extraction, no drift comparison, no indexer wiring, no `pyproject.toml` edit other than the `anthropic` pin needed for the test to import (which lands in the dry-run subtask, not later).

**Dry-run acceptance** (`dryrun_acceptance: "inline"`):
- [ ] `extract_facts(<3 BE-dev user-message records>, project="vecs-test")` returns `list[Triple]` of length ≥ 1.
- [ ] The Anthropic SDK call uses `model="claude-opus-4-7"` and does NOT pass `temperature`. Asserted via monkeypatching `anthropic.Anthropic.messages.create` (recording args dict to a test-local list).
- [ ] First call to `add_fact_with_state_machine(triple, source_id="be_dev_announce", project="vecs-test")` returns `EVENT_INSERT`. Chroma collection `vecs-test-prose-facts` exists under `tmp_path` (NOT under `~/.vecs/chromadb` — the dry-run pins the Chroma path to `tmp_path`) with exactly one row.
- [ ] Second call with the same triple returns `EVENT_NOOP`. Row count unchanged.
- [ ] Third call with the same `(subject, predicate)` but different `object` returns `EVENT_SUPERSEDE`. Collection now has 2 rows; the old row has `is_current=False` and `invalid_at != None`; the new row has `is_current=True`, `invalid_at=None`, `version=2`.
- [ ] **Chroma bool-where verification** (Fix 1 / BLOCKER 1). After the dry-run state-machine sequence (which seeds two rows for the chain: one with `is_current=True`, one with `is_current=False`), assert:
  - `collection.get(where={"is_current": True})["ids"]` returns exactly 1 id.
  - `collection.get(where={"is_current": False})["ids"]` returns exactly 1 id.
  - If Chroma rejects the bool literal (some versions normalize bool to int 0/1), the dry-run FAILS and the row schema is revised pre-implementation: rename the field to `is_current_int: 0|1` (with `0 = False`, `1 = True`) and update the where-clause accordingly. Schema field name + values are pinned post-dry-run.
- [ ] **Chroma multi-key where verification.** The state-machine lookup uses `collection.get(where={"chain_key": "<key>", "is_current": True})` (two keys). Chroma versions ≥0.4.x require multi-key filters to be wrapped in `{"$and": [...]}`; older versions accept the flat dict form. Assert the lookup returns exactly 1 row against the seeded chain using BOTH forms: (a) flat `{"chain_key": k, "is_current": True}`, (b) `{"$and": [{"chain_key": k}, {"is_current": True}]}`. Whichever form Chroma accepts is pinned as the canonical lookup syntax in `src/vecs/prose_drift.py` post-dry-run; the other is removed. If Chroma rejects both forms in the pinned version, the dry-run FAILS.
- [ ] Voyage `voyage-3` model id appears in the embedding call, asserted via monkeypatching the Voyage client's `.embed(...)` (recording args dict to a test-local list).
- [ ] `extraction_cache` SQLite table is created under the dry-run cache dir (also `tmp_path`-pinned); the table includes the `prompt_version` column and the PK is `(text_sha, model, prompt_version)`. A repeated `extract_facts(...)` call with identical messages issues zero new Anthropic calls (cache hit).
- [ ] `tests/test_prose_drift.py` exists and the dry-run test passes under `uv run pytest -q` when `ANTHROPIC_API_KEY` is set; skips with a clear message when unset.
- [ ] No edits to `indexer.py`, `cli.py`, `mcp_server.py`, `config.py`. Edits to `pyproject.toml` are limited to adding the pinned `anthropic==<X>` dependency line.

**Roles.** `architect` (this plan), `builder-small` (≤2 files; the new module + the new test file), `critical-sinker` (gap-find on state-machine boundaries + extraction cache + temperature constraint), `reviewer` (verdict).

**Pass criteria** (per profile `dryrun_pass_criteria`): `[pipeline-pass, review-loop-satisfied]`.

**Expected learning.** Validates four boundaries in one subtask: (1) Anthropic Opus 4.7 auth + the no-temperature constraint + cost, (2) the INSERT/NOOP/SUPERSEDE state machine end-to-end against a real Chroma write, (3) the verdict cache hit/miss path, (4) the Voyage embed-call signature against the new collection. If any of these fails, rollback (`branch-drop`) before further work.

**Sequencing.** Phase 7 dry-run runs FIRST on a feature branch. Full Phase 5 test matrix is authored AFTER dry-run validates the four boundaries above. Indexer wire-in and CLI/MCP surface only follow once dry-run hits all inline acceptance items and reviewer verdict=ship.

**Abort policy** (profile: `branch-drop`). Dry-run failure → drop branch, no partial wiring on trunk. Rollback plan below covers the cleanup.

## Phase 8 — Retrospective

`retro_template: docs/templates/retro.md` (profile).
`gap_log_destination: docs/features/prose-staleness-detector/gaps.md`. Schema per `gaps.md` row: `pass | finding-id | severity | summary | decision | follow-up-link`.
`feedback_targets: [0, 2]` (profile default — Bootstrap and Context phases).
`feedback_artifact: context-doc-edit`.
`feedback_apply_owner: "self"`.

**Pre-known gap candidates** (populated into `gaps.md` after dry-run, not at design time):

1. **First Anthropic API call invariant.** Profile Phase 0 `bootstrap_done_check` does not currently verify `ANTHROPIC_API_KEY`. Consider extending the runbook (`README.md` Install section).
2. **First `anthropic` dependency.** Profile Phase 0 `pipeline_setup_runbook` may need an addendum for the Anthropic install path.
3. **Partial SUPERSEDE recovery.** RESOLVED in this design pass via Fix 2 (reordered crash-safe write + in-line repair). The state machine writes the new row BEFORE flipping the old row, so a mid-SUPERSEDE crash leaves the chain with two `is_current=True` rows rather than zero; the next lookup detects this and repairs by selecting the highest-version row as operative and demoting the lower-version row. Version chain is preserved across crashes. No v2 work required.
4. **Chroma `where`-clause for `None` values.** Chroma does not support `{"invalid_at": None}` filters natively; v1 sidesteps with the `is_current: bool` companion field. If Chroma's filter API gains native NULL support, the companion field can be retired.
5. **Verdict cache size growth.** `extraction_cache` rows accumulate forever. At vecs scale (single dev, ~1500 calls/mo) this is sub-MB. At multi-tenant scale it could grow; future `vecs prose-drift compact --older-than <days>` deferred to v2.
6. **No row-deletion compactor.** `<project>-prose-facts` grows unboundedly with SUPERSEDE chains. At vecs scale this is negligible (~kB/yr); larger deployments will want a compactor. Deferred.
7. **Cost ceiling not enforced.** Opus 4.7 at ~$0.80/mo at vecs scale is well within tolerance but no hard ceiling exists. Consider env-var ceiling (`VECS_PROSE_DRIFT_MAX_CALLS_PER_DAY`) as future work.
8. **Doc-fact cache invalidation under doc renames.** The `doc_facts` cache keys on `(source_relpath, sha256(text))` — a rename with identical text is a cache miss (harmless: re-extracts) but a rename followed by content change is correctly invalidated. Tracking only.
9. **Concurrent `vecs prose-drift` invocations on the same project.** Single-writer SQLite cache assumption (WAL mode + IMMEDIATE write transactions) is sufficient for solo/POC scale. Two simultaneous runs against the same project's `.db` may surface lock contention; out of scope for v1.
10. **Assistant-turn re-enablement.** v1 hard-excludes `role != "user"` turns from extraction because assistant output is hallucination-prone. A future v2 may re-enable assistant turns gated by a pre-classifier. Defer; tracking-only.
11. **Time-travel CLI flag deferred.** `--as-of <ISO-date>` is excluded from the v1 CLI surface entirely (Fix 6/MAJOR 5); argparse's default unknown-flag handling (exit 2 usage error) is sufficient. The full implementation is parked in `docs/features/prose-staleness-detector/v2-roadmap.md`.

Each item lands in `gaps.md` only after the dry-run + full implementation + reviewer-ship cycle completes. Items 1+2 are Bootstrap-targeted; items 3-11 are tracking-only.

---

## Roll-back plan

If the Phase 7 dry-run fails, or if the full state-machine wiring surfaces an unrecoverable issue:

1. **Branch drop** (profile `abort_policy: branch-drop`). Delete the feature branch; no partial state lands on trunk.
2. **Revert config-loader change** if it leaked to trunk. The `prose_drift_enabled` field is opt-in default-False; deleting it is a no-op for any project not declaring it.
3. **Drop `anthropic`** from `pyproject.toml`. `uv lock` + `uv sync` removes it. No vendored Anthropic code exists.
4. **Drop the new ChromaDB collection** if any test or dry-run created `<project>-prose-facts`. Deletion is `chroma_client.delete_collection(name="<project>-prose-facts")`; does not touch `-code`, `-sessions`, or `-docs`. Note: this is the ONLY context in which `delete_collection` against `-prose-facts` is acceptable; the runtime state machine never deletes.
5. **Drop the verdict cache directory** at `~/.vecs/prose_drift_cache/`. Pure side-cache; deletion is `rm -rf`.
6. **Revert the `staleness_check` list promotion** in `src/vecs/CLAUDE.md` back to scalar `commit-sha-tag`.
7. **Escalate to architectural retry.** Locked-decision backup is Graphiti (full graph DB) — strictly heavier; only consider if V+ surfaces fundamental correctness issues, not cost or scale issues. Reopening the locked decision requires a new design doc; this V+ design is sealed once dry-run succeeds.

The rollback is cheaper than the prior Mem0 design by construction: no Mem0 dependency to drop, fewer invariant changes, and the code-vecs path is never touched, so reverting prose-vecs cannot break code search.
