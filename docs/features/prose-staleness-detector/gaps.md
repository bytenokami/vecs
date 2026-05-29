# Gap log — prose-staleness-detector v1 (V+)

Schema (per profile `feedback_artifact` row contract): `pass | finding-id | severity | summary | decision | follow-up-link`.

`decision` values: `applied-in-design` (fix landed in this design pass), `parked-v2` (deferred to roadmap), `wontfix` (intentionally not addressing), `dry-run-pin` (resolved at Phase 7 dry-run, not at design time).

| pass | finding-id | severity | summary | decision | follow-up-link |
|---|---|---|---|---|---|
| 1 | sinker-1.1 | BLOCKER | Chroma `where`-clause cannot match `invalid_at IS NULL` natively | applied-in-design | `is_current: bool` companion field + Phase 7 bool-where dry-run pin (Fix 1) |
| 1 | sinker-1.2 | BLOCKER | SUPERSEDE write order not crash-safe (naive: flip-then-add can drop chain) | applied-in-design | Reordered write (add-new-first / flip-old-second) + transient-state repair branch (Fix 2) |
| 1 | sinker-1.3 | BLOCKER | Cache key non-canonical: dict-key order + per-message timestamps inflate cache | applied-in-design | `_extract_cache_key_text` canonical JSON serialization with `sort_keys=True` over `{role, text}` only (Fix 3) |
| 1 | sinker-1.4 | BLOCKER | Schema `source_type` field never read by query path → dead field | applied-in-design | Field REMOVED from row schema for v1; `add_fact_with_state_machine` signature drops `source_type` arg (Fix 4) |
| 1 | reviewer-1.1 | MAJOR | `--as-of` time-travel flag adds CLI surface without operator demand | parked-v2 | `docs/features/prose-staleness-detector/v2-roadmap.md` (Fix 5) |
| 1 | sinker-1.5 | MAJOR | Empty triple list extraction path undefined (error vs. positive cache) | applied-in-design | Positive-cached as `triples_json="[]"`; per-file log `triples=0`; no special error branch (Fix 6) |
| 1 | sinker-1.6 | MAJOR | Cache does not invalidate on prompt template edits → stale extractions | applied-in-design | `EXTRACTION_PROMPT_VERSION` module constant; PK includes `prompt_version` (Fix 7) |
| 1 | sinker-1.7 | MAJOR | Anthropic missing-import / missing-key crashes entry point | applied-in-design | Lazy import + preflight error codes (`anthropic_unavailable`, `anthropic_key_missing`) (Fix 9) |
| 1 | sinker-1.8 | MAJOR | State-machine invariants not enumerated → silent corruption possible | applied-in-design | Explicit invariant list (1)-(5) in design + acceptance (Fix 10) |
| 1 | sinker-1.9 | MAJOR | Doc-side `source_relpath` lookup ambiguous (which Chroma metadata key?) | applied-in-design | Pinned to `file_path` metadata key (cite `src/vecs/doc_chunker.py:103`) |
| 1 | sinker-1.10 | MAJOR | Bi-temporal hand-wave: which column is "valid time"? | applied-in-design | `valid_from`/`invalid_at` (transaction time) defined; rejected separate "valid time" axis for v1 |
| 1 | sinker-1.11 | MINOR | INSERT/NOOP/SUPERSEDE event names not standardized | applied-in-design | Module constants `EVENT_INSERT`/`EVENT_NOOP`/`EVENT_SUPERSEDE` |
| 1 | sinker-1.12 | MINOR | Indexer wire-in insertion point ambiguous (success-path vs early-return `manifest.save`) | applied-in-design | Pinned to success-path `manifest.save()` at `src/vecs/indexer.py:964`; `:951` early-return untouched |
| 1 | sinker-1.13 | MINOR | Per-message stash strategy undefined (re-parse vs cache) | applied-in-design | `file_messages: dict[Path, list[dict]]` stash during first loop |
| 1 | sinker-1.14 | MINOR | Doc-extraction collision with session-side write path | applied-in-design | Doc triples NEVER written to `<project>-prose-facts` in v1; cached only in SQLite `doc_facts` |
| 1 | sinker-1.15 | MINOR | Role filter undefined → assistant hallucinations pollute drift baseline | applied-in-design | Filter to `role == "user"` at preprocessed-message level |
| 1 | sinker-1.16 | MINOR | Code-vecs vs prose-vecs path bleed possible via shared `chunkers.py` import | applied-in-design | Grep test asserts `anthropic`/`prose_drift` not imported by code-vecs path |
| 1 | sinker-1.17 | MINOR | Cache DDL initialization point undefined | applied-in-design | First `extract_facts` call creates `.db` + tables + WAL mode |
| 1 | sinker-1.18 | MINOR | `facts_scanned` semantics ambiguous (all rows? current-only?) | applied-in-design | Pinned: count of `is_current=True` rows at scan time |
| 1 | sinker-1.19 | MINOR | `--limit N` truncation order undefined | applied-in-design | `(subject, predicate)` sort; first N entries; stderr `drift truncated: showing N of M` |
| 1 | reviewer-1.2 | MINOR | YAML config key drift risk (`prose_drift_enabled` not validated) | applied-in-design | `load_config` parses absent as `False`; explicit acceptance bullet |
| 2 | sinker-2.1 | BLOCKER | MCP `project=None` preflight semantics undefined (global vs per-project errors collide) | applied-in-design | Preflight split into `_preflight_global` + `_preflight_project`; flow contracts per entry-point pinned (Pass-3 Fix 3) |
| 2 | sinker-2.2 | MAJOR | Cache helper reads `m["content"]` but `preprocess_session` emits `m["text"]` → runtime KeyError | applied-in-design | Helper switched to `m["role"]`+`m["text"]`; schema cite added (Pass-3 Fix 1) |
| 2 | reviewer-2.1 | MAJOR | `doc_facts` ↔ `extraction_cache` join column not declared invariant | applied-in-design | `doc_facts.sha256 == extraction_cache.text_sha` invariant documented (Pass-3 Fix 2) |
| 2 | reviewer-2.2 | MINOR | CLI drift-line `<docs-dir-relpath>` ambiguous when multiple projects share filename | applied-in-design | Prefixed with `<project>/` in drift output (Pass-3 Fix 4) |
| 3 | sinker-3.1 | MAJOR | Wire-in spec passes `source_type="session"` after Fix 4 dropped the field | applied-in-design | Pass-4 edit: arg removed from line 70 of design |
| 3 | sinker-3.2 | MAJOR | Phase 7 dry-run subtask signature retains `source_type` arg | applied-in-design | Pass-4 edit: arg removed from line 499 of design |
| 3 | reviewer-3.1 | MINOR | Test description references `role`+`content` not `role`+`text` | applied-in-design | Pass-4 edit: description corrected (line 417) |
| 3 | sinker-3.3 | MAJOR | Multi-key Chroma `where={"chain_key": ..., "is_current": True}` not dry-run-validated | applied-in-design | Pass-4 edit: new dry-run bullet asserts both flat + `$and`-wrapped forms; canonical form pinned post-dry-run |
| dryrun-1 | sinker-d1.1 | MAJOR | `INVALID_AT_NONE_SENTINEL = 0` deviated from doc `invalid_at: int \| None` (Chroma metadata rejects `None`) | applied-in-design | design line 57 + acceptance row-schema bullet amended to pin `invalid_at: int` with `0` sentinel |
| dryrun-1 | sinker-d1.2 | MAJOR | SUPERSEDE write-order (add-FIRST / flip-SECOND) not test-asserted; future refactor could silently reverse it | applied-in-impl | `test_supersede_write_order_add_first_then_flip` spies on `add`+`update`, monkeypatches `update` to raise, asserts call order = `[add, update]`, verifies repair branch returns NOOP on re-run |
| dryrun-1 | sinker-d1.3 | MAJOR | `test_no_deletes_against_prose_facts` path-relative — fails when pytest cwd ≠ repo root | applied-in-impl | Anchored with `Path(__file__).resolve().parents[1]` |
| dryrun-1 | sinker-d1.4 | MAJOR | `.delete(` grep had tautological bypass clause (`or "# delete" in src`) — any future comment containing `# delete` would whitewash a real delete | applied-in-impl | Dropped tolerance clause; assertion now strict |
| dryrun-1 | sinker-d1.5 | MAJOR | Repair-branch NOOP path (all current rows have same object) was unasserted by any test | applied-in-impl | `test_repair_branch_noop_when_operative_matches_incoming` hand-seeds 2 `is_current=True` rows for the same chain_key with same object; asserts NOOP and demotion of lower-version row |
| dryrun-1 | reviewer-d1.1 | MAJOR | Acceptance line 68 requires "unit test asserts pin form" for `anthropic==X.Y.Z`; no such test existed | applied-in-impl | `test_pyproject_pins_anthropic_exact_version` reads pyproject.toml, asserts exactly one `"anthropic==…"` line |
| dryrun-1 | sinker-d1.6 | MINOR | Multi-key-where test didn't pin canonical form; silently flipped to `$and` if flat dict raised in future Chroma | applied-in-impl | Renamed to `test_chroma_multi_key_where_canonical_form_pinned`; asserts `$and` form succeeds unconditionally; flat-dict form observed under XOR gate (raise-XOR-match) |
| wire-in | sinker-w1.1 | MINOR | SQLite cache DDL omits the `extracted_at` columns the design DDL (lines 83/92) lists; no v1 code path reads them | parked-v2 | Deferred to the parked v2 cache-compactor; no functional impact in v1 (no reader). `docs/features/prose-staleness-detector/v2-roadmap.md` |

## Pre-known gap candidates (move to runtime gaps after Phase 7 dry-run)

1. **First Anthropic API call invariant.** Profile Phase 0 `bootstrap_done_check` does not currently verify `ANTHROPIC_API_KEY`. Candidate runbook addendum. **STATUS post-dry-run: confirmed gap** — operator needed to load key from Keychain manually; `README.md` Install section addendum still TODO.
2. **First `anthropic` dependency.** Profile Phase 0 `pipeline_setup_runbook` may need an addendum for the Anthropic install path. **STATUS post-dry-run: addressed indirectly** — `uv sync` picked up `anthropic==0.103.1` pin automatically; no manual step needed.
3. **Chroma `where`-clause for `None` values.** v1 sidesteps with `is_current: bool` companion field. **STATUS post-dry-run: pinned permanently** — `invalid_at: int` with `0` sentinel; documented in design + acceptance.
4. **Chroma bool-vs-int normalization.** Field name `is_current` may need to become `is_current_int: 0|1` per Phase 7 dry-run. **STATUS post-dry-run: pinned to `is_current: bool`** — Chroma 1.0.x accepts bool literals natively; `test_chroma_bool_where_verification` passes.
5. **Chroma multi-key where syntax.** Flat dict vs `$and` wrapping. **STATUS post-dry-run: canonical form = `$and`** — production code uses `{"$and": [{"chain_key": ...}, {"is_current": True}]}`. Flat dict also works on Chroma 1.0.x but `$and` is the pinned form.

## Convergence summary

- Pass 1: 20 findings (4 BLOCKER, 9 MAJOR, 7 MINOR). All BLOCKERs applied in design pass 1.
- Pass 2: 4 findings (1 BLOCKER, 2 MAJOR, 1 MINOR). All applied in design pass 2.
- Pass 3: 3 findings (0 BLOCKER, 2 MAJOR, 1 MINOR). All applied in design pass 3-4.
- Pass 4: reviewer verdict = ship.
- Dry-run pass 1: 7 findings (0 BLOCKER, 6 MAJOR, 1 MINOR). All applied. 1 doc-deviation, 5 test gaps, 1 path-bug.
- Dry-run pass 2: reviewer verdict = ship.

Total: 34 findings tracked; 33 `applied-in-design` or `applied-in-impl`; 1 `parked-v2`. Zero `wontfix`.
