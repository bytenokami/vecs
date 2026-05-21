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

## Pre-known gap candidates (move to runtime gaps after Phase 7 dry-run)

1. **First Anthropic API call invariant.** Profile Phase 0 `bootstrap_done_check` does not currently verify `ANTHROPIC_API_KEY`. Candidate runbook addendum.
2. **First `anthropic` dependency.** Profile Phase 0 `pipeline_setup_runbook` may need an addendum for the Anthropic install path.
3. **Chroma `where`-clause for `None` values.** v1 sidesteps with `is_current: bool` companion field. If Chroma's filter API gains native NULL support, the companion field can be retired (v2 opportunity).
4. **Chroma bool-vs-int normalization.** Field name `is_current` may need to become `is_current_int: 0|1` per Phase 7 dry-run. Decision pinned post-dry-run.
5. **Chroma multi-key where syntax.** Flat dict vs `$and` wrapping. Canonical form pinned post-dry-run.

## Convergence summary

- Pass 1: 20 findings (4 BLOCKER, 9 MAJOR, 7 MINOR). All BLOCKERs applied in design pass 1.
- Pass 2: 4 findings (1 BLOCKER, 2 MAJOR, 1 MINOR). All applied in design pass 2.
- Pass 3: 3 findings (0 BLOCKER, 2 MAJOR, 1 MINOR). All applied in design pass 3-4.
- Pass 4: reviewer verdict = ship.

Total: 27 findings tracked; 26 `applied-in-design`; 1 `parked-v2`. Zero `wontfix`.
