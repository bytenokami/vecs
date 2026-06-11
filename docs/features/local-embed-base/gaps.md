Authored by Claude

# local-embed-base — Phase-4 gap log

Review: 3 lenses (deep-correctness, test-quality, docs-consistency), 14 findings, verdict fix-then-ship.
Raw findings: `phase4-findings.json` (this dir). 13 fixed in-branch; status below.

## Fixed

- **Blocked mismatch-clear could falsely advance markers** (deep MED, the big one): a model mismatch whose sources were unenumerable (transiently missing root / empty scan) cleared nothing, yet `_remodel_record` advanced the marker — interlock permanently disarmed over old-model vectors. Now `_remodel_clear` reports `docs_blocked`/`code_blocked` and `_remodel_record` skips blocked collections (the docs half of this hole pre-existed on master; fixed for both). Test-pinned.
- **Per-call Qwen provider instancing** (deep MED): `get_provider` now memoizes per name (`_provider_cache`), so the multi-GB SentenceTransformer loads once per process, not per search.
- **Config-flip-alone guard** (deep MED): `embed_provider: qwen-local` with voyage model constants now fails loud at provider construction with the flip story (config + model-constants release + reindex, design.md L3). Explicit `name=` requests (A/B arms) bypass the guard. Module docstring corrected.
- **Backfill warning** (deep MED, partial — see Open): backfilling an unmarked store with a non-voyage model id logs a loud warning (the suspicious pre-L1-store-meets-flip-build case).
- Integration test now pins run_index-path backfill (code key kept + marker recorded) (test MED).
- Lazy-import error path test-pinned (`vecs[local]` message) (test MED).
- Timeout-test docstring corrected; string-based rate-limit fallback pinned separately for typed-error-less providers (test MED).
- run_arm/ab_report edge cases pinned (empty set, empty arms, multi-class) (test LOW).
- `_cached_embed` provider/model-id namespace assumption documented in docstring (test LOW).
- CLAUDE.md: interlock attributed to `search_collections()`; searcher + eval_harness module rows updated (docs MED x3).
- `_collection_markers` docstring updated for code markers (docs MED).
- evalsets changelog records freeze SHA `fe0f5cb` (docs LOW).

## Accepted / deferred

- **Lazy voyage client construction** (deep LOW, accepted): client now resolves at first embed, not operation entry. Keyless `search()` degrades to BM25-only instead of raising at entry; keyless `run_index` runs (correctness-preserving) sweeps before failing at the first embed. Arguably an improvement; noted here per review.
- **Backfill precondition for the flip** (deep MED, remainder): the L1.4 backfill assumes the unmarked store was embedded under the era's voyage constants. Enforced precondition deferred to the **L3 flip runbook** (local-embed-ab): "verify `collection_models` markers exist (one post-L1 reindex ran) before installing a flip build". The runtime warning above is the safety net.
- **Model ids stay constants** (deep MED, scoped choice): design.md L3's "set embed_provider + model ids in config.yaml" is implemented as config flip + constants release, guarded. Making model ids config-resolvable is a possible L3-time enhancement if the runbook wants config-only flips.
