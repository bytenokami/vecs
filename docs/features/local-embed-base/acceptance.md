Authored by Claude

# local-embed-base — acceptance

Design: `docs/features/local-embed/design.md` §L1. Plan: `plan.md` (this dir).

- [x] Chroma telemetry off at all 3 PersistentClient sites (test-pinned, `tests/test_telemetry.py`)
- [x] `embed_provider` config field round-trips load->save (auto-configure cannot strip it; `test_save_after_load_preserves_provider`)
- [x] EmbedProvider seam: indexer, searcher, prose_drift all route through it; no direct `voyageai` use outside `clients.py`/`embed_provider.py` (grep-guard clean)
- [x] QwenLocalProvider: query-prompt asymmetry, first-batch dim assertion, lazy import with actionable error, `vecs[local]` extra (sentence-transformers + torch)
- [x] Code-collection markers: unmarked+non-empty => backfill (NO clear); mismatch => clear scoped to `_code_sources`; `_remodel_record` marks code+docs; searcher drops a marker-mismatched code collection from the vector path (test-pinned)
- [x] `search_collections()` extracted; `search()` delegates with production values; existing search tests pass unchanged
- [x] Eval: YAML golden-set loader, recall@5/10 + nDCG@10 + MRR, `run_arm` + `ab_report` (overall + per class, aggregates only) + paired bootstrap CI
- [x] `evalsets/vecs.yaml` (46 cases) + README with schema, livly-never-in-repo rule, authoring protocol, freeze rule
- [x] Full suite green (`uv run pytest -q`, 422 passed); `src/vecs/CLAUDE.md` updated (provider seam, code markers/backfill, telemetry, config round-trip invariants)
- [ ] Phase-4 adversarial review run on the diff; findings fixed or logged in `gaps.md`
