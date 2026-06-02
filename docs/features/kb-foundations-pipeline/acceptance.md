# Acceptance — Increment 1-pipeline (F + B + C)

Format: checklist. Pass threshold: all-pass. Parent: `docs/features/kb-foundations/design.md`; program §6.
Runner: `uv run pytest -q`. Adapter: `scripts/check_acceptance.py kb-foundations-pipeline`.
Sub-features share one reindex. Build order within: C → B → F.

## C — version_id + content-hash embedding cache

- [ ] Each stored chunk carries a `version_id` (git SHA for code, mtime/revision for docs, session/run id for sessions).
- [ ] An embedding cache keyed by chunk content-hash exists.
- [ ] Cache test: in a file with ≥2 chunks where only one chunk's content changed, re-indexing makes a Voyage call for exactly the changed chunk; the unchanged chunks are cache hits.
- [ ] Invariant test: in that same mixed-chunk file, cache-hit chunk ids are counted toward `succeeded_ids`, the file reaches `succeeded == expected`, and is marked indexed in one pass (no perpetual reprocessing).

## B — voyage-4 re-embed for docs/sessions

> Model decision: the re-embed target is **voyage-4**, not the originally-specced voyage-3.5. voyage-4 is the current Voyage frontier (verified live 2026-06-02) and shares voyage-3's 1024-dim default, so it fits the existing Chroma collections in place with no recreate. All "voyage-3.5" references below mean voyage-4.

- [ ] voyage-4 dim recorded vs voyage-3 (equal-dim is necessary, not sufficient).
- [ ] docs + sessions are RE-EMBEDDED under voyage-4 via the C cache (not a model-constant flip against stored voyage-3 vectors). Code stays voyage-code-3.
- [ ] Quality check: a set of known query→expected-source pairs returns the expected source post-re-embed (not merely non-empty results).
- [ ] The facts embedding model is pinned/stamped so swapping `SESSIONS_MODEL` cannot strand facts (note `_voyage_embed` shares it; facts are empty until Inc 2).

## F — .md reroute (sweep + multi-source docs)

- [ ] `.md` removed from every `code_dirs` extension list in `~/.vecs/config.yaml`.
- [ ] An explicit sweep deletes `.md`-sourced chunks from each `-code` collection AND its BM25 sidecar; a test asserts the sweep ran and `-code` contains zero `.md`-sourced chunks afterward.
- [ ] `ProjectConfig` has `docs_dirs: list[Path]`; an existing single `docs_dir` is coerced in with no breakage (migration test).
- [ ] `index_docs` accepts a per-file base dir and emits a source-root-qualified `rel_path`; collision test — two distinct source roots each containing `README.md` both survive a reindex (neither deletes the other).
- [ ] In-repo `.md` under `code_dirs` is routed to the project `-docs` collection, heading-chunked; bloomly, eric, and livly each have a `-docs` collection containing their in-repo `.md`.
- [ ] No `.md` content lost: per project, count of `.md` files under code_dirs (∪ docs_dirs) == `.md` source files tracked in the docs manifest.

## Global (this sub-feature)

- [ ] `uv run pytest -q` green; new/updated tests in `test_config.py`, `test_indexer.py`, `test_clients.py`.
- [ ] `src/vecs/CLAUDE.md` updated for touched modules.
- [ ] Phase 7 dry-run passed (docs_dirs coercion). Phase 4 review verdict = approve.
