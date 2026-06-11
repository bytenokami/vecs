Authored by Claude

# Increment 1.7 — local-embed: conditional Voyage→local-model swap + program reshape

**Parent program:** `docs/vecs-kb-curation-design-2026-06.md` (v4; this doc also carries the program-reshape delta that its v5 revision must absorb — see §Program reshape).
**Workflow profile:** `docs/workflow-vecs-profile-v0.1.md`.
**Research basis:** `docs/research/local-embedding-models-2026-06-11.md` (verified deep-research, 2026-06-11).
**Status:** spec (owner-approved direction 2026-06-11; thresholds in §L3 are a proposal pending owner sign-off at spec review). Adversarially reviewed 2026-06-11 (3 lenses); this revision incorporates all findings.

**Acceptance packaging (decompose rule):** two independently-gated sub-features, each with its own `acceptance.md` and Phase-4 review:

| Sub-feature | Dir | Deliverables |
|---|---|---|
| local-embed-base | `docs/features/local-embed-base/` | L1 (golden sets + A/B runner, provider abstraction, model-id discipline, code-collection markers, telemetry off) |
| local-embed-ab | `docs/features/local-embed-ab/` | L2 (perf spike + shadow A/B) and L3 (decision gate + flip or fallback) |

## Driver

Work (livly) source code currently leaves the work mac on every index/search: chunks go to the Voyage API for embedding. That is the one unauthorized outbound data path — the Anthropic API is explicitly allowed at work, so Inc 2 extraction and Inc 7's Contextual-Retrieval blurbs are unaffected. Inc 7's *reranker* half IS affected: v4 specs the hosted voyage rerank-2.5-lite, which under a swap would re-open the path (see §Program reshape). A second, minor leak: ChromaDB's PostHog telemetry is on (anonymized, but unmanaged).

**The swap is NOT decided.** Owner constraint: if local quality degrades a lot, the fallback is to request Voyage usage approval at work, using our own A/B numbers as evidence. So this increment is decision-gated: build the no-regret base, measure non-destructively, then gate.

## Candidate (from research)

- **Primary arm:** Qwen3-Embedding-4B, MRL-truncated to 1024 dims (`truncate_dim`, sentence-transformers — the verified MRL path), one model for both code and docs. Apache 2.0.
- **Cheap-tier arm:** Qwen3-Embedding-0.6B at native 1024 dims (MTEB-Code 75.41, above prior proprietary SOTA Gemini-Embedding 74.66). Becomes the swap target if 4B fails feasibility but passes quality.
- **Rescue arm (Inc 7 material, measured here only):** Qwen3-Reranker-4B over the qwen-4B arm (+5.8 pts MTEB-Code; the 0.6B reranker actively hurts code — never use it).
- Two honest gaps the research could not close, which L2 exists to close locally: **no Qwen-vs-Voyage head-to-head exists anywhere**, and **zero verified Apple Silicon throughput/RAM data**.

## Current state (verified 2026-06-11)

| Fact | Where |
|---|---|
| Models: `CODE_MODEL=voyage-code-3`, `DOCS_MODEL=voyage-4`, `FACTS_MODEL=voyage-4`; `EMBED_DIMS` all 1024, documented as *descriptive* ("we never send an output_dimension override") | `config.py:15-38` |
| No provider abstraction; contract = `vo.embed(texts, model=…, input_type="document"\|"query") → .embeddings + .usage.total_tokens` | `clients.py:12-17` (singleton), `indexer.py:451`, `searcher.py:31`, `prose_drift.py` `_voyage_embed` (all three call sites — verified exhaustive) |
| Retry tuple is Voyage-specific (`voyageai.error.{Timeout,APIConnectionError,…}`) | `indexer.py:456-464` |
| Embed cache keyed `(model, sha256(text))`; per-collection model markers in `collection_models` | `embed_cache.py` |
| Model-flip trigger (`_model_changed`/`_remodel_clear`/`_remodel_record`) covers **docs collections only**; `_model_changed` treats a never-recorded collection as changed (None != model → re-embed) | `indexer.py:1541-1623` |
| Searcher interlock is already fail-closed for a present-and-mismatched marker (collection dropped from vector path, BM25 carries it); only a None marker / marker-read error is fail-open. Code collections are never marked, so a code-model swap today would silently mix vector spaces — the hole is **indexer-side only** | `searcher.py:41-49, 185-204`; pinned by `tests/test_searcher.py:430`, `tests/test_indexer_run.py:147` |
| `VecsConfig.save()` rewrites config.yaml with ONLY `{"projects": …}` and `load_config` reads only `projects` — any new top-level key is silently stripped on the next `add_document` auto-configure (`mcp_server.py:183`, `cli.py:184,239,253`) | `config.py:126-148, 176-178` |
| Chroma telemetry on (no `Settings`) at all three `PersistentClient` sites | `clients.py:25`, `prose_drift.py:363,370` |
| Eval harness exists: stale-retrieval-rate + eval-set scaffold (Inc 1-E); `run_eval` defaults to production `searcher.search` | `eval_harness.py` |
| RRF fusion constants: `k=60, w_vector=1.0, w_bm25=0.6` | `searcher.py:112-118` |
| Corpus scale: livly ≈ 11,490 code + 1,588 docs chunks (work mac); vecs = 36 + 226 (personal); paused bloomly/eric orphans 599/274 (personal). Chunks ~3K tokens → full livly re-embed ≈ 40M tokens | manifests / Inc 1-A metering |

**Deployment reality (since 2026-06-05): stores are disjoint per machine.** Personal mac holds vecs (+paused bloomly/eric); work mac holds livly and reindexes it daily via launchd. No store is shared; the index-move recipe was a one-time migration. The flip is therefore a **per-machine config edit + full on-device reindex**, not a convergence event. One fleet-wide model id is still policy — it keeps any future store move and Inc-5 sharing valid — but it is enforced by discipline, not mechanics.

## Design

Three phases across the two sub-features. L1 ships value even if the answer is "keep Voyage."

### L1 — no-regret base (sub-feature: local-embed-base)

1. **Retrieval-quality golden sets + A/B runner** (extends `eval_harness.py`).
   - **Location split (security):** the vecs golden set is versioned in-repo (`evalsets/vecs.yaml`) together with the schema and harness. The **livly golden set lives on the work mac only**, outside the repo (`~/.vecs/evalsets/livly.yaml`) — queries and expected paths describe work internals, and the repo's remote is the personally-hosted deploy channel; versioning it in-repo would reopen the exact leak this increment closes. Any A/B report exported off the work mac contains **aggregate metrics only** (no chunk text, no file paths).
   - **Authoring protocol (anti-bias):** queries derived from real information needs (recent agent tasks/sessions), authored **without running vecs search**; expected sets completed by **pooling top-10 results from all arms and adjudicating relevance** before scoring (binary judgments suffice for nDCG); golden file **frozen at a recorded commit before any L2 run** — later edits require a changelog entry and a re-run of all arms.
   - **Size:** livly (the decision corpus) targets **≥100 queries**; vecs ~40 (diagnostic only). Query classes: natural-language intent, exact-identifier, cross-file concept — reported separately (BM25 cushions identifier queries, so embedding deltas concentrate in NL).
   - **Metrics:** recall@5, recall@10, nDCG@10, MRR, with **paired-bootstrap confidence intervals over queries**; arms: hybrid (production config), embed-only, BM25-only.
   - **Runner mechanism:** refactor the search pipeline into a function parameterized by (collections, provider, model, bm25 paths) that `search()` itself calls with production values and the harness calls with arm values — the hybrid arm is the production path *by construction*, not by re-implementation. RRF constants held identical across arms (`k=60, w_bm25=0.6`).
   - This artifact outlives the swap question — it is the regression harness Inc 3 (chunking) and Inc 7 (quality) need anyway.
2. **Provider abstraction.** New `embed_provider.py`: `EmbedProvider` protocol — `embed(texts, *, model, input_type) -> EmbedResult(embeddings, total_tokens)` plus `retryable_errors: tuple[type[BaseException], ...]`. `VoyageProvider` wraps the existing singleton verbatim (token counts from `.usage`). `QwenLocalProvider` lazy-imports sentence-transformers: `device=mps` (cpu fallback), output dim per §L1.3, normalized vectors, `input_type="query"` → the model's query prompt (`prompt_name="query"` — Qwen3 is instruction-asymmetric; missing this silently degrades retrieval), token counts from the HF tokenizer (keeps `AdaptiveBatcher` calibration semantics; embed tokens feed only the batcher, there is no embed metering path). Selected by an `embed_provider:` field in `~/.vecs/config.yaml`, default `voyage` — zero behavior change at merge. **The field must be modeled on `VecsConfig` and persisted through `save()`** (load→save round-trip test required): `save()` currently emits only `projects`, and it fires on every `add_document` auto-configure — an unmodeled field would be silently stripped, reverting the fleet to voyage and triggering a mass re-embed at the next reindex. Packaging: heavy deps (torch + sentence-transformers) behind an optional extra `vecs[local]`; core install unaffected; clear error if the provider is selected without the extra.
3. **Model-id discipline.** Local model ids are distinct strings (e.g. `qwen3-embedding-4b@mrl1024`, `qwen3-embedding-0.6b`) added to `EMBED_DIMS`, which becomes *prescriptive* for local models (it sets `truncate_dim`) — update the `config.py:27-33` comment accordingly. Enforced, not conventional: the provider **asserts `len(vec) == EMBED_DIMS[model]` on the first batch**, and a dim change REQUIRES a new model id (declared in config.py) — the `(model, sha256)` cache key and `collection_models` markers track only the id string, so an in-place dim edit would otherwise serve stale-dim cached vectors with zero invalidation.
4. **Close the code-collection marker hole (indexer-side only).** Extend `_remodel_*` markers to code collections; the searcher interlock already fail-closes on a present-and-mismatched marker — **zero searcher change**, add only a regression test pinning code-marker-mismatch → BM25-only. **First-run backfill:** an unmarked non-empty code collection gets its marker *recorded* as the current `CODE_MODEL` with **no clear** — clear+re-embed fires only on a real recorded-vs-configured mismatch (reusing `_model_changed`'s None→changed semantics verbatim would mass re-embed livly's 11,490 code chunks on the first post-merge reindex, violating no-regret). Add a test pinning "unmarked code collection + unchanged model → no manifest clear". This deliberately retires the documented "code has no trigger" invariant: replace the pinned test `test_remodel_clears_docs_leaves_code_on_model_change` and rewrite the corresponding `src/vecs/CLAUDE.md` bullet.
5. **Chroma telemetry off.** `Settings(anonymized_telemetry=False)` at all three `PersistentClient` sites.

### L2 — measure (non-destructive; live store untouched) (sub-feature: local-embed-ab)

1. **Apple Silicon perf spike** (the research's empty quadrant), for 0.6B and 4B on MPS: fixed batch-size matrix over the real chunk-length distribution, with a **sustained leg (≥20–30 min continuous at steady state)** whose tokens/sec is the projection basis — a minutes-long burst overstates an ~11h run (thermal throttling, sustained memory pressure). Record peak RAM under sustained load. Output: projected full-reindex wall-clock + steady-state RAM at livly scale (~40M tokens). Chunk source on the personal mac: bloomly/eric orphan chunk text (599+274 chunks ≈ 2.6M tokens; vecs alone is only 262 chunks). Repeat the spike on the work M5 48GB before any flip there. The spike's output is the **feasibility precondition** consumed by the L3 gate.
2. **Frozen-snapshot shadow pairing.** Freeze the checkout (record SHAs), then build **both arms as shadow collections from the same frozen tree**: `<p>-code-qwen4b`/`<p>-docs-qwen4b` (and optionally 0.6B arms) AND a Voyage shadow — the `(model, sha256)` embed cache makes the Voyage shadow nearly free (unchanged chunks are cache hits, no API spend), and it makes "the only variable is the embedding" true by construction for the vector AND BM25 sides (one BM25 build from the frozen tree serves all arms; BM25 is embedding-independent). No pulls between arm builds. Shadows are never visible to MCP search and are deleted after the decision.
3. **A/B on the golden sets** via the parameterized runner: per arm × per query class, hybrid + embed-only, bootstrap CIs. Optional rescue arm: qwen-4B + Reranker-4B over top-50, with rerank latency measured on-device (research flags autoregressive rerank latency as a real risk for the interactive MCP path).
4. Personal mac measures vecs (diagnostic); work mac measures livly with the same harness. **The livly numbers are the decision numbers.**

### L3 — owner decision gate (sub-feature: local-embed-ab)

**Step 0 — feasibility precondition (from the spike):** an arm survives only if projected full-reindex wall-clock and steady-state RAM are within owner-set bounds for the machine that must run it (proposal: reindex ≤ 12h i.e. overnight, RAM headroom ≥ 8GB free at steady state). If 4B fails feasibility, the 0.6B arm is evaluated as the swap target in its place.

**Quality rows** — evaluated on the **livly** golden set, hybrid mode, per surviving arm. All thresholds are proposals for owner sign-off; gating is on **bootstrap CI bounds, not point estimates** (~100 queries cannot resolve a 3-pt boundary as a point estimate). Aggregation rule pre-registered here: headline = worst of (code, docs) per metric; plus a per-class floor.

| Outcome | Condition (per surviving arm) | Action |
|---|---|---|
| **Swap** | CI **upper bound** of degradation < 3 pts recall@10 AND < 0.03 nDCG@10 (worst of code/docs), AND NL-class floor holds: hybrid NL recall@10 within 5 pts AND embed-only NL recall@10 within 15 pts | Flip `embed_provider` per machine, full reindex (runbook below), retire `VOYAGE_API_KEY` from the work mac, drop shadow collections |
| **Keep Voyage** | CI **lower bound** of degradation > 8 pts recall@10 OR > 0.08 nDCG@10 (worst of code/docs) | Keep provider=voyage everywhere; take the A/B report (aggregates only) to work as evidence for a Voyage usage request. L1 artifacts all remain in service |
| **Gray zone** | anything else (CI straddles a boundary, or NL floor fails while aggregate passes) | Owner judgment, with two rescue options in cost order: (1) RRF weight sweep for the qwen arm (fusion constants were tuned in the voyage era — a cheap retune may close the gap), (2) the Reranker-4B arm — if qwen+reranker ≥ Voyage and rerank latency is acceptable, swap-with-reranker (pulls that slice of Inc 7 forward) |

**Flip mechanics (disjoint stores — each machine independently):** snapshot store → pause cron (work mac) → set `embed_provider` + model ids in that machine's config.yaml → full reindex (the L1.4 markers drive the code+docs re-embed) → E-harness before/after + golden-set spot-check → resume cron. Steps 2–4 of the old live-reindex runbook (`docs/vecs-direction-review-2026-06.md`: snapshot, pause cron, before/after baseline) generalize; its step 1 (Voyage dim probe) is superseded — the full flip runbook is written at L3 acceptance time, the old one is precedent only. **Order: personal mac flips and soaks first** — soak = N days (proposal: 7) of dogfood search on vecs under qwen with zero interlock warnings and stale-rate unchanged. The soak validates the provider code path and MPS stability, NOT livly-scale quality (that is what the work-mac A/B already measured); then the work mac flips with its own overnight reindex. Facts are still empty (Inc 2 unstarted), so `FACTS_MODEL` flips by constant with no migration — this is exactly why this increment precedes Inc 2 (facts born under the final model; the parked fact-store re-embed migration stays parked).

## Program reshape (parent doc v5 must absorb)

New order: **1.7 (this) → 2 → 3 → 4a/4b → 5 → 7. Inc 6 CUT.** Inc 4c remains a v4 cut-candidate (cut unless Inc-1-E shows redundancy harm — still an open owner decision, not a done cut).

- **Why 1.7 first:** every later increment stamps the embedding model into durable artifacts — Inc 2 births facts under `FACTS_MODEL` (no-delete store, re-embed migration unbuilt), Inc 5 stamps bundles, Inc 1-E baselines assume a fixed model. Swapping after any of them multiplies migration cost; swapping before costs one re-embed of code+docs only.
- **Inc 6 cut:** its premise — session transcripts as an indexed source — died with the 2026-06-04 sessions removal. Its two preserved ideas (per the direction review) **re-home to Inc 2's** chat-triple extraction: stage candidate facts to a jsonl before promotion, and git-anchored (commit-SHA) provenance per distilled fact.
- **Inc 2 absorbs old 1-D** (per-collection RRF refactor + facts FTS5 sidecar): D is unbuilt and Inc 2 is its only consumer.
- **Inc 7:** Contextual Retrieval remains the headline. The reranker slot is **re-pointed from voyage rerank-2.5-lite to Qwen3-Reranker-4B** (under a swap, a Voyage reranker re-opens the outbound path; under keep-Voyage it stays an option). L2's reranker arm gives it a measured head start.
- **v5 must also absorb the 2026-06-04 sessions removal** (owner decision) throughout: drop the `<p>-sessions` topology row, §2 session counts, the sessions down-weight rationale in §5.1/1-D, and re-scope gaps G2a/G2b as closed-by-removal.

## Hygiene rider (lands with 1.7, mostly docs)

- Reconcile `docs/vecs-roadmap.md` (stale 2026-05-31: still lists session indexing as shipped AND "voyage-3 for prose" — false since the 2026-06-04 voyage-4 reindex; missing shipped 1.5c/1-E/1-A items) and bump the parent design doc to v5 with the reshape above.
- Close `kb-foundations-pipeline` Phase-8 (retro.md/gaps.md missing; only acceptance.md exists).
- Orphaned `bloomly`/`eric` collections (599/274 chunks; repos paused, absent from config.yaml): document as paused now; they serve as the L2 spike's chunk-text source; **sweep at flip time if swapping** (voyage vectors are dead weight under a qwen store).
- Reindex backup dir `_backup-reindex-20260604-210203`: **gone from the personal mac** (verified 2026-06-11; likely removed with the livly release). Check the work mac (rina) when reachable; remove there after the gate decision if present, else drop this item.
- Memory index correction for the deployed-editable install state (done 2026-06-11).

## Hardware / deployment context (informative)

| Machine | Spec | Fits (weights arithmetic, UNVERIFIED — L2 measures actual peak RAM) |
|---|---|---|
| Personal mac | M4 Pro, 24GB | 4B fp16 ≈ 8GB weights; 8B tight |
| Work mac (rina) | M5, 48GB | 4B easy; 8B possible |
| HQ Mac Studio (planned CI agent) | ~128GB | 8B + Reranker-4B resident; Inc 5 territory |

Studio note: 8B is a different vector space than 4B. If the Studio indexes and members embed queries locally (share shape A), members must run the same model — so 8B only makes sense with shape B (Studio answers queries; members run nothing). Decision deferred to Inc 5; the fleet default is 4B@1024 everywhere until then.

## Out of scope

Reranker integration in the search path (Inc 7; only measured here). Stage-0/1 team sharing + Studio deployment (Inc 5). Fact-store re-embed migration (parked v2; obviated at birth by swap-early). Higher-dim collections (2560 native 4B) — possible later since a swap recreates collections anyway, but 1024 keeps the A/B apples-to-apples and the store dim-stable. Multi-branch indexing (deferred increment). Embed-cost metering for local providers (no embed metering path exists today; not added here).

## Phase 7 — Dry-run

- `dryrun_selection`: smallest additive real subtask = **Chroma telemetry off** (`Settings(anonymized_telemetry=False)` at the three client sites). Additive, zero ranking semantics.
- `dryrun_acceptance` (inline): `uv run pytest -q` green; new test asserts every constructed Chroma client has `anonymized_telemetry=False` in its settings.
- `dryrun_pass_criteria`: `[pipeline-pass, review-loop-satisfied]`. `abort_policy`: branch-drop.

## Phase 2 — Context docs

On acceptance, update `src/vecs/CLAUDE.md` for: `embed_provider.py` (new), `clients.py` (provider selection + telemetry settings), `config.py` (persisted `embed_provider` field + save() round-trip, qwen model ids, EMBED_DIMS prescriptive-for-local comment), `indexer.py` (provider-routed embed + retry tuple, code-collection markers + backfill — rewrite the "code has no trigger" invariant bullet), `searcher.py` (provider-routed query embed; interlock unchanged), `embed_cache.py` (qualified model ids), `eval_harness.py` (golden-set A/B runner + parameterized pipeline), `prose_drift.py` (provider-routed `_voyage_embed` rename).
