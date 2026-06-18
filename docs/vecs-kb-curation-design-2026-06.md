Authored by Claude

# vecs Knowledge-Base Curation — End-State Design & Increment Program

**Date:** 2026-06-01 (rev v4: 2026-06-03; rev v5: 2026-06-19)
**Revision:** v5 (quality-max re-aim of Increment 7). v1 attacked by 4 reviewers → v2; v2 re-attacked by 4 → v3; v3's shipped Inc-1-pipeline work + the §6 program re-attacked by a 7-lens adversarial critic panel → v4 (`course_correct`: direction right, sequencing wrong — full verdict in `docs/vecs-direction-review-2026-06.md`). **v5 (2026-06-19, owner directive — quality is the dominant axis, latency is not a constraint): Increment 7 re-aimed from a flag-off latency-hedged amplifier to a default-ON, E-gated quality stack — full `rerank-2.5` (not lite) default-ON + task-conditioned `instructions` + Contextual Retrieval + voyage-context-3 + MMR diversification + HyDE (experiment-gated); the query router is parked as a latency-only/contingent lever (see Inc 7).** Material corrections are flagged inline as **[v5-fix]**/**[v5]** (this round), **[v4-fix]**, **[v2-fix]**, or **[v1-fix]** (prior rounds).
**Status:** Approved direction; per-increment specs follow. Scope items marked **(contingent: §7)** depend on an unresolved decision and must not be frozen into an `acceptance.md` until resolved.
**Pillars:** (1) frontier retrieval quality, (2) team-shared knowledge base, (3) agent-facing tool surface.
**Companion docs:** `docs/vecs-platform-strategy-2026-05.md` (platform strategy + SOTA), `docs/vecs-roadmap.md` (platform direction), `src/vecs/CLAUDE.md` (module context).

---

## 0. What this document is

The **end-state design** for vecs as a *frontier-quality, team-shared knowledge base for coding agents*, plus the **sequenced increment program** to get there. Grounded in a verified investigation of the live repo + `~/.vecs/` store, a 2026 SOTA research pass, and two rounds of adversarial review (citation scopes/magnitudes reflect the audits).

### North star

Every decision is judged by one test: **does it make the coding agent a more effective engineer — better recall of the right context, less rediscovery, fewer wrong turns?**

### Hard constraints (non-negotiable)

- **Embedded, zero external servers.** Stores live in ChromaDB or SQLite/FTS5 under `~/.vecs/`. No graph DB, no standing service.
- **Contract-first.** Config-schema and tool-surface changes are designed before code.
- **Appropriate context, not maximum.** Condense to high-signal; keep raw addressable as fallback.
- **Index storage never written inside the repo.** `~/.vecs/` only. (This excludes Git-LFS-into-the-code-repo as a bundle target — see §6 Increment 5.) **[v2-fix]**

### Explicitly out of scope

The **repo-dependency / code-graph** bet (the largest standalone Pillar-1 lever, ~32.8% SWE-bench in the strategy doc) is **not** part of this curation program; it is tracked in `docs/vecs-roadmap.md` (Track B2). This program improves *curation, freshness, facts, and sharing* — not the code graph.

---

## 1. The end-state vision

**One line:** vecs becomes the *most up-to-date, condensed-but-not-lacking* team-shared knowledge base — code, docs, sessions, durable facts — where retrieval is fresh, version-aware, deduplicated, the team's best-curated knowledge is first-class and searchable with provenance, and the index is publishable to teammates *after* it is freshness-defended.

**The shape ("condensed but not lacking"):** index **small, high-signal units** (tight chunks + curated/extracted facts) for precision → keep **raw transcripts/docs addressable** as a lossy fallback → **freshness-stamp and supersede** so stale content is filtered *before scoring* → **then publish** as an immutable bundle for the team → **amplify with contextual retrieval + reranking last**.

### End-state collection topology

| Collection | Role | Built from | Searched | Lifecycle |
|---|---|---|---|---|
| `<p>-code` | code recall | code files (extension-filtered, **no `.md`**) | default | git-driven incremental; `version_id` + validity; tombstoned |
| `<p>-docs` | doc/prose recall | `docs_dir` **+ multi-path per-repo docs + rerouted in-repo `.md`** (heading-chunked, source-root-qualified ids) | default | hash-incremental; `version_id` |
| `<p>-sessions` | raw transcript fallback | chat JSONL (lightly cleaned; later extract-and-link) | default, **down-weighted** (per-collection RRF weight) | append-aware; near-dup deduped |
| `<p>-prose-facts` | **first-class durable facts** | promoted curated `memory/*.md` + chat-extracted triples, gated by a write-time merge | default after Increment 2 (the `is_current` filter ships with facts search in Inc 2); **FTS5 sidecar built in Inc 1** | bi-temporal supersede; provenance + scope tier |

Per-chunk metadata contract (end-state): `version_id`, `valid_from`/`valid_to`, `provenance` (source + actor/agent + scope tier), `kind` (static/versioned/event). Sessions are down-weighted via a **per-collection RRF weight** (§5.1). **[v1-fix: mechanism was unspecified.]**

---

## 2. Verified current state (2026-06-01)

**What is indexed.** Three collections per project (`code`/`sessions`/`docs`) across `bloomly`, `eric`, `livly`, from `~/.vecs/config.yaml`. Code matched by **file extension** (mandatory per `code_dir`, no default — `config.py:39-40`); sessions by globbing `*.jsonl` in one `sessions_dir` per project plus Codex sessions routed by `cwd`; docs from **one** `docs_dir` per project — **only `livly` has one** (`config.py:49`); `bloomly`/`eric` have none. Live counts: **code 12,823** (bloomly 599, eric 274, livly 11,950); **sessions 1,134** (bloomly 108, **eric 0 — empty**, livly 1,026); **docs 1,586** (livly only). `chroma.sqlite3` ≈ 3.6 GB (`chromadb/` dir ≈ 4.2 GB). **[v2-fix: v2 said livly docs 1,580 / sessions 1,025; live is 1,586 / 1,026, and eric-sessions is empty.]**

**Chunking.** Code = tree-sitter AST (C#/TS) else 200-line line-chunks (overlap 50); sessions = 10-message groups (overlap 2); docs = H1/H2 headings, paragraph fallback. Refs: `config.py:24-27`, `ast_chunker.py`, `chunkers.py`, `doc_chunker.py`.

**Sessions are lightly cleaned, not verbatim.** `preprocess_session` (`chunkers.py:46-96`) strips `<system-reminder>` + base64 and reformats to `[role]: text`; embedding is later in `_embed_and_store` (`indexer.py:385-458`).

**Curation / dedup / freshness (today).**
- Incremental only: SHA-256 file-hash manifest skips unchanged files (`needs_indexing`, `indexer.py:129-138`); unchanged files never reach `_embed_and_store`. Sessions tracked by byte-offset + identity-hash.
- **Dedup is search-time only** — Jaccard 0.55 (`searcher.py:57-79`). Chunk ids are path-keyed `{source_key}:{chunk_index}` (`_make_chunk_id`, `indexer.py:348`), **no content hash**.
- **Fusion concatenates all collections into one RRF rank space** with global weights (k=60, w_vector=1.0, w_bm25=0.6; `searcher.py:82-114`, fused once at `:209` over concatenated `all_results`/`bm25_results`). Code hits are appended first (`:145-146`), so they systematically out-rank docs/sessions by append order. **No per-collection weighting, no reranker, no recency.** **[v2-fix: relevant to Inc-1 search work — per-collection weighting is a fusion refactor, not a parameter add.]**
- Stale-chunk cleanup runs after a successful embed for docs (unconditional, `:1118`) and **full** session re-index, but **not for incremental session appends** (`:942-943`). `prune_out_of_scope` (`:189-227`) removes manifest keys under a code-dir root that are no longer in scope (and their chunks) — but it keys on full-path manifest keys and is exclude-dirs-oriented; it is not an asserted path for "extension removed from config." Mid-batch-crash orphans self-heal: a partially-embedded file isn't in `fully_succeeded` (`:604-607`) so the manifest doesn't record it; reprocessed + cleaned next reindex.

**Fact store: write path exists but is OFF; no data.** The fact-write path (`indexer.py:968-1000` → `add_fact_with_state_machine`, `prose_drift.py:685-758`, an **exact `chain_key` INSERT/NOOP/SUPERSEDE merge** flipping `is_current`, never deleting) is gated by `prose_drift_enabled` (default `False`, `config.py:54`), **absent from `~/.vecs/config.yaml`**. Verified live: ChromaDB lists only `bloomly-code/-sessions`, `eric-code/-sessions`, `livly-code/-docs/-sessions` — **no `*-prose-facts` collection; zero triples written.** Facts (when written) embed with voyage-3 via `_voyage_embed` (`prose_drift.py:488-491`, which consumes `SESSIONS_MODEL` — so a sessions-model swap silently changes the facts model too); the collection is created with no `_sync_bm25` → vector-only, no FTS5 sidecar.

**The memory↔vecs inversion (confirmed).** Curated `memory/*.md` facts are never indexed (`sessions_dir` globs `*.jsonl` only, `:1024`); raw chat transcripts are (1,026 livly session chunks). vecs is single-machine, per-user stdio MCP.

---

## 3. Gap map (verified + SOTA-backed)

Severity = impact on the north star.

| # | Gap | Sev | Owner increment | SOTA backing (§9) |
|---|---|---|---|---|
| G1 | **Memory↔vecs disconnect** — curated facts not indexed; fact-write path disabled, prose-facts collection nonexistent | High | 2 | Collaborative Memory |
| G2a | **Chat-transcript redundancy** (cumulative histories repeat) | Cost-only | 4c dedup; largely subsumed by 6 (transcript demotion) | byte-exact dedup ≈zero quality change (cost/latency) |
| G2b | **Chat-transcript topical noise** (verbatim dialogue competes with answers) | Med | 6 (transcript inversion) | distractor harm (analogy) |
| G3 | **`.md` indexed AS code** | Med | 1 (reroute) | distractor harm (analogy) |
| G4 | **Scattered docs** — only `docs_dir` indexed; per-repo `docs/` dirs missed | Med | **1 (in-repo `.md` under code_dirs) + 3 (per-repo `docs/` dirs)** | coverage helps ("Less LLM, More Documents") |
| G5 | **No freshness defense** — no version stamp, no pre-retrieval supersession filter, no tombstones; incremental appends never clean superseded chunks; **`Manifest.prune()` orphans chunks on delete (verified)** | High | 4a | stale worse than no retrieval (code-RAG; HoH) |
| G6 | **No index-time dedup** | Med | 4c (with ownership model) | MinHash-LSH standard infra (qualitative) |
| G7 | **Fact store has no actor/scope/valid-time** | Med | 2 | bi-temporal valid-time + provenance |
| G8 | **No *semantic* promotion gate** (exact `chain_key` merge exists; no top-k semantic merge) | Med | 2 | Mem0 ADD/UPDATE/DELETE/NOOP top-k |
| G9 | **Chunk size not tuned + no parent/window fetch** | Low | 3 | small chunks ≈2× precision; gate on measured answer win |

**[v2-fix: G4's per-repo `docs/` half and G9 are now owned by Increment 3 (Docs & chunking), which was dropped in the v2 rewrite and is restored here. G2a is explicitly addressed by 4c dedup and largely subsumed by Increment 6, so it is not a silent orphan.]**

---

## 4. Design principles from the research (audited twice)

1. **Curate, don't hoard — the lever is noise/staleness/redundancy, not size.** Backing: redundancy pruning (Zero-RAG, high), stale-harm (HoH, high), distractor-harm (Distracting Effect, high *for its NQ single-distractor setting*; our `.md`/chat uses are analogies), coverage helps (Less LLM More Documents, high — used only for G4). **[v1-fix: "quality dominates size" overstated; downgraded.]**
2. **Stale is worse than absent.** Version-stamp, **hard-filter superseded before scoring**, tombstone deletes. (Direction high; magnitudes from small samples — §8.)
3. **Extract beats verbatim for chat — but extraction is lossy** (33–35 pt on some categories; **~7 pt on PersonaMem-v2** — not uniform). Promote extracted facts **and keep raw addressable**. (Medium.)
4. **Dedup is cost/latency, not quality — only byte-exact is proven loss-free.** Near-duplicate (MinHash) dedup is **not** covered by the zero-loss result and must be gated; the ~80%/24% reduction figures are **our corpus estimates**, not from the cited infra sources. **[v1-fix.]**
5. **"Condensed but not lacking" base case is cheap.** Small chunks (~200–400 tok) ≈**2× precision** (Chroma eval 3.6→7.0, "doubled"; **recall trades down somewhat — mitigate with parent/sentence-window fetch**, do not claim equal recall) + the cheap **ClusterSemanticChunker** (no LLM, highest precision); only the **LLM-prompted** chunker isn't worth its cost. **[v1-fix: "4.7×" was fabricated; v2-fix: dropped the unbacked "equal recall" qualifier.]**
6. **Contextual Retrieval is the highest-ROI quality upgrade — amplifies whatever is indexed** (−49% / −67% with reranker). Sequence after curation. (High, verbatim.)
7. **Facts: supersede, don't delete; version first-class.** (High.)
8. **Recency is a conditional prior, never dominant, eval-gated.** Source weights α on the *semantic* term; when wiring α to the *recency* term keep it low. The reported effect is a drop to ~0.667 (not a "collapse") on event-log data that **won't transfer** to code/doc. **[v1-fix.]**
9. **No unsupervised clustering for canonical-version/topic selection** (≈0.08 F1, topic-clustering; version-selection is our inference).
10. **Team memory needs provenance + tiers + redaction + a promotion gate**, consolidation **offline**. (High — Mem0, Collaborative Memory; the naive-accumulation source is an illustrative vendor blog backed by the peer-reviewed pair.) **[v1-fix.]**
11. **Measure version-alignment, not just relevance.** Track stale-retrieval-rate (needs a per-chunk `version_id`/embed-hash anchor); memory on LongMemEval/LoCoMo. Harness seeded in Increment 1. (High.)

---

## 5. Target architecture (concrete)

### 5.1 Retrieval pipeline (end-state)

```
query
  → query transform: HyDE on NL-intent path (embed hypothetical doc; E-gated experiment) [Inc 7]
  → PRE-RETRIEVAL validity filter (metadata predicate, before fusion)        [Inc 4a]
      · facts: is_current = True                                              [the facts predicate ships with facts search in Inc 2]
      · code/docs: not superseded; valid_to unset or in the future ("expired" = valid_to in the past)
  → embed (per-collection model) + BM25 (FTS5, incl. facts sidecar [Inc 1])
  → RRF fuse with PER-COLLECTION rank lists + weights (sessions down-weighted) [Inc 1: fusion refactor]
  → kind-aware recency prior (flag, default-OFF, eval-gated, α low)           [Inc 4b]
  → rerank: voyage rerank-2.5 FULL, default-ON, E-gated + task-conditioned instructions (B1 workflow) [Inc 7]
  → MMR diversification (λ high — near-dup only; bundle coverage)             [Inc 7]
  → result-shaping (detail levels, get_chunk)                                 [later / Pillar 3]
```

**[v1-fix: validity filter moved pre-fusion. v2-fix: the `is_current` facts predicate is inert in Inc 1 (facts empty/unsearched), so it ships with facts search in Inc 2 — Inc 1 builds only the FTS5 plumbing + the fusion refactor. Per-collection weighting requires restructuring `reciprocal_rank_fusion` into per-collection rank lists (today it fuses one concatenated list, `searcher.py:82-114,209`); "preserve current ranking" is NOT a meaningful baseline since current order is partly append-order.]** **[v5: the Inc-7 line is now a default-ON, E-gated quality stack (HyDE query-transform + full `rerank-2.5` + task-conditioned instructions + MMR), not an optional flag-off rerank. The query router that would gate these by query class is parked — under quality-max it is a latency-only optimization; it earns a place only if Inc-1-E shows the expensive path *regresses* a query class (then route rerank away from that class). See Inc 7.]**

### 5.2 Stores

- **code / docs / sessions** — ChromaDB + FTS5 sidecars; add `version_id` + validity; **content-hash embedding cache** (re-embed only changed chunks). The cache must contribute cache-hit chunk ids to `succeeded_ids` and idempotently ensure the chunk is present, so the per-file `succeeded == expected` invariant (`indexer.py:604-607`) still holds — **this is the invariant the v1 dedup idea violated.** **[Inc 1.]**
- **facts** — `<p>-prose-facts`, first-class: **FTS5 sidecar** (Inc 1); bi-temporal; provenance + scope tier; populated and gated in Inc 2; embedding model pinned + dim-stamped on the collection (note `_voyage_embed` shares `SESSIONS_MODEL`).

### 5.3 Memory bridge (end-state)

```
source (curated memory/*.md | chat transcript)
  → extract / parse (lossy; keep source pointer)
  → exact chain_key merge (EXISTS: prose_drift.py)  +  semantic top-k merge gate (NEW)
  → attach provenance + scope tier
  → (shared tier) redact personal/sensitive          [paired with the Stage-0 consumer, Inc 5]
  → write to <p>-prose-facts (searchable)  + keep raw transcript addressable
  (offline / scheduled, not in the hot path)
```

### 5.4 Team-sharing (Stage-0 bundle)

Publish the index once as an **immutable bundle** (Chroma dir + FTS5 `.db` + a manifest stamping **embedding model + dim**, commit-SHA versioned); teammates `vecs pull` and read locally. **Hard requirement = the manifest model/dim stamp + the content-hash cache** (both from Inc 1); per-chunk `version_id` is a later enhancement for *incremental* bundle rebuilds. **The first team-wide publish is gated on Increment 4a's freshness filter** (see §6 Inc 5) — we do not amplify a stale index across the team. Stage-1 shared HTTP MCP and Stage-2 object-store remain later stages (strategy doc).

---

## 6. Increment program

**Seven increments + Inc 1.5 (a no-regret correctness wedge inserted after the 7-lens review)**, dependency- and ROI-ordered. Each is a feature run through the workflow profile (`docs/features/<name>/` with `acceptance.md`, dry-run, Phase-4 review, retro). **Larger increments decompose into independently-gated sub-features, each with its own `acceptance.md`** (separate Phase-4 review + sign-off) — not one coupled gate.

**Order rationale:** cheap no-regret foundations → **fix the one verified "agent retrieves lies" defect + stand up the measurement instrument (Inc 1.5 + Inc-1-E/A) before any unmeasured mutation** → bother-closing facts → docs/coverage → freshness (so a clean+fresh index exists) → *then* team-share → boldest transcript change → quality amplifier last.

**[v4-fix — revised sequencing from the 7-lens direction review (`docs/vecs-direction-review-2026-06.md`):** the program front-loaded correctness-heavy machinery (Inc 2/3/4a) and unmeasured quality bets *ahead of* (a) the one verified north-star defect — `prune()` orphans, live, lies to the agent on every search — and (b) the one instrument (Inc-1-E stale-retrieval-rate) that would tell us whether the later bets matter on this corpus. Correction: **insert Inc 1.5** (prune-orphan fix + searcher `-docs` gate + freshness/trust signal) to ship next; **pull Inc-1-E + a thin Inc-1-A forward** (built right after 1.5a, *ahead of* the increments that gate on them — 3/4b/6/7 and the heavy half of 4a); a **green E number is the hard precondition** for those. **Inc 4c** (MinHash near-dup dedup) is demoted to a **cut-candidate** — defer indefinitely unless E shows real redundancy harm.]**

**Build order (revised):** 1-pipeline (✅ shipped, local commits) → **Inc 1.5a** → **Inc-1-E + thin Inc-1-A** (E right after 1.5a so the stale-retrieval-rate drop is the proof the fix worked) → **Inc 1.5b/1.5c** → live migrating reindex (owner's manual step, gated on a before/after E baseline) → Inc 2 → Inc 3 → Inc 4a → Inc 4b (eval-gated) → Inc 5 → Inc 6 → Inc 7. **Inc 4c cut unless E justifies it.**

### Increment 1 — Foundations & no-regret wins  → **3 independent sub-features**

**[v2-fix: v2 bundled six deliverables under one acceptance, violating the decompose rule. Split into three workflow-profile features, each with its own `acceptance.md`:]**

- **1-pipeline** (`docs/features/kb-foundations-pipeline/`) — share one reindex:
  - **F. `.md`→docs reroute.** Remove `.md` from all `code_dirs`; **explicitly sweep `.md`-sourced chunks out of every `-code` collection + BM25 sidecar** (dropping the extension does NOT delete the ~431 already-embedded livly `.md` code chunks; `index_code` only adds — `prune_out_of_scope` may catch them but is unasserted, so add an explicit sweep + a test asserting zero `.md` in `-code`). Route in-repo `.md` under code_dirs into the project `-docs` collection — **this is `index_docs` surgery**: it currently `return 0`s without a `docs_dir` and uses `relative_to(docs_dir)` (raises for code-dir files) with bare-rel_path chunk ids that collide across roots (`indexer.py:1073,1104,1114`). Build **multi-source `docs_dirs` + per-source base dir + source-root-qualified rel_path** so two repos' `README.md` don't mutually delete. **[v2-fix.]**
  - **B. voyage-3.5 for docs/sessions.** Equal dim ≠ equal vector space — a model swap against an un-re-embedded corpus silently degrades ranking (query vectors from voyage-3.5 vs stored voyage-3 docs). So **re-embed docs/sessions under the content-hash cache** (dim-equality is necessary-not-sufficient); validate with known query→expected-source pairs, not just non-empty results. Code stays voyage-code-3. Note `_voyage_embed` (facts) also reads `SESSIONS_MODEL`. **[v2-fix.]**
  - **C. `version_id` + content-hash embedding cache.** Stamp every chunk; cache by content-hash; cache hits contribute ids to `succeeded_ids` + idempotent upsert (preserve the `succeeded == expected` invariant — test the **mixed changed+unchanged-chunk file** case explicitly). **[v2-fix: the cache test must change one chunk in a file and assert only it re-embeds — a no-change reindex already does zero embeds via the manifest skip and does NOT test the cache.]**
- **1-search** (`docs/features/kb-foundations-search/`):
  - **D. Per-collection RRF refactor + facts FTS5 sidecar.** Restructure `reciprocal_rank_fusion` into per-collection rank lists + weights (sessions down-weighted); acceptance asserts the **new weighted order is correct** (current concatenation order is not a baseline to preserve) and that down-weighting sessions does not spuriously trip the 2×/3× refetch (`searcher.py:154-216`) or `deduplicate_results`. Give `-prose-facts` an FTS5 sidecar via `_sync_bm25` (empty until Inc 2). **[v2-fix.]**
- **1-instrumentation** (`docs/features/kb-foundations-instrumentation/`) — **[v4-fix: moved up — built right after Inc 1.5a, ahead of Inc 3/4b/6/7 and the heavy half of 4a, which all gate on a green E number. Was sequenced last in Inc-1, i.e. *after* the mutations it should measure — backwards.]**
  - **A. Metering spike** — per-call cost record (model/tokens/$), a **hard `MAX_CALLS_PER_DAY` cap**, extraction model = **latest Sonnet (`claude-sonnet-4-6`)**, stage-2 contradiction-judge = **latest Opus (`claude-opus-4-8`)** (owner-decided 2026-06-03: bulk extraction on the cheap-strong model, the rare decisive contradiction call on the strongest). Keep it a **spike** (per-call record + cap), not a dashboard. A prerequisite instrument that informs (and, via the §7 cost-ceiling decision, can gate) Inc 2/6 — not itself an execution gate. **[v2-fix: "gates Inc 2/5/6" overstated.]**
  - **E. Measurement-harness seed** — `stale-retrieval-rate` (defined against the `version_id`/embed-hash anchor from C, with a graceful "legacy/unknown" bucket for un-restamped chunks) + a small local eval-set scaffold, **pointed at the live store**. A **green E number is the hard precondition** for Inc 3/4b/6/7 + the heavy half of 4a. **Depends on C.** **[v2-fix; v4-fix: promoted to gate + live-store target.]**

**Phase-7 dry-run (parent):** smallest additive subtask = the **`docs_dirs` back-compat coercion** (config-load migration mirroring `sessions_dir`→`sessions_dirs`, `config.py:89-95,213-217`), with a clean "existing `docs_dir`-only config behaves identically" assertion. **[v2-fix: the prior choice (per-collection RRF weight) is a refactor whose "ranking unchanged" criterion is unsatisfiable.]**

### Increment 1.5 — No-regret correctness wedge  **[v4-fix: new; ships before Inc 2]**

Own workflow-profile feature (`docs/features/kb-freshness-hotfix/`, its own `acceptance.md` + Phase-4 review). Three independent fixes, each with a test; all reuse already-shipped machinery, none depend on the bi-temporal apparatus of 4a.

- **1.5a — prune-orphan fix (the only verified "agent retrieves lies" defect; pulled out of 4a).** `Manifest.prune()` (`indexer.py:181-189`) clears the manifest for deleted files but returns only a *count*; its `run_index` caller (`~:1850`) never deletes the chunks, so the 4×/day cron leaves deleted files' vectors in Chroma + BM25 forever, ranking against live content. Fix: `prune()` returns the **stale keys** → caller classifies each by collection (code/docs/sessions) → deletes via the shipped `_delete_stale_chunks_after_embed` + `_delete_ids_from_bm25` (call BM25 deletion **directly** — the `_sync_bm25`-only-when-`total_stored > 0` gap means a prune-only run otherwise leaves BM25 rows). Plus a **one-time orphan sweep**: scan each collection's `file_path` metadata, delete chunks whose source is gone on disk (manifest already forgot them; no re-embed). **Independent of `valid_from`/`valid_to`** — 4a keeps only the recurring tombstone *semantics* + the `is_full=False` append-cleanup fix.
- **1.5b — searcher `-docs` gate one-liner (`searcher.py:150`).** F populated `bloomly`/`eric` `-docs` with in-repo `.md`, but search gates the `-docs` target on `proj.docs_dir` (which those projects lack), so F is sunk cost for 2/3 projects. Always attempt `-docs` with skip-on-miss (like code/sessions at `:146-148`). + a test.
- **1.5c — freshness/trust signal.** Surface `version_id`/a freshness bucket in the MCP search-result header, + a **query-time interlock**: when a collection's recorded model marker (`EmbedCache.get_collection_model`) ≠ the configured model, warn (and/or fall back to BM25) instead of silently scoring new-model query vectors against old-model stored vectors. Closes the live model-flip fail-open window (review finding #5) until the migrating reindex runs.

**Depends on:** Inc-1-pipeline C (`version_id` + `EmbedCache`). **Gates:** nothing structurally, but **1.5a's stale-retrieval-rate drop is the first thing Inc-1-E measures** (E is sequenced right after 1.5a as the proof the fix worked).

### Increment 2 — Memory bridge

Enable extraction (metered/capped, Sonnet) to **populate `-prose-facts`**; promote curated `memory/*.md` + chat triples; **semantic top-k merge gate** atop the existing exact `chain_key` machine; provenance + scope tier; wire facts into search incl. the `is_current` pre-retrieval predicate and default blend. Keep raw transcript addressable. Offline/scheduled. **Sub-features:** 2a single-machine fact bridge (populate + gate + provenance + search). Shared-tier + redaction move to Increment 5 (their consumer is the bundle). **Contingent (§7):** promotion source-of-truth; human/critic gate. **Risks:** lossy extraction (keep raw); embedding-model pinning.

**[v5-fix — sessions-removed reconciliation (owner decision 2026-06-04; project memory `vecs-sessions-removed`).** vecs no longer indexes chat transcripts — the store is **code + docs + prose-facts** only, and raw `.jsonl` is a grep verification layer, not an indexed source. So Inc 2's **live extraction source is curated `memory/*.md` promotion + metered extraction over docs/code prose (the memory↔vecs inversion), NOT chat-transcript triples.** The "chat triples" clause above is defunct for Inc 2; transcript distillation stays parked with **Inc 6** (do not build the session-triple path here). Inc 2 = **2a single-machine fact bridge** over the memory-inversion source: populate `-prose-facts` (FTS5 sidecar built in Inc 1) → semantic top-k merge gate on the exact `chain_key` machine → provenance (source path/commit-SHA) + scope tier → wire into search via the `is_current` pre-retrieval predicate + per-collection blend (folds old Inc-1-D's per-collection RRF refactor if not already shipped). Metering cap (`MAX_CALLS_PER_DAY`, Sonnet extraction / Opus contradiction-judge) carries from Inc-1-A.]**

### Increment 3 — Docs & chunking completion  **[v2-fix: restored]**

Owns the rest of G4 + G9. **Scope:** **per-repo `docs/` discovery** — extend the multi-path `docs_dirs` from Inc 1 to auto-discover/configure per-repo `docs/` dirs beyond in-repo `.md` (contract-first config change). Small-chunk tuning (~200–400 tok) + **parent-document / sentence-window fetch**, **gated on a measured precision→answer-quality win** (G9 is Low; build the storage machinery only if the eval shows end-to-end gain). **Depends on:** Inc 1 (docs_dirs, source-root rel_path), a **green Inc-1-E** (the chunking/parent-window win in G9 must be shown end-to-end before building the storage machinery). **[v4-fix: E is now built ahead of this, not after.]**

### Increment 4 — Freshness

- **4a (default-ON, high-confidence):** `valid_from`/`valid_to` + **pre-retrieval supersession/validity hard-filter**; **recurring tombstone semantics** on delete (the *one-time* `prune()` orphan fix + orphan-sweep was pulled out to **Inc 1.5a** — see there for the verified bug + line anchors); **fix the `is_full=False` append-cleanup** (`indexer.py:942-943`); git-driven incremental reindex (Merkle pattern; uses Inc-1 `version_id` + cache). The heavy bi-temporal half is **eval-gated on a green Inc-1-E**. **[v4-fix: the standalone prune-orphan corruption is now Inc 1.5a (shipped before Inc 2), since it is an active "agent retrieves lies" defect independent of `valid_from`/`valid_to`; 4a keeps only the recurring validity/tombstone apparatus. v3-fix: prune() orphan bug verified 2026-06-02.]**
- **4b (default-OFF, eval-gated):** kind-aware recency prior; turns on only if the Inc-1 harness shows stale-retrieval-rate/version-alignment improves with no relevance regression (bound a metric + threshold).
- **4c — CUT-CANDIDATE [v4-fix].** near-dup MinHash dedup **with a chunk-ownership/refcount model + tombstones** (closes the byte-exact G2a redundancy). The 7-lens review flags this as correctness-heavy machinery to save a cost a single user pays once: **defer indefinitely unless Inc-1-E shows real redundancy harm on this corpus.**

**Intra-increment order: 4a before 4c** (4c's safe-deletion reuses 4a's tombstone+refcount machinery); 4b is order-independent + eval-gated. **[v2-fix.]** **Depends on:** clean corpus (1–1.5, 3) + a **green Inc-1-E** (gates 4b + the heavy half of 4a; 4c is cut unless E justifies it). **[v4-fix.]**

### Increment 5 — Stage-0 team-share + shared tier  **[v2-fix: moved after Freshness]**

The Pillar-2 deliverable. `vecs publish` → immutable bundle (Chroma dir + FTS5 + manifest model/dim stamp, commit-SHA); `vecs pull` → verify model/dim (hard-fail on mismatch), read locally. **Shared-tier + redaction** (from Inc 2) land here, now that the bundle is the consumer. **Gated on Inc 4a** — the first team-wide publish must carry the freshness filter so we don't multiply stale-harm across N teammates (§4.2, G5 High). **[v2-fix: v2 placed Stage-0 before freshness.]** **Contingent (§7):** publish target — **S3 / GitHub release** (out-of-repo object transport, embedded-clean); **Git-LFS-into-the-code-repo is excluded** (~4 GB into repo history violates the "never inside the repo" constraint).

**[2026-06-17 — company-wide deployment topology (consumer = the team's *Claudes*, not human readers).]** A design pass (operator + Claude) reframed team-share around the actual consumer: agents calling vecs over MCP, not people reading a pulled index. That yields a **central-build** model with two consumption modes, plus a transport (cocone's MCP gate) that makes the company-wide case deployable. Captured here as design + open decisions; **not lockable** until the gate's capabilities, the embedder choice, and box ownership are pinned.

- **Central build.** One box (HQ Mac Studio, or a GPU VM on UK GCP) indexes the **canonical origin** of each product (not anyone's working tree) on the existing ~6h cadence. The shared index = the canonical product, identical for everyone; local uncommitted WIP stays in a small per-dev local index if wanted.
- **Two consumption modes — pick on one axis (`no local model` ⟺ `per-query network call`; you cannot have both):**
  - **Remote MCP (lighter for agents).** Full vecs (embed model + index) lives on the box behind the gate; each Claude calls `semantic_search` as a remote tool. **Zero local models on laptops**, always-fresh, one thing to maintain. Cost: every search is a network round-trip, and box-down = tool gone (grep/read fallback — degraded, not dead). ⇒ the **box must sit near the team** (UK), not Tokyo, or every query crosses the planet. This is the strategy-doc **Stage-1 shared HTTP MCP**, reachable directly via the gate.
  - **Pull-to-local (offline + zero query latency).** Box builds + publishes the index files (= the Inc 5 Stage-0 bundle); laptops `vecs pull` + atomic-swap and run a **tiny local query-embed model** (one short string per search). Survives box-down (stale, not dead); Tokyo location becomes irrelevant (only the bulk pull goes there, latency on it doesn't matter). Cost: the pull mechanism + a local model + version-pin. Effectively **requires local Qwen** — without it, query-embed falls back to Voyage per search and you've lost the "offline / nothing leaves an outside service" claim.
- **cocone MCP gate = the transport AND the access model.** If the gate does routing + auth, it is the clean front for the remote-MCP path. Key simplification (operator's): **multi-tenancy = per-instance, not in-vecs identity.** Run **one vecs instance per PRODUCT** (a group of related repos — e.g. livly = client-uk + server-uk + proto-uk — **not** strictly per-repo, which would kill cross-repo search within a product; reserve per-repo for genuinely standalone repos). Access control becomes **"which instance the gate lets you reach"** (connection-level), so vecs needs **no row-level identity / result-scoping** — that earlier gap is scrapped. Each silo is naturally isolated by construction.
- **Trap — do NOT duplicate the embed model per instance.** Each instance loads the model resident; N instances × 4B (~8 GB) = N×8 GB RAM, brutal at company scale. **Split embedder from index:** ONE shared resident embed service (one warm Qwen, or one Voyage client) + N lightweight index instances that call it. Gives many cheap silos + a single model footprint + automatic version-pin consistency. (0.6B makes per-instance copies cheap as a fallback, but centralizing is the right call.)
- **Embedder choice is orthogonal but interacts** (gated on **Inc 1.7 L2/L3** eval): remote-MCP works with either (box calls Voyage, or box runs Qwen locally); pull-to-local effectively needs **local Qwen** to be genuinely offline/internal. Either way the manifest model/dim stamp + the **query-model-must-equal-index-model** hard rule (already enforced — vecs drops mismatched collections) carries over; with a shared embedder it is automatic. Cold-start (load weights → RAM) is paid once per server lifetime if the model stays resident in the long-running MCP process, not per query.
- **Ownership / blast-radius flags.** One box now serves the whole company → box-down stalls everyone; this needs a real reliability/restart story (templated instances, health-checks, auto-restart — the no-support-team rule forbids hand-managed sprawl). If **cocone/HQ runs the gate + box as maintained infra**, the no-support-team rule is *satisfied* (a plus); if it is a Tokyo Studio the UK team cannot physically reboot, prefer a UK-GCP box the operator owns end-to-end.

**Open decisions (block locking):**
1. **Gate capabilities** — pure routing+auth proxy, or does it also pass user identity through / do per-tool authz? (The per-instance model only needs the former.) **Who runs + maintains the gate AND the box** — HQ infra (good: satisfies no-support-team) vs operator (the Tokyo-reboot problem).
2. **Embedder** — Voyage vs Qwen (and 4B vs 0.6B) → **Inc 1.7 L2/L3 result** (perf spike + shadow A/B + livly golden set, work mac M5 48GB).
3. **Box location / owner** — UK GCP (latency + control) vs HQ Studio.
4. **New build, not a config flip:** per-product instance templating + the **shared-embedder service** (the embedder/index split) is net-new work beyond Inc 5's publish/pull.

**Supersedes** the Mac-mini + Tailscale + shared-*writes* + identity-threaded sketch in `docs/features/shared-team-vecs-future.md` (that assumed teammates writing into one shared store; this is read-mostly central build behind a gate).

### Increment 6 — Transcript inversion (boldest)

A **per-session distiller** (distinct from Inc-2's triple extractor) distills durable facts/decisions per session into the fact store **with a provenance pointer to the raw jsonl**; demote raw to lower-ranked fallback (subsumes G2a/G2b). **Kill criterion (quantified):** if end-to-end recall on the temporal/coreference subset drops ≥ a set threshold vs raw-primary, do not flip the default. **Depends on:** Inc 2 (gate/provenance), a **green Inc-1-E** (metering + measurement). **[v4-fix: E built ahead.]**

**[v4-fix — session-extraction reframe deferred to here (do NOT build before Inc-1-E/A + Inc 2's gate).** The reframe's two good ideas survive: (1) **stage candidate facts to a jsonl, not the live store** (gradeable/revertible before promotion); (2) **git-anchored provenance** (commit-SHA per distilled fact). Dropped: the "piggyback ai-log for free" claim is **false** — ai-log's `_build_body_prompt` (`~/.claude/skills/ai-log/scripts/entry.py:625`) feeds `claude -p` only the first-N user prompts truncated to 200 chars (a Sonnet *retrospective*, not a triple producer over the full transcript), and coupling a team-KB deliverable to an unversioned personal hook is wrong when vecs already reads the same jsonl via `sessions_dirs`. Zero agent-felt payoff this quarter; it accumulates an ungradeable/unpriced stream before E/A exist.]**

### Increment 7 — Quality layer  **[v5: re-aimed at quality-max — a default-ON, E-gated stack, not a flag-off amplifier]**

**Goal flip (owner 2026-06-19): quality is the dominant axis; latency is not a constraint.** So the v4 framing ("Voyage reranker flag, off by default" for latency) is superseded. Build the full quality stack and **default-ON whatever passes Inc-1-E.** Quality-max ≠ add blind — it means run every viable lever as an independent A/B arm on the golden set (Inc-1-E: 46 cases, per-query-class paired bootstrap) and **ship the union of every arm that measures ≥ baseline per query class**, default-ON the winners, drop any arm that regresses a class. Still sequenced last (amplifier position unchanged — it amplifies a curated+fresh index); only its default and breadth change.

The stack — each an independent E arm:

- **Voyage reranker — full `rerank-2.5` (not the lite/perf variant), default-ON** over the fused top-30-50 before truncation. Single biggest lever (Anthropic Contextual Retrieval: −49% → −67% failure-rate reduction with a reranker — the largest single jump). v4's latency flag rationale is void under quality-max. E-gate per class (a cross-encoder can rarely reorder a correct top hit down — that is the one regression to watch).
- **Task-conditioned reranking.** Pass the B1 workflow (debug / review / implement) as the `rerank-2.5` `instructions` string — rerank toward call-sites for debug, definitions+tests for review. Free once the reranker fires; quality multiplier. Fuses Inc 7 × roadmap B1.
- **Contextual Retrieval** — Anthropic ≈100-token blurb per chunk → embed + FTS5, prompt-cache the doc body. The other half of the −49/−67 combo. Build-time LLM per chunk (metered via Inc-1-A).
- **voyage-context-3 contextual chunk embeddings for docs/sessions** (re-embed; +6.76% chunk / +2.40% doc over manual contextual retrieval). No contextual *code* model exists — code stays `voyage-code-3`. (Docs already on voyage-4 from Inc 1-B; context-3 is the chunk-context upgrade on top, gated on E beating the voyage-4 arm.)
- **MMR diversification — NEW [v5].** At result assembly, penalize a candidate near-identical to one already chosen (**λ high — near-duplicates only**, so it can never drop a relevant *unique* hit), so the bundle covers more of the answer surface. Query-time, **distinct from the cut Inc-4c** (that was index-time MinHash dedup). Bundle coverage is a B1 quality axis.
- **HyDE — NEW [v5], experiment-gated.** On the NL-intent path only: have the agent (already in the loop via MCP) generate a hypothetical code snippet that would answer the query, then embed THAT instead of the raw question — closes the NL→code vocabulary gap dense retrieval struggles with. Query-side, nothing in the store changes. Higher variance (a hallucinated hypothetical can pull retrieval toward a wrong region), so it is **strictly E-gated — keep only if golden-set delta ≥ 0 per class**; never default-ON blind.

**Parked — query router [v5].** A fast/slow router (high-confidence exact-symbol → BM25-only, NL-intent → vector + rerank) is a **latency/cost** optimization. Under quality-max with latency unconstrained it adds nothing on its own — you simply run the full expensive path on every query. It becomes a *quality* lever in exactly one contingent case: if Inc-1-E shows the expensive path (rerank) **hurts** a specific query class, route rerank away from that class. Build only on that E finding, not speculatively. (If the priority ever flips back to performance, the router is the top pick — it cuts the common exact-symbol path to BM25-only, ~10-100× faster, zero Voyage call.)

**Depends on:** a green Inc-1-E (the arbiter for every arm above). **Risks:** build-time LLM per chunk (Contextual Retrieval) + per-NL-query LLM call (HyDE) — both metered via Inc-1-A; context-3 re-embed cost (compute, not quality). **(voyage-3.5/voyage-4 already in Inc 1.)**

---

## 7. Consolidated open decisions

| Decision | Resolved in | Status / recommendation |
|---|---|---|
| `.md`: drop / reroute / defer | 1 | **RESOLVED — reroute now (all 3 projects)** |
| docs sources: auto-create vs multi-path | 1 | **RESOLVED — multi-path `docs_dirs` (auto-create rejected)** **[v2-fix: was stale-open]** |
| Extraction + contradiction-judge model | 1/2 | **RESOLVED (2026-06-03) — extraction = latest Sonnet (`claude-sonnet-4-6`); judge = latest Opus (`claude-opus-4-8`)** **[v4-fix]** |
| Inc 1.5 packaging: own feature vs 1-pipeline follow-up | 1.5 | **RESOLVED (2026-06-03) — own workflow-profile feature (`docs/features/kb-freshness-hotfix/`)** **[v4-fix]** |
| Inc 4c: cut vs keep | 4c | **RESOLVED-ish — CUT-CANDIDATE; keep only behind a green-E redundancy signal** **[v4-fix]** |
| Stage-0: own increment vs strategy-doc | 5 | **RESOLVED — own increment, after Freshness** |
| Facts: explicit vs default-blended | 2 | blend once gate exists |
| Promotion source-of-truth (curated `.md` vs triples) | 2 | curated `.md` authoritative on conflict (open) |
| Human/critic review gate vs auto-promote | 2 | review queue + LLM merge; human gate for shared tier (open) |
| Publish target | 5 | S3 or GitHub release; Git-LFS-into-repo excluded (open between the two) |
| Cost ceiling for fact population (gates Inc 2?) | 1/2 | open — bind a $ threshold from the Inc-1 metering report **[v2-fix]** |
| Recency prior default-off threshold | 4b | open |
| Transcript default-flip threshold | 6 | open |
| Inc 7 quality-stack: per-arm E gate | 7 | open — bind a per-query-class min delta; ship the union of arms ≥ threshold, default-ON winners, drop any class-regressor; HyDE strictly experiment-gated **[v5]** |

---

## 8. Risks, unknowns, and what is NOT verified

**Resolved across review rounds (do not re-introduce):** fact store empty/disabled; index-time exact dedup corruption hazard (→ 4c with ownership model); benchmark IDs corrected (§9, all three now confirmed to resolve); the dropped "Docs & chunking" increment (restored as Inc 3); validity filter pre-retrieval; voyage-3.5 needs re-embed (not a model flip); `.md` reroute needs an explicit `-code` sweep + `index_docs` multi-source surgery; per-collection RRF is a fusion refactor; Stage-0 gated behind freshness; "4.7×"→~2× (no equal-recall claim); MinHash ≠ byte-exact zero-loss; 80/24 are our estimates; α convention; vendor blog illustrative; Git-LFS-into-repo excluded. **[v4-fix — from the 7-lens direction review:** the `prune()` orphan fix is pulled to Inc 1.5a (active "agent retrieves lies" defect, ahead of the program, not buried in 4a); Inc-1-E/A moved ahead of the bets they gate (E is the precondition for 3/4b/6/7 + the heavy half of 4a); Inc 4c demoted to cut-candidate; the searcher `-docs` gate (1.5b) + model-flip trust signal (1.5c) are live-defect hotfixes; extraction model = latest Sonnet, judge = latest Opus; the session-extraction reframe is deferred to Inc 6 (keep only stage-to-jsonl + git-provenance; "piggyback ai-log for free" is false).]**

**Still not independently verified (caveats):** recency-decay magnitudes (event-log, won't transfer to code/doc — tie-breaker only); stale-code-harm pp (n=17 — direction load-bearing); extraction lossiness (version/category-sensitive; ~7 pt on PersonaMem-v2); Power-of-Noise / IGP (PDF-summarized, medium); prose-drift extraction cost **unmetered** (Inc-1 spike gates any cost claim for 2/6); monorepo scale vs the ~7M single-node ceiling **unvalidated** (gates Stage-1/2). **[v5]** Inc-7 per-arm quality gains are **unverified on this corpus until Inc-1-E runs**: rerank-2.5 ≥ baseline on *every* query class (cross-encoder can rarely reorder a correct top hit down), HyDE NL→code gain (higher variance — hallucinated hypotheticals), context-3 beating the voyage-4 arm, and MMR coverage not dropping unique hits — each ships only on a green per-class E number.

**Measurement plan:** stale-retrieval-rate / version-alignment (not just relevance); memory on LongMemEval + LoCoMo; chunking/transcript changes measured **end-to-end**.

---

## 9. References (audited twice)

No fabricated arXiv IDs. **All three benchmark IDs confirmed to resolve** in the v2 evidence audit (LongMemEval `2410.10813`; LoCoMo `2402.17753` = "Evaluating Very Long-Term Conversational Memory of LLM Agents"; BEAM `2510.27246` = "Beyond a Million Tokens…"); the prior wrong ID `2507.05257` (MemoryAgentBench) is removed. **[v1-fix.]**

**Curation:** Distracting Effect `2505.06914` (high; distractor-only, NQ). Less LLM More Documents `2510.02657` (high; coverage-only, G4). Zero-RAG `2511.00505` (high). HoH `2503.04800` (high). Byte-exact dedup `2605.09611` (high; **byte-exact only**). MinHash infra — Milvus 2.6 + LSHBloom `2411.04257` (qualitative only; no 80/24, no zero-quality).

**Freshness:** stale-code harmful `2605.14478` (direction high; n=17). Time-decay/recency `2509.19376` + temporal-rag (technique high; α weights the semantic term). Bi-temporal supersede (Zep) `2501.13956` (high). No-clustering 0.08 F1 `2509.19376` (topic-only; version-selection is our inference). Cursor Merkle incremental — engineerscodex/turbopuffer (high, eng reports).

**Condensation:** Contextual Retrieval −49/−67% — anthropic.com/news/contextual-retrieval (high). Small chunks ~2× (recall trades down) + ClusterSemanticChunker cheapest-best — trychroma.com chunking eval (high). RAPTOR `2401.18059` (high, conditional). Proposition/Dense-X `2312.06648` (high, low priority). IndexRAG `2603.16415` (medium, watch).

**Memory:** Mem0 `2504.19413` (high). Extraction lossy `2603.04814` (medium; non-uniform). Bi-temporal supersede `2501.13956` (high). Naive-accumulation — hindsight.vectorize.io (illustrative vendor blog; backed by the two papers). Sleep-time/async — Letta, Mem0 (high). Collaborative Memory `2505.18279` (high). Benchmarks — LongMemEval `2410.10813`, LoCoMo `2402.17753`, BEAM `2510.27246` (high, all confirmed).

---

## 10. How this runs through the workflow framework

Program parent. Each increment (and each Inc-1 sub-feature) is a feature via `docs/workflow-vecs-profile-v0.1.md`: Phase 1 acceptance → Phase 7 dry-run → build (TDD, commit per task, new tests per touched module) → Phase 4 multi-agent review → Phase 8 retro. Larger increments (1, 2, 4) split into independently-gated sub-features.

**Next step [v4-fix]:** Inc-1-pipeline (F+B+C) is **shipped (local commits; the live migrating reindex is the owner's manual step — runbook in `docs/vecs-direction-review-2026-06.md`)**. Build next, in order: **Inc 1.5a** (prune-orphan fix, `docs/features/kb-freshness-hotfix/acceptance.md`) → **Inc-1-E + thin Inc-1-A** (the green E number gates everything downstream) → **Inc 1.5b/1.5c** → then the owner runs the migrating reindex against a before/after E baseline → Inc 2 onward.
