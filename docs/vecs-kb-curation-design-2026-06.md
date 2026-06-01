# vecs Knowledge-Base Curation — End-State Design & Increment Program

**Date:** 2026-06-01
**Revision:** v2 (post Phase-4 multi-agent review). v1 was attacked by four independent thinking-on reviewers (internal-logic, evidence/SOTA audit, codebase-reality, architecture/sequencing); all returned REVISE. This revision folds in every confirmed finding. Material corrections from v1 are flagged inline as **[v1-fix]**.
**Status:** Approved direction; per-increment specs follow. Several scope items are **contingent** on the open decisions in §7 — those are tagged and must not be frozen into an `acceptance.md` until resolved.
**Pillars:** (1) frontier retrieval quality, (2) team-shared knowledge base, (3) agent-facing tool surface.
**Companion docs:** `docs/vecs-platform-strategy-2026-05.md` (platform strategy + SOTA), `docs/vecs-roadmap.md` (platform direction), `src/vecs/CLAUDE.md` (module context).

---

## 0. What this document is

The **end-state design** for vecs as a *frontier-quality, team-shared knowledge base for coding agents*, plus the **sequenced increment program** to get there. Grounded in (a) a verified investigation of the live repo and `~/.vecs/` store, and (b) a 2026 SOTA research pass, **re-audited** in Phase-4 review — citation scopes and magnitudes below reflect the audit, not the first-pass research.

### North star

Every decision is judged by one test: **does it make the coding agent a more effective engineer — better recall of the right context, less rediscovery, fewer wrong turns?**

### Hard constraints (non-negotiable)

- **Embedded, zero external servers.** Stores live in ChromaDB or SQLite/FTS5 under `~/.vecs/`. No graph DB, no standing service. (Borrow methods from Graphiti/Zep, never the substrate.)
- **Contract-first.** Config-schema and tool-surface changes are designed before code.
- **Appropriate context, not maximum.** Condense to high-signal; keep raw addressable as fallback.
- **Index storage never written inside the repo.** `~/.vecs/` only.

### Explicitly out of scope

The **repo-dependency / code-graph** bet (the largest standalone Pillar-1 lever — code-graph retrieval, ~32.8% SWE-bench in the strategy doc) is **not** part of this curation program. It is tracked in `docs/vecs-roadmap.md` (Track B2) / the strategy doc. This program improves *curation, freshness, facts, and sharing* — not the code graph. **[v1-fix: was implied complete under Pillar 1; now scoped out explicitly.]**

---

## 1. The end-state vision

**One line:** vecs becomes the *most up-to-date, condensed-but-not-lacking* knowledge base the engineering team's agents share — code, docs, sessions, and durable facts — where retrieval is fresh, version-aware, deduplicated, the team's best-curated knowledge is first-class and searchable with provenance, and the index is publishable to teammates.

**The shape ("condensed but not lacking"):** index **small, high-signal units** (tight chunks + curated/extracted facts) for precision → keep **raw transcripts/docs addressable** as a lossy fallback → **freshness-stamp and supersede** so stale content is filtered *before scoring* → **publish as an immutable bundle** for the team → **amplify with contextual retrieval + reranking last**, once the index is clean.

### End-state collection topology

| Collection | Role | Built from | Searched | Lifecycle |
|---|---|---|---|---|
| `<p>-code` | code recall | code files (extension-filtered, **no `.md`**) | default | git-driven incremental; `version_id` + validity; tombstoned |
| `<p>-docs` | doc/prose recall | `docs_dir` **+ multi-path per-repo docs + rerouted in-repo `.md`** (heading-chunked) | default | hash-incremental; `version_id` |
| `<p>-sessions` | raw transcript fallback | chat JSONL (lightly cleaned; later extract-and-link) | default, **down-weighted** (per-collection RRF weight) | append-aware; near-dup deduped |
| `<p>-prose-facts` | **first-class durable facts** | promoted curated `memory/*.md` + chat-extracted triples, gated by a write-time merge | default (after Increment 2) + explicit; **gets an FTS5 sidecar in Increment 1** so it fuses symmetrically | bi-temporal supersede; provenance + scope tier |

Per-chunk metadata contract (end-state): `version_id` (git SHA for code, revision/mtime for docs, session/run id for chat), `valid_from`/`valid_to`, `provenance` (source + actor/agent + scope tier), `kind` (static / versioned / event) for kind-aware recency. **[v1-fix: §1 asserted "sessions: lower weight" with no mechanism; the mechanism (per-collection RRF weight) is now a named Increment-1 deliverable, see §5.1.]**

---

## 2. Verified current state (2026-06-01)

All facts carry repo/live-store evidence.

**What is indexed.** Three collections per project (`code` / `sessions` / `docs`) across projects `bloomly`, `eric`, `livly`, discovered from `~/.vecs/config.yaml`. Code is matched by **file extension** (mandatory per `code_dir`, no default — `config.py:39-40`) within include/exclude dir rules; sessions by globbing `*.jsonl` in one `sessions_dir` per project plus Codex sessions routed by `cwd`; docs from **one** `docs_dir` per project — **only `livly` has one** (`config.py:49`); `bloomly`/`eric` have none. Live counts (as-of 2026-06-01): **12,823 code / 1,134 session / 1,580 doc** chunks, of which **livly = 11,950 / 1,025 / 1,580**. **[v1-fix: the "1,025 session" figure used elsewhere in v1 is livly-only; the cross-project total is 1,133–1,134 (minor live drift). Both now labeled.]** `chroma.sqlite3` ≈ **3.6 GB** (the `chromadb/` dir ≈ 4.2 GB). **[v1-fix: was "3.9 GB".]**

**Chunking.** Code = tree-sitter AST (C#/TS) else 200-line line-chunks (overlap 50); sessions = 10-message groups (overlap 2); docs = H1/H2 heading boundaries with paragraph fallback (min 30 chars). Refs: `config.py:24-27`, `ast_chunker.py`, `chunkers.py`, `doc_chunker.py`.

**Sessions are lightly cleaned, not verbatim.** `preprocess_session` (`chunkers.py:46-96`) parses turns, strips `<system-reminder>` blocks and base64 blobs, and reformats to `[role]: text`; embedding happens later in `_embed_and_store` (`indexer.py:385-458`). **[v1-fix: v1 said "embeds raw turns" and cited the parsing function as the embed site — both wrong; it is light cleaning at parse time, embedding elsewhere.]**

**Curation / dedup / freshness (today).**
- Incremental only: SHA-256 file-hash manifest skips unchanged files; sessions tracked by byte-offset + identity-hash.
- **Dedup is search-time only** — Jaccard 0.55 line-overlap (`searcher.py:57-79`). No index-time dedup; chunk ids are path-keyed `{source_key}:{chunk_index}` (`_make_chunk_id`, `indexer.py:348`) with **no content hash**.
- Fusion is RRF with **global** weights (k=60, w_vector=1.0, w_bm25=0.6; `searcher.py:82-114`) — **no per-collection weighting, no reranker, no recency/time-decay.**
- Stale-chunk cleanup runs after a successful embed for docs (unconditional, `indexer.py:1118`) and for **full** session re-index, but **not for incremental session appends** (`indexer.py:942-943`). Orphan chunks from a mid-batch crash are **self-healing**: a partially-embedded file is not in `fully_succeeded` (`indexer.py:604-607`) so the manifest doesn't record it; it is reprocessed and cleaned on the next reindex of that file. **[v1-fix: v1 implied permanent orphans; they persist only until the next reindex of that file.]**
- The "no content-addressable" comment at `indexer.py:505-506` describes the **BM25/FTS5 sidecar** sync, not the Chroma store; the Chroma `upsert` (`indexer.py:449`) is also content-blind. **[v1-fix: misattributed in v1.]**

**Fact store: write path exists but is OFF; no data.** vecs *has* a fact-write path (`indexer.py:968-1000`) that extracts triples from chat during indexing and writes them through `add_fact_with_state_machine` (`prose_drift.py:685-758`), an **exact `chain_key` (subject|predicate) INSERT/NOOP/SUPERSEDE merge** that flips `is_current` and never deletes. **But it is gated by `prose_drift_enabled` (default `False`, `config.py:54`), which is set on no project — the key is absent from `~/.vecs/config.yaml`.** Verified against the live store: ChromaDB lists only `bloomly-code/-sessions`, `eric-code/-sessions`, `livly-code/-docs/-sessions` — **no `*-prose-facts` collection exists, and zero triples have ever been written.** **[v1-fix — BLOCKER: v1 claimed "the fact store exists but is not searched." It does not exist in the live store. Populating it requires `prose_drift_enabled=True` + a metered extraction reindex (the unmetered cost §8 flags). This moves the facts work out of "quick wins" into Increment 2.]** Facts (when written) embed with voyage-3 via `_voyage_embed` (`prose_drift.py:488-491`; embed sites `:723`, `:743`); `:350-354` is the collection getter, not an embed site. The collection is created via raw `chromadb` with **no `_sync_bm25`** call → vector-only, no FTS5 sidecar.

**The memory ↔ vecs inversion (confirmed, refined).** The team's durable knowledge lives as curated markdown fact files under the agent memory dir. These are **never indexed** — `sessions_dir` globs `*.jsonl` only (`indexer.py:1024`). Meanwhile **raw chat transcripts are indexed (1,025 livly session chunks)**. So curated facts are not searchable and not shared, *and* the would-be fact store is disabled. vecs is single-machine (`~/.vecs/`) with a per-user stdio MCP server.

---

## 3. Gap map (verified + SOTA-backed)

Severity = impact on the north star.

| # | Gap | Sev | Evidence (code) | SOTA backing (see §9) |
|---|---|---|---|---|
| G1 | **Memory↔vecs disconnect** — curated durable facts not indexed; the fact-write path is disabled and the prose-facts collection does not exist | High | memory dirs hold `.md` only; `prose_drift_enabled` off everywhere; no `-prose-facts` collection in live store | Collaborative Memory (private→shared + provenance) |
| G2a | **Chat-transcript redundancy** — cumulative histories repeat prior turns | **Cost, not north-star-rated** | sessions embedded with light cleaning (`chunkers.py:46-96`) | byte-exact dedup ≈zero quality change (cost/latency win, *not* accuracy) |
| G2b | **Chat-transcript topical noise** — verbatim multi-turn dialogue competes with answer-bearing content | Med | raw turns in `-sessions` | distractor harm (analogy from NQ single-distractor setting, ~6–11 pts) |
| G3 | **`.md` indexed AS code** — prose competes with code | Med | all 8 livly `code_dirs` list `.md`; `indexer.py:696-699` | distractor harm (same analogy) |
| G4 | **Scattered docs** — only `docs_dir` indexed; per-repo `docs/` missed | Med | `docs_dir` single-valued; only livly has one | adding unique answer-bearing coverage helps ("Less LLM, More Documents") |
| G5 | **No freshness defense** — no version/build stamp, no recency, no pre-retrieval supersession filter, no tombstones; incremental appends never clean superseded chunks | High | `indexer.py:942-943`; no recency in RRF | stale code context *actively harmful*, worse than no retrieval; outdated docs worse than no retrieval (HoH) |
| G6 | **No index-time dedup** — near-identical chunks across branches/forks/appends coexist | Med | `searcher.py:57-79` post-hoc only; no content hash | MinHash-LSH now standard infra (qualitative) |
| G7 | **Fact store has no actor/scope/valid-time** — can't be *team* memory even when populated | Med | `prose_drift.py:669-684` has `is_current`, no actor/scope | bi-temporal valid-time + provenance |
| G8 | **No *semantic* promotion gate** — an exact `chain_key` merge exists, but no top-k semantic merge to catch paraphrase/cross-predicate dupes; bulk-indexing would surface confidently-wrong stale facts | Med | exact merge `prose_drift.py:685-758`; no top-k semantic step | Mem0 ADD/UPDATE/DELETE/NOOP against top-k; naive accumulation failure mode (illustrative) |
| G9 | **Chunk size not tuned for precision** | Low | `config.py:24` 200-line code; doc heading sections can be large | small chunks ≈2× precision at ≈equal recall; pair with parent/window fetch |

**[v1-fix: G2 was a single "High" gap; its evidence was cost-only or self-admittedly lossy. Split into G2a (cost) + G2b (quality), with the distractor evidence attached to the quality half. G8 reworded — a merge gate exists; the gap is the *semantic* top-k step.]**

---

## 4. Design principles from the research (audited)

Confidence and scope reflect the Phase-4 evidence audit. Several v1 magnitudes were overstated and are corrected here.

1. **Curate, don't hoard — but the lever is noise/staleness/redundancy, not size per se.** Stale, redundant, and topically-confusable content hurts; coverage of *answer-bearing* passages helps. **[v1-fix: v1's "corpus quality dominates size (high confidence, multiple sources)" overstated its citations — the Distracting Effect is about distractors, and "Less LLM, More Documents" argues coverage *helps*. Reframed; confidence medium.]** Backing: redundancy pruning (Zero-RAG, high), stale-harm (HoH, high), distractor-harm (Distracting Effect, high *for its NQ setting*; our `.md`/chat uses are analogies), coverage helps (Less LLM More Documents, high — used only for G4).
2. **Stale is worse than absent.** Stale *code* context actively redirects the model to obsolete state; outdated docs mislead rather than abstain. → version-stamp, **hard-filter superseded before scoring**, tombstone deletes. (Direction high; magnitudes from small samples — §8.)
3. **Extract beats verbatim for chat — but extraction is lossy.** ~35:1 compression drops temporal markers, coreference, ephemeral detail (33–35 pt gaps on some categories; **only ~7 pt on PersonaMem-v2** — the penalty is *not* uniform). → promote extracted facts **and keep the raw transcript addressable**. Never delete the source. (Medium.)
4. **Dedup is a cost/latency win, not a quality claim — and only byte-exact is proven loss-free.** Byte-exact dedup shows zero measured quality change (inference-time RAG study). **Near-duplicate (MinHash) dedup is NOT covered by that zero-loss result** — near-dup is exactly where quality risk lives; source it separately and gate it. The ~80%/≈24% reduction figures are **our corpus estimates**, not from the cited dedup-infra sources. **[v1-fix: v1 attributed zero-loss to MinHash too and sourced the 80/24 figures to the infra papers — both overreaches.]**
5. **"Condensed but not lacking" base case is cheap.** Small chunks (~200–400 tokens) ≈**2× precision** at roughly equal recall (Chroma eval: 3.6→7.0, "doubled" — **not** 4.7×) + parent/sentence-window fetch. The cheap **ClusterSemanticChunker** (no LLM call, reuses the embedding model) posts the eval's *highest* precision; only the **LLM-prompted** chunker isn't worth its cost. **[v1-fix: "4.7×" was fabricated (~2.4× inflated); the "skip semantic chunkers" claim contradicted the source's own headline method — both corrected.]**
6. **Contextual Retrieval is the highest-ROI quality upgrade — but it amplifies whatever is indexed.** ~100-token LLM context blurb per chunk before *both* the embedding and BM25 index (−49% retrieval failures; −67% with a reranker). Sequence it **after** curation. (High — verbatim from Anthropic.)
7. **Facts: supersede, don't delete; make version first-class.** Bi-temporal valid-time + `invalidated_by` + ingestion timestamp. Borrow the method, not the graph substrate. (High.)
8. **Recency is a conditional prior, never dominant.** Kind-aware exponential half-life as a post-RRF re-score, **eval-gated**. Note the source weights α on the *semantic* term (so its "α≥0.9 degrades" means *too little* recency); when we wire α to the *recency* term, keep it low — recency as tie-breaker. The reported effect is a drop to ~0.667, not a "collapse," and is on time-stamped event logs that **won't transfer** to code/doc retrieval. **[v1-fix: α convention was inverted and "collapse" overstated.]**
9. **No unsupervised clustering for canonical-version/topic selection** — topic clustering scored ≈0.08 F1. Use deterministic git/version signals + the existing selective Opus contradiction-judge. (Topic-clustering result is real; applying it to *version* selection is our inference.)
10. **Team memory needs provenance + tiers + redaction + a promotion gate.** Private vs shared tiers; attach who/which-agent/which-session/when; redact on promotion to shared; gate with a top-k LLM merge; run consolidation **offline**. (High — Mem0, Collaborative Memory; the "naive accumulation" failure mode is from a vendor blog, **illustrative**, backed by the peer-reviewed pair.) **[v1-fix: vendor blog was load-bearing at High; downgraded.]**
11. **Measure version-alignment, not just relevance.** A system can score faithfulness 0.95 and still be wrong on a stale chunk. Track stale-retrieval-rate; for memory use LongMemEval / LoCoMo. **The harness for this is built in Increment 1** (§6) — v1 had no owner for it. (High.)

---

## 5. Target architecture (concrete)

### 5.1 Retrieval pipeline (end-state)

```
query
  → PRE-RETRIEVAL validity filter: metadata predicate (Chroma where / FTS5 clause)
      · facts: is_current = True
      · code/docs: not superseded / valid_to unset-or-future
      · drops stale chunks BEFORE they consume candidate slots        [Inc 4; facts predicate in Inc 1]
  → embed (per-collection model) + BM25 (FTS5, incl. facts sidecar)   [facts FTS5 in Inc 1]
  → RRF fuse across code/docs/sessions/facts with PER-COLLECTION weights
      (sessions down-weighted; facts weight defined now they have FTS5) [Inc 1 contract]
  → kind-aware recency prior (flag, default-OFF, eval-gated, α low)    [Inc 4]
  → optional rerank (voyage rerank-2.5-lite, flag)                     [Inc 6]
  → result-shaping (detail levels, get_chunk just-in-time)             [later / Pillar 3]
```

**[v1-fix: v1 drew the validity filter *after* RRF fusion, contradicting principle 2 ("filter before scoring"). Moved to a pre-retrieval metadata predicate so stale chunks never enter fusion. Also: "expired" is now defined (valid_to in the past); per-collection semantics are specified rather than one generic filter.]**

### 5.2 Stores

- **code / docs / sessions** — ChromaDB + FTS5 sidecars. Add `version_id` + validity metadata and a **content-hash embedding cache** so reindex re-embeds only changed chunks (cuts Voyage cost; de-risks the Stage-0 bundle). **[Inc 1.]**
- **facts** — `<p>-prose-facts`, first-class: gets an **FTS5 sidecar** (Inc 1) so it fuses symmetrically; bi-temporal; provenance + scope tier; populated and gated in Inc 2.

### 5.3 Memory bridge (end-state)

```
source (curated memory/*.md | chat transcript)
  → extract / parse                                  (lossy; keep source pointer)
  → exact chain_key merge (EXISTS: prose_drift.py)   INSERT | NOOP | SUPERSEDE
  → + semantic top-k merge gate (NEW)                ADD | UPDATE | SUPERSEDE | NOOP
  → attach provenance + scope tier
  → (shared tier) redact personal/sensitive          [paired with Stage-0 consumer, Inc 3]
  → write to <p>-prose-facts (searchable)            + keep raw transcript addressable
  (offline / scheduled, not in the agent hot path)
```

### 5.4 Team-sharing (Stage-0 bundle)

Publish the index once as an **immutable bundle** (Chroma dir + FTS5 `.db` + a manifest stamping **embedding model + dim**, versioned by commit SHA); teammates `vecs pull` and read locally. **Hard requirement = the manifest model/dim stamp (cheap, available in Inc 1) + the content-hash cache.** Per-chunk `version_id` is **not** a precondition — it is a later enhancement for *incremental* bundle rebuilds. **[v1-fix: v1 gated the whole bundle on Increment-4 per-chunk version stamps; that over-coupling is removed.]** Bundles are re-published as later increments land (freshness, quality). Stage-1 shared HTTP MCP and Stage-2 object-store remain later stages (strategy doc).

---

## 6. Increment program

Seven increments, dependency- and ROI-ordered. Each is a feature run through the workflow profile (`docs/features/<name>/` with `acceptance.md`, dry-run, Phase-4 review, retro). Larger increments (2, 4) **decompose into sub-features** with their own acceptance — they are not single features. **[v1-fix: v1's "each is one feature" framing was violated by increments bundling ~8 deliverables.]** Scope items marked **(contingent: §7)** depend on an unresolved decision and must not be frozen into acceptance until resolved.

### Increment 1 — Foundations & no-regret wins

**Goal:** the cheap, safe, no-hot-path-LLM base everything else needs.
**Scope (in):**
- **Metering spike** — instrument extraction/judge cost (default extraction model **Sonnet**, with `MAX_CALLS_PER_DAY` cap). This is the **gate for all later LLM work (Inc 2, 5, 6).**
- **voyage-3 → voyage-3.5** for docs/sessions after a dim-compat check (code stays voyage-code-3). Cheap recall lift, independent of corpus cleanliness.
- **`version_id` stamp + content-hash embedding cache** — prereqs that make every later reindex cheap and enable the Stage-0 bundle.
- **RRF per-collection weight contract + facts FTS5 sidecar** — design the per-collection weighting (sessions down-weighted) and give `-prose-facts` a BM25 sidecar so it fuses symmetrically (satisfies the `src/vecs/CLAUDE.md` "BM25 in lockstep" invariant).
- **Measurement-harness seed** — a `stale-retrieval-rate` metric + a small local eval-set scaffold, so Inc 4/5 gates can actually fire.
- **`.md` → docs reroute (decided: now)** — remove `.md` from all `code_dirs`, and route in-repo `.md` into the docs collection (heading-chunked), including for `bloomly`/`eric` which have no `docs_dir` (auto-create or multi-path). **No coverage gap.**

**Scope (out):** populating/searching facts (→ Inc 2, needs metered extraction); near-dup dedup (→ Inc 4, needs ownership model); recency, tombstones, supersession filter (→ Inc 4).
**Acceptance outline:** metering report exists with a per-extraction cost + cap; docs/sessions on voyage-3.5 with a recorded dim check; chunks carry `version_id`; re-embedding a project re-embeds only changed chunks; `-prose-facts` has an FTS5 sidecar and per-collection RRF weights are configurable; `.md` absent from `-code` and present in `-docs` for all three projects after reindex; tests per touched module.
**Risks (corrected):** rerouting `.md`→docs **does** re-embed those files as docs (cheap, voyage-3); dropping `.md` does **not** re-embed remaining code (`indexer.py:799-815`). The genuine risks are the dim-compat check and the config-schema migration for multi-path docs — not a full reindex. **[v1-fix: v1's "reindex required" overstated cost and masked the real risks.]**
**Files:** `~/.vecs/config.yaml` (remove `.md` from 8 livly code_dirs; add docs sources) **[v1-fix: not a `config.py` "extension default" — extensions are mandatory per code_dir]**, `config.py` (multi-path docs schema), `indexer.py`, `searcher.py`, `bm25_index.py`, `clients.py` (model), tests.

### Increment 2 — Memory bridge (rest of A)

**Goal:** populate and surface the team's durable knowledge.
**Scope:** **enable extraction** (`prose_drift_enabled`, metered/capped per Inc 1) to **populate `-prose-facts`**; promote curated `memory/*.md` + chat-extracted triples; add a **semantic top-k merge gate** on top of the existing exact `chain_key` state machine; **provenance** (source/session, actor/agent, ts) + **scope tier** (private/shared); wire facts into search and **blend into default** (now that facts exist, are gated, and have an FTS5 sidecar). Keep raw transcript addressable via provenance pointer. Offline/scheduled promotion (not hot path).
**Decompose:** 2a single-machine fact bridge (populate + promote + gate + provenance + search); shared-tier + redaction moves to **Increment 3** (its only consumer is the bundle). **[v1-fix: redaction-to-shared had no destination in v1; now paired with Stage-0.]**
**Open decisions (contingent: §7):** promotion source-of-truth (curated `.md` authoritative on conflict?); human/critic review gate vs auto-promote.
**Depends on:** Inc 1 (metering gate, facts FTS5, RRF weights).
**Risks:** extraction lossy (keep raw); a bad fact pollutes team recall unless the gate + review are real; embedding-model pinning (cosine breaks on model change).
**Measurement:** LongMemEval-style (knowledge-update, temporal, abstention) on the Inc-1 local set.

### Increment 3 — Stage-0 team-share bundle + shared tier (decided: A)

**Goal:** the literal Pillar-2 deliverable — make the index team-shared.
**Scope:** `vecs publish` → immutable bundle (Chroma dir + FTS5 + manifest with **embedding model+dim** stamp, commit-SHA versioned); `vecs pull` → verify manifest model/dim, read locally. Decoupled from per-chunk `version_id` (§5.4). **Shared-tier + redaction** (from Inc 2) lands here, now that the bundle is its consumer.
**Depends on:** Inc 1 (manifest stamp + content-hash cache), Inc 2a (facts to include).
**Open decisions (contingent: §7):** publish target (S3 / GitHub release / Git-LFS); redaction policy for the shared tier.
**Risks:** bundle size; pull-time model/dim mismatch must hard-fail. Re-publish cadence as later increments improve the index.

### Increment 4 — Freshness

**Goal:** defend the strongest empirical finding (stale = harmful). **Split by confidence:**
- **4a (default-ON, high-confidence):** `valid_from`/`valid_to` + **pre-retrieval supersession/validity hard-filter**; tombstones on file/doc delete; **fix the `is_full=False` append-cleanup** (`indexer.py:942-943`); git-driven incremental reindex (Cursor Merkle pattern; `version_id` + cache from Inc 1).
- **4b (default-OFF, eval-gated):** kind-aware recency prior — ships off; turns on only when the Inc-1 harness shows stale-retrieval-rate / version-alignment improves with **no relevance regression**. Bind a metric + threshold (mirrors Inc-5's kill criterion).
- **4c:** near-dup **MinHash** dedup **with a chunk-ownership/refcount model + tombstones** (so deleting a shared near-dup can't corrupt another owner's recall). **[v1-fix: exact dedup was a verified corruption hazard in Inc 1 (breaks the `succeeded==expected` manifest invariant, `indexer.py:604-611`; no content-hash store exists); near-dup work moved here with the ownership model it requires.]**
**Depends on:** clean corpus (1, 2), measurement harness (1).
**Risks:** over-weighting recency (keep α low; §4.8); no unsupervised clustering for canonical-version (§4.9).

### Increment 5 — Transcript inversion (boldest)

**Goal:** stop indexing the noisiest corpus verbatim as primary.
**Scope:** add a **per-session distiller** (distinct from Inc-2's triple extractor) that distills durable facts/decisions per session into the fact store **with a provenance pointer to the raw jsonl**; demote raw to lower-ranked/flagged fallback.
**Depends on:** Inc 2 (gate/provenance), Inc 1 (metering + measurement).
**Kill criterion (quantified):** if end-to-end recall on the temporal/coreference subset drops ≥ a set threshold vs raw-primary, do **not** flip the default — keep raw primary. **[v1-fix: "drops materially" was unquantified; and the distiller is a *new* component, not Inc-2's extractor — so Inc 2 only partially de-risks 5.]**
**Risks:** boldest behavior change; only safe if raw stays addressable; added LLM cost (gated by Inc-1 metering).

### Increment 6 — Quality layer

**Goal:** frontier retrieval quality on a now-clean index.
**Scope:** Anthropic **Contextual Retrieval** (≈100-token blurb per chunk → embed + FTS5, prompt-cache the doc body) + **Voyage reranker** (rerank-2.5-lite over fused top-30–50, flag, off by default). (voyage-3.5 already shipped in Inc 1.)
**Depends on:** curation (1, 2) + freshness (4) — it amplifies whatever is indexed.
**Risks:** build-time LLM per chunk (cost, gated by Inc-1 metering); reranker ~200–800 ms/query (flag, off by default); worth it above ~200k tokens.

---

## 7. Consolidated open decisions

| Decision | Resolved in | Status / recommendation |
|---|---|---|
| `.md`: drop / reroute / defer | 1 | **RESOLVED — reroute now** (drop + route to docs, all 3 projects) |
| Stage-0: own increment vs strategy-doc | 3 | **RESOLVED — A, own increment (Inc 3)** |
| Facts: explicit vs default-blended | 2 | blend once gate exists (Inc 2) |
| Promotion source-of-truth (curated `.md` vs chat triples) | 2 | curated `.md` authoritative on conflict (open) |
| Human/critic review gate vs auto-promote | 2 | staged review queue + LLM merge; human/critic gate for shared tier (open) |
| Publish target (S3 / GH release / Git-LFS) | 3 | open |
| Redaction policy for shared tier | 3 | open |
| Recency prior default-off threshold | 4b | open — bind metric+threshold from Inc-1 harness |
| `docs_dir` auto-create vs multi-path for `bloomly`/`eric` `.md` | 1 | open (small) |
| Transcript default-flip threshold | 5 | open — quantified kill criterion |

---

## 8. Risks, unknowns, and what is NOT verified

**Resolved-in-review (were v1 defects):** fact store empty/disabled (premise corrected, §2); index-time exact dedup is a corruption hazard (moved to Inc 4 with ownership model); benchmark IDs fixed (§9); `.md`-drop coverage gap closed by reroute-now; team-sharing transport now an increment (Inc 3); validity-filter moved pre-retrieval; voyage-3.5 and version_id+cache pulled forward.

**Citation corrections folded in (do not re-introduce v1's numbers):** "4.7× chunking precision" → ~2× (3.6→7.0); semantic-chunker "not worth it" → only the LLM-prompted one (ClusterSemanticChunker is cheap + best); byte-exact zero-loss does **not** extend to MinHash; the ~80%/24% dedup figures are **our estimates**, not from the cited infra sources; "corpus quality dominates size" downgraded (its cited sources don't support dominance); recency α convention is inverted in the source and "collapse" was a ~0.667 drop.

**Still not independently verified (treat as caveats):**
- Recency-decay magnitudes are from time-stamped *event logs* — technique high-confidence, magnitude **won't transfer** to code/doc; use as a tie-breaker only.
- Stale-code-harm percentages (+76–88 pp) are from **n=17** samples — direction load-bearing, not the exact pp.
- Extraction lossiness (33–35 pt; ~7 pt on PersonaMem-v2) and vendor memory leaderboard scores (66–72% peer-reviewed vs 92%+ vendor) are version/method-sensitive.
- "Power of Noise" / IGP figures were PDF-summarized, not line-verified (medium).
- prose-drift extraction cost is **unmetered** — the Inc-1 metering spike gates any team-scale/cost claim for Inc 2/5/6.
- Monorepo scale vs the ~7M-embedding single-node ceiling is **unvalidated** — gates whether sharing stays embedded (Stage-0/1) or needs object-store (Stage-2).
- Replacement benchmark IDs (LoCoMo `2402.17753`, BEAM `2510.27246`) came from reviewer convergence; confirm at edit-time of any text that cites them.

**Measurement plan:** track stale-retrieval-rate / version-alignment (not just relevance/faithfulness); memory bridge on LongMemEval + LoCoMo; chunking/transcript changes measured **end-to-end**, since retrieval-precision gains don't reliably translate to better answers.

---

## 9. References (audited)

Confidence + scope reflect the Phase-4 audit. No fabricated arXiv IDs were found in v1; the defects were one wrong ID and several over-broad scopes, fixed here.

**Curation:** Distracting Effect — arXiv 2505.06914 (high; *distractor harm only*, NQ setting). Less LLM, More Documents — arXiv 2510.02657 (high; *coverage helps only*, used for G4). Zero-RAG redundancy pruning — arXiv 2511.00505 (high). HoH outdated-worse-than-no-retrieval — arXiv 2503.04800 (high). Byte-exact dedup zero-loss — arXiv 2605.09611 (high; **byte-exact only**, inference-time RAG). MinHash-LSH infra — Milvus 2.6 + LSHBloom arXiv 2411.04257 (qualitative "standard infra" only; **no** 80/24 figures, **no** zero-quality claim).

**Freshness:** stale-code harmful — arXiv 2605.14478 (direction high; n=17). Time-decay / kind-aware recency — arXiv 2509.19376 + temporal-rag (technique high; α weights the *semantic* term in source). Bi-temporal supersede (Zep) — arXiv 2501.13956 (high). No unsupervised clustering 0.08 F1 — arXiv 2509.19376 (topic-clustering only; version-selection is our inference). Cursor Merkle incremental reindex — engineerscodex / turbopuffer (high, eng reports).

**Condensation:** Contextual Retrieval −49%/−67% — anthropic.com/news/contextual-retrieval (high, verbatim). Small chunks ~2× precision + ClusterSemanticChunker cheapest-best — trychroma.com/research/evaluating-chunking (high). RAPTOR — arXiv 2401.18059 (high, conditional). Proposition/Dense-X — arXiv 2312.06648 (high, low priority). IndexRAG — arXiv 2603.16415 (medium, watch).

**Memory:** Mem0 extract + ADD/UPDATE/DELETE/NOOP + top-k — arXiv 2504.19413 (high). Extraction lossy — arXiv 2603.04814 (medium; penalty non-uniform). Bi-temporal supersede — arXiv 2501.13956 (high). Naive-accumulation failure mode — hindsight.vectorize.io 2026-05 (**illustrative vendor blog**; backed by the two papers above). Sleep-time/async consolidation — Letta, Mem0 (high). Collaborative Memory (private→shared, provenance, redaction) — arXiv 2505.18279 (high). Benchmarks — **LongMemEval arXiv 2410.10813, LoCoMo arXiv 2402.17753, BEAM arXiv 2510.27246** (high). **[v1-fix — BLOCKER: v1 cited `2507.05257` for LoCoMo/BEAM; that ID is MemoryAgentBench, a different paper. Corrected.]**

---

## 10. How this runs through the workflow framework

This document is the **program parent**. Each increment is a feature executed via `docs/workflow-vecs-profile-v0.1.md`: Phase 1 acceptance (`docs/features/<increment>/acceptance.md`, checklist) → Phase 7 dry-run (smallest real subtask; pipeline-pass + review-loop-satisfied) → build (TDD, commit per task, new tests per touched module) → Phase 4 multi-agent review (architect → critical-sinker → reviewer) → Phase 8 retro (`gaps.md`). Larger increments (2, 4) split into sub-features, each with its own acceptance.

**Next step:** spec **Increment 1 (Foundations & no-regret wins)** — resolve its small open items (§7: docs-dir auto-create vs multi-path), then write its `acceptance.md` and run the dry-run.
