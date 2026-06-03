Authored by Claude

# vecs platform strategy — frontier + team-shared (2026-05)

Owner-level strategy for the mandate: **vecs uses frontier coding tech, and is a shared knowledge + search base for the engineering team's agents — code *and* general info.** Supplements `vecs-roadmap.md` (which covers retrieval shapes) with the two dimensions it omits: team-sharing and the agent-facing tool surface.

Evidence base: a verified research sweep (audit of this repo + 2026 SOTA on code-RAG, agent memory, MCP tool design, shared-embedded deployment), adversarially fact-checked. Verifier corrections are applied inline below. Primary sources cited where load-bearing.

## Three pillars

1. **Frontier retrieval quality** — close the gap to 2026 SOTA (reranking, current embedding models, contextual chunks, a code graph).
2. **Team-shared** — go from single-machine `~/.vecs` to something the team's agents consume, without throwing away the zero-server principle prematurely.
3. **Agent-facing tool surface** — answer "separate tools to reduce data?" with evidence, and shape what each call returns.

---

## Pillar 1 — Frontier retrieval quality

vecs today: Chroma vectors (`voyage-code-3` for code, `voyage-3` for sessions/docs) + SQLite FTS5 BM25, fused via RRF (k=60, w_vector=1.0, w_bm25=0.6), tree-sitter AST chunking (C#/TS only), 2000-char result truncation, **no reranker**.

**Recalibration — AST chunking is not a retrieval-quality differentiator.** "Practical Code RAG at Scale" (arXiv 2510.20609, verbatim-verified): *"Simple line-based chunking matches syntax-aware splitting across budgets."* Keep AST chunking for clean, self-contained chunks and as the substrate for the code graph (B2) — but do **not** claim it improves recall/precision. The paper also confirms vecs's two-engine split is correct: BM25 wins code-to-code / exact-symbol (and is 10-100x faster), dense wins NL-to-code intent queries.

Ranked quick wins (high ROI, low effort):

1. **Reranker (highest ROI).** After RRF fusion, rerank fused top-30-50 with Voyage `rerank-2.5-lite` before truncating to `n_results`. Drop-in via the `voyageai.Client` vecs already holds. Anthropic Contextual Retrieval data: adding a reranker takes failure-rate reduction from 49% → 67% — the single largest jump. **Gate behind a flag** (cross-encoder rerank of 30-50 chunks adds ~200-800ms/query and a per-token cost — conflicts with the "fast local-feeling tool" value for latency-sensitive callers).
2. **Upgrade `voyage-3` → `voyage-3.5`** for sessions/docs (config.py:20-21). Identical price ($0.06/1M), +2.66%/+4.28% quality, 32K context. **Precondition:** confirm output dimension matches the existing collections — a dim change forces collection *recreation*, not just reindex. Reindex sessions+docs only; code stays on `voyage-code-3` (still SOTA).
3. **Contextual chunk embeddings (`voyage-context-3`)** for docs/sessions — embeds each chunk with full-document awareness, reducing the "chunk loses meaning in isolation" problem. Corrected magnitude: **+6.76% (chunk) / +2.40% (doc) over manual contextual retrieval** (the earlier "20.54%" was a transposed Jina-doc number). No contextual *code* model exists, so code stays on `voyage-code-3`.
4. **Verify the FTS5 tokenizer does word-level / identifier-aware splitting** (camelCase/snake_case/dotted). 2510.20609's BM25 win is specifically for word-level splitting; vecs's audit indicates a code-aware tokenizer already exists — confirm it, since this is the lever the paper actually points at.

Bigger bet (medium-term, also = roadmap B2):

5. **Lightweight repo dependency graph** (RepoGraph-style defs/refs/calls, derived from the tree-sitter AST already parsed, stored in the same embedded SQLite). Expand a retrieval hit into its callers/callees/imports — the structure-aware win chunking does *not* give. RepoGraph reports ~32.8% *relative* SWE-bench improvement. Scope as an optional expansion step over fused results, not a rewrite.

**Positioning:** frontier agents (Claude Code) dropped vector RAG for grep+read on freshness/precision grounds. vecs's defensible niche is exactly the gap grep leaves — **cross-repo + cross-transcript + cross-doc semantic recall + persistent session memory**. Position vecs as the semantic+memory layer *complementary* to the agent's own filesystem tools, not a replacement.

## Pillar 1b — Agent / team memory (prose-drift line)

- **stage-2 recall — ✅ SHIPPED (2026-05-30).** Embedding-similarity fallback on a `chain_key` MISS + ONE Opus contradiction-judge. Independently confirmed as the single highest-value memory upgrade (the universal 2026 cheap-retrieval → selective-LLM-judge pattern). See `features/prose-staleness-detector/stage2-recall-design.md`.
- **Multi-scope + actor-aware provenance on the fact store — biggest gap for genuine TEAM memory.** The sessions collection is already agent-tagged (`metadata.agent ∈ {claude_code, codex}`); the prose-fact store has `source_id` but no actor/agent/scope field. Add scope columns (user/agent/run/org-equivalent) + actor to `_new_row_metadata`. Zero new infra — just metadata fields. *Without this, vecs is single-user fact memory, not team memory.*
- **Valid-time axis** (second Snodgrass timeline) — catches "we *used to have* a BE dev" temporal contradictions a single timeline silently overwrites. This is what makes Zep beat Mem0 on temporal benchmarks.
- **Write-time adjudication + constrained read-out** (CUPMem KEEP/STALE/REPLACE/UNKNOWN; STALE benchmark arXiv 2605.06527): surface low-confidence/UNKNOWN rather than silently overwriting, and have the MCP tool *warn the agent at query time* when a fact is flagged stale (recognition ≠ application).
- **SQLite migration of the fact store** — natively expresses current-view + point-in-time (kills the `is_current`/`invalid_at=0` Chroma workarounds), enables the parked `--as-of` and compaction, matches the independently-validated Cloudflare Agent Memory design (SQLite supersession chains + explicit `supersedes_id` forward pointer).
- **Cost:** switch extraction default Opus→**Sonnet**, reserve Opus for the escalated judge; add call-count metering + `VECS_PROSE_DRIFT_MAX_CALLS_PER_DAY` *before* any team-scale claim (the ~$0.80/mo figure is UNMEASURED).
- **AVOID** adopting Graphiti/Zep/Mem0/Letta as dependencies — all need external graph/vector servers (Kuzu, the lightest, archived Oct 2025). Borrow the *method*, never the substrate. (Internal review reached this; broader 2026 evidence confirms it.)

**Known limitation introduced by stage-2:** the contradiction-judge runs cosine over historical `is_current` rows. If the embedding model ever changes, stored fact vectors drift out of the query's vector space and similarity silently degrades. Mitigation: pin the embedding model+version with the fact store; re-embed on model change. (Tracked in the prose-drift v2-roadmap.)

## Pillar 2 — Team sharing (the new track)

The load-bearing fact: vecs is **not** a pure-Chroma store — it's Chroma `PersistentClient` **+ a SQLite FTS5 BM25 sidecar** (one `.db`/collection, kept in lockstep), + manifests/Codex-routing state, all under `~/.vecs`, fronted by a per-user stdio FastMCP server. Every sharing option must carry **both** stores or it breaks BM25 fusion.

Blockers today (ranked): single-user `~/.vecs` paths; no multi-tenant isolation; no auth/RBAC; `fcntl` locks corrupt over NFS; no remote Chroma; no cost control/quota; process-scoped config cache (no cross-machine coherency).

**Staged path — preserve zero-server as long as it pays:**

- **Stage 0 — read-only bundle (now, ~zero new ops). RECOMMENDED FIRST.** Add `vecs publish` (one dev or CI builds the index, tarballs each project's Chroma collection dir + matching FTS5 `.db` + manifest into an immutable artifact versioned by source commit SHA, pushes to S3 / GitHub release / Git-LFS) and `vecs pull` (each teammate syncs into their own `~/.vecs` and reads locally). Safe concurrency (each reads its own copy), BM25 travels with the vectors, and the Voyage embedding cost is paid **once** for the whole team. **Must** stamp embedding model+dim in the manifest and have `vecs pull` verify it, or queries silently mismatch. Only cost: snapshot freshness.
- **Stage 1 — one shared vecs FastMCP over Streamable HTTP (when snapshot staleness hurts / need shared writes).** `mcp.run(transport="http")` + `StaticTokenVerifier` bearer tokens + `stateless_http=True`, owning a single `~/.vecs`, serving **both** Chroma and FTS5 locally. A *vecs* server, not a Chroma server (a Chroma server strands the lexical half). Keep stdio as the default local mode; HTTP is opt-in. Honest cost: a real service (process, token rotation, box holds the HNSW index in RAM). Start single-instance (the `fcntl` routing lock is same-host only).
- **Stage 2 — object-storage-backed store (only if index > ~7M embeddings / single-node RAM ceiling).** turbopuffer (Cursor's namespace-per-codebase pattern, native vector+BM25, ~$0.02/GB) or Chroma Cloud serverless. Explicit, deliberate exception to the embedded principle — retires the SQLite FTS5 sidecar for the host's native full-text. Gate on a concrete RAM/recall/latency trigger, not hype.

**Explicitly REJECT:** (a) mounting `~/.vecs` on NFS/SMB and pointing many MCPs at it — SQLite POSIX-lock + WAL-shm failures cause silent corruption; (b) a shared Chroma-only server — serves the vector half, strands BM25. (Note: Chroma *does* ship first-class Basic/token auth as of 2026 — the reason to prefer vecs-FastMCP is that it carries the FTS5 half, not an auth gap.)

## Pillar 3 — Agent-facing tool surface

**Decision: keep ONE `semantic_search` tool with the `collection` filter. Do NOT split into code-search vs general-info-search.** (Verified against Anthropic "Writing effective tools for AI agents" + "Effective context engineering.") Rationale:

- More tools — especially overlapping ones differing only by corpus — distract agents and create ambiguous tool-selection, measurably lowering accuracy. "If a human engineer can't definitively say which tool to use, an AI agent can't either."
- Splitting entry-point tools does **not** reduce per-call payload — payload is governed by result-side controls (top-k, format, truncation, pagination), not by tool count. So the split wouldn't even serve its stated goal.
- vecs's 8 tools sit in the healthy small-set range; Tool Search / deferred tool loading (shipped Nov 2025, `advanced-tool-use-2025-11-20`) makes tool *count* cheap context-wise — further removing any incentive to split.

The real lever for "less data per call" is **result shaping** (do these instead):

1. Add a `detail`/`response_format` param — **compact default** (header + ~300-500 char snippet + score; Anthropic measured ~3x token savings from a concise default) and `full` on request.
2. Replace the blunt 2000-char truncation marker with **actionable steering** ("result truncated — narrow the query or request detail=full / fetch by id").
3. Add **pagination/offset** so agents page instead of inflating `n_results`. (Implementation note: RRF fusion + Jaccard dedup happen before slicing, so the cursor must page over the post-fusion/post-dedup ordering deterministically.)
4. Add a **`get_chunk(id)` fetch-by-id tool** so compact results carrying a stable `file:line`/id have a just-in-time landing endpoint (the one place adding a tool is justified — it's not an overlapping search tool).
5. Type `collection` as `Literal["code","sessions","docs"]`, not `str | None`, so the MCP schema exposes a real enum.

(Current worst-case payload is already modest: 5 results × 2000 chars ≈ 2.5k tokens — so compaction is a nicety, not urgent. Sequence it accordingly.)

## Content coverage — "all things Livly"

The team's repos are on disk: ~31k files — **Go ~17,949 (57%)**, Python ~3,389, C# (Unity client), plus docs (Markdown + Notion HTML exports, ODS/CSV, etc.). Coverage map:

- **Go AST — not supported.** But per the Pillar-1 recalibration, AST chunking does *not* improve retrieval, so Go line-based fallback is acceptable for **search recall**. Go AST matters specifically for the **code graph (B2)** and bundle structure — so prioritize tree-sitter-go *with* B2, not before it.
- **Generated-code exclusion is the higher-impact, cheaper win:** aggressively `exclude_dirs` `obj/Debug/`, `LibraryOrigin/`, `.gocache/`, `node_modules/`, `.venv/`. This removes far more noise than AST adds signal.
- Python / YAML / Notion-HTML / ODS-CSV chunking are gaps; add per demand (tree-sitter-python and an HTML→Markdown pre-pass are the cheapest).
- **Monorepo scale (31k files) is unvalidated** — benchmark index time, memory, and search latency before declaring vecs production-ready for Livly, and check where the real index sits vs the ~7M-embedding single-node ceiling (gates the Pillar-2 stage transition).

---

## Recommended sequencing

Near-term, highest value-per-effort, moving all three pillars:

1. **MCP result-shaping** (Pillar 3) — `detail=compact` default + `get_chunk(id)` + `Literal` collection + truncation steering. Cheap, directly answers the tool question, improves every agent interaction.
2. **`voyage-3.5` upgrade + reranker behind a flag** (Pillar 1) — biggest quality jump for the effort; verify dim compatibility first.
3. **Stage-0 `vecs publish`/`vecs pull` bundle** (Pillar 2) — the literal "share with the team" mechanism, zero-ops, pays embedding cost once.
4. **Multi-scope/actor provenance on the fact store** (Pillar 1b) — the missing piece that makes prose-drift *team* memory.
5. Then the bigger bets: code graph / B2 (+ tree-sitter-go), B1 bundle assembly, valid-time axis, SQLite fact-store migration.

Each item stays embedded/derived until a concrete trigger forces Stage 1/2. Contract-first throughout.

## Open risks to retire with measurement (not assumption)

- Embedding-model versioning for the no-delete fact store (stage-2 cosine breaks on model change).
- Cost: ~$0.80/mo prose-drift and Opus-on-every-extraction are UNMEASURED — run the metering + accuracy spike before team-scale claims.
- Monorepo scale at 31k files untested on Chroma+Voyage.
- Reranker latency/cost vs the "fast local tool" value — measure before defaulting it on.
