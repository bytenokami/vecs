# vecs Roadmap

vecs is **code-aware agent memory**: index your code, your AI-agent sessions, and your docs; retrieve what's relevant; and flag what's gone stale. This is the platform view — where vecs goes beyond search. The prose-drift staleness feature's own deferrals live in `docs/features/prose-staleness-detector/v2-roadmap.md`.

**Team + frontier strategy** (retrieval-quality quick wins, the staged team-sharing path, and the MCP tool-surface decision, grounded in a verified 2026 SOTA sweep): `docs/vecs-platform-strategy-2026-05.md`. This roadmap covers retrieval *shapes*; the strategy doc covers *sharing* and the *agent-facing tool surface*.

Sketches, not contracts. Updated 2026-05-31.

## Guiding principles

- **Memory, not just search.** Chatbot RAG returns similar chunks. An agent doing multi-step work needs an *assembled bundle* — the entity plus its relationships, history, and provenance. vecs is moving search → memory.
- **Appropriate context, not maximum.** Bigger context windows don't fix context rot. The win is delivering the *right* bundle, and flagging stale context so the agent stops trusting it (prose-drift's job).
- **Contract-first.** Define the bundle the agent needs (exact fields + format) before picking a primitive. Don't choose storage by hype.
- **Embedded / derived only.** vecs runs zero servers — everything lives under `~/.vecs` (Chroma + SQLite). A new "shape" earns its place only if it can be derived locally and stored embedded. This constraint is load-bearing: the prose-drift review rejected adopting Graphiti precisely because it needs an external graph DB, while keeping the temporal *shape* it provides.

## Where vecs is now (shipped)

- **Hybrid retrieval** — Chroma vectors (Voyage `voyage-code-3` for code, `voyage-3` for prose) + SQLite FTS5 BM25, fused.
- **Indexing** — code (tree-sitter AST), AI sessions (Claude Code + Codex, auto-routed by cwd), docs.
- **prose-drift** — bi-temporal staleness detector: the first *structured* shape (temporal facts + provenance + a contract-style output, not chunk similarity). CLI + MCP. **stage-2 recall shipped** (2026-05-30): exact `chain_key` collisions + embedding-similarity + Opus contradiction-judge on MISS, catching cross-predicate/paraphrase drift.
- Embedded, zero servers; CLI (`vecs`) + MCP server.

## Knowledge shapes — vecs coverage

| Shape | Industry example | vecs |
|---|---|---|
| Vector similarity | Pinecone | ✅ Chroma + Voyage |
| Keyword | — | ✅ FTS5 BM25 (fused) |
| Temporal | Zep / Graphiti | ✅ prose-drift (bi-temporal facts) |
| Hierarchical / tree | PageIndex | ❌ docs flattened to chunks |
| Graph / relational | GraphRAG | ❌ none |
| Tabular | SAP / Dremio | n/a (code tool) |

## Track A — prose-drift feature v2

Recall + temporal + cost hardening for the staleness detector. Full list: `docs/features/prose-staleness-detector/v2-roadmap.md`. Headline item **stage-2 embedding-similarity + LLM contradiction-judge** on a `chain_key` miss — ✅ **shipped 2026-05-30** (Graphiti's per-edge method, no graph DB). Remaining: valid-time axis, multi-scope/actor provenance (team memory), write-time adjudication, SQLite migration, Sonnet-extraction + metering — see the strategy doc's Pillar 1b.

## Track C — team sharing

Make vecs a shared base for the engineering team's agents without a premature infra jump. Staged path (detail in `docs/vecs-platform-strategy-2026-05.md`, Pillar 2): **Stage 0** read-only published bundle (`vecs publish`/`vecs pull`, zero new ops, embeds once for the whole team) → **Stage 1** one shared vecs FastMCP over Streamable HTTP + token auth (carries both Chroma *and* the FTS5 sidecar) → **Stage 2** object-storage-backed store, only past the ~7M-embedding single-node ceiling. Rejected: NFS-mounted `~/.vecs` (SQLite corruption) and a Chroma-only server (strands BM25).

## Track B — platform: shapes & bundles

### B1 — Retrieval contract / bundle assembly (biggest)

Today `search()` returns ranked chunks (chatbot-shaped). Agents need bundles. For a code agent the bundle is concrete: **given a symbol → its definition + call sites + tests + the session decisions that touched it + relevant docs + recent diffs** — assembled, deduped, carrying provenance. Define the contract per workflow (debug vs review vs implement), then assemble from the collections vecs already holds. This is the search → memory pivot, and the precondition for vecs being a *development* tool rather than a search box.

### B2 — Code-relationship graph (infra-free GraphRAG)

Code is natively relational — imports, calls, type refs, defines/uses. vecs **already parses AST via tree-sitter**, so the graph is derivable for free and stores in the **same embedded SQLite** as BM25/cache. No external graph server — so the infra objection that ruled out Graphiti for staleness does not apply here. Unlocks "what breaks if I change X", "who calls Y", and dependency-aware bundles for B1. This is the apt GraphRAG-for-code shape, and the place where "relational knowledge vector search misses" actually pays off for a dev tool.

### B3 — Hierarchical / tree retrieval

Stop flattening docs into isolated chunks (the PageIndex critique). Keep the heading/section tree; retrieve sub-trees with their context. Double-duty: also attacks prose-drift's flattening recall hole *structurally* — a different lever than Track A's LLM-judge (extraction over structured sub-trees beats extraction over flat chunks).

### B4 — Rediscovery-tax instrumentation

Agents re-fetch and re-summarize the same information across runs — the headline waste in the agent-memory thesis. Measure it: track repeated retrievals / re-extractions and surface a context-budget signal. Turns "appropriate not maximum context" into a metric rather than a slogan. prose-drift's verdict cache already kills re-extraction of identical text; extend the idea to the retrieval path.

## Sequencing

Contract-first: **B1** defines the bundles agents actually need; **B2** and **B3** are the shapes that fill those bundles; **Track A** hardens the temporal shape already shipped; **B4** measures whether any of it reduces real waste. Each item earns its place by a concrete bundle need and must stay embedded/derived.

Industry framing — Pinecone Nexus (retrieval contracts carrying intent + provenance), PageIndex (hierarchy over chunks), Microsoft GraphRAG (relations), Zep/Graphiti (temporal memory) — corroborates the direction. vecs's differentiator is doing it all **embedded and code-native**. The staleness-specific SOTA analysis is in `docs/research/prose-drift-review-and-sota-2026-05-29.md`.
