# Session Handoff — 2026-05-18

Handoff for the next Claude Code session picking up vecs workflow design work. Self-contained; read this first, then the cited files.

## Session goal (completed)

Generalize a senior peer's AI-driven feature workflow into a base framework + project-specific profile + worked-example feature design, all using a manager-pattern dogfood: the manager (this session) delegates research and authoring to subagents and composes the deliverables.

Three deliverables were the brief. All landed and were signed off by the user.

## Deliverables (signed, uncommitted)

| Path | Step | What it is |
|---|---|---|
| `docs/workflow-framework-v0.1.md` | Step 1 | Project-agnostic base. 9 phases (0 Bootstrap → 8 Retrospective). Typed extension-point slots per phase. `enabled: bool` + `cadence` per phase. Composition stance documented. ~230 lines. |
| `docs/workflow-vecs-profile-v0.1.md` | Step 2 | vecs-specific profile filling every required base slot. 0 open gaps (4 → 0 after iteration). Roster maps roles to real Claude Code subagent types. ~180 lines. |
| `docs/features/context-staleness-detector-design-v1.md` | Step 3 | Feature design walking the profile's phases. 6-pass reviewer loop terminated at `verdict: ship`. ~245 lines. |

## Supporting stubs (uncommitted)

| Path | Purpose |
|---|---|
| `src/vecs/CLAUDE.md` | Phase 2 root context doc; module map, invariants. |
| `scripts/check_acceptance.py` | Phase 3 acceptance adapter (interactive + `--non-interactive` modes). |
| `scripts/register_self.py` | Phase 0 helper — registers vecs as a vecs project (idempotent edit of `~/.vecs/config.yaml`). |
| `docs/templates/retro.md` | Phase 8 retrospective template. |
| `.claude/prompts/critical-sinker.md` | Phase 6 standing-skeptic prompt file. |

## Key design decisions (the thin trail)

These choices shaped every downstream output. Don't relitigate without reason.

- **Composition, not inheritance.** Base = thin slots + intent; profile fills slots. New project = new profile, not new subclass.
- **Mindset is stance, not phase.** Dropped from phase list per user; framework describes steps, not team adoption posture.
- **No iteration cap on loops.** Peer's "7 lucky" was a joke. Loops terminate on reviewer-satisfied OR no-progress.
- **No budget slot.** User on company plan.
- **9 phases**, numbered 0–8. Phase 0 Bootstrap, Phase 8 Retrospective are production-additions to the original 7-point seed.
- **Typed schema slots**, not prose. Closed enums where universally applicable; `"custom:<id>"` extension on enums likely to vary.
- **`enabled: bool` per phase**, `cadence` enum per phase. Lets profiles disable phases that don't apply.
- **Roster real-checked against Claude Code subagent types.** No invented agent types. Critical-sinker has no native backing → `general-purpose` + prompt file.
- **Architect and Planner/Lead collapsed in vecs profile.** Acceptable at solo scale; split later if parallel features land.
- **Manager dispatch as routing authority.** Vecs profile picks `manager-dispatch` over static-map or lead-agent.
- **Gaps must close, not just be crossed out.** User pushed back hard on gap-inflation; profile and feature-design both reflect this.

## Manager pattern (what to do, what not to do)

- Delegate before reading. For any task touching >200 lines / >2 files / multi-step verification, dispatch a subagent.
- Parallel where possible. Independent investigations → one message with multiple `Agent` tool calls.
- Spot-check 1–2 cited paths after each subagent returns; do not silently trust.
- Subagents brief includes: goal, scope boundary, expected output shape, word cap.
- Reviewer loop fires per profile `loop_kill_criteria`. Reviewer's verdict line is the falsifiable signal; do not declare ship without it.

## Open decisions (next session picks one)

1. **Commit the work.** Single commit, per-step splits, or per-file splits. Nothing committed yet.
2. **Implement `context-staleness-detector` feature** from `docs/features/context-staleness-detector-design-v1.md`. Phase 7 dry-run subtask is `resolve_baseline_sha(doc_path: Path, repo_root: Path) -> str | None`, then full Phase 5 test matrix, then CLI/MCP wiring.
3. **Author Phase 1 acceptance** at `docs/features/context-staleness-detector/acceptance.md` (currently inline in design). Required before `scripts/check_acceptance.py` can run against the feature.
4. **Run `scripts/register_self.py`** to register vecs as a vecs project in `~/.vecs/config.yaml`. Closes the last operator-action gap from Step 2.
5. **Scope a follow-up feature** that adds a profile-validator / dispatcher CLI (`vecs workflow validate` / `vecs workflow run`), which would turn the framework from documentation-only to automatable.
6. **Iterate on any of the three signed docs** if a real defect surfaces.
7. **Research thread (highest priority for the next session).** See "Open research thread" below.

## Open research thread — split vecs into code-vecs and prose-vecs?

**Trigger.** Late in this session the user observed that the `context-staleness-detector` feature only catches code-vs-doc drift via git. Out-of-repo facts (team composition, decisions made in chats, external-system state) never trigger drift because nothing in the repo's git history changes when they go stale. Example given: doc says "we have no BE dev", team hires a BE dev, repo unchanged, detector silent forever.

**Proposed split.**

- **Code-vecs.** What vecs is today, plus the staleness detector. Indexes code, runs hybrid semantic + BM25 search, detects code-doc drift via git commit-sha-tag comparison.
- **Prose-vecs.** New, separate. Indexes prose context — chats (Claude Code + Codex sessions, already partially handled), team notes, decisions, anything that lives outside source files. Different model for staleness: not git-driven. Different retrieval semantics: synthesis-oriented, not just chunk-return.

Open question for the search interface: does an agent query both halves and get a fused result, or does the agent pick which half to query? Same question for the MCP tool surface — one `semantic_search` with a `corpus` arg vs two distinct tools (`search_code`, `search_prose`).

**Tech directions to research for prose-vecs.**

The user's instruction: do not stop at the seed list. Discover the full space, then pick the best fit for vecs-scale (solo or small-team, POC moving toward production).

**Seed list (examples, not exhaustive):**

- **Obsidian-like manual graph.** Notes + bidirectional links + tags. Operator-maintained.
- **Local LLM extraction.** Small local model (llama.cpp, Ollama, 7–14B) extracts facts from chats nightly, writes a structured fact-store. Diffs surface new / contradicted / deprecated facts.
- **Cloud LLM extraction.** Same shape, Anthropic / OpenAI / other hosted API instead.
- **Transformer-only retrieval, no extraction.** Embeddings only; rely on retrieval + query-time LLM synthesis.
- **Hybrid retrieval + synthesis.** Embedding narrows candidate set; LLM produces a single current-state answer.

**Expand into at least these adjacent categories (research session must enumerate concrete tools per category):**

- **Knowledge graphs.** RDF triple stores, property graphs (Neo4j, Memgraph). Auto-built from chats vs. operator-curated.
- **Vector DB alternatives.** Qdrant, Weaviate, Pinecone, Milvus, LanceDB, pgvector. Different retrieval/filter/metadata semantics than ChromaDB; some support hybrid graph-and-vector natively.
- **Retrieval frameworks.** LlamaIndex, Haystack, RAGFlow, Dify. Higher-level orchestration; question is whether vecs should adopt one or stay bespoke.
- **Memory-as-a-service products.** Mem0, Letta / MemGPT, Cognee, Zep. Pre-built fact-store + drift / contradiction detection. Evaluate licensing, self-host story, lock-in.
- **LLM-native memory primitives.** Anthropic memory tool, OpenAI Assistants memory, Gemini long-context. Skips local fact-store entirely; the LLM holds the state.
- **Graph-RAG variants.** HippoRAG, GraphRAG (Microsoft), LightRAG. Embeddings + graph fused at retrieval time.
- **Personal knowledge graph tools.** Logseq, Reflect, RemNote, Tana. Same problem shape as Obsidian; different architectures.
- **Event-log / append-only fact stores.** Datalog-style (Datomic), event-sourced state derivation. Time-travel queries answer "what did we believe on date X".
- **Specialized chat-summarization tools.** Anything that turns long-running chat history into a maintained summary doc.
- **Anything the search surfaces that does not fit these buckets.** Note it.

**Evaluation criteria — "best" means highest score across:**

1. **Setup cost.** How long from zero to working prototype on a Mac?
2. **Ongoing cost.** $ per month at vecs scale (~10s of chats per day, dozens of facts in scope).
3. **Locality.** What runs offline vs. requires network. Privacy implications of chat content leaving the machine.
4. **Accuracy on the BE-dev case.** Can it catch "doc says no BE dev; chat from last week says we hired one"? Concretely, not handwaved.
5. **Latency.** Query-time response, batch-time fact-extraction.
6. **Integration with existing vecs pipeline.** Reuses Voyage embeddings, ChromaDB, BM25 FTS5? Or replaces them? Or runs alongside?
7. **Maintenance burden.** Manual link-keeping (Obsidian) vs. automated drift detection vs. periodic re-extraction.
8. **Workflow-profile compatibility.** Does the chosen approach fit Phase 2's `staleness_check` slot via `"custom:<id>"` extension, or does it require profile/base changes?
9. **Lock-in risk.** Proprietary format / hosted service vs. open standards.
10. **Vibes / dogfooding.** Vecs is a search tool. The chosen prose-vecs approach should ideally be something a vecs-style project would actually want to ship, not just glue around someone else's product.

**What the research session should produce.**

Not code. A research write-up document at `docs/research/code-vs-prose-vecs-2026-XX.md`, that:

1. Defines the corpus boundary precisely (what counts as code-context vs prose-context; where chats sit; where docs like CLAUDE.md sit — code or prose?).
2. Enumerates the full candidate space (seed list + adjacent categories above, plus anything else surfaced during research). At minimum one named, concrete option per category — no abstract "knowledge graph" entry without a specific tool.
3. Scores each candidate against the 10 evaluation criteria above. Numeric or low/mid/high ratings; not prose.
4. Recommends one primary approach with reasoning, plus one backup, plus a smallest viable prototype for the primary.
5. Identifies how the existing vecs pipeline (Voyage embeddings, ChromaDB, BM25 FTS5) gets reused vs replaced for prose-vecs.
6. Sketches how the workflow profile's Phase 2 `staleness_check` slot extends: open the enum with `"custom:prose-llm"` or add a `prose_staleness_check` companion slot? Profile already accepts `"custom:<id>"`.
7. Lists what was NOT evaluated and why (time, scope, irrelevance) — be honest about coverage.

**Constraints for the research session.**

- Manager pattern: dispatch parallel research subagents per category, not per seed option. Each subagent enumerates concrete tools in its category, scores them, returns a structured digest. Manager composes the final write-up.
- Use `WebSearch` / `WebFetch` for current-state-of-the-art (the field moves fast; training-data answers will be stale).
- Spot-check claims. Don't trust "local LLM is 50ms latency" without a citation. Prefer benchmarks dated within the last 12 months.
- Honest about unknowns. If a direction is plausible but unbenchmarked at vecs scale, say so. Don't invent numbers.
- Honest about the seed list. If a seed option turns out to be dominated by something else, drop it. The seed list is a starting point, not a required slate.
- This is research, not commitment. Output is a recommendation, not a green-lit feature.
- Deep, not wide-only. After the breadth pass, go 2-3 levels deep on the top 3 candidates: actual setup commands, actual integration points with vecs's existing code, actual failure modes.

**Pre-reading list for the research session.**

- Current vecs implementation, especially the session-indexing path: `src/vecs/codex_chunker.py`, `src/vecs/codex_routing.py`, `src/vecs/indexer.py` (sessions branch around line 868), `src/vecs/mcp_server.py` (search tools).
- Vecs profile Phase 2 + Phase 6 (`docs/workflow-vecs-profile-v0.1.md`).
- The handoff this section lives in — for the design-decision trail.

## Caveats / known gaps not blocking

- Framework has no profile-loader code. Slots are validated by humans/manager, not by tooling.
- `framework_version_pin` is documentation only.
- vecs CLI has no `add-project` subcommand; `scripts/register_self.py` is the workaround.
- Roster prompt files (`.claude/prompts/critical-sinker.md`) are loaded by manager convention per profile's `specialist_scope_rule`; no harness enforcement.
- `context_tree_root` field does not yet exist on `ProjectConfig` — adding it is the first implementation step of the `context-staleness-detector` feature.

## Conversation state

- Caveman mode active (`full`). Switch with `/caveman lite|full|ultra` or "stop caveman".
- All 8 session tasks completed (see `TaskList`).
- Last user input: "handoff this session".
- Working directory: `/Users/darynavoloshyna/repo/vecs`. Branch: `master`. No uncommitted changes outside the files listed above.

## How to resume

1. Read this file.
2. Read the three signed deliverables in order: framework → profile → feature design.
3. Read `src/vecs/CLAUDE.md` for source-tree orientation.
4. Pick one of the six open decisions; confirm with user before acting on commits, registrations, or implementation.
