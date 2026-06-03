Authored by Claude

# Prose-Staleness-Detector v1 — Architecture Review & SOTA Research (2026-05-29)

## Executive summary

**Decision: PROCEED WITH MODIFICATIONS.** The two architectural bets that govern the whole feature — **keep the hand-build** (B6) and **the bi-temporal no-delete state machine** (B2) — are sound on verified evidence and should not be re-litigated. The state machine is genuinely converged: 34 findings tracked, 33 resolved, crash-safe SUPERSEDE test-asserted, a live `VECS_TEST_REAL_LLM=1` integration test passed.

But proceeding *as-is* is not warranted, for two evidence-bound reasons:

1. **The feature does not yet detect drift.** A code read of `src/vecs/prose_drift.py` (272 LOC) shows it contains only the Phase-7 dry-run subtask — the state machine, `extract_facts`, and the SQLite cache. `find_prose_drift`, `extract_facts_from_doc`, `iterate_indexed_docs`, and the preflight functions **do not exist**, and there is **zero wire-in** to `cli.py`, `mcp_server.py`, `indexer.py`, or `config.py` (all grep-verified to 0 matches). The drift comparison is the core v1 deliverable and is unbuilt.
2. **The two highest-risk recall bets (B1, B4) have verified holes.** The exact-string `chain_key = f"{subject}|{predicate}"` collision (B1) silently misses paraphrase and cross-predicate contradictions — and the team's *own* paraphrase wedge (`docs/spikes/approach1-spike.md:85-89`) has no test. B4's object-collision-only definition leaves omission, temporal ("used to have"), and self-contradiction invisible.

This is **not a pivot**: the fact-store architecture is the right shape and dominates the alternatives on infrastructure grounds. It is a bounded set of mods — finish the comparison, encode the known recall boundary honestly, add cheap canonicalization — with the LLM contradiction-judge staged as the explicit v2 recall upgrade.

## Architecture verdict

| Bet | Verdict | What SOTA says (verified) | Recommendation |
|---|---|---|---|
| **B1** Triple / exact-predicate `chain_key` | **weak** | Graphiti runs an LLM invalidation call *per new edge* over retrieved semantically-similar edges — not a deterministic key match ([Zep arXiv](https://arxiv.org/html/2501.13956v1), [getzep blog](https://blog.getzep.com/beyond-static-knowledge-graphs/), 3-vote confirmed). 2025-26 SOTA is staged: cheap retrieval → NLI → selective LLM-judge. | Keep the exact key as the cheap **first** stage; add a **second** stage — embedding-similarity over `is_current` rows + one LLM contradiction-judge on miss. |
| **B2** Hand-built bi-temporal state machine | **sound-with-caveat** | Graphiti ships the full 4-timestamp Snodgrass model (transaction-time `created_at`/`expired_at` + valid-time `valid_at`/`invalid_at`); vecs ships **one** timeline (`valid_from`=ingestion, conflated). No-delete/point-in-time invariant is the faithful mature pattern. | Sound for v1. Caveat: the single timeline makes B4's temporal "used to have" class unsolvable — document as the v1 boundary, not a bug. |
| **B3** Opus-4-7, no-temperature, SQLite determinism | **sound-with-caveat** | No-temperature constraint verified + coded (`prose_drift.py:147-152`). Opus-on-every-extraction is the most expensive point; SOTA escalates to a judge only on low-confidence. Extraction accuracy at vecs scale is **unbenchmarked**. | Keep (one-constant swap escape hatch). Meter real call counts; run the parked accuracy spike; consider Sonnet as extraction default. |
| **B4** Object-collision only, write-time | **weak** | Omission, temporal, and self-contradiction are invisible. SOTA splits the problem: Slite detects the *signal* via scheduled activity-cross-reference; Graphiti the *representation* via bi-temporal edges. Mature pattern is scheduled recrawl + diff (Diffbot, Wikidata) — never consumer-noticing. | Acceptable minimal v1 contract **if** the boundary is surfaced. Reframe as scheduled/on-demand recrawl (the comparison is already query-time). Defer the missing classes to v2. |
| **B5** Chroma as bi-temporal store | **sound-with-caveat** | Impedance mismatch is real (no `invalid_at IS NULL`, no `None`, `$and` wrapping) — all dry-run-pinned + test-asserted. The per-row embedding is **dead weight** on the collision path (written, never read). SQLite (already in vecs) natively expresses the current-view. | Sound (pinned + working). Either add the stage-2 embedding fallback so the embedding earns its keep, or migrate the fact store to SQLite. Don't reopen the pinned quirks. |
| **B6** Build vs adopt Graphiti | **sound-with-caveat** | Graphiti has **no embedded storage** — requires Neo4j/FalkorDB/Kuzu/Neptune, "user-managed infrastructure" ([PyPI](https://pypi.org/project/graphiti-core/), [GitHub](https://github.com/getzep/graphiti), 3-vote). Kuzu (lightest backend) archived Oct 2025. Apache-2.0, 26.7k stars (mature; license not a blocker). Mem0 v3 is ADD-only ([migration doc](https://docs.mem0.ai/migration/oss-v2-to-v3)). The Mem0-vs-Graphiti benchmark was never run. | **Keep the hand-build.** Borrow Graphiti's *method* (per-edge LLM judge) without its substrate. |

## Build-vs-adopt analysis

**Recommendation: keep the hand-build. Do not adopt Graphiti, Zep, Mem0, Letta, or Cognee for v1.**

Three verified facts decide it:

1. **Graphiti requires an external graph database** — Neo4j 5.26+, FalkorDB 1.1.2+, Kuzu 0.11.2+, or Amazon Neptune; deployment "requires user-managed infrastructure." vecs runs solo/POC-scale on **embedded** Chroma + Voyage with zero servers. Adopting Graphiti is a categorical infra jump that breaks the design's zero-new-storage-system constraint. Its lightest backend, **Kuzu, was archived Oct 2025** (project memory), so the cheapest adoption path is unmaintained.
2. **Mem0 is dead for this purpose.** v3 is single-pass **ADD-only** (UPDATE/DELETE removed), confirmed by the official `oss-v2-to-v3` migration doc and [mem0#2344](https://github.com/mem0ai/mem0/issues/2344). The self-editing-on-write mechanism the prior design depended on no longer exists.
3. **License/maturity is not the blocker.** Graphiti is Apache-2.0 (not MIT, as commonly assumed), 26.7k stars, peer-reviewed, `graphiti-core 0.29.1` (2026-05-21) — well past "too young." The blocker is purely operational weight.

The honest caveat: the **Mem0-vs-Graphiti benchmark at vecs scale was never run**, so "build wins on capability" is unproven. Build wins on **infra ergonomics**, which at this scale is the dominant axis. The brainstorm ranked the shipped design *third* (behind Approach B retrieval-judge and Approach A SQLite event log), but Approach B's own spike extrapolates ~15k Opus calls/mo vs the fact-store's ~1500, and the bi-temporal store yields the parked `--as-of` time-travel for free. **Top alternative: Graphiti** — adopt its *method* (per-edge LLM contradiction judge over retrieved similar facts), not its substrate.

## Ranked improvements

| # | Improvement | Lever | Impact | Effort | v1? |
|---|---|---|---|---|---|
| 1 | Implement + test the actual drift comparison (`find_prose_drift`, `extract_facts_from_doc`, `iterate_indexed_docs`, preflight) — currently absent | both | high | med | ✅ |
| 2 | Add the paraphrase-miss test (B1's own falsifiable wedge; no coverage today) | accuracy | high | low | ✅ |
| 3 | Predicate/subject canonicalization in `EXTRACTION_PROMPT` (+ bump `EXTRACTION_PROMPT_VERSION`) | accuracy | med | low | ✅ |
| 4 | Reframe + document prose-drift as scheduled recrawl; surface v1 boundary in output | staleness | med | low | ✅ |
| 5 | Stage-2 embedding-similarity + LLM contradiction-judge on `chain_key` miss (Graphiti's method) | accuracy | high | med | ⏸ v2 |
| 6 | Meter real Opus call counts + run extraction-accuracy spike; reconsider Sonnet default | accuracy | med | low | ⏸ v2 |
| 7 | SQLite fact-store migration OR justify/drop the dead embedding write | accuracy | med | med | ⏸ v2 |
| 8 | Drift-confidence score with PSI-calibrated thresholds + dual-detector confirmation | both | low | med | ⏸ v2 |

The **most important missing test (#2)** is verified absent: `tests/test_prose_drift.py` has zero paraphrase/synonym/cross-predicate coverage, yet the approach1 spike was explicitly designed to show the exact-key join misses `"no backend engineer"` vs `"leading our server-side work"`. Encode it as an xfail that flips to pass when the stage-2 judge lands.

## Proceed recommendation

**PROCEED WITH MODIFICATIONS.** The architecture is the right shape and dominates the alternatives on infra grounds; the state machine is converged and should not be reopened. But the feature does not yet detect drift, and B1/B4 leave verified recall holes. Fix with bounded mods, not a pivot.

**Fold into v1:**
- Implement + test the drift comparison (core deliverable, currently absent)
- Predicate canonicalization in the extraction prompt (bump prompt version)
- Paraphrase-miss test encoding B1's known hole
- Reframe as scheduled recrawl + document the v1 object-collision boundary in user-facing output
- Wire the CLI / MCP / indexer / `ProjectConfig.prose_drift_enabled` per the converged design

**Park for v2:**
- Stage-2 embedding-similarity + LLM contradiction-judge (the high-value recall upgrade)
- Second valid-time axis for soft/temporal contradictions
- Drift-confidence scoring with PSI thresholds
- SQLite fact-store migration or dropping the dead embedding write
- Call-count metering + extraction-accuracy spike; Sonnet-default reconsideration
- `--as-of` time-travel (already parked)

## Sources

- Zep: A Temporal Knowledge Graph Architecture for Agent Memory — https://arxiv.org/html/2501.13956v1
- Beyond Static Knowledge Graphs (Zep blog) — https://blog.getzep.com/beyond-static-knowledge-graphs/
- getzep/graphiti GitHub — https://github.com/getzep/graphiti
- graphiti-core on PyPI — https://pypi.org/project/graphiti-core/
- Graphiti + Neo4j — https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/
- Graphiti + FalkorDB — https://www.falkordb.com/blog/building-temporal-knowledge-graphs-graphiti/
- Mem0 v2→v3 migration (ADD-only) — https://docs.mem0.ai/migration/oss-v2-to-v3
- Mem0 issue #2344 — https://github.com/mem0ai/mem0/issues/2344
- Letta dedup issue #3116 — https://github.com/letta-ai/letta/issues/3116
- XTDB v2 launch — https://xtdb.com/blog/launching-xtdb-v2
- Datomic history — https://docs.datomic.com/client-tutorial/history.html
- Fiddler PSI thresholds — https://www.fiddler.ai/blog/measuring-data-drift-population-stability-index
- Slite knowledge base freshness — https://slite.com/solutions/knowledge-base
- Panthaplackel et al., Deep JIT Inconsistency Detection — https://arxiv.org/pdf/2010.01625

### Repo evidence (verified by direct read, 2026-05-29)
- `src/vecs/prose_drift.py` (272 LOC) — state machine + `extract_facts` + SQLite cache only; `find_prose_drift`/`extract_facts_from_doc`/`iterate_indexed_docs`/preflight absent
- `src/vecs/cli.py`, `mcp_server.py`, `indexer.py`, `config.py` — 0 matches for prose-drift wire-in
- `tests/test_prose_drift.py` — no paraphrase/synonym/cross-predicate test
- `pyproject.toml` — `anthropic==0.103.1` pinned
- `docs/features/prose-staleness-detector/gaps.md` — 34 findings, 33 resolved, 1 parked
- `docs/spikes/approach1-spike.md:85-89` — the paraphrase wedge

---

*Generated by the `prose-drift-review-and-sota` workflow: 47 agents, 6 adversarial review dimensions + 6 web-research angles, 42 unique claims, 8 survived 3-vote adversarial verification (2 killed). Repo claims re-verified by direct read before publication.*
