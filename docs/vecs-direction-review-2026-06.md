Authored by Claude

# vecs — Phase-4 direction review (2026-06-03)

Adversarial 7-lens critic panel (workflow run `wf_e205def2-b79`, 7 critics + synthesis) on:
shipped Inc-1-pipeline work (C/B/F) + the 7-increment program (`vecs-kb-curation-design-2026-06.md §6`) + the session-extraction reframe.

**Verdict: `course_correct` — the direction is right, the sequencing is wrong.** The spine (cheap foundations → facts → docs → freshness → share → transcript inversion → quality) is sound; the program front-loads correctness-heavy machinery and unmeasured quality bets *ahead of* the one verified north-star defect and the one instrument that would tell us whether the later bets matter on this corpus.

All findings below were re-verified against the live code/store by the main thread (not taken on the thinking-off critics' word).

## Two convergent, verified facts (6/7 critics)

1. **`prune()` orphan corruption is LIVE and lies to the agent on every search.** `Manifest.prune()` (`indexer.py:181-189`) returns only a count; its caller in `run_index` (`indexer.py:~1850`) never deletes the chunks from Chroma or BM25. The 4×/day cron prunes the manifest but leaves the vectors forever, so deleted files keep ranking against live content. Design commit `39f6d3d` deliberately folded the ~50-LOC fix into Inc 4a's bi-temporal apparatus — parking an active corruption behind months of Inc 2/3/4a. The fix is **independent** of `valid_from`/`valid_to` and reuses already-shipped deletion code (`_delete_stale_chunks_after_embed`, `_delete_ids_from_bm25`).
2. **The eval harness (Inc-1-E) and metering (Inc-1-A) do not exist** (only the `version_id` anchor + spec dirs). Yet Inc 3/4b/6/7 are "eval-gated" on E and Inc 2's cost is gated on A. E is currently sequenced *last* in Inc-1, *after* the mutations it should measure — backwards.

## Verified findings (severity-ordered)

| # | Severity | Finding | Verified anchor |
|---|---|---|---|
| 1 | 🔴 critical | **Live migrating reindex is a content swap, not in-place re-embed.** All 1631 `livly-docs` chunks are pre-F bare-scheme → `_partition_docs_by_root` classes all as orphans → deletes them; only ~365 surviving files re-embed (rest are dead orphans the prune bug never cleaned), + 1051 sessions re-embed under voyage-4 — one blind command, no eval gate, no run-lock vs cron. Orphan delete (`indexer.py:1425`) commits **before** the first voyage-4 upsert (`:1492`); if voyage-4 default dim ≠ 1024 (a `config.py:39-43` *comment*, not a checked fact) `-docs` empties unrecoverably. | live store: `embed_cache.db` 0 bytes (unrun); `livly-docs`=1631 bare-scheme |
| 2 | 🟠 high | **`prune()` orphan delete** — see fact ①. The only verified "agent retrieves lies" defect. | `indexer.py:181-189` + caller `~1850` |
| 3 | 🟠 high | **Inc-1-E/A don't exist** but 4 increments gate on them — see fact ②. | grep: no metering/harness code |
| 4 | 🟠 high | **`searcher.py:150` gates `-docs` on `proj.docs_dir`** → bloomly/eric `-docs` (which F just populated with in-repo `.md`) are **never searched**. F is sunk cost for 2/3 projects until fixed. code/sessions targets have no such gate. | `searcher.py:150` confirmed |
| 5 | 🟠 high | **Model-flip degradation window is LIVE and fails open** — markers unset + reindex unrun = new-model query vectors hitting old-model stored vectors, with no trust signal surfaced to the agent. | `version_id` stamped but never returned/filtered |
| 6 | 🟡 med | **Extraction-model contradiction** → **RESOLVED (owner, 2026-06-03): use latest Sonnet always.** Set `prose_drift.py:25 PROSE_EXTRACTION_MODEL = "claude-sonnet-4-6"` (was `claude-opus-4-7`; design §6 already said Sonnet). Both extraction + judge are dormant today (extraction disabled). **Open sub-decision:** the stage-2 contradiction-judge `prose_drift.py:29 PROSE_JUDGE_MODEL` is also `claude-opus-4-7` — does "latest Sonnet always" apply to the judge too, or keep it on (latest) Opus for contradiction-detection quality? | `prose_drift.py:25,29` |
| 7 | 🟡 med | voyage-4 dim==1024 unverified against live API (delete-before-upsert, no rollback). | `config.py:39-43` comment |

## Revised sequencing (delta vs §6)

Insert **Inc 1.5 (no-regret, ship next, before Inc 2):**
- **1.5a — prune-orphan fix** (pulled out of 4a): `prune()` returns stale keys → caller classifies each by collection (code/docs) → deletes chunks + BM25 + a one-time orphan sweep (scan each collection's `file_path` metadata, delete chunks whose source is gone on disk). Leave only the recurring `valid_from`/`valid_to` tombstone *semantics* in 4a.
- **1.5b — searcher `-docs` gate one-liner** (`searcher.py:150`): always attempt the `-docs` collection (skip-on-miss like code/sessions); don't wait for Inc-1-search D.
- **1.5c — freshness/trust signal**: surface `version_id`/freshness in the MCP result header + a query-time interlock that warns/falls back to BM25 when a collection's recorded model marker ≠ configured model.

**Then build Inc-1-E + thin Inc-1-A** (currently last in Inc-1 → move up). Point E at the live store; make a green E number the hard precondition for Inc 3/4b/6/7 and the heavy half of 4a. Keep A a spike (per-call {model,tokens,$} + hard `MAX_CALLS_PER_DAY`), not a dashboard. Sequence E right after 1.5a so the stale-retrieval-rate drop is the proof the prune fix worked.

**Cut candidate:** Inc 4c (MinHash near-dup dedup w/ refcount+tombstone ownership) — correctness-heavy machinery to save a cost a single user pays once; defer indefinitely unless E shows a real redundancy harm.

## Session-extraction reframe → deferred (do NOT build now)

Keep the two good ideas (stage candidate facts to jsonl, not the live store; git-anchored provenance). But:
- "Piggyback ai-log for free" is **false** — `_build_body_prompt` (`~/.claude/skills/ai-log/scripts/entry.py:625`) feeds `claude -p` only "User prompts (first N, each truncated to 200 chars)" — user turns only, a Sonnet *retrospective*, not a triple producer over the full transcript.
- Coupling a team-KB deliverable to an unversioned personal hook is wrong when vecs already reads the same jsonl via `sessions_dirs`.
- It accumulates an ungradeable/unpriced stream before E/A exist.

It is Inc-6-class with zero agent-felt payoff this quarter. Revisit after Inc-1-E/A + Inc 2's gate.

## Live-reindex runbook (owner's manual step — do NOT auto-run)

Before the single migrating reindex:
1. Verify voyage-4 default dim == 1024 with ONE real API call (`len(embeddings[0])`).
2. Snapshot `~/.vecs/chromadb` (the 1631 deletions are irreversible; some HQ/superpowers/fork sources still exist and re-embed correctly, dead orphans are gone for good).
3. Pause the 4×/day reindex cron for the duration (no run-level lock; `prune_out_of_scope` is unsafe under concurrent runs).
4. Capture a before/after `mcp__vecs__semantic_search` baseline on known docs queries (manual stand-in for the unbuilt E harness).

## Open decisions for the owner
- ~~Extraction model: Opus vs Sonnet~~ → **RESOLVED: latest Sonnet always (`claude-sonnet-4-6`)**. Still run the Inc-1-A spike to get the actual Sonnet $/run before enabling extraction.
- Stage-2 contradiction-judge (`PROSE_JUDGE_MODEL`): latest Sonnet too, or keep (latest) Opus for quality? (sub-decision of the above)
- Inc 4c: cut or keep-behind-eval.
- Whether 1.5 ships as its own workflow-profile feature or folds into a 1-pipeline follow-up.
