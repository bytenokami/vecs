Authored by Claude

# Est-cost-to-populate-facts (Inc 1-A metering spike)

Acceptance: `docs/features/kb-foundations-instrumentation/acceptance.md` (A line 4).
Instrument: `src/vecs/metering.py` (`estimate_extraction_cost`, `price_call`).

Rough estimate, verify before trusting. This prices a ONE-PASS bulk fact
extraction (`prose_drift.extract_facts_from_doc`, one LLM call per indexed doc
chunk) at the configured extraction model. It does NOT include Voyage embedding
cost (separate budget) and assumes a cold extraction cache (re-runs are
near-free — every chunk hashes to a cache hit). Facts are empty/extraction is
dormant today, so no spend has occurred; this is the pre-enable estimate.

## Method (measured, not guessed)

- Corpus: the `vecs` project's doc sources on disk (the repo `docs/**` tree +
  in-repo `*.md`), chunked with the real `doc_chunker.chunk_doc` — so chunk
  counts/sizes match what the indexer would extract over.
- Input tokens: `chunk_chars // 3` (a FLOOR for code-ish prose, not `//4`) plus
  the per-chunk extraction-prompt wrapper (`DOC_EXTRACTION_PROMPT`, ~237 tok).
- Output tokens: assumed 200 tok/chunk (extraction returns a short JSON triple
  list; `max_tokens=2048` caps it, most chunks yield far less). This is the one
  unmeasured input — it can only be pinned by sampling real responses.
- Pricing: `metering.PRICING_USD_PER_MTOK`, list prices noted 2026-06. **Verify
  against current Anthropic pricing** — Sonnet 4.x assumed $3 / $15 per MTok
  (input / output), Opus 4.x $15 / $75.

## vecs project (measured 2026-06-04)

| Metric | Value |
|---|---|
| Doc files | 24 |
| Total doc chars | 294,262 |
| Chunks (`chunk_doc`) | 223 |
| Input tokens (floor) | ~150,700 |
| Output tokens (assumed 200/chunk) | ~44,600 |
| Extraction model | `claude-sonnet-4-6` |
| **Est. one-pass cost** | **~$1.12** |

At the Opus judge tier the same token volume would be ~5× ($15/$75 vs $3/$15) —
which is why extraction is pinned to Sonnet (acceptance A line 3) and only the
rare stage-2 contradiction judge uses Opus.

Other configured projects (e.g. `livly-docs`) scale linearly with their doc
corpus; not measured here (sources external to this machine). Re-run the
measurement per project once its docs are local.

## Daily cap

`metering.MAX_CALLS_PER_DAY` (default 500, env `VECS_MAX_CALLS_PER_DAY`) is a
hard ceiling on real (cache-miss) calls. At ~223 chunks, a full vecs extraction
pass fits under the default cap in one day; a larger corpus would stop at the
cap and resume next day (the cache makes the resumed run skip already-extracted
chunks). `find_prose_drift` stops gracefully at the cap (`cap_hit=True` in its
payload) rather than aborting.

## Not a gate (acceptance A line 5)

Metering is a prerequisite *instrument*: it gives per-call cost records + a daily
ceiling so enabling extraction (Inc 2) is observable and bounded. It does NOT
auto-gate Inc 2/6. A cost-ceiling kill criterion, if wanted, is a §7 program
decision — deliberately not baked into the instrument.
