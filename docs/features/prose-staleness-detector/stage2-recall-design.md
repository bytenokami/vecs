# prose-drift stage-2 recall — design

Status: approved 2026-05-30. Scope: **minimal stage-2** (close the paraphrase/cross-predicate xfail). Sits under platform pillar 1 (frontier retrieval quality). Sibling roadmap: `v2-roadmap.md`. SOTA basis: `docs/research/prose-drift-review-and-sota-2026-05-29.md`.

## Problem

v1 joins doc-side and chat-side facts on an exact `(subject, predicate)` `chain_key`. Paraphrase and cross-predicate contradictions slip through: doc `team|has_role:"no backend developer"` vs chat `team|employs:"sasha"` never collide because the chains differ. Encoded as `xfail(strict)` `test_cross_predicate_paraphrase_drift_is_detected` (`tests/test_prose_drift.py`).

## Approach (Graphiti's method, no graph DB)

On a `chain_key` MISS, fall back to embedding similarity over the `is_current` facts and escalate the single best candidate to ONE LLM contradiction-judge. Keeps the exact key as the cheap first stage; adds a selective second stage. Stays inside vecs's embedded, zero-server constraint. Makes the per-row Voyage embedding — written on every fact, never read on the collision path — load-bearing.

Decisions (locked): minimal scope; **top-1** candidate per miss; **Opus 4.7** judge (extraction model unchanged); **always-on** when prose-drift runs, bounded by similarity threshold + top-1 + verdict cache.

## Data flow — `find_prose_drift`

Today the doc-driven loop does `if cur is None ... continue` (`src/vecs/prose_drift.py:373`); that `cur is None` is the MISS. New flow:

1. Load `is_current` rows **once** at scan start, with embeddings: `_load_current_rows(facts) -> list[(meta, embedding)]` plus a `chain_key -> meta` view. Replaces the per-triple `_current_row_for_chain` calls; preserves the `>1 is_current` max-version tie-break (read-only, no repair writes — same as today).
2. For each doc triple:
   - **Exact match, differing object** -> exact drift entry (unchanged behaviour; gains `match_type: "exact"`).
   - **Exact match, same object** -> NOOP, no drift.
   - **MISS** (`cur is None`) -> stage-2:
     1. Embed the doc triple: `_voyage_embed(f"{subject} {predicate} {object}")` (voyage-3 / `SESSIONS_MODEL`, the same model the facts were embedded with — symmetric).
     2. `_best_semantic_candidate(doc_emb, current_rows)` -> highest cosine, excluding any row whose `chain_key` equals the doc triple's (none, by definition of MISS).
     3. If `sim >= STAGE2_SIM_THRESHOLD` -> `_judge_contradiction(doc_triple, chat_meta, project)` (cached).
     4. If `verdict.contradicts` -> semantic drift entry.

Cosine similarity in stdlib `math` (dot / (‖a‖·‖b‖)); the `is_current` set is one row per chain, small. No numpy coupling.

## New constants

```
STAGE2_SIM_THRESHOLD = 0.85
PROSE_JUDGE_MODEL    = "claude-opus-4-7"
JUDGE_PROMPT_VERSION = "v1"
JUDGE_PROMPT         = <contradiction-judge template>
```

## Judge — `_judge_contradiction(doc_triple, chat_meta, project) -> Verdict`

- `Verdict{contradicts: bool, confidence: float, reason: str}` (frozen dataclass).
- Prompt presents both facts and asks for a verdict **with a `reason`** — forcing inline reasoning before the boolean. Returns JSON `{"contradicts","confidence","reason"}`, fence-stripped via existing `_strip_fence`.
- No `temperature` kwarg (Opus 4.7 rejects it — same constraint as extraction). Lazy `import anthropic`.
- Cached in a new SQLite table in the existing per-project cache db:
  ```sql
  CREATE TABLE IF NOT EXISTS judge_cache (
      doc_triple_json  TEXT NOT NULL,
      chat_triple_json TEXT NOT NULL,
      model            TEXT NOT NULL,
      prompt_version   TEXT NOT NULL,
      verdict_json     TEXT NOT NULL,
      PRIMARY KEY (doc_triple_json, chat_triple_json, model, prompt_version)
  )
  ```
  Triple JSON is canonical (`json.dumps(..., sort_keys=True, separators=(",",":"))`). Cache hit -> no anthropic call. Gives rerun-determinism + cost control.
- A single judge call failure (API error or unparseable JSON) is caught: conservative skip (no drift entry for that candidate), scan continues, counted in `stage2_judge_errors`. Global anthropic-unavailable / key-missing is already blocked by preflight before the scan starts.

## Output contract (additive, backward-compatible)

Existing exact entries gain `"match_type": "exact"` and `chat.subject` / `chat.predicate` (equal to the top-level pair). New semantic entry:

```
{
  "subject": <doc subject>, "predicate": <doc predicate>,
  "match_type": "semantic",
  "similarity": 0.91, "confidence": 0.8,        # from judge verdict
  "doc":  {"object": <doc obj>, "source": <relpath>},
  "chat": {"subject": <chat subj>, "predicate": <chat pred>,
           "object": <chat obj>, "session_id": <id>},
  "chat_history_versions": <int>
}
```

Report dict gains `"stage2_judge_calls"` and `"stage2_judge_errors"`. Drift sort key gains a `match_type` tiebreak so output is deterministic.

## CLI render — `src/vecs/cli.py:126-140`

Semantic lines show both predicates and a `[semantic sim=.. conf=..]` tag; exact lines unchanged. Trailing note updated: cross-predicate / paraphrase contradictions are now partially covered (exact + semantic similarity-judge); omission and soft/temporal "used to have" remain out of scope (see v2-roadmap). MCP `prose_drift` tool: no signature change — semantic entries flow through the report dict.

## Tests — `tests/test_prose_drift.py`

- Enhance `fake_anthropic`: route by prompt — a judge-prompt marker returns a verdict JSON (new `judge_response` state slot); otherwise the extraction response. Lets one fake serve both call types in a single scan.
- New fixture supplying controllable text->unit-vector embeddings for the below-threshold case (existing `fake_voyage` untouched: its constant vector yields cosine 1.0, comfortably above threshold).
- **Promote** `test_cross_predicate_paraphrase_drift_is_detected`: drop the `xfail`, set judge_response to `contradicts=true`, assert exactly 1 semantic drift with `match_type=="semantic"`, `doc.predicate=="has_role"`, `chat.predicate=="employs"`, `similarity` present.
- New unit tests: cosine helper correctness; `_load_current_rows` returns embeddings + metas; below-threshold candidate -> no judge call, no drift; above-threshold but judge says not-contradiction -> no drift; judge verdict cached (second scan makes no anthropic call); judge error -> candidate skipped, scan continues, `stage2_judge_errors` reflects it; exact-path regression (all existing exact assertions hold, `match_type=="exact"` added).

Runner: `uv run pytest -q`. Live: `VECS_TEST_REAL_LLM=1 uv run pytest -k test_integration_real_anthropic`.

## Files

- `src/vecs/prose_drift.py` — core (`_load_current_rows`, `_cosine`, `_best_semantic_candidate`, `_judge_contradiction`, `Verdict`, `judge_cache` DDL, stage-2 branch in `find_prose_drift`, new constants).
- `src/vecs/cli.py` — semantic render + updated note.
- `tests/test_prose_drift.py` — fixtures + promoted xfail + new tests.
- Docs: tick stage-2 in `v2-roadmap.md`; update the v1-boundary notes in `src/vecs/CLAUDE.md` and the `find_prose_drift` docstring.

## Out of scope (deferred, unchanged)

PSI drift-confidence calibration; SQLite fact-store migration; valid-time (Snodgrass second axis); extraction-model cost metering; top-k candidates; assistant-turn re-enablement. Each remains its own `v2-roadmap.md` entry.
