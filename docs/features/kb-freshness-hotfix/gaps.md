Authored by Claude

# Gap log: kb-freshness-hotfix

Phase-8 gap log per `docs/workflow-vecs-profile-v0.1.md`. Records what this feature deliberately did NOT cover, and follow-ups it surfaced — so a silent scope boundary doesn't read as "covered."

## Deliberately out of scope (by design)
- **Freshness *bucket* (age class), not just the raw anchor.** 1.5c surfaces the `version_id` verbatim (sha/mtime) as the per-hit trust signal. It does NOT derive a fresh/recent/stale age bucket — that needs a reference clock and is non-uniform across code (sha) vs docs (mtime). Deferred until Inc-1-E defines the stale-retrieval-rate metric against the same anchor.
- **No recency prior / re-ranking on freshness.** The signal is informational (surfaced to the agent); search ranking is unchanged. Recency weighting stays in Inc 4a.
- **Interlock marker is docs-only in practice.** Only the docs re-embed path records a marker (`set_collection_model`). Code collections are never marked, so the interlock can only ever fire for `-docs`. That is intended (code stays `voyage-code-3`, never flips) — but it means the interlock is a no-op safety net for code by construction.

## Follow-ups surfaced (not blocking; candidates for later increments)
- **Stale-retrieval-rate harness (Inc-1-E).** The `version_id` anchor 1.5c surfaces is the input to E's metric. E is the next sequenced increment and the hard precondition for Inc 3/4b/6/7.
- **`version_id` type at the surface.** `_freshness_tag` calls `str(v)`; the indexer stamps code sha as `str` and docs mtime as a value coerced to str at chunk construction. If a future chunk source stamps a non-str/non-hex token, it surfaces verbatim — acceptable, but worth a glance when new chunk sources are added.
- **Interlock observability.** A marker mismatch logs to stderr only. If/when search runs unattended, a counter or a surfaced banner in the `semantic_search` payload (so the agent sees "this collection fell back to BM25") would beat a stderr line nobody reads. Not built — would be a small follow-up if the fallback ever fires in the field.

## Known superseded acceptance items (audit trail)
- 1.5a acceptance lines for session-chunk deletion and the `-sessions` collection describe machinery removed by `5dedb17` (session/codex indexing rip). The shipped 1.5a prune fix still handles a legacy `session:{path}` manifest key (it reads as a non-existent literal path and is pruned as stale junk), but there is no longer a `-sessions` collection to sweep. Boxes left unchecked rather than rewritten, to preserve the original-spec record.
