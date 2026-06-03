Authored by Claude

# Future Feature: shared-team-vecs (placeholder)

Status: idea logged 2026-05-21. NOT scheduled for v1. Separate from prose-staleness-detector v1.

## Trigger

Mid-design discussion (2026-05-21) on prose-staleness-detector v1. Operator floated: "How about I train a local LLM on Mac mini as knowledge base and make it shared with a team?"

Decomposition surfaced three separable goals:
- **A. Shared team vecs server** (chosen — this doc).
- B. Local synthesis LLM (file separately if pursued).
- C. Fine-tune local LLM ON the KB (rejected — RAG dominates fine-tuning for fact recall).

## Feature shape (sketch only)

vecs MCP server deployed on a team-owned Mac mini (≥M4 Pro, ≥64GB). Network-accessible via Tailscale / WireGuard / VPN. All team members' Claude Code (and Codex) sessions index into a single shared `~/.vecs/chromadb/` + BM25 sidecar.

Operations the v1 (single-user POC) does NOT need but a shared deployment does:

- Authentication / authorization on the MCP server endpoint (mTLS or bearer token).
- Per-team-member identity threaded through index calls so `source_id` in `<project>-prose-facts` rows distinguishes "Alice's chat" from "Bob's chat".
- Concurrent-write coordination (existing `fcntl.flock` on manifest handles single-process serial; shared deployment needs cross-machine locking via Postgres advisory locks, Redis, or a queue).
- Index-update fan-out: when one teammate adds a doc, others' caches invalidate.
- Backup / disaster recovery (current vecs is single-user laptop scale; team scale needs nightly snapshots).
- Operator UI for team admin (list projects, prune stale collections, view drift across team).

## Composes with V+ (prose-staleness-detector v1)

V+ ships single-user first. The shared variant inherits V+'s INSERT/NOOP/SUPERSEDE state machine unchanged; the `<project>-prose-facts` Chroma collection becomes naturally shared because all team members write to the same collection. SUPERSEDE then surfaces team-wide contradictions (Alice indexed "no BE dev"; Bob's chat 3 weeks later says "hired Sasha" → SUPERSEDE event, drift visible to anyone querying).

## Why not v1

- Networking + auth + concurrency are independent product surface, not drift-detection logic.
- Single-user V+ is testable in isolation; shared adds 3 invariants (auth, locking, identity).
- Mac mini deployment requires hardware purchase + ops overhead the operator hasn't committed to.
- Cleaner: ship V+ single-user, validate the drift mechanism, then deploy on Mac mini once V+'s patterns are proven.

## Not committed

This file is a placeholder. No design work. No timeline. Revisit only after V+ ships and is in production use for the solo operator. The shape above is the operator's brainstorm, not a locked plan.
