Authored by Claude

# vecs — HQ Talk: Findings & Decisions (2026-06-17)

Working summary of this session's findings, for composing the HQ presentation. All
numbers are rough and labelled — quote *ratios and tradeoffs*, not precise dollars.
Items tagged **[DECISION]** are open and need owner/HQ input. Items tagged
**[VERIFIED]** were confirmed against a primary source this session; **[MEASURED]**
came from a run on our own corpus.

---

## ▶ Instructions for Claude (work mac) — read this first

You are being pointed at this file as **source material to build an HQ presentation
about vecs.** Your job is to turn the findings below into presentation slides + speaker
notes. The operator will tell you the exact tool/format (likely a deck/composer tool);
this document is the content, not the format.

How to use it:

1. **Spine = §0 (TL;DR).** Build the narrative from those 5 points, then draw
   supporting slides from §2–§8. §10 has ready copy-paste lines.
2. **Respect the tags.** `[VERIFIED]` = confirmed against a primary source (safe to
   state as fact; cite the Appendix source). `[MEASURED]` = a number from our own run
   (always state it with its assumptions). `[DECISION]` = **NOT decided** — present as
   an open question with the tradeoff shown; do **not** assert a choice or invent an
   answer. Bring `[DECISION]` items to the operator.
3. **Lead with ratios and tradeoffs, not dollars.** The ~5× locate ratio and the
   sovereignty story are the substance; the dollar figures are small and easy to
   attack — keep them as a footnote, not a headline.
4. **The arc:** vecs is thin local glue → it measurably saves agent effort (~5×) →
   sovereignty is egress-minimization and the **agent tier is the #1 leak** (fix it
   first) → local embedding is the endgame, ~11 pts behind today with a clear ladder to
   close it → therefore we need the Mac Studio.
5. **Writing constraints for slide copy:** no em-dashes; no local filesystem paths,
   hostnames, emails, or personal identifiers; keep the honest caveats in (they are
   what make HQ trust the rest); plain declarative prose.
6. **Keep the honesty.** Do not overclaim. Where the doc flags a counter-case or a
   "buy wins" point, keep it — a one-sided deck reads as a pitch and loses the room.
7. **Open verifications before the talk:** confirm Voyage's SOC 2 / ISO on its Vanta
   portal; confirm whether we were opted out (Voyage and Claude) before embedding the
   existing corpus; note the Voyage pre-opt-out retention duration is unstated.

Primary sources for every `[VERIFIED]` claim are in the Appendix. Related detail lives
in `docs/vecs-ab-locate-payload-2026-06-17.md` (locate A/B + cost + sovereignty) and
`docs/vecs-kb-curation-design-2026-06.md` (deployment topology, Inc 5).

---

## 0. TL;DR — the spine of the talk

1. **vecs is thin glue, not a product.** It wires best-of-breed parts —
   ChromaDB (vector) + sqlite FTS5 (keyword/BM25) + Voyage or local Qwen
   (embeddings) + Claude (the agent) — into an agent-facing retrieval layer over
   our code + docs, exposed via MCP. We own the integration (~a few thousand lines),
   not the hard parts.
2. **It measurably saves agent effort.** On our own repo, semantic search located
   code in **~5× fewer tokens** than grep-and-read (~3K vs ~16K per query), rising
   to ~8× when the answer is buried in a large file.
3. **Sovereignty is an egress-minimization policy, and the #1 leak is the AGENT, not
   the embedder.** On personal Claude Pro/Max subscriptions, Claude Code trains on
   our code *by default*. Fix the agent tier first; the embedder second.
4. **Local embedding (Qwen) is the sovereignty endgame.** Today it trails Voyage by
   ~11 points recall@10 (off-the-shelf 4B), with a clear ladder of levers to close
   the gap (8B → reranker → higher dim → fusion tuning → domain fine-tune).
5. **We need a Mac Studio** to run that research *and* serve the team's shared index.
   We are already memory-bound on the 48GB machine running the smaller model.

---

## 1. What vecs is (and isn't)

- **Not** a proprietary vector database competing with Pinecone. The vector store
  (Chroma), keyword index (FTS5), embedding model (Voyage / Qwen), and agent
  (Claude) are all off-the-shelf. vecs is the integration layer.
- **Consumer = a coding agent, via MCP** — not a human at a search box. This drives
  the design: code-aware AST chunking, git-SHA freshness, model-pin interlocks,
  ranked-chunk output (not a chat answer).
- **Corpus = our own code + working docs**, on hardware we control.
- Today's embedding models: docs = `voyage-4`, code = `voyage-code-3` (both
  1024-dim), via the Voyage API.

---

## 2. Measured value — locate-payload A/B  **[MEASURED]**

Question: how many tokens does an agent ingest to *locate* code — semantic search
vs grep-and-read? Run read-only against the vecs repo's own index (36 code + 226
docs chunks). Measures the locate phase only (not a full agent session); token
estimate uses bytes ÷ 3 for code (a floor).

| Query | grep payload (whole file) | vecs payload (ranked chunks) |
|---|--:|--:|
| orphan prune/sweep | 24,750 | ~3,000 |
| embed cache key | 1,917 | ~3,000 |
| model-flip interlock | 24,750 | ~3,000 |
| rank fusion | 3,735 | ~3,000 |
| bm25 sync/delete | 24,750 | ~3,000 |

- **Average: grep ~16,000 → vecs ~3,000 tokens. ≈5× less; ~8× when the answer is
  buried in a large file.**
- **Honest counter-case:** one query's answer sat in a small, well-named file — grep
  tied vecs there. vecs's edge is largest exactly where grep is worst (logic buried
  in big files).
- **Beyond tokens:** ~3–5 fewer tool round-trips per locate (faster), and less
  context-window pressure (fewer compactions, longer sessions). These are the real
  agent-efficiency wins and don't dollarize cleanly.

---

## 3. Cost projection — now vs after vecs vs save

**Assumptions (state on slide):** 40 locate/understand ops per dev/day; 220 working
days/year; Opus 4.8 input = $5 / 1M tokens; locate-phase *input* tokens only
(output, edits, conversation not addressed). Reconciled to the measured 16K → 3K.

| Scope | NOW (no vecs) | AFTER vecs | SAVE |
|---|---|---|---|
| Per dev / day  | 640K tok · **$3.20** | 120K tok · **$0.60** | 520K tok · **$2.60** (81%) |
| Per dev / year | 140.8M tok · **$704** | 26.4M tok · **$132** | 114.4M tok · **$572** |
| 10 devs / year | 1.41B tok · **$7,040** | 264M tok · **$1,320** | 1.14B tok · **$5,720** |

- **Cache-realism:** re-read context bills at ~0.1× input, so divide the dollars by
  ~3–10 (10-dev/year saving ≈ **$0.6–1.7K**).
- The dollar saving is modest. **Lead with the ratio (~5× fewer tokens, measured)
  and the speed/quality story, not the dollar figure.**
- Reference pricing (claude-api, cached 2026-06-04): Opus 4.8 $5 in / $25 out;
  Sonnet 4.6 $3 in / $15 out (per 1M).

---

## 4. Build vs buy — market landscape (2026)

Three categories; none lands on vecs's exact spot.

- **Code assistants** — Cody, Cursor, Copilot, Continue, Tabby, Augment, Sourcebot.
- **Enterprise wiki-RAG** — Glean, Onyx (Danswer), Dust, Notion AI, Rovo, M365.
- **Vector DBs + frameworks** — Pinecone, Qdrant, Chroma, Weaviate; LlamaIndex,
  txtai; PrivateGPT, LocalGPT. (Components/DIY, not a finished product; vecs is built
  *on* one — Chroma.)

### vecs vs the 5 nearest

| | self-host / sovereign | agent MCP | hybrid vec+BM25 | code-aware | docs too | OSS | maturity |
|---|---|---|---|---|---|---|---|
| **vecs** | ✅ | ✅ | ✅ | ✅ | ✅ | own | low (bespoke) |
| Sourcebot | ✅ | ✅ | ❌ keyword/trigram | symbol ✅ / dense ❌ | ❌ | ✅ | med |
| Augment | remote ❌ / local ~ | ✅ | ? | ✅ | ? | ❌ | high |
| Onyx (Danswer) | ✅ | ✅ | ✅ | ❌ connector-text | ✅ | ✅ | high |
| Cody | ❌ SaaS ($59/seat) | API (not MCP) | ✅ | ✅ | ~ | ❌ | high |
| Continue.dev | ✅ local | ❌ IDE-bound | ✅ | ✅ | ~ | ✅ | med |
| Glean | ❌ SaaS ($60K+/yr) | ✅ | ✅ | ❌ connector-text | ✅ | ❌ | high |

**Verdict:** the intersection vecs occupies — local + hybrid-dense + MCP-agent-facing
+ code AND docs — is unoccupied by any single off-the-shelf product. Each nearest
miss drops one axis.

**Honest caveats (include — they age badly otherwise):**
- The gap is **closing fast** (Augment MCP context engine Feb 2026; Sourcebot OSS
  MCP-native; Zilliz "Claude Context MCP" does BM25+dense; Chroma/Qdrant/Pinecone all
  ship MCP). Don't claim "nothing like this exists."
- Durable build-case is **(1) sovereignty, (2) code-semantic retrieval quality**
  (Onyx/Glean treat code as text — no function/class chunking, no code-tuned
  embeddings), **(3) we own the eval + pipeline**.
- Where **buy wins**: connectors/breadth, human browser UI, RBAC/SSO/multi-tenant
  (vecs has none today), scale + maturity.

---

## 5. Sovereignty — what actually leaves, and under what terms

Frame the whole topic as **egress minimization**: every party that sees our code held
to no-train terms, in priority order. "Nothing leaves your machine" is **false** the
moment any cloud agent reads a file — do not slide it.

### 5.1 The #1 egress path is the AGENT  **[VERIFIED]**

The coding agent reads every file it touches — far more code than the embedder ever
sees. And on **personal Claude Pro/Max subscriptions, Claude Code trains on that code
by default.**

Verbatim (code.claude.com/docs/en/data-usage): *"We will train new models using data
from Free, Pro, and Max accounts when this setting is on (including when you use
Claude Code from these accounts)."*

- Training toggle **defaults ON**; retention **5 years** when on, 30 days when off.
- **No org-level control** at Pro/Max — only a per-developer toggle
  (claude.ai/settings/data-privacy-controls).
- **API / Console keys + Team / Enterprise = no-train by policy** (commercial terms:
  *"Anthropic may not train models on Customer Content"*); 30-day retention; ZDR for
  qualifying Enterprise.
- Turning the toggle off is **not retroactive** — stops future training, not past
  sessions already ingested.

**Current de-facto state:** if the team runs Claude Code on personal subs and nobody
flipped the toggle, company code is in Anthropic's training pipeline with 5-year
retention. **[DECISION]** This is an IP question for HQ/legal, independent of vecs.

**Fix, in priority order:**
1. **Agent tier (biggest, first):** move the team to **Team/Enterprise** (org-enforced
   no-train + flat seats + admin control) — *not* personal API keys (loses the
   subscription economics, no governance). Stopgap today: every dev turns training
   **off** in the privacy controls (keeps Max limits, drops retention 5yr→30d).
2. **Embedder** (Voyage opt-out → local Qwen) — a smaller surface than the agent.

Note on API keys vs subscription: you do **not** need API keys to stop training as an
individual — the toggle does it and keeps Max limits. API keys are no-train but
metered (lose the flat Max allowance). The org-level answer is Team/Enterprise.

### 5.2 Voyage data policy  **[VERIFIED]**

Primary sources: voyageai.com/privacy (updated 2025-02-20), voyageai.com/tos (updated
2026-05-27), voyageai.com/dpa.

- **Training is ON by default (opt-out).** Unless you opt out, you grant Voyage *"a
  worldwide, irrevocable, perpetual, royalty-free... license to use, copy, reproduce,
  distribute, prepare derivative works of, display and perform the Customer Content...
  to train, improve, and otherwise further develop the Service."*
- **Opt-out is NOT retroactive** — *"applies only to Customer Content you submit after
  the time at which you opt out"*; prior data *"may continue to be subject."*
- **Retention:** zero-day deletion applies **only post-opt-out** (*"immediately
  deleted... after it is processed"*). Pre-opt-out retention duration is **unstated**
  (the one open gap).
- **DPA exists** (voyageai.com/dpa, incorporated by ToS §6, covers the embeddings
  API); sub-processors **AWS + Google**; region **United States**.
- **MongoDB does NOT govern this** — Voyage (acquired ~Feb 2025) runs standalone
  ToS/DPA and is not named in MongoDB's SOC 2 / ISO / DPA, so its certs can't be leaned
  on. Voyage's own SOC 2 / ISO status is on its Vanta trust portal —
  **[DECISION] confirm certs manually** (the page is JS-rendered, not machine-readable).

### 5.3 Recommended posture — opt out from day one

If the opt-out is enabled *before* any embed, every submission is post-opt-out: not
trained on, deleted after processing. Accurate claim becomes *"our code is processed
transiently under Voyage's DPA, not retained, not trained on."* Caveats: (a) it still
**transits** Voyage (US, AWS/Google) — "not retained/trained", not "never leaves";
(b) it's a **contractual** promise, not a technical guarantee — local Qwen is the
trust-free version. **[DECISION] Verify we were opted out before the existing corpus
(livly, bloomly, vecs) was embedded** — if not, that already-sent text is under the
perpetual license regardless.

### 5.4 Rebuttals to likely HQ objections

- *"Why not let Voyage train on us — they'll get better for us?"* → You get the
  improved models **whether you opt in or not** (your data is a rounding error in
  their training set); opting in adds no private upside but grants a broad perpetual
  license over the product's source. Bad trade. If you want a model better *for us
  specifically*, that's local fine-tuning (§6), the opposite of donating to a vendor.
- *"We already send code to Claude/ChatGPT/Gemini, so who cares about Voyage?"* →
  True that air-gap is already broken by the agent — but the agents on no-train tiers
  are the *good* terms; Voyage-default (perpetual train) is *worse* than the egress we
  already accepted. The principle is to hold every vendor to the agent's no-train bar.
- *"Why build instead of buy?"* → see §4. Sovereignty + code-semantic quality + owning
  the eval; and for human wiki search or SSO/RBAC at scale, we'd buy.

---

## 6. Local embedding — Qwen vs Voyage, and closing the gap

### Current measured state  **[MEASURED]** (L2 A/B, work mac, 2026-06-17)

- **Qwen3-Embedding-4B (MRL-truncated to 1024-dim) LOSES to Voyage by ~11 points
  recall@10.** Split: natural-language queries **−13** (Qwen worse), ID/symbol queries
  **+14** (Qwen better). Confidence intervals exclude 0 → statistically real. Latency
  fine (44 ms). Current lean: keep Voyage.
- Production blockers surfaced on the 48GB machine (memory-pressure symptoms): giant
  chunks (91K tokens) need a sequence-length cap, cache clearing between batches, and
  length-bucketed batching.
- **Rescues not yet run** (these are the levers below).

### The lever ladder (cheapest → most effort) — close the gap, measure each

1. **Use 8B instead of 4B (off-the-shelf).** Qwen3-Embedding-8B tops the open family
   (MTEB-Code ~80.7). Biggest free win — no training, just more RAM. Try first.
2. **Add the reranker (Qwen3-Reranker-4B).** Research: +5.8 pts on code; targets exactly
   the NL queries where 4B lost. Note: **vecs uses no reranker today** (verified — zero
   `rerank` references in the source); both Voyage's and Qwen's are available-but-unused,
   so a reranker is a lever for *either* path.
3. **Drop the MRL truncation / use a higher dim** (1024 → 2560 or native) — we're
   discarding vector information today.
4. **Tune the RRF fusion weights.**
5. **Domain fine-tune (last lever).** A domain-specialized embedder can beat a general
   commercial one *on its own domain* — this is the only path that plausibly makes a
   local model **beat** Voyage on our code, and it's the strong-form sovereignty
   endgame (a model better *for us*, fully local, nothing to trust).

### Fine-tuning — the correct mental model  **[DECISION]**

- You do **not** train an embedder by running it over raw code (next-token). Retrieval
  quality comes from **contrastive fine-tuning on (query, relevant-chunk) pairs + hard
  negatives.** The hard part is the **training data** — labeled query↔code pairs you
  must synthesize (LLM-generated) or mine from usage; data quality is the ceiling.
- A fine-tune is **domain-specific** — a model tuned on one project does not transfer
  to another. Validate the pipeline on **bloomly** (lower-stakes test bed), but treat
  the fine-tune number as bloomly-specific; re-measure on livly before claiming a livly
  win. The cheap levers (8B, reranker, dim) *do* generalize across projects.
- Likely outcome: **8B + reranker may erase the 11-point gap before fine-tuning is
  needed.** Sequence it; don't pre-commit.

---

## 7. Deployment topology (team / company-wide)

Captured in the increment program (`docs/vecs-kb-curation-design-2026-06.md`, Inc 5).
Consumer is the team's agents over MCP.

- **Central build, local read.** One box indexes the canonical origin of each product
  on a cadence; the shared index is identical for everyone.
- **Two consumption modes** on one axis (*no local model* ⟺ *per-query network call*):
  remote MCP (zero laptop models, but every query is a round-trip + single point of
  failure) vs pull-to-local (offline + zero query latency, needs a local query-embed
  model).
- **cocone MCP gate = transport + access model.** Multi-tenancy via **per-PRODUCT
  instances** (group related repos — not strictly per-repo, which would kill cross-repo
  search), so access control = "which instance the gate lets you reach" (no in-vecs
  identity work).
- **Trap:** do not duplicate the embed model per instance. Split embedder from index —
  one shared resident embedder + N lightweight index instances.
- **[DECISION]** gate capabilities (routing/auth only vs identity passthrough); who
  runs+maintains the box (HQ infra vs us); box location; embedder (Voyage vs Qwen);
  per-product instance granularity. The shared-embedder service is net-new build.

---

## 8. Hardware — why the Mac Studio (now)  **[DECISION]**

The Studio is **two jobs in one box**: the research machine *and* the central
build-and-serve index for the team.

Needs that the 48GB work machine cannot carry:
- Qwen-8B inference (~16GB) **plus** the reranker resident together — stacking models
  exhausts 48GB.
- **Fine-tuning 8B** — RAM-hungry (model + optimizer state + data + evals); ~128GB
  makes LoRA/contrastive feasible.
- Central bulk-embed of the corpus + running A/B evals at scale.

**Evidence we're already undersized:** the memory-pressure workarounds above (§6) were
hit running the *smaller 4B* evaluation on the 48GB machine. 8B and any fine-tuning
don't fit there.

Deck line:
> To build a sovereign retrieval model that matches or beats the commercial vendor on
> our own code, and to run the team's shared index, we need the Mac Studio. We are
> already memory-bound on the 48GB machine running the smaller 4B evaluation; the 8B
> model and any fine-tuning do not fit there. The Studio is both the research machine
> and the central build-and-serve box for the team.

---

## 9. Open decisions (what we need to decide)

1. **Agent tier:** move the team to Team/Enterprise (no-train, org-enforced)? Stopgap:
   everyone toggles training off now. (Biggest, most urgent — current default is
   train-on-our-code.)
2. **Were we opted out** (Voyage *and* Claude) before embedding the existing corpus? If
   not, accept past data is covered and protect going forward.
3. **Embedder direction:** keep Voyage (with opt-out + DPA signed) short-term; commit to
   the Qwen lever ladder for the endgame?
4. **Fine-tune?** Fund the domain fine-tune R&D (needs training-pair generation + GPU),
   or stop at 8B + reranker if that closes the gap?
5. **Mac Studio:** approve/expedite — research machine + central index server.
6. **Deployment:** remote-MCP vs pull-to-local; per-product instances behind the cocone
   gate; who owns the box.
7. **Confirm Voyage SOC 2 / ISO** via its Vanta portal; **sign the Voyage DPA** if we
   keep Voyage.

---

## 10. Deck-ready lines (copy-paste, plain text)

vecs is a thin local layer over off-the-shelf parts (ChromaDB, Voyage or local Qwen embeddings, Claude). We own the integration, not the hard parts, because the consumer is a coding agent, the corpus is our own code, and owning the pipeline lets us measure and tune it.

On our own codebase, semantic search located code in about 5x fewer tokens than grep-and-read (around 3K vs 16K per query, up to 8x when the answer sits in a large file). This is locate-effort; speed and context savings are on top.

Sovereignty is about minimizing who sees our code and on what terms. The largest exposure is the coding agent itself: on personal Claude subscriptions it trains on our code by default, with five-year retention and no org control. The first step is moving the team to no-train terms (Team or Enterprise). The embedder is the second step.

Local embeddings are the endgame where nothing leaves our hardware and there is nothing to take on trust. Today a local model trails the commercial one by about 11 points; we have a clear, measured path to close that, and the strongest version is a model fine-tuned on our own code that beats the vendor in our domain.

To do that research and to serve the team's shared index, we need the Mac Studio. We are already memory-bound on the 48GB machine running the smaller model.

---

## Appendix — primary sources

- Claude data usage: code.claude.com/docs/en/data-usage; anthropic.com/legal/commercial-terms
- Voyage: voyageai.com/privacy, voyageai.com/tos, voyageai.com/dpa; MongoDB trust: mongodb.com/trust
- Competitors: augmentcode.com (context engine MCP), docs.sourcebot.dev (MCP server),
  onyx.app, docs.continue.dev, cursor.com, glean.com
- Embedding model research: docs/research/local-embedding-models-2026-06-11.md (repo)
- Pricing: claude-api skill (cached 2026-06-04)
- Increment program / topology: docs/vecs-kb-curation-design-2026-06.md (Inc 5)
- Locate A/B + cost + sovereignty detail: docs/vecs-ab-locate-payload-2026-06-17.md (repo)
