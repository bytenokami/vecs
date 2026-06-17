Authored by Claude

# vecs — HQ deck supporting data (2026-06-17)

Evidence pack for the HQ talk: (1) a measured locate-payload A/B, (2) a per-dev /
per-company cost projection (now vs after vecs vs save), (3) the competitive
landscape + build-vs-buy, (4) the sovereignty (Qwen vs Voyage) tradeoff. Started
as the locate-payload measurement; filename kept (`vecs-ab-locate-payload-2026-06-17.md`).

All numbers are rough and labelled. Quote the *ratios* and the *tradeoff*, not the
dollar figures — the dollars are small; the agent-efficiency and sovereignty
stories are the point.

---

## 1. Locate-payload A/B (measured, read-only)

Measures one thing vecs saves: tokens an agent ingests to **locate** code,
semantic-search vs grep-and-read. Run against the vecs repo's own store (live
index: 36 code + 226 docs chunks, 88 tracked files).

- **Measures:** locate-phase payload — with vecs = the ranked chunks
  `semantic_search` returns; without = the file(s) a grep-driven agent opens whole.
- **Does NOT measure:** a full agent session. Turn-count, latency, and
  context-headroom savings are real and additional, not captured here.
- **Token estimate:** code tokens = bytes ÷ 3 (a floor); vecs payload = returned
  search-text size (~5 chunks/query). Exact `count_tokens` would refine, not
  overturn, the ratio.

### Method
5 realistic "where/how does X work" queries about vecs internals. Per query:
vecs = `semantic_search(collection="code", n_results=5)`; grep = the keyword grep
an agent would run, then open the highest-hit file whole (conservative: one file;
agents often open 2-3, which widens the gap).

### Result (per query)

| # | Query | grep primary file (hits) | grep tok | vecs tok |
|---|-------|--------------------------|---------:|---------:|
| 1 | orphan prune / sweep        | indexer.py (127)    | 24,750 | ~3,000 |
| 2 | embed cache keyed by hash   | embed_cache.py (16) |  1,917 | ~3,000 |
| 3 | model-flip interlock        | indexer.py (25)     | 24,750 | ~3,000 |
| 4 | reciprocal rank fusion      | searcher.py (2)     |  3,735 | ~3,000 |
| 5 | bm25 sync / delete ids      | indexer.py (14)     | 24,750 | ~3,000 |

- **Average locate payload: grep ~16,000 tok → vecs ~3,000 tok. ≈5× less, ~81% reduction, ~13K saved per locate.**
- Buried-in-a-big-file cases (Q1/3/5, answer inside the 1,845-line `indexer.py`):
  grep ~24,750 → vecs ~3,000 = **~8× less**.
- **Honest counter-case (Q2):** answer lived in the small, well-named
  `embed_cache.py` (1.9K tok) — grep ties/beats vecs. vecs's edge is largest
  exactly where grep is worst: logic buried in large files.

---

## 2. Cost projection — now vs after vecs vs save

**Assumptions (state these on the slide):** 40 locate/understand ops per dev/day;
220 working days/year; Opus 4.8 input = $5/1M; locate-phase **input tokens only**
(output, targeted edits, conversation are not addressed by vecs); per-query Voyage
embed cost is negligible (~$0.0001/day). Reconciled to the measured 16K→3K.

| Scope | NOW (no vecs) | AFTER vecs | SAVE |
|---|---|---|---|
| Per dev / day  | 640K tok · **$3.20** | 120K tok · **$0.60** | 520K tok · **$2.60** (81%) |
| Per dev / year | 140.8M tok · **$704** | 26.4M tok · **$132** | 114.4M tok · **$572** |
| 10 devs / year | 1.41B tok · **$7,040** | 264M tok · **$1,320** | 1.14B tok · **$5,720** |

**Cache-realism:** re-read context bills at ~0.1× input, so effective price is
~$0.5-1.5/1M, not $5. Divide the $ above by ~3-10 → 10-dev/year saving ≈
**$0.6-1.7K**. The dollar saving is modest; the real wins are fewer turns
(3-5/locate → faster), context-window headroom (fewer compactions, longer
sessions), and answer quality (semantic recall > grep). Those serve the
agent-efficiency north star and don't dollarize cleanly.

**Caveats:** locate-phase only (total daily Claude spend is higher); 40 ops/day is
an assumption — scale linearly with the real number; full-price $ is the ceiling;
"after vecs" assumes the 3K payload holds at scale (bigger repos may shift it);
excludes vecs's own cost (Voyage ~$5/full embed, or Qwen compute).

For a production figure: run Inc-1-A metering + Inc-1-E + the A/B harness over the
livly corpus.

---

## 3. Market landscape — build vs buy (2026)

Three categories; none lands on vecs's exact spot.

- **Code assistants** (Cody, Cursor, Copilot, Continue, Tabby, Augment, Sourcebot)
  — built for code, but mostly IDE-embedded or cloud.
- **Enterprise wiki-RAG** (Glean, Onyx/Danswer, Dust, Notion AI, Rovo, M365
  Copilot) — built for humans searching docs; code = connector-level text only.
- **Vector DBs + frameworks** (Pinecone, Qdrant, Chroma, Weaviate; LlamaIndex,
  txtai; PrivateGPT, LocalGPT) — components/DIY, not a finished product. vecs is
  built *on* one (Chroma).

### vecs vs the 5 nearest

| | self-host / sovereign | agent MCP | hybrid vec+BM25 | code-aware (AST + code-embed) | docs too | OSS | maturity |
|---|---|---|---|---|---|---|---|
| **vecs** | ✅ | ✅ | ✅ | ✅ | ✅ | own | low (bespoke) |
| Sourcebot | ✅ | ✅ | ❌ keyword/trigram | symbol ✅ / dense ❌ | ❌ | ✅ | med |
| Augment | remote ❌ / local ~ | ✅ | ? unconfirmed | ✅ | ? | ❌ | high |
| Onyx (Danswer) | ✅ | ✅ | ✅ | ❌ connector-text | ✅ | ✅ | high |
| Cody (Sourcegraph) | ❌ SaaS | API (not MCP) | ✅ | ✅ | ~ | ❌ | high |
| Continue.dev | ✅ local | ❌ IDE-bound | ✅ | ✅ | ~ | ✅ | med |

**Verdict:** the intersection vecs occupies — local + hybrid-dense +
MCP-agent-facing + code AND docs + code-tuned embeddings — is unoccupied by any
single off-the-shelf product. Each nearest miss drops one axis: Sourcebot drops
dense vector, Augment drops sovereignty (cloud) + closed, Onyx drops code-semantic
quality (code as text), Cody drops self-host, Continue drops MCP (IDE-locked).

**Honest caveats (say these — they age badly otherwise):**
- The gap is **closing fast**: Augment shipped an MCP context engine (Feb 2026),
  Sourcebot is OSS MCP-native, Zilliz "Claude Context MCP" does BM25+dense,
  Chroma/Qdrant/Pinecone all ship MCP. Don't slide "nothing like this exists."
- Durable build-case is NOT "it doesn't exist." It is: **(1) data sovereignty**
  (Augment/Cody/Glean touch cloud), **(2) code-semantic retrieval quality**
  (Onyx/Glean treat code as text — no function/class chunking, no code-tuned
  embeddings), **(3) we own the eval + pipeline** (caught/fixed our own
  "retrieves deleted content" bug; tune to a measured locate metric).
- Where **buy wins outright**: connectors/breadth (Glean, Onyx ingest
  Slack/Jira/Drive — vecs none), human browser UI (vecs is agent-only),
  RBAC/SSO/multi-tenant (vecs zero today), scale + maturity.

---

## 4. Sovereignty — what actually leaves, and under what terms

### 4.0 The #1 code-egress path is the agent itself (verified 2026-06-17)

Before the embedder debate: the biggest channel by which our code leaves is the
**coding agent**, which reads every file it touches. On **personal Claude Pro/Max
subscriptions that code is used for training by default.** Verbatim
(code.claude.com/docs/en/data-usage): "We will train new models using data from
Free, Pro, and Max accounts when this setting is on (**including when you use Claude
Code from these accounts**)." The training toggle **defaults ON**; retention is
**5 years** when on, 30 days when off. There is **no org-level control** at Pro/Max —
only a per-developer toggle. API/Console keys + Team/Enterprise are **no-train by
policy** (commercial terms: "Anthropic may not train models on Customer Content").

**Current de-facto state:** if the team runs Claude Code on personal subs and nobody
flipped the toggle, company code (incl. livly) is in Anthropic's training pipeline,
5-year retention — far more code than Voyage ever sees. Like the Voyage opt-out,
turning it off is **not retroactive**: stops future training, not past sessions
already ingested. This is an IP flag for HQ/legal independent of the embedder.

**Priority order this implies (reverses the naive one):**
1. **Agent tier — biggest, do first:** move the team to **API/Console keys or
   Team/Enterprise** (org-enforced no-train); stopgap today = every dev turns
   training OFF at claude.ai/settings/data-privacy-controls (5yr→30d).
2. **Embedder:** Voyage opt-out → local Qwen (a smaller surface than the agent).

Honest deck framing: sovereignty is an **egress-minimization policy** — every party
that sees our code held to no-train terms, **agent FIRST**, then the embedder.
Leading with "we built vecs so code stays local" while the agent trains-by-default
on personal subs is the contradiction HQ will spot.

---

### 4.1 Embedder — Qwen vs Voyage

The competitive case leans on **sovereignty**. But the L2 A/B (ran 2026-06-17 on
the team's work mac) shows the tension is real and currently unresolved:

- **Qwen-4B@mrl1024 LOSES to Voyage ≈11 pts recall@10.** Split: natural-language
  queries **−13** (Qwen worse), ID/symbol queries **+14** (Qwen better). CIs
  exclude 0 → statistically real. Latency fine (44ms). Lean: keep-Voyage for now.
- **You can't split the embedder** — query model must equal index model — so it's
  Voyage (quality, text leaves to a third-party API) **or** Qwen (full local
  sovereignty, −11 today), not a blend.

So: **ship Voyage → "nothing leaves your machine" is false.** Don't put that line
on a slide while on Voyage. But this is a strength to present, not a hole:

1. **−11 is the naked-4B number, not the ceiling.** Three rescues not yet run:
   Qwen3-**Reranker-4B** (research: +5.8 pts on code), higher MRL dim (2560 vs
   1024), RRF weight sweep. The gap is concentrated in NL queries — what the
   reranker targets — while Qwen already **wins** on ID/symbol. Note: **vecs uses
   no reranker today** (grep-confirmed, zero `rerank` refs in `src/`); both Voyage's
   reranker and Qwen3-Reranker-4B are available-but-unused, so a reranker is a lever
   for *either* path — and turning on Voyage's would also raise the bar Qwen must beat.
2. **Optionality is the moat, and it survives on Voyage.** vecs-on-Voyage leaks no
   more than the cloud competitors (Augment/Cody/Glean all send data out) — and
   vecs is the **only** one that can flip to fully air-gapped (local Qwen) without
   changing vendors. The embedder is a swappable seam (shipped in L1).
3. **It's a config flip, not a rebuild** — Voyage today, Qwen when the rescues
   close the gap; same store, same pipeline.

**Voyage data policy (verified 2026-06-17, primary source):** no-train is an
**opt-out, NOT the default — and the default grant is strong.** Voyage's Privacy
Policy (last updated 2025-02-20, voyageai.com/privacy): unless you opt out, you
grant Voyage "a worldwide, irrevocable, **perpetual**, royalty-free, fully paid-up
... license to use, copy, reproduce ... prepare derivative works of ... the
Customer Content ... to train, improve, and otherwise further develop the Service
(such as by training the artificial intelligence models we use)." Two stings:
(a) it is a **perpetual** license by default; (b) **opting out is NOT retroactive**
— "it will apply only to Customer Content you submit after the time at which you
opt out", so any corpus already embedded without opting out is covered. The FAQ
adds the opt-out also yields "zero-day retention" via a self-serve dashboard toggle
(paying org + Admin). The **ToS (last updated 2026-05-27, directly
fetched) confirms** zero-day deletion applies ONLY post-opt-out ("your Customer
Content provided after such opt out will be immediately deleted ... after it is
processed"); the **pre-opt-out retention window stays unstated** (the one open gap),
and prior data "may continue to be subject to" the training license. A **DPA exists**
(voyageai.com/dpa, incorporated by ToS §6, covers the embeddings API), naming
sub-processors **AWS + Google**, region **United States**. **MongoDB does NOT govern
this** — Voyage runs standalone ToS/DPA and is not named in MongoDB's SOC 2 / ISO /
DPA, so its certs can't be leaned on. Voyage's own SOC 2 / ISO status sits on its
Vanta trust portal (app.vanta.com/voyageai.com/trust), but that page is JS-rendered
and could not be read by the fetch tool — **confirm certs manually**. Sources
(license/retention verified by direct fetch of voyageai.com/tos + /privacy;
sub-processors/region from /dpa): voyageai.com/tos, /privacy, /dpa,
docs.voyageai.com/docs/faq.

**So the accurate slide line is NOT "Voyage doesn't retain or train on our data."**
It is: "we have enabled Voyage's zero-retention opt-out" — and that is contingent on
actually having flipped it.

**Action items before the deck:**
1. **Confirm the no-train / zero-retention opt-out is ENABLED** on our
   Voyage/MongoDB account (dashboard toggle; needs Admin + a payment method).
   **The opt-out is NOT retroactive** — it only covers content submitted *after*
   opting out. If the livly corpus was embedded before opting out, that text is
   already under Voyage's perpetual training license. Default (no opt-out) = inputs
   licensed perpetually for training.
2. Mostly resolved: Voyage has its own ToS + self-serve **DPA** (voyageai.com/dpa)
   covering the embeddings API; sub-processors AWS + Google, region US; MongoDB's
   umbrella does NOT cover the standalone Voyage API. **Still open:** pre-opt-out
   retention duration (unstated in ToS/DPA), and Voyage's SOC 2 / ISO status (on its
   Vanta portal, not machine-readable — confirm manually). If we keep Voyage, **sign
   the DPA**.

**Strongest honest posture — opt out from day one.** If the opt-out is enabled
*before* any embed, every submission is post-opt-out: Voyage does not train on it and
deletes it immediately after processing (zero-day). That neutralizes the
perpetual-license + non-retroactive problem going forward, and the accurate claim
becomes "our code is processed transiently under Voyage's DPA, not retained, not
trained on." Two honesty caveats: (a) it still **transits** Voyage (US, AWS/Google)
to be embedded — strong "not retained/trained", not "never leaves"; (b) it is a
**contractual** promise (trust + DPA), not a technical guarantee — local Qwen is the
trust-free version. **Verify first:** our corpus was already embedded (livly, vecs)
over prior weeks — if the opt-out was NOT on then, that already-sent text is under the
perpetual license regardless (re-embedding cannot un-grant it); opting out now
protects only future submissions.

**Pick one honest deck posture (don't claim full sovereignty on Voyage):**
- *Ship-today:* "Runs on our hardware; embedding calls go to Voyage with their
  zero-retention opt-out enabled (it is opt-out, not their default). Full air-gap
  available via local Qwen, in evaluation."
- *Endgame:* "Fully sovereign once local-embed closes a measured 11-pt gap
  (reranker/dim/RRF, in eval)."

**Net:** the verified finding *strengthens* the Qwen endgame — going local removes
the opt-out dependency AND the unverified-umbrella risk entirely.
