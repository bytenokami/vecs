Authored by Claude

# Local (open-weights) embedding models to replace Voyage AI — deep-research report

**Date:** 2026-06-11. **Method:** 5-angle web sweep (code benchmarks, prose leaderboards, specs/licensing, Apple Silicon practice, rerankers/field reports), 21 sources fetched, 104 claims extracted, top 25 adversarially verified by 3-vote panels (20 confirmed, 5 killed). Consumed by `docs/features/local-embed/design.md`.

**Question:** best open-weights, on-device replacement for voyage-code-3 (code) and voyage-4 (prose/docs) in vecs, running on Apple Silicon, as of June 2026. Constraints: 1024-dim pinned store (Chroma collections recreated on swap, so other dims possible), commercial/team license required.

## Recommendation (confidence: medium — verdict inferential, see gaps)

- **Primary:** **Qwen3-Embedding-4B**, MRL-truncated to **1024 dims**, for both code and docs. 8B if RAM allows (+~0.7–1.1 pts across benchmarks).
- **Reranker:** **Qwen3-Reranker-4B** for code queries (verified +5.8 pts MTEB-Code over embedding-only). Do NOT use the 0.6B reranker on code — it scores *below* the embedding-only baseline. The 8B reranker adds only +0.02 over 4B.
- **Cheap tier:** **Qwen3-Embedding-0.6B** at native 1024 dims — exact drop-in for the pinned store, MTEB-Code 75.41, still above Gemini-Embedding (74.66), the prior proprietary SOTA.
- **Serving stack:** sentence-transformers with `truncate_dim` is the only MRL path verified working (vLLM rejects the dimensions param without `--hf-overrides`; config.json lacks `is_matryoshka: true`). MLX / Ollama / llama.cpp performance on Apple Silicon is entirely unverified.
- **License:** all six Qwen3 Embedding + Reranker models (0.6B/4B/8B × 2) are **Apache 2.0** (verified live on all HF cards + Qwen blog). Clear for commercial/team use.

## Verified findings

| # | Finding | Confidence |
|---|---|---|
| 1 | Qwen3-Embedding is the strongest verified open-weights family on MTEB-Code: 8B = **80.68**, 0.6B = **75.41**, both above Gemini-Embedding 74.66 (prior proprietary SOTA, Table 3 of the tech report). No Voyage model appears in any Qwen3 table. | high |
| 2 | MRL dims: native 1024 (0.6B) / 2560 (4B) / 4096 (8B), user-truncatable 32→native; 32K context. 0.6B natively emits 1024-dim. | high |
| 3 | Licensing: all six Qwen3 embed+rerank models Apache 2.0. (Open issue QwenLM/Qwen3-Embedding#166 about MS MARCO training-data terms — does not change the Apache grant on weights.) | high |
| 4 | Prose (voyage-4 side): 8B = 70.58 MTEB-Multilingual Mean(Task) (#1 at June 2025 release; still top broadly-licensed open model in a May 2026 aggregator snapshot, #2 overall behind Tencent KaLM-Gemma3-12B which carries an explicit not-for-EU clause) and **75.22 MTEB-English-v2**; 4B = 69.45/74.60; 0.6B = 64.33/70.70. 4B/8B beat gemini-embedding-exp-03-07 (68.37/73.3). | high |
| 5 | Reranker on MTEB-Code (rerank top-100 from 0.6B-embed baseline 75.41): 4B = **81.20**, 8B = 81.22, 0.6B = **73.42 (hurts)**. Competing open rerankers far lower (Jina-v2-base 58.98, BGE-v2-m3 41.38). Caveat: Qwen self-run figures; independent evals flag 4B weaker on some general-domain text benchmarks + high autoregressive-decoding latency. | high |
| 6 | Runner-up (code): **nomic-embed-code 7B** — fully Apache 2.0 across weights+training code+data. But quality claim is vendor-framed, CodeSearchNet-only; per-language it beats voyage-code-3 on Python/PHP/Go, ties Java, **loses Ruby (81.8 vs 84.6) and JavaScript (77.1 vs 79.2)**. ~14GB fp16, 2048-token context. | high |
| 7 | **CodeXEmbed/SFR disqualified:** CoIR SOTA claim was vs the obsolete Voyage-Code-002; the 7B weights were never released; released 400M/2B are CC-BY-NC-4.0 research-only. | high |
| 8 | **EmbeddingGemma (308M) dominated:** max 768 dims (can't hit 1024), scores strictly below Qwen3-0.6B on code/English/multilingual. No successor as of June 2026. | high |
| 9 | Incumbent: voyage-code-3 is hosted-only (API/SageMaker/Azure), no open weights. Voyage's only open-weights release is **voyage-4-nano** (0.3B, general-purpose, 2026-01-15) — does not cover code, unbenchmarked in surviving claims. | high |

## Honest gaps (drive the local spike in Inc local-embed)

1. **No head-to-head Qwen3-vs-Voyage exists anywhere.** voyage-code-3's headline is CoIR ~77.3 NDCG@10 — overlapping but not identical to MTEB-Code, so numbers can't share an axis. "Likely parity or better" is triangulation through Gemini-Embedding only. The A/B eval on our own corpus is the only real source of truth.
2. **Zero verified Apple Silicon performance data** — tokens/sec, RAM, 10–15K-chunk reindex wall time, rerank-50 latency: every claim failed verification. Needs an empirical spike.
3. MRL-truncation cost (4B @ 2560→1024) on code retrieval is unmeasured vs 0.6B native-1024.
4. Post-mid-2025 contenders (jina v5-text etc.) effectively unassessed — the aggregator blogs claiming them failed verification 0-3. All Qwen3 numbers are vendor self-runs / June-2025 MTEB snapshot.

## Refuted claims (3-vote panels)

- "Qwen3-8B ranks #1 MTEB multilingual as of April 2026" (mixpeek) — 0-3.
- "Jina v5-text-small 71.7 MTEB v2, Apache 2.0" (mixpeek) — 0-3.
- "Jina v4 is CC-BY-NC" (bentoml) — 0-3.
- awesomeagents "April 2026 leaderboard" — 0-3.
- modal.com nomic per-language figures — 1-2 (superseded by HF-card-verified figures in finding 6).

## Key sources

Primary: arxiv.org/pdf/2506.05176 (Qwen3-Embedding tech report), github.com/QwenLM/Qwen3-Embedding, HF model cards (all six Qwen3 + nomic + EmbeddingGemma + SFR), arxiv.org/pdf/2411.12644 (CodeXEmbed), docs.voyageai.com, nomic.ai announcement, ai.google.dev EmbeddingGemma card. Secondary cross-checks: codesota.com MTEB aggregate (2026-05-17), vllm issue #20899 (MRL serving), QwenLM blog.
