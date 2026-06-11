Authored by Claude

# Golden eval sets

Schema (one YAML per project):

    cases:
      - query: <natural-language or identifier query>
        project: <vecs config project name>
        collection: code | docs
        class: nl | identifier | concept
        expected: [<path substring>, ...]   # ANY match in metadata.file_path = hit

## Rules (design: docs/features/local-embed/design.md §L1.1)

- **livly's golden set NEVER enters this repo.** It lives on the work mac only
  at `~/.vecs/evalsets/livly.yaml` — its queries and expected paths describe
  work internals, and this repo's remote is a personally-hosted deploy channel.
  Exported A/B reports contain aggregate metrics only (`ab_report` emits no
  chunk text or paths).
- **Authoring protocol:** derive queries from real information needs WITHOUT
  running vecs search while authoring (selecting queries the incumbent already
  answers rigs the A/B). Before scoring an A/B, complete the expected sets by
  pooling top-10 from all arms and adjudicating relevance.
- **Freeze:** the set is frozen at a recorded commit before any A/B run. Later
  edits require a changelog line here + re-running all arms.

## Changelog

- 2026-06-11: `vecs.yaml` authored (46 cases) per protocol; frozen pre-A/B.
