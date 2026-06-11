"""Inc 1-E: stale-retrieval-rate harness + eval-set scaffold (measurement seed).

Defines `stale-retrieval-rate` against the per-chunk ``version_id`` anchor
(Inc 1-C): a stored/retrieved chunk is STALE when its recorded ``version_id`` no
longer matches the ``version_id`` its source file would be stamped with NOW.
Chunks with no recorded ``version_id`` (pre-C, never re-stamped) — or whose
current version can't be computed (source gone, git/IO error) — fall into a
graceful UNKNOWN bucket and are excluded from the rate denominator rather than
guessed.

The "current version" is computed by the SAME rule the indexer stamps with, so
the comparison is apples-to-apples:
  - code chunk -> ``_git_sha(code_dir)``; non-git fallback = ``sha256`` of the
    file BYTES (mirrors ``Manifest._file_hash``, NOT a text hash).
  - docs chunk -> ``str(file mtime)`` (mirrors ``index_docs``).

Seed scope (per acceptance): a harness + a small local eval-set scaffold + a
runner stub, NOT a production dashboard. Collections are read read-only and a
missing collection / unconfigured project / absent source degrades gracefully.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from vecs.config import ProjectConfig, VecsConfig, load_config
from vecs.indexer import _git_sha

FRESH = "fresh"
STALE = "stale"
UNKNOWN = "unknown"


@dataclass
class StaleStats:
    """Fresh/stale/unknown chunk counts + the stale-retrieval-rate."""

    fresh: int = 0
    stale: int = 0
    unknown: int = 0

    @property
    def classified(self) -> int:
        """Chunks we could classify either way (rate denominator)."""
        return self.fresh + self.stale

    @property
    def total(self) -> int:
        return self.fresh + self.stale + self.unknown

    @property
    def rate(self) -> float | None:
        """stale / (fresh + stale). None when nothing is classifiable (all
        unknown, or empty) — distinct from 0.0 (classifiable, none stale)."""
        return self.stale / self.classified if self.classified else None

    def add(self, bucket: str) -> None:
        if bucket not in (FRESH, STALE, UNKNOWN):
            raise ValueError(f"unknown bucket {bucket!r}")
        setattr(self, bucket, getattr(self, bucket) + 1)

    def __iadd__(self, other: "StaleStats") -> "StaleStats":
        self.fresh += other.fresh
        self.stale += other.stale
        self.unknown += other.unknown
        return self


def bucket_chunk(stored_version_id, current_version_id) -> str:
    """Classify one chunk by comparing its stored ``version_id`` to the current.

    UNKNOWN (excluded from the rate) when either side is missing/empty: a chunk
    with no recorded ``version_id`` or a source whose current version can't be
    computed is never guessed. Compared as strings (the indexer stamps strings).
    """
    if not stored_version_id or not current_version_id:
        return UNKNOWN
    return FRESH if str(stored_version_id) == str(current_version_id) else STALE


def _kind_for_collection(collection_name: str) -> str | None:
    if collection_name.endswith("-code"):
        return "code"
    if collection_name.endswith("-docs"):
        return "docs"
    return None  # e.g. -prose-facts: no version_id anchor


def _resolve_source(
    file_path: str, kind: str, proj: ProjectConfig
) -> tuple[Path | None, Path | None]:
    """Map a root-qualified ``{root.name}/{rel}`` back to (source_path, code_root).

    code: the owning root is a ``code_dir`` (code_root returned for the git-sha
    anchor). docs: ``docs_dirs`` win, then in-repo ``code_dirs`` (F routes in-repo
    ``.md`` to ``-docs``), mirroring ``_docs_sources``. Returns (None, None) when
    the leading segment matches no current root.
    """
    if "/" not in file_path:
        return None, None
    root_name, rel = file_path.split("/", 1)
    if kind == "code":
        for cd in proj.code_dirs:
            if cd.path.name == root_name:
                return cd.path / rel, cd.path
        return None, None
    for d in proj.docs_dirs:
        if d.name == root_name:
            return d / rel, None
    for cd in proj.code_dirs:
        if cd.path.name == root_name:
            return cd.path / rel, cd.path
    return None, None


def _current_version_id(file_path: str, kind: str, proj: ProjectConfig) -> str | None:
    """The ``version_id`` the source WOULD be stamped with now, or None if the
    source is unresolvable / missing / errors (-> UNKNOWN, never guessed)."""
    src, code_root = _resolve_source(file_path, kind, proj)
    if src is None:
        return None
    try:
        if not src.exists():
            return None
        if kind == "code":
            sha = _git_sha(code_root) if code_root is not None else None
            if sha:
                return sha
            return hashlib.sha256(src.read_bytes()).hexdigest()
        return str(src.stat().st_mtime)
    except OSError:
        return None


def stale_stats_for_chunks(
    metadatas: list[dict], kind: str, proj: ProjectConfig
) -> StaleStats:
    """Bucket a list of chunk-metadata dicts (each may carry ``version_id`` +
    ``file_path``) into fresh/stale/unknown."""
    stats = StaleStats()
    for meta in metadatas:
        meta = meta or {}
        stored = meta.get("version_id")
        fp = meta.get("file_path")
        current = _current_version_id(fp, kind, proj) if fp else None
        stats.add(bucket_chunk(stored, current))
    return stats


def collection_stale_rate(
    collection_name: str,
    config: VecsConfig | None = None,
    db=None,
) -> StaleStats:
    """stale-retrieval-rate over ALL chunks of one collection (read-only).

    Returns empty stats (``rate is None``) for an absent collection, an
    unconfigured project, or a non-anchored collection (e.g. ``-prose-facts``):
    the harness tolerates a live store that has not been (re)indexed yet.
    """
    config = config or load_config()
    kind = _kind_for_collection(collection_name)
    if kind is None:
        return StaleStats()
    proj_name = collection_name[: -len(kind) - 1]  # strip "-code"/"-docs"
    proj = config.projects.get(proj_name)
    if proj is None:
        return StaleStats()
    if db is None:
        from vecs.clients import get_chromadb_client

        db = get_chromadb_client()
    try:
        col = db.get_collection(collection_name)
        got = col.get(include=["metadatas"])
    except Exception:
        return StaleStats()
    return stale_stats_for_chunks(got.get("metadatas") or [], kind, proj)


# --- eval-set scaffold + runner stub -----------------------------------------


@dataclass
class EvalCase:
    """One eval pair: a query plus the source(s) the answer SHOULD surface from."""

    query: str
    project: str
    collection: str  # "code" | "docs"
    expected_path_substring: str = ""  # legacy single-substring form
    expected: list[str] = field(default_factory=list)
    query_class: str = "nl"  # nl | identifier | concept

    def __post_init__(self):
        if self.expected_path_substring and not self.expected:
            self.expected = [self.expected_path_substring]


# Seed eval set — a handful of query -> expected-source pairs over the vecs
# repo's OWN knowledge base. Scaffold only: extend / repoint per the live store's
# projects. (Mirrors the gated REEMBED_EVAL_SET in tests/test_searcher.py, but is
# runnable offline against an injected search_fn for the stub.)
DEFAULT_EVAL_SET: list[EvalCase] = [
    EvalCase(
        "content-addressable embedding cache keyed by content hash",
        "vecs", "docs", "kb-foundations",
    ),
    EvalCase(
        "prose staleness detector stage 2 semantic recall",
        "vecs", "docs", "prose-staleness-detector",
    ),
    EvalCase(
        "hybrid search reciprocal rank fusion across collections",
        "vecs", "docs", "vecs",
    ),
]


@dataclass
class EvalCaseResult:
    case: EvalCase
    hit: bool
    n_results: int
    sources: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    results: list[EvalCaseResult]
    retrieved_stale: StaleStats

    @property
    def hit_rate(self) -> float | None:
        """Fraction of cases whose expected source appeared in the top-n.
        None for an empty eval set."""
        if not self.results:
            return None
        return sum(1 for r in self.results if r.hit) / len(self.results)


def run_eval(
    eval_set: list[EvalCase] | None = None,
    config: VecsConfig | None = None,
    search_fn=None,
    n_results: int = 5,
) -> EvalReport:
    """Runner STUB: per case, search and check the expected source is in the
    top-n; also accumulate stale-retrieval-rate over the retrieved chunks.

    ``search_fn`` defaults to ``vecs.searcher.search``; inject a fake for offline
    tests. A per-case search failure degrades that case to a miss with zero
    results rather than aborting the run.
    """
    eval_set = DEFAULT_EVAL_SET if eval_set is None else eval_set
    config = config or load_config()
    if search_fn is None:
        from vecs.searcher import search as search_fn  # type: ignore[assignment]

    retrieved_stale = StaleStats()
    results: list[EvalCaseResult] = []
    for case in eval_set:
        try:
            hits = search_fn(
                case.query,
                collection_name=case.collection,
                n_results=n_results,
                project=case.project,
            )
        except Exception:
            hits = []
        sources = [(h.get("metadata") or {}).get("file_path", "") for h in hits]
        hit = any(any(e in (s or "") for e in case.expected) for s in sources)
        results.append(EvalCaseResult(case, hit, len(hits), sources))

        proj = config.projects.get(case.project)
        if proj is not None and case.collection in ("code", "docs"):
            metas = [h.get("metadata") or {} for h in hits]
            retrieved_stale += stale_stats_for_chunks(metas, case.collection, proj)

    return EvalReport(results, retrieved_stale)


# --- L1.1 (local-embed-base): golden-set loader + ranking metrics -------------


def load_eval_set(path: Path) -> list[EvalCase]:
    """Load a golden set YAML: {cases: [{query, project, collection, class, expected: [..]}]}.

    The livly golden set lives OUTSIDE the repo (~/.vecs/evalsets/livly.yaml) --
    work-derived queries/paths must not travel to the repo's remote
    (design.md L1.1). Only the schema and the vecs set are versioned in-repo.
    """
    import yaml

    raw = yaml.safe_load(path.read_text()) or {}
    return [
        EvalCase(
            query=c["query"],
            project=c["project"],
            collection=c["collection"],
            expected=[str(e) for e in c["expected"]],
            query_class=c.get("class", "nl"),
        )
        for c in raw.get("cases", [])
    ]


def _hit_rank(sources: list[str], expected: list[str]) -> int | None:
    """0-based rank of the first source matching ANY expected substring."""
    for i, s in enumerate(sources):
        if any(e in (s or "") for e in expected):
            return i
    return None


def recall_at_k(sources: list[str], expected: list[str], k: int) -> float:
    rank = _hit_rank(sources[:k], expected)
    return 1.0 if rank is not None else 0.0


def mrr(sources: list[str], expected: list[str]) -> float:
    rank = _hit_rank(sources, expected)
    return 0.0 if rank is None else 1.0 / (rank + 1)


def ndcg_at_k(sources: list[str], expected: list[str], k: int) -> float:
    """Binary-relevance nDCG: one relevant doc => DCG = 1/log2(rank+2), IDCG = 1."""
    import math

    rank = _hit_rank(sources[:k], expected)
    return 0.0 if rank is None else 1.0 / math.log2(rank + 2)
