Authored by Claude

# local-embed-base (L1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the no-regret base of Inc 1.7 (spec: `docs/features/local-embed/design.md` §L1): provider abstraction with a Qwen local provider, persisted provider config, code-collection model markers with backfill, Chroma telemetry off, and the golden-set + A/B measurement machinery — with `embed_provider: voyage` default, zero behavior change at merge.

**Architecture:** A new `EmbedProvider` seam (`embed_provider.py`) replaces direct `voyageai` calls at the three embed call sites (indexer, searcher, prose_drift). Model-id strings stay the cache/marker key, so voyage/qwen vectors segregate by construction. The search pipeline is extracted into a parameterized `search_collections()` so the A/B harness runs the production path against arbitrary collections. Eval metrics + paired bootstrap CI extend `eval_harness.py`.

**Tech Stack:** Python 3.12, ChromaDB, SQLite FTS5, voyageai, sentence-transformers (optional extra `vecs[local]`), pytest. Runner: `uv run pytest -q`.

**Verified line refs are against commit `fcba93d`.** If they drift, search for the quoted code.

---

### Task 1: Chroma telemetry off (dry-run task per design Phase 7)

**Files:**
- Modify: `src/vecs/clients.py:20-26`
- Modify: `src/vecs/prose_drift.py:360-371` (the two `chromadb.PersistentClient` sites)
- Test: `tests/test_telemetry.py` (create)

- [ ] **Step 1: Write the failing tests**

```python
"""Chroma telemetry must be off at every PersistentClient construction site."""
from unittest.mock import MagicMock

import vecs.clients as clients
import vecs.prose_drift as prose_drift


def _capture_pc(captured):
    def fake_pc(path, settings=None):
        captured["settings"] = settings
        client = MagicMock()
        return client
    return fake_pc


def test_get_chromadb_client_disables_telemetry(monkeypatch):
    captured = {}
    monkeypatch.setattr(clients.chromadb, "PersistentClient", _capture_pc(captured))
    monkeypatch.setattr(clients, "_db_client", None)
    clients.get_chromadb_client()
    assert captured["settings"] is not None
    assert captured["settings"].anonymized_telemetry is False
    monkeypatch.setattr(clients, "_db_client", None)


def test_prose_facts_collection_disables_telemetry(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(prose_drift.chromadb, "PersistentClient", _capture_pc(captured))
    monkeypatch.setattr(prose_drift, "_chroma_path", lambda: tmp_path)
    prose_drift._get_prose_facts_collection("p")
    assert captured["settings"].anonymized_telemetry is False


def test_docs_collection_disables_telemetry(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(prose_drift.chromadb, "PersistentClient", _capture_pc(captured))
    monkeypatch.setattr(prose_drift, "_chroma_path", lambda: tmp_path)
    prose_drift._get_docs_collection("p")
    assert captured["settings"].anonymized_telemetry is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest -q tests/test_telemetry.py`
Expected: 3 FAIL (`captured["settings"] is None` — no settings passed today).

- [ ] **Step 3: Implement**

In `src/vecs/clients.py` add the import and pass settings:

```python
from chromadb.config import Settings
```

```python
        _db_client = chromadb.PersistentClient(
            path=str(CHROMADB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
```

In `src/vecs/prose_drift.py` add `from chromadb.config import Settings` to the imports, and in BOTH `_get_prose_facts_collection` and `_get_docs_collection` replace `chromadb.PersistentClient(path=str(path))` with:

```python
    client = chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: all green (389+ tests; the new 3 pass).

- [ ] **Step 5: Commit**

```bash
git add src/vecs/clients.py src/vecs/prose_drift.py tests/test_telemetry.py
git commit -m "feat(clients): disable ChromaDB anonymized telemetry at all client sites"
```

---

### Task 2: `embed_provider` config field + qwen model ids in EMBED_DIMS

**Files:**
- Modify: `src/vecs/config.py:27-38` (EMBED_DIMS + comment), `:93-148` (VecsConfig field + save), `:176-178` (load)
- Test: `tests/test_config.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
class TestEmbedProvider:
    def test_default_is_voyage_when_absent(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("projects: {}\n")
        cfg = load_config(p)
        assert cfg.embed_provider == "voyage"

    def test_round_trips_through_save(self, tmp_path):
        p = tmp_path / "config.yaml"
        cfg = VecsConfig(path=p)
        cfg.embed_provider = "qwen-local"
        cfg.save()
        loaded = load_config(p)
        assert loaded.embed_provider == "qwen-local"

    def test_save_after_load_preserves_provider(self, tmp_path):
        """The add_document auto-configure path (load -> mutate projects -> save)
        must NOT strip the provider field (design.md L1.2: save() currently
        rewrites the whole file)."""
        p = tmp_path / "config.yaml"
        p.write_text("embed_provider: qwen-local\nprojects: {}\n")
        cfg = load_config(p)
        cfg.add_project("x", code_dirs=[CodeDir(path=tmp_path, extensions={".py"})])
        cfg.save()
        assert "qwen-local" in p.read_text()
        assert load_config(p).embed_provider == "qwen-local"

    def test_qwen_model_ids_in_embed_dims(self):
        from vecs.config import EMBED_DIMS
        assert EMBED_DIMS["qwen3-embedding-4b@mrl1024"] == 1024
        assert EMBED_DIMS["qwen3-embedding-0.6b"] == 1024
```

(Use the file's existing imports; it already imports `VecsConfig`, `CodeDir`, `load_config` — add any missing.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest -q tests/test_config.py -k EmbedProvider`
Expected: FAIL — `VecsConfig` has no attribute `embed_provider`; KeyError on EMBED_DIMS.

- [ ] **Step 3: Implement in `src/vecs/config.py`**

Extend EMBED_DIMS (keep the existing three entries and the existing comment; append below them inside the dict):

```python
EMBED_DIMS = {
    "voyage-3": 1024,
    "voyage-4": 1024,
    "voyage-code-3": 1024,
    # Local (Qwen3) model ids — for these, the dim is PRESCRIPTIVE, not
    # descriptive: QwenLocalProvider truncates to this width (MRL). A dim
    # change REQUIRES a new model id — the embed cache and collection_models
    # markers key on the id string alone, so editing a dim in place would
    # serve stale-width cached vectors with zero invalidation.
    "qwen3-embedding-4b@mrl1024": 1024,
    "qwen3-embedding-0.6b": 1024,
}
```

Also update the comment block above the dict (`config.py:27-33`): after "we never send an output_dimension override" add "— for Voyage models. Qwen entries below are prescriptive (see inline note)."

Add the field to `VecsConfig`:

```python
@dataclass
class VecsConfig:
    """Top-level config holding all projects."""
    path: Path
    projects: dict[str, ProjectConfig] = field(default_factory=dict)
    # Which embedding provider serves all embed calls: "voyage" (hosted API,
    # default) or "qwen-local" (on-device, optional extra vecs[local]).
    # MUST round-trip through save() — save() rewrites the whole file and runs
    # on every add_document auto-configure; an unmodeled field would be
    # silently stripped, reverting the fleet to voyage (design.md L1.2).
    embed_provider: str = "voyage"
```

In `save()` change the data init line to:

```python
        data: dict = {"embed_provider": self.embed_provider, "projects": {}}
```

In `load_config()` after `config = VecsConfig(path=path)` (line 176) add:

```python
    config.embed_provider = raw.get("embed_provider", "voyage") if (raw := yaml.safe_load(path.read_text()) or {}) else "voyage"
```

— careful: `raw` is already loaded on the next line in current code. Correct minimal edit: keep the existing `raw = yaml.safe_load(...)` line, then add after it:

```python
    config.embed_provider = raw.get("embed_provider", "voyage")
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: green. (Existing save() tests that assert exact file content may need the new `embed_provider: voyage` top-level line — update those assertions if any fail; that IS the intended new persisted shape.)

- [ ] **Step 5: Commit**

```bash
git add src/vecs/config.py tests/test_config.py
git commit -m "feat(config): persisted embed_provider field + qwen model ids in EMBED_DIMS"
```

---

### Task 3: `embed_provider.py` — EmbedResult, protocol, VoyageProvider, factory

**Files:**
- Create: `src/vecs/embed_provider.py`
- Test: `tests/test_embed_provider.py` (create)

- [ ] **Step 1: Write the failing tests**

```python
"""Provider seam: VoyageProvider wraps the voyage client verbatim; the factory
routes on config.embed_provider; unknown names fail loud."""
from unittest.mock import MagicMock

import pytest

from vecs.config import VecsConfig
from vecs.embed_provider import EmbedResult, VoyageProvider, get_provider


def _fake_vo(n_embeddings=2, total_tokens=99):
    vo = MagicMock()
    result = MagicMock()
    result.embeddings = [[0.1] * 4 for _ in range(n_embeddings)]
    result.usage.total_tokens = total_tokens
    vo.embed.return_value = result
    return vo


def test_voyage_provider_passes_through_model_and_input_type():
    vo = _fake_vo()
    p = VoyageProvider(client=vo)
    out = p.embed(["a", "b"], model="voyage-code-3", input_type="document")
    vo.embed.assert_called_once_with(["a", "b"], model="voyage-code-3", input_type="document")
    assert isinstance(out, EmbedResult)
    assert out.embeddings == [[0.1] * 4, [0.1] * 4]
    assert out.total_tokens == 99


def test_voyage_provider_tolerates_missing_usage():
    vo = MagicMock()
    result = MagicMock(spec=["embeddings"])
    result.embeddings = [[0.1]]
    vo.embed.return_value = result
    out = VoyageProvider(client=vo).embed(["a"], model="voyage-4", input_type="query")
    assert out.total_tokens is None


def test_voyage_provider_retryable_errors_match_voyage_exceptions():
    import voyageai.error
    errs = VoyageProvider(client=MagicMock()).retryable_errors
    assert voyageai.error.RateLimitError in errs
    assert voyageai.error.Timeout in errs


def test_factory_routes_on_config(tmp_path):
    cfg = VecsConfig(path=tmp_path / "c.yaml")
    assert isinstance(get_provider(cfg), VoyageProvider)


def test_factory_unknown_name_raises():
    with pytest.raises(ValueError, match="qwen-local|voyage"):
        get_provider(name="banana")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest -q tests/test_embed_provider.py`
Expected: FAIL — `ModuleNotFoundError: vecs.embed_provider`.

- [ ] **Step 3: Create `src/vecs/embed_provider.py`**

```python
"""Embedding-provider seam (Inc 1.7 L1, design: docs/features/local-embed/design.md).

One protocol, two implementations: VoyageProvider (hosted API, the default) and
QwenLocalProvider (on-device sentence-transformers, behind the optional extra
``vecs[local]``). The three embed call sites (indexer, searcher, prose_drift)
route through a provider, so an embedding-model swap is a config flip + reindex,
not a code change. Model-id strings remain the embed-cache / collection-marker
key, which keeps voyage and qwen vectors segregated by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vecs.config import EMBED_DIMS, VecsConfig


@dataclass
class EmbedResult:
    """Provider-neutral embed response."""

    embeddings: list[list[float]]
    # None when the provider cannot report token usage (drives AdaptiveBatcher
    # calibration only — there is no embed metering path).
    total_tokens: int | None


class EmbedProvider(Protocol):
    name: str
    # Transient errors the indexer's retry loop should sleep-and-retry on.
    retryable_errors: tuple[type[BaseException], ...]

    def embed(
        self, texts: list[str], *, model: str, input_type: str
    ) -> EmbedResult: ...


class VoyageProvider:
    """Wraps the existing singleton voyage client verbatim."""

    name = "voyage"

    def __init__(self, client=None):
        self._client = client

    @property
    def retryable_errors(self) -> tuple[type[BaseException], ...]:
        import voyageai.error

        return (
            voyageai.error.Timeout,
            voyageai.error.APIConnectionError,
            voyageai.error.RateLimitError,
            voyageai.error.ServiceUnavailableError,
            voyageai.error.ServerError,
            voyageai.error.TryAgain,
            voyageai.error.APIError,
        )

    def embed(self, texts: list[str], *, model: str, input_type: str) -> EmbedResult:
        if self._client is None:
            from vecs.clients import get_voyage_client

            self._client = get_voyage_client()
        result = self._client.embed(texts, model=model, input_type=input_type)
        total = getattr(getattr(result, "usage", None), "total_tokens", None)
        return EmbedResult(embeddings=result.embeddings, total_tokens=total)


def get_provider(
    config: VecsConfig | None = None, name: str | None = None
) -> EmbedProvider:
    """Resolve the configured provider. ``name`` overrides config (tests, A/B arms)."""
    if name is None:
        if config is None:
            from vecs.config import load_config

            config = load_config()
        name = config.embed_provider
    if name == "voyage":
        return VoyageProvider()
    if name == "qwen-local":
        return QwenLocalProvider()
    raise ValueError(
        f"Unknown embed_provider {name!r} (expected 'voyage' or 'qwen-local')"
    )
```

(`QwenLocalProvider` lands in Task 4 — for this commit add a stub so the factory line imports:)

```python
class QwenLocalProvider:  # implemented in Task 4
    name = "qwen-local"
    retryable_errors: tuple[type[BaseException], ...] = ()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q tests/test_embed_provider.py`
Expected: PASS (5).

- [ ] **Step 5: Commit**

```bash
git add src/vecs/embed_provider.py tests/test_embed_provider.py
git commit -m "feat(embed): EmbedProvider seam with VoyageProvider + factory"
```

---

### Task 4: QwenLocalProvider + `vecs[local]` extra

**Files:**
- Modify: `src/vecs/embed_provider.py` (replace the stub)
- Modify: `pyproject.toml`
- Test: `tests/test_embed_provider.py` (append)

- [ ] **Step 1: Write the failing tests** (append; a fake SentenceTransformer is injected — no model download, no torch import in CI)

```python
class _FakeST:
    """Stands in for sentence_transformers.SentenceTransformer."""

    def __init__(self, dim=1024):
        self.dim = dim
        self.calls = []

        class _Tok:
            def __call__(self, text):
                return {"input_ids": list(range(len(text.split())))}

        self.tokenizer = _Tok()

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return [[0.5] * self.dim for _ in texts]


def _qwen(dim=1024):
    from vecs.embed_provider import QwenLocalProvider

    fake = _FakeST(dim=dim)
    return QwenLocalProvider(model_loader=lambda model_id: fake), fake


def test_qwen_documents_have_no_prompt_queries_get_query_prompt():
    p, fake = _qwen()
    p.embed(["doc text"], model="qwen3-embedding-0.6b", input_type="document")
    p.embed(["a query"], model="qwen3-embedding-0.6b", input_type="query")
    doc_kwargs, query_kwargs = fake.calls[0][1], fake.calls[1][1]
    assert "prompt_name" not in doc_kwargs
    assert query_kwargs["prompt_name"] == "query"
    assert doc_kwargs["normalize_embeddings"] is True


def test_qwen_counts_tokens_with_hf_tokenizer():
    p, _ = _qwen()
    out = p.embed(["one two three", "four five"], model="qwen3-embedding-0.6b", input_type="document")
    assert out.total_tokens == 5


def test_qwen_asserts_dim_matches_embed_dims():
    p, _ = _qwen(dim=768)
    import pytest as _pytest

    with _pytest.raises(ValueError, match="768.*1024|1024.*768"):
        p.embed(["x"], model="qwen3-embedding-0.6b", input_type="document")


def test_qwen_unknown_model_id_fails_loud():
    p, _ = _qwen()
    import pytest as _pytest

    with _pytest.raises(ValueError, match="qwen3-embedding"):
        p.embed(["x"], model="voyage-4", input_type="document")


def test_qwen_loads_model_once_per_id():
    from vecs.embed_provider import QwenLocalProvider

    loads = []

    def loader(model_id):
        loads.append(model_id)
        return _FakeST()

    p = QwenLocalProvider(model_loader=loader)
    p.embed(["a"], model="qwen3-embedding-0.6b", input_type="document")
    p.embed(["b"], model="qwen3-embedding-0.6b", input_type="document")
    assert loads == ["qwen3-embedding-0.6b"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q tests/test_embed_provider.py -k qwen`
Expected: FAIL — stub takes no `model_loader`, has no `embed`.

- [ ] **Step 3: Replace the stub in `src/vecs/embed_provider.py`**

```python
# Qwen model id -> Hugging Face repo. The "@mrl1024" suffix is naming
# convention; the ENFORCED dim source is EMBED_DIMS (asserted on first batch).
_QWEN_HF_REPOS = {
    "qwen3-embedding-4b@mrl1024": "Qwen/Qwen3-Embedding-4B",
    "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
}


class QwenLocalProvider:
    """On-device Qwen3 embeddings via sentence-transformers (verified MRL path:
    ``truncate_dim``). Heavy deps live behind the ``vecs[local]`` extra and are
    imported lazily — selecting this provider without the extra fails with an
    actionable message. Local inference has no transient network errors, so
    ``retryable_errors`` is empty.
    """

    name = "qwen-local"
    retryable_errors: tuple[type[BaseException], ...] = ()

    def __init__(self, model_loader=None):
        self._models: dict[str, object] = {}
        self._loader = model_loader or _load_sentence_transformer
        self._dim_checked: set[str] = set()

    def _get_model(self, model_id: str):
        if model_id not in _QWEN_HF_REPOS:
            raise ValueError(
                f"Unknown qwen-local model id {model_id!r}; "
                f"known: {sorted(_QWEN_HF_REPOS)}"
            )
        if model_id not in self._models:
            self._models[model_id] = self._loader(model_id)
        return self._models[model_id]

    def embed(self, texts: list[str], *, model: str, input_type: str) -> EmbedResult:
        st = self._get_model(model)
        kwargs: dict = {"normalize_embeddings": True}
        if input_type == "query":
            # Qwen3 is instruction-asymmetric: queries need the model's query
            # prompt or retrieval silently degrades (research doc, finding 2).
            kwargs["prompt_name"] = "query"
        vectors = st.encode(texts, **kwargs)
        embeddings = [[float(x) for x in v] for v in vectors]
        if model not in self._dim_checked and embeddings:
            got, want = len(embeddings[0]), EMBED_DIMS[model]
            if got != want:
                raise ValueError(
                    f"{model}: provider emitted {got}-dim vectors but "
                    f"EMBED_DIMS declares {want} — a dim change requires a NEW model id"
                )
            self._dim_checked.add(model)
        total: int | None = None
        tokenizer = getattr(st, "tokenizer", None)
        if tokenizer is not None:
            total = sum(len(tokenizer(t)["input_ids"]) for t in texts)
        return EmbedResult(embeddings=embeddings, total_tokens=total)


def _load_sentence_transformer(model_id: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "embed_provider 'qwen-local' requires the optional local extra: "
            "uv tool install --editable . --with-extras local  (or: uv sync --extra local)"
        ) from e
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer(
        _QWEN_HF_REPOS[model_id],
        device=device,
        truncate_dim=EMBED_DIMS[model_id],
    )
```

(Place `_QWEN_HF_REPOS` + `QwenLocalProvider` + `_load_sentence_transformer` where the stub was, i.e. above `get_provider`.)

Add to `pyproject.toml` after `[project.scripts]`:

```toml
[project.optional-dependencies]
local = [
    "sentence-transformers>=3.3",
    "torch>=2.4",
]
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/vecs/embed_provider.py tests/test_embed_provider.py pyproject.toml
git commit -m "feat(embed): QwenLocalProvider (lazy MPS sentence-transformers) + vecs[local] extra"
```

---

### Task 5: route the indexer through the provider

**Files:**
- Modify: `src/vecs/indexer.py` — `_embed_and_store` (`:383-520`), `run_index` (`:1684+`, the `vo = get_voyage_client()` line), `index_code` (`:1015`), `index_docs` (`:1283`), `index_single_doc` (`:1416`) signatures/threading
- Modify: `tests/indexer_helpers.py` and the `tests/test_indexer_*.py` files that pass a raw `vo` mock

- [ ] **Step 1: Write the failing test** (append to `tests/test_indexer_embed.py`)

```python
def test_embed_and_store_routes_through_provider(tmp_path):
    """_embed_and_store must call provider.embed (not vo.embed) and calibrate
    the batcher from EmbedResult.total_tokens."""
    from vecs.embed_provider import VoyageProvider
    from vecs.indexer import AdaptiveBatcher, _embed_and_store

    vo = MagicMock()
    vo.embed.return_value = FakeEmbedResult(2)
    provider = VoyageProvider(client=vo)
    collection = MagicMock()
    batcher = AdaptiveBatcher()
    chunks = [
        {"id": "a", "text": "alpha", "metadata": {}},
        {"id": "b", "text": "beta", "metadata": {}},
    ]
    ids = _embed_and_store(chunks, collection, "voyage-code-3", provider, batcher=batcher)
    assert sorted(ids) == ["a", "b"]
    vo.embed.assert_called_once()
    assert vo.embed.call_args.kwargs["input_type"] == "document"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q tests/test_indexer_embed.py -k provider`
Expected: FAIL (today `_embed_and_store` calls `vo.embed` on whatever is passed; passing a `VoyageProvider` breaks the `.usage` access — assert the failure mode, then fix).

- [ ] **Step 3: Implement**

In `_embed_and_store` (indexer.py:383):
- Signature: `def _embed_and_store(chunks, collection, model, provider, batcher=None, cache=None)` — rename the `vo: voyageai.Client` param to `provider` (type hint `EmbedProvider`); update the docstring's "Voyage call" wording to "provider call".
- The embed call (`:451`): `result = provider.embed(texts, model=model, input_type="document")`.
- The retry classification (`:455-467`) becomes:

```python
                is_transient = (
                    isinstance(e, provider.retryable_errors)
                    or isinstance(e, (TimeoutError, ConnectionError))
                    or ("rate" in error_msg and "limit" in error_msg)
                )
```

- The calibration (`:484-485`) becomes:

```python
        if result.total_tokens is not None:
            batcher.calibrate(batch_chars, result.total_tokens)
```

- `result.embeddings` accesses stay as-is (EmbedResult has `.embeddings`).
- Imports: add `from vecs.embed_provider import EmbedProvider, get_provider`; the `import voyageai` / `import voyageai.error` at `:13-14` can be DELETED once nothing else in the module references them (grep the module first).

In `run_index`: replace `vo = get_voyage_client()` with `provider = get_provider(config)` and thread `provider` where `vo` was passed (`index_code(project, provider, db, cache=cache)`, same for `index_docs`). Rename the `vo` params of `index_code` / `index_docs` / `_index_collection` (and any helper between them that forwards it) to `provider`. In `index_single_doc` replace its internal `get_voyage_client()` with `get_provider(config)` (it already loads config; verify and reuse).

In `tests/indexer_helpers.py` and `tests/test_indexer_*.py`: wherever a `MagicMock()` vo is passed into `index_code`/`index_docs`/`_embed_and_store`/`_index_collection`, wrap it: `provider = VoyageProvider(client=vo)` and pass `provider`. The mocks' `.embed.call_args` assertions (`_embedded_texts`) still see the inner client's calls — VoyageProvider passes through verbatim. Add `from vecs.embed_provider import VoyageProvider` to `indexer_helpers.py` and re-export it.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: green. Iterate on remaining `vo`-shaped test call sites until green — the failure list IS the call-site list.

- [ ] **Step 5: Commit**

```bash
git add src/vecs/indexer.py tests/
git commit -m "refactor(indexer): route embedding through EmbedProvider seam"
```

---

### Task 6: route the searcher through the provider

**Files:**
- Modify: `src/vecs/searcher.py:26-33` (`_cached_embed`), `:163` (`vo = get_voyage_client()`)
- Test: `tests/test_searcher.py` (update fakes the same way)

- [ ] **Step 1: Write the failing test** (append to `tests/test_searcher.py`)

```python
def test_cached_embed_uses_provider_query_input_type():
    from unittest.mock import MagicMock
    from vecs.embed_provider import VoyageProvider
    from vecs.searcher import _cached_embed, _clear_caches

    _clear_caches()
    vo = MagicMock()
    result = MagicMock()
    result.embeddings = [[0.2] * 4]
    vo.embed.return_value = result
    provider = VoyageProvider(client=vo)
    emb = _cached_embed(provider, "q", "voyage-code-3")
    assert emb == [0.2] * 4
    assert vo.embed.call_args.kwargs["input_type"] == "query"
    # second call: cache hit, no new provider call
    _cached_embed(provider, "q", "voyage-code-3")
    assert vo.embed.call_count == 1
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest -q tests/test_searcher.py -k cached_embed_uses_provider`. Expected: FAIL (`.embeddings[0]` shape mismatch — `_cached_embed` calls `vo.embed(...).embeddings[0]` on the provider).

- [ ] **Step 3: Implement**

```python
def _cached_embed(provider, query: str, model: str) -> list[float]:
    """Embed a query through the provider, using cache when available."""
    key = (query, model)
    if key in _embedding_cache:
        return _embedding_cache[key]
    embedding = provider.embed([query], model=model, input_type="query").embeddings[0]
    _embedding_cache[key] = embedding
    return embedding
```

In `search()`: replace `vo = get_voyage_client()` with `provider = get_provider(config)` — NOTE: move it AFTER `config = load_config()` so the provider follows config; pass `provider` to `_cached_embed`. Update the import at `:8`: drop `get_voyage_client`, add `from vecs.embed_provider import get_provider`. Update existing searcher tests that monkeypatch `get_voyage_client` in `vecs.searcher` — they now patch `vecs.searcher.get_provider` (return a `VoyageProvider(client=fake_vo)`).

- [ ] **Step 4: Run the full suite** — `uv run pytest -q`. Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/vecs/searcher.py tests/test_searcher.py
git commit -m "refactor(searcher): route query embedding through EmbedProvider"
```

---

### Task 7: route prose_drift through the provider

**Files:**
- Modify: `src/vecs/prose_drift.py:514-535` (`_voyage_embed` → `_embed_fact`, `_voyage_embed_cached` → `_embed_fact_cached`) and the call sites at `:471` and `:750`
- Test: `tests/test_prose_drift.py` (mechanical rename of any `_voyage_embed` references)

- [ ] **Step 1: Implement (rename + reroute; tests pin behavior already)**

```python
def _embed_fact(text: str) -> list[float]:
    provider = get_provider()
    result = provider.embed([text], model=FACTS_MODEL, input_type="document")
    return result.embeddings[0]
```

Rename `_voyage_embed_cached` → `_embed_fact_cached` (body unchanged except it calls `_embed_fact`; its SQLite cache stays keyed by `(text_sha, FACTS_MODEL)` — model-keyed, so a provider/model flip segregates rows by construction). Update call sites `:471` and `:750` and any test references. Import `from vecs.embed_provider import get_provider`; remove the now-unused `get_voyage_client` import if nothing else uses it.

- [ ] **Step 2: Run the full suite** — `uv run pytest -q`. Expected: green (rename-only fallout fixed as it surfaces).

- [ ] **Step 3: Commit**

```bash
git add src/vecs/prose_drift.py tests/test_prose_drift.py
git commit -m "refactor(prose-drift): route fact embedding through EmbedProvider"
```

---

### Task 8: code-collection model markers with backfill

**Files:**
- Modify: `src/vecs/indexer.py` — `_remodel_clear` (`:1577`), `_remodel_record` (`:1607`); add `_code_sources` + `_clear_code_manifest_entries` next to `_docs_sources`/`_clear_docs_manifest_entries`
- Test: `tests/test_indexer_run.py` (replace `test_remodel_clears_docs_leaves_code_on_model_change` at `:147`), `tests/test_searcher.py` (code-marker interlock regression)

Semantics (design.md L1.4 — this deliberately retires the "code has no trigger" invariant):
- docs: UNCHANGED legacy semantics (None marker ⇒ treated as changed ⇒ clear+re-embed; that is the shipped pre-marker-store migration path).
- code: **backfill-first.** None marker + non-empty collection ⇒ record `CODE_MODEL`, NO clear (reusing docs' None⇒changed semantics would mass re-embed livly's ~11,490 code chunks on the first post-merge reindex). Mismatched marker + non-empty ⇒ clear code manifest entries (clear-scope ≡ rescan-scope via the same enumerator `index_code` scans with).

- [ ] **Step 1: Write the failing tests**

Replace `test_remodel_clears_docs_leaves_code_on_model_change` with (reuse the test file's existing fixtures/builders for project + cache + db — mirror the old test's setup exactly, only the assertions change):

```python
def test_remodel_backfills_unmarked_code_without_clearing(...existing fixture args...):
    """Unmarked non-empty code collection + unchanged model => marker recorded,
    NO manifest clear (the no-regret backfill, design.md L1.4)."""
    # setup: code collection count > 0, no marker, manifest has code keys
    cleared = _remodel_clear(project, db, cache)
    assert cleared["code"] == 0
    assert cache.get_collection_model(project.code_collection) == CODE_MODEL
    # manifest keys untouched
    ...

def test_remodel_clears_code_on_real_model_mismatch(...):
    """Recorded marker != CODE_MODEL + non-empty => code manifest entries cleared."""
    cache.set_collection_model(project.code_collection, "voyage-code-2")
    cleared = _remodel_clear(project, db, cache)
    assert cleared["code"] > 0

def test_remodel_record_marks_both_docs_and_code(...):
    _remodel_record(project, cache)
    assert cache.get_collection_model(project.docs_collection) == DOCS_MODEL
    assert cache.get_collection_model(project.code_collection) == CODE_MODEL
```

Add to `tests/test_searcher.py` (model the existing interlock tests — the docs-mismatch one — but for a `-code` collection):

```python
def test_model_flip_interlock_drops_mismatched_code_collection(...):
    """A code collection marked under a different model is dropped from the
    vector path (BM25-only) — the interlock needs zero searcher change, this
    pins that code markers (new in L1.4) engage it."""
```

- [ ] **Step 2: Run to verify failure** — the replaced test fails (`cleared` has no `"code"` key; marker not backfilled).

- [ ] **Step 3: Implement in `src/vecs/indexer.py`**

Extract the file enumeration `index_code` already uses into a shared helper (DRY — lift the EXACT existing scan expression out of `index_code` and call the helper from `index_code` too, so clear-scope ≡ rescan-scope by construction):

```python
def _code_sources(project: ProjectConfig) -> list[Path]:
    """Every file index_code will scan across all code_dirs (mirrors
    _docs_sources; shared with the code-model clear so the two cannot drift).
    `.md` is excluded exactly as index_code excludes it (routes to -docs)."""
    files: list[Path] = []
    for code_dir in project.code_dirs:
        extensions = {e for e in code_dir.extensions if e != ".md"}
        if not extensions:
            continue
        files.extend(_scan_code_dir(code_dir, extensions))
    return files


def _clear_code_manifest_entries(manifest: Manifest, code_files: list[Path]) -> int:
    """Code twin of _clear_docs_manifest_entries: same key scheme (bare abs path)."""
    cleared = 0
    for f in code_files:
        key = str(f)
        if key in manifest.data:
            del manifest.data[key]
            cleared += 1
    return cleared
```

(Verify `index_code`'s scan line at `:1015+` matches this `.md`-stripping expression before lifting; if `index_code` filters differently, mirror IT, not this sketch.)

Extend `_remodel_clear` (keep the docs block verbatim, add the code block + update docstring):

```python
def _remodel_clear(
    project: ProjectConfig, db: chromadb.ClientAPI, cache: EmbedCache
) -> dict[str, int]:
    """run_index PRE-pass: clear manifest entries when an embedding model changed.

    Docs keep legacy semantics: a None marker reads as changed (the shipped
    pre-marker-store migration). Code is BACKFILL-FIRST (L1.4): an unmarked
    non-empty code collection predates code markers and is assumed current —
    record CODE_MODEL with no clear; clear+re-embed fires only on a real
    recorded-vs-configured mismatch. New markers are written AFTER the index
    passes by _remodel_record.
    """
    cleared = {"docs": 0, "code": 0}
    if _model_changed(cache, db, project.docs_collection, DOCS_MODEL):
        # ... existing docs body verbatim ...

    code_marker = cache.get_collection_model(project.code_collection)
    if code_marker is None:
        if _collection_count(db, project.code_collection) > 0:
            cache.set_collection_model(project.code_collection, CODE_MODEL)
            _log(f"[{project.name}] backfilled code model marker = {CODE_MODEL}")
    elif code_marker != CODE_MODEL and _collection_count(db, project.code_collection) > 0:
        manifest = Manifest(project.name)
        cleared["code"] = _clear_code_manifest_entries(manifest, _code_sources(project))
        if cleared["code"]:
            manifest.save()
            _log(
                f"[{project.name}] Code embedding-model change: cleared "
                f"{cleared['code']} code manifest entries to force a re-embed."
            )
    return cleared
```

`_remodel_record` records both:

```python
def _remodel_record(project: ProjectConfig, cache: EmbedCache) -> None:
    """run_index POST-pass: record what docs AND code are now embedded under."""
    cache.set_collection_model(project.docs_collection, DOCS_MODEL)
    cache.set_collection_model(project.code_collection, CODE_MODEL)
```

- [ ] **Step 4: Run the full suite** — `uv run pytest -q`. Expected: green. (`tests/test_searcher.py:430`'s "code collections never marked" pin will fail — rewrite it to the new reality: code collections ARE marked after a reindex; the fail-open-on-None branch still covers pre-backfill stores and cache errors, pinned by `test_collection_markers_fail_open_on_cache_error` which must stay green.)

- [ ] **Step 5: Commit**

```bash
git add src/vecs/indexer.py tests/test_indexer_run.py tests/test_searcher.py
git commit -m "feat(indexer): code-collection model markers with no-regret backfill"
```

---

### Task 9: extract parameterized `search_collections()`

**Files:**
- Modify: `src/vecs/searcher.py:147-267`
- Test: `tests/test_searcher.py` (append; existing tests pin that `search()` is unchanged)

- [ ] **Step 1: Write the failing test**

```python
def test_search_collections_accepts_explicit_targets_and_bm25_paths(tmp_path, monkeypatch):
    """The A/B harness entry point: same pipeline as search(), but targets,
    provider and bm25 paths are injected instead of derived from config."""
    from vecs.embed_provider import VoyageProvider
    from vecs.searcher import search_collections, _clear_caches

    _clear_caches()
    vo = MagicMock()
    emb_result = MagicMock()
    emb_result.embeddings = [[0.3] * 4]
    vo.embed.return_value = emb_result
    provider = VoyageProvider(client=vo)

    collection = MagicMock()
    collection.query.return_value = {
        "ids": [["c1"]],
        "documents": [["hello world"]],
        "metadatas": [[{"file_path": "src/x.py"}]],
        "distances": [[0.1]],
    }
    db = MagicMock()
    db.get_collection.return_value = collection
    monkeypatch.setattr("vecs.searcher.get_chromadb_client", lambda: db)

    out = search_collections(
        "q",
        targets=[("shadow-code-qwen", "qwen3-embedding-0.6b", "vecs")],
        provider=provider,
        n_results=3,
        bm25_paths={"shadow-code-qwen": tmp_path / "absent.db"},  # missing db => vector-only
        check_markers=False,
    )
    assert out and out[0]["id"] == "c1"
    assert out[0]["collection"] == "shadow-code-qwen"
    assert vo.embed.call_args.kwargs["model"] == "qwen3-embedding-0.6b"
```

- [ ] **Step 2: Run to verify failure** — ImportError: no `search_collections`.

- [ ] **Step 3: Implement**

Move the body of `search()` from the interlock block through the final return into:

```python
def search_collections(
    query: str,
    targets: list[tuple[str, str, str]],  # (collection_name, model, project_name)
    *,
    provider,
    n_results: int = 5,
    path_filter: str | None = None,
    bm25_paths: dict[str, Path] | None = None,  # collection_name -> bm25 .db path
    check_markers: bool = True,
) -> list[dict]:
    """The full hybrid pipeline (interlock -> vector -> BM25 -> RRF -> dedup ->
    fetch-multiplier escalation) over EXPLICIT targets. search() wraps this with
    config-derived production values; the A/B harness calls it with arm values —
    one code path, so the 'hybrid (production config)' arm is production by
    construction (design.md L1.1)."""
```

Inside: the interlock block runs only `if check_markers:` (else `vector_targets = list(targets)`); the BM25 loop replaces the path derivation with `bm25_path = (bm25_paths or {}).get(col_name)` + `if bm25_path is None: continue`; everything else moves verbatim (fetch-multiplier loop, dedup, RRF, sort fallback). `db = get_chromadb_client()` stays inside (tests patch it).

`search()` becomes the thin wrapper — build `targets` exactly as today (`:173-182`), then:

```python
    bm25_dir = VECS_DIR / "bm25"
    bm25_paths = {}
    for col_name, _model, proj_name in targets:
        suffix = "code" if col_name.endswith("-code") else "docs"
        bm25_paths[col_name] = bm25_dir / f"{proj_name}_{suffix}.db"
    return search_collections(
        query, targets, provider=provider, n_results=n_results,
        path_filter=path_filter, bm25_paths=bm25_paths,
    )
```

Add `from pathlib import Path` to searcher imports.

- [ ] **Step 4: Run the full suite** — `uv run pytest -q`. Expected: green — every existing `search()` test passes unchanged (behavior identical by construction).

- [ ] **Step 5: Commit**

```bash
git add src/vecs/searcher.py tests/test_searcher.py
git commit -m "refactor(searcher): extract parameterized search_collections for A/B arms"
```

---

### Task 10: eval harness — EvalCase extension, YAML loader, ranking metrics

**Files:**
- Modify: `src/vecs/eval_harness.py:186-213` (EvalCase + DEFAULT_EVAL_SET untouched semantics), append loader + metrics
- Test: `tests/test_eval_harness.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
class TestGoldenSetLoader:
    def test_loads_yaml_cases(self, tmp_path):
        from vecs.eval_harness import load_eval_set
        p = tmp_path / "g.yaml"
        p.write_text(
            "cases:\n"
            "  - query: how does fusion work\n"
            "    project: vecs\n"
            "    collection: code\n"
            "    class: nl\n"
            "    expected: [searcher.py]\n"
        )
        cases = load_eval_set(p)
        assert len(cases) == 1
        assert cases[0].expected == ["searcher.py"]
        assert cases[0].query_class == "nl"

    def test_legacy_single_substring_back_compat(self):
        from vecs.eval_harness import EvalCase
        c = EvalCase("q", "vecs", "docs", expected_path_substring="kb-foundations")
        assert c.expected == ["kb-foundations"]


class TestRankingMetrics:
    def test_recall_at_k(self):
        from vecs.eval_harness import recall_at_k
        sources = ["a/x.py", "b/y.py", "c/z.py"]
        assert recall_at_k(sources, ["y.py"], k=2) == 1.0
        assert recall_at_k(sources, ["z.py"], k=2) == 0.0

    def test_mrr_first_relevant_rank(self):
        from vecs.eval_harness import mrr
        assert mrr(["a", "hit/b", "c"], ["hit"]) == 0.5
        assert mrr(["x"], ["hit"]) == 0.0

    def test_ndcg_binary_relevance(self):
        from vecs.eval_harness import ndcg_at_k
        # relevant at rank 1 => 1.0; relevant only at rank 2 => 1/log2(3)
        assert ndcg_at_k(["hit/a", "b"], ["hit"], k=10) == 1.0
        import math
        assert abs(ndcg_at_k(["b", "hit/a"], ["hit"], k=10) - 1 / math.log2(3)) < 1e-9
```

- [ ] **Step 2: Run to verify failure** — ImportErrors / missing fields.

- [ ] **Step 3: Implement in `src/vecs/eval_harness.py`**

Extend `EvalCase` (defaults keep `DEFAULT_EVAL_SET` and existing callers valid):

```python
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
```

(`run_eval`'s hit check changes from `case.expected_path_substring in (s or "")` to `any(e in (s or "") for e in case.expected)` — keep everything else.)

Append loader + metrics:

```python
def load_eval_set(path: Path) -> list[EvalCase]:
    """Load a golden set YAML: {cases: [{query, project, collection, class, expected: [..]}]}.

    The livly golden set lives OUTSIDE the repo (~/.vecs/evalsets/livly.yaml) —
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
```

- [ ] **Step 4: Run the full suite** — `uv run pytest -q`. Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/vecs/eval_harness.py tests/test_eval_harness.py
git commit -m "feat(eval): golden-set YAML loader + recall/MRR/nDCG metrics"
```

---

### Task 11: paired bootstrap CI + A/B arm runner

**Files:**
- Modify: `src/vecs/eval_harness.py` (append)
- Test: `tests/test_eval_harness.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
class TestBootstrapAB:
    def test_paired_bootstrap_ci_zero_deltas(self):
        from vecs.eval_harness import paired_bootstrap_ci
        lo, hi = paired_bootstrap_ci([0.0] * 50)
        assert lo == 0.0 and hi == 0.0

    def test_paired_bootstrap_ci_brackets_the_mean(self):
        from vecs.eval_harness import paired_bootstrap_ci
        deltas = [0.1] * 30 + [-0.1] * 10  # mean = 0.05
        lo, hi = paired_bootstrap_ci(deltas, seed=7)
        assert lo < 0.05 < hi
        assert lo > -0.1 and hi < 0.15

    def test_run_arm_scores_each_case(self):
        from vecs.eval_harness import EvalCase, run_arm

        def fake_search(query, collection_name=None, n_results=10, project=None):
            return [{"metadata": {"file_path": "src/searcher.py"}}]

        cases = [EvalCase("q1", "vecs", "code", expected=["searcher.py"], query_class="nl"),
                 EvalCase("q2", "vecs", "code", expected=["nowhere.py"], query_class="identifier")]
        scores = run_arm(cases, fake_search, n_results=10)
        assert scores[0].recall10 == 1.0 and scores[0].ndcg10 == 1.0
        assert scores[1].recall10 == 0.0

    def test_ab_report_pairs_and_breaks_down_by_class(self):
        from vecs.eval_harness import EvalCase, run_arm, ab_report

        cases = [EvalCase("q1", "vecs", "code", expected=["a.py"], query_class="nl"),
                 EvalCase("q2", "vecs", "code", expected=["b.py"], query_class="identifier")]

        def arm_hits_all(query, **kw):
            return [{"metadata": {"file_path": "a.py"}}, {"metadata": {"file_path": "b.py"}}]

        def arm_hits_none(query, **kw):
            return [{"metadata": {"file_path": "z.py"}}]

        report = ab_report(run_arm(cases, arm_hits_all), run_arm(cases, arm_hits_none))
        r10 = report["overall"]["recall10"]
        assert r10["mean_a"] == 1.0 and r10["mean_b"] == 0.0 and r10["delta"] == -1.0
        assert r10["ci"][0] <= -1.0 <= r10["ci"][1]
        assert set(report["by_class"]) == {"nl", "identifier"}
```

- [ ] **Step 2: Run to verify failure** — ImportErrors.

- [ ] **Step 3: Implement (append to `eval_harness.py`)**

```python
@dataclass
class QueryScore:
    """Per-query metrics for one arm (paired by eval-set order)."""

    case: EvalCase
    recall5: float
    recall10: float
    ndcg10: float
    mrr: float


def run_arm(
    eval_set: list[EvalCase], search_fn, n_results: int = 10
) -> list[QueryScore]:
    """Score one arm. A per-case search failure degrades to all-zero metrics
    (mirrors run_eval), never aborts the arm."""
    scores: list[QueryScore] = []
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
        scores.append(
            QueryScore(
                case=case,
                recall5=recall_at_k(sources, case.expected, 5),
                recall10=recall_at_k(sources, case.expected, 10),
                ndcg10=ndcg_at_k(sources, case.expected, 10),
                mrr=mrr(sources, case.expected),
            )
        )
    return scores


def paired_bootstrap_ci(
    deltas: list[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired per-query deltas.

    The L3 gate reads CI BOUNDS, not point estimates — at ~100 queries a 3-pt
    boundary is inside one standard error (design.md L3)."""
    import random

    if not deltas:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(
        sum(rng.choice(deltas) for _ in range(n)) / n for _ in range(n_boot)
    )
    lo = means[max(0, int((alpha / 2) * n_boot) - 1)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


_METRICS = ("recall5", "recall10", "ndcg10", "mrr")


def _summarize(arm_a: list[QueryScore], arm_b: list[QueryScore]) -> dict:
    out: dict = {}
    for m in _METRICS:
        a = [getattr(s, m) for s in arm_a]
        b = [getattr(s, m) for s in arm_b]
        deltas = [bb - aa for aa, bb in zip(a, b)]
        out[m] = {
            "mean_a": sum(a) / len(a) if a else None,
            "mean_b": sum(b) / len(b) if b else None,
            "delta": (sum(b) - sum(a)) / len(a) if a else None,
            "ci": paired_bootstrap_ci(deltas),
            "n": len(a),
        }
    return out


def ab_report(arm_a: list[QueryScore], arm_b: list[QueryScore]) -> dict:
    """Paired A/B summary: overall + per query class. Arms MUST be run over the
    same eval set in the same order (paired deltas)."""
    assert len(arm_a) == len(arm_b), "arms must score the same eval set"
    report = {"overall": _summarize(arm_a, arm_b), "by_class": {}}
    classes = {s.case.query_class for s in arm_a}
    for qc in sorted(classes):
        a = [s for s in arm_a if s.case.query_class == qc]
        b = [s for s in arm_b if s.case.query_class == qc]
        report["by_class"][qc] = _summarize(a, b)
    return report
```

- [ ] **Step 4: Run the full suite** — `uv run pytest -q`. Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/vecs/eval_harness.py tests/test_eval_harness.py
git commit -m "feat(eval): run_arm + paired-bootstrap A/B report (overall + per query class)"
```

---

### Task 12: the vecs golden set + schema/protocol README

**Files:**
- Create: `evalsets/README.md`
- Create: `evalsets/vecs.yaml`

- [ ] **Step 1: Create `evalsets/README.md`**

```markdown
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
  Exported A/B reports contain aggregate metrics only.
- **Authoring protocol:** derive queries from real information needs WITHOUT
  running vecs search while authoring (selecting queries the incumbent already
  answers rigs the A/B). Before scoring an A/B, complete the expected sets by
  pooling top-10 from all arms and adjudicating relevance.
- **Freeze:** the set is frozen at a recorded commit before any A/B run. Later
  edits require a changelog line here + re-running all arms.

## Changelog

- 2026-06-11: `vecs.yaml` authored (46 cases) per protocol; frozen pre-A/B.
```

- [ ] **Step 2: Create `evalsets/vecs.yaml`** — the diagnostic set, authored per protocol (from real information needs in repo work; no search ran during authoring):

```yaml
# vecs golden set — diagnostic arm (livly is the decision set; see README.md).
# 46 cases: 24 code (12 nl / 8 identifier / 4 concept), 22 docs (12 nl / 6 identifier / 4 concept).
cases:
  # ---- code / nl ----
  - {query: how are chunks batched to stay under the embedding API token limit, project: vecs, collection: code, class: nl, expected: [indexer.py]}
  - {query: where do deleted files get their orphaned chunks removed, project: vecs, collection: code, class: nl, expected: [indexer.py]}
  - {query: how does search merge vector and keyword results, project: vecs, collection: code, class: nl, expected: [searcher.py]}
  - {query: where are near duplicate search results filtered out, project: vecs, collection: code, class: nl, expected: [searcher.py]}
  - {query: how does the system avoid re-embedding unchanged chunks, project: vecs, collection: code, class: nl, expected: [embed_cache.py, indexer.py]}
  - {query: what happens when the configured embedding model differs from what a collection was embedded with, project: vecs, collection: code, class: nl, expected: [searcher.py, indexer.py]}
  - {query: how is a query embedding cached between searches, project: vecs, collection: code, class: nl, expected: [searcher.py]}
  - {query: where does markdown get split into chunks, project: vecs, collection: code, class: nl, expected: [doc_chunker.py]}
  - {query: how are C sharp files chunked using the syntax tree, project: vecs, collection: code, class: nl, expected: [ast_chunker.py]}
  - {query: where is the daily cost cap for LLM calls enforced, project: vecs, collection: code, class: nl, expected: [metering.py]}
  - {query: how do I add a single document to the index without a full reindex, project: vecs, collection: code, class: nl, expected: [indexer.py, cli.py, mcp_server.py]}
  - {query: where are stale chunks detected by comparing version ids, project: vecs, collection: code, class: nl, expected: [eval_harness.py]}
  # ---- code / identifier ----
  - {query: reciprocal_rank_fusion, project: vecs, collection: code, class: identifier, expected: [searcher.py]}
  - {query: AdaptiveBatcher, project: vecs, collection: code, class: identifier, expected: [indexer.py]}
  - {query: _prune_and_sweep_orphans, project: vecs, collection: code, class: identifier, expected: [indexer.py]}
  - {query: collection_models marker table, project: vecs, collection: code, class: identifier, expected: [embed_cache.py]}
  - {query: metered_create, project: vecs, collection: code, class: identifier, expected: [metering.py]}
  - {query: find_prose_drift, project: vecs, collection: code, class: identifier, expected: [prose_drift.py]}
  - {query: _docs_sources, project: vecs, collection: code, class: identifier, expected: [indexer.py]}
  - {query: get_bm25 FTS5, project: vecs, collection: code, class: identifier, expected: [bm25_index.py]}
  # ---- code / concept ----
  - {query: how does a model flip stay safe between deploy and reindex, project: vecs, collection: code, class: concept, expected: [searcher.py, indexer.py]}
  - {query: what keeps the BM25 sidecar consistent with the vector store, project: vecs, collection: code, class: concept, expected: [indexer.py, bm25_index.py]}
  - {query: lifecycle of a changed file from detection to embedded chunks, project: vecs, collection: code, class: concept, expected: [indexer.py]}
  - {query: how are projects and their directories configured, project: vecs, collection: code, class: concept, expected: [config.py]}
  # ---- docs / nl ----
  - {query: why were session transcripts removed from indexing, project: vecs, collection: docs, class: nl, expected: [CLAUDE.md]}
  - {query: what is the plan for sharing the knowledge base with the team, project: vecs, collection: docs, class: nl, expected: [shared-team-vecs, vecs-platform-strategy, vecs-roadmap]}
  - {query: how does the prose staleness detector decide a doc contradicts a fact, project: vecs, collection: docs, class: nl, expected: [prose-staleness-detector, CLAUDE.md]}
  - {query: what did the direction review decide about increment ordering, project: vecs, collection: docs, class: nl, expected: [vecs-direction-review]}
  - {query: which local embedding models could replace voyage and why, project: vecs, collection: docs, class: nl, expected: [local-embed]}
  - {query: what gate decides whether we swap to a local embedding model, project: vecs, collection: docs, class: nl, expected: [local-embed]}
  - {query: how should a feature move through the workflow phases, project: vecs, collection: docs, class: nl, expected: [workflow-framework, workflow-vecs-profile]}
  - {query: what does the freshness tag on search results mean, project: vecs, collection: docs, class: nl, expected: [CLAUDE.md, kb-freshness]}
  - {query: how is the stale retrieval rate defined, project: vecs, collection: docs, class: nl, expected: [CLAUDE.md, kb-foundations-instrumentation]}
  - {query: why does the embedding cache key include the model, project: vecs, collection: docs, class: nl, expected: [CLAUDE.md, kb-foundations]}
  - {query: what would it cost to populate the facts store, project: vecs, collection: docs, class: nl, expected: [est-cost-to-populate-facts]}
  - {query: how do we plan to detect when documentation goes stale, project: vecs, collection: docs, class: nl, expected: [prose-staleness-detector, context-staleness-detector]}
  # ---- docs / identifier ----
  - {query: dryrun_pass_criteria, project: vecs, collection: docs, class: identifier, expected: [workflow-vecs-profile, local-embed, kb-foundations]}
  - {query: Qwen3-Embedding-4B, project: vecs, collection: docs, class: identifier, expected: [local-embed]}
  - {query: stale-retrieval-rate, project: vecs, collection: docs, class: identifier, expected: [kb-foundations-instrumentation, CLAUDE.md, local-embed]}
  - {query: _remodel_clear, project: vecs, collection: docs, class: identifier, expected: [CLAUDE.md, kb-foundations]}
  - {query: voyage rerank-2.5-lite, project: vecs, collection: docs, class: identifier, expected: [vecs-kb-curation-design, local-embed]}
  - {query: anonymized_telemetry, project: vecs, collection: docs, class: identifier, expected: [local-embed]}
  # ---- docs / concept ----
  - {query: what order do the remaining increments run in and why, project: vecs, collection: docs, class: concept, expected: [local-embed, vecs-kb-curation-design, vecs-direction-review]}
  - {query: what invariants protect against deleting live chunks by mistake, project: vecs, collection: docs, class: concept, expected: [CLAUDE.md]}
  - {query: how does the system know which model a collection was embedded under, project: vecs, collection: docs, class: concept, expected: [CLAUDE.md, kb-foundations, local-embed]}
  - {query: what happened in the retro of the freshness hotfix, project: vecs, collection: docs, class: concept, expected: [kb-freshness-hotfix]}
```

- [ ] **Step 3: Validate the file loads**

Run: `uv run python -c "from pathlib import Path; from vecs.eval_harness import load_eval_set; cs = load_eval_set(Path('evalsets/vecs.yaml')); print(len(cs), sum(1 for c in cs if c.collection=='code'))"`
Expected: `46 24`.

- [ ] **Step 4: Commit**

```bash
git add evalsets/
git commit -m "feat(eval): vecs golden set (46 cases) + schema/authoring protocol"
```

---

### Task 13: context docs, acceptance, full suite

**Files:**
- Modify: `src/vecs/CLAUDE.md` (invariants)
- Create: `docs/features/local-embed-base/acceptance.md`

- [ ] **Step 1: Update `src/vecs/CLAUDE.md`**

- Modules table: add `embed_provider.py` row ("EmbedProvider seam: VoyageProvider (default) + QwenLocalProvider (optional `vecs[local]` extra). Selected by `config.embed_provider`; model-id strings stay the cache/marker key.").
- Invariants: REWRITE the "Code has no trigger" sentence inside the model-change invariant to: "Code collections are markered too (L1.4): an unmarked non-empty `-code` collection is BACKFILLED (marker recorded, no clear — it predates code markers and is assumed current); clear+re-embed fires only on a real recorded≠configured mismatch, scoped by `_code_sources` (the same enumerator `index_code` scans with)."
- Invariants: update the interlock bullet's "code collections are never marked" parenthetical to "(pre-backfill stores and cache errors still read None)".
- Invariants: add: "`embed_provider` is a persisted `VecsConfig` field — `save()` writes it; never add config keys outside the dataclass+save round-trip (they get stripped on the next auto-configure save)."
- Invariants: add: "Chroma telemetry is off (`Settings(anonymized_telemetry=False)`) at every `PersistentClient` site — keep it off in new sites."

- [ ] **Step 2: Create `docs/features/local-embed-base/acceptance.md`**

```markdown
Authored by Claude

# local-embed-base — acceptance

Design: `docs/features/local-embed/design.md` §L1. Plan: `plan.md` (this dir).

- [ ] Chroma telemetry off at all 3 PersistentClient sites (test-pinned)
- [ ] `embed_provider` config field round-trips load->save (auto-configure cannot strip it)
- [ ] EmbedProvider seam: indexer, searcher, prose_drift all route through it; no direct `voyageai` use outside `clients.py`/`embed_provider.py`
- [ ] QwenLocalProvider: query-prompt asymmetry, first-batch dim assertion, lazy import with actionable error, `vecs[local]` extra installs sentence-transformers+torch
- [ ] Code-collection markers: unmarked+non-empty => backfill (NO clear); mismatch => clear scoped to `_code_sources`; `_remodel_record` marks code+docs; searcher drops a marker-mismatched code collection from the vector path (test-pinned)
- [ ] `search_collections()` extracted; `search()` delegates with production values; existing search tests pass unchanged
- [ ] Eval: YAML golden-set loader, recall@5/10 + nDCG@10 + MRR, `run_arm` + `ab_report` (overall + per class) + paired bootstrap CI
- [ ] `evalsets/vecs.yaml` (46 cases) + README with schema, livly-never-in-repo rule, authoring protocol, freeze rule
- [ ] Full suite green (`uv run pytest -q`); `src/vecs/CLAUDE.md` updated (provider, markers, telemetry, config round-trip invariants)
- [ ] Phase-4 adversarial review run on the diff; findings fixed or logged in `gaps.md`
```

- [ ] **Step 3: Full suite + grep guard**

Run: `uv run pytest -q` — green.
Run: `grep -rn "get_voyage_client\|voyageai" src/vecs/ --include="*.py" | grep -v "clients.py\|embed_provider.py"` — expected: no hits (all embed call sites routed).

- [ ] **Step 4: Commit**

```bash
git add src/vecs/CLAUDE.md docs/features/local-embed-base/
git commit -m "docs(local-embed-base): context-doc invariants + acceptance checklist"
```

---

## Out of plan (later work in this increment)

L2 (perf spike, shadow indexing, A/B execution) and L3 (gate) are `local-embed-ab` — planned after this sub-feature ships. The hygiene rider (roadmap reconcile, parent v5, kb-foundations-pipeline Phase-8 close) is docs-only and rides separately.
