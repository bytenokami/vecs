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
    # calibration only -- there is no embed metering path).
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


# Qwen model id -> Hugging Face repo. The "@mrl1024" suffix is naming
# convention; the ENFORCED dim source is EMBED_DIMS (asserted on first batch).
_QWEN_HF_REPOS = {
    "qwen3-embedding-4b@mrl1024": "Qwen/Qwen3-Embedding-4B",
    "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
}


class QwenLocalProvider:
    """On-device Qwen3 embeddings via sentence-transformers (verified MRL path:
    ``truncate_dim``). Heavy deps live behind the ``vecs[local]`` extra and are
    imported lazily -- selecting this provider without the extra fails with an
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
                    f"EMBED_DIMS declares {want} -- a dim change requires a NEW model id"
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
            "uv tool install --editable '.[local]'  (or: uv sync --extra local)"
        ) from e
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer(
        _QWEN_HF_REPOS[model_id],
        device=device,
        truncate_dim=EMBED_DIMS[model_id],
    )


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
