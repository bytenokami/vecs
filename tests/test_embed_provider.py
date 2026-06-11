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


# ---- QwenLocalProvider (fake SentenceTransformer injected; no downloads) ----


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
    out = p.embed(
        ["one two three", "four five"], model="qwen3-embedding-0.6b", input_type="document"
    )
    assert out.total_tokens == 5


def test_qwen_asserts_dim_matches_embed_dims():
    p, _ = _qwen(dim=768)
    with pytest.raises(ValueError, match="768"):
        p.embed(["x"], model="qwen3-embedding-0.6b", input_type="document")


def test_qwen_unknown_model_id_fails_loud():
    p, _ = _qwen()
    with pytest.raises(ValueError, match="qwen3-embedding"):
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


# ---- Phase-4 fixes: memoization, config-flip guard, lazy-import error -------


def test_get_provider_memoizes_per_name(tmp_path):
    from vecs.embed_provider import _clear_provider_cache

    _clear_provider_cache()
    cfg = VecsConfig(path=tmp_path / "c.yaml")
    p1 = get_provider(cfg)
    p2 = get_provider(cfg)
    assert p1 is p2  # one instance per process; Qwen model cache survives calls
    _clear_provider_cache()


def test_config_flip_alone_to_qwen_is_guarded(tmp_path):
    """embed_provider: qwen-local with voyage model constants must fail LOUD at
    provider construction with the flip story, not crash on the first embed."""
    from vecs.embed_provider import _clear_provider_cache

    _clear_provider_cache()
    cfg = VecsConfig(path=tmp_path / "c.yaml")
    cfg.embed_provider = "qwen-local"
    with pytest.raises(RuntimeError, match="voyage|design.md L3"):
        get_provider(cfg)
    _clear_provider_cache()


def test_explicit_qwen_name_skips_config_guard():
    """name= override (A/B arms, tests) legitimately runs qwen models while the
    configured constants stay voyage — no guard."""
    from vecs.embed_provider import QwenLocalProvider, _clear_provider_cache

    _clear_provider_cache()
    p = get_provider(name="qwen-local")
    assert isinstance(p, QwenLocalProvider)
    _clear_provider_cache()


def test_qwen_loader_missing_extra_raises_actionable(monkeypatch):
    """Selecting qwen-local without the vecs[local] extra raises RuntimeError
    naming the install command (pins the lazy-import try/except)."""
    import sys

    from vecs.embed_provider import _load_sentence_transformer

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match=r"vecs\[local\]"):
        _load_sentence_transformer("qwen3-embedding-0.6b")
