from vecs.indexer import _make_batches, AdaptiveBatcher


def test_small_chunks_pack_into_one_batch():
    chunks = [{"text": "short text"} for _ in range(10)]
    batches = list(_make_batches(chunks))
    assert len(batches) == 1
    assert len(batches[0]) == 10


def test_large_chunks_split_into_multiple_batches():
    big_text = "x" * 300_000
    chunks = [{"text": big_text} for _ in range(5)]
    batches = list(_make_batches(chunks))
    assert len(batches) == 5


def test_batch_respects_max_size():
    chunks = [{"text": "a"} for _ in range(300)]
    batches = list(_make_batches(chunks))
    assert all(len(b) <= 128 for b in batches)
    assert sum(len(b) for b in batches) == 300


def test_empty_chunks():
    assert list(_make_batches([])) == []


def test_adaptive_batcher_default_uses_div2():
    """Before calibration, uses len(text) // 2 as estimate."""
    batcher = AdaptiveBatcher()
    assert batcher.estimate_tokens("abcdef") == 3


def test_adaptive_batcher_calibrate_single():
    """First calibration sets the ratio directly."""
    batcher = AdaptiveBatcher()
    batcher.calibrate(total_chars=1000, actual_tokens=250)
    assert batcher.ratio == 4.0
    # 100 chars / 4.0 ratio = 25 tokens, but floor is 100 // 2 = 50
    # Floor dominates when ratio > 2.0
    assert batcher.estimate_tokens("x" * 100) == 50
    # With longer text where calibration gives higher estimate than floor:
    # 1000 chars / 4.0 = 250, floor = 1000 // 2 = 500 -- floor still wins
    # To see calibration effect, use ratio < 2.0:
    batcher2 = AdaptiveBatcher()
    batcher2.calibrate(total_chars=1000, actual_tokens=1000)  # ratio = 1.0
    # 100 chars / 1.0 = 100, floor = 50 -> calibrated estimate wins
    assert batcher2.estimate_tokens("x" * 100) == 100


def test_adaptive_batcher_calibrate_ema():
    """Subsequent calibrations use EMA smoothing (0.8 old + 0.2 new)."""
    batcher = AdaptiveBatcher()
    batcher.calibrate(total_chars=1000, actual_tokens=250)
    batcher.calibrate(total_chars=1000, actual_tokens=500)
    assert abs(batcher.ratio - 3.6) < 0.01


def test_adaptive_batcher_floor_div2():
    """Calibrated estimate never goes below len(text) // 2."""
    batcher = AdaptiveBatcher()
    batcher.calibrate(total_chars=1000, actual_tokens=100)
    assert batcher.estimate_tokens("x" * 100) == 50


def test_adaptive_batcher_make_batches_integration():
    """AdaptiveBatcher works with _make_batches."""
    batcher = AdaptiveBatcher()
    batcher.calibrate(total_chars=400, actual_tokens=100)
    chunks = [{"text": "x" * 400} for _ in range(10)]
    batches = list(_make_batches(chunks, batcher))
    assert len(batches) == 1
    assert len(batches[0]) == 10


def test_oversized_single_chunk_is_truncated():
    """A single chunk exceeding MAX_BATCH_TOKENS is truncated to fit."""
    from vecs.indexer import MAX_BATCH_TOKENS
    oversized_text = "x" * (MAX_BATCH_TOKENS * 2 + 1000)
    chunks = [{"text": oversized_text}]
    batches = list(_make_batches(chunks))
    assert len(batches) == 1
    assert len(batches[0]) == 1
    truncated_text = batches[0][0]["text"]
    assert len(truncated_text) // 2 <= MAX_BATCH_TOKENS
    assert len(truncated_text) < len(oversized_text)


def test_oversized_chunk_logs_warning(capsys):
    """Truncating an oversized chunk emits a warning to stderr."""
    from vecs.indexer import MAX_BATCH_TOKENS
    oversized_text = "x" * (MAX_BATCH_TOKENS * 2 + 1000)
    chunks = [{"text": oversized_text}]
    list(_make_batches(chunks))
    captured = capsys.readouterr()
    assert "truncat" in captured.err.lower()


def test_oversized_chunk_warning_includes_chunk_id(capsys):
    """Truncation warning must name the offending chunk so the file is identifiable."""
    from vecs.indexer import MAX_BATCH_TOKENS
    oversized_text = "x" * (MAX_BATCH_TOKENS * 2 + 1000)
    chunk_id = "code:client-uk/Assets/Foo/Generated/Big.cs:7"
    chunks = [{"id": chunk_id, "text": oversized_text}]
    list(_make_batches(chunks))
    captured = capsys.readouterr()
    assert chunk_id in captured.err


def test_normal_chunks_not_affected_by_truncation():
    """Chunks within budget are not truncated."""
    original_text = "hello world this is normal text"
    chunks = [{"text": original_text}]
    batches = list(_make_batches(chunks))
    assert batches[0][0]["text"] == original_text
