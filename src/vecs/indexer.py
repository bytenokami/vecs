from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import chromadb

from vecs.ast_chunker import chunk_code_file_ast
from vecs.bm25_index import BM25Index
from vecs.clients import get_chromadb_client
from vecs.config import (
    CHROMADB_DIR,
    CODE_CHUNK_LINES,
    CODE_CHUNK_OVERLAP,
    CODE_MODEL,
    CodeDir,
    DOCS_MODEL,
    MANIFESTS_DIR,
    MANIFEST_PATH,
    VECS_DIR,
    ProjectConfig,
    VecsConfig,
    load_config,
)
from vecs.doc_chunker import chunk_doc, extract_pdf_text
from vecs.embed_cache import EmbedCache
from vecs.embed_provider import EmbedProvider, get_provider


MAX_BATCH_TOKENS = 80_000  # Voyage limit is 120K; char-based estimation is unreliable, so leave wide margin
MAX_BATCH_SIZE = 128

# Doc source extensions for the -docs collection. .md / .txt / .pdf under a
# docs_dir; only .md is routed from inside code_dirs (F).
DOC_EXTENSIONS = {".md", ".txt", ".pdf"}


class AdaptiveBatcher:
    """Estimates token counts using EMA-calibrated char-to-token ratio.

    Starts with conservative len(text) // 2. Calibrates from Voyage API
    responses. Never estimates fewer tokens than len(text) // 2 (floor).
    """

    def __init__(self, max_tokens: int = MAX_BATCH_TOKENS):
        self.max_tokens = max_tokens
        self.ratio: float | None = None  # chars per token, learned

    def estimate_tokens(self, text: str) -> int:
        floor = len(text) // 2
        if self.ratio is None:
            return floor
        estimate = int(len(text) / self.ratio)
        return max(estimate, floor)

    def calibrate(self, total_chars: int, actual_tokens: int) -> None:
        if actual_tokens <= 0:
            return
        batch_ratio = total_chars / actual_tokens
        if self.ratio is None:
            self.ratio = batch_ratio
        else:
            self.ratio = 0.8 * self.ratio + 0.2 * batch_ratio  # EMA smoothing


def _make_batches(chunks: list[dict], batcher: AdaptiveBatcher | None = None) -> Iterator[list[dict]]:
    """Pack chunks into batches by estimated token count.

    If a single chunk exceeds MAX_BATCH_TOKENS, it is truncated to fit
    and a warning is logged. This is better than silent data loss.
    """
    batch: list[dict] = []
    batch_tokens = 0
    for chunk in chunks:
        if batcher:
            tokens = batcher.estimate_tokens(chunk["text"])
        else:
            tokens = len(chunk["text"]) // 2
        # Truncate oversized single chunks to fit the token budget
        if tokens > MAX_BATCH_TOKENS:
            max_chars = MAX_BATCH_TOKENS * 2
            chunk_id = chunk.get("id", "<unknown>")
            _log(f"  WARNING: chunk truncated from {len(chunk['text'])} to {max_chars} chars "
                 f"(exceeded {MAX_BATCH_TOKENS} token budget) [chunk_id={chunk_id}]")
            chunk = {**chunk, "text": chunk["text"][:max_chars]}
            tokens = max_chars // 2
        if batch and (batch_tokens + tokens > MAX_BATCH_TOKENS or len(batch) >= MAX_BATCH_SIZE):
            yield batch
            batch = []
            batch_tokens = 0
        batch.append(chunk)
        batch_tokens += tokens
    if batch:
        yield batch


class Manifest:
    """Tracks which files have been indexed and their content hashes.

    Each project gets its own manifest at {manifests_dir}/{project_name}.json
    with fcntl.flock-based locking for inter-process safety.
    """

    def __init__(self, project_name: str, manifests_dir: Path | None = None):
        self.project_name = project_name
        self.manifests_dir = manifests_dir or MANIFESTS_DIR
        self.path = self.manifests_dir / f"{project_name}.json"
        self.lock_path = self.manifests_dir / f"{project_name}.lock"
        self.data: dict[str, str] = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text())

    def _file_hash(self, file_path: Path) -> str:
        content = file_path.read_bytes()
        return hashlib.sha256(content).hexdigest()

    def needs_indexing(self, file_path: Path) -> tuple[bool, str]:
        """Check if file needs indexing. Returns (needs_indexing, current_hash).

        H6: Returns pre-computed hash so callers can pass it to mark_indexed
        without re-reading the file.
        """
        current_hash = self._file_hash(file_path)
        key = str(file_path)
        needs = key not in self.data or self.data[key] != current_hash
        return needs, current_hash

    def mark_indexed(self, file_path: Path, file_hash: str) -> None:
        """Mark file as indexed using pre-computed hash.

        H6: Accepts hash instead of re-reading the file.
        """
        self.data[str(file_path)] = file_hash

    def prune(self) -> list[str]:
        """Remove entries whose source file no longer exists on disk.

        Returns the REMOVED KEYS (was a bare count) so the caller can delete the
        corresponding chunks from chroma + BM25 -- without that, a deleted file's
        vectors live forever and keep ranking against current content (the
        prune-orphan fix, Inc 1.5a). A legacy `session:{path}` key (sessions are
        no longer indexed) reads as a non-existent literal path and is pruned as
        stale junk -- harmless cleanup of old manifests.
        """
        stale = [key for key in self.data if not Path(key).exists()]
        for key in stale:
            del self.data[key]
        return stale

    def prune_out_of_scope(
        self,
        in_scope: set[Path],
        roots: list[Path],
        extensions: set[str] | None = None,
    ) -> list[str]:
        """Remove code-file entries that are under any of `roots` but not in `in_scope`.

        Used after exclude_dirs filtering: a file may still exist on disk but be
        out of scope for indexing. Only entries beneath one of the supplied
        `roots` are considered -- code files for unrelated projects/code_dirs
        sharing this manifest must be left alone.

        When `extensions` is given, only keys whose file suffix is in that set
        are eligible for pruning. F relies on this: an in-repo `.md` under a
        code_dir is a DOCS source (its key is owned by `index_docs`), so scoping
        the code prune to the code extensions stops it from deleting that key
        every run -- which would otherwise make `index_docs` re-embed the `.md`
        perpetually (the manifest namespace is shared by bare absolute path).
        `.md` code chunks are removed by the explicit `_sweep_md_code_chunks`
        instead, not by this manifest prune.

        Returns the list of removed manifest keys (string paths) so callers can
        delete the corresponding chunks from chromadb.

        NOT SAFE under concurrent runs: state read in __init__ is mutated here
        and persisted in save() under LOCK_EX, but the read is unlocked. A
        concurrent run that writes a hash for a file this run has decided to
        prune will lose that hash on rename. Cron serializes runs in practice;
        prune sets are tiny; no operational impact observed.
        """
        resolved_roots = [r.resolve() for r in roots]
        in_scope_str = {str(p) for p in in_scope}
        stale: list[str] = []
        for key in list(self.data):
            if key.startswith("session:"):
                continue
            if key in in_scope_str:
                continue
            if extensions is not None and Path(key).suffix not in extensions:
                continue
            try:
                key_resolved = Path(key).resolve()
            except OSError:
                continue
            for root in resolved_roots:
                try:
                    key_resolved.relative_to(root)
                except ValueError:
                    continue
                stale.append(key)
                break
        for key in stale:
            del self.data[key]
        return stale

    def save(self) -> None:
        import os
        import tempfile

        self.manifests_dir.mkdir(parents=True, exist_ok=True)

        # Atomic write: temp file + rename prevents corruption on crash
        fd, tmp_path = tempfile.mkstemp(dir=str(self.manifests_dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.data, f, indent=2)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        try:
            try:
                import fcntl
                lock_fd = open(self.lock_path, "w")
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                    os.replace(tmp_path, str(self.path))
                finally:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
            except ImportError:
                # fcntl unavailable (Windows) — rename without locking
                os.replace(tmp_path, str(self.path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"{ts} {msg}", file=sys.stderr)


def migrate_global_manifest(
    global_path: Path,
    manifests_dir: Path,
    config: VecsConfig,
) -> None:
    """Migrate global manifest.json to per-project manifest files.

    Algorithm:
    1. Load global manifest
    2. For each entry, check if file_path starts with any project's
       code_dirs[].path or docs_dir
    3. Write matched entries to per-project manifest files
    4. Write unmatched entries to _orphaned.json
    5. Rename old manifest to manifest.json.bak
    """
    if not global_path.exists():
        return

    # Skip if per-project manifests already exist (already migrated)
    if manifests_dir.exists() and any(manifests_dir.glob("*.json")):
        return

    data = json.loads(global_path.read_text())
    if not data:
        return

    # Build prefix -> project_name mapping
    project_buckets: dict[str, dict[str, str]] = {}  # project_name -> {path: hash}
    orphaned: dict[str, str] = {}

    for file_path_str, file_hash in data.items():
        matched_project = None
        for proj_name, proj in config.projects.items():
            # Check code_dirs
            for cd in proj.code_dirs:
                if file_path_str.startswith(str(cd.path)):
                    matched_project = proj_name
                    break
            if matched_project:
                break
            # Check docs_dir
            if proj.docs_dir and file_path_str.startswith(str(proj.docs_dir)):
                matched_project = proj_name
                break

        if matched_project:
            project_buckets.setdefault(matched_project, {})[file_path_str] = file_hash
        else:
            orphaned[file_path_str] = file_hash

    # Write per-project manifests
    manifests_dir.mkdir(parents=True, exist_ok=True)
    for proj_name, entries in project_buckets.items():
        (manifests_dir / f"{proj_name}.json").write_text(json.dumps(entries, indent=2))

    # Write orphaned entries
    if orphaned:
        (manifests_dir / "_orphaned.json").write_text(json.dumps(orphaned, indent=2))

    # Backup and remove old manifest
    backup_path = global_path.with_suffix(".json.bak")
    global_path.rename(backup_path)

    _log(f"Migrated {len(data)} manifest entries to per-project files "
         f"({len(orphaned)} orphaned).")


def _make_chunk_id(source_key: str, chunk_index: int) -> str:
    return f"{source_key}:{chunk_index}"


def _git_sha(path: Path) -> str | None:
    """Best-effort current commit SHA of the git work tree containing `path`.

    Returns None when `path` is not inside a git repo or git is unavailable.
    Used as the code `version_id` (C): the repo HEAD at index time, so a later
    HEAD lets stale-retrieval detection (E) tell when a chunk is out of date.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _delete_stale_chunks_after_embed(
    collection: chromadb.Collection,
    metadata_key: str,
    metadata_value: str,
    new_ids: set[str],
) -> None:
    """Delete orphaned chunks that no longer exist in the new version of a file.

    Called AFTER embedding succeeds. Queries ChromaDB for all chunks matching
    the metadata filter, diffs against new_ids, and deletes orphans.
    Failures are swallowed -- duplicates self-heal on next index run.
    """
    try:
        existing = collection.get(where={metadata_key: metadata_value})
        orphan_ids = [eid for eid in existing["ids"] if eid not in new_ids]
        if orphan_ids:
            collection.delete(ids=orphan_ids)
    except Exception as e:
        _log(f"  Warning: stale chunk cleanup failed for {metadata_key}={metadata_value}: {e}")


def _calibrate_from_error(batcher: AdaptiveBatcher, batch_chars: int, error_msg: str) -> None:
    """Try to extract token count from Voyage error message for calibration."""
    # Voyage errors typically contain something like "total tokens: 150000"
    match = re.search(r"(\d[\d,]+)\s*tokens", error_msg)
    if match:
        try:
            actual_tokens = int(match.group(1).replace(",", ""))
            if actual_tokens > 0:
                batcher.calibrate(batch_chars, actual_tokens)
        except ValueError:
            pass


def _embed_and_store(
    chunks: list[dict],
    collection: chromadb.Collection,
    model: str,
    provider: EmbedProvider,
    batcher: AdaptiveBatcher | None = None,
    cache: EmbedCache | None = None,
) -> list[str]:
    """Embed and store chunks. Returns list of successfully stored chunk IDs.

    Uses AdaptiveBatcher for token estimation when provided. Calibrates from
    EmbedResult.total_tokens when the provider reports usage.

    When a `cache` is supplied (C), byte-identical chunks already embedded under
    this `model` are served from the cache without a provider call -- but they
    are STILL upserted and counted in the returned ids, so the caller's
    `succeeded == expected` manifest invariant holds (see _track_embed_success).
    Freshly-embedded (cache-miss) vectors are written back to the cache.
    """
    if not chunks:
        return []

    if batcher is None:
        batcher = AdaptiveBatcher()

    succeeded_ids: list[str] = []

    # C: partition into cache hits (served + upserted, no Voyage call) and
    # misses (embedded below, then written back to the cache). A cache READ
    # failure (e.g. "database is locked" under overlapping reindex) degrades to
    # embedding everything this run -- it must never abort indexing.
    chunks_to_embed = chunks
    if cache is not None:
        cached: dict[str, list[float]] = {}
        hashes: list[str] = []
        try:
            hashes = [EmbedCache.content_hash(c["text"]) for c in chunks]
            cached = cache.get(model, hashes)
        except Exception as e:
            _log(f"  Cache read failed ({e}); embedding all chunks this run.")
            cached = {}
        if cached:
            hit_chunks: list[dict] = []
            hit_embeddings: list[list[float]] = []
            misses: list[dict] = []
            for c, h in zip(chunks, hashes):
                if h in cached:
                    hit_chunks.append(c)
                    hit_embeddings.append(cached[h])
                else:
                    misses.append(c)
            if hit_chunks:
                collection.upsert(
                    ids=[c["id"] for c in hit_chunks],
                    embeddings=hit_embeddings,
                    documents=[c["text"] for c in hit_chunks],
                    metadatas=[c["metadata"] for c in hit_chunks],
                )
                succeeded_ids.extend(c["id"] for c in hit_chunks)
                _log(f"  Cache: {len(hit_chunks)} hit, {len(misses)} to embed")
            chunks_to_embed = misses

    for batch in _make_batches(chunks_to_embed, batcher):
        texts = [c["text"] for c in batch]
        batch_chars = sum(len(t) for t in texts)

        for attempt in range(5):
            try:
                result = provider.embed(texts, model=model, input_type="document")
                break
            except Exception as e:
                error_msg = str(e).lower()
                is_transient = (
                    isinstance(e, provider.retryable_errors)
                    or isinstance(e, (TimeoutError, ConnectionError))
                    or ("rate" in error_msg and "limit" in error_msg)
                )
                if is_transient:
                    wait = 20 * (attempt + 1)
                    err_excerpt = str(e).splitlines()[0][:160] if str(e) else ""
                    _log(f"  {type(e).__name__} (attempt {attempt + 1}/5), waiting {wait}s: {err_excerpt}")
                    time.sleep(wait)
                else:
                    # H2: try to parse token count from error for calibration
                    _calibrate_from_error(batcher, batch_chars, str(e))
                    raise
        else:
            sample_id = batch[0].get("id", "<unknown>") if batch else "<empty>"
            _log(f"  Failed after 5 retries, skipping batch of {len(batch)} chunks "
                 f"(sample chunk_id={sample_id}, {len(succeeded_ids)} stored so far)")
            continue

        # H2: calibrate from successful response (provider-neutral usage field)
        if result.total_tokens is not None:
            batcher.calibrate(batch_chars, result.total_tokens)

        ids = [c["id"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        collection.upsert(
            ids=ids,
            embeddings=result.embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        succeeded_ids.extend(ids)
        # C: write freshly-embedded vectors back to the cache, keyed by
        # (model, content_hash of the EMBEDDED text). Hashing the post-batch
        # text -- not the pre-truncation original -- keeps the key aligned with
        # the bytes actually embedded (_make_batches may truncate an oversized
        # chunk). Never fatal: a failed write only costs a re-embed next run.
        if cache is not None:
            try:
                cache.put(
                    model,
                    [
                        (EmbedCache.content_hash(t), emb)
                        for t, emb in zip(texts, result.embeddings)
                    ],
                )
            except Exception as e:
                _log(f"  Cache write failed ({e}); continuing.")
        _log(f"  Indexed {len(succeeded_ids)}/{len(chunks)} chunks")

    return succeeded_ids


def _paginated_delete(
    collection: chromadb.Collection,
    ids: list[str],
    batch_size: int = 5000,
) -> None:
    """Delete chunks in batches to stay under SQLITE_MAX_VARIABLE_NUMBER."""
    for i in range(0, len(ids), batch_size):
        collection.delete(ids=ids[i:i + batch_size])


def _paginated_get(
    collection: chromadb.Collection,
    batch_size: int = 5000,
    **kwargs,
) -> Iterator[dict]:
    """Yield pages of `collection.get()` results, paginating by limit/offset.

    chromadb's unbounded `collection.get()` builds an internal SQL query whose
    IN-clause can blow past SQLITE_MAX_VARIABLE_NUMBER (32766 modern, 999 old).
    Paginating with limit=5000 stays safely under both caps.

    Stops as soon as a page returns zero ids.
    """
    offset = 0
    while True:
        page = collection.get(limit=batch_size, offset=offset, **kwargs)
        ids = page.get("ids") or []
        if not ids:
            return
        yield page
        if len(ids) < batch_size:
            return
        offset += len(ids)


def _sync_bm25(collection: chromadb.Collection, project_name: str, suffix: str) -> None:
    """Incrementally sync the BM25 FTS5 index with the given ChromaDB collection.

    Diff strategy:
      1. Fetch all chunk ids + documents + metadatas from chromadb (paginated).
      2. Compare ids against the existing BM25 index's id set.
      3. Delete BM25 rows whose ids are no longer in chromadb.
      4. Upsert all chromadb-resident chunks (insert new + overwrite existing).

    Performance: deletes are O(diff size); upserts are O(N) since every
    chromadb-resident chunk is upserted unconditionally (no content-addressable
    diffing). The win over the previous _rebuild_bm25 is eliminating the
    in-memory BM25Okapi rebuild and the full pickle write — tokenization
    still runs for every chunk, but FTS5's incremental writes amortize.

    Migrates transparently: when no .db exists yet, step 4 simply inserts
    everything (equivalent to a fresh build). Old `.pkl` files (if any) are
    deleted on first successful sync.
    """
    bm25_dir = VECS_DIR / "bm25"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    db_path = bm25_dir / f"{project_name}_{suffix}.db"
    legacy_pkl = bm25_dir / f"{project_name}_{suffix}.pkl"

    bm25 = BM25Index(db_path)
    try:
        bm25.load()
        chroma_docs: list[dict] = []
        chroma_ids: set[str] = set()
        for page in _paginated_get(collection, include=["documents", "metadatas"]):
            for id_, text, meta in zip(
                page["ids"], page["documents"], page["metadatas"]
            ):
                chroma_ids.add(id_)
                chroma_docs.append({"id": id_, "text": text, "metadata": meta or {}})

        existing_ids = bm25.all_ids()
        to_delete = sorted(existing_ids - chroma_ids)
        if to_delete:
            bm25.delete(to_delete)
        if chroma_docs:
            bm25.upsert(chroma_docs)

        # One-shot cleanup of the obsolete pickle, if it exists
        if legacy_pkl.exists():
            try:
                legacy_pkl.unlink()
            except OSError:
                pass
    except Exception as e:
        _log(f"  Warning: BM25 sync failed for {project_name}_{suffix}: {e}")
    finally:
        bm25.close()


def _delete_ids_from_bm25(project_name: str, suffix: str, ids: list[str]) -> None:
    """Delete specific `doc_id`s from a project's BM25 FTS5 sidecar.

    Used by the targeted F sweeps (`-code` `.md` sweep, `-docs` orphan sweep) to
    keep the sidecar in lockstep with chroma when the run's end-of-pass
    `_sync_bm25` would not fire -- it only runs when `total_stored > 0`, so a
    sweep-only run (nothing new to embed, but chunks deleted) would otherwise
    leave stale BM25 rows behind. No-op when `ids` is empty or no `.db` exists.
    """
    if not ids:
        return
    db_path = VECS_DIR / "bm25" / f"{project_name}_{suffix}.db"
    if not db_path.exists():
        return
    bm25 = BM25Index(db_path)
    try:
        bm25.load()
        bm25.delete(ids)
    except Exception as e:
        _log(f"  Warning: BM25 targeted delete failed for {project_name}_{suffix}: {e}")
    finally:
        bm25.close()


def _sweep_excluded_chunks(
    collection: chromadb.Collection,
    code_dir: CodeDir,
) -> int:
    """Delete chromadb chunks whose file_path is under any of code_dir.exclude_dirs.

    Manifest-driven cleanup misses orphan chunks for files that were embedded
    mid-batch in a prior run that died before manifest.save(). This sweep is
    metadata-only -- it scans every chunk in the collection and drops any whose
    `file_path` starts with `{code_dir.path.name}/{excluded_subdir}/`.

    Returns the number of chunks deleted. Failures are logged and swallowed.
    """
    if not code_dir.exclude_dirs:
        return 0

    dir_prefix = code_dir.path.name
    excluded_prefixes = tuple(
        f"{dir_prefix}/{sub.rstrip('/')}/" for sub in code_dir.exclude_dirs
    )

    orphan_ids: list[str] = []
    try:
        for page in _paginated_get(collection, include=["metadatas"]):
            for cid, meta in zip(page["ids"], page["metadatas"]):
                fp = (meta or {}).get("file_path")
                if isinstance(fp, str) and fp.startswith(excluded_prefixes):
                    orphan_ids.append(cid)
        if orphan_ids:
            _paginated_delete(collection, orphan_ids)
    except Exception as e:
        _log(f"  Warning: orphan-chunk sweep failed for {dir_prefix}: {e}")
        return 0

    return len(orphan_ids)


def _sweep_out_of_scope_docs(
    collection: chromadb.Collection,
    stale_file_paths: set[str],
    root_map: dict[str, Path],
) -> list[str]:
    """Delete `-docs` chunks whose qualified `file_path` is in `stale_file_paths`.

    These are chunks present under a still-VALID source root but whose source
    file is no longer an in-scope `_docs_sources` member — a newly-hidden dir
    (`.claude/…`) or a newly-excluded subdir. `_partition_docs_by_root` keeps them
    (root is valid, not an orphan) and the Inc 1.5a deleted-source sweep keeps
    them (file still on disk), so without this they would linger forever. The
    caller computes `present_file_paths - in_scope_qualified`.

    Deletes ONLY a chunk whose source file STILL EXISTS on disk under a known,
    on-disk `root_map` root (present-but-out-of-scope is unambiguously stale). A
    candidate whose root basename is absent from `root_map` (root not on disk
    this scan, or non-unique basename) or whose file no longer exists is KEPT —
    a transiently-missing root must never wipe its live chunks (mirrors
    `_safe_sweep_root_map` + `_sweep_deleted_source_chunks`), and a genuinely
    deleted file is the Inc 1.5a deleted-source sweep's job. Metadata-only scan;
    returns deleted ids for the BM25 mirror. Failures logged + swallowed.
    """
    if not stale_file_paths or not root_map:
        return []

    # Stat each distinct candidate file at most once. True = "present under a
    # known root -> out of scope, delete"; False = "missing root / gone / unstattable
    # -> keep (an orphan is safer than deleting a transiently-absent root's data)".
    present: dict[str, bool] = {}

    def _present_under_known_root(fp: str) -> bool:
        if fp in present:
            return present[fp]
        if "/" not in fp:
            present[fp] = False
            return False
        root_name, rel = fp.split("/", 1)
        root = root_map.get(root_name)
        if root is None:
            present[fp] = False  # root absent this scan -> keep, never wipe
        else:
            try:
                present[fp] = (root / rel).exists()
            except OSError:
                present[fp] = False  # unstattable -> keep (never delete live data)
        return present[fp]

    ids: list[str] = []
    try:
        for page in _paginated_get(collection, include=["metadatas"]):
            for cid, meta in zip(page["ids"], page["metadatas"]):
                fp = (meta or {}).get("file_path")
                if (
                    isinstance(fp, str)
                    and fp in stale_file_paths
                    and _present_under_known_root(fp)
                ):
                    ids.append(cid)
        if ids:
            _paginated_delete(collection, ids)
    except Exception as e:
        _log(f"  Warning: out-of-scope -docs sweep failed: {e}")
        return []
    return ids


def _sweep_md_code_chunks(collection: chromadb.Collection) -> list[str]:
    """Delete chunks whose source file is a `.md` from a `-code` collection.

    F drops `.md` from `code_dirs` extensions (`.md` routes to the `-docs`
    collection instead), so `index_code` stops emitting `.md` chunks -- but the
    already-embedded `.md` code chunks remain, because `index_code` only adds.
    This metadata-only sweep removes them: it scans every chunk and drops any
    whose `file_path` ends in `.md` (code chunk `file_path` =
    `{code_dir.path.name}/{rel}`). Returns the deleted ids so the caller can
    mirror the deletion into the BM25 sidecar. Failures are logged + swallowed.
    """
    md_ids: list[str] = []
    try:
        for page in _paginated_get(collection, include=["metadatas"]):
            for cid, meta in zip(page["ids"], page["metadatas"]):
                fp = (meta or {}).get("file_path")
                if isinstance(fp, str) and fp.endswith(".md"):
                    md_ids.append(cid)
        if md_ids:
            _paginated_delete(collection, md_ids)
    except Exception as e:
        _log(f"  Warning: .md code-chunk sweep failed: {e}")
        return []

    return md_ids


def _partition_docs_by_root(
    collection: chromadb.Collection,
    valid_root_names: set[str],
) -> tuple[list[str], set[str]]:
    """Single scan of a `-docs` collection, partitioning chunks by source root.

    F qualifies docs chunk ids/file_paths as `{root.name}/{rel}` for every source
    root (docs_dirs + code_dirs). Returns `(orphan_ids, present_file_paths)`:

    - `orphan_ids`: chunks whose `file_path` is NOT under any current source-root
      prefix -- legacy pre-F chunks (bare `relative_to` paths, e.g. `HQ/x.md`)
      and chunks whose source root was removed from config. The caller deletes
      these (chroma + BM25).
    - `present_file_paths`: the set of `file_path` values that ARE source-root-
      qualified under a current root. `index_docs` uses this to force a re-embed
      of any source file whose qualified chunks are ABSENT -- the half of the
      id-scheme migration that the delete alone cannot do, since the shared
      bare-abs-path manifest key still carries a matching content hash (so
      `needs_indexing` would otherwise skip the file and the content would be
      lost, not migrated). This makes the migration self-converge WITHOUT
      depending on a coincident embedding-model change.

    Empty `valid_root_names` -> `([], set())` WITHOUT scanning: a degenerate
    empty prefix set must never classify every chunk as an orphan (which would
    wipe the collection). Failures logged + swallowed.
    """
    valid_prefixes = tuple(f"{name}/" for name in valid_root_names)
    if not valid_prefixes:
        return [], set()

    orphan_ids: list[str] = []
    present: set[str] = set()
    try:
        for page in _paginated_get(collection, include=["metadatas"]):
            for cid, meta in zip(page["ids"], page["metadatas"]):
                fp = (meta or {}).get("file_path")
                if not isinstance(fp, str):
                    continue
                if fp.startswith(valid_prefixes):
                    present.add(fp)
                else:
                    orphan_ids.append(cid)
    except Exception as e:
        _log(f"  Warning: -docs partition scan failed: {e}")
        return [], set()

    return orphan_ids, present


def _sweep_deleted_source_chunks(
    collection: chromadb.Collection,
    root_map: dict[str, Path],
) -> list[str]:
    """Delete chunks whose source-root-qualified `file_path` is gone on disk.

    The prune-orphan fix for code/docs (Inc 1.5a): a deleted file's chunks keep
    their `file_path` (`{root.name}/{rel}`), so we resolve it back to disk via
    `root_map` (root basename -> root path) and drop the chunk when the source no
    longer exists. Catches BOTH newly-deleted files and the backlog the buggy
    `prune()` leaked (their manifest entry is already gone, so a manifest diff
    can't find them). Returns the deleted ids so the caller mirrors the deletion
    into the BM25 sidecar.

    Conservative on unknown roots: a `file_path` whose first segment is not a
    current root is LEFT ALONE -- legacy bare-scheme `-docs` chunks are owned by
    `_partition_docs_by_root`, and an unknown code root is out of scope here.
    Empty `root_map` -> no scan, no deletes (a degenerate/misconfigured root set
    must never be read as "every chunk is an orphan" and wipe the collection).
    Failures logged + swallowed.
    """
    if not root_map:
        return []

    orphan_ids: list[str] = []
    # Cache existence by file_path: many chunks share one source file, so we
    # stat each distinct file at most once (keeps the every-reindex sweep cheap
    # on large collections). True = "source present, keep"; False = "gone, drop".
    present: dict[str, bool] = {}

    def _source_present(fp: str) -> bool:
        if fp in present:
            return present[fp]
        root_name, rel = fp.split("/", 1)
        root = root_map.get(root_name)
        if root is None:
            present[fp] = True  # unknown root -> not this sweep's job, keep
        else:
            try:
                present[fp] = (root / rel).exists()
            except OSError:
                present[fp] = True  # unstattable -> keep (never delete live data)
        return present[fp]

    try:
        for page in _paginated_get(collection, include=["metadatas"]):
            for cid, meta in zip(page["ids"], page["metadatas"]):
                fp = (meta or {}).get("file_path")
                if not isinstance(fp, str) or "/" not in fp:
                    continue
                if not _source_present(fp):
                    orphan_ids.append(cid)
        if orphan_ids:
            _paginated_delete(collection, orphan_ids)
    except Exception as e:
        _log(f"  Warning: deleted-source orphan sweep failed: {e}")
        return []

    return orphan_ids


def _track_embed_success(
    succeeded_ids: list[str],
    chunk_to_file: dict[str, Path],
    file_expected_count: dict[Path, int],
    file_cleanup: dict[Path, tuple[str, str, set[str]]],
    collection: chromadb.Collection,
) -> set[Path]:
    """Determine which files had ALL chunks embedded. Clean up orphaned chunks.

    Returns the set of file paths where every chunk was successfully stored.
    """
    succeeded_per_file: dict[Path, int] = {}
    for cid in succeeded_ids:
        fp = chunk_to_file[cid]
        succeeded_per_file[fp] = succeeded_per_file.get(fp, 0) + 1

    fully_succeeded: set[Path] = set()
    for fp, expected in file_expected_count.items():
        if succeeded_per_file.get(fp, 0) == expected:
            fully_succeeded.add(fp)
            # C3: delete orphans after successful embed
            if fp in file_cleanup:
                mk, mv, expected_ids = file_cleanup[fp]
                _delete_stale_chunks_after_embed(collection, mk, mv, expected_ids)

    return fully_succeeded


def _index_collection(
    chunks: list[dict],
    collection: chromadb.Collection,
    model: str,
    provider: EmbedProvider,
    manifest: Manifest,
    chunk_to_file: dict[str, Path],
    file_expected_count: dict[Path, int],
    file_cleanup: dict[Path, tuple[str, str, set[str]]],
    files_to_process: list[Path],
    file_hashes: dict[Path, str],
    cache: EmbedCache | None = None,
) -> int:
    """Shared indexing pipeline: embed, track success, cleanup orphans, save manifest.

    Args:
        chunks: All chunks to embed (batching handled internally).
        collection: ChromaDB collection to upsert into.
        model: Embedding model name.
        provider: Embedding provider (the seam over voyage / local models).
        manifest: Manifest instance for tracking indexed files.
        chunk_to_file: Map from chunk ID to source file path.
        file_expected_count: Map from file path to expected chunk count.
        file_cleanup: Map from file path to (metadata_key, metadata_value, chunk_ids).
        files_to_process: Ordered list of files that produced chunks.
        file_hashes: Map from file path to pre-computed SHA-256 hash (H6).

    Returns:
        Number of successfully stored chunks.
    """
    if not chunks:
        return 0

    succeeded_ids = _embed_and_store(chunks, collection, model, provider, cache=cache)

    fully_succeeded = _track_embed_success(
        succeeded_ids, chunk_to_file, file_expected_count, file_cleanup, collection,
    )
    for f in files_to_process:
        if f in fully_succeeded:
            manifest.mark_indexed(f, file_hashes[f])

    manifest.save()
    return len(succeeded_ids)


def _under_hidden_dir(path: Path, root: Path) -> bool:
    """True if `path` lives under a hidden DIRECTORY relative to `root` (any
    parent path component starts with ``.``).

    Hidden dirs (``.git``, ``.claude``, ``.github``, ``.agents``, caches, …) are
    tooling/config/scaffolding — essentially never the code or docs a coding
    agent should retrieve — so the scanners skip them everywhere. Only hidden
    DIRECTORIES are excluded here; a hidden FILE is left to the extension filter.
    Returns False when `path` is not under `root`.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    return any(part.startswith(".") for part in rel.parts[:-1])


def _scan_code_dir(code_dir: CodeDir, extensions: set[str]) -> list[Path]:
    """Discover files under `code_dir` matching `extensions`, skipping hidden
    dirs and applying the code_dir's include_dirs / exclude_dirs filters (exclude
    wins on overlap).

    Extracted so `index_code` and `index_docs` scope a code_dir scan identically:
    F routes in-repo `.md` to the -docs collection using the SAME include/exclude
    scope the code scan uses, so no in-scope `.md` is dropped in the move and no
    third-party `.md` (under an excluded subdir) is pulled in.
    """
    if code_dir.include_dirs:
        files: list[Path] = []
        for subdir in code_dir.include_dirs:
            d = code_dir.path / subdir
            if d.exists():
                files.extend(
                    f for f in d.rglob("*")
                    if f.suffix in extensions and f.is_file()
                )
    else:
        files = [
            f for f in code_dir.path.rglob("*")
            if f.suffix in extensions and f.is_file()
        ]

    # Skip hidden dirs (.git/.claude/.github/.agents/caches/…): tooling, not code/docs.
    files = [f for f in files if not _under_hidden_dir(f, code_dir.path)]

    # exclude_dirs: drop files whose path is under any excluded subdir.
    if code_dir.exclude_dirs:
        excluded_roots = [
            (code_dir.path / sub).resolve() for sub in code_dir.exclude_dirs
        ]
        kept: list[Path] = []
        for f in files:
            f_resolved = f.resolve()
            drop = False
            for ex in excluded_roots:
                try:
                    f_resolved.relative_to(ex)
                    drop = True
                    break
                except ValueError:
                    continue
            if not drop:
                kept.append(f)
        files = kept

    return files


def index_code(
    project: ProjectConfig,
    provider: EmbedProvider,
    db: chromadb.ClientAPI,
    cache: EmbedCache | None = None,
) -> int:
    """Index code files for a project. Returns count of new chunks."""
    if not project.code_dirs:
        _log(f"[{project.name}] No code_dirs configured, skipping.")
        return 0

    manifest = Manifest(project.name)
    collection = db.get_or_create_collection(project.code_collection)

    all_chunks: list[dict] = []
    files_to_process: list[Path] = []
    chunk_to_file: dict[str, Path] = {}
    file_expected_count: dict[Path, int] = {}
    file_cleanup: dict[Path, tuple[str, str, set[str]]] = {}
    file_hashes: dict[Path, str] = {}

    # Track in-scope files (+ code extensions) per code_dir so we can compute
    # proper chromadb rel_paths when pruning out-of-scope manifest entries, and
    # so the prune stays scoped to code extensions (it must not touch in-repo
    # .md keys, which index_docs owns).
    in_scope_by_root: list[tuple[Path, set[Path], set[str]]] = []

    for code_dir in project.code_dirs:
        if not code_dir.path.exists():
            _log(f"[{project.name}] {code_dir.path}: directory not found, skipping.")
            continue

        # F: .md is a -docs source, never a code extension. Strip it defensively
        # so a stale config (.md still listed) cannot make index_code embed .md
        # as code and fight the .md sweep (embed-then-sweep thrash + dual store).
        code_exts = set(code_dir.extensions)
        if ".md" in code_exts:
            code_exts.discard(".md")
            _log(
                f"[{project.name}] {code_dir.path}: '.md' routes to the -docs "
                f"collection (F), not -code -- ignoring it as a code extension. "
                f"Remove '.md' from this code_dir's extensions in config.yaml."
            )

        files = _scan_code_dir(code_dir, code_exts)

        in_scope_by_root.append((code_dir.path, set(files), code_exts))

        # H6: needs_indexing returns (bool, hash) -- hash computed once
        to_index: list[Path] = []
        for f in files:
            needs, fhash = manifest.needs_indexing(f)
            if needs:
                to_index.append(f)
                file_hashes[f] = fhash

        if not to_index:
            _log(f"[{project.name}] {code_dir.path}: nothing new to index.")
            continue

        _log(f"[{project.name}] {code_dir.path}: {len(to_index)} files to index ({len(files)} total)")

        # C: version_id for code = repo HEAD sha at index time (one git call per
        # code_dir). Falls back to the file content hash for non-git trees.
        code_dir_sha = _git_sha(code_dir.path)

        for f in to_index:
            content = f.read_text(errors="replace")
            dir_prefix = code_dir.path.name
            rel_path = f"{dir_prefix}/{str(f.relative_to(code_dir.path))}"
            chunks = chunk_code_file_ast(
                content, rel_path, chunk_lines=CODE_CHUNK_LINES, overlap=CODE_CHUNK_OVERLAP
            )
            file_version = code_dir_sha or file_hashes[f]
            chunk_ids_for_file: set[str] = set()
            for c in chunks:
                c["id"] = _make_chunk_id(f"code:{rel_path}", c["metadata"]["chunk_index"])
                c["metadata"]["version_id"] = file_version
                chunk_to_file[c["id"]] = f
                chunk_ids_for_file.add(c["id"])
            file_expected_count[f] = len(chunks)
            file_cleanup[f] = ("file_path", rel_path, chunk_ids_for_file)
            all_chunks.extend(chunks)
            files_to_process.append(f)

    # Prune manifest entries now out of scope (e.g. newly excluded subdirs).
    # We iterate per code_dir so we can compute the same rel_path that was
    # used as chromadb metadata, enabling targeted chunk deletion.
    total_pruned = 0
    for root, in_scope, extensions in in_scope_by_root:
        stale_keys = manifest.prune_out_of_scope(in_scope, [root], extensions)
        if not stale_keys:
            continue
        total_pruned += len(stale_keys)
        dir_prefix = root.name
        root_resolved = root.resolve()
        for key in stale_keys:
            try:
                rel = Path(key).resolve().relative_to(root_resolved)
                rel_path = f"{dir_prefix}/{rel}"
                _delete_stale_chunks_after_embed(
                    collection, "file_path", rel_path, set()
                )
            except Exception as e:
                _log(f"  Warning: chromadb cleanup for {key} failed: {e}")
    if total_pruned:
        _log(
            f"[{project.name}] pruned {total_pruned} manifest entries "
            f"now out of scope (excluded subdirs)."
        )
        manifest.save()

    # Sweep chromadb for orphan chunks under exclude_dirs that the manifest
    # never knew about (e.g. files embedded mid-batch in a prior failed run).
    # Manifest-driven prune above only touches tracked entries -- this is the
    # belt-and-suspenders pass that catches everything else.
    for code_dir in project.code_dirs:
        if not code_dir.exclude_dirs or not code_dir.path.exists():
            continue
        swept = _sweep_excluded_chunks(collection, code_dir)
        if swept:
            _log(
                f"[{project.name}] swept {swept} orphan chromadb chunks "
                f"under excluded subdirs of {code_dir.path}."
            )

    # F: .md is no longer a code extension (it routes to -docs). Sweep any
    # already-embedded .md chunks out of the -code collection + its BM25
    # sidecar. Runs unconditionally (idempotent no-op once clean), because the
    # residue persists even on runs with nothing new to index.
    swept_md = _sweep_md_code_chunks(collection)
    if swept_md:
        _delete_ids_from_bm25(project.name, "code", swept_md)
        _log(
            f"[{project.name}] swept {len(swept_md)} .md-sourced chunks out of "
            f"the -code collection (.md now routes to -docs)."
        )

    if not all_chunks:
        return 0

    total_stored = _index_collection(
        chunks=all_chunks,
        collection=collection,
        model=CODE_MODEL,
        provider=provider,
        manifest=manifest,
        chunk_to_file=chunk_to_file,
        file_expected_count=file_expected_count,
        file_cleanup=file_cleanup,
        files_to_process=files_to_process,
        file_hashes=file_hashes,
        cache=cache,
    )

    if total_stored > 0:
        _sync_bm25(collection, project.name, "code")

    return total_stored


def _docs_source_root_names(project: ProjectConfig) -> set[str]:
    """Configured source-root basenames for the -docs collection.

    Drawn from config (not disk), so a transiently-missing dir is not treated
    as removed. Used by the -docs orphan sweep to decide which chunks are still
    rooted in a current source.
    """
    names = {d.name for d in project.docs_dirs}
    names |= {cd.path.name for cd in project.code_dirs}
    return names


def _safe_sweep_root_map(roots: list[Path]) -> dict[str, Path]:
    """Root-basename -> root path for the deleted-source orphan sweep, including
    ONLY roots that are present on disk AND have a unique basename among `roots`.

    Two guards against false-positive deletion of LIVE chunks (the worst outcome
    per the north star — an orphan is strictly safer than deleting live data);
    both degrade to "leave the chunk", matching the index passes:
      - **transiently-missing root** (unmounted volume / renamed dir): without
        this, every `(root/rel).exists()` would be False and the sweep would wipe
        the whole collection for that root. `index_code` / `_docs_sources` skip a
        missing root (`if not path.exists(): continue`); the sweep must too.
      - **basename collision** (two roots share `.name`): the qualified file_path
        `{root.name}/{rel}` can't distinguish them, so a chunk could resolve
        against the wrong root and a live file be deleted. Such names are dropped
        from the sweep (the documented unique-roots assumption; current roots are
        unique). A collision degrades to leaving an orphan, never a false delete.
    """
    counts: dict[str, int] = {}
    for r in roots:
        counts[r.name] = counts.get(r.name, 0) + 1
    return {r.name: r for r in roots if counts[r.name] == 1 and r.is_dir()}


def _docs_sources(project: ProjectConfig) -> list[tuple[Path, Path]]:
    """Enumerate (source_root, file) for every doc file `index_docs` will index.

    Sources, de-duplicated by resolved path (docs_dirs win on overlap):
      - every `.md`/`.txt`/`.pdf` under each `docs_dir` (root = that docs_dir)
      - every in-repo `.md` under each `code_dir`, using the code_dir's own
        include/exclude scope (root = that `code_dir.path`)

    `root.name` is the rel_path qualifier (mirrors `index_code`). This is the
    single source of truth shared by `index_docs` and `_remodel_clear`, so the
    model-change clear scope can never drift from what `index_docs` re-scans.
    """
    sources: list[tuple[Path, Path]] = []
    seen: set[Path] = set()

    def _add(root: Path, f: Path) -> None:
        rf = f.resolve()
        if rf not in seen:
            seen.add(rf)
            sources.append((root, f))

    for d in project.docs_dirs:
        if not d.exists():
            continue
        for f in d.rglob("*"):
            if (
                f.suffix in DOC_EXTENSIONS
                and f.is_file()
                and not _under_hidden_dir(f, d)
            ):
                _add(d, f)

    for code_dir in project.code_dirs:
        if not code_dir.path.exists():
            continue
        for f in _scan_code_dir(code_dir, {".md"}):
            _add(code_dir.path, f)

    return sources


def _owning_doc_root(project: ProjectConfig, file_path: Path) -> Path | None:
    """The source root a doc file belongs to, for rel_path qualification.

    Mirrors `_docs_sources` precedence (docs_dirs win over code_dirs on overlap)
    so `index_single_doc` (add-document) qualifies a file the SAME way a full
    `index_docs` reindex would -- otherwise add + reindex would disagree on the
    chunk id and double-store the file. Returns None if the file is under no
    configured root.

    Deliberately does NOT honor a code_dir's `exclude_dirs` / hidden-dir skip
    (unlike `_docs_sources` via `_scan_code_dir`): add-document is an EXPLICIT
    per-file user opt-in, so a `.md` under an excluded or hidden (`.`-prefixed)
    subdir is still indexed when added by hand, with an id that AGREES with what
    a full reindex would produce.

    Caveat (changed by the out-of-scope sweep): such a hand-added out-of-scope
    file is TEMPORARY. A later full `index_docs` of that `-docs` collection won't
    re-scan it (out of scan scope) AND `_sweep_out_of_scope_docs` reclaims it
    (present on disk under a valid root, but not an in-scope `_docs_sources`
    member) -- so it survives only until the next full reindex of its project.
    Re-add it afterwards if it must persist.
    """
    for root in [*project.docs_dirs, *(cd.path for cd in project.code_dirs)]:
        try:
            file_path.relative_to(root)
            return root
        except ValueError:
            continue
    return None


def index_docs(
    project: ProjectConfig,
    provider: EmbedProvider,
    db: chromadb.ClientAPI,
    cache: EmbedCache | None = None,
) -> int:
    """Index documentation files for a project. Returns count of new chunks.

    F: multi-source. Doc files come from every `docs_dir` AND in-repo `.md`
    under each `code_dir`; each chunk's id/file_path is qualified with its
    source-root basename (`docs:{root.name}/{rel}`) so two roots' `README.md`
    cannot collide. Every pass also sweeps `-docs` chunks no longer rooted in a
    current source (legacy bare-id chunks, removed roots).
    """
    manifest = Manifest(project.name)
    collection = db.get_or_create_collection(project.docs_collection)

    # One scan partitions the collection into orphans (not under any current
    # source root) and the set of currently-present qualified file_paths.
    orphan_ids, present_paths = _partition_docs_by_root(
        collection, _docs_source_root_names(project)
    )
    if orphan_ids:
        _paginated_delete(collection, orphan_ids)
        _delete_ids_from_bm25(project.name, "docs", orphan_ids)
        _log(
            f"[{project.name}] swept {len(orphan_ids)} -docs chunks not under any "
            f"current source root (legacy/removed-root cleanup)."
        )

    sources = _docs_sources(project)
    if not sources:
        return 0

    # Sweep chunks now OUT OF SCOPE under a still-valid root: a source file that
    # left _docs_sources (newly-hidden dir, newly-excluded subdir) but whose root
    # is unchanged is NOT an orphan (_partition_docs_by_root keeps it) and NOT a
    # deleted source (1.5a keeps it -- file still on disk). present_file_paths
    # (valid-root chunks) minus the in-scope qualified set = exactly those stale
    # chunks. Guarded by the non-empty `sources` check above, so an empty scope
    # can never wipe the collection.
    in_scope_qualified = {f"{root.name}/{f.relative_to(root)}" for root, f in sources}
    stale_paths = present_paths - in_scope_qualified
    if stale_paths:
        # Only sweep candidates whose root is on disk THIS scan with a unique
        # basename (same guard as the 1.5a deleted-source sweep): a transiently
        # missing root degrades to "keep the chunk", never "wipe the root".
        docs_root_map = _safe_sweep_root_map(
            [*project.docs_dirs, *(cd.path for cd in project.code_dirs)]
        )
        stale_ids = _sweep_out_of_scope_docs(collection, stale_paths, docs_root_map)
        if stale_ids:
            _delete_ids_from_bm25(project.name, "docs", stale_ids)
            _log(
                f"[{project.name}] swept {len(stale_ids)} -docs chunks now out of "
                f"scope under a valid root (hidden/excluded subdir)."
            )

    # Migration self-heal: a source file whose qualified chunks are ABSENT from
    # the collection must be re-embedded even when its (shared, possibly
    # code-era) manifest hash matches. Clearing the key forces needs_indexing
    # True. This converges the .md->docs move and the bare->qualified id-scheme
    # change in the SAME run, independent of any embedding-model change.
    for root, f in sources:
        qualified_rel = f"{root.name}/{f.relative_to(root)}"
        if qualified_rel not in present_paths:
            manifest.data.pop(str(f), None)

    to_index: list[tuple[Path, Path]] = []
    file_hashes: dict[Path, str] = {}
    for root, f in sources:
        needs, fhash = manifest.needs_indexing(f)
        if needs:
            to_index.append((root, f))
            file_hashes[f] = fhash

    if not to_index:
        _log(f"[{project.name}] Docs: nothing new to index.")
        return 0

    _log(f"[{project.name}] Docs: {len(to_index)} files to index ({len(sources)} total)")

    all_chunks: list[dict] = []
    chunk_to_file: dict[str, Path] = {}
    file_expected_count: dict[Path, int] = {}
    file_cleanup: dict[Path, tuple[str, str, set[str]]] = {}
    files_to_process: list[Path] = []

    for root, f in to_index:
        rel_path = f"{root.name}/{f.relative_to(root)}"

        if f.suffix == ".pdf":
            content = extract_pdf_text(str(f))
        else:
            content = f.read_text(errors="replace")

        chunks = chunk_doc(content, rel_path)
        # C: version_id for docs = file mtime (revision proxy).
        file_version = str(f.stat().st_mtime)
        chunk_ids_for_file: set[str] = set()
        for c in chunks:
            c["id"] = _make_chunk_id(f"docs:{rel_path}", c["metadata"]["chunk_index"])
            c["metadata"]["version_id"] = file_version
            chunk_to_file[c["id"]] = f
            chunk_ids_for_file.add(c["id"])
        file_expected_count[f] = len(chunks)
        file_cleanup[f] = ("file_path", rel_path, chunk_ids_for_file)
        all_chunks.extend(chunks)
        files_to_process.append(f)

    if not all_chunks:
        return 0

    total_stored = _index_collection(
        chunks=all_chunks,
        collection=collection,
        model=DOCS_MODEL,
        provider=provider,
        manifest=manifest,
        chunk_to_file=chunk_to_file,
        file_expected_count=file_expected_count,
        file_cleanup=file_cleanup,
        files_to_process=files_to_process,
        file_hashes=file_hashes,
        cache=cache,
    )

    if total_stored > 0:
        _sync_bm25(collection, project.name, "docs")

    return total_stored


def index_single_doc(project_name: str, file_path: Path) -> int:
    """Index a single doc file immediately. Returns chunk count."""
    config = load_config()
    if project_name not in config.projects:
        raise ValueError(f"Project '{project_name}' not found.")

    project = config.projects[project_name]
    # F: qualify by the file's OWN source root (any docs_dir, or a code_dir for
    # an in-repo .md), matching index_docs -- so add-document and a later full
    # reindex agree on the chunk id rather than double-storing the file.
    root = _owning_doc_root(project, file_path)
    if root is None:
        raise ValueError(
            f"Project '{project_name}': {file_path} is under no configured "
            f"docs_dir or code_dir."
        )

    provider = get_provider(config)
    db = get_chromadb_client()
    collection = db.get_or_create_collection(project.docs_collection)
    manifest = Manifest(project_name)

    # H6: compute hash once
    _, file_hash = manifest.needs_indexing(file_path)

    rel_path = f"{root.name}/{file_path.relative_to(root)}"

    if file_path.suffix == ".pdf":
        content = extract_pdf_text(str(file_path))
    else:
        content = file_path.read_text(errors="replace")

    chunks = chunk_doc(content, rel_path)
    # C: stamp the same mtime version_id index_docs uses, so add-document and a
    # later full reindex agree on the version_id for this -docs file.
    file_version = str(file_path.stat().st_mtime)
    chunk_to_file: dict[str, Path] = {}
    chunk_ids: set[str] = set()
    for c in chunks:
        c["id"] = _make_chunk_id(f"docs:{rel_path}", c["metadata"]["chunk_index"])
        c["metadata"]["version_id"] = file_version
        chunk_to_file[c["id"]] = file_path
        chunk_ids.add(c["id"])

    total_stored = _index_collection(
        chunks=chunks,
        collection=collection,
        model=DOCS_MODEL,
        provider=provider,
        manifest=manifest,
        chunk_to_file=chunk_to_file,
        file_expected_count={file_path: len(chunks)},
        file_cleanup={file_path: ("file_path", rel_path, chunk_ids)},
        files_to_process=[file_path],
        file_hashes={file_path: file_hash},
    )

    if total_stored > 0:
        _sync_bm25(collection, project_name, "docs")

    return total_stored


def get_status(project_name: str | None = None) -> dict:
    """Get index status info, optionally filtered to one project."""
    config = load_config()

    if project_name is not None and project_name not in config.projects:
        available = ", ".join(sorted(config.projects.keys())) or "(none)"
        raise ValueError(
            f"Project '{project_name}' not found. Available projects: {available}"
        )

    db = get_chromadb_client()
    projects = (
        {project_name: config.projects[project_name]}
        if project_name
        else config.projects
    )

    status: dict = {"projects": {}, "total_code_chunks": 0, "total_docs_chunks": 0}

    for name, p in projects.items():
        code_count = 0
        docs_count = 0
        try:
            col = db.get_collection(p.code_collection)
            code_count = col.count()
        except Exception:
            pass
        try:
            col = db.get_collection(p.docs_collection)
            docs_count = col.count()
        except Exception:
            pass
        status["projects"][name] = {
            "code_chunks": code_count,
            "docs_chunks": docs_count,
        }
        status["total_code_chunks"] += code_count
        status["total_docs_chunks"] += docs_count

    total_manifest_entries = 0
    if MANIFESTS_DIR.exists():
        for mf in MANIFESTS_DIR.glob("*.json"):
            if mf.name.startswith("_"):
                continue  # skip _orphaned.json
            try:
                data = json.loads(mf.read_text())
                total_manifest_entries += len(data)
            except Exception:
                pass
    status["manifest_entries"] = total_manifest_entries

    return status


def _collection_count(db: chromadb.ClientAPI, name: str) -> int:
    """Chunk count for a collection, or 0 if it does not exist yet."""
    try:
        return db.get_collection(name).count()
    except Exception:
        return 0


def _model_changed(
    cache: EmbedCache, db: chromadb.ClientAPI, collection: str, model: str
) -> bool:
    """True if `collection` must be re-embedded under `model`.

    Re-embed is needed when the model recorded for the collection differs from
    the configured `model` AND the collection actually holds vectors (count>0).
    A never-recorded collection reads None != model, so the first post-deploy
    reindex migrates an existing live store in place; an empty/absent collection
    reads count 0 and is skipped (the next index pass embeds it fresh anyway).
    """
    if cache.get_collection_model(collection) == model:
        return False
    return _collection_count(db, collection) > 0


def _clear_docs_manifest_entries(manifest: Manifest, docs_files: list[Path]) -> int:
    """Drop the manifest key for each doc source file so index_docs re-embeds it.

    F: keyed on the EXACT files index_docs will re-scan (`_docs_sources`:
    docs_dirs ∪ in-repo `.md` under code_dirs), NOT on a directory list. This
    makes clear-scope ≡ rescan-scope by construction -- clearing a key index_docs
    would not re-create strands that file's old-model vectors after the marker
    advances. It also auto-protects code keys: `_docs_sources` enumerates only
    `.md` under code_dirs, never `.cs`, so a code file's key is never cleared.
    Returns the number of keys removed.
    """
    cleared = 0
    for f in docs_files:
        key = str(f)
        if key in manifest.data:
            del manifest.data[key]
            cleared += 1
    return cleared


def _code_sources(project: ProjectConfig) -> list[Path]:
    """Every file index_code will scan, across all code_dirs (mirrors
    `_docs_sources`; shared with the code-model clear so clear-scope ≡
    rescan-scope by construction). `.md` is stripped exactly as index_code
    strips it (it routes to -docs), and a missing code_dir is skipped exactly
    as index_code skips it."""
    files: list[Path] = []
    for code_dir in project.code_dirs:
        if not code_dir.path.exists():
            continue
        code_exts = set(code_dir.extensions)
        code_exts.discard(".md")
        if not code_exts:
            continue
        files.extend(_scan_code_dir(code_dir, code_exts))
    return files


def _clear_code_manifest_entries(manifest: Manifest, code_files: list[Path]) -> int:
    """Code twin of `_clear_docs_manifest_entries`: same bare-abs-path key
    scheme, keyed on the EXACT files index_code will re-scan. Returns the
    number of keys removed."""
    cleared = 0
    for f in code_files:
        key = str(f)
        if key in manifest.data:
            del manifest.data[key]
            cleared += 1
    return cleared


def _remodel_clear(
    project: ProjectConfig, db: chromadb.ClientAPI, cache: EmbedCache
) -> dict[str, int]:
    """run_index PRE-pass: clear manifest entries when an embedding model changed.

    Docs keep their legacy semantics: a None marker reads as changed (the
    shipped pre-marker-store migration path). Code is BACKFILL-FIRST (L1.4):
    an unmarked non-empty code collection predates code markers and is assumed
    current -- record CODE_MODEL with NO clear; clear+re-embed fires only on a
    real recorded-vs-configured mismatch (reusing the docs None=>changed
    semantics here would mass re-embed the whole live code store on the first
    post-merge reindex). New markers are written AFTER the index passes by
    `_remodel_record`.

    BLOCKED clears (Phase-4 finding): a detected mismatch whose sources are
    unenumerable THIS run (a source dir transiently missing, or the enumerator
    returned nothing while the collection is non-empty) sets `docs_blocked` /
    `code_blocked` — `_remodel_record` must then NOT advance that collection's
    marker, else old-model vectors get a falsely 'current' marker, the
    interlock disarms, and no re-embed ever happens. Next run with sources
    present re-detects the mismatch and converges.
    """
    cleared = {"docs": 0, "code": 0, "docs_blocked": False, "code_blocked": False}
    if _model_changed(cache, db, project.docs_collection, DOCS_MODEL):
        # Clear ONLY the files index_docs will actually re-scan, so we never
        # invalidate a key the indexer won't re-embed (which would strand that
        # file's old-model vectors after the marker advances). F: index_docs
        # scans ALL docs_dirs ∪ in-repo .md under code_dirs (`_docs_sources`);
        # clear and re-embed share that one enumerator, so they cannot drift.
        docs_files = [f for _root, f in _docs_sources(project)]
        docs_roots_missing = any(not d.exists() for d in project.docs_dirs)
        if not docs_files or docs_roots_missing:
            cleared["docs_blocked"] = True
            _log(
                f"[{project.name}] docs model mismatch but sources are "
                f"unenumerable this run (missing root or empty scan); marker "
                f"NOT advanced — will retry next run."
            )
        else:
            manifest = Manifest(project.name)
            cleared["docs"] = _clear_docs_manifest_entries(manifest, docs_files)
            if cleared["docs"]:
                manifest.save()
                _log(
                    f"[{project.name}] Embedding-model change: cleared "
                    f"{cleared['docs']} docs manifest entries to force a re-embed."
                )

    code_marker = cache.get_collection_model(project.code_collection)
    if code_marker is None:
        if _collection_count(db, project.code_collection) > 0:
            cache.set_collection_model(project.code_collection, CODE_MODEL)
            msg = f"[{project.name}] backfilled code model marker = {CODE_MODEL}"
            if not CODE_MODEL.startswith("voyage"):
                # A non-voyage build backfilling an unmarked store is the
                # suspicious case: the L1.4 backfill assumes the store was
                # embedded under the era's voyage constants (design.md L1.4
                # precondition — at least one reindex under a voyage-constants
                # L1 build before any model-id change).
                msg += (
                    " -- WARNING: non-voyage model id stamped onto an unmarked "
                    "store; verify these vectors were really embedded under "
                    "this model, else delete the marker and reindex."
                )
            _log(msg)
    elif code_marker != CODE_MODEL and _collection_count(db, project.code_collection) > 0:
        code_files = _code_sources(project)
        code_roots_missing = any(
            not cd.path.exists() for cd in project.code_dirs
        )
        if not code_files or code_roots_missing:
            cleared["code_blocked"] = True
            _log(
                f"[{project.name}] code model mismatch but sources are "
                f"unenumerable this run (missing root or empty scan); marker "
                f"NOT advanced — will retry next run."
            )
        else:
            manifest = Manifest(project.name)
            cleared["code"] = _clear_code_manifest_entries(manifest, code_files)
            if cleared["code"]:
                manifest.save()
                _log(
                    f"[{project.name}] Code embedding-model change: cleared "
                    f"{cleared['code']} code manifest entries to force a re-embed."
                )
    return cleared


def _remodel_record(
    project: ProjectConfig, cache: EmbedCache, blocked: dict | None = None
) -> None:
    """run_index POST-pass: record what docs AND code are now embedded under.

    Set per project so the next run reads marker == configured model and the
    pre-pass is a no-op. Code markers (new in L1.4) arm the searcher's
    model-flip interlock for code collections. A collection whose mismatch-clear
    was BLOCKED this run (see `_remodel_clear`) keeps its old marker — advancing
    it without the re-embed would permanently disarm the interlock over
    old-model vectors.
    """
    blocked = blocked or {}
    if not blocked.get("docs_blocked"):
        cache.set_collection_model(project.docs_collection, DOCS_MODEL)
    if not blocked.get("code_blocked"):
        cache.set_collection_model(project.code_collection, CODE_MODEL)


def _prune_and_sweep_orphans(
    project: ProjectConfig,
    db: chromadb.ClientAPI,
) -> dict[str, int]:
    """Prune deleted files from the manifest AND delete their orphaned chunks.

    The prune-orphan fix (Inc 1.5a): `Manifest.prune()` forgot deleted files but
    never removed their vectors, so deleted files kept ranking against live
    content forever. Here we:
      - prune the manifest (now returns the removed keys);
      - sweep `-code` and `-docs` for chunks whose source-root-qualified
        `file_path` no longer resolves on disk (this also clears the already-
        leaked backlog, whose manifest entry is already gone).
    BM25 sidecars are updated via `_delete_ids_from_bm25` directly -- the prune
    path never reaches `_sync_bm25` (which only fires when `total_stored > 0`, so
    a delete-only run would otherwise leave stale BM25 rows). Crash window: each
    sweep deletes from chroma THEN BM25; an interruption in that gap leaves a
    BM25-only strand (its chroma twin already gone). The prune RETRY does NOT
    re-mirror it -- the retry re-derives orphans from chroma membership, and the
    twin is no longer a member. That strand self-heals via `_sync_bm25` step-3
    (drops BM25 rows absent from chroma) on the next content-adding run, not via
    the prune. Independent of Inc 4a's `valid_from`/`valid_to`.
    """
    stats = {"pruned_keys": 0, "code_orphans": 0, "docs_orphans": 0}

    manifest = Manifest(project.name)
    pruned = manifest.prune()
    stats["pruned_keys"] = len(pruned)

    def _col(name: str) -> chromadb.Collection | None:
        try:
            return db.get_collection(name)
        except Exception:
            return None

    # Code: {code_dir.name}/{rel} -> code_dir.path / rel on disk. The map is
    # liveness- and collision-guarded so a transiently-missing or basename-
    # colliding root can never be read as "all its files deleted" (see
    # _safe_sweep_root_map).
    code_map = _safe_sweep_root_map([cd.path for cd in project.code_dirs])
    if code_map:
        col = _col(project.code_collection)
        if col is not None:
            ids = _sweep_deleted_source_chunks(col, code_map)
            if ids:
                _delete_ids_from_bm25(project.name, "code", ids)
            stats["code_orphans"] = len(ids)

    # Docs: roots = docs_dirs ∪ code_dirs (in-repo .md), same guards. Unknown-root
    # chunks are left to _partition_docs_by_root (legacy bare-scheme migration).
    docs_map = _safe_sweep_root_map(
        [cd.path for cd in project.code_dirs] + list(project.docs_dirs)
    )
    if docs_map:
        col = _col(project.docs_collection)
        if col is not None:
            ids = _sweep_deleted_source_chunks(col, docs_map)
            if ids:
                _delete_ids_from_bm25(project.name, "docs", ids)
            stats["docs_orphans"] = len(ids)

    if pruned:
        manifest.save()

    return stats


def run_index(project_name: str | None = None) -> None:
    """Run incremental index for one or all projects."""
    VECS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMADB_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()
    if not config.projects:
        _log("No projects configured. Use 'vecs project add' to register one.")
        return

    if project_name is not None and project_name not in config.projects:
        available = ", ".join(sorted(config.projects.keys())) or "(none)"
        raise ValueError(
            f"Project '{project_name}' not found. Available projects: {available}"
        )

    # Migrate global manifest to per-project manifests (one-time)
    migrate_global_manifest(MANIFEST_PATH, MANIFESTS_DIR, config)

    provider = get_provider(config)
    db = get_chromadb_client()

    projects = (
        {project_name: config.projects[project_name]}
        if project_name
        else config.projects
    )

    run_start = time.monotonic()
    _log(f"Starting index... ({len(projects)} project(s): {', '.join(projects)})")
    total_code = 0
    total_docs = 0
    # C: one content-hash embedding cache shared across every project and source
    # this run, so unchanged chunks skip the Voyage call. Closed in finally.
    cache = EmbedCache(VECS_DIR / "embed_cache.db")
    try:
        for name, project in projects.items():
            proj_start = time.monotonic()
            _log(f"Project: {name}")
            # B2 PRE-pass: detect a docs embedding-model change and clear the
            # affected manifest entries so index_docs re-embeds under the new model.
            remodel = _remodel_clear(project, db, cache)
            total_code += index_code(project, provider, db, cache=cache)
            total_docs += index_docs(project, provider, db, cache=cache)
            # B2 POST-pass: record the model docs are now embedded under.
            _remodel_record(project, cache, blocked=remodel)
            _log(f"[{name}] project finished in {time.monotonic() - proj_start:.1f}s")
    finally:
        cache.close()

    # Prune manifest entries for deleted files AND delete their orphaned chunks
    # (the prune-orphan fix, Inc 1.5a): a deleted file's vectors must not keep
    # ranking against live content. Also clears the already-accumulated backlog.
    for name, project in projects.items():
        try:
            stats = _prune_and_sweep_orphans(project, db)
        except Exception as e:
            # Isolate per-project failure so one project's prune/save I/O error
            # can't skip the sweep for the rest of the run (code/docs self-heal
            # next run).
            _log(f"[{name}] prune+sweep failed: {e}")
            continue
        if any(stats.values()):
            _log(
                f"[{name}] prune+sweep: pruned {stats['pruned_keys']} manifest "
                f"entries; deleted {stats['code_orphans']} code + "
                f"{stats['docs_orphans']} docs orphan chunks."
            )

    duration = time.monotonic() - run_start
    _log(f"Done. Indexed {total_code} code chunks, {total_docs} doc chunks "
         f"in {duration:.1f}s.")
