from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import chromadb
import voyageai
import voyageai.error

from vecs.ast_chunker import chunk_code_file_ast
from vecs.bm25_index import BM25Index
from vecs.chunkers import preprocess_session, chunk_session
from vecs.clients import get_voyage_client, get_chromadb_client
from vecs.codex_chunker import preprocess_codex_session
from vecs.codex_routing import CodexRoutingState, discover_codex_sessions
from vecs.config import (
    CHROMADB_DIR,
    CODE_CHUNK_LINES,
    CODE_CHUNK_OVERLAP,
    CODE_MODEL,
    CodeDir,
    DOCS_MODEL,
    MANIFESTS_DIR,
    MANIFEST_PATH,
    SESSION_CHUNK_MESSAGES,
    SESSION_CHUNK_OVERLAP,
    SESSIONS_MODEL,
    VECS_DIR,
    ProjectConfig,
    VecsConfig,
    load_config,
)
from vecs.doc_chunker import chunk_doc, extract_pdf_text


# Module-level set: payload types we logged-once-per-run as unrecognized.
# Reset at the top of run_index so each indexer invocation gets fresh telemetry.
_codex_unknown_payload_seen: set[str] = set()


MAX_BATCH_TOKENS = 80_000  # Voyage limit is 120K; char-based estimation is unreliable, so leave wide margin
MAX_BATCH_SIZE = 128


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

    def _session_identity_hash(self, file_path: Path, num_bytes: int = 1024) -> str:
        """Hash of first num_bytes of a file -- used as identity check.

        Uses a fixed number of bytes so appending to the file does not
        change the hash as long as the beginning is unchanged.
        """
        with open(file_path, "rb") as fh:
            head = fh.read(num_bytes)
        return hashlib.sha256(head).hexdigest()

    def get_session_info(self, file_path: Path) -> dict | None:
        """Get stored session tracking info, or None if not tracked."""
        key = f"session:{file_path}"
        entry = self.data.get(key)
        if not isinstance(entry, dict):
            return None
        return entry

    def mark_session_indexed(self, file_path: Path, byte_offset: int, chunk_count: int = 0) -> None:
        """Record session file identity, indexed byte offset, and chunk count.

        Stores the identity hash over min(1024, byte_offset) bytes so that
        appending to the file does not change the identity hash.
        """
        identity_bytes = min(1024, byte_offset)
        self.data[f"session:{file_path}"] = {
            "byte_offset": byte_offset,
            "identity_hash": self._session_identity_hash(file_path, num_bytes=identity_bytes),
            "identity_bytes": identity_bytes,
            "chunk_count": chunk_count,
        }

    def prune(self) -> int:
        """Remove entries for files that no longer exist on disk. Returns count removed."""
        stale = [
            key for key in self.data
            if not key.startswith("session:") and not Path(key).exists()
        ]
        for key in stale:
            del self.data[key]
        return len(stale)

    def prune_out_of_scope(self, in_scope: set[Path], roots: list[Path]) -> list[str]:
        """Remove code-file entries that are under any of `roots` but not in `in_scope`.

        Used after exclude_dirs filtering: a file may still exist on disk but be
        out of scope for indexing. Only entries beneath one of the supplied
        `roots` are considered -- code files for unrelated projects/code_dirs
        sharing this manifest must be left alone.

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
       code_dirs[].path, sessions_dir, or docs_dir
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
            # Check sessions_dirs
            for sd in proj.sessions_dirs:
                if file_path_str.startswith(str(sd)):
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
    vo: voyageai.Client,
    batcher: AdaptiveBatcher | None = None,
) -> list[str]:
    """Embed and store chunks. Returns list of successfully stored chunk IDs.

    Uses AdaptiveBatcher for token estimation when provided. Calibrates
    from Voyage API response usage data.
    """
    if not chunks:
        return []

    if batcher is None:
        batcher = AdaptiveBatcher()

    succeeded_ids: list[str] = []
    for batch in _make_batches(chunks, batcher):
        texts = [c["text"] for c in batch]
        batch_chars = sum(len(t) for t in texts)

        for attempt in range(5):
            try:
                result = vo.embed(texts, model=model, input_type="document")
                break
            except Exception as e:
                error_msg = str(e).lower()
                is_transient = (
                    isinstance(e, (
                        voyageai.error.Timeout,
                        voyageai.error.APIConnectionError,
                        voyageai.error.RateLimitError,
                        voyageai.error.ServiceUnavailableError,
                        voyageai.error.ServerError,
                        voyageai.error.TryAgain,
                        voyageai.error.APIError,
                    ))
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

        # H2: calibrate from successful response
        if hasattr(result, "usage") and hasattr(result.usage, "total_tokens"):
            batcher.calibrate(batch_chars, result.usage.total_tokens)

        ids = [c["id"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        collection.upsert(
            ids=ids,
            embeddings=result.embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        succeeded_ids.extend(ids)
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
    vo: voyageai.Client,
    manifest: Manifest,
    chunk_to_file: dict[str, Path],
    file_expected_count: dict[Path, int],
    file_cleanup: dict[Path, tuple[str, str, set[str]]],
    files_to_process: list[Path],
    file_hashes: dict[Path, str],
) -> int:
    """Shared indexing pipeline: embed, track success, cleanup orphans, save manifest.

    Args:
        chunks: All chunks to embed (batching handled internally).
        collection: ChromaDB collection to upsert into.
        model: Voyage AI model name.
        vo: Voyage AI client.
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

    succeeded_ids = _embed_and_store(chunks, collection, model, vo)

    fully_succeeded = _track_embed_success(
        succeeded_ids, chunk_to_file, file_expected_count, file_cleanup, collection,
    )
    for f in files_to_process:
        if f in fully_succeeded:
            manifest.mark_indexed(f, file_hashes[f])

    manifest.save()
    return len(succeeded_ids)


def index_code(project: ProjectConfig, vo: voyageai.Client, db: chromadb.ClientAPI) -> int:
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

    # Track in-scope files per code_dir so we can compute proper chromadb
    # rel_paths when pruning out-of-scope manifest entries.
    in_scope_by_root: list[tuple[Path, set[Path]]] = []

    for code_dir in project.code_dirs:
        if not code_dir.path.exists():
            _log(f"[{project.name}] {code_dir.path}: directory not found, skipping.")
            continue

        if code_dir.include_dirs:
            files: list[Path] = []
            for subdir in code_dir.include_dirs:
                d = code_dir.path / subdir
                if d.exists():
                    files.extend(
                        f for f in d.rglob("*")
                        if f.suffix in code_dir.extensions and f.is_file()
                    )
        else:
            files = [
                f for f in code_dir.path.rglob("*")
                if f.suffix in code_dir.extensions and f.is_file()
            ]

        # exclude_dirs: drop files whose path is under any excluded subdir.
        # Excludes apply on top of includes -- exclude wins on overlap.
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

        in_scope_by_root.append((code_dir.path, set(files)))

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

        for f in to_index:
            content = f.read_text(errors="replace")
            dir_prefix = code_dir.path.name
            rel_path = f"{dir_prefix}/{str(f.relative_to(code_dir.path))}"
            chunks = chunk_code_file_ast(
                content, rel_path, chunk_lines=CODE_CHUNK_LINES, overlap=CODE_CHUNK_OVERLAP
            )
            chunk_ids_for_file: set[str] = set()
            for c in chunks:
                c["id"] = _make_chunk_id(f"code:{rel_path}", c["metadata"]["chunk_index"])
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
    for root, in_scope in in_scope_by_root:
        stale_keys = manifest.prune_out_of_scope(in_scope, [root])
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

    if not all_chunks:
        return 0

    total_stored = _index_collection(
        chunks=all_chunks,
        collection=collection,
        model=CODE_MODEL,
        vo=vo,
        manifest=manifest,
        chunk_to_file=chunk_to_file,
        file_expected_count=file_expected_count,
        file_cleanup=file_cleanup,
        files_to_process=files_to_process,
        file_hashes=file_hashes,
    )

    if total_stored > 0:
        _sync_bm25(collection, project.name, "code")

    return total_stored


def _get_session_new_content(
    file_path: Path,
    manifest: Manifest,
) -> tuple[str, int, bool]:
    """Determine what content to index for a session file.

    Returns:
        (content_to_index, new_byte_offset, is_full_reindex)
    """
    file_size = file_path.stat().st_size
    info = manifest.get_session_info(file_path)

    if info is None:
        # Never indexed -- full read
        content = file_path.read_text(errors="replace")
        return content, file_size, True

    identity_bytes = info.get("identity_bytes", 1024)
    current_identity = manifest._session_identity_hash(file_path, num_bytes=identity_bytes)
    if current_identity != info["identity_hash"]:
        # File rewritten/compacted -- full re-index
        content = file_path.read_text(errors="replace")
        return content, file_size, True

    stored_offset = info["byte_offset"]
    if file_size <= stored_offset:
        # File hasn't grown (or shrunk) -- nothing new
        return "", stored_offset, False

    # Append-only growth -- read only new bytes
    with open(file_path, "rb") as fh:
        fh.seek(stored_offset)
        new_bytes = fh.read()
    content = new_bytes.decode("utf-8", errors="replace")
    return content, file_size, False


def _index_session_files(
    project: ProjectConfig,
    files: list[Path],
    parser_fn,
    agent_tag: str,
    vo: voyageai.Client,
    db: chromadb.ClientAPI,
    *,
    log_label: str,
    manifest: Manifest | None = None,
) -> int:
    """Shared session indexing pipeline.

    Both the Claude Code (`index_sessions`) and Codex (`index_codex_sessions`)
    code paths call this with their respective parser_fn and agent_tag. The
    embedding model, BM25 sidecar suffix, and chunking parameters are constant
    across agents; only parsing differs.

    Args:
        project: Target project. Output goes to project.sessions_collection.
        files: Session files to consider (already routed and filtered).
        parser_fn: Callable[[str], list[dict]] mapping raw JSONL to messages
            in the shared `{role, text, timestamp}` shape.
        agent_tag: "claude_code" | "codex". Stamped onto every chunk's
            metadata so search results can distinguish sources.
        vo: Voyage client.
        db: Chromadb client.
        log_label: Human-readable label for log lines.
        manifest: Optional pre-built Manifest. Pass when caller already opened
            one to avoid double-loading.

    Returns:
        Number of successfully embedded+stored chunks.
    """
    if not files:
        return 0

    if manifest is None:
        manifest = Manifest(project.name)
    collection = db.get_or_create_collection(project.sessions_collection)

    all_chunks: list[dict] = []
    chunk_to_file: dict[str, Path] = {}
    file_expected_count: dict[Path, int] = {}
    file_cleanup: dict[Path, tuple[str, str, set[str]]] = {}
    # Per-file: (new_byte_offset, total_chunk_count, is_full_reindex)
    file_session_meta: dict[Path, tuple[int, int, bool]] = {}

    indexed_count = 0
    for f in files:
        content, new_offset, is_full = _get_session_new_content(f, manifest)
        if not content:
            continue

        indexed_count += 1
        session_id = f.stem

        messages = parser_fn(content)
        if not messages:
            # Mark indexed anyway so byte_offset advances; otherwise we re-read
            # the same uninteresting bytes every run.
            manifest.mark_session_indexed(f, byte_offset=new_offset)
            continue

        chunk_index_offset = 0
        if not is_full:
            info = manifest.get_session_info(f)
            if info and "chunk_count" in info:
                chunk_index_offset = info["chunk_count"]

        chunks = chunk_session(
            messages, session_id, SESSION_CHUNK_MESSAGES, overlap=SESSION_CHUNK_OVERLAP
        )
        chunk_ids_for_file: set[str] = set()
        for c in chunks:
            c["metadata"]["chunk_index"] += chunk_index_offset
            c["metadata"]["agent"] = agent_tag
            c["id"] = _make_chunk_id(f"session:{session_id}", c["metadata"]["chunk_index"])
            chunk_to_file[c["id"]] = f
            chunk_ids_for_file.add(c["id"])

        file_expected_count[f] = len(chunks)
        total_chunk_count = chunk_index_offset + len(chunks)
        file_session_meta[f] = (new_offset, total_chunk_count, is_full)

        if is_full:
            file_cleanup[f] = ("session_id", session_id, chunk_ids_for_file)

        all_chunks.extend(chunks)

    if indexed_count == 0:
        _log(f"[{project.name}] {log_label}: nothing new to index.")
    else:
        _log(f"[{project.name}] {log_label}: {indexed_count} files to index ({len(files)} total)")

    if not all_chunks:
        manifest.save()
        return 0

    succeeded_ids = _embed_and_store(all_chunks, collection, SESSIONS_MODEL, vo)

    fully_succeeded = _track_embed_success(
        succeeded_ids, chunk_to_file, file_expected_count, file_cleanup, collection,
    )
    for f in fully_succeeded:
        if f in file_session_meta:
            new_offset, total_chunk_count, _ = file_session_meta[f]
            manifest.mark_session_indexed(f, byte_offset=new_offset, chunk_count=total_chunk_count)

    manifest.save()

    if len(succeeded_ids) > 0:
        _sync_bm25(collection, project.name, "sessions")

    return len(succeeded_ids)


def index_sessions(project: ProjectConfig, vo: voyageai.Client, db: chromadb.ClientAPI) -> int:
    """Index Claude Code session transcripts for a project.

    Thin wrapper around `_index_session_files` for backward-compat.
    """
    if not project.sessions_dirs:
        return 0

    manifest = Manifest(project.name)
    total = 0
    for sessions_dir in project.sessions_dirs:
        if not sessions_dir.exists():
            _log(f"[{project.name}] Sessions dir not found: {sessions_dir}")
            continue
        files = sorted(sessions_dir.glob("*.jsonl"))
        total += _index_session_files(
            project,
            files,
            preprocess_session,
            agent_tag="claude_code",
            vo=vo,
            db=db,
            log_label=f"Sessions ({sessions_dir})",
            manifest=manifest,
        )
    return total


def _make_codex_parser(unknown_seen: set[str]):
    """Bind the per-run `unknown_payload_seen` set onto preprocess_codex_session.

    The shared `_index_session_files` expects parser_fn(content) -> messages.
    Codex parser also accepts an optional set arg; we close over the run-level
    set so unknown payload types are deduplicated across all files.
    """
    def parse(content: str) -> list[dict]:
        return preprocess_codex_session(content, unknown_payload_seen=unknown_seen)
    return parse


def index_codex_sessions(
    project: ProjectConfig,
    files: list[Path],
    vo: voyageai.Client,
    db: chromadb.ClientAPI,
) -> int:
    """Index Codex CLI session transcripts already routed to this project."""
    if not files:
        return 0
    parser = _make_codex_parser(_codex_unknown_payload_seen)
    return _index_session_files(
        project,
        files,
        parser,
        agent_tag="codex",
        vo=vo,
        db=db,
        log_label="Codex sessions",
    )


def index_docs(project: ProjectConfig, vo: voyageai.Client, db: chromadb.ClientAPI) -> int:
    """Index documentation files for a project. Returns count of new chunks."""
    if not project.docs_dir or not project.docs_dir.exists():
        return 0

    manifest = Manifest(project.name)
    collection = db.get_or_create_collection(project.docs_collection)

    files = [
        f for f in project.docs_dir.rglob("*")
        if f.suffix in {".md", ".txt", ".pdf"} and f.is_file()
    ]

    to_index: list[Path] = []
    file_hashes: dict[Path, str] = {}
    for f in files:
        needs, fhash = manifest.needs_indexing(f)
        if needs:
            to_index.append(f)
            file_hashes[f] = fhash

    if not to_index:
        _log(f"[{project.name}] Docs: nothing new to index.")
        return 0

    _log(f"[{project.name}] Docs: {len(to_index)} files to index ({len(files)} total)")

    all_chunks: list[dict] = []
    chunk_to_file: dict[str, Path] = {}
    file_expected_count: dict[Path, int] = {}
    file_cleanup: dict[Path, tuple[str, str, set[str]]] = {}

    for f in to_index:
        rel_path = str(f.relative_to(project.docs_dir))

        if f.suffix == ".pdf":
            content = extract_pdf_text(str(f))
        else:
            content = f.read_text(errors="replace")

        chunks = chunk_doc(content, rel_path)
        chunk_ids_for_file: set[str] = set()
        for c in chunks:
            c["id"] = _make_chunk_id(f"docs:{rel_path}", c["metadata"]["chunk_index"])
            chunk_to_file[c["id"]] = f
            chunk_ids_for_file.add(c["id"])
        file_expected_count[f] = len(chunks)
        file_cleanup[f] = ("file_path", rel_path, chunk_ids_for_file)
        all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    total_stored = _index_collection(
        chunks=all_chunks,
        collection=collection,
        model=DOCS_MODEL,
        vo=vo,
        manifest=manifest,
        chunk_to_file=chunk_to_file,
        file_expected_count=file_expected_count,
        file_cleanup=file_cleanup,
        files_to_process=to_index,
        file_hashes=file_hashes,
    )

    if total_stored > 0:
        _sync_bm25(collection, project.name, "docs")

    return total_stored


def purge_session_files_from_project(
    project_name: str,
    file_paths: list[Path],
    session_ids: list[str],
    db: chromadb.ClientAPI | None = None,
) -> dict:
    """Drop manifest entries + sweep chunks for a set of session files in a project.

    Used by `codex_assign` / `codex_ignore` to invalidate previous routing so a
    subsequent `vecs index` re-emits sessions into the correct project (or skips
    them, in the ignore case).

    Args:
        project_name: Project whose `sessions_collection` and manifest are touched.
        file_paths: Absolute paths whose `session:{path}` manifest entries are
            dropped. Empty list is a no-op.
        session_ids: Session ids whose chunks are deleted from the project's
            sessions collection (chroma) AND BM25 sidecar. Empty list skips the
            chunk sweep but still drops manifest entries.
        db: Chromadb client. If None, opened lazily.

    Returns:
        {
            "manifest_entries_dropped": int,
            "chunks_deleted": int,
            "session_ids_swept": int,
        }
    """
    result = {"manifest_entries_dropped": 0, "chunks_deleted": 0, "session_ids_swept": 0}

    # 1. Drop manifest entries (no chromadb needed if list empty).
    if file_paths:
        manifest = Manifest(project_name)
        for fp in file_paths:
            key = f"session:{fp}"
            if key in manifest.data:
                del manifest.data[key]
                result["manifest_entries_dropped"] += 1
        if result["manifest_entries_dropped"]:
            manifest.save()

    if not session_ids:
        return result

    # 2. Sweep chunks from chroma collection by session_id metadata.
    if db is None:
        db = get_chromadb_client()
    config = load_config()
    if project_name not in config.projects:
        return result
    project = config.projects[project_name]
    try:
        collection = db.get_collection(project.sessions_collection)
    except Exception:
        return result

    deleted = 0
    for sid in session_ids:
        try:
            existing = collection.get(where={"session_id": sid})
            ids = existing.get("ids") or []
            if ids:
                _paginated_delete(collection, ids)
                deleted += len(ids)
                result["session_ids_swept"] += 1
        except Exception as e:
            _log(f"  Warning: chunk sweep failed for session_id={sid}: {e}")
    result["chunks_deleted"] = deleted

    if deleted:
        # Keep BM25 sidecar in sync with the now-shorter chroma collection.
        _sync_bm25(collection, project_name, "sessions")

    return result


def index_single_doc(project_name: str, file_path: Path) -> int:
    """Index a single doc file immediately. Returns chunk count."""
    config = load_config()
    if project_name not in config.projects:
        raise ValueError(f"Project '{project_name}' not found.")

    project = config.projects[project_name]
    if not project.docs_dir:
        raise ValueError(f"Project '{project_name}' has no docs_dir configured.")

    vo = get_voyage_client()
    db = get_chromadb_client()
    collection = db.get_or_create_collection(project.docs_collection)
    manifest = Manifest(project_name)

    # H6: compute hash once
    _, file_hash = manifest.needs_indexing(file_path)

    rel_path = str(file_path.relative_to(project.docs_dir))

    if file_path.suffix == ".pdf":
        content = extract_pdf_text(str(file_path))
    else:
        content = file_path.read_text(errors="replace")

    chunks = chunk_doc(content, rel_path)
    chunk_to_file: dict[str, Path] = {}
    chunk_ids: set[str] = set()
    for c in chunks:
        c["id"] = _make_chunk_id(f"docs:{rel_path}", c["metadata"]["chunk_index"])
        chunk_to_file[c["id"]] = file_path
        chunk_ids.add(c["id"])

    total_stored = _index_collection(
        chunks=chunks,
        collection=collection,
        model=DOCS_MODEL,
        vo=vo,
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

    status: dict = {"projects": {}, "total_code_chunks": 0, "total_session_chunks": 0, "total_docs_chunks": 0}

    for name, p in projects.items():
        code_count = 0
        session_count = 0
        docs_count = 0
        try:
            col = db.get_collection(p.code_collection)
            code_count = col.count()
        except Exception:
            pass
        try:
            col = db.get_collection(p.sessions_collection)
            session_count = col.count()
        except Exception:
            pass
        try:
            col = db.get_collection(p.docs_collection)
            docs_count = col.count()
        except Exception:
            pass
        status["projects"][name] = {
            "code_chunks": code_count,
            "session_chunks": session_count,
            "docs_chunks": docs_count,
        }
        status["total_code_chunks"] += code_count
        status["total_session_chunks"] += session_count
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

    vo = get_voyage_client()
    db = get_chromadb_client()

    projects = (
        {project_name: config.projects[project_name]}
        if project_name
        else config.projects
    )

    # Reset run-level Codex telemetry.
    _codex_unknown_payload_seen.clear()

    # Codex discovery is global (one walk of codex_sessions_root) so we route
    # ALL codex files here, then hand each project its own slice. Cheaper than
    # rerouting per project, and orphan tracking lives in one place.
    codex_routing: dict[str, list[Path]] = {}
    codex_state: CodexRoutingState | None = None
    if not config.codex_disabled:
        try:
            codex_routing, codex_state = discover_codex_sessions(config)
        except Exception as e:
            _log(f"  Warning: codex discovery failed: {e}")
            codex_routing, codex_state = {}, None

    run_start = time.monotonic()
    _log(f"Starting index... ({len(projects)} project(s): {', '.join(projects)})")
    total_code = 0
    total_sessions = 0
    total_docs = 0
    for name, project in projects.items():
        proj_start = time.monotonic()
        _log(f"Project: {name}")
        total_code += index_code(project, vo, db)
        total_sessions += index_sessions(project, vo, db)
        codex_files = codex_routing.get(name, [])
        if codex_files:
            total_sessions += index_codex_sessions(project, codex_files, vo, db)
        total_docs += index_docs(project, vo, db)
        _log(f"[{name}] project finished in {time.monotonic() - proj_start:.1f}s")

    if _codex_unknown_payload_seen:
        _log(
            "  Codex parser saw unknown payload types this run: "
            + ", ".join(sorted(_codex_unknown_payload_seen))
        )

    if codex_state is not None and codex_state.orphans:
        _log(
            f"  Codex orphans: {codex_state.total_orphan_sessions()} sessions across "
            f"{len(codex_state.orphans)} cwd(s). Triage with `vecs codex orphans` "
            f"or the codex_orphans MCP tool."
        )

    # Prune manifest entries for deleted files
    for proj_name in projects:
        manifest = Manifest(proj_name)
        pruned = manifest.prune()
        if pruned > 0:
            manifest.save()
            _log(f"Pruned {pruned} stale manifest entries for {proj_name}.")

    duration = time.monotonic() - run_start
    _log(f"Done. Indexed {total_code} code chunks, {total_sessions} session chunks, "
         f"{total_docs} doc chunks in {duration:.1f}s.")
