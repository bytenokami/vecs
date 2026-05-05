"""Codex session discovery, cwd routing, and orphan state.

Single entry point for the indexer is `discover_codex_sessions(config)`,
which returns a `(routing, state)` pair:
  - `routing`: dict[project_name, list[Path]] of session files routed to each
    configured project.
  - `state`: `CodexRoutingState` instance holding the current orphan map
    (cwd -> count + first/last-seen) plus a (path -> mtime, cwd, session_id)
    cache that lets subsequent runs skip parsing line 1 of unchanged files.

Routing precedence:
  1. cwd in `config.codex_ignore_cwds` -> drop (silent).
  2. cwd resolves under any `project.codex_cwds[]` entry -> that project.
  3. Bidirectional containment vs `project.code_dirs[].path` (resolved):
     - if cwd is under a code_dir, candidate (depth-scored).
     - if a code_dir is under the cwd (cwd is ancestor), candidate (ancestor).
  4. Multiple ancestor matches across distinct projects -> orphan (ambiguous).
  5. Otherwise: longest-prefix wins among non-ancestor matches.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from vecs.codex_chunker import extract_session_meta
from vecs.config import (
    CODEX_ROUTING_PATH,
    VecsConfig,
)


CODEX_FILE_GLOB = "rollout-*.jsonl"


@dataclass
class CodexRoutingState:
    """Persisted Codex routing metadata.

    Lives at `~/.vecs/manifests/_codex_routing.json` and shares the manifests
    dir's `fcntl.flock` infrastructure (locked on save).
    """
    path: Path = CODEX_ROUTING_PATH
    version: int = 1
    updated_at: str = ""
    # cwd_str -> {sessions, first_seen, last_seen}
    orphans: dict[str, dict] = field(default_factory=dict)
    # session_file_path_str -> {mtime, cwd, session_id}
    cwd_cache: dict[str, dict] = field(default_factory=dict)
    # Per-day banner dedupe so MCP doesn't repeat itself.
    last_banner_day: str = ""

    @classmethod
    def load(cls, path: Path = CODEX_ROUTING_PATH) -> "CodexRoutingState":
        if not path.exists():
            return cls(path=path)
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return cls(path=path)
        return cls(
            path=path,
            version=int(data.get("version", 1)),
            updated_at=str(data.get("updated_at", "")),
            orphans=dict(data.get("orphans", {}) or {}),
            cwd_cache=dict(data.get("cwd_cache", {}) or {}),
            last_banner_day=str(data.get("last_banner_day", "")),
        )

    def prune_dead_cache(self) -> int:
        """Drop cwd_cache entries whose underlying session file no longer exists.

        Returns the number of entries pruned.
        """
        dead = [k for k in self.cwd_cache if not Path(k).exists()]
        for k in dead:
            del self.cwd_cache[k]
        return len(dead)

    def get_or_load_meta(self, file_path: Path) -> dict | None:
        """Return cached (or freshly-parsed) session_meta dict for a file.

        Cache key: absolute path. Cache validity: file mtime unchanged.
        On miss / mtime change, reads line 1 only via extract_session_meta.
        Returns dict with at least `cwd` and `session_id` on success.
        """
        try:
            stat = file_path.stat()
        except OSError:
            return None
        key = str(file_path)
        cached = self.cwd_cache.get(key)
        if cached and abs(cached.get("mtime", 0.0) - stat.st_mtime) < 1e-6:
            return {
                "cwd": cached.get("cwd", ""),
                "id": cached.get("session_id", ""),
            }
        # Parse line 1. session_meta lines can be large (Codex embeds the full
        # base_instructions there, often ~14KB). Read by-line so we always
        # have a complete JSON record regardless of size.
        try:
            with open(file_path, "rb") as fh:
                first_line = fh.readline()
        except OSError:
            return None
        try:
            text = first_line.decode("utf-8", errors="replace")
        except Exception:
            return None
        meta = extract_session_meta(text)
        if not meta:
            return None
        cwd = str(meta.get("cwd", "") or "")
        session_id = str(meta.get("id", "") or "")
        self.cwd_cache[key] = {
            "mtime": stat.st_mtime,
            "cwd": cwd,
            "session_id": session_id,
        }
        return {"cwd": cwd, "id": session_id}

    def record_orphan(self, cwd: str, when_iso: str) -> None:
        entry = self.orphans.setdefault(
            cwd, {"sessions": 0, "first_seen": when_iso, "last_seen": when_iso}
        )
        entry["sessions"] = int(entry.get("sessions", 0)) + 1
        entry["last_seen"] = when_iso

    def reset_orphans(self) -> None:
        self.orphans = {}

    def total_orphan_sessions(self) -> int:
        return sum(int(v.get("sessions", 0)) for v in self.orphans.values())

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        data = {
            "version": self.version,
            "updated_at": self.updated_at,
            "orphans": self.orphans,
            "cwd_cache": self.cwd_cache,
            "last_banner_day": self.last_banner_day,
        }
        # Atomic write under flock so concurrent indexers and MCP tool calls
        # don't tear the file.
        import tempfile
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        try:
            try:
                import fcntl
                # Lock file lives next to the state file so callers passing a
                # custom path (tests, alt locations) get consistent behavior.
                lock_path = self.path.parent / f".{self.path.name}.lock"
                lock_fd = open(lock_path, "w")
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                    os.replace(tmp_path, str(self.path))
                finally:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
            except ImportError:
                os.replace(tmp_path, str(self.path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def _safe_resolve(p: Path) -> Path | None:
    """Resolve a path without raising on missing entries / loops."""
    try:
        return p.expanduser().resolve()
    except (OSError, RuntimeError):
        return None


def _is_under(child: Path, parent: Path) -> bool:
    """Return True if `child` is `parent` or under `parent` (resolved)."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def route_cwd(cwd_str: str, config: VecsConfig) -> str | None:
    """Decide which project owns a session captured at `cwd_str`.

    Returns the project name, or None if the session is an orphan / ignored.
    """
    if not cwd_str:
        return None
    cwd = _safe_resolve(Path(cwd_str))
    if cwd is None:
        return None

    # Ignore list (silent drop).
    for ignore in config.codex_ignore_cwds:
        i_resolved = _safe_resolve(ignore)
        if i_resolved is None:
            continue
        if cwd == i_resolved or _is_under(cwd, i_resolved):
            return None

    # Explicit per-project override wins outright.
    for proj in config.projects.values():
        for explicit in proj.codex_cwds:
            e_resolved = _safe_resolve(explicit)
            if e_resolved is None:
                continue
            if cwd == e_resolved or _is_under(cwd, e_resolved):
                return proj.name

    # Bidirectional containment vs code_dirs.
    descendant_matches: list[tuple[int, str]] = []  # (depth, project)
    ancestor_matches: dict[str, set[str]] = {}  # project -> set of code_dir paths
    for proj in config.projects.values():
        for cd in proj.code_dirs:
            cd_resolved = _safe_resolve(cd.path)
            if cd_resolved is None:
                continue
            if _is_under(cwd, cd_resolved):
                descendant_matches.append((len(cd_resolved.parts), proj.name))
            elif _is_under(cd_resolved, cwd):
                ancestor_matches.setdefault(proj.name, set()).add(str(cd_resolved))

    if descendant_matches:
        descendant_matches.sort(reverse=True)
        return descendant_matches[0][1]

    if ancestor_matches:
        if len(ancestor_matches) > 1:
            # cwd is ancestor of code_dirs from multiple projects -> ambiguous.
            return None
        # Single project: cwd is an ancestor of one of its code_dirs. Accept.
        return next(iter(ancestor_matches))

    return None


def discover_codex_sessions(
    config: VecsConfig,
    state: CodexRoutingState | None = None,
) -> tuple[dict[str, list[Path]], CodexRoutingState]:
    """Walk codex_sessions_root, route each session to a project (or orphan).

    Returns (routing, state) where:
      - routing: dict[project_name, list[Path]] of session files per project,
        sorted by path so subsequent indexing is deterministic.
      - state: the CodexRoutingState used (loaded fresh if None was passed).
        State has been mutated to reflect this run's orphan census.
    """
    if state is None:
        state = CodexRoutingState.load()

    state.reset_orphans()
    state.prune_dead_cache()

    routing: dict[str, list[Path]] = {}
    if config.codex_disabled:
        state.save()
        return routing, state

    root = config.codex_sessions_root
    if not root.exists():
        state.save()
        return routing, state

    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    for path in sorted(root.rglob(CODEX_FILE_GLOB)):
        if not path.is_file():
            continue
        meta = state.get_or_load_meta(path)
        if not meta:
            continue
        cwd = meta.get("cwd", "")
        project_name = route_cwd(cwd, config)
        if project_name is None:
            if cwd:
                state.record_orphan(cwd, now_iso)
            continue
        routing.setdefault(project_name, []).append(path)

    for files in routing.values():
        files.sort()

    state.save()
    return routing, state
