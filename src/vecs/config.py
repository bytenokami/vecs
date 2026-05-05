from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

VECS_DIR = Path.home() / ".vecs"
CHROMADB_DIR = VECS_DIR / "chromadb"
MANIFEST_PATH = VECS_DIR / "manifest.json"
MANIFESTS_DIR = VECS_DIR / "manifests"
DEFAULT_CONFIG_PATH = VECS_DIR / "config.yaml"

DEFAULT_CODEX_SESSIONS_ROOT = Path.home() / ".codex" / "sessions"
CODEX_ROUTING_PATH = MANIFESTS_DIR / "_codex_routing.json"

# Embedding models
CODE_MODEL = "voyage-code-3"
SESSIONS_MODEL = "voyage-3"
DOCS_MODEL = "voyage-3"

# Chunking defaults
CODE_CHUNK_LINES = 200
CODE_CHUNK_OVERLAP = 50
SESSION_CHUNK_MESSAGES = 10
SESSION_CHUNK_OVERLAP = 2


@dataclass
class CodeDir:
    """A single code directory with its own extensions and filters."""
    path: Path
    extensions: set[str] = field(default_factory=set)
    include_dirs: list[str] = field(default_factory=list)
    exclude_dirs: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.extensions:
            raise ValueError(f"CodeDir '{self.path}' requires at least one extension (e.g. '.py', '.ts')")


@dataclass
class ProjectConfig:
    """Configuration for a single project (may span multiple repos)."""
    name: str
    code_dirs: list[CodeDir] = field(default_factory=list)
    sessions_dirs: list[Path] = field(default_factory=list)
    docs_dir: Path | None = None
    # Explicit Codex routing cwds: any session whose session_meta.cwd resolves
    # under one of these is routed to this project. Bypasses bidirectional
    # containment matching against code_dirs.
    codex_cwds: list[Path] = field(default_factory=list)

    @property
    def code_collection(self) -> str:
        return f"{self.name}-code"

    @property
    def sessions_collection(self) -> str:
        return f"{self.name}-sessions"

    @property
    def docs_collection(self) -> str:
        return f"{self.name}-docs"


@dataclass
class VecsConfig:
    """Top-level config holding all projects."""
    path: Path
    projects: dict[str, ProjectConfig] = field(default_factory=dict)
    codex_sessions_root: Path = DEFAULT_CODEX_SESSIONS_ROOT
    codex_disabled: bool = False
    codex_ignore_cwds: list[Path] = field(default_factory=list)

    def add_project(
        self,
        name: str,
        code_dirs: list[CodeDir] | None = None,
        sessions_dirs: list[Path] | None = None,
        docs_dir: Path | None = None,
        codex_cwds: list[Path] | None = None,
        # Backward compat keyword -- callers using sessions_dir= still work
        sessions_dir: Path | None = None,
    ) -> None:
        # Handle legacy single sessions_dir kwarg
        resolved_dirs = sessions_dirs or []
        if not resolved_dirs and sessions_dir:
            resolved_dirs = [sessions_dir]
        self.projects[name] = ProjectConfig(
            name=name,
            code_dirs=code_dirs or [],
            sessions_dirs=resolved_dirs,
            docs_dir=docs_dir,
            codex_cwds=codex_cwds or [],
        )

    def remove_project(self, name: str) -> None:
        self.projects.pop(name, None)

    def find_project_by_path(self, target: Path) -> str | None:
        """Find which project owns a directory path."""
        resolved = target.resolve()
        for name, p in self.projects.items():
            for cd in p.code_dirs:
                try:
                    resolved.relative_to(cd.path.resolve())
                    return name
                except ValueError:
                    continue
        return None

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {"projects": {}}
        for name, p in self.projects.items():
            proj: dict = {}
            if p.code_dirs:
                proj["code_dirs"] = []
                for cd in p.code_dirs:
                    cd_dict: dict = {
                        "path": str(cd.path),
                        "extensions": sorted(cd.extensions),
                    }
                    if cd.include_dirs:
                        cd_dict["include_dirs"] = list(cd.include_dirs)
                    if cd.exclude_dirs:
                        cd_dict["exclude_dirs"] = list(cd.exclude_dirs)
                    proj["code_dirs"].append(cd_dict)
            if p.sessions_dirs:
                proj["sessions_dirs"] = [str(d) for d in p.sessions_dirs]
            if p.docs_dir:
                proj["docs_dir"] = str(p.docs_dir)
            if p.codex_cwds:
                proj["codex_cwds"] = [str(c) for c in p.codex_cwds]
            data["projects"][name] = proj
        # Top-level Codex settings: only emit non-default values to keep YAML clean.
        if self.codex_sessions_root != DEFAULT_CODEX_SESSIONS_ROOT:
            data["codex_sessions_root"] = str(self.codex_sessions_root)
        if self.codex_disabled:
            data["codex_disabled"] = True
        if self.codex_ignore_cwds:
            data["codex_ignore_cwds"] = [str(c) for c in self.codex_ignore_cwds]
        self.path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
        # Invalidate cache for this path so next load_config sees fresh data
        _config_cache.pop(str(self.path), None)


# Module-level config cache: {str(path): (mtime, VecsConfig)}
_config_cache: dict[str, tuple[float, VecsConfig]] = {}


def _clear_config_cache() -> None:
    """Clear the config cache. Used in tests and when config is saved."""
    _config_cache.clear()


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> VecsConfig:
    """Load config from YAML. Returns empty config if file missing.

    Results are cached by file path with mtime-based invalidation.
    """
    key = str(path)

    if path.exists():
        current_mtime = path.stat().st_mtime
        if key in _config_cache and _config_cache[key][0] == current_mtime:
            return _config_cache[key][1]
    else:
        # File doesn't exist -- return empty config (not cached)
        _config_cache.pop(key, None)
        return VecsConfig(path=path)

    config = VecsConfig(path=path)
    raw = yaml.safe_load(path.read_text()) or {}
    for name, proj in raw.get("projects", {}).items():
        code_dirs = []
        for cd_raw in proj.get("code_dirs", []):
            raw_extensions = cd_raw.get("extensions")
            if not raw_extensions:
                raise ValueError(
                    f"Project '{name}', code_dir '{cd_raw['path']}': "
                    f"'extensions' is required (e.g. ['.py', '.ts'])"
                )
            code_dirs.append(CodeDir(
                path=Path(cd_raw["path"]),
                extensions=set(raw_extensions),
                include_dirs=list(cd_raw.get("include_dirs", [])),
                exclude_dirs=list(cd_raw.get("exclude_dirs", [])),
            ))

        # Backward compat: migrate legacy single code_dir
        if not code_dirs and "code_dir" in proj:
            raw_extensions = proj.get("extensions")
            if not raw_extensions:
                raise ValueError(
                    f"Project '{name}', code_dir '{proj['code_dir']}': "
                    f"'extensions' is required (e.g. ['.py', '.ts'])"
                )
            code_dirs = [CodeDir(
                path=Path(proj["code_dir"]),
                extensions=set(raw_extensions),
                include_dirs=list(proj.get("include_dirs", [])),
                exclude_dirs=list(proj.get("exclude_dirs", [])),
            )]

        # Support both plural sessions_dirs (list) and legacy singular sessions_dir
        sessions_dirs_raw = proj.get("sessions_dirs", [])
        if not sessions_dirs_raw and proj.get("sessions_dir"):
            sessions_dirs_raw = [proj["sessions_dir"]]
        sessions_dirs = [Path(s) for s in sessions_dirs_raw]

        codex_cwds_raw = proj.get("codex_cwds", []) or []
        codex_cwds = [Path(c) for c in codex_cwds_raw]

        config.projects[name] = ProjectConfig(
            name=name,
            code_dirs=code_dirs,
            sessions_dirs=sessions_dirs,
            docs_dir=Path(proj["docs_dir"]) if proj.get("docs_dir") else None,
            codex_cwds=codex_cwds,
        )

    # Top-level Codex settings (all optional, sensible defaults).
    if raw.get("codex_sessions_root"):
        config.codex_sessions_root = Path(raw["codex_sessions_root"]).expanduser()
    config.codex_disabled = bool(raw.get("codex_disabled", False))
    # Env-var escape hatch: VECS_CODEX_DISABLED=1 forces codex indexing off
    # without editing the config file.
    if os.environ.get("VECS_CODEX_DISABLED", "").lower() in ("1", "true", "yes"):
        config.codex_disabled = True
    ignore_raw = raw.get("codex_ignore_cwds", []) or []
    config.codex_ignore_cwds = [Path(c) for c in ignore_raw]

    _config_cache[key] = (current_mtime, config)
    return config
