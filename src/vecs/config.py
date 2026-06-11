from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

VECS_DIR = Path.home() / ".vecs"
CHROMADB_DIR = VECS_DIR / "chromadb"
MANIFEST_PATH = VECS_DIR / "manifest.json"
MANIFESTS_DIR = VECS_DIR / "manifests"
DEFAULT_CONFIG_PATH = VECS_DIR / "config.yaml"

# Embedding models
CODE_MODEL = "voyage-code-3"
# Inc 1-B: docs re-embed target is voyage-4 (current frontier). The voyage-3 ->
# voyage-4 migration is NOT a bare constant flip against stored voyage-3 vectors
# -- the run_index model-change trigger (indexer._remodel_clear) detects the
# change and re-embeds every docs chunk under voyage-4.
DOCS_MODEL = "voyage-4"
# Facts (prose-drift) embedding model. Pinned separately from DOCS_MODEL so a
# docs re-embed cannot silently restrand fact vectors into a different vector
# space. Facts are empty until Inc 2, so this can be set to the current frontier
# with no migration.
FACTS_MODEL = "voyage-4"

# Embedding output dimensions (for Voyage models: the model DEFAULT -- we never
# send an output_dimension override, so each model emits its default-width
# vector. For local Qwen entries the value is PRESCRIPTIVE -- see inline note).
# Recorded so the Inc 1-B in-place docs re-embed is provably dim-safe: voyage-4's
# default 1024 == voyage-3's 1024 == voyage-code-3's 1024, so re-embedded
# vectors overwrite existing chunk ids in the same Chroma collection with no
# recreate. Equal dim is NECESSARY (vectors fit) but NOT SUFFICIENT -- a
# different vector space still requires the real re-embed the trigger delivers.
EMBED_DIMS = {
    "voyage-3": 1024,
    "voyage-4": 1024,
    "voyage-code-3": 1024,
    # Local (Qwen3) model ids -- for these, the dim is PRESCRIPTIVE, not
    # descriptive: QwenLocalProvider truncates to this width (MRL). A dim
    # change REQUIRES a new model id -- the embed cache and collection_models
    # markers key on the id string alone, so editing a dim in place would
    # serve stale-width cached vectors with zero invalidation.
    "qwen3-embedding-4b@mrl1024": 1024,
    "qwen3-embedding-0.6b": 1024,
}

# Chunking defaults
CODE_CHUNK_LINES = 200
CODE_CHUNK_OVERLAP = 50


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
    docs_dirs: list[Path] = field(default_factory=list)
    prose_drift_enabled: bool = False

    @property
    def docs_dir(self) -> Path | None:
        """Legacy singular accessor: the first configured docs dir, or None.

        `docs_dirs` is the canonical store; this property keeps existing read
        sites (searcher, indexer) and the add_document auto-configure writes
        (cli, mcp_server) working unchanged. Multi-source docs land in Inc 1-F.
        """
        return self.docs_dirs[0] if self.docs_dirs else None

    @docs_dir.setter
    def docs_dir(self, value: Path | None) -> None:
        self.docs_dirs = [value] if value is not None else []

    @property
    def code_collection(self) -> str:
        return f"{self.name}-code"

    @property
    def docs_collection(self) -> str:
        return f"{self.name}-docs"

    @property
    def prose_facts_collection(self) -> str:
        return f"{self.name}-prose-facts"


@dataclass
class VecsConfig:
    """Top-level config holding all projects."""
    path: Path
    projects: dict[str, ProjectConfig] = field(default_factory=dict)
    # Which embedding provider serves all embed calls: "voyage" (hosted API,
    # default) or "qwen-local" (on-device, optional extra vecs[local]).
    # MUST round-trip through save() -- save() rewrites the whole file and runs
    # on every add_document auto-configure; an unmodeled field would be
    # silently stripped, reverting the fleet to voyage (design.md L1.2).
    embed_provider: str = "voyage"

    def add_project(
        self,
        name: str,
        code_dirs: list[CodeDir] | None = None,
        docs_dir: Path | None = None,
    ) -> None:
        self.projects[name] = ProjectConfig(
            name=name,
            code_dirs=code_dirs or [],
            docs_dirs=[docs_dir] if docs_dir else [],
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
        data: dict = {"embed_provider": self.embed_provider, "projects": {}}
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
            if p.docs_dirs:
                proj["docs_dirs"] = [str(d) for d in p.docs_dirs]
            data["projects"][name] = proj
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
    config.embed_provider = raw.get("embed_provider", "voyage")
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

        # Support both plural docs_dirs (list) and legacy singular docs_dir
        docs_dirs_raw = proj.get("docs_dirs", [])
        if not docs_dirs_raw and proj.get("docs_dir"):
            docs_dirs_raw = [proj["docs_dir"]]
        docs_dirs = [Path(d) for d in docs_dirs_raw]

        config.projects[name] = ProjectConfig(
            name=name,
            code_dirs=code_dirs,
            docs_dirs=docs_dirs,
            prose_drift_enabled=bool(proj.get("prose_drift_enabled", False)),
        )

    _config_cache[key] = (current_mtime, config)
    return config
