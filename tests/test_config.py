import yaml
from pathlib import Path
from vecs.config import load_config, ProjectConfig, CodeDir


def test_load_config_multi_code_dirs(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {
            "livly": {
                "code_dirs": [
                    {"path": "/tmp/client", "extensions": [".ts", ".tsx"]},
                    {"path": "/tmp/server", "extensions": [".cs"]},
                ],
            }
        }
    }))
    config = load_config(config_file)
    p = config.projects["livly"]
    assert len(p.code_dirs) == 2
    assert p.code_dirs[0].path == Path("/tmp/client")
    assert p.code_dirs[0].extensions == {".ts", ".tsx"}
    assert p.code_dirs[1].path == Path("/tmp/server")
    assert p.code_dirs[1].extensions == {".cs"}


def test_load_config_with_docs_dir(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {
            "proj": {
                "code_dirs": [{"path": "/tmp/code", "extensions": [".cs"]}],
                "docs_dir": "/tmp/docs",
            }
        }
    }))
    config = load_config(config_file)
    assert config.projects["proj"].docs_dir == Path("/tmp/docs")


def test_load_config_missing_file(tmp_path):
    config = load_config(tmp_path / "nonexistent.yaml")
    assert config.projects == {}


def test_collection_names():
    p = ProjectConfig(
        name="livly",
        code_dirs=[CodeDir(path=Path("/tmp"), extensions={".cs"})],
    )
    assert p.code_collection == "livly-code"
    assert p.docs_collection == "livly-docs"


def test_save_and_reload(tmp_path):
    config_file = tmp_path / "config.yaml"
    config = load_config(config_file)
    config.add_project(
        "test",
        code_dirs=[CodeDir(path=Path("/tmp/code"), extensions={".cs", ".ts"})],
        docs_dir=Path("/tmp/docs"),
    )
    config.save()
    reloaded = load_config(config_file)
    assert "test" in reloaded.projects
    p = reloaded.projects["test"]
    assert len(p.code_dirs) == 1
    assert p.code_dirs[0].path == Path("/tmp/code")
    assert p.code_dirs[0].extensions == {".cs", ".ts"}
    assert p.docs_dir == Path("/tmp/docs")


def test_remove_project(tmp_path):
    config_file = tmp_path / "config.yaml"
    config = load_config(config_file)
    config.add_project(
        "test",
        code_dirs=[CodeDir(path=Path("/tmp/code"), extensions={".cs"})],
    )
    config.save()
    config.remove_project("test")
    config.save()
    reloaded = load_config(config_file)
    assert "test" not in reloaded.projects


def test_include_dirs_on_code_dir(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {
            "proj": {
                "code_dirs": [{
                    "path": "/tmp/code",
                    "extensions": [".cs"],
                    "include_dirs": ["Scripts", "UI"],
                }],
            }
        }
    }))
    config = load_config(config_file)
    assert config.projects["proj"].code_dirs[0].include_dirs == ["Scripts", "UI"]


def test_exclude_dirs_default_empty():
    """CodeDir.exclude_dirs defaults to an empty list."""
    cd = CodeDir(path=Path("/tmp/code"), extensions={".cs"})
    assert cd.exclude_dirs == []


def test_exclude_dirs_on_code_dir(tmp_path):
    """exclude_dirs is parsed from YAML alongside include_dirs."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {
            "proj": {
                "code_dirs": [{
                    "path": "/tmp/code",
                    "extensions": [".cs"],
                    "exclude_dirs": ["Library", "Temp", "obj"],
                }],
            }
        }
    }))
    config = load_config(config_file)
    assert config.projects["proj"].code_dirs[0].exclude_dirs == ["Library", "Temp", "obj"]


def test_exclude_dirs_save_and_reload(tmp_path):
    """exclude_dirs survives save/reload roundtrip."""
    config_file = tmp_path / "config.yaml"
    config = load_config(config_file)
    config.add_project(
        "proj",
        code_dirs=[CodeDir(
            path=Path("/tmp/code"),
            extensions={".cs"},
            exclude_dirs=["Library", "Temp"],
        )],
    )
    config.save()
    reloaded = load_config(config_file)
    assert reloaded.projects["proj"].code_dirs[0].exclude_dirs == ["Library", "Temp"]


def test_find_project_by_path(tmp_path):
    config_file = tmp_path / "config.yaml"
    config = load_config(config_file)
    config.add_project(
        "livly",
        code_dirs=[
            CodeDir(path=Path("/repos/livly-client"), extensions={".ts"}),
            CodeDir(path=Path("/repos/livly-server"), extensions={".cs"}),
        ],
    )
    assert config.find_project_by_path(Path("/repos/livly-client/src")) == "livly"
    assert config.find_project_by_path(Path("/repos/livly-server")) == "livly"
    assert config.find_project_by_path(Path("/repos/other")) is None


def test_backward_compat_legacy_code_dir(tmp_path):
    """Old configs with single code_dir still load correctly."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {
            "old_project": {
                "code_dir": "/tmp/code",
                "extensions": [".cs", ".ts"],
                "include_dirs": ["Scripts"],
            }
        }
    }))
    config = load_config(config_file)
    p = config.projects["old_project"]
    assert len(p.code_dirs) == 1
    assert p.code_dirs[0].path == Path("/tmp/code")
    assert p.code_dirs[0].extensions == {".cs", ".ts"}
    assert p.code_dirs[0].include_dirs == ["Scripts"]


# --- docs_dirs multi-path coercion (Inc 1-pipeline, Phase-7 dry-run) ---
# ProjectConfig gains canonical docs_dirs: list[Path]; legacy singular docs_dir
# is coerced into the list (mirrors the sessions_dir -> sessions_dirs precedent).
# docs_dir survives as a get/set property so every downstream read site
# (searcher, indexer) and the add_document auto-configure writes (cli, mcp_server)
# behave identically.

def test_load_config_singular_docs_dir_coerced_to_list(tmp_path):
    """Legacy singular docs_dir loads as a single-element docs_dirs list."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {
            "proj": {
                "code_dirs": [{"path": "/tmp/code", "extensions": [".cs"]}],
                "docs_dir": "/tmp/docs",
            }
        }
    }))
    p = load_config(config_file).projects["proj"]
    assert p.docs_dirs == [Path("/tmp/docs")]
    # downstream back-compat: the singular accessor still resolves identically
    assert p.docs_dir == Path("/tmp/docs")


def test_load_config_multi_docs_dirs(tmp_path):
    """Multiple docs_dirs load as a list; singular accessor returns the first."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {
            "proj": {
                "code_dirs": [{"path": "/tmp/code", "extensions": [".cs"]}],
                "docs_dirs": ["/tmp/docs1", "/tmp/docs2"],
            }
        }
    }))
    p = load_config(config_file).projects["proj"]
    assert p.docs_dirs == [Path("/tmp/docs1"), Path("/tmp/docs2")]
    assert p.docs_dir == Path("/tmp/docs1")


def test_load_config_no_docs_dirs(tmp_path):
    """No docs config -> empty docs_dirs list and None singular accessor."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {
            "proj": {
                "code_dirs": [{"path": "/tmp/code", "extensions": [".cs"]}],
            }
        }
    }))
    p = load_config(config_file).projects["proj"]
    assert p.docs_dirs == []
    assert p.docs_dir is None


def test_docs_dir_setter_writes_through_to_docs_dirs():
    """Assigning docs_dir (the cli/mcp_server auto-configure path) updates docs_dirs."""
    p = ProjectConfig(name="proj")
    assert p.docs_dirs == []
    assert p.docs_dir is None
    p.docs_dir = Path("/tmp/auto/docs")
    assert p.docs_dirs == [Path("/tmp/auto/docs")]
    assert p.docs_dir == Path("/tmp/auto/docs")
    p.docs_dir = None
    assert p.docs_dirs == []


def test_save_and_reload_docs_dir_roundtrip(tmp_path):
    """A single docs_dir set via add_project survives save/load identically."""
    config_file = tmp_path / "config.yaml"
    config = load_config(config_file)
    config.add_project(
        "test",
        code_dirs=[CodeDir(path=Path("/tmp/code"), extensions={".cs"})],
        docs_dir=Path("/tmp/docs"),
    )
    config.save()
    p = load_config(config_file).projects["test"]
    assert p.docs_dirs == [Path("/tmp/docs")]
    assert p.docs_dir == Path("/tmp/docs")


def test_save_and_reload_multi_docs_dirs(tmp_path):
    """Multi docs_dirs round-trip through save/load with no data loss."""
    from vecs.config import VecsConfig, _clear_config_cache

    config_file = tmp_path / "config.yaml"
    config = VecsConfig(path=config_file)
    config.projects["test"] = ProjectConfig(
        name="test",
        code_dirs=[CodeDir(path=Path("/tmp/code"), extensions={".cs"})],
        docs_dirs=[Path("/tmp/d1"), Path("/tmp/d2")],
    )
    config.save()
    _clear_config_cache()
    p = load_config(config_file).projects["test"]
    assert p.docs_dirs == [Path("/tmp/d1"), Path("/tmp/d2")]


def test_save_empty_docs_dirs_omitted(tmp_path):
    """Empty docs_dirs writes neither the plural nor the legacy singular key."""
    config_file = tmp_path / "config.yaml"
    config = load_config(config_file)
    config.add_project(
        "test",
        code_dirs=[CodeDir(path=Path("/tmp/code"), extensions={".cs"})],
    )
    config.save()
    raw = yaml.safe_load(config_file.read_text())
    assert "docs_dirs" not in raw["projects"]["test"]
    assert "docs_dir" not in raw["projects"]["test"]


# --- Config caching tests (M1) ---

def test_config_cache_returns_same_object(tmp_path):
    """Calling load_config twice without changes returns cached object."""
    from vecs.config import load_config, _clear_config_cache

    _clear_config_cache()

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {
            "proj": {
                "code_dirs": [{"path": "/tmp/code", "extensions": [".cs"]}],
            }
        }
    }))
    c1 = load_config(config_file)
    c2 = load_config(config_file)
    assert c1 is c2


def test_config_cache_invalidates_on_mtime_change(tmp_path):
    """Modifying the config file causes reload on next call."""
    import os
    import time
    from vecs.config import load_config, _clear_config_cache

    _clear_config_cache()

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {
            "proj": {
                "code_dirs": [{"path": "/tmp/code", "extensions": [".cs"]}],
            }
        }
    }))
    c1 = load_config(config_file)
    assert "proj" in c1.projects

    # Modify the file with a new mtime
    time.sleep(0.05)
    config_file.write_text(yaml.dump({
        "projects": {
            "new_proj": {
                "code_dirs": [{"path": "/tmp/new", "extensions": [".ts"]}],
            }
        }
    }))
    # Force mtime difference on filesystems with 1s resolution
    mtime = os.path.getmtime(str(config_file))
    os.utime(str(config_file), (mtime + 1, mtime + 1))

    c2 = load_config(config_file)
    assert c2 is not c1
    assert "new_proj" in c2.projects
    assert "proj" not in c2.projects


def test_config_cache_different_paths_not_shared(tmp_path):
    """Different config file paths are cached independently."""
    from vecs.config import load_config, _clear_config_cache

    _clear_config_cache()

    f1 = tmp_path / "a.yaml"
    f2 = tmp_path / "b.yaml"
    f1.write_text(yaml.dump({"projects": {"a": {"code_dirs": [{"path": "/a", "extensions": [".cs"]}]}}}))
    f2.write_text(yaml.dump({"projects": {"b": {"code_dirs": [{"path": "/b", "extensions": [".cs"]}]}}}))

    c1 = load_config(f1)
    c2 = load_config(f2)
    assert "a" in c1.projects
    assert "b" in c2.projects
    assert c1 is not c2


def test_clear_config_cache(tmp_path):
    """_clear_config_cache forces a fresh reload."""
    from vecs.config import load_config, _clear_config_cache

    _clear_config_cache()

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {
            "proj": {
                "code_dirs": [{"path": "/tmp/code", "extensions": [".cs"]}],
            }
        }
    }))
    c1 = load_config(config_file)

    _clear_config_cache()

    c2 = load_config(config_file)
    assert c1 is not c2  # fresh object after cache clear


# --- Extension validation tests (L6) ---

def test_code_dir_requires_extensions():
    """CodeDir without extensions raises an error."""
    import pytest
    with pytest.raises(ValueError):
        CodeDir(path=Path("/tmp/code"), extensions=set())


def test_load_config_missing_extensions_raises(tmp_path):
    """Config with code_dir missing extensions raises an error."""
    import pytest
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {
            "proj": {
                "code_dirs": [{"path": "/tmp/code"}],
            }
        }
    }))
    with pytest.raises(ValueError, match="extensions"):
        load_config(config_file)


# --- Inc 1-B: voyage-4 re-embed target + dim safety -------------------------

def test_docs_model_is_voyage_4():
    """B1: docs re-embed target is voyage-4 (current frontier). The in-place
    migration is delivered by the run_index model-change trigger (B2), so
    flipping the constant is safe. Code stays on voyage-code-3 (no trigger)."""
    from vecs.config import DOCS_MODEL, CODE_MODEL
    assert DOCS_MODEL == "voyage-4"
    assert CODE_MODEL == "voyage-code-3"


def test_voyage4_dim_matches_voyage3_for_in_place_reembed():
    """B3: record voyage-4 dim vs voyage-3. Equal dim (both 1024) is NECESSARY
    so re-embedded vectors overwrite existing chunk ids in the same Chroma
    collection with no recreate -- but NOT SUFFICIENT: a different vector space
    still requires a real re-embed (delivered by B2)."""
    from vecs.config import EMBED_DIMS, DOCS_MODEL, CODE_MODEL
    assert EMBED_DIMS["voyage-4"] == EMBED_DIMS["voyage-3"] == 1024
    # Every configured model resolves to the same dim -> in-place overwrite safe.
    assert EMBED_DIMS[DOCS_MODEL] == EMBED_DIMS[CODE_MODEL] == 1024


class TestEmbedProvider:
    def test_default_is_voyage_when_absent(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("projects: {}\n")
        cfg = load_config(p)
        assert cfg.embed_provider == "voyage"

    def test_round_trips_through_save(self, tmp_path):
        from vecs.config import VecsConfig
        p = tmp_path / "config.yaml"
        cfg = VecsConfig(path=p)
        cfg.embed_provider = "qwen-local"
        cfg.save()
        loaded = load_config(p)
        assert loaded.embed_provider == "qwen-local"

    def test_save_after_load_preserves_provider(self, tmp_path):
        """The add_document auto-configure path (load -> mutate projects -> save)
        must NOT strip the provider field (design.md L1.2: save() rewrites the
        whole file)."""
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
