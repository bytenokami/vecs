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


def test_load_config_with_sessions(tmp_path):
    """Legacy singular sessions_dir loads as single-element list."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {
            "proj": {
                "code_dirs": [{"path": "/tmp/code", "extensions": [".cs"]}],
                "sessions_dir": "/tmp/sessions",
            }
        }
    }))
    config = load_config(config_file)
    assert config.projects["proj"].sessions_dirs == [Path("/tmp/sessions")]


def test_load_config_missing_file(tmp_path):
    config = load_config(tmp_path / "nonexistent.yaml")
    assert config.projects == {}


def test_collection_names():
    p = ProjectConfig(
        name="livly",
        code_dirs=[CodeDir(path=Path("/tmp"), extensions={".cs"})],
    )
    assert p.code_collection == "livly-code"
    assert p.sessions_collection == "livly-sessions"
    assert p.docs_collection == "livly-docs"


def test_save_and_reload(tmp_path):
    config_file = tmp_path / "config.yaml"
    config = load_config(config_file)
    config.add_project(
        "test",
        code_dirs=[CodeDir(path=Path("/tmp/code"), extensions={".cs", ".ts"})],
        sessions_dirs=[Path("/tmp/sessions")],
        docs_dir=Path("/tmp/docs"),
    )
    config.save()
    reloaded = load_config(config_file)
    assert "test" in reloaded.projects
    p = reloaded.projects["test"]
    assert len(p.code_dirs) == 1
    assert p.code_dirs[0].path == Path("/tmp/code")
    assert p.code_dirs[0].extensions == {".cs", ".ts"}
    assert p.sessions_dirs == [Path("/tmp/sessions")]
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


def test_load_config_multi_sessions_dirs(tmp_path):
    """Multiple sessions_dirs load as a list."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {
            "proj": {
                "code_dirs": [{"path": "/tmp/code", "extensions": [".cs"]}],
                "sessions_dirs": ["/tmp/sessions1", "/tmp/sessions2"],
            }
        }
    }))
    config = load_config(config_file)
    p = config.projects["proj"]
    assert p.sessions_dirs == [Path("/tmp/sessions1"), Path("/tmp/sessions2")]


def test_load_config_singular_sessions_dir_compat(tmp_path):
    """Legacy singular sessions_dir loads as single-element list."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {
            "proj": {
                "code_dirs": [{"path": "/tmp/code", "extensions": [".cs"]}],
                "sessions_dir": "/tmp/sessions",
            }
        }
    }))
    config = load_config(config_file)
    p = config.projects["proj"]
    assert p.sessions_dirs == [Path("/tmp/sessions")]


def test_load_config_no_sessions_dirs(tmp_path):
    """No sessions config results in empty list."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {
            "proj": {
                "code_dirs": [{"path": "/tmp/code", "extensions": [".cs"]}],
            }
        }
    }))
    config = load_config(config_file)
    p = config.projects["proj"]
    assert p.sessions_dirs == []


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


def test_save_and_reload_multi_sessions_dirs(tmp_path):
    """Multi sessions_dirs round-trip through save/load."""
    config_file = tmp_path / "config.yaml"
    config = load_config(config_file)
    config.add_project(
        "test",
        code_dirs=[CodeDir(path=Path("/tmp/code"), extensions={".cs"})],
        sessions_dirs=[Path("/tmp/s1"), Path("/tmp/s2")],
    )
    config.save()
    reloaded = load_config(config_file)
    p = reloaded.projects["test"]
    assert p.sessions_dirs == [Path("/tmp/s1"), Path("/tmp/s2")]


def test_save_empty_sessions_dirs_omitted(tmp_path):
    """Empty sessions_dirs list is not written to YAML."""
    config_file = tmp_path / "config.yaml"
    config = load_config(config_file)
    config.add_project(
        "test",
        code_dirs=[CodeDir(path=Path("/tmp/code"), extensions={".cs"})],
    )
    config.save()
    raw = yaml.safe_load(config_file.read_text())
    assert "sessions_dirs" not in raw["projects"]["test"]
    assert "sessions_dir" not in raw["projects"]["test"]


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
