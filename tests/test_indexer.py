from pathlib import Path

from vecs.indexer import Manifest


def test_manifest_new_file(tmp_path):
    """A new file is detected as needing indexing."""
    manifest_path = tmp_path / "manifest.json"
    m = Manifest(manifest_path)
    test_file = tmp_path / "test.cs"
    test_file.write_text("hello")
    assert m.needs_indexing(test_file) is True


def test_manifest_already_indexed(tmp_path):
    """A file that hasn't changed is skipped."""
    manifest_path = tmp_path / "manifest.json"
    m = Manifest(manifest_path)
    test_file = tmp_path / "test.cs"
    test_file.write_text("hello")
    m.mark_indexed(test_file)
    m.save()
    # Reload
    m2 = Manifest(manifest_path)
    assert m2.needs_indexing(test_file) is False


def test_manifest_changed_file(tmp_path):
    """A file with new content is detected as needing re-indexing."""
    manifest_path = tmp_path / "manifest.json"
    m = Manifest(manifest_path)
    test_file = tmp_path / "test.cs"
    test_file.write_text("hello")
    m.mark_indexed(test_file)
    m.save()
    test_file.write_text("changed")
    m2 = Manifest(manifest_path)
    assert m2.needs_indexing(test_file) is True
