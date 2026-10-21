import pytest

from tools import verify_delivery


def documents(monkeypatch, root, paths):
    monkeypatch.setattr(verify_delivery, "ROOT", root)
    monkeypatch.setattr(verify_delivery, "indexed_paths", lambda: paths)


def test_release_links_only_reference_versioned_files(monkeypatch, tmp_path):
    readme = tmp_path / "README.md"
    ignored = tmp_path / "private.md"
    ignored.write_text("local archive")
    readme.write_text("[archive](private.md)")
    documents(monkeypatch, tmp_path, [readme])
    with pytest.raises(ValueError, match="Broken delivery link"):
        verify_delivery.check_links()


def test_release_links_include_images_and_nested_documents(monkeypatch, tmp_path):
    guide = tmp_path / "guide.md"
    guide.write_text("![product](missing.png)")
    documents(monkeypatch, tmp_path, [guide])
    with pytest.raises(ValueError, match="missing.png"):
        verify_delivery.check_links()


def test_release_links_allow_urls_anchors_and_encoded_paths(monkeypatch, tmp_path):
    readme = tmp_path / "README.md"
    linked = tmp_path / "guide name.md"
    linked.write_text("# Guide")
    readme.write_text("[guide](guide%20name.md#guide) [web](https://example.com) [local](#title)")
    documents(monkeypatch, tmp_path, [readme, linked])
    verify_delivery.check_links()


def test_public_documentation_links_resolve():
    verify_delivery.check_links()
