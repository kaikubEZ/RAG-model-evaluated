import pytest

from rageval import store


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Point the tool at an empty project directory."""
    monkeypatch.setattr(store, "ROOT", tmp_path)
    return tmp_path
