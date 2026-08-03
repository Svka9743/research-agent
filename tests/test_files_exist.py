from pathlib import Path


def test_required_files_exist():
    assert Path("app.py").exists()
    assert Path("rag.py").exists()
    assert Path("README.md").exists()