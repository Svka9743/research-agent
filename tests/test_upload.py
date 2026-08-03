from pathlib import Path


def test_documents_folder_exists():
    assert Path("documents").exists()


def test_vector_store_folder_exists():
    assert Path("vector_store").exists()