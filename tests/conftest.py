import atexit
import shutil
import tempfile
from pathlib import Path

import pytest

CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"


def _restore(backup: Path) -> None:
    try:
        shutil.copytree(backup, CHROMA_DIR, dirs_exist_ok=True)
    except OSError as e:
        print(f"\nCould not restore {CHROMA_DIR} ({e}); run: git checkout -- data/chroma_db/")


@pytest.fixture(scope="session", autouse=True)
def restore_committed_chroma_db():
    """Opening a chromadb PersistentClient rewrites its files even for reads, and chromadb flushes
    again at interpreter exit. Restore the committed index in an atexit hook registered before any
    client exists — atexit runs hooks in reverse order, so this one runs after chromadb's."""
    if CHROMA_DIR.exists():
        backup = Path(tempfile.mkdtemp(prefix="chroma_backup_")) / "chroma_db"
        shutil.copytree(CHROMA_DIR, backup)
        atexit.register(_restore, backup)
    yield
