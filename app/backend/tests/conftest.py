from pathlib import Path
import importlib.util
import sys

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = BACKEND_DIR / "main.py"

spec = importlib.util.spec_from_file_location("backend_main", MODULE_PATH)
backend_main = importlib.util.module_from_spec(spec)
sys.modules["backend_main"] = backend_main
assert spec.loader is not None
spec.loader.exec_module(backend_main)


@pytest.fixture
def backend_module():
    return backend_main


@pytest.fixture
def client():
    return TestClient(backend_main.app)


class DummyCollection:
    def insert_one(self, record):
        return None

    def find(self, *args, **kwargs):
        return self

    def sort(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return []


class DummyDatabase:
    def __getitem__(self, name):
        return DummyCollection()


class DummyAdmin:
    def command(self, name):
        return {"ok": 1}


class DummyMongoClient:
    admin = DummyAdmin()

    def __getitem__(self, name):
        return DummyDatabase()


@pytest.fixture
def mock_mongo(monkeypatch, backend_module):
    monkeypatch.setattr(
        backend_module.pymongo,
        "MongoClient",
        lambda *args, **kwargs: DummyMongoClient(),
    )
