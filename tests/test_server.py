import pytest
from fastapi.testclient import TestClient

import engram
from engram.server import app


@pytest.fixture(autouse=True)
def fresh_store(tmp_path):
    engram.configure(tmp_path)
    yield
    engram.reset()


client = TestClient(app)


def test_remember_and_recall():
    res = client.post(
        "/remember",
        json={"text": "Marie Curie was born in Warsaw.", "dataset": "science"},
    )
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "dataset": "science"}

    recall_res = client.post(
        "/recall",
        json={
            "query": "Where was Marie Curie born?",
            "dataset": "science",
            "top_k": 1,
        },
    )
    assert recall_res.status_code == 200
    results = recall_res.json()["results"]
    assert len(results) == 1
    assert "Marie Curie" in results[0]["text"]


def test_remember_invalid_strength():
    res = client.post(
        "/remember",
        json={"text": "Invalid strength test.", "dataset": "test", "strength": 1.5},
    )
    assert res.status_code == 422


def test_improve_endpoint():
    res = client.post(
        "/improve",
        json={"text": "Python is a programming language.", "dataset": "coding"},
    )
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "dataset": "coding"}


def test_prune_endpoint():
    client.post(
        "/remember",
        json={"text": "Weak memory.", "dataset": "temp", "strength": 0.1},
    )
    res = client.post("/prune", json={"dataset": "temp", "threshold": 0.5})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["pruned"] == 1


def test_forget_endpoint():
    client.post(
        "/remember",
        json={"text": "Temporary note.", "dataset": "ephemeral"},
    )
    forget_res = client.post("/forget", json={"dataset": "ephemeral"})
    assert forget_res.status_code == 200
    assert forget_res.json() == {"status": "ok", "dataset": "ephemeral"}

    recall_res = client.post(
        "/recall",
        json={"query": "Temporary note", "dataset": "ephemeral"},
    )
    assert recall_res.json()["results"] == []
