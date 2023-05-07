import json
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_chat_completion():
    response = client.post("/chat_completion", json={"text": "What is the capital"})
    assert response.status_code == 200

    data = response.json()
    assert "completions" in data
    assert len(data["completions"]) == 5
    assert all(isinstance(completion, str) for completion in data["completions"])
