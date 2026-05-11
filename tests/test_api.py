from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_chat_clarify() -> None:
    res = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "I need an assessment"}]},
    )
    body = res.json()
    assert res.status_code == 200
    assert isinstance(body["reply"], str)
    assert body["recommendations"] == []
    assert body["end_of_conversation"] is False


def test_chat_recommendation_shape() -> None:
    res = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "Hiring a Java developer, mid-level, needs communication and personality."}
            ]
        },
    )
    body = res.json()
    assert res.status_code == 200
    assert "reply" in body
    assert "recommendations" in body
    assert "end_of_conversation" in body
    assert isinstance(body["recommendations"], list)
    for rec in body["recommendations"]:
        assert set(rec.keys()) == {"name", "url", "test_type"}
        assert rec["url"].startswith("https://www.shl.com/")
