def test_health(client, mock_mongo):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "backend"
    assert data["mongodb_status"] == "connected"
