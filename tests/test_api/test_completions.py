def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_post_completions_returns_text_and_meta(client):
    r = client.post("/v1/completions", json={"prompt": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert "text" in body
    assert "meta" in body
    meta = body["meta"]
    assert meta["tier"] in {"simple", "moderate", "complex"}
    assert "candidate_model" in meta
    assert "final_model" in meta
    assert isinstance(meta["escalated"], bool)
    assert "routing_reason" in meta


def test_completions_rejects_empty_prompt(client):
    r = client.post("/v1/completions", json={"prompt": ""})
    assert r.status_code == 422


def test_completions_increments_stats(client):
    s_before = client.get("/v1/stats").json()
    client.post("/v1/completions", json={"prompt": "hello"})
    s_after = client.get("/v1/stats").json()
    assert s_after["total_requests"] == s_before["total_requests"] + 1
