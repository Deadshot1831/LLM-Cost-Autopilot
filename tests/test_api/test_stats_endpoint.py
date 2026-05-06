def test_stats_starts_at_zero(client):
    r = client.get("/v1/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total_requests"] == 0
    assert body["savings_pct"] == 0.0


def test_stats_after_one_request(client):
    client.post("/v1/completions", json={"prompt": "hello world"})
    body = client.get("/v1/stats").json()
    assert body["total_requests"] == 1
    assert body["baseline_cost_total"] > 0
