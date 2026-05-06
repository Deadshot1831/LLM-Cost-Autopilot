def test_get_models_returns_all_registered(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    models = r.json()["models"]
    ids = {m["model_id"] for m in models}
    assert {"gpt-4o", "gpt-4o-mini"}.issubset(ids)
    sample = next(m for m in models if m["model_id"] == "gpt-4o")
    assert sample["provider"] == "openai"
    assert sample["quality_tier"] == "complex"
    assert sample["input_cost_per_1k"] > 0
