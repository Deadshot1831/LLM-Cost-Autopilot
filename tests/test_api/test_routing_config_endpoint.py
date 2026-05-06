def test_get_routing_config(client):
    r = client.get("/v1/routing-config")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"simple", "moderate", "complex"}


def test_put_routing_config_updates(client):
    new_cfg = {
        "simple": "gpt-4o-mini",
        "moderate": "gpt-4o-mini",
        "complex": "gpt-4o",
    }
    r = client.put("/v1/routing-config", json=new_cfg)
    assert r.status_code == 200
    assert r.json() == new_cfg
    after = client.get("/v1/routing-config").json()
    assert after == new_cfg


def test_put_routing_config_rejects_unknown_model(client):
    r = client.put("/v1/routing-config", json={
        "simple": "nonexistent",
        "moderate": "gpt-4o-mini",
        "complex": "gpt-4o",
    })
    assert r.status_code == 400


def test_put_routing_config_persists_to_yaml(client, app_state):
    new_cfg = {
        "simple": "gpt-4o-mini",
        "moderate": "claude-haiku-4-5",
        "complex": "gpt-4o",
    }
    client.put("/v1/routing-config", json=new_cfg)
    text = app_state.routing_path.read_text()
    assert "gpt-4o-mini" in text
    assert "claude-haiku-4-5" in text
