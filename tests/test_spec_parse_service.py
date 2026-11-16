import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from recon import service


def _reply(payload):
    return SimpleNamespace(
        prompt_feedback=SimpleNamespace(block_reason=None),
        candidates=[SimpleNamespace(finish_reason="STOP")],
        text=json.dumps(payload),
    )


def test_parse_spec_requires_buyer_key(tmp_path):
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
        spec_parser=lambda _prose: _reply({"features": [], "assumptions": []}),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/spec/parse", json={"prose": "two holes within 2 mm"}
        )

    assert response.status_code == 401
    assert response.json()["code"] == "buyer_auth_required"


def test_parse_spec_returns_a03_shape(tmp_path):
    payload = {
        "features": [
            {
                "name": "mounting holes",
                "kind": "hole",
                "tolerance_mm": 2.0,
                "critical": True,
            },
            {
                "name": "sealing face",
                "kind": "face",
                "tolerance_mm": None,
                "critical": True,
            },
        ],
        "assumptions": ["No numeric tolerance was given for the sealing face."],
    }
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
        spec_parser=lambda _prose: _reply(payload),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/spec/parse",
            headers={"x-proofshape-buyer-key": "buyer-secret"},
            json={"prose": "mounting holes within 2 mm; sealing face flat"},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "parsed", "spec": payload, "reason": None}


def test_parse_spec_missing_api_key_is_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/spec/parse",
            headers={"x-proofshape-buyer-key": "buyer-secret"},
            json={"prose": "mounting holes within 2 mm"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "unparsed"
    assert response.json()["reason"] == "missing_api_key"


def test_parse_spec_blank_prose_is_contract_error(tmp_path):
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
        spec_parser=lambda _prose: None,
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/spec/parse",
            headers={"x-proofshape-buyer-key": "buyer-secret"},
            json={"prose": "   "},
        )

    assert response.status_code == 422
    assert response.json()["code"] == "malformed_request"
