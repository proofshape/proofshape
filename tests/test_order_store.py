"""R-34 tests for the file-backed order store and order API."""

import json
import re

from fastapi.testclient import TestClient

from recon import service
from recon.order_store import OrderStore

ORDER_RE = re.compile(r"^PO-[0-9ABCDEFGHJKMNPQRSTVWXYZ]{8}$")


def _ascii_stl() -> bytes:
    return b"""solid part
facet normal 0 0 1
 outer loop
  vertex 0 0 0
  vertex 10 0 0
  vertex 0 20 0
 endloop
endfacet
endsolid part
"""


def test_order_store_round_trip_atomic_write_and_code_format(tmp_path):
    store = OrderStore(tmp_path)
    first = store.create(
        created_at="2026-11-15T00:00:00+00:00",
        spec={"critical_region": "hole"},
        placement_note="Arrow forward",
        buyer_reference="BUYER-42",
    )
    second = store.create(
        created_at="2026-11-15T00:00:01+00:00",
        spec={},
        placement_note=None,
        buyer_reference=None,
    )

    assert ORDER_RE.fullmatch(first["order_code"])
    assert ORDER_RE.fullmatch(second["order_code"])
    assert first["order_code"] != second["order_code"]
    assert store.load(first["order_code"]) == first
    assert store.record_path(first["order_code"]).exists()
    assert not store.record_path(first["order_code"]).with_suffix(".json.tmp").exists()


def test_order_store_keeps_cad_bytes_unchanged(tmp_path):
    store = OrderStore(tmp_path)
    raw = _ascii_stl()
    record = store.create(
        created_at="2026-11-15T00:00:00+00:00",
        spec={},
        placement_note=None,
        buyer_reference=None,
        cad_name="buyer-part.stl",
        cad_bytes=raw,
        cad_units="mm",
        cad_bbox_mm=[[0.0, 0.0, 0.0], [10.0, 20.0, 0.0]],
    )

    assert record["has_cad"] is True
    assert store.cad_path(record["order_code"]).read_bytes() == raw


def test_create_order_round_trip_and_session_gate(tmp_path):
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
    )
    with TestClient(app) as client:
        created = client.post(
            "/v1/orders",
            headers={"x-proofshape-buyer-key": "buyer-secret"},
            files={
                "spec": (
                    "spec.json",
                    json.dumps({"tolerance_mm": 4.0}),
                    "application/json",
                ),
                "cad": ("part.stl", _ascii_stl(), "model/stl"),
            },
            data={"placement_note": "Board arrow up", "buyer_reference": "PO-EXTERNAL-7"},
        )
        assert created.status_code == 201
        body = created.json()
        assert ORDER_RE.fullmatch(body["order_code"])
        assert body["has_cad"] is True
        assert body["cad_units"] == "mm"
        assert body["cad_bounding_box_mm"] == [[0.0, 0.0, 0.0], [10.0, 20.0, 0.0]]
        assert body["spec"] == {"tolerance_mm": 4.0}

        fetched = client.get(f"/v1/orders/{body['order_code']}")
        assert fetched.status_code == 200
        assert fetched.json() == body

        session = client.post("/v1/sessions", json={"order_code": body["order_code"]})
        assert session.status_code == 201
        assert session.json()["order_code"] == body["order_code"]


def test_arbitrary_order_code_is_rejected_by_default(tmp_path):
    app = service.create_app(tmp_path / "sessions", orders_dir=tmp_path / "orders")
    with TestClient(app) as client:
        response = client.post("/v1/sessions", json={"order_code": "PO-NOTREAL"})

    assert response.status_code == 400
    assert response.json()["code"] == "order_code_unknown"


def test_known_order_codes_override_store_for_tests(tmp_path):
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        known_order_codes={"PO-TEST"},
    )
    with TestClient(app) as client:
        assert client.post("/v1/sessions", json={"order_code": "PO-TEST"}).status_code == 201
        assert (
            client.post("/v1/sessions", json={"order_code": "PO-OTHER"}).json()["code"]
            == "order_code_unknown"
        )


def test_create_order_requires_buyer_key(tmp_path):
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/orders",
            files={"spec": ("spec.json", "{}", "application/json")},
        )

    assert response.status_code == 401
    assert response.json()["code"] == "buyer_auth_required"


def test_unreadable_cad_returns_named_error(tmp_path):
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/orders",
            headers={"x-proofshape-buyer-key": "buyer-secret"},
            files={
                "spec": ("spec.json", "{}", "application/json"),
                "cad": ("broken.stl", b"not an stl", "model/stl"),
            },
        )

    assert response.status_code == 422
    assert response.json()["code"] == "cad_unreadable"


def test_verdict_distinguishes_no_cad_from_not_available(tmp_path):
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
    )
    with TestClient(app) as client:
        no_cad = client.post(
            "/v1/orders",
            headers={"x-proofshape-buyer-key": "buyer-secret"},
            files={"spec": ("spec.json", "{}", "application/json")},
        ).json()
        with_cad = client.post(
            "/v1/orders",
            headers={"x-proofshape-buyer-key": "buyer-secret"},
            files={
                "spec": ("spec.json", "{}", "application/json"),
                "cad": ("part.stl", _ascii_stl(), "model/stl"),
            },
        ).json()

        sid_no_cad = client.post(
            "/v1/sessions", json={"order_code": no_cad["order_code"]}
        ).json()["session_id"]
        sid_with_cad = client.post(
            "/v1/sessions", json={"order_code": with_cad["order_code"]}
        ).json()["session_id"]

        assert client.get(f"/v1/sessions/{sid_no_cad}/verdict").json()["code"] == "no_cad_uploaded"
        assert (
            client.get(f"/v1/sessions/{sid_with_cad}/verdict").json()["code"]
            == "verdict_not_available"
        )
