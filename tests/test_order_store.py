"""R-34 tests for the file-backed order store and order API."""

import json
import re
import secrets
import struct

import pytest
from fastapi.testclient import TestClient

from recon import service
from recon.order_store import OrderStore

ORDER_RE = re.compile(r"^PO-[0-9ABCDEFGHJKMNPQRSTVWXYZ]{8}$")


def _malformed_glb_accessor_reference() -> bytes:
    document = json.dumps(
        {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": 0}],
            "bufferViews": [],
            "accessors": [],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
            "nodes": [{"mesh": 0}],
            "scenes": [{"nodes": [0]}],
            "scene": 0,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    document += b" " * ((-len(document)) % 4)
    total_length = 12 + 8 + len(document)
    return (
        b"glTF"
        + struct.pack("<II", 2, total_length)
        + struct.pack("<I4s", len(document), b"JSON")
        + document
    )


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
    source = tmp_path / "upload.stl"
    source.write_bytes(raw)
    record = store.create(
        created_at="2026-11-15T00:00:00+00:00",
        spec={},
        placement_note=None,
        buyer_reference=None,
        cad_name="buyer-part.stl",
        cad_source_path=(tmp_path / "upload.stl"),
        cad_units="mm",
        cad_bbox_mm=[[0.0, 0.0, 0.0], [10.0, 20.0, 0.0]],
    )

    assert record["has_cad"] is True
    assert store.cad_path(record["order_code"]).read_bytes() == raw


def test_order_code_path_traversal_is_rejected(tmp_path):
    orders = tmp_path / "orders"
    orders.mkdir()
    (tmp_path / "order.json").write_text(
        json.dumps(
            {
                "order_code": "..",
                "created_at": "2026-11-15T00:00:00+00:00",
                "has_cad": False,
                "cad": None,
                "spec": {"secret": "must-not-leak"},
                "placement_note": "private",
                "buyer_reference": "PRIVATE-PO",
            }
        ),
        encoding="utf-8",
    )
    app = service.create_app(tmp_path / "sessions", orders_dir=orders)

    with TestClient(app) as client:
        fetched = client.get("/v1/orders/%2e%2e")
        session = client.post("/v1/sessions", json={"order_code": ".."})

    assert fetched.status_code == 404
    assert fetched.json()["code"] == "order_code_unknown"
    assert session.status_code == 400
    assert session.json()["code"] == "order_code_unknown"


def test_order_code_collision_retries_atomically(tmp_path, monkeypatch):
    store = OrderStore(tmp_path)
    (tmp_path / "PO-AAAAAAAA").mkdir()
    choices = iter("A" * 8 + "B" * 8)
    monkeypatch.setattr(secrets, "choice", lambda _alphabet: next(choices))

    record = store.create(
        created_at="2026-11-15T00:00:00+00:00",
        spec={},
        placement_note=None,
        buyer_reference=None,
    )

    assert record["order_code"] == "PO-BBBBBBBB"


def test_whitespace_optional_fields_are_normalized_to_none(tmp_path):
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/orders",
            headers={"x-proofshape-buyer-key": "buyer-secret"},
            files={"spec": ("spec.json", "{}", "application/json")},
            data={"placement_note": "   ", "buyer_reference": "\t"},
        )

    assert response.status_code == 201
    assert "placement_note" not in response.json()
    assert "buyer_reference" not in response.json()


def test_malformed_glb_reference_returns_cad_unreadable(tmp_path):
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
                "cad": (
                    "broken.glb",
                    _malformed_glb_accessor_reference(),
                    "model/gltf-binary",
                ),
            },
        )

    assert response.status_code == 422
    assert response.json()["code"] == "cad_unreadable"


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
            data={
                "placement_note": "Board arrow up",
                "buyer_reference": "PO-EXTERNAL-7",
            },
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
        assert (
            client.post("/v1/sessions", json={"order_code": "PO-TEST"}).status_code
            == 201
        )
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

        assert (
            client.get(f"/v1/sessions/{sid_no_cad}/verdict").json()["code"]
            == "no_cad_uploaded"
        )
        assert (
            client.get(f"/v1/sessions/{sid_with_cad}/verdict").json()["code"]
            == "verdict_not_available"
        )


def test_order_response_surfaces_intake_check_for_an_unverifiable_feature(tmp_path):
    """I-09: a feature whose tolerance sits below the board tier's k*sigma_floor is flagged
    Unverifiable in both the createOrder and getOrder responses -- the same endpoint round
    trip test_create_order_round_trip_and_session_gate already exercises, with a real
    A-03-shaped spec this time."""
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
    )
    with TestClient(app) as client:
        # Both in-plane kinds (hole/edge), checked against D-078's measured floor -- a
        # height-like kind (face/overall) is refused instead, since neither states its axis
        # (see test_order_response_refuses_face_and_overall_for_their_axis_ambiguity below).
        spec = {
            "features": [
                {
                    "name": "hole_A",
                    "kind": "hole",
                    "tolerance_mm": 1.0,
                    "critical": True,
                },
                {
                    "name": "edge_B",
                    "kind": "edge",
                    "tolerance_mm": 5.0,
                    "critical": False,
                },
            ]
        }
        created = client.post(
            "/v1/orders",
            headers={"x-proofshape-buyer-key": "buyer-secret"},
            files={"spec": ("spec.json", json.dumps(spec), "application/json")},
        )

        assert created.status_code == 201
        checks = {
            r["feature_name"]: r["status"] for r in created.json()["intake_check"]
        }
        assert checks == {"hole_A": "unverifiable", "edge_B": "ok"}

        fetched = client.get(f"/v1/orders/{created.json()['order_code']}")
        assert fetched.json()["intake_check"] == created.json()["intake_check"]


def test_order_response_refuses_face_and_overall_for_their_axis_ambiguity(tmp_path):
    """Found on review (@TabeenRaoof, PR #372): neither kind states whether it means the
    board-perpendicular dimension (D-073) or an in-plane one (D-078), so neither floor is
    assumed -- both are refused, regardless of tolerance."""
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
    )
    with TestClient(app) as client:
        spec = {
            "features": [
                {
                    "name": "top_face_A",
                    "kind": "face",
                    "tolerance_mm": 0.1,
                    "critical": True,
                },
                {
                    "name": "overall_B",
                    "kind": "overall",
                    "tolerance_mm": 50.0,
                    "critical": False,
                },
            ]
        }
        created = client.post(
            "/v1/orders",
            headers={"x-proofshape-buyer-key": "buyer-secret"},
            files={"spec": ("spec.json", json.dumps(spec), "application/json")},
        )

        assert created.status_code == 201
        checks = {
            r["feature_name"]: r["status"] for r in created.json()["intake_check"]
        }
        assert checks == {"top_face_A": "refused", "overall_B": "refused"}


def test_order_response_refuses_a_null_tolerance_feature(tmp_path):
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
    )
    with TestClient(app) as client:
        spec = {
            "features": [
                {
                    "name": "edge_C",
                    "kind": "edge",
                    "tolerance_mm": None,
                    "critical": True,
                }
            ]
        }
        created = client.post(
            "/v1/orders",
            headers={"x-proofshape-buyer-key": "buyer-secret"},
            files={"spec": ("spec.json", json.dumps(spec), "application/json")},
        )

    # response_model_exclude_none=True drops tolerance_mm (null) from the JSON entirely.
    assert created.json()["intake_check"] == [
        {
            "feature_name": "edge_C",
            "status": "refused",
            "reason": "tolerance_mm is missing or null; refused, not defaulted.",
        }
    ]


def test_order_response_refuses_a_non_numeric_tolerance_without_server_error(tmp_path):
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
                    json.dumps(
                        {
                            "features": [
                                {
                                    "name": "height",
                                    "kind": "face",
                                    "tolerance_mm": "1.0",
                                    "critical": True,
                                }
                            ]
                        }
                    ),
                    "application/json",
                )
            },
        )

    assert created.status_code == 201
    assert created.json()["intake_check"] == [
        {
            "feature_name": "height",
            "status": "refused",
            "reason": "tolerance_mm must be a finite positive number; refused, not defaulted.",
        }
    ]


@pytest.mark.parametrize(
    ("spec", "expected_statuses"),
    [
        ({"features": None}, []),
        ({"features": "abc"}, ["refused"]),
        ({"features": {"a": 1}}, ["refused"]),
        ({"features": [5]}, ["refused"]),
        (
            {"features": [{"name": 7, "kind": "edge", "tolerance_mm": 1.0}]},
            ["unverifiable"],
        ),
    ],
)
def test_order_response_refuses_malformed_opaque_specs_without_server_error(
    tmp_path, spec, expected_statuses
):
    """Found on review (@TabeenRaoof, PR #372): each of these previously returned 500 on
    createOrder, and -- because the order is saved before the response is built -- 500 on
    every later getOrder too, from a spec that `main` accepts with 201. R-34 stores a spec as
    opaque JSON, so a buyer-supplied `features` shape this malformed reaches the server as-is.
    The last case's `name` is malformed, but its `kind`/`tolerance_mm` are not -- the real check
    still runs, with the name replaced by a placeholder, rather than every malformed field
    forcing a blanket refusal."""
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
    )
    with TestClient(app) as client:
        created = client.post(
            "/v1/orders",
            headers={"x-proofshape-buyer-key": "buyer-secret"},
            files={"spec": ("spec.json", json.dumps(spec), "application/json")},
        )

        assert created.status_code == 201
        statuses = [r["status"] for r in created.json()["intake_check"]]
        assert statuses == expected_statuses

        fetched = client.get(f"/v1/orders/{created.json()['order_code']}")
        assert fetched.status_code == 200
        assert fetched.json()["intake_check"] == created.json()["intake_check"]


def test_order_response_checks_an_in_plane_feature_against_the_in_plane_floor_not_height(
    tmp_path,
):
    """The exact bug found on review (@mbj1994, PR #372): use D-078's in-plane floor, not height."""
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
    )
    with TestClient(app) as client:
        spec = {
            "features": [
                {
                    "name": "edge_1",
                    "kind": "edge",
                    "tolerance_mm": 1.0,
                    "critical": True,
                },
                {
                    "name": "edge_2",
                    "kind": "edge",
                    "tolerance_mm": 5.0,
                    "critical": True,
                },
            ]
        }
        created = client.post(
            "/v1/orders",
            headers={"x-proofshape-buyer-key": "buyer-secret"},
            files={"spec": ("spec.json", json.dumps(spec), "application/json")},
        )

    checks = {r["feature_name"]: r["status"] for r in created.json()["intake_check"]}
    assert checks == {"edge_1": "unverifiable", "edge_2": "ok"}


def test_order_response_is_empty_intake_check_when_spec_has_no_features(tmp_path):
    # Existing orders (and R-34's own placeholder-shaped specs, e.g. {"tolerance_mm": 4.0})
    # have no "features" key at all -- D-004, nothing to check yet, not an error.
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
                )
            },
        )

    assert created.json()["intake_check"] == []
