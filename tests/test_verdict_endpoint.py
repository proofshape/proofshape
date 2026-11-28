"""I-07: getVerdict end to end, through the live service -- registration, ambiguity and
deviation are otherwise dormant, so this is the first test that exercises the whole wiring
(run_session -> run_post_reconstruction_measurement -> getVerdict), not just compute_verdict
in isolation (see tests/test_verdict.py for that)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from inspection import cad, registration, synthetic

from recon import scale_agreement, service
from recon.board_pose import BOARD_SQUARE_MM
from recon.mesh_cleanup import write_glb
from recon.provenance import labels_to_colors
from recon.report_narrative import VerdictInput, template_narrative
from recon.uncertainty import write_sigma_npz
from recon.vlm_check import VlmSupervisor

ROOT = Path(__file__).resolve().parents[1]
# A scalene right-triangle prism (40x30x20mm) -- deliberately not g01_calibration_cube_30mm.stl
# (cubical symmetry, 24-fold ambiguous) nor a rectangular box (still 4-fold ambiguous under
# 180-degree flips about each axis even with three unequal dimensions -- confirmed directly
# against real `registration.global_register`/`check_ambiguity`, not assumed: a rectangular
# box's "no two dimensions equal" does not remove its rotational symmetry). A scalene triangle's
# cross-section has none, confirmed the same way, which is what every "clean part" test in this
# file actually needs.
ASYMMETRIC_PART = (
    ROOT / "fixtures" / "reference_cad" / "g02_03_right_triangle_40x30x20mm.stl"
)

FLAT_BOARD_SURFACE = {
    "slope_x": 0.0,
    "slope_y": 0.0,
    "z_at_origin_mm": 0.0,
    "offset_mm": 0.0,
    "tilt_deg": 0.0,
    "residual_sigma_mm": 0.0,
    "inlier_count": 0,
    "candidate_count": 0,
    "board_layer_top_mm": None,
    "height_threshold_mm": 2.0,
}


def _make_pipeline_runner(*, sigma_mm: float = 0.3):
    """A fake pipeline_runner that writes a real, fully-Observed, noise-free synthetic cube
    (I-01's generator) as the served GLB, plus a real sigma_mm.npz and a real
    fusion/tsdf_fusion.json -- everything run_post_reconstruction_measurement and getVerdict
    need to run for real, not stubbed."""

    def pipeline_runner(capture_dir: Path, artifact_dir: Path) -> dict:
        cad_vertices, cad_faces, _units = cad.load_mesh(ASYMMETRIC_PART)
        sample = synthetic.generate_synthetic_sample(
            cad_vertices, cad_faces, seed=0, sigma_mm=0.0, patch_fraction=0.0
        )

        uncertainty_dir = artifact_dir / "uncertainty"
        uncertainty_dir.mkdir(parents=True, exist_ok=True)
        glb_path = uncertainty_dir / "mesh_uncertainty.glb"
        write_glb(
            glb_path, sample.vertices_mm, sample.faces, labels_to_colors(sample.labels)
        )
        write_sigma_npz(
            uncertainty_dir / "sigma_mm.npz",
            np.full(len(sample.labels), sigma_mm),
            np.full(len(sample.labels), 10, dtype=np.int64),
        )

        fusion_dir = artifact_dir / "fusion"
        fusion_dir.mkdir(parents=True, exist_ok=True)
        (fusion_dir / "tsdf_fusion.json").write_text(
            json.dumps({"board_surface": FLAT_BOARD_SURFACE})
        )

        return {
            "glb_path": str(glb_path),
            "reference_tier_used": "charuco_board",
            "metric": True,
            "observed_fraction": 1.0,
            "stage_timings_s": {},
            "warnings": [],
        }

    return pipeline_runner


def _clean_vlm_reply(frame_path: Path, reference_path: Path | None) -> SimpleNamespace:
    """A-02/D-084: every sampled frame comes back `unchecked`/`missing_api_key` by default in
    tests (no `GEMINI_API_KEY` set), which the veto now correctly downgrades to Rescan -- tests
    that need a clean `Pass`/`Fail` through the real verdict path inject this instead, mirroring
    `tests/test_vlm_check.py`'s own fake-reply shape."""
    observation = {
        "same_part": None if reference_path is None else True,
        "hands_occluding": False,
        "screen_photo": False,
        "lighting_adequate": True,
        "hint": "Keep the part visible.",
    }
    return SimpleNamespace(
        text=json.dumps(observation),
        model_version="test-model-version",
        candidates=[SimpleNamespace(finish_reason="STOP")],
        prompt_feedback=None,
    )


def _finish_a_session_with_cad(client: TestClient, spec: dict) -> str:
    created = client.post(
        "/v1/orders",
        headers={"x-proofshape-buyer-key": "buyer-secret"},
        files={
            "spec": ("spec.json", json.dumps(spec), "application/json"),
            "cad": ("part.stl", ASYMMETRIC_PART.read_bytes(), "model/stl"),
        },
    ).json()
    session_id = client.post(
        "/v1/sessions", json={"order_code": created["order_code"]}
    ).json()["session_id"]
    gyro = json.dumps({"alpha": 1.0, "beta": 2.0, "gamma": 3.0})
    for frame_index in range(3):
        client.post(
            f"/v1/sessions/{session_id}/frames",
            files={
                "frame": (f"frame{frame_index}.jpg", b"jpeg-data", "image/jpeg"),
                "gyro": ("blob", gyro, "application/json"),
            },
        )
    response = client.post(f"/v1/sessions/{session_id}/finish")
    assert response.status_code == 202
    return session_id


def test_getverdict_passes_a_clean_part_end_to_end(tmp_path):
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
        pipeline_runner=_make_pipeline_runner(),
        vlm_supervisor=VlmSupervisor(inspector=_clean_vlm_reply),
    )
    with TestClient(app) as client:
        spec = {
            "features": [{"name": "overall_A", "kind": "overall", "tolerance_mm": 5.0}]
        }
        session_id = _finish_a_session_with_cad(client, spec)

        response = client.get(f"/v1/sessions/{session_id}/verdict")

    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "Pass"
    assert body["declared_at"] == "post_reconstruction"
    assert body["orientation_ambiguous"] is False
    assert body["regions"][0]["name"] == "overall_A"
    assert body["regions"][0]["contributes_to_verdict"] is True
    # A-09/D-082: narrative is the deterministic template over this same served payload, never
    # a model's -- pins the wiring itself, not report_narrative's own already-tested logic.
    verdict_input = VerdictInput.model_validate(
        {key: value for key, value in body.items() if key != "narrative"}
    )
    assert body["narrative"] == template_narrative(verdict_input)


def test_getverdict_downgrades_to_rescan_when_vlm_checks_are_unchecked_end_to_end(
    tmp_path,
):
    # A-02/D-084: unchecked blocks a Pass -- no GEMINI_API_KEY configured (the default for every
    # other test in this file) means every sampled frame comes back missing_api_key, so a part
    # that would otherwise Pass is downgraded to Rescan instead.
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
        pipeline_runner=_make_pipeline_runner(),
    )
    with TestClient(app) as client:
        spec = {
            "features": [{"name": "overall_A", "kind": "overall", "tolerance_mm": 5.0}]
        }
        session_id = _finish_a_session_with_cad(client, spec)

        response = client.get(f"/v1/sessions/{session_id}/verdict")

    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "Rescan"
    assert any(code.startswith("vlm_unchecked_frame_") for code in body["reason_codes"])


def test_getverdict_downgrades_to_rescan_on_a_screen_photo_end_to_end(tmp_path):
    def screen_photo_reply(
        frame_path: Path, reference_path: Path | None
    ) -> SimpleNamespace:
        observation = {
            "same_part": None if reference_path is None else True,
            "hands_occluding": False,
            "screen_photo": True,
            "lighting_adequate": True,
            "hint": "This looks like a photo of a screen.",
        }
        return SimpleNamespace(
            text=json.dumps(observation),
            model_version="test-model-version",
            candidates=[SimpleNamespace(finish_reason="STOP")],
            prompt_feedback=None,
        )

    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
        pipeline_runner=_make_pipeline_runner(),
        vlm_supervisor=VlmSupervisor(inspector=screen_photo_reply),
    )
    with TestClient(app) as client:
        spec = {
            "features": [{"name": "overall_A", "kind": "overall", "tolerance_mm": 5.0}]
        }
        session_id = _finish_a_session_with_cad(client, spec)

        response = client.get(f"/v1/sessions/{session_id}/verdict")

    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "Rescan"
    assert any(
        code.startswith("vlm_screen_photo_frame_") for code in body["reason_codes"]
    )


def test_getverdict_vetoes_instead_of_500ing_when_vlm_checks_file_is_missing(tmp_path):
    # Review, @TabeenRaoof, PR #396: vlm_checks.json is read unguarded in get_verdict, but
    # should never 500 on a state it can still answer safely.
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
        pipeline_runner=_make_pipeline_runner(),
        vlm_supervisor=VlmSupervisor(inspector=_clean_vlm_reply),
    )
    with TestClient(app) as client:
        spec = {
            "features": [{"name": "overall_A", "kind": "overall", "tolerance_mm": 5.0}]
        }
        session_id = _finish_a_session_with_cad(client, spec)
        (tmp_path / "sessions" / session_id / "vlm_checks.json").unlink()

        response = client.get(f"/v1/sessions/{session_id}/verdict")

    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "Rescan"
    assert "vlm_checks_missing" in body["reason_codes"]


def test_getverdict_refuses_a_null_tolerance_feature_end_to_end(tmp_path):
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
        pipeline_runner=_make_pipeline_runner(),
    )
    with TestClient(app) as client:
        spec = {
            "features": [{"name": "overall_A", "kind": "overall", "tolerance_mm": None}]
        }
        session_id = _finish_a_session_with_cad(client, spec)

        response = client.get(f"/v1/sessions/{session_id}/verdict")

    assert response.status_code == 422
    assert response.json()["code"] == "tolerance_missing"


def test_getverdict_404s_an_empty_spec_not_a_dressed_up_pass_end_to_end(tmp_path):
    """PR #379 review (@TabeenRaoof, @mbj1994): a spec with no features at all previously
    returned a bare Pass through the live service, the same as compute_verdict's own unit
    tests caught in isolation -- this exercises the real wiring end to end."""
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
        pipeline_runner=_make_pipeline_runner(),
    )
    with TestClient(app) as client:
        session_id = _finish_a_session_with_cad(client, {"features": []})

        response = client.get(f"/v1/sessions/{session_id}/verdict")

    assert response.status_code == 404
    assert response.json()["code"] == "verdict_not_available"


def test_getverdict_404s_in_the_gap_before_measurement_finishes_not_rescan(tmp_path):
    """PR #379 review (@TabeenRaoof): run_session marks reconstruction "complete" and saves the
    record before calling run_post_reconstruction_measurement, which then writes `ambiguity`
    separately. A request landing in that window must not see a Rescan that looks like a final
    answer for a part that was never actually measured. TestClient runs background tasks to
    completion before a request returns, so the two steps can't actually race inside this test
    -- instead, the record is rewritten straight after a normal finish to the exact state that
    window looks like (reconstruction complete, no `ambiguity` key yet), which is what
    get_verdict's own check above this test's sibling exercises; this test checks the other
    `reconstruction.status == "complete"` branch specifically."""
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
        pipeline_runner=_make_pipeline_runner(),
    )
    with TestClient(app) as client:
        spec = {
            "features": [{"name": "overall_A", "kind": "overall", "tolerance_mm": 5.0}]
        }
        session_id = _finish_a_session_with_cad(client, spec)

        record_path = tmp_path / "sessions" / session_id / "session.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["reconstruction"]["status"] == "complete"
        assert (
            "ambiguity" in record
        )  # sanity: a normal finish really did write this key
        del record["ambiguity"]
        record_path.write_text(json.dumps(record), encoding="utf-8")

        response = client.get(f"/v1/sessions/{session_id}/verdict")

    assert response.status_code == 404
    assert response.json()["code"] == "verdict_not_available"


def test_getverdict_still_404s_before_reconstruction_completes(tmp_path):
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
        pipeline_runner=_make_pipeline_runner(),
    )
    with TestClient(app) as client:
        created = client.post(
            "/v1/orders",
            headers={"x-proofshape-buyer-key": "buyer-secret"},
            files={
                "spec": ("spec.json", "{}", "application/json"),
                "cad": ("part.stl", ASYMMETRIC_PART.read_bytes(), "model/stl"),
            },
        ).json()
        session_id = client.post(
            "/v1/sessions", json={"order_code": created["order_code"]}
        ).json()["session_id"]

        response = client.get(f"/v1/sessions/{session_id}/verdict")

    assert response.status_code == 404
    assert response.json()["code"] == "verdict_not_available"


def test_i10_getverdict_returns_intake_unverifiable_before_reconstruction(tmp_path):
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
        pipeline_runner=_make_pipeline_runner(),
    )
    with TestClient(app) as client:
        spec = {
            "features": [
                {
                    "name": "hole_A",
                    "kind": "hole",
                    "tolerance_mm": 1.0,
                    "critical": True,
                }
            ]
        }
        created = client.post(
            "/v1/orders",
            headers={"x-proofshape-buyer-key": "buyer-secret"},
            files={
                "spec": ("spec.json", json.dumps(spec), "application/json"),
                "cad": ("part.stl", ASYMMETRIC_PART.read_bytes(), "model/stl"),
            },
        ).json()
        session_id = client.post(
            "/v1/sessions", json={"order_code": created["order_code"]}
        ).json()["session_id"]
        response = client.get(f"/v1/sessions/{session_id}/verdict")
    assert response.status_code == 200
    expected = {
        "verdict": "Unverifiable",
        "declared_at": "intake",
        "orientation_ambiguous": False,
        "regions": [],
        "reason_codes": ["intake_tolerance_below_k_sigma_floor:hole_A"],
    }
    # A-09/D-082: narrative is the deterministic template over this same payload, never a
    # model's.
    expected["narrative"] = template_narrative(VerdictInput.model_validate(expected))
    assert response.json() == expected


@pytest.mark.parametrize(
    ("residual", "raises", "reference_tier", "expected_flag"),
    [
        (0.0, False, "charuco_board", False),
        (0.03, False, "charuco_board", True),
        (None, False, "charuco_board", None),
        (0.03, True, "charuco_board", None),
        (0.03, False, "blank_sheet", None),
    ],
)
def test_r42_scale_agreement_after_registration(
    tmp_path, monkeypatch, caplog, residual, raises, reference_tier, expected_flag
):
    real_ambiguity = registration.check_ambiguity
    real_check = scale_agreement.check_scale_agreement
    calls = []

    def ambiguity_with_residual(*args, **kwargs):
        result = real_ambiguity(*args, **kwargs)
        return replace(result, best=replace(result.best, free_scale_residual=residual))

    def check(*, board_pitch_mm, cad_registration_residual):
        calls.append((board_pitch_mm, cad_registration_residual))
        # Simulate an agent message arriving after the measurement loaded its record.
        app.state.store.update_agent_message(
            session_id_for_check[0], fallback_reason=None, message="Latest guidance"
        )
        if raises:
            raise ValueError("synthetic scale-check failure")
        return real_check(
            board_pitch_mm=board_pitch_mm,
            cad_registration_residual=cad_registration_residual,
        )

    pipeline = _make_pipeline_runner()
    session_id_for_check = []

    def pipeline_with_reference(capture_dir, artifact_dir):
        session_id_for_check.append(capture_dir.parent.name)
        return {
            **pipeline(capture_dir, artifact_dir),
            "reference_tier_used": reference_tier,
        }

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(registration, "check_ambiguity", ambiguity_with_residual)
    monkeypatch.setattr(scale_agreement, "check_scale_agreement", check)
    app = service.create_app(
        tmp_path / "sessions",
        orders_dir=tmp_path / "orders",
        buyer_key="buyer-secret",
        pipeline_runner=pipeline_with_reference,
        vlm_supervisor=VlmSupervisor(inspector=_clean_vlm_reply),
    )
    with TestClient(app) as client:
        spec = {
            "features": [{"name": "overall_A", "kind": "overall", "tolerance_mm": 5.0}]
        }
        sid = _finish_a_session_with_cad(client, spec)
        response = client.get(f"/v1/sessions/{sid}/reconstruction")
        assert response.status_code == 200
        body = response.json()
        record = app.state.store.load(sid)

        assert body["status"] == "complete"
        assert "error" not in record["ambiguity"]
        assert record["reconstruction"]["glb_path"].endswith("_deviation.glb")
        assert client.get(f"/v1/sessions/{sid}/verdict").status_code == 200
        if residual is None or reference_tier != "charuco_board":
            assert calls == []
        else:
            assert calls == [(BOARD_SQUARE_MM, residual)]
            assert record["agent_message"]["message"] == "Latest guidance"

        if expected_flag is None:
            assert "scale_agreement" not in body
            assert "scale_agreement" not in record["reconstruction"]
        else:
            expected = {
                "sources_compared": ["board_pitch", "cad_registration_residual"],
                "flagged": expected_flag,
            }
            assert body["scale_agreement"] == expected
            assert record["reconstruction"]["scale_agreement"] == expected
        if raises:
            assert "Scale agreement failed for session" in caplog.text
