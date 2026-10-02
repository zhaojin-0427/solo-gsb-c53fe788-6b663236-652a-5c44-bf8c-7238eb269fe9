"""End-to-end tests for the Bayesian device fault diagnosis API."""

from __future__ import annotations

import itertools
import json
import time

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


# ---------------------------------------------------------------------------
# Fixtures: a small pump-line model (5 variables) and a 20-variable model
# ---------------------------------------------------------------------------

def pump_model() -> dict:
    """5 variables: 2 fault roots, 3 symptoms (one deterministic)."""
    return {
        "model_id": "pump-line-a",
        "version": "1.0.0",
        "variables": [
            {"name": "MotorFault", "states": ["ok", "fail"]},
            {"name": "BearingFault", "states": ["ok", "fail"]},
            {"name": "VibrationHigh", "states": ["no", "yes"]},
            {"name": "TempHigh", "states": ["no", "yes"]},
            {"name": "FlowLow", "states": ["no", "yes"]},
        ],
        "edges": [
            ["MotorFault", "VibrationHigh"],
            ["BearingFault", "VibrationHigh"],
            ["MotorFault", "TempHigh"],
            ["BearingFault", "FlowLow"],
        ],
        "cpts": {
            "MotorFault": {
                "parents": [],
                "rows": [{"when": {}, "probs": {"ok": 0.98, "fail": 0.02}}],
            },
            "BearingFault": {
                "parents": [],
                "rows": [{"when": {}, "probs": {"ok": 0.97, "fail": 0.03}}],
            },
            "VibrationHigh": {
                "parents": ["MotorFault", "BearingFault"],
                "rows": [
                    {"when": {"MotorFault": "ok", "BearingFault": "ok"},
                     "probs": {"no": 0.95, "yes": 0.05}},
                    {"when": {"MotorFault": "ok", "BearingFault": "fail"},
                     "probs": {"no": 0.30, "yes": 0.70}},
                    {"when": {"MotorFault": "fail", "BearingFault": "ok"},
                     "probs": {"no": 0.20, "yes": 0.80}},
                    {"when": {"MotorFault": "fail", "BearingFault": "fail"},
                     "probs": {"no": 0.02, "yes": 0.98}},
                ],
            },
            "TempHigh": {
                "parents": ["MotorFault"],
                "rows": [
                    {"when": {"MotorFault": "ok"}, "probs": {"no": 0.90, "yes": 0.10}},
                    {"when": {"MotorFault": "fail"}, "probs": {"no": 0.15, "yes": 0.85}},
                ],
            },
            # Deterministic CPT: FlowLow=yes is impossible unless BearingFault=fail.
            "FlowLow": {
                "parents": ["BearingFault"],
                "rows": [
                    {"when": {"BearingFault": "ok"}, "probs": {"no": 1.0, "yes": 0.0}},
                    {"when": {"BearingFault": "fail"}, "probs": {"no": 0.25, "yes": 0.75}},
                ],
            },
        },
    }


def big_model(n_faults: int = 5, n_symptoms: int = 15) -> dict:
    """20 variables: layered fault/symptom network with deterministic CPTs."""
    faults = [f"F{i}" for i in range(n_faults)]
    symptoms = [f"S{i}" for i in range(n_symptoms)]
    variables = [{"name": v, "states": ["ok", "fail"]} for v in faults]
    variables += [{"name": v, "states": ["no", "yes"]} for v in symptoms]

    edges: list[list[str]] = []
    parents: dict[str, list[str]] = {}
    for i, s in enumerate(symptoms):
        # Two distinct fault parents (offset 2 is coprime with 5) + chain.
        ps = [faults[i % n_faults], faults[(i + 2) % n_faults]]
        if i >= 1:
            ps.append(symptoms[i - 1])  # chain between symptoms
        parents[s] = ps
        for p in ps:
            edges.append([p, s])

    cpts: dict[str, dict] = {}
    for i, f in enumerate(faults):
        p_fail = 0.01 + 0.005 * i
        cpts[f] = {
            "parents": [],
            "rows": [{"when": {}, "probs": {"ok": 1.0 - p_fail, "fail": p_fail}}],
        }
    for i, s in enumerate(symptoms):
        ps = parents[s]
        rows = []
        # Fault parents are ok/fail, symptom parents are no/yes.
        p_states = [["ok", "fail"] if p in faults else ["no", "yes"] for p in ps]
        for combo in itertools.product(*p_states):
            bad = sum(
                1
                for p, v in zip(ps, combo)
                if (p in faults and v == "fail") or (p in symptoms and v == "yes")
            )
            p_yes = min(0.02 + 0.3 * bad, 0.97)
            rows.append(
                {"when": dict(zip(ps, combo)), "probs": {"no": 1.0 - p_yes, "yes": p_yes}}
            )
        cpts[s] = {"parents": ps, "rows": rows}

    return {
        "model_id": "line-20",
        "version": "2.1.0",
        "variables": variables,
        "edges": edges,
        "cpts": cpts,
    }


@pytest.fixture()
def client(tmp_path):
    app = create_app(str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def pump_client(client):
    resp = client.post("/models", json=pump_model())
    assert resp.status_code == 201, resp.text
    return client


# ---------------------------------------------------------------------------
# Brute-force reference inference (enumeration over the joint)
# ---------------------------------------------------------------------------

def brute_force(model: dict, evidence: dict, do: dict | None = None):
    """Returns (p_evidence, {var: {state: p}}, mpe_assignment, mpe_prob)."""
    do = do or {}
    names = [v["name"] for v in model["variables"]]
    states = {v["name"]: v["states"] for v in model["variables"]}
    cpts = model["cpts"]

    def joint_prob(assign: dict) -> float:
        p = 1.0
        for var in names:
            if var in do:
                if assign[var] != do[var]:
                    return 0.0
                continue
            row_probs = None
            for row in cpts[var]["rows"]:
                if all(assign[p] == s for p, s in row["when"].items()):
                    row_probs = row["probs"]
                    break
            p *= row_probs[assign[var]]
        return p

    totals: dict[str, dict[str, float]] = {
        v: {s: 0.0 for s in states[v]} for v in names if v not in evidence and v not in do
    }
    p_e = 0.0
    best_assign, best_p = None, -1.0
    free = [v for v in names if v not in evidence]
    for combo in itertools.product(*(states[v] for v in free)):
        assign = dict(zip(free, combo))
        assign.update(evidence)
        p = joint_prob(assign)
        if p == 0.0:
            continue
        p_e += p
        for var in totals:
            totals[var][assign[var]] += p
        if p > best_p:
            best_p, best_assign = p, dict(assign)
    posterior = {
        v: {s: totals[v][s] / p_e for s in totals[v]} for v in totals
    }
    mpe_assignment = {v: best_assign[v] for v in names if v not in evidence}
    return p_e, posterior, mpe_assignment, best_p / p_e


# ---------------------------------------------------------------------------
# Model registration
# ---------------------------------------------------------------------------

class TestModelRegistration:
    def test_register_ok(self, client):
        resp = client.post("/models", json=pump_model())
        assert resp.status_code == 201
        body = resp.json()
        assert body["model_id"] == "pump-line-a"
        assert body["variables"] == 5
        assert len(body["spec_hash"]) == 64

    def test_register_idempotent_same_content(self, client):
        r1 = client.post("/models", json=pump_model())
        r2 = client.post("/models", json=pump_model())
        assert r1.status_code == 201 and r2.status_code == 200
        assert r1.json()["spec_hash"] == r2.json()["spec_hash"]
        assert r1.json()["created_at"] == r2.json()["created_at"]

    def test_register_conflict_same_version(self, client):
        client.post("/models", json=pump_model())
        changed = pump_model()
        changed["cpts"]["MotorFault"]["rows"][0]["probs"] = {"ok": 0.9, "fail": 0.1}
        resp = client.post("/models", json=changed)
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "model_version_conflict"

    def test_row_not_normalized(self, client):
        bad = pump_model()
        bad["cpts"]["TempHigh"]["rows"][0]["probs"] = {"no": 0.5, "yes": 0.6}
        resp = client.post("/models", json=bad)
        assert resp.status_code == 422
        items = resp.json()["detail"]["items"]
        assert any("not normalized" in i["message"] for i in items)
        assert any(i["path"] == "cpts.TempHigh.rows[0]" for i in items)

    def test_cycle_rejected(self, client):
        bad = pump_model()
        bad["edges"].append(["FlowLow", "BearingFault"])
        resp = client.post("/models", json=bad)
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "model_invalid"
        assert any("acyclic" in i["message"] for i in resp.json()["detail"]["items"])

    def test_incomplete_cpt_rejected(self, client):
        bad = pump_model()
        bad["cpts"]["VibrationHigh"]["rows"] = bad["cpts"]["VibrationHigh"]["rows"][:3]
        resp = client.post("/models", json=bad)
        assert resp.status_code == 422
        assert any("incomplete CPT" in i["message"] for i in resp.json()["detail"]["items"])

    def test_cpt_parents_must_match_edges(self, client):
        bad = pump_model()
        bad["cpts"]["TempHigh"]["parents"] = ["BearingFault"]
        bad["cpts"]["TempHigh"]["rows"] = [
            {"when": {"BearingFault": "ok"}, "probs": {"no": 0.9, "yes": 0.1}},
            {"when": {"BearingFault": "fail"}, "probs": {"no": 0.2, "yes": 0.8}},
        ]
        resp = client.post("/models", json=bad)
        assert resp.status_code == 422
        assert any("do not match" in i["message"] for i in resp.json()["detail"]["items"])

    def test_variable_count_bounds(self, client):
        small = pump_model()
        small["variables"] = small["variables"][:4]  # only 4 variables
        resp = client.post("/models", json=small)
        assert resp.status_code == 422

    def test_get_model(self, pump_client):
        resp = pump_client.get("/models/pump-line-a/versions/1.0.0")
        assert resp.status_code == 200
        assert resp.json()["spec"]["model_id"] == "pump-line-a"
        assert pump_client.get("/models/nope/versions/9").status_code == 404


# ---------------------------------------------------------------------------
# Diagnosis: exactness vs brute force
# ---------------------------------------------------------------------------

def diag_request(key: str = "req-1", **overrides) -> dict:
    req = {
        "model_id": "pump-line-a",
        "model_version": "1.0.0",
        "idempotency_key": key,
        "evidence": {"VibrationHigh": "yes", "TempHigh": "yes"},
        "queries": {
            "posterior": ["MotorFault", "BearingFault"],
            "mpe": True,
            "interventions": [
                {"name": "replace_motor", "do": {"MotorFault": "ok"},
                 "observe": ["BearingFault", "FlowLow"]}
            ],
        },
    }
    req.update(overrides)
    return req


class TestDiagnosisExactness:
    def test_posterior_matches_brute_force(self, pump_client):
        req = diag_request()
        resp = pump_client.post("/diagnoses", json=req)
        assert resp.status_code == 201, resp.text
        body = resp.json()

        model = pump_model()
        p_e, post, mpe_assign, mpe_p = brute_force(model, req["evidence"])
        assert body["p_evidence"] == pytest.approx(p_e, abs=1e-9)
        for var, dist in body["posterior"].items():
            for state, p in dist.items():
                assert p == pytest.approx(post[var][state], abs=1e-9), (var, state)

    def test_mpe_matches_brute_force(self, pump_client):
        req = diag_request()
        body = pump_client.post("/diagnoses", json=req).json()
        _, _, mpe_assign, mpe_p = brute_force(pump_model(), req["evidence"])
        assert body["mpe"]["assignment"] == mpe_assign
        assert body["mpe"]["probability"] == pytest.approx(mpe_p, abs=1e-9)

    def test_intervention_matches_brute_force(self, pump_client):
        req = diag_request()
        body = pump_client.post("/diagnoses", json=req).json()
        iv = body["interventions"][0]
        assert iv["name"] == "replace_motor"

        p_e_do, post_do, _, _ = brute_force(
            pump_model(), req["evidence"], do={"MotorFault": "ok"}
        )
        assert iv["p_evidence"] == pytest.approx(p_e_do, abs=1e-9)
        for var, dist in iv["posterior"].items():
            for state, p in dist.items():
                assert p == pytest.approx(post_do[var][state], abs=1e-9), (var, state)
        # delta = posterior under do minus base posterior
        _, base, _, _ = brute_force(pump_model(), req["evidence"])
        for var, dist in iv["delta"].items():
            for state, d in dist.items():
                assert d == pytest.approx(post_do[var][state] - base[var][state], abs=1e-9)

    def test_default_posterior_covers_all_unobserved(self, pump_client):
        req = diag_request(key="req-default")
        req["queries"] = {"posterior": [], "mpe": False, "interventions": []}
        body = pump_client.post("/diagnoses", json=req).json()
        assert set(body["posterior"]) == {"MotorFault", "BearingFault", "FlowLow"}

    def test_elimination_summary_recorded(self, pump_client):
        body = pump_client.post("/diagnoses", json=diag_request(key="req-el")).json()
        elim = body["elimination"]
        assert elim["variables"] == 5
        assert elim["evidence_count"] == 2
        assert elim["total_duration_ms"] >= 0
        tasks = {t["task"] for t in elim["tasks"]}
        assert "p_evidence" in tasks
        assert "posterior:MotorFault" in tasks
        assert "mpe" in tasks
        assert any(t.startswith("intervention:replace_motor") for t in tasks)
        for t in elim["tasks"]:
            assert t["order"] and t["max_factor_entries"] >= 1


# ---------------------------------------------------------------------------
# Evidence contradiction
# ---------------------------------------------------------------------------

class TestEvidenceContradiction:
    def test_contradiction_located_and_not_saved(self, pump_client):
        # FlowLow=yes is impossible when BearingFault=ok (deterministic CPT).
        req = diag_request(key="req-conflict")
        req["evidence"] = {
            "BearingFault": "ok",
            "FlowLow": "yes",
            "VibrationHigh": "yes",
        }
        req["queries"] = {"posterior": ["MotorFault"], "mpe": True, "interventions": []}
        resp = pump_client.post("/diagnoses", json=req)
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail["error"] == "evidence_contradiction"
        conflict_vars = {item["variable"] for item in detail["items"]}
        # Minimal conflict is {BearingFault=ok, FlowLow=yes}; VibrationHigh is irrelevant.
        assert conflict_vars == {"BearingFault", "FlowLow"}
        assert all(item["path"].startswith("evidence.") for item in detail["items"])

        # Nothing was persisted: replaying the same key computes afresh (still 422),
        # and no diagnosis id exists to fetch.
        again = pump_client.post("/diagnoses", json=req)
        assert again.status_code == 422

    def test_single_zero_prior_item(self, pump_client):
        req = diag_request(key="req-conflict-1")
        req["evidence"] = {"BearingFault": "ok", "FlowLow": "yes"}
        req["queries"] = {"posterior": ["MotorFault"], "mpe": False, "interventions": []}
        resp = pump_client.post("/diagnoses", json=req)
        assert resp.status_code == 422
        assert {i["variable"] for i in resp.json()["detail"]["items"]} == {
            "BearingFault",
            "FlowLow",
        }


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_replay_byte_identical(self, pump_client):
        req = diag_request(key="req-idem")
        r1 = pump_client.post("/diagnoses", json=req)
        r2 = pump_client.post("/diagnoses", json=req)
        assert r1.status_code == 201 and r2.status_code == 200
        assert r2.headers["X-Idempotent-Replay"] == "true"
        assert r1.content == r2.content  # byte-identical

    def test_get_by_id_returns_same_bytes(self, pump_client):
        req = diag_request(key="req-get")
        r1 = pump_client.post("/diagnoses", json=req)
        diag_id = r1.json()["diagnosis_id"]
        r2 = pump_client.get(f"/diagnoses/{diag_id}")
        assert r2.status_code == 200
        assert r2.content == r1.content

    def test_key_reuse_with_different_payload_conflicts(self, pump_client):
        req = diag_request(key="req-idem-x")
        pump_client.post("/diagnoses", json=req)
        changed = diag_request(key="req-idem-x")
        changed["evidence"] = {"VibrationHigh": "no"}
        resp = pump_client.post("/diagnoses", json=changed)
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "idempotency_key_conflict"

    def test_same_key_different_model_version_is_independent(self, pump_client):
        other = pump_model()
        other["version"] = "1.0.1"
        pump_client.post("/models", json=other)
        req = diag_request(key="req-shared")
        r1 = pump_client.post("/diagnoses", json=req)
        req2 = diag_request(key="req-shared")
        req2["model_version"] = "1.0.1"
        r2 = pump_client.post("/diagnoses", json=req2)
        assert r1.status_code == 201 and r2.status_code == 201
        assert r1.json()["diagnosis_id"] != r2.json()["diagnosis_id"]

    def test_unknown_model_404(self, client):
        resp = client.post("/diagnoses", json=diag_request())
        assert resp.status_code == 404
        assert resp.json()["detail"]["error"] == "model_not_found"

    def test_unknown_diagnosis_404(self, client):
        assert client.get("/diagnoses/dg_missing").status_code == 404


# ---------------------------------------------------------------------------
# Validation of diagnosis requests
# ---------------------------------------------------------------------------

class TestDiagnosisValidation:
    def test_unknown_evidence_state(self, pump_client):
        req = diag_request(key="req-bad-state")
        req["evidence"] = {"VibrationHigh": "maybe"}
        resp = pump_client.post("/diagnoses", json=req)
        assert resp.status_code == 422
        items = resp.json()["detail"]["items"]
        assert any(i["path"] == "evidence.VibrationHigh" for i in items)

    def test_do_overlaps_evidence(self, pump_client):
        req = diag_request(key="req-do-overlap")
        req["evidence"] = {"MotorFault": "ok"}
        req["queries"]["interventions"] = [
            {"name": "x", "do": {"MotorFault": "fail"}}
        ]
        resp = pump_client.post("/diagnoses", json=req)
        assert resp.status_code == 422
        assert any("disjoint" in i["message"] for i in resp.json()["detail"]["items"])

    def test_posterior_of_evidence_rejected(self, pump_client):
        req = diag_request(key="req-post-ev")
        req["queries"]["posterior"] = ["VibrationHigh"]
        resp = pump_client.post("/diagnoses", json=req)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 20-variable performance
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_twenty_variables_under_two_seconds(self, client):
        model = big_model()
        resp = client.post("/models", json=model)
        assert resp.status_code == 201, resp.text

        req = {
            "model_id": "line-20",
            "model_version": "2.1.0",
            "idempotency_key": "perf-1",
            "evidence": {"S3": "yes", "S7": "yes", "S11": "yes"},
            "queries": {
                "posterior": [f"F{i}" for i in range(5)],
                "mpe": True,
                "interventions": [
                    {"name": "fix_f0", "do": {"F0": "ok"}, "observe": ["S4", "S8"]}
                ],
            },
        }
        start = time.perf_counter()
        resp = client.post("/diagnoses", json=req)
        elapsed = time.perf_counter() - start
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert elapsed < 2.0, f"diagnosis took {elapsed:.3f}s"
        assert body["elimination"]["variables"] == 20
        assert len(body["posterior"]) == 5
        assert set(body["mpe"]["assignment"]) == {
            v["name"] for v in model["variables"]
        } - set(req["evidence"])
        # MPE assignment values are valid states.
        states = {v["name"]: v["states"] for v in model["variables"]}
        for var, val in body["mpe"]["assignment"].items():
            assert val in states[var]

    def test_replay_of_big_model_is_byte_identical(self, client):
        client.post("/models", json=big_model())
        req = {
            "model_id": "line-20",
            "model_version": "2.1.0",
            "idempotency_key": "perf-2",
            "evidence": {"S0": "yes"},
            "queries": {"posterior": ["F0", "F1"], "mpe": True, "interventions": []},
        }
        r1 = client.post("/diagnoses", json=req)
        r2 = client.post("/diagnoses", json=req)
        assert r1.content == r2.content
        assert json.loads(r2.content)["elimination"]["variables"] == 20
