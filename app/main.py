"""FastAPI application: model registration, diagnosis, result query."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import __version__
from .bayes import BayesianNetwork, ElimStats
from .db import Database
from .network import SpecErrorGroup, validate_and_build
from .schemas import DiagnosisRequest, ModelSpec
from .util import canonical_json, round12, sha256_hex

DEFAULT_DB_PATH = os.environ.get("BAYES_DB_PATH", "./data/bayesdiag.db")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, items: Optional[list] = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.items = items or []


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_response(body: str, status: int = 200, headers: Optional[dict] = None) -> Response:
    return Response(content=body, status_code=status, media_type="application/json", headers=headers)


def create_app(db_path: Optional[str] = None) -> FastAPI:
    db = Database(db_path or DEFAULT_DB_PATH)
    app = FastAPI(title="Bayesian Device Fault Diagnosis API", version=__version__)
    app.state.db = db
    # Parsed networks are immutable per (model_id, version); cache them.
    net_cache: dict[tuple[str, str], BayesianNetwork] = {}

    # -- error handling ------------------------------------------------------

    @app.exception_handler(ApiError)
    async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status,
            content={
                "detail": {
                    "error": exc.code,
                    "message": exc.message,
                    "items": exc.items,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        items = []
        for err in exc.errors():
            loc = [str(p) for p in err["loc"] if p not in ("body", "query", "path")]
            items.append({"path": ".".join(loc), "message": err["msg"]})
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "error": "validation_error",
                    "message": "request failed schema validation",
                    "items": items,
                }
            },
        )

    # -- helpers ---------------------------------------------------------------

    def _load_network(model_id: str, version: str) -> BayesianNetwork:
        key = (model_id, version)
        if key not in net_cache:
            row = db.get_model(model_id, version)
            if row is None:
                raise ApiError(
                    404,
                    "model_not_found",
                    f"model '{model_id}' version '{version}' is not registered",
                )
            spec = ModelSpec.model_validate_json(row["spec_json"])
            net_cache[key] = validate_and_build(spec)
        return net_cache[key]

    # -- routes ------------------------------------------------------------------

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.post("/models", status_code=201)
    def register_model(spec: ModelSpec) -> Response:
        try:
            validate_and_build(spec)
        except SpecErrorGroup as group:
            raise ApiError(
                422,
                "model_invalid",
                f"model spec has {len(group.errors)} error(s)",
                [{"path": e.path, "message": e.message} for e in group.errors],
            )

        canonical = canonical_json(spec.model_dump(mode="json"))
        spec_hash = sha256_hex(canonical)

        existing = db.get_model(spec.model_id, spec.version)
        if existing is not None:
            if existing["spec_hash"] != spec_hash:
                raise ApiError(
                    409,
                    "model_version_conflict",
                    f"model '{spec.model_id}' version '{spec.version}' already exists"
                    " with different content",
                )
            body = canonical_json(
                {
                    "model_id": spec.model_id,
                    "version": spec.version,
                    "spec_hash": spec_hash,
                    "variables": len(spec.variables),
                    "edges": len(spec.edges),
                    "created_at": existing["created_at"],
                    "replayed": True,
                }
            )
            return _json_response(body, status=200)

        created_at = _utc_now()
        try:
            db.insert_model(spec.model_id, spec.version, spec_hash, canonical, created_at)
        except sqlite3.IntegrityError:
            # Lost a registration race; the stored row is authoritative.
            return register_model(spec)
        net_cache.pop((spec.model_id, spec.version), None)
        body = canonical_json(
            {
                "model_id": spec.model_id,
                "version": spec.version,
                "spec_hash": spec_hash,
                "variables": len(spec.variables),
                "edges": len(spec.edges),
                "created_at": created_at,
                "replayed": False,
            }
        )
        return _json_response(body, status=201)

    @app.get("/models/{model_id}/versions/{version}")
    def get_model(model_id: str, version: str) -> Response:
        row = db.get_model(model_id, version)
        if row is None:
            raise ApiError(
                404,
                "model_not_found",
                f"model '{model_id}' version '{version}' is not registered",
            )
        body = canonical_json(
            {
                "model_id": row["model_id"],
                "version": row["version"],
                "spec_hash": row["spec_hash"],
                "created_at": row["created_at"],
                "spec": json.loads(row["spec_json"]),
            }
        )
        return _json_response(body)

    @app.post("/diagnoses", status_code=201)
    def submit_diagnosis(req: DiagnosisRequest) -> Response:
        net = _load_network(req.model_id, req.model_version)

        request_json = canonical_json(req.model_dump(mode="json"))
        request_hash = sha256_hex(request_json)
        diagnosis_id = "dg_" + sha256_hex(
            f"{req.model_id}|{req.model_version}|{req.idempotency_key}"
        )[:24]

        # Idempotent replay: same model version + key + payload -> stored bytes.
        existing = db.get_diagnosis_by_key(req.model_id, req.model_version, req.idempotency_key)
        if existing is not None:
            if existing["request_hash"] != request_hash:
                raise ApiError(
                    409,
                    "idempotency_key_conflict",
                    "idempotency_key was already used with a different request body",
                )
            return _json_response(
                existing["response_json"],
                status=200,
                headers={"X-Idempotent-Replay": "true"},
            )

        _validate_diagnosis(net, req)
        response_obj, elimination = _run_diagnosis(net, req, diagnosis_id)

        body = canonical_json(response_obj)
        try:
            db.insert_diagnosis(
                diagnosis_id=diagnosis_id,
                model_id=req.model_id,
                version=req.model_version,
                idempotency_key=req.idempotency_key,
                request_hash=request_hash,
                request_json=request_json,
                response_json=body,
                elimination_json=canonical_json(elimination),
                created_at=response_obj["created_at"],
            )
        except sqlite3.IntegrityError:
            # Concurrent identical submission: serve the stored row.
            stored = db.get_diagnosis_by_key(req.model_id, req.model_version, req.idempotency_key)
            if stored is not None and stored["request_hash"] == request_hash:
                return _json_response(
                    stored["response_json"],
                    status=200,
                    headers={"X-Idempotent-Replay": "true"},
                )
            raise ApiError(409, "idempotency_key_conflict", "idempotency key conflict")

        return _json_response(body, status=201, headers={"X-Idempotent-Replay": "false"})

    @app.get("/diagnoses/{diagnosis_id}")
    def get_diagnosis(diagnosis_id: str) -> Response:
        row = db.get_diagnosis_by_id(diagnosis_id)
        if row is None:
            raise ApiError(404, "diagnosis_not_found", f"diagnosis '{diagnosis_id}' not found")
        return _json_response(row["response_json"], status=200)

    return app


# ---------------------------------------------------------------------------
# Diagnosis validation & computation
# ---------------------------------------------------------------------------

def _validate_diagnosis(net: BayesianNetwork, req: DiagnosisRequest) -> None:
    items: list[dict] = []

    for var, state in req.evidence.items():
        if var not in net.states:
            items.append({"path": f"evidence.{var}", "message": f"unknown variable '{var}'"})
        elif state not in net.state_idx[var]:
            items.append(
                {
                    "path": f"evidence.{var}",
                    "message": f"unknown state '{state}'; valid: {net.states[var]}",
                }
            )

    seen_post: set[str] = set()
    for i, var in enumerate(req.queries.posterior):
        if var not in net.states:
            items.append(
                {"path": f"queries.posterior[{i}]", "message": f"unknown variable '{var}'"}
            )
        elif var in req.evidence:
            items.append(
                {
                    "path": f"queries.posterior[{i}]",
                    "message": f"'{var}' is observed evidence; its posterior is degenerate",
                }
            )
        elif var in seen_post:
            items.append(
                {"path": f"queries.posterior[{i}]", "message": f"duplicate variable '{var}'"}
            )
        seen_post.add(var)

    seen_names: set[str] = set()
    for i, iv in enumerate(req.queries.interventions):
        ipath = f"queries.interventions[{i}]"
        if iv.name in seen_names:
            items.append({"path": f"{ipath}.name", "message": f"duplicate intervention name '{iv.name}'"})
        seen_names.add(iv.name)
        for var, state in iv.do.items():
            if var not in net.states:
                items.append({"path": f"{ipath}.do.{var}", "message": f"unknown variable '{var}'"})
            elif state not in net.state_idx[var]:
                items.append(
                    {
                        "path": f"{ipath}.do.{var}",
                        "message": f"unknown state '{state}'; valid: {net.states[var]}",
                    }
                )
            if var in req.evidence:
                items.append(
                    {
                        "path": f"{ipath}.do.{var}",
                        "message": f"'{var}' is also fixed by evidence; do() and evidence must be disjoint",
                    }
                )
        for j, var in enumerate(iv.observe or []):
            if var not in net.states:
                items.append(
                    {"path": f"{ipath}.observe[{j}]", "message": f"unknown variable '{var}'"}
                )
            elif var in iv.do:
                items.append(
                    {
                        "path": f"{ipath}.observe[{j}]",
                        "message": f"'{var}' is intervened; its posterior under do() is degenerate",
                    }
                )
            elif var in req.evidence:
                items.append(
                    {
                        "path": f"{ipath}.observe[{j}]",
                        "message": f"'{var}' is observed evidence; its posterior is degenerate",
                    }
                )

    if items:
        raise ApiError(422, "diagnosis_invalid", f"diagnosis request has {len(items)} error(s)", items)


def _run_diagnosis(
    net: BayesianNetwork, req: DiagnosisRequest, diagnosis_id: str
) -> tuple[dict, dict]:
    stats: list[ElimStats] = []
    started = time.perf_counter()
    evidence = dict(req.evidence)

    # Evidence contradiction: locate the minimal conflicting subset and
    # refuse the request without persisting anything.
    p_e = net.p_evidence(evidence, stats=stats)
    if p_e == 0.0:
        conflict = net.minimal_conflict(evidence)
        raise ApiError(
            422,
            "evidence_contradiction",
            "evidence has zero probability under the model; nothing was stored",
            [
                {
                    "path": f"evidence.{var}",
                    "variable": var,
                    "state": state,
                    "message": "part of a minimal zero-probability evidence set",
                }
                for var, state in conflict.items()
            ],
        )

    query_vars = req.queries.posterior or [v for v in net.variables if v not in evidence]

    # Base posteriors cover the query set plus every intervention's observe set.
    observe_sets: list[tuple[Any, list[str]]] = []
    extra: list[str] = []
    for iv in req.queries.interventions:
        observe = iv.observe if iv.observe is not None else [
            v for v in query_vars if v not in iv.do
        ]
        observe_sets.append((iv, observe))
        for v in observe:
            if v not in query_vars and v not in extra:
                extra.append(v)
    base = net.posterior(evidence, query_vars + extra, stats=stats)

    result: dict[str, Any] = {
        "diagnosis_id": diagnosis_id,
        "model_id": req.model_id,
        "model_version": req.model_version,
        "idempotency_key": req.idempotency_key,
        "evidence": evidence,
        "p_evidence": round12(p_e),
        "posterior": {
            q: {s: round12(p) for s, p in base[q].items()} for q in query_vars
        },
        "mpe": None,
        "interventions": [],
        "created_at": _utc_now(),
    }

    if req.queries.mpe:
        assignment, prob = net.mpe(evidence, stats=stats, p_e=p_e)
        result["mpe"] = {
            "assignment": assignment,
            "probability": round12(prob),
        }

    for idx, (iv, observe) in enumerate(observe_sets):
        p_e_do = net.p_evidence(
            evidence, do=iv.do, stats=stats, task=f"intervention:{iv.name}:p_evidence"
        )
        if p_e_do == 0.0:
            raise ApiError(
                422,
                "evidence_contradiction",
                f"evidence has zero probability under do() of intervention '{iv.name}';"
                " nothing was stored",
                [
                    {
                        "path": f"queries.interventions[{idx}].do.{var}",
                        "variable": var,
                        "state": state,
                        "message": "intervention conflicts with the evidence",
                    }
                    for var, state in iv.do.items()
                ],
            )
        post_do = net.posterior(
            evidence, observe, do=iv.do, stats=stats, prefix=f"intervention:{iv.name}"
        )
        result["interventions"].append(
            {
                "name": iv.name,
                "do": dict(iv.do),
                "p_evidence": round12(p_e_do),
                "posterior": {q: {s: round12(p) for s, p in post_do[q].items()} for q in observe},
                "delta": {
                    q: {s: round12(post_do[q][s] - base[q][s]) for s in net.states[q]}
                    for q in observe
                },
            }
        )

    elimination = {
        "variables": len(net.variables),
        "evidence_count": len(evidence),
        "tasks": [st.to_dict() for st in stats],
        "total_duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    result["elimination"] = elimination
    return result, elimination


app = create_app()
