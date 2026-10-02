"""Model-spec validation and construction of the internal BayesianNetwork.

Every violation is reported with a locatable JSON-ish path so clients can
pinpoint the offending row / edge / variable.
"""

from __future__ import annotations

import numpy as np

from .bayes import BayesianNetwork
from .schemas import ModelSpec

ROW_SUM_TOL = 1e-6
MAX_FAMILY_ENTRIES = 2_000_000  # card(var) * prod(card(parents)) safety cap


class SpecError(Exception):
    """A single model-spec violation with a locatable path."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(message)
        self.path = path
        self.message = message


def _check_acyclic(variables: list[str], edges: list[tuple[str, str]]) -> None:
    children: dict[str, list[str]] = {v: [] for v in variables}
    indegree: dict[str, int] = {v: 0 for v in variables}
    for src, dst in edges:
        children[src].append(dst)
        indegree[dst] += 1
    # Kahn's algorithm with deterministic (sorted) ready set.
    ready = sorted(v for v in variables if indegree[v] == 0)
    seen = 0
    while ready:
        node = ready.pop(0)
        seen += 1
        for nxt in sorted(children[node]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
        ready.sort()
    if seen != len(variables):
        remaining = sorted(v for v in variables if indegree[v] > 0)
        raise SpecError(
            "edges",
            f"graph must be acyclic; cycle involves variables: {remaining}",
        )


def validate_and_build(spec: ModelSpec) -> BayesianNetwork:
    errors: list[SpecError] = []

    # -- variables & states --------------------------------------------------
    names = [v.name for v in spec.variables]
    if len(set(names)) != len(names):
        seen: set[str] = set()
        for i, v in enumerate(spec.variables):
            if v.name in seen:
                errors.append(
                    SpecError(f"variables[{i}].name", f"duplicate variable '{v.name}'")
                )
            seen.add(v.name)
    states: dict[str, list[str]] = {}
    for i, v in enumerate(spec.variables):
        for j, s in enumerate(v.states):
            if not s:
                errors.append(SpecError(f"variables[{i}].states[{j}]", "empty state"))
        if len(set(v.states)) != len(v.states):
            errors.append(
                SpecError(f"variables[{i}].states", f"duplicate states for '{v.name}'")
            )
        states[v.name] = list(v.states)

    # -- edges ---------------------------------------------------------------
    parents: dict[str, list[str]] = {v: [] for v in names}
    seen_edges: set[tuple[str, str]] = set()
    for i, (src, dst) in enumerate(spec.edges):
        if src == dst:
            errors.append(SpecError(f"edges[{i}]", f"self-loop on '{src}'"))
        for endpoint in (src, dst):
            if endpoint not in states:
                errors.append(
                    SpecError(f"edges[{i}]", f"unknown variable '{endpoint}'")
                )
        if (src, dst) in seen_edges:
            errors.append(SpecError(f"edges[{i}]", f"duplicate edge {src!r}->{dst!r}"))
        seen_edges.add((src, dst))
        if dst in parents and src in states:
            parents[dst].append(src)
    try:
        _check_acyclic(names, [(s, d) for s, d in spec.edges if s in states and d in states])
    except SpecError as exc:
        errors.append(exc)

    # -- CPTs ----------------------------------------------------------------
    cpt_arrays: dict[str, np.ndarray] = {}
    for var in names:
        if var not in spec.cpts:
            errors.append(SpecError(f"cpts", f"missing CPT for variable '{var}'"))
    for var, cpt in spec.cpts.items():
        path = f"cpts.{var}"
        if var not in states:
            errors.append(SpecError(path, f"CPT declared for unknown variable '{var}'"))
            continue
        edge_parents = sorted(parents[var])
        if sorted(cpt.parents) != edge_parents:
            errors.append(
                SpecError(
                    f"{path}.parents",
                    f"CPT parents {sorted(cpt.parents)} do not match edges {edge_parents}",
                )
            )
            continue
        p_order = list(cpt.parents)
        var_states = states[var]

        # Row coverage: exactly one row per parent-state combination.
        seen_rows: set[tuple[str, ...]] = set()
        for r, row in enumerate(cpt.rows):
            rpath = f"{path}.rows[{r}]"
            if set(row.when.keys()) != set(p_order):
                errors.append(
                    SpecError(
                        f"{rpath}.when",
                        f"keys {sorted(row.when.keys())} != parents {p_order}",
                    )
                )
                continue
            bad = [
                f"{rpath}.when.{p}"
                for p in p_order
                if row.when[p] not in states.get(p, [])
            ]
            if bad:
                errors.append(SpecError(bad[0], "unknown parent state"))
                continue
            key = tuple(row.when[p] for p in p_order)
            if key in seen_rows:
                errors.append(SpecError(rpath, f"duplicate row for {dict(zip(p_order, key))}"))
                continue
            seen_rows.add(key)

            if set(row.probs.keys()) != set(var_states):
                errors.append(
                    SpecError(
                        f"{rpath}.probs",
                        f"states {sorted(row.probs.keys())} != variable states {sorted(var_states)}",
                    )
                )
                continue
            vals = [row.probs[s] for s in var_states]
            for s, p in zip(var_states, vals):
                if not (0.0 <= p <= 1.0):
                    errors.append(
                        SpecError(f"{rpath}.probs.{s}", f"probability {p} out of [0, 1]")
                    )
            total = sum(vals)
            if abs(total - 1.0) > ROW_SUM_TOL:
                errors.append(
                    SpecError(
                        rpath,
                        f"row not normalized: probabilities sum to {total!r} (tol {ROW_SUM_TOL})",
                    )
                )

        expected = 1
        for p in p_order:
            expected *= len(states[p])
        if len(seen_rows) != expected and not any(
            e.path.startswith(path) for e in errors
        ):
            missing = expected - len(seen_rows)
            errors.append(
                SpecError(
                    path,
                    f"incomplete CPT: {len(seen_rows)}/{expected} parent combinations covered"
                    + (f" ({missing} missing)" if missing > 0 else ""),
                )
            )

        # Build the dense CPT array (only when rows are complete & valid).
        entries = len(var_states) * expected
        if entries > MAX_FAMILY_ENTRIES:
            errors.append(
                SpecError(path, f"family too large: {entries} CPT entries")
            )
            continue
        if len(seen_rows) == expected and not any(e.path.startswith(path) for e in errors):
            shape = (len(var_states),) + tuple(len(states[p]) for p in p_order)
            arr = np.zeros(shape, dtype=np.float64)
            idx_of = {p: {s: i for i, s in enumerate(states[p])} for p in p_order}
            for row in cpt.rows:
                pidx = tuple(idx_of[p][row.when[p]] for p in p_order)
                arr[(slice(None),) + pidx] = [row.probs[s] for s in var_states]
            cpt_arrays[var] = arr

    if errors:
        raise SpecErrorGroup(errors)

    return BayesianNetwork(
        variables=names,
        states=states,
        parents={v: list(spec.cpts[v].parents) for v in names},
        cpt_arrays=cpt_arrays,
    )


class SpecErrorGroup(Exception):
    def __init__(self, errors: list[SpecError]) -> None:
        super().__init__(f"{len(errors)} spec error(s)")
        self.errors = errors
