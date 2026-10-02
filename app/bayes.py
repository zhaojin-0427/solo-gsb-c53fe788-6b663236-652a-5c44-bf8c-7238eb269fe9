"""Exact inference over discrete Bayesian networks via variable elimination.

All inference is deterministic: factors are aligned to canonical variable
orders, the elimination heuristic breaks ties lexicographically, and NumPy
operations are order-stable, so identical inputs always produce identical
output bytes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Factors
# ---------------------------------------------------------------------------

@dataclass
class Factor:
    """A non-negative factor over an ordered set of discrete variables.

    ``vars[i]`` labels axis ``i`` of ``values``; ``values.shape[i]`` is the
    cardinality of that variable.
    """

    vars: list[str]
    values: np.ndarray

    @property
    def size(self) -> int:
        return int(self.values.size)


def _aligned_values(f: Factor, target_vars: list[str]) -> np.ndarray:
    """Re-express ``f`` over ``target_vars`` (broadcastable, size-1 axes added)."""
    vals = f.values
    present = [v for v in target_vars if v in f.vars]
    if f.vars != present:
        perm = [f.vars.index(v) for v in present]
        vals = np.transpose(vals, axes=perm) if vals.ndim else vals
    dims = iter(vals.shape)
    shape = [next(dims) if v in f.vars else 1 for v in target_vars]
    return vals.reshape(shape)


def multiply(a: Factor, b: Factor) -> Factor:
    vars_ = a.vars + [v for v in b.vars if v not in a.vars]
    return Factor(vars_, _aligned_values(a, vars_) * _aligned_values(b, vars_))


def restrict(f: Factor, var: str, state_idx: int) -> Factor:
    """Condition a factor on ``var == state_idx`` (evidence / do reduction)."""
    ax = f.vars.index(var)
    return Factor(
        [v for v in f.vars if v != var],
        np.take(f.values, state_idx, axis=ax),
    )


def sum_out(f: Factor, var: str) -> Factor:
    ax = f.vars.index(var)
    return Factor(
        [v for v in f.vars if v != var],
        f.values.sum(axis=ax),
    )


def max_out(f: Factor, var: str) -> tuple[Factor, Factor]:
    """Maximize ``var`` out; also return the argmax indices as a factor."""
    ax = f.vars.index(var)
    rest = [v for v in f.vars if v != var]
    return (
        Factor(rest, f.values.max(axis=ax)),
        Factor(rest, f.values.argmax(axis=ax).astype(np.int64)),
    )


# ---------------------------------------------------------------------------
# Elimination bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class ElimStats:
    """Summary of one variable-elimination run (recorded per task)."""

    task: str
    order: list[str] = field(default_factory=list)
    multiplies: int = 0
    max_factor_vars: int = 0
    max_factor_entries: int = 0
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "order": self.order,
            "eliminated": len(self.order),
            "multiplies": self.multiplies,
            "max_factor_vars": self.max_factor_vars,
            "max_factor_entries": self.max_factor_entries,
            "duration_ms": round(self.duration_ms, 3),
        }


def _choose_elim_var(
    factors: list[Factor], remaining: set[str], cards: dict[str, int]
) -> str:
    """Min-fill heuristic; ties broken by scope weight then variable name."""
    best_key = None
    best_var = None
    for var in sorted(remaining):
        scope: set[str] = set()
        for f in factors:
            if var in f.vars:
                scope.update(f.vars)
        scope.discard(var)
        ordered = sorted(scope)
        fill = 0
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                if not any(a in f.vars and b in f.vars for f in factors):
                    fill += 1
        weight = sum(math.log2(cards[u]) for u in ordered)
        key = (fill, weight, var)
        if best_key is None or key < best_key:
            best_key, best_var = key, var
    return best_var  # type: ignore[return-value]


def _join(factors: list[Factor], stats: ElimStats) -> Factor:
    prod = factors[0]
    for f in factors[1:]:
        prod = multiply(prod, f)
        stats.multiplies += 1
    stats.max_factor_vars = max(stats.max_factor_vars, len(prod.vars))
    stats.max_factor_entries = max(stats.max_factor_entries, prod.size)
    return prod


def eliminate(
    factors: list[Factor], elim_vars: list[str], cards: dict[str, int], task: str
) -> tuple[list[Factor], ElimStats]:
    """Sum out ``elim_vars``; return the remaining factors and run stats."""
    stats = ElimStats(task=task)
    start = time.perf_counter()
    factors = list(factors)
    remaining = set(elim_vars)
    while remaining:
        var = _choose_elim_var(factors, remaining, cards)
        remaining.discard(var)
        stats.order.append(var)
        involved = [f for f in factors if var in f.vars]
        factors = [f for f in factors if var not in f.vars]
        prod = _join(involved, stats)
        factors.append(sum_out(prod, var))
    stats.duration_ms = (time.perf_counter() - start) * 1000.0
    return factors, stats


def eliminate_max(
    factors: list[Factor], elim_vars: list[str], cards: dict[str, int], task: str
) -> tuple[list[Factor], dict[str, Factor], ElimStats]:
    """Max-product elimination; returns remaining factors and back-pointers."""
    stats = ElimStats(task=task)
    start = time.perf_counter()
    factors = list(factors)
    back: dict[str, Factor] = {}
    remaining = set(elim_vars)
    while remaining:
        var = _choose_elim_var(factors, remaining, cards)
        remaining.discard(var)
        stats.order.append(var)
        involved = [f for f in factors if var in f.vars]
        factors = [f for f in factors if var not in f.vars]
        prod = _join(involved, stats)
        maxed, arg = max_out(prod, var)
        back[var] = arg
        factors.append(maxed)
    stats.duration_ms = (time.perf_counter() - start) * 1000.0
    return factors, back, stats


def join_all(factors: list[Factor]) -> Factor:
    if not factors:
        return Factor([], np.array(1.0))
    prod = factors[0]
    for f in factors[1:]:
        prod = multiply(prod, f)
    return prod


# ---------------------------------------------------------------------------
# Bayesian network
# ---------------------------------------------------------------------------

class BayesianNetwork:
    """Validated discrete Bayesian network with exact-inference routines."""

    def __init__(
        self,
        variables: list[str],
        states: dict[str, list[str]],
        parents: dict[str, list[str]],
        cpt_arrays: dict[str, np.ndarray],
    ) -> None:
        self.variables = list(variables)
        self.states = states
        self.parents = parents
        self.cards = {v: len(states[v]) for v in self.variables}
        self.state_idx = {
            v: {s: i for i, s in enumerate(states[v])} for v in self.variables
        }
        # One factor per variable: axes = [var] + parents (CPT order).
        self.factors: dict[str, Factor] = {
            v: Factor([v] + parents[v], cpt_arrays[v]) for v in self.variables
        }

    # -- factor preparation -------------------------------------------------

    def _joint_factors(self, do: dict[str, str] | None = None) -> list[Factor]:
        """CPT factors, with do-intervened variables' mechanisms removed and
        their values pinned in every other factor (graph mutilation)."""
        do = do or {}
        out: list[Factor] = []
        for var in self.variables:
            if var in do:
                continue  # replace P(var|parents) by the intervention point mass
            f = self.factors[var]
            for dv, ds in do.items():
                if dv in f.vars:
                    f = restrict(f, dv, self.state_idx[dv][ds])
            out.append(f)
        return out

    def _with_evidence(
        self, evidence: dict[str, str], do: dict[str, str] | None = None
    ) -> list[Factor]:
        factors = self._joint_factors(do)
        out: list[Factor] = []
        for f in factors:
            for var, state in evidence.items():
                if var in f.vars:
                    f = restrict(f, var, self.state_idx[var][state])
            out.append(f)
        return out

    # -- queries -------------------------------------------------------------

    def p_evidence(
        self,
        evidence: dict[str, str],
        do: dict[str, str] | None = None,
        stats: list[ElimStats] | None = None,
        task: str = "p_evidence",
    ) -> float:
        """P(evidence | do) by full elimination. Exactly 0.0 on contradiction."""
        if not evidence:
            return 1.0
        factors = self._with_evidence(evidence, do)
        elim = [v for v in self.variables if v not in evidence and v not in (do or {})]
        remaining, st = eliminate(factors, elim, self.cards, task)
        if stats is not None:
            stats.append(st)
        return float(join_all(remaining).values)

    def posterior(
        self,
        evidence: dict[str, str],
        query_vars: list[str],
        do: dict[str, str] | None = None,
        stats: list[ElimStats] | None = None,
        prefix: str = "posterior",
    ) -> dict[str, dict[str, float]]:
        """Exact posterior marginals P(var | evidence, do) per query variable."""
        factors = self._with_evidence(evidence, do)
        pinned = set(evidence) | set(do or {})
        result: dict[str, dict[str, float]] = {}
        for q in query_vars:
            elim = [v for v in self.variables if v != q and v not in pinned]
            remaining, st = eliminate(factors, elim, self.cards, f"{prefix}:{q}")
            if stats is not None:
                stats.append(st)
            joint = join_all(remaining)
            vals = joint.values.reshape(-1)
            total = float(vals.sum())
            probs = vals / total if total > 0 else np.full_like(vals, np.nan)
            result[q] = {
                state: float(probs[i]) for i, state in enumerate(self.states[q])
            }
        return result

    def mpe(
        self,
        evidence: dict[str, str],
        stats: list[ElimStats] | None = None,
        p_e: float | None = None,
    ) -> tuple[dict[str, str], float]:
        """Most probable explanation over all non-evidence variables.

        Returns (assignment, P(assignment | evidence)).
        """
        factors = self._with_evidence(evidence)
        elim = [v for v in self.variables if v not in evidence]
        remaining, back, st = eliminate_max(factors, elim, self.cards, "mpe")
        if stats is not None:
            stats.append(st)
        p_star = float(join_all(remaining).values)  # P(mpe_assignment, evidence)

        assigned: dict[str, int] = {}
        for var in reversed(st.order):
            arg = back[var]
            key = tuple(assigned[v] for v in arg.vars)
            assigned[var] = int(arg.values[key]) if arg.vars else int(arg.values)

        if p_e is None:
            p_e = self.p_evidence(evidence, stats=stats)
        probability = p_star / p_e if p_e > 0 else 0.0
        assignment = {
            var: self.states[var][assigned[var]] for var in self.variables if var not in evidence
        }
        return assignment, probability

    def minimal_conflict(self, evidence: dict[str, str]) -> dict[str, str]:
        """Greedy minimal subset of contradictory evidence (P == 0).

        Removing any remaining item makes the evidence satisfiable again,
        so every returned item is a locatable contributor to the conflict.
        """
        current = dict(evidence)
        for var in list(current):
            if len(current) <= 1:
                break
            trial = {k: v for k, v in current.items() if k != var}
            if self.p_evidence(trial) == 0.0:
                current = trial
        return current
