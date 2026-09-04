"""Capacity-repair optimization over the immutable pairwise-v3 hypergraph.

The implementation has four deliberately separate stages:

* A: maximize the minimum retained fraction ``lambda`` over all 18 targets;
* B: hold that fraction fixed and maximize the number of selected rows;
* C: hold A/B fixed and maximize quotas in the frozen public target order;
* D is implemented in :mod:`allocation` and only consumes C's quota vector.

No text, answer, support, or model outcome is accepted by this module.  The
only row identity used by the optimizer is the privacy-safe source projection
hash from pairwise-v3.
"""

from __future__ import annotations

import math
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .common import ProtocolError, json_exact, require_finite_float, require_fresh_python_bootstrap


_BINARY_TOLERANCE = 1e-7
_REQUIRED_TARGET_COUNT = 18


@dataclass(frozen=True)
class Target:
    index: int
    partition: str
    cell: str
    dataset_id: str
    stratum: str
    upper_quota: int


@dataclass(frozen=True)
class VariableLayout:
    variables: tuple[tuple[int, int], ...]
    variable_index: Mapping[tuple[int, int], int]
    target_variables: Mapping[int, tuple[int, ...]]
    record_variables: Mapping[int, tuple[int, ...]]
    lambda_index: int | None


@dataclass(frozen=True)
class MatrixBuild:
    constraints: LinearConstraint
    layout: VariableLayout
    row_kinds: tuple[str, ...]


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ProtocolError(f"{name} must be a nonnegative integer")
    return value


def targets_from_contract(contract: Mapping[str, Any], upper_quotas: Mapping[str, Any] | None = None) -> tuple[Target, ...]:
    raw = contract.get("targets")
    if not isinstance(raw, list) or len(raw) != _REQUIRED_TARGET_COUNT:
        raise ProtocolError("quota repair requires exactly 18 target definitions")
    targets: list[Target] = []
    seen_cells: set[tuple[str, str]] = set()
    for expected, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ProtocolError("target definition must be an object")
        required = {"target_index", "partition", "cell", "dataset_id", "stratum", "exact_quota"}
        if set(row) != required:
            raise ProtocolError(f"target {expected} typed keys mismatch")
        if row.get("target_index") != expected or type(row.get("target_index")) is not int:
            raise ProtocolError("target indices must be contiguous public order")
        if row.get("partition") not in {"screen", "confirm"}:
            raise ProtocolError("target partition must be screen or confirm")
        if not all(isinstance(row.get(k), str) for k in ("cell", "dataset_id", "stratum")):
            raise ProtocolError("target identity fields must be strings")
        if row["cell"] != f"{row['dataset_id']}|{row['stratum']}":
            raise ProtocolError("target cell/dataset/stratum mismatch")
        upper = _strict_int(row.get("exact_quota"), f"target {expected} exact_quota")
        if upper_quotas is not None:
            key = str(expected)
            if key not in upper_quotas or type(upper_quotas[key]) is not int or upper_quotas[key] != upper:
                raise ProtocolError(f"target {expected} upper quota differs from v3 exact quota")
        identity = (row["partition"], row["cell"])
        if identity in seen_cells:
            raise ProtocolError("duplicate partition/cell target")
        seen_cells.add(identity)
        targets.append(Target(expected, row["partition"], row["cell"], row["dataset_id"], row["stratum"], upper))
    cells = {target.cell for target in targets}
    for cell in cells:
        if {(partition, cell) for partition in ("screen", "confirm")} - seen_cells:
            raise ProtocolError(f"missing screen/confirm target for cell {cell}")
    return tuple(targets)


def _validate_rows(records: Sequence[Mapping[str, Any]], hyperedges: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise ProtocolError("quota repair records cannot be empty")
    previous_hash = ""
    for index, record in enumerate(records):
        if (
            not isinstance(record, Mapping)
            or set(record) != {"record_index", "dataset_id", "stratum", "source_index", "source_projection_sha256"}
            or type(record.get("record_index")) is not int
            or record["record_index"] != index
            or type(record.get("source_index")) is not int
            or record["source_index"] < 0
            or not isinstance(record.get("dataset_id"), str)
            or not isinstance(record.get("stratum"), str)
            or not isinstance(record.get("source_projection_sha256"), str)
            or len(record["source_projection_sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in record["source_projection_sha256"])
            or record["source_projection_sha256"] <= previous_hash
        ):
            raise ProtocolError(f"quota repair record schema/order mismatch at {index}")
        previous_hash = record["source_projection_sha256"]
    for index, edge in enumerate(hyperedges):
        members = edge.get("member_record_indices") if isinstance(edge, Mapping) else None
        if (
            not isinstance(edge, Mapping)
            or set(edge) != {"hyperedge_id", "member_record_indices", "member_count", "relation_kinds", "derivation_count_by_kind"}
            or not isinstance(edge.get("hyperedge_id"), str)
            or len(edge["hyperedge_id"]) != 64
            or any(char not in "0123456789abcdef" for char in edge["hyperedge_id"])
            or not isinstance(members, list)
            or len(members) < 2
            or any(type(value) is not int for value in members)
            or members != sorted(set(members))
            or members[0] < 0
            or members[-1] >= len(records)
            or type(edge.get("member_count")) is not int
            or edge["member_count"] != len(members)
            or not isinstance(edge.get("relation_kinds"), list)
            or edge["relation_kinds"] != sorted(set(edge["relation_kinds"]))
            or any(not isinstance(value, str) for value in edge["relation_kinds"])
            or not isinstance(edge.get("derivation_count_by_kind"), dict)
            or set(edge["derivation_count_by_kind"]) != set(edge["relation_kinds"])
            or any(type(value) is not int or value <= 0 for value in edge["derivation_count_by_kind"].values())
        ):
            raise ProtocolError(f"quota repair hyperedge schema mismatch at {index}")


def _build_layout(records: Sequence[Mapping[str, Any]], targets: Sequence[Target], *, with_lambda: bool) -> VariableLayout:
    records_by_cell: dict[str, list[int]] = defaultdict(list)
    for record in records:
        records_by_cell[f"{record['dataset_id']}|{record['stratum']}"].append(record["record_index"])
    variables: list[tuple[int, int]] = []
    target_variables: dict[int, list[int]] = defaultdict(list)
    record_variables: dict[int, list[int]] = defaultdict(list)
    for target in targets:
        for record_index in records_by_cell.get(target.cell, []):
            column = len(variables)
            variables.append((target.index, record_index))
            target_variables[target.index].append(column)
            record_variables[record_index].append(column)
    if not variables:
        raise ProtocolError("no target/record variables available")
    lambda_index = len(variables) if with_lambda else None
    return VariableLayout(
        tuple(variables),
        {pair: index for index, pair in enumerate(variables)},
        {key: tuple(value) for key, value in target_variables.items()},
        {key: tuple(value) for key, value in record_variables.items()},
        lambda_index,
    )


def _append_row(
    row_indices: list[int],
    column_indices: list[int],
    values: list[float],
    lower: list[float],
    upper: list[float],
    columns: Sequence[int],
    minimum: float,
    maximum: float,
) -> None:
    row = len(lower)
    row_indices.extend([row] * len(columns))
    column_indices.extend(columns)
    values.extend([1.0] * len(columns))
    lower.append(float(minimum))
    upper.append(float(maximum))


def build_matrix(
    records: Sequence[Mapping[str, Any]],
    hyperedges: Sequence[Mapping[str, Any]],
    targets: Sequence[Target],
    *,
    with_lambda: bool,
    minimum_counts: Mapping[int, int] | None = None,
    fixed_counts: Mapping[int, int] | None = None,
    total_selected: int | None = None,
) -> MatrixBuild:
    """Build all constraints for one phase.

    ``minimum_counts`` represents the exact rational lambda threshold after
    phase A (``q_t >= ceil(U_t * lambda*)``).  ``fixed_counts`` and
    ``total_selected`` are phase-C tie-break constraints.
    """

    _validate_rows(records, hyperedges)
    layout = _build_layout(records, targets, with_lambda=with_lambda)
    targets_by_cell: dict[str, dict[str, int]] = defaultdict(dict)
    for target in targets:
        targets_by_cell[target.cell][target.partition] = target.index
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row_kinds: list[str] = []

    def row(columns: Sequence[int], lo: float, hi: float, kind: str) -> None:
        _append_row(row_indices, column_indices, values, lower, upper, columns, lo, hi)
        row_kinds.append(kind)

    # Every record can be assigned to at most one partition target.
    for record_index in range(len(records)):
        row(layout.record_variables.get(record_index, ()), -np.inf, 1.0, "record_assignment")
    # Every direct collision hyperedge is consumed exactly as published; no
    # connected-component closure is introduced here.
    for edge in hyperedges:
        columns: list[int] = []
        for record_index in edge["member_record_indices"]:
            cell = f"{records[record_index]['dataset_id']}|{records[record_index]['stratum']}"
            for target_index in targets_by_cell[cell].values():
                key = (target_index, record_index)
                if key not in layout.variable_index:
                    raise ProtocolError("hyperedge member has no target variable")
                columns.append(layout.variable_index[key])
        row(columns, -np.inf, 1.0, "direct_hyperedge")

    # Upper bounds are explicit in addition to binary/eligibility bounds.  It
    # makes the upper-quota claim visible in the matrix and tamper detectable.
    for target in targets:
        cols = layout.target_variables.get(target.index, ())
        row(cols, -np.inf, target.upper_quota, "target_upper")

    # Phase A's lambda is a continuous variable constrained to [0, 1].
    if with_lambda:
        assert layout.lambda_index is not None
        for target in targets:
            target_cols = list(layout.target_variables.get(target.index, ()))
            # sum(y_t) - U_t * lambda >= 0.  Every coefficient gets an
            # explicit row index; constructing the COO triplets piecemeal
            # here is intentional because an empty target still needs its
            # ``-U*lambda`` term.
            row_number = len(lower)
            row_indices.extend([row_number] * len(target_cols))
            column_indices.extend(target_cols)
            values.extend([1.0] * len(target_cols))
            row_indices.append(row_number)
            column_indices.append(layout.lambda_index)
            values.append(float(-target.upper_quota))
            lower.append(0.0)
            upper.append(np.inf)
            row_kinds.append("lambda_minimum")

    # Minimum count constraints are exact integer consequences of lambda*.
    if minimum_counts is not None:
        for target in targets:
            needed = minimum_counts.get(target.index, 0)
            if type(needed) is not int or needed < 0 or needed > target.upper_quota:
                raise ProtocolError("lambda minimum count is outside target bounds")
            row(layout.target_variables.get(target.index, ()), needed, np.inf, "lambda_floor")

    # screen:confirm ratio, cell by cell.
    for cell in sorted(targets_by_cell):
        pair = targets_by_cell[cell]
        screen = pair.get("screen")
        confirm = pair.get("confirm")
        if screen is None or confirm is None:
            raise ProtocolError(f"cell lacks screen/confirm target: {cell}")
        screen_cols = layout.target_variables.get(screen, ())
        confirm_cols = layout.target_variables.get(confirm, ())
        # confirm - 2*screen <= 1
        row_indices.extend([len(lower)] * len(confirm_cols))
        column_indices.extend(confirm_cols)
        values.extend([1.0] * len(confirm_cols))
        row_indices.extend([len(lower)] * len(screen_cols))
        column_indices.extend(screen_cols)
        values.extend([-2.0] * len(screen_cols))
        lower.append(-1.0)
        upper.append(1.0)
        row_kinds.append("screen_confirm_ratio")

    # The strict-clean MuSiQue 4-hop pool must leave two rows unselected.
    four_hop = [target.index for target in targets if target.cell == "musique_train|4hop"]
    if len(four_hop) == 2:
        cols = [column for index in four_hop for column in layout.target_variables.get(index, ())]
        row(cols, -np.inf, 117.0, "musique_4hop_buffer")

    if fixed_counts is not None:
        for target in targets:
            if target.index in fixed_counts:
                value = fixed_counts[target.index]
                if type(value) is not int or value < 0 or value > target.upper_quota:
                    raise ProtocolError("fixed quota is outside target upper bound")
                row(layout.target_variables.get(target.index, ()), value, value, "lexicographic_fixed")
    if total_selected is not None:
        if type(total_selected) is not int or total_selected < 0:
            raise ProtocolError("total selected must be a nonnegative integer")
        columns = [column for target in targets for column in layout.target_variables.get(target.index, ())]
        row(columns, total_selected, total_selected, "total_selected_fixed")

    if not lower:
        raise ProtocolError("empty quota-repair constraint matrix")
    matrix = coo_matrix(
        (np.asarray(values, dtype=float), (np.asarray(row_indices), np.asarray(column_indices))),
        shape=(len(lower), len(layout.variables) + (1 if with_lambda else 0)),
    ).tocsr()
    return MatrixBuild(LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)), layout, tuple(row_kinds))


def _environment_gate(required: Mapping[str, str]) -> None:
    observed = {name: os.environ.get(name) for name in required}
    if observed != dict(required):
        raise ProtocolError(f"solver thread environment mismatch: {observed}")


def _solver_options(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    required = config.get("solver", {}).get("highs_options", {"threads": 1, "presolve": True, "mip_rel_gap": 0.0})
    options = {"disp": False, **dict(required), "random_seed": int(seed)}
    return options


def _run_milp(
    objective: np.ndarray,
    layout: VariableLayout,
    matrix: MatrixBuild,
    *,
    seed: int,
    config: Mapping[str, Any],
    phase: str,
    allow_explicit_infeasible: bool = False,
) -> tuple[Any, dict[str, Any]]:
    if scipy.__version__ != config.get("solver", {}).get("scipy_version"):
        raise ProtocolError("SciPy version mismatch")
    options = _solver_options(config, seed)
    count = len(layout.variables) + (1 if layout.lambda_index is not None else 0)
    integrality = np.ones(count, dtype=np.uint8)
    if layout.lambda_index is not None:
        integrality[layout.lambda_index] = 0
    lower_bounds = np.zeros(count, dtype=float)
    upper_bounds = np.ones(count, dtype=float)
    with_lambda = layout.lambda_index is not None
    if with_lambda:
        lower_bounds[layout.lambda_index] = 0.0
        upper_bounds[layout.lambda_index] = 1.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected:.*", category=RuntimeWarning)
        result = milp(
            np.asarray(objective, dtype=float),
            integrality=integrality,
            bounds=Bounds(lower_bounds, upper_bounds),
            constraints=matrix.constraints,
            options=options,
        )
    status = getattr(result, "status", None)
    success = getattr(result, "success", None)
    message = getattr(result, "message", None)
    if type(status) is not int or type(success) is not bool or not isinstance(message, str):
        raise ProtocolError(f"{phase} solver result status/success/message typed mismatch")
    raw_fun = getattr(result, "fun", None)
    objective_value = None
    if raw_fun is not None:
        try:
            objective_value = float(raw_fun)
        except (TypeError, ValueError):
            objective_value = None
    report = {
        "phase": phase,
        "status": status,
        "success": success,
        "message": message,
        "classification": "optimal" if status == 0 and success else "explicit_infeasible" if status == 2 and not success else "nonpublishable",
        "process_id": os.getpid(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "highs_random_seed": int(seed),
        "objective": objective_value,
    }
    if allow_explicit_infeasible and status == 2:
        if success is not False or getattr(result, "x", None) is not None or not message.startswith("The problem is infeasible.") or "model_status is Infeasible" not in message:
            raise ProtocolError(f"{phase} status-2 result lacks explicit infeasibility semantics")
        return result, report
    # Every quota-search phase must be proven optimal.  In particular, a
    # timeout/unknown result is never silently treated as a capacity bound.
    if (
        status != 0
        or success is not True
        or not message.startswith("Optimization terminated successfully.")
        or "HiGHS Status 7: Optimal" not in message
    ):
        raise ProtocolError(f"{phase} did not prove optimality: status={status} message={message}")
    if not isinstance(report["objective"], float) or not math.isfinite(report["objective"]):
        raise ProtocolError(f"{phase} objective is missing/nonfinite")
    return result, report


def _validated_incumbent(result: Any, matrix: MatrixBuild, *, phase: str) -> np.ndarray:
    count = matrix.constraints.A.shape[1]
    raw = getattr(result, "x", None)
    values = np.asarray(raw, dtype=float)
    if values.shape != (count,) or not np.all(np.isfinite(values)):
        raise ProtocolError(f"{phase} incumbent missing, wrong-length, or nonfinite")
    layout = matrix.layout
    if layout.lambda_index is not None:
        binary_values = values.copy()
        binary = values[: layout.lambda_index]
        rounded = np.rint(binary)
        if np.any(np.abs(binary - rounded) > _BINARY_TOLERANCE) or np.any((rounded != 0.0) & (rounded != 1.0)):
            raise ProtocolError(f"{phase} record incumbent is not binary")
        if values[layout.lambda_index] < -_BINARY_TOLERANCE or values[layout.lambda_index] > 1.0 + _BINARY_TOLERANCE:
            raise ProtocolError(f"{phase} lambda is outside [0,1]")
        binary_values[: layout.lambda_index] = rounded
        activity = np.asarray(matrix.constraints.A @ binary_values, dtype=float)
        return np.concatenate([rounded.astype(float), np.asarray([values[layout.lambda_index]], dtype=float)]) if not (
            np.any(activity < matrix.constraints.lb - _BINARY_TOLERANCE)
            or np.any(activity > matrix.constraints.ub + _BINARY_TOLERANCE)
        ) else (_raise_incumbent_violation(phase))
    rounded = np.rint(values)
    if np.any(np.abs(values - rounded) > _BINARY_TOLERANCE) or np.any((rounded != 0.0) & (rounded != 1.0)):
        raise ProtocolError(f"{phase} incumbent is not binary")
    activity = np.asarray(matrix.constraints.A @ rounded, dtype=float)
    if np.any(activity < matrix.constraints.lb - _BINARY_TOLERANCE) or np.any(activity > matrix.constraints.ub + _BINARY_TOLERANCE):
        raise ProtocolError(f"{phase} incumbent violates quota-repair constraints")
    return rounded.astype(float)


def _raise_incumbent_violation(phase: str) -> np.ndarray:
    raise ProtocolError(f"{phase} incumbent violates quota-repair constraints")


def _target_counts(values: np.ndarray, layout: VariableLayout, targets: Sequence[Target]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for target in targets:
        counts[target.index] = int(round(float(np.sum(values[list(layout.target_variables.get(target.index, ())) ]))))
    return counts


def _fractional_lambda(counts: Mapping[int, int], targets: Sequence[Target]) -> Fraction:
    ratios: list[Fraction] = []
    for target in targets:
        if target.upper_quota <= 0:
            ratios.append(Fraction(1, 1))
        else:
            ratios.append(Fraction(counts[target.index], target.upper_quota))
    return min(ratios, default=Fraction(0, 1))


def _ceil_fraction_product(upper: int, value: Fraction) -> int:
    product = upper * value.numerator
    denominator = value.denominator
    return (product + denominator - 1) // denominator


def _ratio_map(counts: Mapping[int, int], targets: Sequence[Target]) -> dict[str, float]:
    return {
        str(target.index): (float(counts[target.index]) / target.upper_quota if target.upper_quota else 1.0)
        for target in targets
    }


def _lambda_candidates(targets: Sequence[Target]) -> tuple[Fraction, ...]:
    """All attainable min-retention breakpoints for integer quotas.

    A minimum of ``q_t / U_t`` can only change at one of these finite
    fractions.  Keeping the set exact (``Fraction`` rather than float) avoids
    making fairness depend on a solver's arbitrary continuous incumbent.
    """

    candidates: set[Fraction] = {Fraction(0, 1)}
    for target in targets:
        if target.upper_quota <= 0:
            continue
        candidates.update(Fraction(value, target.upper_quota) for value in range(target.upper_quota + 1))
    return tuple(sorted(candidates))


def _threshold_feasibility(
    records: Sequence[Mapping[str, Any]],
    hyperedges: Sequence[Mapping[str, Any]],
    targets: Sequence[Target],
    config: Mapping[str, Any],
    candidate: Fraction,
    *,
    seed: int,
) -> tuple[bool, dict[str, Any]]:
    """Prove feasibility/infeasibility for one exact rational lambda floor."""

    minimum_counts = {target.index: _ceil_fraction_product(target.upper_quota, candidate) for target in targets}
    matrix = build_matrix(
        records,
        hyperedges,
        targets,
        with_lambda=False,
        minimum_counts=minimum_counts,
    )
    count = matrix.constraints.A.shape[1]
    result, report = _run_milp(
        np.zeros(count, dtype=float),
        matrix.layout,
        matrix,
        seed=seed,
        config=config,
        phase=f"A_CERT_{candidate.numerator}_{candidate.denominator}",
        allow_explicit_infeasible=True,
    )
    if report["status"] == 2:
        return False, report
    # A feasible certificate still needs a finite binary incumbent satisfying
    # every direct hyperedge/ratio/buffer row.
    _validated_incumbent(result, matrix, phase=report["phase"])
    return True, report


def _exact_lambda_certificate(
    records: Sequence[Mapping[str, Any]],
    hyperedges: Sequence[Mapping[str, Any]],
    targets: Sequence[Target],
    config: Mapping[str, Any],
    *,
    seed: int,
    continuous_lambda: float,
) -> tuple[Fraction, dict[str, Any]]:
    """Binary-search the finite breakpoint set and prove the adjacent bound.

    The continuous A solve is retained as a diagnostic, but it cannot choose
    the final fairness value.  The exact certificate below does that using
    integer quota floors and a monotone feasibility search.
    """

    candidates = _lambda_candidates(targets)
    lo = 0
    hi = len(candidates) - 1
    tested: list[dict[str, Any]] = []
    # Zero is always feasible (provided the graph rows themselves are valid),
    # but explicitly test it so a malformed matrix cannot pass silently.
    zero_ok, zero_report = _threshold_feasibility(records, hyperedges, targets, config, candidates[0], seed=seed)
    tested.append({"candidate_numerator": 0, "candidate_denominator": 1, "feasible": zero_ok, "solver": zero_report})
    if not zero_ok:
        raise ProtocolError("zero quota floor is unexpectedly infeasible")
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = candidates[mid]
        feasible, report = _threshold_feasibility(records, hyperedges, targets, config, candidate, seed=seed)
        tested.append({"candidate_numerator": candidate.numerator, "candidate_denominator": candidate.denominator, "feasible": feasible, "solver": report})
        if feasible:
            lo = mid
        else:
            hi = mid - 1
    optimum = candidates[lo]
    # If the binary search did not directly test the optimum (possible when it
    # is the initial lower endpoint), obtain a positive feasibility certificate.
    if not any(item["candidate_numerator"] == optimum.numerator and item["candidate_denominator"] == optimum.denominator and item["feasible"] for item in tested):
        feasible, report = _threshold_feasibility(records, hyperedges, targets, config, optimum, seed=seed)
        tested.append({"candidate_numerator": optimum.numerator, "candidate_denominator": optimum.denominator, "feasible": feasible, "solver": report})
        if not feasible:
            raise ProtocolError("binary-search optimum lacks feasible certificate")
    next_candidate = candidates[lo + 1] if lo + 1 < len(candidates) else None
    next_feasible: bool | None = None
    next_report: dict[str, Any] | None = None
    if next_candidate is not None:
        next_feasible, next_report = _threshold_feasibility(records, hyperedges, targets, config, next_candidate, seed=seed)
        tested.append({"candidate_numerator": next_candidate.numerator, "candidate_denominator": next_candidate.denominator, "feasible": next_feasible, "solver": next_report})
        if next_feasible:
            raise ProtocolError("binary-search lambda certificate has feasible higher breakpoint")
    continuous_gap = abs(float(continuous_lambda) - float(optimum))
    if not math.isfinite(float(continuous_lambda)) or continuous_gap > 1e-5:
        raise ProtocolError("continuous phase-A lambda disagrees with exact breakpoint certificate")
    return optimum, {
        "method": "finite_breakpoint_binary_search",
        "candidate_count": len(candidates),
        "tested_count": len(tested),
        "continuous_lambda": float(continuous_lambda),
        "continuous_to_exact_abs_gap": continuous_gap,
        "optimum": {"numerator": optimum.numerator, "denominator": optimum.denominator},
        "next_breakpoint": None if next_candidate is None else {"numerator": next_candidate.numerator, "denominator": next_candidate.denominator},
        "next_breakpoint_feasible": next_feasible,
        "tested": tested,
    }


def solve_quota_search(
    records: Sequence[Mapping[str, Any]],
    hyperedges: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    seed: int = 1,
    require_environment: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run phases A/B/C and return ``(quota_vector, proof_report)``.

    The returned quota vector is a JSON-safe mapping keyed by public target
    index.  No allocation rows are emitted by this function; phase D remains a
    separate call after this proof is frozen.
    """

    # A solver call is itself a formal-capable entrypoint when used by an
    # embedding caller.  Require the same fresh no-bytecode bootstrap as the
    # command-line workers, even though this module has already been imported.
    require_fresh_python_bootstrap()
    if type(seed) is not int or seed <= 0:
        raise ProtocolError("quota-search seed must be a positive integer")
    if require_environment:
        _environment_gate(config["solver"]["required_environment"])
    targets = targets_from_contract(contract, config.get("upper_quotas"))
    _validate_rows(records, hyperedges)

    # A: maximize lambda.  The objective is -lambda because scipy minimizes.
    phase_a_matrix = build_matrix(records, hyperedges, targets, with_lambda=True)
    a_count = phase_a_matrix.constraints.A.shape[1]
    objective_a = np.zeros(a_count, dtype=float)
    assert phase_a_matrix.layout.lambda_index is not None
    objective_a[phase_a_matrix.layout.lambda_index] = -1.0
    result_a, report_a = _run_milp(objective_a, phase_a_matrix.layout, phase_a_matrix, seed=seed, config=config, phase="A")
    incumbent_a = _validated_incumbent(result_a, phase_a_matrix, phase="A")
    counts_a = _target_counts(incumbent_a, phase_a_matrix.layout, targets)
    lambda_value = float(getattr(result_a, "x")[phase_a_matrix.layout.lambda_index])
    lambda_fraction, lambda_certificate = _exact_lambda_certificate(
        records,
        hyperedges,
        targets,
        config,
        seed=seed,
        continuous_lambda=lambda_value,
    )
    minimum_counts = {
        target.index: _ceil_fraction_product(target.upper_quota, lambda_fraction)
        for target in targets
    }

    # B: fixed exact rational lambda floor, maximize total selected.
    phase_b_matrix = build_matrix(
        records, hyperedges, targets, with_lambda=False, minimum_counts=minimum_counts
    )
    b_count = phase_b_matrix.constraints.A.shape[1]
    objective_b = -np.ones(b_count, dtype=float)
    result_b, report_b = _run_milp(objective_b, phase_b_matrix.layout, phase_b_matrix, seed=seed, config=config, phase="B")
    incumbent_b = _validated_incumbent(result_b, phase_b_matrix, phase="B")
    counts_b = _target_counts(incumbent_b, phase_b_matrix.layout, targets)
    total_b = int(sum(counts_b.values()))
    if any(counts_b[target.index] < minimum_counts[target.index] for target in targets):
        raise ProtocolError("phase B incumbent violates fixed lambda floor")
    # Phase A is a max-min proof; B is not allowed to accidentally improve its
    # exact fraction due to a malformed/underconstrained matrix.
    lambda_b = _fractional_lambda(counts_b, targets)
    if lambda_b != lambda_fraction:
        raise ProtocolError("phase B changed the proven lambda fraction")

    # C: sequentially maximize each quota in frozen public target order while
    # keeping the B total and all prior quota choices fixed.
    fixed: dict[int, int] = {}
    phase_c_reports: list[dict[str, Any]] = []
    final_counts: dict[int, int] = {}
    for target in targets:
        matrix_c = build_matrix(
            records,
            hyperedges,
            targets,
            with_lambda=False,
            minimum_counts=minimum_counts,
            fixed_counts=fixed,
            total_selected=total_b,
        )
        c_count = matrix_c.constraints.A.shape[1]
        objective_c = np.zeros(c_count, dtype=float)
        for column in matrix_c.layout.target_variables.get(target.index, ()):
            objective_c[column] = -1.0
        result_c, report_c = _run_milp(objective_c, matrix_c.layout, matrix_c, seed=seed, config=config, phase=f"C_{target.index}")
        incumbent_c = _validated_incumbent(result_c, matrix_c, phase=f"C_{target.index}")
        counts_c = _target_counts(incumbent_c, matrix_c.layout, targets)
        chosen = counts_c[target.index]
        if chosen < minimum_counts[target.index] or chosen > target.upper_quota:
            raise ProtocolError(f"phase C target {target.index} quota outside bounds")
        fixed[target.index] = chosen
        final_counts[target.index] = chosen
        phase_c_reports.append({
            **report_c,
            "target_index": target.index,
            "chosen_quota": chosen,
            "fixed_prior_targets": {str(k): fixed[k] for k in sorted(fixed) if k != target.index},
        })
    if sum(final_counts.values()) != total_b:
        raise ProtocolError("phase C quotas do not preserve phase B total")
    # Ratio and 4-hop invariants are checked independently of the MILP matrix.
    for cell in sorted({target.cell for target in targets}):
        pair = {target.partition: target for target in targets if target.cell == cell}
        if abs(final_counts[pair["confirm"].index] - 2 * final_counts[pair["screen"].index]) > 1:
            raise ProtocolError(f"phase C screen/confirm ratio violated for {cell}")
    four_hop_targets = [target for target in targets if target.cell == "musique_train|4hop"]
    if len(four_hop_targets) == 2 and sum(final_counts[target.index] for target in four_hop_targets) > 117:
        raise ProtocolError("phase C MuSiQue 4-hop buffer violated")

    quota_vector = {str(target.index): final_counts[target.index] for target in targets}
    proof = {
        "schema_version": 1,
        "status": "quota_search_proven_optimal",
        "seed": seed,
        "process_id": os.getpid(),
        "target_order": [target.index for target in targets],
        "upper_quotas": {str(target.index): target.upper_quota for target in targets},
        "phase_a": {
            **report_a,
            "lambda": float(lambda_fraction),
            "lambda_numerator": lambda_fraction.numerator,
            "lambda_denominator": lambda_fraction.denominator,
            "continuous_lambda": lambda_value,
            "incumbent_quota_counts": {str(k): counts_a[k] for k in sorted(counts_a)},
            "minimum_counts_fixed_for_B": {str(k): minimum_counts[k] for k in sorted(minimum_counts)},
            "exact_lambda_certificate": lambda_certificate,
        },
        "phase_b": {
            **report_b,
            "lambda_numerator": lambda_fraction.numerator,
            "lambda_denominator": lambda_fraction.denominator,
            "quota_counts": {str(k): counts_b[k] for k in sorted(counts_b)},
            "selected_total": total_b,
        },
        "phase_c": {
            "status": "lexicographic_quota_tiebreak_complete",
            "public_target_order": [target.index for target in targets],
            "quota_counts": quota_vector,
            "selected_total": total_b,
            "steps": phase_c_reports,
        },
        "invariants": {
            "all_targets_retain_at_least_lambda": all(
                Fraction(final_counts[target.index], target.upper_quota) >= lambda_fraction
                if target.upper_quota else True for target in targets
            ),
            "screen_confirm_ratio_max_abs_error": max(
                abs(final_counts[next(t.index for t in targets if t.cell == cell and t.partition == "confirm")] - 2 * final_counts[next(t.index for t in targets if t.cell == cell and t.partition == "screen")])
                for cell in {target.cell for target in targets}
            ),
            "selected_total": total_b,
        },
        "privacy": {
            "answer_values_read": False,
            "model_outcomes_read": False,
            "raw_text_persisted": False,
        },
    }
    return quota_vector, proof


def validate_quota_vector(
    quota_vector: Mapping[str, Any],
    targets: Sequence[Target],
    *,
    expected_lambda: Fraction | None = None,
    expected_total: int | None = None,
) -> dict[str, Any]:
    """Strictly validate a C-stage quota vector (used by deep/tamper tests)."""

    if not isinstance(quota_vector, Mapping):
        raise ProtocolError("quota vector must be an object")
    expected_keys = {str(target.index) for target in targets}
    if set(quota_vector) != expected_keys:
        raise ProtocolError("quota vector target keys mismatch")
    counts: dict[int, int] = {}
    for target in targets:
        value = quota_vector[str(target.index)]
        if type(value) is not int or value < 0 or value > target.upper_quota:
            raise ProtocolError(f"quota {target.index} is not an integer within v3 upper bound")
        counts[target.index] = value
    for cell in {target.cell for target in targets}:
        screen = next(target for target in targets if target.cell == cell and target.partition == "screen")
        confirm = next(target for target in targets if target.cell == cell and target.partition == "confirm")
        if abs(counts[confirm.index] - 2 * counts[screen.index]) > 1:
            raise ProtocolError(f"quota screen/confirm ratio mismatch for {cell}")
    four = [target for target in targets if target.cell == "musique_train|4hop"]
    if len(four) == 2 and sum(counts[target.index] for target in four) > 117:
        raise ProtocolError("quota 4-hop buffer mismatch")
    if expected_lambda is not None and _fractional_lambda(counts, targets) != expected_lambda:
        raise ProtocolError("quota vector lambda changed")
    total = sum(counts.values())
    if expected_total is not None and total != expected_total:
        raise ProtocolError("quota vector selected total changed")
    return {
        "quota_counts": {str(index): counts[index] for index in sorted(counts)},
        "selected_total": total,
        "lambda_numerator": _fractional_lambda(counts, targets).numerator,
        "lambda_denominator": _fractional_lambda(counts, targets).denominator,
    }


def validate_quota_proof(
    proof: Mapping[str, Any],
    targets: Sequence[Target],
    quota_vector: Mapping[str, Any] | None = None,
    *,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    """Fail-closed validator for a published A/B/C proof.

    This is intentionally independent of the solver call.  Deep validators
    and tamper tests use it to reject bool-as-int fields, status 1/3/4,
    non-finite objectives, malformed adjacent-breakpoint evidence, and seed or
    target-order drift.  If ``quota_vector`` is supplied it is the final
    phase-C vector (the phase-B maximizer may have a different distribution).
    """

    if not isinstance(proof, Mapping):
        raise ProtocolError("quota proof must be an object")
    expected_keys = {
        "schema_version", "status", "seed", "process_id", "target_order", "upper_quotas",
        "phase_a", "phase_b", "phase_c", "invariants", "privacy",
    }
    if set(proof) != expected_keys or proof.get("schema_version") != 1 or proof.get("status") != "quota_search_proven_optimal":
        raise ProtocolError("quota proof top-level schema/status mismatch")
    if type(proof.get("seed")) is not int or proof["seed"] <= 0 or (expected_seed is not None and proof["seed"] != expected_seed):
        raise ProtocolError("quota proof seed is not a positive integer or differs")
    if type(proof.get("process_id")) is not int or proof["process_id"] <= 0:
        raise ProtocolError("quota proof process_id is invalid")
    order = proof.get("target_order")
    if order != [target.index for target in targets] or any(type(value) is not int for value in order):
        raise ProtocolError("quota proof public target order mismatch")
    upper = proof.get("upper_quotas")
    expected_upper = {str(target.index): target.upper_quota for target in targets}
    if not isinstance(upper, Mapping) or set(upper) != set(expected_upper) or any(type(upper[k]) is not int for k in upper) or dict(upper) != expected_upper:
        raise ProtocolError("quota proof upper quota map mismatch")

    proof_pid = proof["process_id"]
    proof_seed = proof["seed"]

    def _validate_solver_evidence(
        value: Any,
        name: str,
        *,
        allow_infeasible: bool = False,
        exact_keys: bool = True,
    ) -> Mapping[str, Any]:
        """Validate one serialized HiGHS result, including every typed field.

        This validator is deliberately stricter than ``json``'s normal
        number semantics: ``bool`` is not an integer, status 1/3/4 (or any
        unknown status) is never publishable, and an explicit infeasible
        result must carry the exact status-2/no-incumbent wording produced by
        the pinned HiGHS build.
        """
        if not isinstance(value, Mapping):
            raise ProtocolError(f"{name} solver evidence must be an object")
        expected = {
            "phase", "status", "success", "message", "classification",
            "process_id", "python_hash_seed", "highs_random_seed", "objective",
        }
        if (set(value) != expected) if exact_keys else (not expected.issubset(set(value))):
            raise ProtocolError(f"{name} solver evidence keys mismatch")
        if not isinstance(value.get("phase"), str) or not value["phase"]:
            raise ProtocolError(f"{name} solver phase is malformed")
        if type(value.get("status")) is not int:
            raise ProtocolError(f"{name} solver status must be an integer")
        if type(value.get("success")) is not bool:
            raise ProtocolError(f"{name} solver success must be boolean")
        if not isinstance(value.get("message"), str):
            raise ProtocolError(f"{name} solver message must be text")
        if type(value.get("process_id")) is not int or value["process_id"] <= 0 or value["process_id"] != proof_pid:
            raise ProtocolError(f"{name} solver process binding mismatch")
        if value.get("python_hash_seed") is not None and not isinstance(value.get("python_hash_seed"), str):
            raise ProtocolError(f"{name} python hash seed type mismatch")
        if type(value.get("highs_random_seed")) is not int or value["highs_random_seed"] != proof_seed:
            raise ProtocolError(f"{name} HiGHS seed binding mismatch")
        status = value["status"]
        success = value["success"]
        message = value["message"]
        classification = value.get("classification")
        objective = value.get("objective")
        if status == 0:
            if success is not True or classification != "optimal" or not message.startswith("Optimization terminated successfully.") or "HiGHS Status 7: Optimal" not in message:
                raise ProtocolError(f"{name} solver is not strict Optimal")
            if isinstance(objective, bool) or not isinstance(objective, float) or not math.isfinite(objective):
                raise ProtocolError(f"{name} optimal objective is nonfinite/not float")
        elif status == 2 and allow_infeasible:
            if success is not False or classification != "explicit_infeasible" or objective is not None or not message.startswith("The problem is infeasible.") or "model_status is Infeasible" not in message:
                raise ProtocolError(f"{name} status-2 evidence is not explicit infeasibility")
        else:
            # HiGHS statuses 1/3/4 and all future/unknown statuses are
            # intentionally rejected instead of being interpreted as a
            # weak or partial certificate.
            raise ProtocolError(f"{name} has nonpublishable solver status {status}")
        return value

    def optimal_phase(value: Any, name: str, *, expected_phase: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise ProtocolError(f"{name} proof must be an object")
        expected_keys = {
            "phase", "status", "success", "message", "classification",
            "process_id", "python_hash_seed", "highs_random_seed", "objective",
        }
        if expected_phase == "A":
            expected_keys |= {
                "lambda", "lambda_numerator", "lambda_denominator",
                "continuous_lambda", "incumbent_quota_counts",
                "minimum_counts_fixed_for_B", "exact_lambda_certificate",
            }
        elif expected_phase == "B":
            expected_keys |= {"lambda_numerator", "lambda_denominator", "quota_counts", "selected_total"}
        if set(value) != expected_keys:
            raise ProtocolError(f"{name} proof keys mismatch")
        evidence = _validate_solver_evidence(value, name, exact_keys=False)
        if evidence["phase"] != expected_phase or evidence["classification"] != "optimal":
            raise ProtocolError(f"{name} phase/classification mismatch")
        return evidence

    phase_a = optimal_phase(proof.get("phase_a"), "phase_a", expected_phase="A")
    phase_b = optimal_phase(proof.get("phase_b"), "phase_b", expected_phase="B")
    if phase_a["process_id"] != proof_pid or phase_b["process_id"] != proof_pid:
        raise ProtocolError("quota proof phase PID binding mismatch")
    if phase_b["python_hash_seed"] != phase_a["python_hash_seed"]:
        raise ProtocolError("quota proof Python hash seed disagreement")
    if type(phase_a.get("lambda_numerator")) is not int or type(phase_a.get("lambda_denominator")) is not int or phase_a["lambda_denominator"] <= 0 or phase_a["lambda_numerator"] < 0 or phase_a["lambda_numerator"] > phase_a["lambda_denominator"]:
        raise ProtocolError("phase A exact lambda fraction is malformed")
    if isinstance(phase_a.get("lambda"), bool) or not isinstance(phase_a.get("lambda"), float) or not math.isfinite(phase_a["lambda"]):
        raise ProtocolError("phase A lambda is nonfinite/not float")
    if abs(phase_a["lambda"] - float(Fraction(phase_a["lambda_numerator"], phase_a["lambda_denominator"]))) > 1e-12:
        raise ProtocolError("phase A lambda does not equal its exact fraction")
    if isinstance(phase_a.get("continuous_lambda"), bool) or not isinstance(phase_a.get("continuous_lambda"), float) or not math.isfinite(phase_a["continuous_lambda"]):
        raise ProtocolError("phase A continuous lambda is nonfinite/not float")
    if not isinstance(phase_a.get("incumbent_quota_counts"), Mapping) or not isinstance(phase_a.get("minimum_counts_fixed_for_B"), Mapping):
        raise ProtocolError("phase A quota maps are missing")
    expected_map_keys = {str(target.index) for target in targets}
    for map_name in ("incumbent_quota_counts", "minimum_counts_fixed_for_B"):
        map_value = phase_a[map_name]
        if set(map_value) != expected_map_keys or any(type(map_value[key]) is not int for key in map_value):
            raise ProtocolError(f"phase A {map_name} has non-integer or incomplete targets")
        for target in targets:
            if map_value[str(target.index)] < 0 or map_value[str(target.index)] > target.upper_quota:
                raise ProtocolError(f"phase A {map_name} exceeds target bounds")
    expected_minimums = {
        str(target.index): _ceil_fraction_product(
            target.upper_quota,
            Fraction(phase_a["lambda_numerator"], phase_a["lambda_denominator"]),
        )
        for target in targets
    }
    if dict(phase_a["minimum_counts_fixed_for_B"]) != expected_minimums:
        raise ProtocolError("phase A lambda floor map does not bind exact fraction")
    if abs(phase_a["objective"] + phase_a["continuous_lambda"]) > 1e-9:
        raise ProtocolError("phase A objective does not bind continuous lambda")
    certificate = phase_a.get("exact_lambda_certificate")
    if not isinstance(certificate, Mapping):
        raise ProtocolError("phase A adjacent-breakpoint certificate is malformed")
    cert_keys = {
        "method", "candidate_count", "tested_count", "continuous_lambda",
        "continuous_to_exact_abs_gap", "optimum", "next_breakpoint",
        "next_breakpoint_feasible", "tested",
    }
    if set(certificate) != cert_keys or certificate.get("method") != "finite_breakpoint_binary_search":
        raise ProtocolError("phase A certificate keys/method mismatch")
    candidates = _lambda_candidates(targets)
    if type(certificate.get("candidate_count")) is not int or certificate["candidate_count"] != len(candidates) or certificate["candidate_count"] <= 0:
        raise ProtocolError("phase A certificate candidate count mismatch")
    if type(certificate.get("tested_count")) is not int or certificate["tested_count"] <= 0:
        raise ProtocolError("phase A certificate tested count is invalid")
    if isinstance(certificate.get("continuous_lambda"), bool) or not isinstance(certificate.get("continuous_lambda"), float) or not math.isfinite(certificate["continuous_lambda"]):
        raise ProtocolError("phase A certificate continuous lambda is nonfinite")
    if isinstance(certificate.get("continuous_to_exact_abs_gap"), bool) or not isinstance(certificate.get("continuous_to_exact_abs_gap"), float) or not math.isfinite(certificate["continuous_to_exact_abs_gap"]):
        raise ProtocolError("phase A certificate lambda gap is nonfinite")
    if abs(certificate["continuous_lambda"] - phase_a["continuous_lambda"]) > 1e-12 or abs(certificate["continuous_to_exact_abs_gap"] - abs(certificate["continuous_lambda"] - phase_a["lambda"])) > 1e-9:
        raise ProtocolError("phase A certificate continuous diagnostics do not bind phase A")
    optimum = certificate.get("optimum")
    if not isinstance(optimum, Mapping) or set(optimum) != {"numerator", "denominator"} or type(optimum.get("numerator")) is not int or type(optimum.get("denominator")) is not int or optimum["denominator"] <= 0 or optimum["numerator"] < 0 or optimum["numerator"] > optimum["denominator"]:
        raise ProtocolError("phase A certificate optimum fraction is malformed")
    if optimum["numerator"] != phase_a["lambda_numerator"] or optimum["denominator"] != phase_a["lambda_denominator"]:
        raise ProtocolError("phase A certificate optimum does not bind lambda")
    optimum_fraction = Fraction(optimum["numerator"], optimum["denominator"])
    if optimum_fraction not in candidates:
        raise ProtocolError("phase A certificate optimum is not an attainable breakpoint")
    next_expected = candidates[candidates.index(optimum_fraction) + 1] if candidates.index(optimum_fraction) + 1 < len(candidates) else None
    next_value = certificate.get("next_breakpoint")
    if next_expected is None:
        if next_value is not None or certificate.get("next_breakpoint_feasible") is not None:
            raise ProtocolError("phase A terminal breakpoint certificate must have no adjacent value")
    else:
        if not isinstance(next_value, Mapping) or set(next_value) != {"numerator", "denominator"} or type(next_value.get("numerator")) is not int or type(next_value.get("denominator")) is not int or next_value["denominator"] <= 0 or next_value["numerator"] < 0 or next_value["numerator"] > next_value["denominator"] or Fraction(next_value["numerator"], next_value["denominator"]) != next_expected or certificate.get("next_breakpoint_feasible") is not False:
            raise ProtocolError("phase A adjacent breakpoint is not the exact next candidate")
    tested = certificate.get("tested")
    if not isinstance(tested, list) or len(tested) != certificate["tested_count"]:
        raise ProtocolError("phase A tested breakpoint list length mismatch")
    tested_feasibility: dict[Fraction, bool] = {}
    for index, item in enumerate(tested):
        if not isinstance(item, Mapping) or set(item) != {"candidate_numerator", "candidate_denominator", "feasible", "solver"}:
            raise ProtocolError(f"phase A tested breakpoint {index} schema mismatch")
        if type(item["candidate_numerator"]) is not int or type(item["candidate_denominator"]) is not int or item["candidate_denominator"] <= 0 or item["candidate_numerator"] < 0 or item["candidate_numerator"] > item["candidate_denominator"] or type(item["feasible"]) is not bool:
            raise ProtocolError(f"phase A tested breakpoint {index} typed mismatch")
        candidate = Fraction(item["candidate_numerator"], item["candidate_denominator"])
        if candidate not in candidates:
            raise ProtocolError(f"phase A tested breakpoint {index} is outside finite candidate set")
        if candidate in tested_feasibility and tested_feasibility[candidate] is not item["feasible"]:
            raise ProtocolError(f"phase A tested breakpoint {index} has conflicting feasibility")
        tested_feasibility[candidate] = item["feasible"]
        solver_evidence = _validate_solver_evidence(item["solver"], f"phase_a.tested[{index}]", allow_infeasible=True)
        if solver_evidence["phase"] != f"A_CERT_{item['candidate_numerator']}_{item['candidate_denominator']}" or solver_evidence["python_hash_seed"] != phase_a["python_hash_seed"]:
            raise ProtocolError(f"phase A tested breakpoint {index} solver binding mismatch")
        solver_status = item["solver"]["status"]
        if item["feasible"] and solver_status != 0:
            raise ProtocolError(f"phase A feasible breakpoint {index} lacks status-0 proof")
        if not item["feasible"] and solver_status != 2:
            raise ProtocolError(f"phase A infeasible breakpoint {index} lacks status-2 proof")
    if tested_feasibility.get(optimum_fraction) is not True:
        raise ProtocolError("phase A optimum lacks a feasible tested certificate")
    if next_expected is not None and tested_feasibility.get(next_expected) is not False:
        raise ProtocolError("phase A adjacent breakpoint lacks explicit infeasibility")
    for candidate, feasible in tested_feasibility.items():
        if (candidate <= optimum_fraction and not feasible) or (candidate > optimum_fraction and feasible):
            raise ProtocolError("phase A tested feasibility is not monotone around optimum")
    if type(phase_b.get("lambda_numerator")) is not int or type(phase_b.get("lambda_denominator")) is not int or phase_b["lambda_numerator"] != phase_a["lambda_numerator"] or phase_b["lambda_denominator"] != phase_a["lambda_denominator"]:
        raise ProtocolError("phase B exact lambda binding mismatch")
    if type(phase_b.get("selected_total")) is not int or phase_b["selected_total"] < 0:
        raise ProtocolError("phase B selected_total is not a nonnegative integer")
    phase_b_counts = phase_b.get("quota_counts")
    if not isinstance(phase_b_counts, Mapping) or set(phase_b_counts) != expected_map_keys or any(type(phase_b_counts[key]) is not int for key in phase_b_counts) or sum(phase_b_counts.values()) != phase_b["selected_total"]:
        raise ProtocolError("phase B quota counts are incomplete/non-integer or total mismatch")
    for target in targets:
        value = phase_b_counts[str(target.index)]
        if value < phase_a["minimum_counts_fixed_for_B"][str(target.index)] or value > target.upper_quota:
            raise ProtocolError("phase B quota violates exact lambda floor or upper bound")
    if abs(phase_b["objective"] + float(phase_b["selected_total"])) > 1e-9:
        raise ProtocolError("phase B objective does not bind selected total")
    phase_c = proof.get("phase_c")
    if not isinstance(phase_c, Mapping) or set(phase_c) != {"status", "public_target_order", "quota_counts", "selected_total", "steps"} or phase_c.get("status") != "lexicographic_quota_tiebreak_complete" or phase_c.get("public_target_order") != [target.index for target in targets] or any(type(value) is not int for value in phase_c.get("public_target_order", [])) or type(phase_c.get("selected_total")) is not int or phase_c["selected_total"] != phase_b["selected_total"]:
        raise ProtocolError("phase C lexicographic proof schema mismatch")
    c_counts = phase_c.get("quota_counts")
    if not isinstance(c_counts, Mapping):
        raise ProtocolError("phase C quota vector missing")
    validate_quota_vector(c_counts, targets, expected_total=phase_b["selected_total"])
    if quota_vector is not None and not json_exact(c_counts, quota_vector):
        raise ProtocolError("final quota vector binding mismatch")
    steps = phase_c.get("steps")
    if not isinstance(steps, list) or len(steps) != len(targets):
        raise ProtocolError("phase C must contain one step per target")
    for target, step in zip(targets, steps, strict=True):
        if not isinstance(step, Mapping):
            raise ProtocolError("phase C step must be an object")
        expected_step_keys = {"phase", "status", "success", "message", "classification", "process_id", "python_hash_seed", "highs_random_seed", "objective", "target_index", "chosen_quota", "fixed_prior_targets"}
        if set(step) != expected_step_keys or step.get("target_index") != target.index or type(step.get("target_index")) is not int or type(step.get("chosen_quota")) is not int or step["chosen_quota"] != c_counts[str(target.index)]:
            raise ProtocolError("phase C step target/quota binding mismatch")
        evidence = _validate_solver_evidence(step, f"phase_c.step_{target.index}", exact_keys=False)
        if evidence["phase"] != f"C_{target.index}" or evidence["classification"] != "optimal":
            raise ProtocolError("phase C step solver phase/classification mismatch")
        if evidence["python_hash_seed"] != phase_a["python_hash_seed"]:
            raise ProtocolError("phase C Python hash seed disagreement")
        if step["chosen_quota"] < 0 or step["chosen_quota"] > target.upper_quota:
            raise ProtocolError("phase C chosen quota outside target bounds")
        prior = step.get("fixed_prior_targets")
        expected_prior = {str(previous.index): c_counts[str(previous.index)] for previous in targets if previous.index < target.index}
        if not isinstance(prior, Mapping) or set(prior) != set(expected_prior) or any(type(prior[key]) is not int for key in prior) or dict(prior) != expected_prior:
            raise ProtocolError("phase C prior-quota binding mismatch")
        if step["objective"] != -float(step["chosen_quota"]):
            raise ProtocolError("phase C objective does not bind chosen quota")
    invariants = proof.get("invariants")
    expected_invariant_keys = {"all_targets_retain_at_least_lambda", "screen_confirm_ratio_max_abs_error", "selected_total"}
    if not isinstance(invariants, Mapping) or set(invariants) != expected_invariant_keys or invariants.get("all_targets_retain_at_least_lambda") is not True or type(invariants.get("screen_confirm_ratio_max_abs_error")) is not int or invariants["screen_confirm_ratio_max_abs_error"] < 0 or type(invariants.get("selected_total")) is not int or invariants["selected_total"] != phase_b["selected_total"]:
        raise ProtocolError("quota proof invariants are malformed")
    privacy = proof.get("privacy")
    if privacy != {"answer_values_read": False, "model_outcomes_read": False, "raw_text_persisted": False}:
        raise ProtocolError("quota proof privacy flags are not strict false")
    return validate_quota_vector(c_counts, targets, expected_total=phase_b["selected_total"])
