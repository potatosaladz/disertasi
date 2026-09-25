from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Sequence, TypeVar

from z3 import Real, RealVal, Solver, sat

from backend.core_algorithms import STATUTORY_DEFICIT_CEILING_PERCENT_GDP

T = TypeVar("T")
STATUTORY_DEFICIT_CEILING = Decimal(str(STATUTORY_DEFICIT_CEILING_PERCENT_GDP))
CAR_HARD_STOP_MESSAGES = {
    "id": "Simulasi dibatalkan: Benturan batas keras terdeteksi pada defisit",
    "en": "Simulation cancelled: A verified hard-limit conflict was detected in the deficit.",
}


@dataclass(frozen=True)
class CarEvaluation:
    feasible: list[Any]
    rejected: list[dict[str, Any]]
    solver_status: str
    selected: Any | None
    hard_constraints: list[dict[str, Any]]
    status: str
    messages: dict[str, str] | None = None


def _value(item: object, field: str) -> Any:
    if isinstance(item, dict):
        return item[field]
    return getattr(item, field)


def _name(item: object, index: int) -> str:
    try:
        return str(_value(item, "name"))
    except (AttributeError, KeyError, TypeError):
        return f"Alternative {index + 1}"


def _finite_decimal(value: object, field: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, not boolean")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not number.is_finite():
        raise ValueError(f"{field} must be finite")
    return number


def _rejection(
    index: int,
    alternative: object,
    deficit: Decimal | None,
    ceiling: Decimal,
    violated_constraints: list[str],
    reason: str,
) -> dict[str, Any]:
    return {
        "index": index,
        "name": _name(alternative, index),
        "projected_deficit_percent_gdp": float(deficit) if deficit is not None else None,
        "ceiling_percent_gdp": float(ceiling),
        "excess_percent_gdp": (
            float(max(Decimal("0"), deficit - ceiling)) if deficit is not None else None
        ),
        "violated_constraints": violated_constraints,
        "solver_status": "unsat",
        "reason": reason,
    }


def evaluate_car_constraints(
    alternatives: Sequence[T],
    max_deficit: float,
    *,
    hard_stop_reason: str | None = None,
) -> CarEvaluation:
    requested_ceiling = _finite_decimal(max_deficit, "max_deficit")
    if requested_ceiling < 0:
        raise ValueError("max_deficit must be nonnegative")
    ceiling = min(requested_ceiling, STATUTORY_DEFICIT_CEILING)
    feasible: list[T] = []
    rejected: list[dict[str, Any]] = []
    for index, alternative in enumerate(alternatives):
        try:
            deficit = _finite_decimal(_value(alternative, "deficit"), "deficit")
            utility = float(_finite_decimal(_value(alternative, "utility"), "utility"))
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            rejected.append(
                _rejection(
                    index,
                    alternative,
                    None,
                    ceiling,
                    ["VERIFIED_FISCAL_INPUT"],
                    str(error),
                )
            )
            continue
        if not 0 <= utility <= 1:
            rejected.append(
                _rejection(
                    index,
                    alternative,
                    deficit,
                    ceiling,
                    ["UTILITY_SCALE_0_1"],
                    "Utility must use the verified 0..1 scale.",
                )
            )
            continue
        projected_deficit = Real(f"projected_deficit_{index}")
        solver = Solver()
        solver.add(projected_deficit == RealVal(str(deficit)))
        solver.add(projected_deficit >= RealVal("0"))
        solver.add(projected_deficit <= RealVal(str(ceiling)))
        if solver.check() == sat:
            feasible.append(alternative)
            continue
        violated = (
            ["PROJECTED_DEFICIT_NONNEGATIVE"]
            if deficit < 0
            else [
                *(
                    ["DEFICIT_3PCT"]
                    if deficit > STATUTORY_DEFICIT_CEILING
                    else []
                ),
                *(
                    ["SCENARIO_DEFICIT_CEILING"]
                    if requested_ceiling < STATUTORY_DEFICIT_CEILING
                    and deficit > requested_ceiling
                    else []
                ),
            ]
        )
        rejected.append(
            _rejection(
                index,
                alternative,
                deficit,
                ceiling,
                violated,
                "Projected deficit violates a non-overridable CAR constraint.",
            )
        )
    selected = (
        max(
            feasible,
            key=lambda item: (
                float(_value(item, "utility")),
                -float(_value(item, "deficit")),
            ),
        )
        if feasible
        else None
    )
    if hard_stop_reason is not None:
        existing_rejections = {
            int(item["index"]): item
            for item in rejected
            if isinstance(item.get("index"), int)
        }
        hard_stop_rejections: list[dict[str, Any]] = []
        for index, alternative in enumerate(alternatives):
            existing_rejection = existing_rejections.get(index)
            if existing_rejection is not None:
                violated_constraints = list(
                    existing_rejection.get("violated_constraints", [])
                )
                if "DDR_DC_HARD_STOP" not in violated_constraints:
                    violated_constraints.append("DDR_DC_HARD_STOP")
                hard_stop_rejections.append(
                    {
                        **existing_rejection,
                        "violated_constraints": violated_constraints,
                        "reason": (
                            f"{existing_rejection['reason']} {hard_stop_reason}"
                        ),
                    }
                )
                continue
            try:
                hard_stop_deficit = _finite_decimal(
                    _value(alternative, "deficit"), "deficit"
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                hard_stop_deficit = None
            hard_stop_rejections.append(
                _rejection(
                    index,
                    alternative,
                    hard_stop_deficit,
                    ceiling,
                    ["DDR_DC_HARD_STOP"],
                    hard_stop_reason,
                )
            )
        rejected = hard_stop_rejections
        feasible = []
        selected = None
    deficit_constraint_codes = {"DEFICIT_3PCT", "SCENARIO_DEFICIT_CEILING"}
    deficit_rejections = [
        item
        for item in rejected
        if deficit_constraint_codes.intersection(item["violated_constraints"])
    ]
    statutory_rejections = [
        item for item in rejected if "DEFICIT_3PCT" in item["violated_constraints"]
    ]
    scenario_policy_rejections = [
        item
        for item in rejected
        if "SCENARIO_DEFICIT_CEILING" in item["violated_constraints"]
    ]
    non_deficit_rejections = [
        item
        for item in rejected
        if not deficit_constraint_codes.intersection(item["violated_constraints"])
    ]
    calculation_status = (
        "calculated"
        if alternatives and not non_deficit_rejections
        else "partially-calculated"
        if alternatives and deficit_rejections
        else "not-calculated"
    )
    hard_constraints = [
        {
            "code": "DEFICIT_3PCT",
            "formula": "projected_deficit_percent_gdp <= 3.0",
            "requested_ceiling_percent_gdp": float(requested_ceiling),
            "ceiling_percent_gdp": float(STATUTORY_DEFICIT_CEILING),
            "source_tags": ["UU17_2003_P12"],
            "calculation_status": calculation_status,
            "status": (
                "violated"
                if statutory_rejections
                else "not-calculated"
                if non_deficit_rejections or not alternatives
                else "satisfied"
            ),
            "solver": "z3",
        }
    ]
    if requested_ceiling < STATUTORY_DEFICIT_CEILING:
        hard_constraints.append(
            {
                "code": "SCENARIO_DEFICIT_CEILING",
                "formula": f"projected_deficit_percent_gdp <= {requested_ceiling}",
                "ceiling_percent_gdp": float(requested_ceiling),
                "source_tags": [],
                "calculation_status": calculation_status,
                "status": (
                    "violated"
                    if scenario_policy_rejections
                    else "not-calculated"
                    if non_deficit_rejections or not alternatives
                    else "satisfied"
                ),
                "solver": "z3",
            }
        )
    hard_constraints.append(
        {
            "code": "EDUCATION_20PCT",
            "formula": "education_spending_share >= 20%",
            "value": None,
            "source_tags": ["UUD45_P31"],
            "calculation_status": "not-calculated",
            "status": "not-calculated",
            "reason": "Verified post-policy education spending share was not supplied.",
        }
    )
    if hard_stop_reason is not None:
        hard_constraints.insert(
            0,
            {
                "code": "DDR_DC_HARD_STOP",
                "formula": "verified_constraint_violation => INFEASIBLE",
                "source_tags": ["UU17_2003_P12"],
                "calculation_status": "calculated",
                "status": "violated",
                "solver": "z3",
                "reason": hard_stop_reason,
            },
        )
    return CarEvaluation(
        feasible=list(feasible),
        rejected=rejected,
        solver_status=(
            "unsat"
            if hard_stop_reason is not None
            else "sat"
            if feasible
            else "unsat"
            if deficit_rejections and not non_deficit_rejections
            else "invalid-input"
            if alternatives
            else "not-evaluated"
        ),
        selected=selected,
        hard_constraints=hard_constraints,
        status=(
            "INFEASIBLE"
            if hard_stop_reason is not None or alternatives and not feasible
            else "FEASIBLE"
            if feasible
            else "NOT_EVALUATED"
        ),
        messages=CAR_HARD_STOP_MESSAGES.copy()
        if hard_stop_reason is not None
        else None,
    )
