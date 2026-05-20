from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


STYLE_TILTS = {"momentum", "value", "quality", "low_vol", "balanced"}
ADOPTION_CHOICES = {"adopt", "partial", "skip"}
MIN_SINGLE_STOCK_WEIGHT = 0.01
MIN_RISK_BUDGET = 0.01
MAX_PORTFOLIO_HOLDINGS = 100


def _require_number(payload: dict[str, Any], key: str, lower: float, upper: float) -> float:
    if key not in payload:
        raise ValueError(f"Missing required field: {key}")
    try:
        number = float(payload[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field {key} must be numeric") from exc
    if number < lower or number > upper:
        raise ValueError(f"Field {key} must be in [{lower}, {upper}]")
    return number


def _require_text(payload: dict[str, Any], key: str) -> str:
    if key not in payload:
        raise ValueError(f"Missing required field: {key}")
    text = str(payload[key]).strip()
    if not text:
        raise ValueError(f"Field {key} must be non-empty")
    return text


def _require_greater_than(payload: dict[str, Any], key: str, lower: float, upper: float) -> float:
    number = _require_number(payload, key, lower, upper)
    if number <= lower:
        raise ValueError(f"Field {key} must be greater than {lower} and at most {upper}")
    return number


@dataclass(frozen=True)
class PreferenceVector:
    risk_budget: float
    max_single_weight: float
    turnover_cap: float
    diversification_target: int
    style_tilt: str

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "PreferenceVector":
        style_tilt = _require_text(payload, "style_tilt")
        if style_tilt not in STYLE_TILTS:
            raise ValueError(f"Invalid style_tilt: {style_tilt}")
        diversification_target = int(round(_require_number(payload, "diversification_target", 1, MAX_PORTFOLIO_HOLDINGS)))
        return PreferenceVector(
            risk_budget=_require_number(payload, "risk_budget", MIN_RISK_BUDGET, 1.00),
            max_single_weight=_require_greater_than(payload, "max_single_weight", MIN_SINGLE_STOCK_WEIGHT, 1.00),
            turnover_cap=_require_number(payload, "turnover_cap", 0.05, 0.50),
            diversification_target=diversification_target,
            style_tilt=style_tilt,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HardConstraints:
    risk_budget: float
    max_single_weight: float
    turnover_cap: float
    diversification_target: int

    @staticmethod
    def from_preference(preference: PreferenceVector) -> "HardConstraints":
        return HardConstraints(
            risk_budget=preference.risk_budget,
            max_single_weight=preference.max_single_weight,
            turnover_cap=preference.turnover_cap,
            diversification_target=preference.diversification_target,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InitialUserInput:
    utterance: str
    next_focus: str

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "InitialUserInput":
        return InitialUserInput(
            utterance=_require_text(payload, "utterance"),
            next_focus=_require_text(payload, "next_focus"),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SimulatedUserResponse:
    utterance: str
    adoption: str
    rating: float
    next_focus: str

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "SimulatedUserResponse":
        adoption = _require_text(payload, "adoption")
        if adoption not in ADOPTION_CHOICES:
            raise ValueError(f"Invalid adoption: {adoption}")
        return SimulatedUserResponse(
            utterance=_require_text(payload, "utterance"),
            adoption=adoption,
            rating=_require_number(payload, "rating", 1.0, 5.0),
            next_focus=_require_text(payload, "next_focus"),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreferenceAgentResponse:
    preference_vector: PreferenceVector
    hard_constraints: HardConstraints

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "PreferenceAgentResponse":
        preference_payload = payload.get("preference_vector", payload)
        preference_vector = PreferenceVector.from_dict(preference_payload)
        hard_constraints = HardConstraints.from_preference(preference_vector)
        return PreferenceAgentResponse(
            preference_vector=preference_vector,
            hard_constraints=hard_constraints,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "preference_vector": self.preference_vector.as_dict(),
            "hard_constraints": self.hard_constraints.as_dict(),
        }


@dataclass(frozen=True)
class AdvisorResponse:
    recommended_action: str
    rationale: str
    preference_alignment: str
    risk_note: str

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "AdvisorResponse":
        action_name = _require_text(payload, "recommended_action")
        return AdvisorResponse(
            recommended_action=action_name,
            rationale=_require_text(payload, "rationale"),
            preference_alignment=_require_text(payload, "preference_alignment"),
            risk_note=_require_text(payload, "risk_note"),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
