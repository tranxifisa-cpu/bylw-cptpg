from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .config import ExperimentConfig
from .llm import DashScopeClient
from .schemas import AdvisorResponse, HardConstraints, InitialUserInput, PreferenceAgentResponse, PreferenceVector, SimulatedUserResponse


@dataclass
class AgentResult:
    payload: Any


class MultiAgentSystem:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.user_client = DashScopeClient(
            cache_dir=config.llm_cache_dir,
            config=config.user_agent_llm,
        )
        self.preference_client = DashScopeClient(
            cache_dir=config.llm_cache_dir,
            config=config.preference_agent_llm,
        )
        self.advisor_client = DashScopeClient(
            cache_dir=config.llm_cache_dir,
            config=config.advisor_agent_llm,
        )

    def simulate_user(
        self,
        run_key: str,
        trade_date: str,
        market_summary: dict[str, Any],
        reference_point: float,
        budget_limit: float,
        current_portfolio_value: float,
    ) -> AgentResult:
        system_prompt = (
            "你是一名普通A股投资用户的模拟器。"
            "你的任务是在首次交互时，用自然语言表达自己的投资目标、风险偏好和关注点。"
            "只输出 JSON，不要 Markdown，不要解释。"
        )
        user_prompt = (
            f"交易日: {trade_date}\n"
            f"市场摘要: {market_summary}\n"
            f"当前参考点净收益率: {reference_point:.6f}\n"
            f"账户总预算上限: {budget_limit:.2f}\n"
            f"当前账户总价值: {current_portfolio_value:.2f}\n"
            "输出字段: utterance, next_focus。"
        )
        return self._call_agent(
            client=self.user_client,
            namespace=f"user_init/{run_key}/{trade_date}",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            validator=lambda payload: InitialUserInput.from_dict(payload).as_dict(),
            constructor=InitialUserInput.from_dict,
        )

    def simulate_feedback(
        self,
        run_key: str,
        trade_date: str,
        market_summary: dict[str, Any],
        reference_point: float,
        day_return: float,
        portfolio_value: float,
        advisor_response: AdvisorResponse,
        action_summary: dict[str, Any],
    ) -> AgentResult:
        system_prompt = (
            "你是一名普通A股投资用户的模拟器，必须符合累积前景理论。"
            "你表现出损失厌恶、参考依赖、对亏损更慢的适应，以及面对回撤时更敏感。"
            "只输出 JSON，不要 Markdown，不要解释。"
        )
        user_prompt = (
            f"交易日: {trade_date}\n"
            f"市场摘要: {market_summary}\n"
            f"当前参考点净收益率: {reference_point:.6f}\n"
            f"当日组合净盈亏金额: {day_return:.2f}\n"
            f"收盘后账户总价值: {portfolio_value:.2f}\n"
            f"投顾建议: {advisor_response.as_dict()}\n"
            f"建议动作摘要: {action_summary}\n"
            "输出字段: utterance, adoption, rating, next_focus。"
            "adoption 只能是 adopt / partial / skip，rating 在 1 到 5 之间。"
        )
        return self._call_agent(
            client=self.user_client,
            namespace=f"user_feedback/{run_key}/{trade_date}",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            validator=lambda payload: SimulatedUserResponse.from_dict(payload).as_dict(),
            constructor=SimulatedUserResponse.from_dict,
        )

    def infer_preference(
        self,
        run_key: str,
        trade_date: str,
        user_response: dict[str, Any],
        previous_preference: PreferenceAgentResponse | None,
    ) -> AgentResult:
        system_prompt = (
            "你是用户偏好Agent。"
            "你的任务是从用户自然语言和近期反馈中抽取结构化偏好约束向量。"
            "只输出 JSON，不要 Markdown，不要解释。"
        )
        user_prompt = (
            f"交易日: {trade_date}\n"
            f"用户反馈: {user_response}\n"
            f"上一期偏好: {previous_preference.as_dict() if previous_preference else {}}\n"
            "输出字段: preference_vector。"
            "preference_vector 必须包含 risk_budget, max_single_weight, turnover_cap, diversification_target, style_tilt。"
            "请严格遵守结构化校验规则："
            "risk_budget 必须为数字且在 [0.01, 1.00]，表示单期最多可动用当前现金买入股票的比例；"
            "max_single_weight 必须为数字且在 (0.01, 1.00]；"
            "turnover_cap 必须为数字且在 [0.05, 0.50]，表示单期最大股票调仓比例；"
            "diversification_target 必须为整数且在 [1, 100]，表示从A股非金融行业资产池中选入投资组合的最多持仓股票数量，不允许小数；"
            "style_tilt 只能是 momentum/value/quality/low_vol/balanced。"
            "只输出合法 JSON 对象，不要输出注释、解释或额外字段。"
        )
        return self._call_agent(
            client=self.preference_client,
            namespace=f"preference/{run_key}/{trade_date}",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            validator=lambda payload: PreferenceAgentResponse.from_dict(payload).as_dict(),
            constructor=PreferenceAgentResponse.from_dict,
        )

    def advise(
        self,
        run_key: str,
        trade_date: str,
        preference: PreferenceVector,
        hard_constraints: HardConstraints,
        action_name: str,
        action_summary: dict[str, Any],
        reference_point: float,
    ) -> AgentResult:
        system_prompt = (
            "你是投资顾问Agent。"
            "你的任务是说明推荐哪些动作，并解释动作为什么符合用户偏好，且提醒主要风险。"
            "只输出 JSON，不要 Markdown，不要解释。"
        )
        user_prompt = (
            f"交易日: {trade_date}\n"
            f"当前参考点净收益率: {reference_point:.6f}\n"
            f"偏好向量: {preference.as_dict()}\n"
            f"硬约束: {hard_constraints.as_dict()}\n"
            f"推荐动作: {action_name}\n"
            f"动作摘要: {action_summary}\n"
            "输出字段: recommended_action, rationale, preference_alignment, risk_note。"
        )
        return self._call_agent(
            client=self.advisor_client,
            namespace=f"advisor/{run_key}/{trade_date}/{action_name}",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            validator=lambda payload: AdvisorResponse.from_dict(payload).as_dict(),
            constructor=AdvisorResponse.from_dict,
        )

    def _call_agent(
        self,
        client: DashScopeClient,
        namespace: str,
        system_prompt: str,
        user_prompt: str,
        validator: Callable[[dict[str, Any]], dict[str, Any]],
        constructor: Callable[[dict[str, Any]], Any],
    ) -> AgentResult:
        last_error = ""
        for _ in range(2):
            try:
                normalized = client.chat_json(
                    namespace=namespace,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    validator=validator,
                )
                return AgentResult(payload=constructor(normalized))
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)
        raise RuntimeError(f"Agent call failed: {last_error}")
