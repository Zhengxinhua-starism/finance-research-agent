"""跨期对比工具。

解决什么问题
    "近三年毛利率变化趋势"这类问题，如果只是把三期数据丢给 LLM 让它自己比，
    会出现三种错误：算错同比（尤其是负基数）、把"增速放缓"说成"下滑"、
    漏掉某一期的异常跳变。这三件事都是确定性计算，不该交给概率模型。
    本模块把对比逻辑做成工具：LLM 只拿结论，不做算术。

核心设计决策
    1. 输出的每个指标都带四个维度：原始值序列、同比变动序列、趋势标签、
       异常标记。趋势标签（growth_accelerating / growth_decelerating / ...）
       是给 LLM 用的受控词汇——比让它自己总结"趋势如何"稳定得多，
       评测时也能直接比对。
    2. 严格区分"增速放缓"和"下滑"。营收从 +42% 降到 +29% 是增速放缓
       （规模仍在增长），不是下滑。这个区分在投研里是常识，但 LLM 经常搞错，
       所以在 classify_trend 里用两期增速的符号和大小关系显式判定。
    3. 比率型指标（毛利率）的变动用**百分点**而非百分比表达。
       17.1% → 18.5% 是 +1.4pp，说成 "+8.2%" 虽然数学正确但会被读者误解。
       输出里两种都给，展示串统一用 pp。
    4. 异常检测阈值 ±30%（md 规定）。但对比率型指标不套用这个阈值——
       毛利率从 20% 变到 26% 是 +30% 相对变动，可这在业务上是重大改善
       而不是"数据异常"。比率型改用 ±5pp 作为显著变动线。

为什么不用其他方案
    - 不在 Writer 的 prompt 里要求"请计算同比"：prompt 约束对算术不起作用，
      模型该错还是错，而且错得没有规律，评测里表现为随机失分。
"""

from __future__ import annotations

import logging
from typing import Any

from data.schemas import (
    SIGNIFICANT_CHANGE_THRESHOLD,
    TREND_LABELS,
    BalanceSheetRow,
    CashFlowRow,
    IncomeRow,
    MetricComparison,
    classify_trend,
    compute_dupont,
    detect_risk_alerts,
    format_amount,
    format_pp,
    format_ratio,
    growth_rate,
    safe_divide,
)
from tools.akshare_tools import AKShareDataClient, get_data_client
from tools.data_client import BaseTool, DataSourceError, normalize_ticker

logger = logging.getLogger(__name__)

# 比率型指标的显著变动线（百分点）。与绝对值指标的 ±30% 相对变动区分。
RATIO_SIGNIFICANT_CHANGE_PP = 5.0

# 参与对比的指标定义：(内部键, 中文名, 取值函数, 类型)
# 类型决定展示格式和异常判定方式
METRIC_SPECS: list[tuple[str, str, str]] = [
    ("revenue", "营业收入", "amount"),
    ("operating_cost", "营业成本", "amount"),
    ("gross_margin", "毛利率", "ratio"),
    ("net_profit", "归母净利润", "amount"),
    ("deducted_net_profit", "扣非净利润", "amount"),
    ("net_margin", "净利率", "ratio"),
    ("rd_expense", "研发费用", "amount"),
    ("rd_ratio", "研发费用占比", "ratio"),
    ("total_assets", "总资产", "amount"),
    ("net_assets", "归母净资产", "amount"),
    ("accounts_receivable", "应收账款", "amount"),
    ("debt_ratio", "资产负债率", "ratio"),
    ("roe", "ROE", "ratio"),
    ("operating_cashflow", "经营活动现金流净额", "amount"),
    ("cashflow_to_profit", "经营现金流/净利润", "raw"),
]


class ComparePeriodsTool(BaseTool):
    """跨期对比工具。"""

    name = "compare_periods"
    description = (
        "对指定 A 股公司做多期（默认 3 期年报）财务指标对比分析。"
        "返回每个指标的历年数值、同比变动、趋势判断（增长加速/增速放缓/改善/恶化/平稳/波动）"
        "以及显著变动标记，并附带杜邦分解（ROE = 净利率 × 资产周转率 × 权益乘数）"
        "和风险预警。适用于「近三年变化趋势」「同比变动」「为什么下降」这类"
        "跨期对比和归因分析问题。所有同比和趋势均由程序计算，可直接引用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "A股股票代码，6位数字，例如 002594",
            },
            "periods": {
                "type": "integer",
                "description": "对比的年报期数，默认 3（趋势分析的最小可用期数），最大 5",
                "default": 3,
            },
            "metrics": {
                "type": "string",
                "description": (
                    "可选，逗号分隔的指标名，用于只对比关心的指标以节省篇幅。"
                    "可选值：revenue, operating_cost, gross_margin, net_profit, "
                    "deducted_net_profit, net_margin, rd_expense, rd_ratio, total_assets, "
                    "net_assets, accounts_receivable, debt_ratio, roe, operating_cashflow, "
                    "cashflow_to_profit。留空则返回全部。"
                ),
            },
        },
        "required": ["ticker"],
    }

    def __init__(self, client: AKShareDataClient | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.client = client or get_data_client()

    def fetch(self, ticker: str, periods: int = 3, metrics: str | None = None) -> dict[str, Any]:
        periods = max(2, min(int(periods), 5))
        code = normalize_ticker(ticker)

        incomes = self._safe_load(lambda: self.client.get_income_statements(code, periods), "利润表")
        balances = self._safe_load(
            lambda: self.client.get_balance_sheets(code, periods), "资产负债表"
        )
        cashflows = self._safe_load(
            lambda: self.client.get_cash_flows(code, periods), "现金流量表"
        )

        if not incomes and not balances:
            raise DataSourceError(f"{code} 未获取到可对比的报表数据")

        # 以三张表报告期的并集为时间轴，缺失的期用 None 占位，
        # 这样 LLM 能看出"这一期某表没数据"，而不是误以为指标为 0
        period_axis = sorted(
            {row.period_end for row in incomes}
            | {row.period_end for row in balances}
            | {row.period_end for row in cashflows}
        )[-periods:]

        income_map = {row.period_end: row for row in incomes}
        balance_map = {row.period_end: row for row in balances}
        cashflow_map = {row.period_end: row for row in cashflows}

        selected = self._parse_metric_filter(metrics)
        comparisons: list[MetricComparison] = []
        anomalies: list[str] = []

        for key, label, value_type in METRIC_SPECS:
            if selected and key not in selected:
                continue
            values = [
                self._extract(key, income_map.get(p), balance_map.get(p), cashflow_map.get(p))
                for p in period_axis
            ]
            if all(value is None for value in values):
                continue
            comparison = self._build_comparison(key, label, value_type, values)
            comparisons.append(comparison)
            if comparison.anomaly:
                anomalies.append(f"{label}: {comparison.anomaly_reason}")

        # 杜邦分解与风险预警：与 get_risk_snapshot 共用同一套实现，
        # 保证两个工具对同一家公司给出的结论不会互相打架
        latest = period_axis[-1] if period_axis else None
        previous = period_axis[-2] if len(period_axis) > 1 else None
        dupont = None
        if latest and income_map.get(latest) and balance_map.get(latest):
            dupont = compute_dupont(
                income_map[latest],
                balance_map[latest],
                income_map.get(previous) if previous else None,
                balance_map.get(previous) if previous else None,
            )

        ordered_desc = sorted(period_axis, reverse=True)
        alerts = detect_risk_alerts(
            [income_map[p] for p in ordered_desc if p in income_map],
            [balance_map[p] for p in ordered_desc if p in balance_map],
            [cashflow_map[p] for p in ordered_desc if p in cashflow_map],
        )

        return {
            "company": self.client.get_company_name(code),
            "ticker": code,
            "periods": [p.isoformat() for p in period_axis],
            "metrics_comparison": [self._comparison_payload(c) for c in comparisons],
            "anomalies": anomalies,
            "dupont": dupont.to_display() if dupont else None,
            "risk_alerts": [alert.model_dump() for alert in alerts],
            "source": "AKShare 新浪财报接口 · 合并报表 · 年报口径",
            "calculation_note": (
                "同比变动、趋势判断、异常标记均由程序按固定公式计算，可直接引用；"
                "比率型指标的变动以百分点(pp)表示，金额型以百分比表示。"
                "ROE 计算口径：归母净利润 / 期末归母股东权益（摊薄）；"
                "杜邦拆解中资产周转率与权益乘数使用期末数，而非期初期末平均。"
            ),
            "trend_legend": TREND_LABELS,
        }

    # ---------------- 内部实现 ----------------

    @staticmethod
    def _parse_metric_filter(metrics: str | None) -> set[str]:
        if not metrics:
            return set()
        valid = {key for key, _, _ in METRIC_SPECS}
        requested = {item.strip() for item in str(metrics).split(",") if item.strip()}
        unknown = requested - valid
        if unknown:
            logger.warning("忽略未知的对比指标: %s", unknown)
        return requested & valid

    @staticmethod
    def _extract(
        key: str,
        income: IncomeRow | None,
        balance: BalanceSheetRow | None,
        cashflow: CashFlowRow | None,
    ) -> float | None:
        """从三张表里取指标值。跨表指标（ROE）在这里组合。"""
        if key in {"revenue", "operating_cost", "net_profit", "deducted_net_profit", "rd_expense"}:
            return getattr(income, key, None) if income else None
        if key in {"gross_margin", "net_margin", "rd_ratio"}:
            return getattr(income, key, None) if income else None
        if key in {"total_assets", "net_assets", "accounts_receivable", "debt_ratio"}:
            return getattr(balance, key, None) if balance else None
        if key == "roe":
            if income is None or balance is None:
                return None
            return safe_divide(income.net_profit, balance.net_assets)
        if key == "operating_cashflow":
            return cashflow.operating_cashflow if cashflow else None
        if key == "cashflow_to_profit":
            if cashflow is None:
                return None
            profit = income.net_profit if income and income.net_profit else cashflow.net_profit
            return safe_divide(cashflow.operating_cashflow, profit)
        return None

    def _build_comparison(
        self, key: str, label: str, value_type: str, values: list[float | None]
    ) -> MetricComparison:
        yoy_values: list[float | None] = [None]
        yoy_display: list[str] = [""]

        for index in range(1, len(values)):
            current, previous = values[index], values[index - 1]
            if value_type == "ratio" and current is not None and previous is not None:
                # 比率用百分点差表达，同时保留相对变动供程序使用
                change_pp = (current - previous) * 100
                yoy_values.append(change_pp)
                yoy_display.append(format_pp(change_pp))
            else:
                change = growth_rate(current, previous)
                yoy_values.append(change)
                yoy_display.append(format_ratio(change) if change is not None else "—")

        trend, anomaly, anomaly_reason = classify_trend(values)

        # 比率型指标改用百分点阈值判定异常，避免"20%→26%"被误报
        if value_type == "ratio":
            ratio_changes = [v for v in yoy_values[1:] if v is not None]
            anomaly = any(abs(change) > RATIO_SIGNIFICANT_CHANGE_PP for change in ratio_changes)
            anomaly_reason = (
                f"单期变动超过 ±{RATIO_SIGNIFICANT_CHANGE_PP}pp："
                + ", ".join(format_pp(c) for c in ratio_changes if abs(c) > RATIO_SIGNIFICANT_CHANGE_PP)
                if anomaly
                else ""
            )

        return MetricComparison(
            metric=key,
            metric_label=label,
            values=values,
            values_display=[self._format_value(v, value_type) for v in values],
            yoy_changes=yoy_display,
            yoy_values=yoy_values,
            trend=trend,  # type: ignore[arg-type]
            anomaly=anomaly,
            anomaly_reason=anomaly_reason,
        )

    @staticmethod
    def _format_value(value: float | None, value_type: str) -> str:
        if value is None:
            return "—"
        if value_type == "amount":
            return format_amount(value)
        if value_type == "ratio":
            return format_ratio(value)
        return f"{value:.2f}"

    @staticmethod
    def _comparison_payload(comparison: MetricComparison) -> dict[str, Any]:
        return {
            "metric": comparison.metric,
            "metric_label": comparison.metric_label,
            "values": comparison.values,
            "values_display": comparison.values_display,
            "yoy_changes": comparison.yoy_changes,
            "trend": comparison.trend,
            "trend_label": TREND_LABELS.get(comparison.trend, comparison.trend),
            "anomaly": comparison.anomaly,
            "anomaly_reason": comparison.anomaly_reason,
        }

    @staticmethod
    def _safe_load(loader: Any, label: str) -> list[Any]:
        try:
            return loader()
        except DataSourceError as exc:
            logger.warning("跨期对比获取%s失败: %s", label, exc)
            return []


__all__ = ["ComparePeriodsTool", "RATIO_SIGNIFICANT_CHANGE_PP", "SIGNIFICANT_CHANGE_THRESHOLD"]
