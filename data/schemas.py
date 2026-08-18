"""财务数据模型与指标计算规则。

解决什么问题
    AKShare 返回的是中文列名的宽表 DataFrame，列名在不同接口、不同年份之间
    还会变（"营业总收入" / "营业收入" / "一、营业总收入" 都出现过）。
    如果让每个工具各自去 df["营业收入"] 取值，任何一次列名变动都会造成
    KeyError 或更糟的静默错值。本模块定义标准化后的数据模型，并把
    md "金融业务领域知识"章节里的指标公式和预警阈值集中实现一次。

核心设计决策
    1. 单位在模型层就统一死：金额一律为**元**，比率一律为**小数**
       （0.218 表示 21.8%）。AKShare 有的接口返回元、有的返回万元、
       财务摘要接口甚至返回"7771.02亿"这样的字符串。不在入口统一，
       后面证据门禁比对数字时会出现"差了 1e8 倍"的经典事故。
    2. 所有派生指标（毛利率、ROE、杜邦拆解、风险预警）都实现成模型上的
       computed 属性或独立纯函数，**不让 LLM 算**。LLM 做算术是不可靠的，
       而这些公式是确定的；把它们交给代码，LLM 只负责解释"为什么变化"。
    3. 字段全部可选（float | None）。财报缺项是常态：非制造业没有营业成本、
       未上市子公司没有商誉。用 0 填充会让"没有商誉"和"商誉为 0"无法区分，
       进而在预警计算里产生假阴性。所有计算函数都显式处理 None。
    4. 预警阈值做成模块级常量而不是散落在判断语句里，因为它们是业务规则，
       会被评测脚本和研报模板同时引用，必须只有一个来源。

为什么不用其他方案
    - 不用 pandas DataFrame 在模块间传递：DataFrame 没有 schema 约束，
      而且不能直接 JSON 序列化进 trace 和 Redis。DataFrame 只在
      akshare_tools.py 的解析阶段用，出了那一层就转成这里的模型。
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

# ============================================================
# 业务阈值常量（来自 md "核心财务指标体系"）
# ============================================================

# 盈利能力预警
GROSS_MARGIN_DROP_ALERT_PP = 3.0  # 毛利率同比下降超过 3 个百分点
NET_MARGIN_DROP_ALERT_PP = 2.0  # 净利率同比下降超过 2 个百分点
ROE_LOW_THRESHOLD = 0.08  # ROE 低于 8% 视为偏低
PROFIT_REVENUE_DIVERGENCE_PP = 10.0  # 扣非净利增速与营收增速背离超过 10pp

# 风险预警
RECEIVABLE_GROWTH_MULTIPLE = 1.5  # 应收增速 > 营收增速 × 1.5
CASHFLOW_TO_PROFIT_ALERT = 0.8  # 经营现金流/净利润 < 0.8
DEBT_RATIO_ALERT = 0.70  # 资产负债率 > 70%
GOODWILL_TO_EQUITY_ALERT = 0.30  # 商誉/净资产 > 30%

# 跨期对比：变动率超过 ±30% 标记为显著变动
SIGNIFICANT_CHANGE_THRESHOLD = 0.30

# 单位换算
YI = 100_000_000.0  # 亿
WAN = 10_000.0  # 万


class FinanceModel(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)


# ============================================================
# 基础工具函数
# ============================================================


def safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    """安全除法。分母为 0 或任一方为 None 时返回 None 而不是 0。

    返回 0 会让下游把"无法计算"误读成"值为零"——毛利率算不出来
    和毛利率为 0 在投研上是完全不同的结论。
    """
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def growth_rate(current: float | None, previous: float | None) -> float | None:
    """同比增速 = (本期 - 上期) / |上期|。

    分母取绝对值：上期为负（亏损）时，用有符号分母会让"亏损收窄"
    算出负增长，符号完全反了。这是财务计算里的经典坑。
    """
    if current is None or previous is None or previous == 0:
        return None
    return (current - previous) / abs(previous)


def to_percentage_points(ratio: float | None) -> float | None:
    """小数比率 → 百分点。"""
    return None if ratio is None else ratio * 100


def format_amount(value: float | None, unit: Literal["auto", "yi", "wan", "yuan"] = "auto") -> str:
    """把元金额格式化成中文习惯写法。"""
    if value is None:
        return "—"
    if unit == "yuan":
        return f"{value:,.0f}元"
    if unit == "wan" or (unit == "auto" and abs(value) < YI):
        return f"{value / WAN:,.2f}万元"
    return f"{value / YI:,.2f}亿元"


def format_ratio(value: float | None, digits: int = 2) -> str:
    """小数比率 → 百分数字符串。"""
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def format_pp(value: float | None, digits: int = 2) -> str:
    """百分点变动，带正负号。"""
    if value is None:
        return "—"
    return f"{value:+.{digits}f}pp"


# ============================================================
# 报表行模型
# ============================================================


class IncomeRow(FinanceModel):
    """单期利润表（标准化后）。金额单位：元。"""

    period_end: date
    report_type: Literal["annual", "interim", "q1", "q3", "unknown"] = "unknown"
    revenue: float | None = None  # 营业总收入
    operating_cost: float | None = None  # 营业成本
    operating_profit: float | None = None  # 营业利润
    total_profit: float | None = None  # 利润总额
    net_profit: float | None = None  # 归母净利润（默认口径）
    total_net_profit: float | None = None  # 净利润总额（含少数股东）
    deducted_net_profit: float | None = None  # 扣非归母净利润
    rd_expense: float | None = None  # 研发费用
    selling_expense: float | None = None
    admin_expense: float | None = None
    finance_expense: float | None = None
    # 以下科目决定"净利率降幅为何小于毛利率降幅"，是利润构成分析的关键。
    # 它们一直在新浪利润表的 83 列里，只是之前没接（见 docs/data_coverage.md）。
    tax_and_surcharges: float | None = None  # 营业税金及附加
    asset_impairment: float | None = None  # 资产减值损失（通常为负）
    credit_impairment: float | None = None  # 信用减值损失（通常为负）
    investment_income: float | None = None  # 投资收益
    fair_value_change: float | None = None  # 公允价值变动收益
    other_income: float | None = None  # 其他收益（政府补助为主）
    non_operating_income: float | None = None  # 营业外收入
    non_operating_expense: float | None = None  # 营业外支出
    income_tax: float | None = None  # 所得税费用

    @property
    def gross_profit(self) -> float | None:
        if self.revenue is None or self.operating_cost is None:
            return None
        return self.revenue - self.operating_cost

    @property
    def period_expense(self) -> float | None:
        """期间费用 = 销售 + 管理 + 财务 + 研发。

        任一项缺失就返回 None 而不是按 0 处理：财务费用为负（利息收入大于
        支出）是正常的，用 0 填充缺失项会让费用总额被低估，
        而这个数会直接进研报。
        """
        parts = [
            self.selling_expense,
            self.admin_expense,
            self.finance_expense,
            self.rd_expense,
        ]
        if any(part is None for part in parts):
            return None
        return sum(part for part in parts if part is not None)

    @property
    def period_expense_ratio(self) -> float | None:
        """期间费用率 = 期间费用 / 营收。"""
        return safe_divide(self.period_expense, self.revenue)

    @property
    def selling_ratio(self) -> float | None:
        return safe_divide(self.selling_expense, self.revenue)

    @property
    def admin_ratio(self) -> float | None:
        return safe_divide(self.admin_expense, self.revenue)

    @property
    def non_operating_net(self) -> float | None:
        """营业外收支净额。"""
        if self.non_operating_income is None and self.non_operating_expense is None:
            return None
        return (self.non_operating_income or 0.0) - (self.non_operating_expense or 0.0)

    @property
    def non_core_profit(self) -> float | None:
        """非经营性损益合计（投资收益 + 公允价值变动 + 其他收益 + 营业外净额
        + 减值损失）。

        这是"利润里有多少不是主营业务挣的"的直接度量。注意减值损失在报表里
        本身就是负值，直接相加即可，不要再取负。
        """
        parts = [
            self.investment_income,
            self.fair_value_change,
            self.other_income,
            self.non_operating_net,
            self.asset_impairment,
            self.credit_impairment,
        ]
        if all(part is None for part in parts):
            return None
        return sum(part for part in parts if part is not None)

    @property
    def non_core_profit_ratio(self) -> float | None:
        """非经营性损益占归母净利润的比例。越高说明利润质量越依赖非主营。"""
        return safe_divide(self.non_core_profit, self.net_profit)

    @property
    def effective_tax_rate(self) -> float | None:
        """实际税率 = 所得税费用 / (归母净利润 + 所得税费用)。

        分母用税前利润的近似值。严格应该用利润表的"利润总额"，
        但那个字段在部分标的上缺失，这个近似在归母与全口径差异不大时可用。
        """
        if self.income_tax is None or self.net_profit is None:
            return None
        pretax = self.net_profit + self.income_tax
        return safe_divide(self.income_tax, pretax)

    @property
    def gross_margin(self) -> float | None:
        """毛利率 = (营收 - 营业成本) / 营收。"""
        return safe_divide(self.gross_profit, self.revenue)

    @property
    def net_margin(self) -> float | None:
        """净利率 = 归母净利润 / 营收。"""
        return safe_divide(self.net_profit, self.revenue)

    @property
    def rd_ratio(self) -> float | None:
        """研发费用占比 = 研发费用 / 营收。"""
        return safe_divide(self.rd_expense, self.revenue)

    def to_display(self) -> dict[str, Any]:
        return {
            "date": self.period_end.isoformat(),
            "revenue": self.revenue,
            "revenue_display": format_amount(self.revenue),
            "operating_cost": self.operating_cost,
            "gross_profit": self.gross_profit,
            "gross_margin": self.gross_margin,
            "gross_margin_display": format_ratio(self.gross_margin),
            "net_profit": self.net_profit,
            "net_profit_display": format_amount(self.net_profit),
            "deducted_net_profit": self.deducted_net_profit,
            "net_margin": self.net_margin,
            "rd_expense": self.rd_expense,
            "rd_ratio": self.rd_ratio,
            # 期间费用：数据一直抓到了，但之前没输出，导致研报反复写
            # "缺乏期间费用明细"（docs/data_coverage.md 里标为 🟡）
            "selling_expense": self.selling_expense,
            "admin_expense": self.admin_expense,
            "finance_expense": self.finance_expense,
            "period_expense": self.period_expense,
            "period_expense_ratio": self.period_expense_ratio,
            "period_expense_ratio_display": format_ratio(self.period_expense_ratio),
            # 利润构成：解释"净利率降幅为何小于毛利率降幅"
            "tax_and_surcharges": self.tax_and_surcharges,
            "asset_impairment": self.asset_impairment,
            "credit_impairment": self.credit_impairment,
            "investment_income": self.investment_income,
            "fair_value_change": self.fair_value_change,
            "other_income": self.other_income,
            "non_operating_net": self.non_operating_net,
            "non_core_profit": self.non_core_profit,
            "non_core_profit_ratio": self.non_core_profit_ratio,
            "income_tax": self.income_tax,
            "effective_tax_rate": self.effective_tax_rate,
        }


class BalanceSheetRow(FinanceModel):
    """单期资产负债表（标准化后）。金额单位：元。"""

    period_end: date
    total_assets: float | None = None
    total_liabilities: float | None = None
    net_assets: float | None = None  # 归母股东权益
    total_equity: float | None = None  # 所有者权益合计（含少数股东）
    accounts_receivable: float | None = None  # 应收账款
    inventory: float | None = None  # 存货
    goodwill: float | None = None  # 商誉
    short_term_debt: float | None = None  # 短期借款
    long_term_debt: float | None = None  # 长期借款
    monetary_funds: float | None = None  # 货币资金
    bonds_payable: float | None = None  # 应付债券
    current_portion_non_current_liabilities: float | None = None  # 一年内到期的非流动负债
    contract_liabilities: float | None = None  # 合同负债（原预收账款）
    fixed_assets: float | None = None  # 固定资产净值
    construction_in_progress: float | None = None  # 在建工程
    development_expenditure: float | None = None  # 开发支出（研发资本化的存量）

    @property
    def debt_ratio(self) -> float | None:
        """资产负债率 = 总负债 / 总资产。"""
        return safe_divide(self.total_liabilities, self.total_assets)

    @property
    def interest_bearing_debt(self) -> float | None:
        """有息负债 = 短期借款 + 长期借款 + 应付债券 + 一年内到期的非流动负债。

        与"总负债"的区别很关键：总负债里包含应付账款、合同负债这些
        无息的经营性负债。用总负债评价偿债压力会高估风险——
        对比亚迪这种对上游占款能力强的公司尤其失真。
        这里全为 None 才返回 None，部分缺失按 0 计（缺失项通常确实不存在）。
        """
        parts = [
            self.short_term_debt,
            self.long_term_debt,
            self.bonds_payable,
            self.current_portion_non_current_liabilities,
        ]
        if all(part is None for part in parts):
            return None
        return sum(part for part in parts if part is not None)

    @property
    def interest_bearing_debt_ratio(self) -> float | None:
        """有息负债率 = 有息负债 / 总资产。"""
        return safe_divide(self.interest_bearing_debt, self.total_assets)

    @property
    def net_cash(self) -> float | None:
        """净现金 = 货币资金 - 有息负债。为负表示净负债。"""
        if self.monetary_funds is None or self.interest_bearing_debt is None:
            return None
        return self.monetary_funds - self.interest_bearing_debt

    @property
    def capex_intensity(self) -> float | None:
        """在建工程 / 固定资产净值，衡量产能扩张强度。"""
        return safe_divide(self.construction_in_progress, self.fixed_assets)

    @property
    def goodwill_to_equity(self) -> float | None:
        """商誉占净资产比。"""
        return safe_divide(self.goodwill, self.net_assets)

    @property
    def equity_multiplier(self) -> float | None:
        """权益乘数 = 总资产 / 净资产（杜邦分析的杠杆因子）。"""
        return safe_divide(self.total_assets, self.net_assets)

    def to_display(self) -> dict[str, Any]:
        return {
            "date": self.period_end.isoformat(),
            "total_assets": self.total_assets,
            "total_liabilities": self.total_liabilities,
            "net_assets": self.net_assets,
            "accounts_receivable": self.accounts_receivable,
            "inventory": self.inventory,
            "goodwill": self.goodwill,
            "short_term_debt": self.short_term_debt,
            "debt_ratio": self.debt_ratio,
            "debt_ratio_display": format_ratio(self.debt_ratio),
            "goodwill_to_equity": self.goodwill_to_equity,
            "monetary_funds": self.monetary_funds,
            "bonds_payable": self.bonds_payable,
            "contract_liabilities": self.contract_liabilities,
            "fixed_assets": self.fixed_assets,
            "construction_in_progress": self.construction_in_progress,
            "development_expenditure": self.development_expenditure,
            # 有息负债比总负债更能反映真实偿债压力：总负债里含应付账款、
            # 合同负债这些无息经营性负债
            "interest_bearing_debt": self.interest_bearing_debt,
            "interest_bearing_debt_ratio": self.interest_bearing_debt_ratio,
            "net_cash": self.net_cash,
            "capex_intensity": self.capex_intensity,
        }


class CashFlowRow(FinanceModel):
    """单期现金流量表（标准化后）。金额单位：元。"""

    period_end: date
    operating_cashflow: float | None = None  # 经营活动产生的现金流量净额
    investing_cashflow: float | None = None
    financing_cashflow: float | None = None
    net_increase_in_cash: float | None = None
    capital_expenditure: float | None = None  # 购建固定资产等支付的现金
    # 用于计算 cashflow_to_profit，从利润表带过来
    net_profit: float | None = None

    @property
    def cashflow_to_profit(self) -> float | None:
        """经营现金流 / 净利润，衡量利润质量。"""
        return safe_divide(self.operating_cashflow, self.net_profit)

    @property
    def free_cashflow(self) -> float | None:
        if self.operating_cashflow is None or self.capital_expenditure is None:
            return None
        return self.operating_cashflow - self.capital_expenditure

    @property
    def capex_to_ocf(self) -> float | None:
        """资本开支 / 经营现金流。大于 1 表示扩张速度超过自身造血能力，
        差额要靠融资或消耗存量现金补。实测比亚迪 2025 年为 2.65。
        """
        return safe_divide(self.capital_expenditure, self.operating_cashflow)

    def to_display(self) -> dict[str, Any]:
        return {
            "date": self.period_end.isoformat(),
            "operating_cashflow": self.operating_cashflow,
            "operating_cashflow_display": format_amount(self.operating_cashflow),
            "investing_cashflow": self.investing_cashflow,
            "financing_cashflow": self.financing_cashflow,
            "net_profit": self.net_profit,
            "cashflow_to_profit": self.cashflow_to_profit,
            "capital_expenditure": self.capital_expenditure,
            "capital_expenditure_display": format_amount(self.capital_expenditure),
            "free_cashflow": self.free_cashflow,
            "free_cashflow_display": format_amount(self.free_cashflow),
            "capex_to_ocf": self.capex_to_ocf,
        }


class SegmentRow(FinanceModel):
    """主营构成的一行：某个分部在某个报告期的收入、成本、毛利率、占比。"""

    period_end: date
    category_type: str  # 按行业分类 / 按产品分类 / 按地区分类
    segment_name: str
    revenue: float | None = None
    cost: float | None = None
    profit: float | None = None
    revenue_share: float | None = None  # 收入占比（小数）
    gross_margin: float | None = None

    def to_display(self) -> dict[str, Any]:
        return {
            "segment": self.segment_name,
            "revenue": self.revenue,
            "revenue_display": format_amount(self.revenue),
            "revenue_share": self.revenue_share,
            "revenue_share_display": format_ratio(self.revenue_share),
            "gross_margin": self.gross_margin,
            "gross_margin_display": format_ratio(self.gross_margin),
        }


class SegmentEffect(FinanceModel):
    """整体毛利率变动的结构分解结果。

    这是分部分析的核心产出：把整体毛利率的变动拆成两部分——
    - 结构效应：各分部占比变了（高毛利业务占比下降会拉低整体，
      即使每个分部自身毛利率都没动）；
    - 自身效应：各分部的毛利率自己变了。
    不做这个拆解，就只能说"整体毛利率降了 1.7pp"，说不清是业务结构变化
    还是经营恶化——而这两者的投资含义完全不同。
    """

    overall_margin_base: float | None = None
    overall_margin_current: float | None = None
    overall_change_pp: float | None = None
    mix_effect_pp: float | None = None  # 结构效应（百分点）
    own_effect_pp: float | None = None  # 各分部自身变化效应（百分点）
    dominant: str = ""  # 主导因素的文字说明
    segment_contributions: list[dict[str, Any]] = Field(default_factory=list)


class PriceBar(FinanceModel):
    """单日 K 线。"""

    trade_date: date
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    turnover: float | None = None
    change_pct: float | None = None


class NewsItem(FinanceModel):
    """一条新闻。"""

    title: str
    url: str | None = None
    published_at: date | None = None
    source: str = "未知来源"
    summary: str = ""

    def to_display(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "source": self.source,
            "summary": self.summary[:300],
        }


# ============================================================
# 核心指标与分析结果
# ============================================================


# ROE 自算默认口径。必须在证据正文和研报里写明：摊薄 vs 平均净资产
# 实测可差 2~4pp，混比会把口径差当成经营变化。
ROE_SCOPE_DILUTED = "归母净利润 / 期末归母股东权益（摊薄口径）"
ROE_SCOPE_AVERAGE = "净资产收益率（平均口径，来自财务摘要；资产负债表获取失败）"


class DupontDecomposition(FinanceModel):
    """杜邦分解：ROE = 净利率 × 资产周转率 × 权益乘数。"""

    roe: float | None = None
    net_margin: float | None = None
    asset_turnover: float | None = None
    equity_multiplier: float | None = None
    # 与上期对比时，各因子的变动（百分比变化，非百分点）
    net_margin_change: float | None = None
    asset_turnover_change: float | None = None
    equity_multiplier_change: float | None = None
    driver: str | None = None  # 主导因子的文字说明
    roe_scope: str = ROE_SCOPE_DILUTED

    def to_display(self) -> dict[str, Any]:
        return {
            "roe": self.roe,
            "roe_display": format_ratio(self.roe),
            "net_margin": self.net_margin,
            "asset_turnover": self.asset_turnover,
            "equity_multiplier": self.equity_multiplier,
            "driver": self.driver,
            "formula": "ROE = 净利率 × 资产周转率 × 权益乘数",
            "roe_scope": self.roe_scope,
            "scope_note": (
                f"ROE 计算口径：{self.roe_scope}。"
                "资产周转率与权益乘数使用期末总资产/净资产，而非期初期末平均。"
            ),
        }


class RiskAlert(FinanceModel):
    """一条风险预警。"""

    code: str  # 机器可读的规则标识，例如 "receivable_divergence"
    level: Literal["info", "warning", "danger"]
    title: str
    detail: str
    metrics: dict[str, float | None] = Field(default_factory=dict)
    threshold_desc: str = ""


class FinancialMetrics(FinanceModel):
    """核心财务指标汇总，对应 md 中 get_financial_metrics 的返回结构。"""

    company: str
    ticker: str
    as_of: date
    revenue: float | None = None
    net_profit: float | None = None
    deducted_net_profit: float | None = None
    gross_margin: float | None = None
    net_margin: float | None = None
    roe: float | None = None
    roe_scope: str = ROE_SCOPE_DILUTED
    revenue_yoy: float | None = None
    net_profit_yoy: float | None = None
    deducted_net_profit_yoy: float | None = None
    rd_ratio: float | None = None
    debt_ratio: float | None = None
    cashflow_to_profit: float | None = None
    source: str = "AKShare"
    disclosure_date: date | None = None

    def to_tool_payload(self) -> dict[str, Any]:
        """转成 md 规定的工具返回格式。"""
        return {
            "company": self.company,
            "ticker": self.ticker,
            "as_of": self.as_of.isoformat(),
            "metrics": {
                "revenue": self.revenue,
                "net_profit": self.net_profit,
                "deducted_net_profit": self.deducted_net_profit,
                "gross_margin": self.gross_margin,
                "net_margin": self.net_margin,
                "roe": self.roe,
                "revenue_yoy": self.revenue_yoy,
                "net_profit_yoy": self.net_profit_yoy,
                "deducted_net_profit_yoy": self.deducted_net_profit_yoy,
                "rd_ratio": self.rd_ratio,
                "debt_ratio": self.debt_ratio,
                "cashflow_to_profit": self.cashflow_to_profit,
            },
            "metrics_display": {
                "revenue": format_amount(self.revenue),
                "net_profit": format_amount(self.net_profit),
                "gross_margin": format_ratio(self.gross_margin),
                "net_margin": format_ratio(self.net_margin),
                "roe": format_ratio(self.roe),
                "revenue_yoy": format_ratio(self.revenue_yoy),
                "net_profit_yoy": format_ratio(self.net_profit_yoy),
            },
            "source": self.source,
            "roe_scope": self.roe_scope,
            "disclosure_date": self.disclosure_date.isoformat() if self.disclosure_date else None,
            "unit_note": "金额单位为元，比率为小数（0.218 表示 21.8%）",
        }

    def numbers_for_evidence(self) -> dict[str, float]:
        """抽出可被证据门禁校验的数字。None 值不进证据池。"""
        candidates = {
            "revenue": self.revenue,
            "net_profit": self.net_profit,
            "deducted_net_profit": self.deducted_net_profit,
            "gross_margin": self.gross_margin,
            "net_margin": self.net_margin,
            "roe": self.roe,
            "revenue_yoy": self.revenue_yoy,
            "net_profit_yoy": self.net_profit_yoy,
            "rd_ratio": self.rd_ratio,
            "debt_ratio": self.debt_ratio,
            "cashflow_to_profit": self.cashflow_to_profit,
        }
        return {key: value for key, value in candidates.items() if value is not None}


class MetricComparison(FinanceModel):
    """单个指标的跨期对比结果（对应 md compare_periods 的返回结构）。"""

    metric: str
    metric_label: str = ""
    values: list[float | None] = Field(default_factory=list)
    values_display: list[str] = Field(default_factory=list)
    yoy_changes: list[str] = Field(default_factory=list)
    yoy_values: list[float | None] = Field(default_factory=list)
    trend: Literal[
        "growth_accelerating",
        "growth_decelerating",
        "improving",
        "deteriorating",
        "stable",
        "volatile",
        "insufficient_data",
    ] = "insufficient_data"
    anomaly: bool = False
    anomaly_reason: str = ""


class PeriodComparison(FinanceModel):
    """跨期对比工具的完整返回。"""

    company: str
    ticker: str
    periods: list[str] = Field(default_factory=list)
    metrics_comparison: list[MetricComparison] = Field(default_factory=list)
    anomalies: list[str] = Field(default_factory=list)
    dupont: DupontDecomposition | None = None
    risk_alerts: list[RiskAlert] = Field(default_factory=list)


# ============================================================
# 指标计算（md "金融业务领域知识"的代码化实现）
# ============================================================


def compute_dupont(
    income: IncomeRow,
    balance: BalanceSheetRow,
    previous_income: IncomeRow | None = None,
    previous_balance: BalanceSheetRow | None = None,
) -> DupontDecomposition:
    """杜邦分解，并在有上期数据时判断 ROE 变动的主导因子。

    资产周转率用期末总资产而非平均总资产：AKShare 单次调用拿不到期初数，
    再拉一期会多一次网络请求和一次可能的失败。用期末值的偏差在
    资产规模稳定的公司上很小，而对定性判断（哪个因子主导）没有影响。
    这个近似必须写在研报的口径说明里，不能假装它是精确值。
    """
    net_margin = income.net_margin
    asset_turnover = safe_divide(income.revenue, balance.total_assets)
    equity_multiplier = balance.equity_multiplier
    roe = safe_divide(income.net_profit, balance.net_assets)

    decomposition = DupontDecomposition(
        roe=roe,
        net_margin=net_margin,
        asset_turnover=asset_turnover,
        equity_multiplier=equity_multiplier,
        roe_scope=ROE_SCOPE_DILUTED,
    )

    if previous_income is None or previous_balance is None:
        return decomposition

    previous_net_margin = previous_income.net_margin
    previous_turnover = safe_divide(previous_income.revenue, previous_balance.total_assets)
    previous_multiplier = previous_balance.equity_multiplier

    decomposition.net_margin_change = growth_rate(net_margin, previous_net_margin)
    decomposition.asset_turnover_change = growth_rate(asset_turnover, previous_turnover)
    decomposition.equity_multiplier_change = growth_rate(equity_multiplier, previous_multiplier)

    # 主导因子 = 三个因子中相对变动绝对值最大的那个。
    # 严格做法是对数分解（ln(ROE) = ln(a)+ln(b)+ln(c)，各项贡献可加），
    # 但相对变动比较在方向判断上与之一致，且更容易向非专业用户解释。
    changes = {
        "净利率": decomposition.net_margin_change,
        "资产周转率": decomposition.asset_turnover_change,
        "权益乘数": decomposition.equity_multiplier_change,
    }
    valid = {name: value for name, value in changes.items() if value is not None}
    if valid:
        dominant = max(valid, key=lambda name: abs(valid[name]))
        direction = "上升" if valid[dominant] > 0 else "下降"
        interpretation = {
            "净利率": "盈利能力变化主导（关注成本与费用控制）",
            "资产周转率": "运营效率变化主导（关注资产利用率）",
            "权益乘数": "财务杠杆变化主导（ROE 变动的质量存疑，需关注债务风险）",
        }[dominant]
        decomposition.driver = (
            f"{dominant}{direction} {abs(valid[dominant]) * 100:.1f}%，{interpretation}"
        )
    return decomposition


def detect_risk_alerts(
    incomes: list[IncomeRow],
    balances: list[BalanceSheetRow],
    cashflows: list[CashFlowRow],
) -> list[RiskAlert]:
    """按 md "风险指标"表逐条执行预警规则。

    入参列表按报告期倒序（最新在前）。任一数据缺失时跳过对应规则而不是
    默认"无风险"——把"没查到"说成"没问题"是这类系统最危险的失效模式。
    """
    alerts: list[RiskAlert] = []
    latest_income = incomes[0] if incomes else None
    previous_income = incomes[1] if len(incomes) > 1 else None
    latest_balance = balances[0] if balances else None
    previous_balance = balances[1] if len(balances) > 1 else None
    latest_cashflow = cashflows[0] if cashflows else None

    # 规则 1：应收账款增速 vs 营收增速
    if latest_balance and previous_balance and latest_income and previous_income:
        receivable_growth = growth_rate(
            latest_balance.accounts_receivable, previous_balance.accounts_receivable
        )
        revenue_growth = growth_rate(latest_income.revenue, previous_income.revenue)
        if receivable_growth is not None and revenue_growth is not None:
            triggered = (
                receivable_growth > revenue_growth * RECEIVABLE_GROWTH_MULTIPLE
                and receivable_growth > 0
            )
            if triggered:
                alerts.append(
                    RiskAlert(
                        code="receivable_divergence",
                        level="warning",
                        title="应收账款增速显著超过营收增速",
                        detail=(
                            f"应收账款同比 {format_ratio(receivable_growth)}，"
                            f"营收同比 {format_ratio(revenue_growth)}，"
                            f"前者超过后者的 {RECEIVABLE_GROWTH_MULTIPLE} 倍。"
                            "可能存在放宽信用政策刺激收入或收入确认激进的情况，"
                            "需结合现金流进一步核实。"
                        ),
                        metrics={
                            "receivable_yoy": receivable_growth,
                            "revenue_yoy": revenue_growth,
                        },
                        threshold_desc=f"应收增速 > 营收增速 × {RECEIVABLE_GROWTH_MULTIPLE}",
                    )
                )

    # 规则 2：经营现金流 / 净利润
    if latest_cashflow:
        ratio = latest_cashflow.cashflow_to_profit
        if ratio is not None and ratio < CASHFLOW_TO_PROFIT_ALERT:
            alerts.append(
                RiskAlert(
                    code="low_cashflow_quality",
                    level="warning" if ratio > 0 else "danger",
                    title="经营现金流对净利润的覆盖不足",
                    detail=(
                        f"经营活动现金流净额 / 净利润 = {ratio:.2f}，"
                        f"低于预警线 {CASHFLOW_TO_PROFIT_ALERT}。"
                        "利润未能同步转化为现金，利润质量存疑。"
                    ),
                    metrics={"cashflow_to_profit": ratio},
                    threshold_desc=f"< {CASHFLOW_TO_PROFIT_ALERT}",
                )
            )

    # 规则 3：资产负债率
    if latest_balance:
        debt_ratio = latest_balance.debt_ratio
        if debt_ratio is not None and debt_ratio > DEBT_RATIO_ALERT:
            alerts.append(
                RiskAlert(
                    code="high_leverage",
                    level="warning",
                    title="资产负债率偏高",
                    detail=(
                        f"资产负债率 {format_ratio(debt_ratio)}，"
                        f"高于 {format_ratio(DEBT_RATIO_ALERT)} 的预警线。"
                        "需注意：制造业与金融业的合理区间差异很大，"
                        "该阈值为通用口径，行业对比结论请以同业数据为准。"
                    ),
                    metrics={"debt_ratio": debt_ratio},
                    threshold_desc=f"> {format_ratio(DEBT_RATIO_ALERT)}",
                )
            )

    # 规则 4：商誉占净资产比
    if latest_balance:
        goodwill_ratio = latest_balance.goodwill_to_equity
        if goodwill_ratio is not None and goodwill_ratio > GOODWILL_TO_EQUITY_ALERT:
            alerts.append(
                RiskAlert(
                    code="goodwill_impairment_risk",
                    level="warning",
                    title="商誉占净资产比过高",
                    detail=(
                        f"商誉 / 净资产 = {format_ratio(goodwill_ratio)}，"
                        f"高于 {format_ratio(GOODWILL_TO_EQUITY_ALERT)}。"
                        "若并购标的业绩不达预期，存在集中减值风险。"
                    ),
                    metrics={"goodwill_to_equity": goodwill_ratio},
                    threshold_desc=f"> {format_ratio(GOODWILL_TO_EQUITY_ALERT)}",
                )
            )

    # 规则 5：扣非净利增速与营收增速背离
    if latest_income and previous_income:
        revenue_growth = growth_rate(latest_income.revenue, previous_income.revenue)
        deducted_growth = growth_rate(
            latest_income.deducted_net_profit, previous_income.deducted_net_profit
        )
        if revenue_growth is not None and deducted_growth is not None:
            divergence_pp = (deducted_growth - revenue_growth) * 100
            if abs(divergence_pp) > PROFIT_REVENUE_DIVERGENCE_PP:
                alerts.append(
                    RiskAlert(
                        code="profit_revenue_divergence",
                        level="info" if divergence_pp > 0 else "warning",
                        title="扣非净利润增速与营收增速背离",
                        detail=(
                            f"营收同比 {format_ratio(revenue_growth)}，"
                            f"扣非净利同比 {format_ratio(deducted_growth)}，"
                            f"差异 {divergence_pp:+.1f}pp。"
                            + (
                                "利润增长快于收入，通常来自毛利率提升或费用率下降。"
                                if divergence_pp > 0
                                else "收入增长未能带来同步的利润增长，需检查成本、费用或减值。"
                            )
                        ),
                        metrics={
                            "revenue_yoy": revenue_growth,
                            "deducted_net_profit_yoy": deducted_growth,
                        },
                        threshold_desc=f"背离 > {PROFIT_REVENUE_DIVERGENCE_PP}pp",
                    )
                )

    # 规则 6：毛利率同比下降
    if latest_income and previous_income:
        current_margin = latest_income.gross_margin
        previous_margin = previous_income.gross_margin
        if current_margin is not None and previous_margin is not None:
            change_pp = (current_margin - previous_margin) * 100
            if change_pp < -GROSS_MARGIN_DROP_ALERT_PP:
                alerts.append(
                    RiskAlert(
                        code="gross_margin_decline",
                        level="warning",
                        title="毛利率显著下滑",
                        detail=(
                            f"毛利率由 {format_ratio(previous_margin)} 降至 "
                            f"{format_ratio(current_margin)}，变动 {change_pp:+.2f}pp，"
                            f"降幅超过 {GROSS_MARGIN_DROP_ALERT_PP}pp 预警线。"
                        ),
                        metrics={
                            "gross_margin": current_margin,
                            "gross_margin_prev": previous_margin,
                        },
                        threshold_desc=f"同比下降 > {GROSS_MARGIN_DROP_ALERT_PP}pp",
                    )
                )

    # 规则 7：ROE 偏低
    if latest_income and latest_balance:
        roe = safe_divide(latest_income.net_profit, latest_balance.net_assets)
        if roe is not None and roe < ROE_LOW_THRESHOLD:
            alerts.append(
                RiskAlert(
                    code="low_roe",
                    level="info",
                    title="ROE 偏低",
                    detail=(
                        f"ROE {format_ratio(roe)}，低于 {format_ratio(ROE_LOW_THRESHOLD)}，"
                        "股东回报水平偏弱。"
                    ),
                    metrics={"roe": roe},
                    threshold_desc=f"< {format_ratio(ROE_LOW_THRESHOLD)}",
                )
            )

    return alerts


def classify_trend(values: list[float | None]) -> tuple[str, bool, str]:
    """按数值序列（按时间正序）判断趋势，并检测显著异动。

    返回 (趋势标签, 是否异常, 异常原因)。
    """
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return "insufficient_data", False, ""

    changes = [
        growth_rate(clean[i], clean[i - 1]) for i in range(1, len(clean))
    ]
    changes = [c for c in changes if c is not None]
    if not changes:
        return "insufficient_data", False, ""

    anomaly_items = [c for c in changes if abs(c) > SIGNIFICANT_CHANGE_THRESHOLD]
    anomaly = bool(anomaly_items)
    anomaly_reason = (
        f"存在单期变动超过 ±{SIGNIFICANT_CHANGE_THRESHOLD:.0%} 的显著变化："
        + ", ".join(format_ratio(c) for c in anomaly_items)
        if anomaly
        else ""
    )

    latest = changes[-1]
    if len(changes) >= 2:
        previous = changes[-2]
        if latest > 0 and previous > 0:
            trend = "growth_accelerating" if latest > previous else "growth_decelerating"
        elif latest > 0 >= previous:
            trend = "improving"
        elif latest <= 0 < previous:
            trend = "deteriorating"
        elif abs(latest) < 0.02 and abs(previous) < 0.02:
            trend = "stable"
        else:
            trend = "volatile"
    else:
        if latest > 0.02:
            trend = "improving"
        elif latest < -0.02:
            trend = "deteriorating"
        else:
            trend = "stable"

    return trend, anomaly, anomaly_reason


def decompose_margin_change(
    base: Sequence[SegmentRow], current: Sequence[SegmentRow]
) -> SegmentEffect:
    """把整体毛利率的变动拆成「结构效应」与「各分部自身效应」。

    公式（标准的两因素分解）：
        整体毛利率 = Σ wᵢ × mᵢ        （wᵢ=分部收入占比，mᵢ=分部毛利率）
        Δ整体 = Σ (Δwᵢ × mᵢ_base)     ← 结构效应：占比变了
              + Σ (wᵢ_cur × Δmᵢ)      ← 自身效应：各分部毛利率变了

    交叉项 Σ(Δwᵢ × Δmᵢ) 被并入自身效应（用期末占比而非期初占比加权），
    这是常见处理方式，避免多出一个难以解释的第三项。
    """
    base_map = {row.segment_name: row for row in base}
    current_map = {row.segment_name: row for row in current}
    names = [name for name in current_map if name in base_map]
    if not names:
        return SegmentEffect(dominant="两期分部口径不可比，无法分解")

    def overall(rows: dict[str, SegmentRow]) -> float | None:
        pairs = [
            (r.revenue_share, r.gross_margin)
            for name, r in rows.items()
            if name in names and r.revenue_share is not None and r.gross_margin is not None
        ]
        if not pairs:
            return None
        return sum(w * m for w, m in pairs)

    base_overall = overall(base_map)
    current_overall = overall(current_map)
    if base_overall is None or current_overall is None:
        return SegmentEffect(dominant="分部数据不完整，无法分解")

    mix_effect = 0.0
    own_effect = 0.0
    contributions: list[dict[str, Any]] = []
    for name in names:
        b, c = base_map[name], current_map[name]
        if None in (b.revenue_share, b.gross_margin, c.revenue_share, c.gross_margin):
            continue
        mix = (c.revenue_share - b.revenue_share) * b.gross_margin
        own = c.revenue_share * (c.gross_margin - b.gross_margin)
        mix_effect += mix
        own_effect += own
        contributions.append(
            {
                "segment": name,
                "revenue_share_base": b.revenue_share,
                "revenue_share_current": c.revenue_share,
                "share_change_pp": (c.revenue_share - b.revenue_share) * 100,
                "gross_margin_base": b.gross_margin,
                "gross_margin_current": c.gross_margin,
                "margin_change_pp": (c.gross_margin - b.gross_margin) * 100,
                "mix_effect_pp": mix * 100,
                "own_effect_pp": own * 100,
                "total_effect_pp": (mix + own) * 100,
            }
        )

    contributions.sort(key=lambda item: abs(item["total_effect_pp"]), reverse=True)
    mix_pp, own_pp = mix_effect * 100, own_effect * 100

    if abs(mix_pp) > abs(own_pp) * 1.5:
        dominant = f"结构效应主导（{mix_pp:+.2f}pp）：业务占比变化是整体毛利率变动的主因"
    elif abs(own_pp) > abs(mix_pp) * 1.5:
        dominant = f"自身效应主导（{own_pp:+.2f}pp）：各业务自身盈利能力变化是主因"
    else:
        dominant = (
            f"结构效应（{mix_pp:+.2f}pp）与自身效应（{own_pp:+.2f}pp）共同作用，量级相当"
        )
    if contributions:
        top = contributions[0]
        dominant += (
            f"；影响最大的分部是「{top['segment']}」"
            f"（贡献 {top['total_effect_pp']:+.2f}pp，"
            f"占比 {top['share_change_pp']:+.2f}pp、毛利率 {top['margin_change_pp']:+.2f}pp）"
        )

    return SegmentEffect(
        overall_margin_base=base_overall,
        overall_margin_current=current_overall,
        overall_change_pp=(current_overall - base_overall) * 100,
        mix_effect_pp=mix_pp,
        own_effect_pp=own_pp,
        dominant=dominant,
        segment_contributions=contributions,
    )


TREND_LABELS: dict[str, str] = {
    "growth_accelerating": "增长加速",
    "growth_decelerating": "增速放缓",
    "improving": "改善",
    "deteriorating": "恶化",
    "stable": "平稳",
    "volatile": "波动较大",
    "insufficient_data": "数据不足",
}
