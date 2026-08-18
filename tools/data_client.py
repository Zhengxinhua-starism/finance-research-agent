"""金融数据客户端契约与通用基础设施。

解决什么问题
    工具层需要一个稳定的"数据源"抽象。如果每个工具直接 import akshare，
    那么 (1) 换数据源（Wind / Tushare / 自建数仓）要改 6 个文件；
    (2) 单元测试必须联网；(3) 缓存、超时、列名归一这些横切逻辑会被复制 6 遍。
    本模块定义 FinancialDataClient Protocol，并提供所有工具共用的
    基类（缓存接入、超时、错误归类）和 AKShare 列名归一化工具。

核心设计决策
    1. 数据源抽象用 Protocol 而不是 ABC。工具持有的是"满足这个接口的东西"，
       不关心继承关系；Protocol 让实现方（AKShareClient）不需要 import 本模块，
       依赖方向更干净，也方便测试时传一个简单的假对象。
    2. BaseTool 基类承担缓存与错误归类，子类只实现 `fetch()`。
       这是本层唯一使用继承的地方——因为这些行为对所有工具**完全一致**，
       用 mixin 或装饰器反而会让 name/description/parameters 的定义位置变散。
    3. 列名归一化用"别名列表 + 包含匹配 + 优先级"三段式。
       AKShare 的列名在不同接口和年份间会变（"营业总收入"/"营业收入"/
       "一、营业总收入"），硬编码列名是这类项目最常见的线上故障源。
       匹配时先精确、后前缀、再包含，避免"营业收入"误匹配到"营业收入其他"。
    4. 金额解析支持中文单位字符串（"7,771.02亿"）。同花顺系接口返回的是
       带单位的字符串而不是数字，直接 float() 会抛异常。

为什么不用其他方案
    - 不做统一的 DataFrame → 模型自动映射（比如按字段注解反射）：
      财务报表的列名映射规则包含大量业务判断（"净利润"到底取归母还是含少数股东），
      自动映射会把这些判断藏起来，出错时无从查起。显式别名表虽然啰嗦，
      但每一条都能对应到一个业务决策。
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any, Iterable, Protocol, runtime_checkable

from config import get_config
from data.schemas import (
    BalanceSheetRow,
    CashFlowRow,
    FinancialMetrics,
    IncomeRow,
    NewsItem,
    PriceBar,
)
from tools.cache import ToolResultCache, get_tool_cache

logger = logging.getLogger(__name__)


class DataSourceError(Exception):
    """数据源层面的错误（网络失败、接口变更、标的不存在）。"""

    def __init__(self, message: str, error_type: str = "execution_error"):
        super().__init__(message)
        self.error_type = error_type


class TickerNotFoundError(DataSourceError):
    """标的代码不存在。单独成类，因为它对应的是"应当拒答"而不是"重试"。"""

    def __init__(self, ticker: str):
        super().__init__(f"未找到股票代码 {ticker} 对应的上市公司", error_type="param_error")
        self.ticker = ticker


@runtime_checkable
class FinancialDataClient(Protocol):
    """金融数据源契约。AKShare 实现见 tools/akshare_tools.py。"""

    def get_company_name(self, ticker: str) -> str: ...

    def get_income_statements(self, ticker: str, periods: int) -> list[IncomeRow]: ...

    def get_balance_sheets(self, ticker: str, periods: int) -> list[BalanceSheetRow]: ...

    def get_cash_flows(self, ticker: str, periods: int) -> list[CashFlowRow]: ...

    def get_financial_metrics(self, ticker: str) -> FinancialMetrics: ...

    def get_price_history(self, ticker: str, days: int) -> list[PriceBar]: ...

    def get_news(self, ticker: str, days: int) -> list[NewsItem]: ...


# ============================================================
# 工具基类
# ============================================================


class BaseTool:
    """所有 Agent 工具的基类，实现 ToolProtocol。

    子类只需要：设置 name / description / parameters 三个类属性，
    并实现 fetch(**kwargs) -> dict。缓存、错误归类由基类处理。
    """

    name: str = "base_tool"
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    # 是否启用缓存。行情类工具时效性强，可以关掉
    cacheable: bool = True

    def __init__(self, cache: ToolResultCache | None = None):
        self.cache = cache or get_tool_cache()
        self.config = get_config()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """ToolProtocol 入口。命中缓存时在返回值里带 _cache 元信息。"""
        if self.cacheable:
            cached, layer = self.cache.get(self.name, kwargs)
            if cached is not None:
                if isinstance(cached, dict):
                    return {**cached, "_cache": {"hit": True, "layer": layer}}
                return {"result": cached, "_cache": {"hit": True, "layer": layer}}

        result = self.fetch(**kwargs)

        if self.cacheable and isinstance(result, dict) and not result.get("error"):
            self.cache.set(self.name, kwargs, result)
        return result

    def fetch(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} 必须实现 fetch()")

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


# ============================================================
# AKShare 数据清洗工具
# ============================================================

# 中文数量单位 → 倍数
CHINESE_UNITS: dict[str, float] = {
    "万亿": 1e12,
    "千亿": 1e11,
    "百亿": 1e10,
    "亿": 1e8,
    "千万": 1e7,
    "百万": 1e6,
    "万": 1e4,
    "千": 1e3,
}

# 表示"无数据"的占位符，AKShare 各接口用法不一
NULL_TOKENS = {"", "--", "—", "-", "nan", "NaN", "None", "null", "不适用", "无"}


def parse_amount(value: Any, default_unit: float = 1.0) -> float | None:
    """把 AKShare 返回的金额（数字或带单位的中文字符串）解析成"元"。

    default_unit 用于接口本身以某单位计价的场景（例如新浪财报以元为单位则传 1，
    某些接口以万元为单位则传 1e4）。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        # pandas 的 NaN 是 float，且 NaN != NaN
        if value != value:  # noqa: PLR0124
            return None
        return float(value) * default_unit

    text = str(value).strip().replace(",", "").replace("，", "")
    if text in NULL_TOKENS:
        return None

    negative = text.startswith("-") or (text.startswith("(") and text.endswith(")"))
    text = text.lstrip("-").strip("()")

    multiplier = default_unit
    for unit, factor in CHINESE_UNITS.items():
        if unit in text:
            multiplier = factor
            text = text.replace(unit, "")
            break

    if text.endswith("%"):
        # 百分比不是金额，交给 parse_ratio 处理；这里返回 None 避免误用
        return None

    match = re.search(r"-?\d+\.?\d*", text)
    if not match:
        return None
    try:
        amount = float(match.group()) * multiplier
    except ValueError:
        return None
    return -amount if negative else amount


def parse_ratio(value: Any) -> float | None:
    """把比率解析成小数。"21.8%" → 0.218；21.8 → 0.218；0.218 → 0.218。

    裸数字的歧义处理：|x| > 1.5 视为百分数形式。理由与 EvidenceGate
    里的归一逻辑一致——A 股极少有超过 150% 的毛利率/ROE。
    """
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value != value:  # NaN  # noqa: PLR0124
            return None
        number = float(value)
        return number / 100 if abs(number) > 1.5 else number

    text = str(value).strip().replace(",", "")
    if text in NULL_TOKENS:
        return None
    is_percent = "%" in text
    match = re.search(r"-?\d+\.?\d*", text)
    if not match:
        return None
    try:
        number = float(match.group())
    except ValueError:
        return None
    if is_percent:
        return number / 100
    return number / 100 if abs(number) > 1.5 else number


# 匹配"年-月-日"三段。分隔符限定为真正的日期分隔符，不用宽泛的 \D——
# 后者会把"2024年第13期"回溯匹配成 2024-01-03，产生一个看起来合法的假日期。
# 假日期比 None 危险得多：它会变成证据的 disclosure_date，进而影响前视偏差判定。
# 覆盖 20241231 / 2024-12-31 / 2024/12/31 / 2024.12.31 / 2024年12月31日
# / 2026-08-13 10:30:00（带时分秒，只取日期部分）
DATE_PATTERN = re.compile(r"(\d{4})[-/.\s年]{0,2}(\d{1,2})[-/.\s月]{0,2}(\d{1,2})")


def parse_report_date(value: Any) -> date | None:
    """解析日期。支持纯数字、各种分隔符、中文年月日、以及带时分秒的时间戳。

    # TODO: md 没规定日期解析要覆盖哪些格式。最初只处理了 8 位数字和
    # 三种纯日期格式，结果东财新闻接口的「发布时间」是
    # "2026-08-13 10:30:00"（带时分秒）——8 位数字分支拿到 14 位数字直接返回 None，
    # 纯日期分支又因为多了时间部分匹配失败，于是所有新闻的 published_at 都是 None，
    # 时间窗过滤完全失效（days=1 和 days=30 返回同一批结果）。
    # 改成正则提取年月日三段，不再依赖整串完全匹配某个格式。
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = str(value).strip()
    if text in NULL_TOKENS:
        return None

    # 纯 8 位数字（20241231）是财报接口最常见的形式，走快速路径
    digits = re.sub(r"\D", "", text)
    if len(digits) == 8:
        try:
            return datetime.strptime(digits, "%Y%m%d").date()
        except ValueError:
            return None

    match = DATE_PATTERN.search(text)
    if match is None:
        return None
    year, month, day = (int(group) for group in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        # 匹配到了但不是合法日期（例如把"2024年第13期"读成 13 月）
        return None


def classify_report_type(period_end: date) -> str:
    """按报告期月份判断报表类型。"""
    return {12: "annual", 6: "interim", 3: "q1", 9: "q3"}.get(period_end.month, "unknown")


def estimate_disclosure_date(period_end: date) -> date:
    """在数据源没给披露日期时，用法定截止日作为保守估计。

    保守方向的选择很关键：估早了会让本该被前视偏差拦截的数据通过门禁，
    估晚了最多是把一条合规数据误判为"尚未披露"。宁可误拦，不可漏拦。
    """
    from harness.evidence_gate import disclosure_deadline

    deadline = disclosure_deadline(period_end)
    if deadline is not None:
        return deadline
    # 非标准报告期：按报告期后 4 个月估算
    month = period_end.month + 4
    year = period_end.year + (month - 1) // 12
    return date(year, (month - 1) % 12 + 1, min(period_end.day, 28))


def find_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    """在 DataFrame 列名里按别名找列。

    三段式匹配（精确 → 去修饰后精确 → 包含），按 aliases 的顺序优先，
    保证"净利润"这类高优先别名不会被"净利润(扣除...)"抢先命中。
    """
    column_list = [str(c) for c in columns]
    normalized = {c: _strip_decorations(c) for c in column_list}

    for alias in aliases:
        for column in column_list:
            if column == alias:
                return column
    for alias in aliases:
        for column in column_list:
            if normalized[column] == alias:
                return column
    for alias in aliases:
        for column in column_list:
            if alias in normalized[column]:
                return column
    return None


def _strip_decorations(column: str) -> str:
    """去掉列名里的序号前缀和空白。"一、营业总收入" → "营业总收入"。"""
    text = str(column).strip()
    text = re.sub(r"^[一二三四五六七八九十（）()\d\.、\s　]+", "", text)
    return text.replace(" ", "").replace("　", "")


def pick(row: Any, columns: Iterable[str], aliases: Iterable[str], parser: Any = parse_amount) -> Any:
    """从一行数据里按别名取值并解析。找不到列返回 None。"""
    column = find_column(columns, aliases)
    if column is None:
        return None
    try:
        raw = row[column]
    except (KeyError, IndexError, TypeError):
        return None
    return parser(raw)


# ============================================================
# 报表字段别名表
# ============================================================
# TODO: md 没有给出 AKShare 列名到标准字段的映射规则，而这是整个数据层
# 最容易出线上故障的地方（接口改一个字列名就取不到值）。这里显式维护别名表，
# 按业务优先级排序，并在取不到关键字段时记 warning 而不是静默返回 None。

INCOME_FIELD_ALIASES: dict[str, list[str]] = {
    "revenue": ["营业总收入", "营业收入", "其中：营业收入", "总营业收入"],
    "operating_cost": ["营业成本", "其中：营业成本", "营业总成本"],
    "operating_profit": ["营业利润"],
    "total_profit": ["利润总额"],
    # 归母净利润优先，退而求其次才用净利润总额
    "net_profit": [
        "归属于母公司所有者的净利润",
        "归属于母公司股东的净利润",
        "归属母公司股东的净利润",
        "归母净利润",
    ],
    "total_net_profit": ["净利润", "五、净利润", "净利润(含少数股东损益)"],
    "deducted_net_profit": [
        "扣除非经常性损益后的净利润",
        "归属于母公司所有者的扣除非经常性损益的净利润",
        "扣非净利润",
    ],
    "rd_expense": ["研发费用", "研发支出"],
    "selling_expense": ["销售费用"],
    "admin_expense": ["管理费用"],
    "finance_expense": ["财务费用"],
    # 以下科目此前未接，但一直在新浪利润表的 83 列里。
    # 它们是"净利率降幅为何小于毛利率降幅"的直接答案来源。
    "tax_and_surcharges": ["营业税金及附加", "税金及附加"],
    "asset_impairment": ["资产减值损失"],
    "credit_impairment": ["信用减值损失"],
    # 「投资收益」要放在「对联营企业和合营企业的投资收益」之前：
    # 后者是前者的子项，别名顺序决定优先级
    "investment_income": ["投资收益"],
    "fair_value_change": ["公允价值变动收益"],
    "other_income": ["其他收益"],
    "non_operating_income": ["营业外收入"],
    "non_operating_expense": ["营业外支出"],
    "income_tax": ["所得税费用", "所得税"],
}

BALANCE_FIELD_ALIASES: dict[str, list[str]] = {
    "total_assets": ["资产总计", "资产总额", "总资产"],
    "total_liabilities": ["负债合计", "负债总计", "总负债"],
    "net_assets": [
        "归属于母公司所有者权益合计",
        "归属于母公司股东权益合计",
        "归属母公司股东的权益",
        "归属于母公司所有者权益",
    ],
    "total_equity": ["所有者权益(或股东权益)合计", "所有者权益合计", "股东权益合计"],
    "accounts_receivable": ["应收账款", "应收账款净额"],
    "inventory": ["存货"],
    "goodwill": ["商誉"],
    "short_term_debt": ["短期借款"],
    "long_term_debt": ["长期借款"],
    "monetary_funds": ["货币资金"],
    "bonds_payable": ["应付债券"],
    "current_portion_non_current_liabilities": ["一年内到期的非流动负债"],
    "contract_liabilities": ["合同负债", "预收款项", "预收账款"],
    # 「固定资产净值」优先于「固定资产原值」：分析用的是净值。
    # 只写"固定资产"会被包含匹配命中"固定资产原值"（列顺序在前），
    # 那是未扣折旧的原值，会高估产能资产规模。
    "fixed_assets": ["固定资产净值", "固定资产净额", "固定资产"],
    "construction_in_progress": ["在建工程合计", "在建工程"],
    "development_expenditure": ["开发支出"],
}

CASHFLOW_FIELD_ALIASES: dict[str, list[str]] = {
    "operating_cashflow": [
        "经营活动产生的现金流量净额",
        "经营活动现金流量净额",
        "经营活动产生的现金流量净额(元)",
    ],
    "investing_cashflow": ["投资活动产生的现金流量净额", "投资活动现金流量净额"],
    "financing_cashflow": ["筹资活动产生的现金流量净额", "筹资活动现金流量净额"],
    "net_increase_in_cash": ["现金及现金等价物净增加额"],
    # F-013：实际列名是「购建固定资产、无形资产和其他长期资产**所**支付的现金」，
    # 之前的别名少了一个"所"字。包含匹配也救不了——目标串比实际列名多字时，
    # 实际列名不可能"包含"它。两种写法都列上以适配不同数据源。
    "capital_expenditure": [
        "购建固定资产、无形资产和其他长期资产所支付的现金",
        "购建固定资产、无形资产和其他长期资产支付的现金",
    ],
    "net_profit": ["净利润"],
}


def normalize_ticker(ticker: str) -> str:
    """规范化 A 股代码：去空格、补足 6 位、去交易所前缀。"""
    text = str(ticker).strip().upper()
    text = re.sub(r"^(SH|SZ|BJ)\.?", "", text)
    text = re.sub(r"\.(SH|SZ|BJ)$", "", text)
    digits = re.sub(r"\D", "", text)
    if not digits:
        raise TickerNotFoundError(ticker)
    return digits.zfill(6)


def exchange_prefix(ticker: str) -> str:
    """推断交易所前缀。新浪财报接口要求 sh600519 / sz000001 这种格式。"""
    code = normalize_ticker(ticker)
    if code.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return f"sh{code}"
    if code.startswith(("000", "001", "002", "003", "300", "301", "200")):
        return f"sz{code}"
    if code.startswith(("4", "8", "920")):
        return f"bj{code}"
    # 兜底按深市处理，并记日志——比抛异常好，至少还有一次成功的机会
    logger.warning("无法判断 %s 所属交易所，按深市处理", code)
    return f"sz{code}"
