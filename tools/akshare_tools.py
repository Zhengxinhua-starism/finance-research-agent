"""AKShare 数据源实现与 5 个财务数据工具。

解决什么问题
    把 AKShare 这个"免费但不稳定、列名会变、返回格式各接口不一致"的数据源，
    包装成 Agent 可以放心调用的工具。核心难点不是调接口，而是：
    (1) 同一份数据在不同接口里单位不同（元 / 万元 / "亿"字符串）；
    (2) 列名随年份和接口版本漂移；
    (3) 接口偶发挂起或返回空表；
    (4) 财报数据本身有"报告期"和"披露日期"两个时间维度，混用会造成前视偏差。

核心设计决策
    1. 数据获取与工具封装分离：AKShareDataClient 负责"拿到干净的模型对象"，
       各 Tool 类负责"组装成 LLM 友好的返回格式"。这样 compare_periods
       和 MCP Server 可以直接复用 Client，不必绕一层工具协议。
    2. 每次 AKShare 调用都套 _call_with_timeout。ToolRegistry 已有 30s 超时，
       但 Client 会被 compare_periods 直接调用（不经过 Registry），
       那条路径上没有保护。防御要放在最靠近风险源的地方。
    3. 财报数据一律走"新浪财报接口"（stock_financial_report_sina）取三大报表，
       "同花顺财务摘要"（stock_financial_abstract_ths）只用于交叉校验和补充
       ROE 这类现成指标。原因：新浪接口返回的是原始报表科目，
       口径明确（合并报表、单位元）；摘要类接口的口径和单位随时可能调整。
    4. 所有派生指标（毛利率、ROE、同比）在本地用 data/schemas.py 的公式算，
       即使数据源提供了现成值也优先自算。数据源给的指标口径不透明
       （ROE 是加权还是摊薄？净利润是否归母？），自算才能保证与研报口径一致。
    5. 披露日期在数据源没提供时用法定截止日估算，并在返回里标注
       is_estimated=True。宁可让证据门禁误拦一条合规数据，
       也不能让前视偏差混进研报。

为什么不用其他方案
    - 不用 Tushare：需要积分，注册门槛高，评审复现不便。
    - 不做本地财报数据库：项目定位是 Agent 能力展示，不是数据工程；
      引入 ETL 会喧宾夺主。缓存层已经解决了重复请求的性能问题。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import date
from typing import Any, Callable

from config import get_config
from data.schemas import (
    BalanceSheetRow,
    CashFlowRow,
    FinancialMetrics,
    IncomeRow,
    NewsItem,
    PriceBar,
    ROE_SCOPE_AVERAGE,
    ROE_SCOPE_DILUTED,
    SegmentRow,
    compute_dupont,
    detect_risk_alerts,
    format_ratio,
    growth_rate,
    safe_divide,
)
from tools.data_client import (
    BALANCE_FIELD_ALIASES,
    CASHFLOW_FIELD_ALIASES,
    INCOME_FIELD_ALIASES,
    BaseTool,
    DataSourceError,
    TickerNotFoundError,
    classify_report_type,
    estimate_disclosure_date,
    exchange_prefix,
    find_column,
    normalize_ticker,
    parse_amount,
    parse_ratio,
    parse_report_date,
)

logger = logging.getLogger(__name__)

# 供 _call_with_timeout 使用的共享线程池。AKShare 内部是同步 requests，
# 唯一可靠的超时手段就是丢到别的线程里等。
_AKSHARE_EXECUTOR = ThreadPoolExecutor(max_workers=6, thread_name_prefix="akshare")

# 新浪财报接口的报表类型参数
SINA_REPORT_TYPES = {
    "income": "利润表",
    "balance": "资产负债表",
    "cashflow": "现金流量表",
}

# 新浪三张报表的报告期列名都是「报告日」，值形如 "20241231"。
# 这里刻意**不放**宽泛的「日期」别名：表里还有「公告日期」和「更新日期」，
# 前者是公告日（2026-04-29）而非报告期（2026-03-31）。
# 用「日期」做包含匹配会命中「公告日期」，导致所有报告期都变成公告日，
# 年报筛选（month == 12）随之全部落空——这是个不报错的静默失效。
REPORT_DATE_ALIASES = ("报告日", "报表日期", "报告日期", "报告期")

# 财务摘要接口（stock_financial_abstract）的指标名 → 标准字段。
# 该接口是**转置**结构：行是指标，列是报告期（20241231 这样的字符串）。
# 只取「常用指标」分组：同一个指标名在多个分组里会重复出现，值可能不同口径。
ABSTRACT_METRIC_MAP: dict[str, str] = {
    "扣非净利润": "deducted_net_profit",
    "归母净利润": "net_profit",
    "营业总收入": "revenue",
    "净资产收益率(ROE)": "roe",
}
# 上述指标里哪些是百分数形态（需要 /100 转成小数）
ABSTRACT_PERCENT_FIELDS = {"roe"}

# 财务分析指标接口（stock_financial_analysis_indicator，86 列）里我们要用的部分。
# 周转率是"倍"、周转天数是"天"，都不是比率，解析时不能走比率归一化逻辑。
EFFICIENCY_FIELD_ALIASES: dict[str, list[str]] = {
    "receivable_turnover": ["应收账款周转率(次)"],
    "receivable_days": ["应收账款周转天数(天)"],
    "inventory_turnover": ["存货周转率(次)"],
    "inventory_days": ["存货周转天数(天)"],
    "fixed_asset_turnover": ["固定资产周转率(次)"],
    "total_asset_turnover": ["总资产周转率(次)"],
    "total_asset_days": ["总资产周转天数(天)"],
    "current_asset_turnover": ["流动资产周转率(次)"],
    "equity_turnover": ["股东权益周转率(次)"],
    "three_expense_ratio": ["三项费用比重"],
    "cost_profit_ratio": ["成本费用利润率(%)"],
}
# 这些字段在接口里是百分数，需要 /100
EFFICIENCY_PERCENT_FIELDS = {"three_expense_ratio", "cost_profit_ratio"}


def _call_with_timeout(func: Callable[..., Any], *args: Any, timeout: int | None = None, **kwargs: Any) -> Any:
    """带超时地执行一次 AKShare 调用。

    超时不抛 TimeoutError 而是抛 DataSourceError(error_type="timeout")，
    让上层的错误归类逻辑只需要认识一种异常类型。
    """
    seconds = timeout or get_config().akshare_timeout_seconds
    future = _AKSHARE_EXECUTOR.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=seconds)
    except FutureTimeoutError as exc:
        future.cancel()
        raise DataSourceError(
            f"数据源调用超时（>{seconds}s）: {getattr(func, '__name__', func)}",
            error_type="timeout",
        ) from exc
    except ImportError as exc:
        raise DataSourceError(f"AKShare 未安装或导入失败: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 — AKShare 抛的异常类型不可枚举
        raise DataSourceError(
            f"数据源调用失败 {getattr(func, '__name__', func)}: {type(exc).__name__}: {exc}"
        ) from exc


class AKShareDataClient:
    """AKShare 数据源实现，满足 FinancialDataClient 协议。"""

    def __init__(self) -> None:
        self.config = get_config()
        self._name_cache: dict[str, str] = {}
        # 财务摘要按报告期缓存：一次研究里 get_income_statements 会被
        # 多个工具各调一次，摘要接口没必要跟着重复请求
        self._abstract_cache: dict[str, dict[date, dict[str, float]]] = {}

    # ---------------- 基础信息 ----------------

    def get_company_name(self, ticker: str) -> str:
        """查公司简称。失败时返回代码本身而不是抛异常——

        公司名只影响研报标题的可读性，为它中断整次研究不值得。
        """
        code = normalize_ticker(ticker)
        if code in self._name_cache:
            return self._name_cache[code]
        try:
            from tools.price_history import fetch_company_name

            quote_name = fetch_company_name(code)
            if quote_name:
                self._name_cache[code] = quote_name
                return quote_name
        except Exception as exc:  # noqa: BLE001
            logger.debug("腾讯简称获取失败，回退东财: %s", exc)
        try:
            import akshare as ak

            info = _call_with_timeout(ak.stock_individual_info_em, symbol=code, timeout=15)
            name = code
            if info is not None and not info.empty:
                rows = info.set_index("item")["value"].to_dict()
                name = str(rows.get("股票简称") or rows.get("简称") or code).strip()
            self._name_cache[code] = name
            return name
        except Exception as exc:  # noqa: BLE001
            logger.warning("获取 %s 公司名称失败，回退为代码: %s", code, exc)
            return code

    def remember_company_name(self, ticker: str, name: str) -> None:
        code = normalize_ticker(ticker)
        cleaned = (name or "").strip()
        if cleaned and cleaned != code:
            self._name_cache[code] = cleaned

    # ---------------- 财务摘要（补充新浪报表缺失的指标）----------------

    def get_abstract_metrics(self, ticker: str) -> dict[date, dict[str, float]]:
        """按报告期返回财务摘要指标。

        存在的理由很具体：**新浪的利润表里没有「扣非净利润」这一列**
        （实测 83 列里一列都没有）。而扣非净利润是 md 指标体系的必需项——
        成长性分析要它，风险规则"扣非净利增速 vs 营收增速背离"也要它。
        摘要接口（stock_financial_abstract）是转置表：行是指标、列是报告期，
        里面有扣非净利润，单位已是元。

        失败返回空 dict：这是补充数据源，拿不到就让对应字段保持 None，
        不能因为它挂了而让整张利润表不可用。
        """
        code = normalize_ticker(ticker)
        if code in self._abstract_cache:
            return self._abstract_cache[code]

        result: dict[date, dict[str, float]] = {}
        try:
            import akshare as ak

            frame = _call_with_timeout(ak.stock_financial_abstract, symbol=code)
            if frame is None or frame.empty:
                raise DataSourceError(f"{code} 财务摘要返回为空")

            group_column = find_column(frame.columns, ["选项", "分类"])
            metric_column = find_column(frame.columns, ["指标", "项目"])
            if metric_column is None:
                raise DataSourceError("财务摘要缺少「指标」列，接口结构可能已变更")

            # 报告期列名形如 "20241231"，与指标列区分开
            period_columns: list[tuple[str, date]] = []
            for column in frame.columns:
                parsed = parse_report_date(column)
                if parsed is not None and str(column).strip().isdigit():
                    period_columns.append((str(column), parsed))

            for _, row in frame.iterrows():
                # 同一指标名会在「常用指标」「盈利能力」等多个分组里重复出现，
                # 且口径不同（例如 ROE 有平均/摊薄两种），只认常用指标分组
                if group_column is not None and str(row[group_column]).strip() != "常用指标":
                    continue
                field = ABSTRACT_METRIC_MAP.get(str(row[metric_column]).strip())
                if field is None:
                    continue
                for column_name, period_end in period_columns:
                    value = parse_amount(row[column_name])
                    if value is None:
                        continue
                    if field in ABSTRACT_PERCENT_FIELDS:
                        value = value / 100
                    result.setdefault(period_end, {})[field] = value

            logger.debug("%s 财务摘要解析出 %d 个报告期", code, len(result))
        except Exception as exc:  # noqa: BLE001 — 补充数据源，失败不影响主流程
            logger.warning("获取 %s 财务摘要失败（扣非净利润等指标将缺失）: %s", code, exc)

        self._abstract_cache[code] = result
        return result

    # ---------------- 三大报表 ----------------

    def _fetch_sina_report(self, ticker: str, report_key: str) -> Any:
        """拉取新浪财报原始 DataFrame。"""
        import akshare as ak
        import pandas as pd

        symbol = SINA_REPORT_TYPES[report_key]
        stock = exchange_prefix(ticker)
        frame = _call_with_timeout(ak.stock_financial_report_sina, stock=stock, symbol=symbol)
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            raise DataSourceError(f"{ticker} 的{symbol}返回为空，可能该标的不存在或接口异常")
        return frame

    @staticmethod
    def _select_annual_rows(frame: Any, periods: int) -> list[tuple[date, Any]]:
        """从财报宽表里挑出最近 N 个**年报**期，按报告期倒序。

        只取年报而不是"最近 N 期"：md 的指标体系（营收增速、毛利率同比）
        都是同比口径，混入季报会让"上一期"变成上一季度，同比变环比，
        算出来的增速在业务上完全没有意义。
        """
        date_column = find_column(frame.columns, REPORT_DATE_ALIASES)
        if date_column is None:
            # 有的接口把报告期放在索引上
            candidates = [(parse_report_date(idx), row) for idx, row in frame.iterrows()]
        else:
            candidates = [
                (parse_report_date(row[date_column]), row) for _, row in frame.iterrows()
            ]

        annual = [
            (period, row)
            for period, row in candidates
            if period is not None and period.month == 12
        ]
        annual.sort(key=lambda item: item[0], reverse=True)
        if not annual:
            # 该标的可能只有季报数据（次新股），退回全部报告期
            fallback = [(p, r) for p, r in candidates if p is not None]
            fallback.sort(key=lambda item: item[0], reverse=True)
            logger.warning("未找到年报数据，退回使用全部报告期（同比口径将不准确）")
            return fallback[:periods]
        return annual[:periods]

    def get_income_statements(self, ticker: str, periods: int = 3) -> list[IncomeRow]:
        frame = self._fetch_sina_report(ticker, "income")
        rows: list[IncomeRow] = []
        for period_end, raw in self._select_annual_rows(frame, periods):
            values: dict[str, Any] = {"period_end": period_end}
            for field, aliases in INCOME_FIELD_ALIASES.items():
                column = find_column(frame.columns, aliases)
                values[field] = parse_amount(raw[column]) if column is not None else None
            # 归母净利润缺失时退回净利润总额，并在日志里留痕。
            # 这是口径降级，Evidence 的 statement_scope 仍标 consolidated，
            # 但研报里必须说明"该期未披露归母口径"。
            if values.get("net_profit") is None and values.get("total_net_profit") is not None:
                logger.info("%s %s 缺少归母净利润，回退使用净利润总额", ticker, period_end)
                values["net_profit"] = values["total_net_profit"]
            values["report_type"] = classify_report_type(period_end)
            rows.append(IncomeRow(**values))
        if not rows:
            raise DataSourceError(f"{ticker} 未解析出任何利润表数据")

        self._enrich_with_abstract(ticker, rows)
        return rows

    def _enrich_with_abstract(self, ticker: str, rows: list[IncomeRow]) -> None:
        """用财务摘要补齐新浪利润表缺失的字段（目前只有扣非净利润）。

        只在字段为 None 时填充，不覆盖报表原值——报表是一手来源，
        摘要是二手加工，两者冲突时以报表为准。
        """
        if all(row.deducted_net_profit is not None for row in rows):
            return
        abstract = self.get_abstract_metrics(ticker)
        if not abstract:
            return
        for row in rows:
            metrics = abstract.get(row.period_end)
            if not metrics:
                continue
            if row.deducted_net_profit is None and "deducted_net_profit" in metrics:
                row.deducted_net_profit = metrics["deducted_net_profit"]

    def get_balance_sheets(self, ticker: str, periods: int = 3) -> list[BalanceSheetRow]:
        frame = self._fetch_sina_report(ticker, "balance")
        rows: list[BalanceSheetRow] = []
        for period_end, raw in self._select_annual_rows(frame, periods):
            values: dict[str, Any] = {"period_end": period_end}
            for field, aliases in BALANCE_FIELD_ALIASES.items():
                column = find_column(frame.columns, aliases)
                values[field] = parse_amount(raw[column]) if column is not None else None
            if values.get("net_assets") is None and values.get("total_equity") is not None:
                values["net_assets"] = values["total_equity"]
            # 总负债缺失时用"总资产 - 所有者权益"倒算，这是恒等式，不是估算
            if (
                values.get("total_liabilities") is None
                and values.get("total_assets") is not None
                and values.get("total_equity") is not None
            ):
                values["total_liabilities"] = values["total_assets"] - values["total_equity"]
            rows.append(BalanceSheetRow(**values))
        if not rows:
            raise DataSourceError(f"{ticker} 未解析出任何资产负债表数据")
        return rows

    def get_cash_flows(self, ticker: str, periods: int = 3) -> list[CashFlowRow]:
        frame = self._fetch_sina_report(ticker, "cashflow")
        rows: list[CashFlowRow] = []
        for period_end, raw in self._select_annual_rows(frame, periods):
            values: dict[str, Any] = {"period_end": period_end}
            for field, aliases in CASHFLOW_FIELD_ALIASES.items():
                column = find_column(frame.columns, aliases)
                values[field] = parse_amount(raw[column]) if column is not None else None
            rows.append(CashFlowRow(**values))
        if not rows:
            raise DataSourceError(f"{ticker} 未解析出任何现金流量表数据")
        return rows

    # ---------------- 核心指标 ----------------

    def get_financial_metrics(self, ticker: str) -> FinancialMetrics:
        """汇总核心指标。

        需要三张表联合计算（ROE 要资产负债表、现金流质量要现金流量表），
        因此这里串联三次调用。任一张表失败不整体失败，缺哪部分指标置 None——
        缺 ROE 的研报仍然有价值，报错的研报没有。
        """
        code = normalize_ticker(ticker)
        company = self.get_company_name(code)

        incomes: list[IncomeRow] = []
        balances: list[BalanceSheetRow] = []
        cashflows: list[CashFlowRow] = []

        try:
            incomes = self.get_income_statements(code, periods=2)
        except DataSourceError as exc:
            logger.warning("获取 %s 利润表失败: %s", code, exc)
        try:
            balances = self.get_balance_sheets(code, periods=2)
        except DataSourceError as exc:
            logger.warning("获取 %s 资产负债表失败: %s", code, exc)
        try:
            cashflows = self.get_cash_flows(code, periods=1)
        except DataSourceError as exc:
            logger.warning("获取 %s 现金流量表失败: %s", code, exc)

        if not incomes and not balances:
            raise DataSourceError(f"{code} 三大报表均获取失败，无法计算财务指标")

        latest_income = incomes[0] if incomes else None
        previous_income = incomes[1] if len(incomes) > 1 else None
        latest_balance = balances[0] if balances else None
        latest_cashflow = cashflows[0] if cashflows else None

        as_of = (
            latest_income.period_end
            if latest_income
            else (latest_balance.period_end if latest_balance else date.today())
        )

        cashflow_ratio = None
        if latest_cashflow is not None:
            # 现金流量表里的"净利润"是含少数股东的口径；
            # 为与研报口径一致，优先用利润表的归母净利润做分母。
            profit_base = (
                latest_income.net_profit
                if latest_income and latest_income.net_profit
                else latest_cashflow.net_profit
            )
            cashflow_ratio = safe_divide(latest_cashflow.operating_cashflow, profit_base)

        # 自算口径：归母净利润 / 期末归母股东权益（摊薄）
        roe = safe_divide(
            latest_income.net_profit if latest_income else None,
            latest_balance.net_assets if latest_balance else None,
        )
        roe_scope = ROE_SCOPE_DILUTED
        if roe is None:
            # 资产负债表获取失败时退回摘要接口的 ROE。
            # 注意这是**不同口径**：摘要给的是平均净资产收益率（分母用期初期末平均，
            # 且剔除其他权益工具如永续债），实测比摊薄口径高 2~4pp。
            # 必须把口径差异写进 source 和 roe_scope，否则读者会拿它和其他期的摊薄 ROE 直接比。
            fallback_roe = self.get_abstract_metrics(code).get(as_of, {}).get("roe")
            if fallback_roe is not None:
                roe = fallback_roe
                roe_scope = ROE_SCOPE_AVERAGE
                logger.info("%s 资产负债表缺失，ROE 退回摘要平均口径", code)

        return FinancialMetrics(
            company=company,
            ticker=code,
            as_of=as_of,
            revenue=latest_income.revenue if latest_income else None,
            net_profit=latest_income.net_profit if latest_income else None,
            deducted_net_profit=latest_income.deducted_net_profit if latest_income else None,
            gross_margin=latest_income.gross_margin if latest_income else None,
            net_margin=latest_income.net_margin if latest_income else None,
            roe=roe,
            revenue_yoy=growth_rate(
                latest_income.revenue if latest_income else None,
                previous_income.revenue if previous_income else None,
            ),
            net_profit_yoy=growth_rate(
                latest_income.net_profit if latest_income else None,
                previous_income.net_profit if previous_income else None,
            ),
            deducted_net_profit_yoy=growth_rate(
                latest_income.deducted_net_profit if latest_income else None,
                previous_income.deducted_net_profit if previous_income else None,
            ),
            rd_ratio=latest_income.rd_ratio if latest_income else None,
            debt_ratio=latest_balance.debt_ratio if latest_balance else None,
            cashflow_to_profit=cashflow_ratio,
            roe_scope=roe_scope,
            source=f"AKShare（新浪财报接口，合并报表口径）；ROE 口径：{roe_scope}",
            disclosure_date=estimate_disclosure_date(as_of),
        )

    # ---------------- 分部构成与运营效率 ----------------

    def get_segment_rows(self, ticker: str, category: str = "按产品分类") -> list[SegmentRow]:
        """主营构成：分行业/分产品/分地区的收入、成本、毛利率、占比。

        这是三大报表之外最有分析价值的一块数据：整体毛利率的变动可能完全
        来自业务占比变化（结构效应），而各业务自身毛利率并未变动。
        没有它，归因只能停在"成本增速快于收入增速"这种同义反复上。
        """
        import akshare as ak

        code = normalize_ticker(ticker)
        symbol = f"{exchange_prefix(code)[:2].upper()}{code}"
        frame = _call_with_timeout(ak.stock_zygc_em, symbol=symbol)
        if frame is None or frame.empty:
            raise DataSourceError(f"{code} 未返回主营构成数据")

        rows: list[SegmentRow] = []
        for _, raw in frame.iterrows():
            if str(raw.get("分类类型", "")).strip() != category:
                continue
            period_end = parse_report_date(raw.get("报告日期"))
            if period_end is None:
                continue
            revenue = parse_amount(raw.get("主营收入"))
            cost = parse_amount(raw.get("主营成本"))
            rows.append(
                SegmentRow(
                    period_end=period_end,
                    category_type=category,
                    segment_name=str(raw.get("主营构成", "")).strip() or "未命名分部",
                    revenue=revenue,
                    cost=cost,
                    profit=parse_amount(raw.get("主营利润")),
                    revenue_share=parse_ratio(raw.get("收入比例")),
                    # 接口自带毛利率，但仍优先自算：口径可控且与报表层一致，
                    # 接口值只在自算不出来时兜底
                    gross_margin=safe_divide(revenue - cost, revenue)
                    if revenue is not None and cost is not None
                    else parse_ratio(raw.get("毛利率")),
                )
            )
        if not rows:
            raise DataSourceError(f"{code} 的主营构成中没有「{category}」维度的数据")
        return rows

    def get_efficiency_indicators(
        self, ticker: str, start_year: str | None = None
    ) -> dict[date, dict[str, float]]:
        """运营效率指标：应收/存货/固定资产/总资产周转率与周转天数。

        杜邦分析的第二个因子就是资产周转率，此前是用「营收/期末总资产」
        自算的粗略值（F-007 记录了这个偏差）。这个接口提供现成的、
        用平均资产计算的周转率，同时还带周转天数——
        后者在解释"应收账款风险"时比周转率更直观。
        """
        import akshare as ak

        code = normalize_ticker(ticker)
        year = start_year or str(date.today().year - 3)
        frame = _call_with_timeout(
            ak.stock_financial_analysis_indicator, symbol=code, start_year=year
        )
        if frame is None or frame.empty:
            raise DataSourceError(f"{code} 未返回财务分析指标")

        result: dict[date, dict[str, float]] = {}
        for _, raw in frame.iterrows():
            period_end = parse_report_date(raw.get("日期"))
            if period_end is None:
                continue
            metrics: dict[str, float] = {}
            for field, aliases in EFFICIENCY_FIELD_ALIASES.items():
                column = find_column(frame.columns, aliases)
                if column is None:
                    continue
                # 周转率/天数是倍数和天数，不是比率，不能走 parse_ratio 的
                # "大于 1.5 就除以 100" 逻辑——存货周转率 5.2 会被变成 0.052
                value = parse_amount(raw[column])
                if value is None:
                    continue
                metrics[field] = value / 100 if field in EFFICIENCY_PERCENT_FIELDS else value
            if metrics:
                result[period_end] = metrics
        return result

    # ---------------- 行情与新闻 ----------------

    def get_price_history(self, ticker: str, days: int = 30) -> list[PriceBar]:
        """日 K。实现委托给 tools/price_history.py（腾讯 → 新浪 → 东财）。"""
        from tools.price_history import fetch_price_history

        return fetch_price_history(normalize_ticker(ticker), days=days)

    def get_news(self, ticker: str, days: int = 7) -> list[NewsItem]:
        """个股新闻。实现委托给 tools/news_search.py，避免逻辑重复。"""
        from tools.news_search import fetch_stock_news

        return fetch_stock_news(normalize_ticker(ticker), days=days)


def warmup_akshare_stack() -> None:
    """在主线程完成 numpy/pandas/akshare 的首次 import。

    这三套库的首次加载不是线程安全的。BrokerToolRegistry 并发执行时，
    多个工具会同时 `import akshare`，numpy 会卡在
    「partially initialized module」上（评测用例 1 已复现）。
    主线程预热后，工作线程只是读 sys.modules，不再竞态。
    """
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import akshare  # noqa: F401


# 进程内共享的数据源实例（内部有公司名缓存）
_SHARED_CLIENT: list[AKShareDataClient] = []


def get_data_client() -> AKShareDataClient:
    if not _SHARED_CLIENT:
        warmup_akshare_stack()
        _SHARED_CLIENT.append(AKShareDataClient())
    return _SHARED_CLIENT[0]


# ============================================================
# 工具实现
# ============================================================

TICKER_PARAM = {
    "type": "string",
    "description": "A股股票代码，6位数字，例如 002594（比亚迪）、600519（贵州茅台）、000001（平安银行）",
}


class GetFinancialMetricsTool(BaseTool):
    """核心财务指标（毛利率/净利率/ROE/增速/负债率/现金流质量）。"""

    name = "get_financial_metrics"
    description = (
        "获取指定 A 股公司最新一期的核心财务指标，包括营业收入、归母净利润、扣非净利润、"
        "毛利率、净利率、ROE、营收同比增速、净利润同比增速、研发费用占比、资产负债率、"
        "经营现金流/净利润。适用于「某公司某指标是多少」这类单点事实问题。"
        "返回的金额单位为元，比率为小数（0.218 表示 21.8%）。"
    )
    parameters = {
        "type": "object",
        "properties": {"ticker": TICKER_PARAM},
        "required": ["ticker"],
    }

    def __init__(self, client: AKShareDataClient | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.client = client or get_data_client()

    def fetch(self, ticker: str) -> dict[str, Any]:
        metrics = self.client.get_financial_metrics(ticker)
        payload = metrics.to_tool_payload()
        payload["interpretation"] = self._interpret(metrics)
        return payload

    @staticmethod
    def _interpret(metrics: FinancialMetrics) -> list[str]:
        """把关键指标翻译成一句话结论，减少 LLM 自行推算出错的机会。"""
        notes: list[str] = []
        if metrics.roe is not None:
            level = "偏低" if metrics.roe < 0.08 else ("较高" if metrics.roe > 0.15 else "中等")
            notes.append(
                f"ROE {format_ratio(metrics.roe)}，处于{level}水平；"
                f"计算口径：{metrics.roe_scope or ROE_SCOPE_DILUTED}"
                "（期末摊薄与平均净资产口径可能相差数个百分点，不可直接混比）"
            )
        if metrics.gross_margin is not None:
            notes.append(f"毛利率 {format_ratio(metrics.gross_margin)}")
        if metrics.revenue_yoy is not None and metrics.net_profit_yoy is not None:
            divergence = (metrics.net_profit_yoy - metrics.revenue_yoy) * 100
            notes.append(
                f"营收同比 {format_ratio(metrics.revenue_yoy)}、"
                f"净利同比 {format_ratio(metrics.net_profit_yoy)}，"
                f"利润增速{'快于' if divergence > 0 else '慢于'}收入 {abs(divergence):.1f}pp"
            )
        if metrics.cashflow_to_profit is not None and metrics.cashflow_to_profit < 0.8:
            notes.append(
                f"经营现金流/净利润 {metrics.cashflow_to_profit:.2f}，低于 0.8 预警线，利润质量存疑"
            )
        return notes


class GetIncomeHistoryTool(BaseTool):
    """多期利润表。"""

    name = "get_income_history"
    description = (
        "获取指定 A 股公司最近 N 期年报的利润表数据，含营业收入、营业成本、毛利、"
        "归母净利润、扣非净利润、研发费用，并自动计算各期毛利率、净利率及同比增速。"
        "适用于趋势分析和归因分析（至少需要 3 期才能看出趋势）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "ticker": TICKER_PARAM,
            "periods": {
                "type": "integer",
                "description": "获取的年报期数，默认 3，最大 5。分析趋势建议至少 3 期",
                "default": 3,
            },
        },
        "required": ["ticker"],
    }

    def __init__(self, client: AKShareDataClient | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.client = client or get_data_client()

    def fetch(self, ticker: str, periods: int = 3) -> dict[str, Any]:
        periods = max(1, min(int(periods), 5))
        code = normalize_ticker(ticker)
        rows = self.client.get_income_statements(code, periods=periods)
        company = self.client.get_company_name(code)

        # 按时间正序输出，便于 LLM 顺着读趋势
        ordered = sorted(rows, key=lambda row: row.period_end)
        periods_payload: list[dict[str, Any]] = []
        for index, row in enumerate(ordered):
            item = row.to_display()
            previous = ordered[index - 1] if index > 0 else None
            item["revenue_yoy"] = growth_rate(
                row.revenue, previous.revenue if previous else None
            )
            item["net_profit_yoy"] = growth_rate(
                row.net_profit, previous.net_profit if previous else None
            )
            item["gross_margin_change_pp"] = (
                (row.gross_margin - previous.gross_margin) * 100
                if previous and row.gross_margin is not None and previous.gross_margin is not None
                else None
            )
            periods_payload.append(item)

        return {
            "company": company,
            "ticker": code,
            "period_count": len(periods_payload),
            "periods": periods_payload,
            "source": "AKShare 新浪财报接口 · 合并报表 · 年报口径",
            "disclosure_dates": {
                row.period_end.isoformat(): estimate_disclosure_date(row.period_end).isoformat()
                for row in ordered
            },
            "unit_note": "金额单位为元；毛利率/净利率为小数；同比增速为小数（0.42 表示 +42%）",
        }


class GetBalanceSheetTool(BaseTool):
    """多期资产负债表。"""

    name = "get_balance_sheet"
    description = (
        "获取指定 A 股公司最近 N 期年报的资产负债表关键科目，含总资产、总负债、"
        "归母净资产、应收账款、存货、商誉、短期借款，并自动计算资产负债率、"
        "商誉占净资产比、权益乘数。适用于风险评估和杜邦分析。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "ticker": TICKER_PARAM,
            "periods": {
                "type": "integer",
                "description": "获取的年报期数，默认 3，最大 5",
                "default": 3,
            },
        },
        "required": ["ticker"],
    }

    def __init__(self, client: AKShareDataClient | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.client = client or get_data_client()

    def fetch(self, ticker: str, periods: int = 3) -> dict[str, Any]:
        periods = max(1, min(int(periods), 5))
        code = normalize_ticker(ticker)
        rows = sorted(
            self.client.get_balance_sheets(code, periods=periods),
            key=lambda row: row.period_end,
        )
        payload: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            item = row.to_display()
            previous = rows[index - 1] if index > 0 else None
            item["accounts_receivable_yoy"] = growth_rate(
                row.accounts_receivable, previous.accounts_receivable if previous else None
            )
            payload.append(item)

        return {
            "company": self.client.get_company_name(code),
            "ticker": code,
            "period_count": len(payload),
            "periods": payload,
            "source": "AKShare 新浪财报接口 · 合并报表 · 年报口径",
            "thresholds": {
                "debt_ratio_alert": "资产负债率 > 70% 视为高杠杆（金融业除外）",
                "goodwill_alert": "商誉/净资产 > 30% 存在减值风险",
            },
            "unit_note": "金额单位为元；比率为小数",
        }


class GetCashFlowTool(BaseTool):
    """多期现金流量表。"""

    name = "get_cash_flow"
    description = (
        "获取指定 A 股公司最近 N 期年报的现金流量表，含经营、投资、筹资三大活动现金流净额，"
        "并计算经营现金流/净利润比值（利润质量指标，低于 0.8 为预警）。"
        "适用于利润质量核查和风险评估。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "ticker": TICKER_PARAM,
            "periods": {
                "type": "integer",
                "description": "获取的年报期数，默认 3，最大 5",
                "default": 3,
            },
        },
        "required": ["ticker"],
    }

    def __init__(self, client: AKShareDataClient | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.client = client or get_data_client()

    def fetch(self, ticker: str, periods: int = 3) -> dict[str, Any]:
        periods = max(1, min(int(periods), 5))
        code = normalize_ticker(ticker)
        cashflows = sorted(
            self.client.get_cash_flows(code, periods=periods), key=lambda row: row.period_end
        )

        # 现金流量表自带的"净利润"是含少数股东口径，用利润表的归母口径覆盖，
        # 保证 cashflow_to_profit 与研报其他地方的净利润口径一致
        try:
            incomes = {
                row.period_end: row
                for row in self.client.get_income_statements(code, periods=periods)
            }
        except DataSourceError as exc:
            logger.warning("补充利润表失败，现金流比值将使用报表自带净利润: %s", exc)
            incomes = {}

        payload: list[dict[str, Any]] = []
        for row in cashflows:
            income = incomes.get(row.period_end)
            if income is not None and income.net_profit is not None:
                row.net_profit = income.net_profit
            item = row.to_display()
            item["profit_scope"] = "归母净利润" if income else "报表自带净利润（可能含少数股东）"
            payload.append(item)

        return {
            "company": self.client.get_company_name(code),
            "ticker": code,
            "period_count": len(payload),
            "periods": payload,
            "source": "AKShare 新浪财报接口 · 合并报表 · 年报口径",
            "threshold_note": "经营现金流/净利润 < 0.8 表示利润未充分转化为现金，利润质量存疑",
            "unit_note": "金额单位为元",
        }


class GetStockPriceTool(BaseTool):
    """日 K 线行情。"""

    name = "get_stock_price"
    description = (
        "获取指定 A 股公司最近 N 个交易日的日 K 线（开高低收、成交量、涨跌幅），"
        "并给出区间涨跌幅和波动率。默认走腾讯前复权，失败则新浪、东方财富依次兜底。"
        "用于补充市场表现视角，不用于财务分析。"
    )
    # 行情时效性强，缓存会导致盘中数据陈旧，因此关闭缓存
    cacheable = False
    parameters = {
        "type": "object",
        "properties": {
            "ticker": TICKER_PARAM,
            "days": {
                "type": "integer",
                "description": "获取最近多少个交易日，默认 30，最大 250",
                "default": 30,
            },
        },
        "required": ["ticker"],
    }

    def __init__(self, client: AKShareDataClient | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.client = client or get_data_client()

    def fetch(self, ticker: str, days: int = 30) -> dict[str, Any]:
        from tools.price_history import fetch_price_history_with_meta

        days = max(1, min(int(days), 250))
        code = normalize_ticker(ticker)
        fetched = fetch_price_history_with_meta(code, days=days)
        bars = fetched.bars
        if not bars:
            raise DataSourceError(f"{code} 未获取到行情数据")

        if fetched.company_name:
            self.client.remember_company_name(code, fetched.company_name)
            company = fetched.company_name
        else:
            company = self.client.get_company_name(code)

        closes = [bar.close for bar in bars if bar.close is not None]
        period_return = (
            (closes[-1] - closes[0]) / closes[0] if len(closes) >= 2 and closes[0] else None
        )
        volatility = None
        if len(closes) >= 3:
            daily_returns = [
                (closes[i] - closes[i - 1]) / closes[i - 1]
                for i in range(1, len(closes))
                if closes[i - 1]
            ]
            if daily_returns:
                mean_return = sum(daily_returns) / len(daily_returns)
                variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
                volatility = variance**0.5

        adjust_note = (
            "价格为前复权，可跨除权日比较"
            if fetched.is_qfq
            else "该路数据为不复权；窗口内若有除权，区间涨跌可能含缺口"
        )
        return {
            "company": company,
            "ticker": code,
            "trading_days": len(bars),
            "start_date": bars[0].trade_date.isoformat(),
            "end_date": bars[-1].trade_date.isoformat(),
            "latest_close": closes[-1] if closes else None,
            "period_return": period_return,
            "period_return_display": format_ratio(period_return),
            "daily_volatility": volatility,
            # 只回传最近 20 根，完整序列对 LLM 没有增量信息但很占 token
            "recent_bars": [
                {
                    "date": bar.trade_date.isoformat(),
                    "close": bar.close,
                    "change_pct": bar.change_pct,
                    "volume": bar.volume,
                }
                for bar in bars[-20:]
            ],
            "source": fetched.source_label,
            "provider": fetched.provider,
            "adjust_note": adjust_note,
            "retrieval_notes": fetched.notes,
        }


class GetRiskSnapshotTool(BaseTool):
    """风险预警快照（组合三张报表跑完整规则集）。

    # TODO: md 的工具表里没有这个工具，但"风险评估类问题（risk）"要求
    # 同时读资产负债表 + 现金流量表并逐条比对预警阈值。让 LLM 自己串三次工具
    # 再心算七条规则，出错概率很高且不可复现。把规则集做成确定性工具，
    # LLM 只负责解释结果，这与"金融计算不交给 LLM"的整体原则一致。
    """

    name = "get_risk_snapshot"
    description = (
        "对指定 A 股公司执行完整的财务风险规则检查，一次性返回所有触发的预警项，"
        "包括：应收账款增速与营收增速背离、经营现金流对净利润覆盖不足、资产负债率过高、"
        "商誉占净资产比过高、扣非净利与营收增速背离、毛利率显著下滑、ROE 偏低。"
        "适用于「某公司有什么风险」「应收账款风险大吗」这类风险评估问题。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "ticker": TICKER_PARAM,
            "periods": {
                "type": "integer",
                "description": "用于比较的年报期数，默认 3",
                "default": 3,
            },
        },
        "required": ["ticker"],
    }

    def __init__(self, client: AKShareDataClient | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.client = client or get_data_client()

    def fetch(self, ticker: str, periods: int = 3) -> dict[str, Any]:
        periods = max(2, min(int(periods), 5))
        code = normalize_ticker(ticker)

        incomes = self._safe(lambda: self.client.get_income_statements(code, periods), "利润表")
        balances = self._safe(lambda: self.client.get_balance_sheets(code, periods), "资产负债表")
        cashflows = self._safe(lambda: self.client.get_cash_flows(code, periods), "现金流量表")

        if not any([incomes, balances, cashflows]):
            raise DataSourceError(f"{code} 三大报表均获取失败，无法执行风险检查")

        # detect_risk_alerts 约定倒序输入（最新在前）
        incomes = sorted(incomes, key=lambda row: row.period_end, reverse=True)
        balances = sorted(balances, key=lambda row: row.period_end, reverse=True)
        cashflows = sorted(cashflows, key=lambda row: row.period_end, reverse=True)

        if incomes and cashflows and cashflows[0].net_profit is None:
            cashflows[0].net_profit = incomes[0].net_profit

        alerts = detect_risk_alerts(incomes, balances, cashflows)
        dupont = None
        if incomes and balances:
            dupont = compute_dupont(
                incomes[0],
                balances[0],
                incomes[1] if len(incomes) > 1 else None,
                balances[1] if len(balances) > 1 else None,
            )

        missing = [
            label
            for label, rows in (("利润表", incomes), ("资产负债表", balances), ("现金流量表", cashflows))
            if not rows
        ]

        return {
            "company": self.client.get_company_name(code),
            "ticker": code,
            "as_of": incomes[0].period_end.isoformat() if incomes else None,
            "alert_count": len(alerts),
            "alerts": [alert.model_dump() for alert in alerts],
            "dupont": dupont.to_display() if dupont else None,
            "checked_rules": [
                "应收账款增速 vs 营收增速（阈值：1.5 倍）",
                "经营现金流/净利润（阈值：0.8）",
                "资产负债率（阈值：70%）",
                "商誉/净资产（阈值：30%）",
                "扣非净利增速 vs 营收增速（阈值：10pp）",
                "毛利率同比变动（阈值：-3pp）",
                "ROE 水平（阈值：8%）",
            ],
            # 显式声明哪些规则因数据缺失未能执行，避免把"没查"说成"没风险"
            "unavailable_data": missing,
            "coverage_note": (
                "以下报表缺失，相关规则未执行：" + "、".join(missing)
                if missing
                else "三大报表齐全，全部规则已执行"
            ),
            "source": "AKShare 新浪财报接口 · 合并报表 · 年报口径",
        }

    @staticmethod
    def _safe(loader: Callable[[], list[Any]], label: str) -> list[Any]:
        try:
            return loader()
        except DataSourceError as exc:
            logger.warning("风险快照获取%s失败: %s", label, exc)
            return []


def build_akshare_tools(client: AKShareDataClient | None = None) -> list[BaseTool]:
    """构造全部 AKShare 工具实例。

    共享同一个 client，让公司名缓存和连接复用生效。
    """
    shared = client or get_data_client()
    return [
        GetFinancialMetricsTool(client=shared),
        GetIncomeHistoryTool(client=shared),
        GetBalanceSheetTool(client=shared),
        GetCashFlowTool(client=shared),
        GetStockPriceTool(client=shared),
        GetRiskSnapshotTool(client=shared),
    ]


__all__ = [
    "AKShareDataClient",
    "GetBalanceSheetTool",
    "GetCashFlowTool",
    "GetFinancialMetricsTool",
    "GetIncomeHistoryTool",
    "GetRiskSnapshotTool",
    "GetStockPriceTool",
    "TickerNotFoundError",
    "build_akshare_tools",
    "get_data_client",
]
