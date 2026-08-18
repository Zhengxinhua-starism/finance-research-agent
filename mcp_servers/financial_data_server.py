"""金融数据 MCP Server。

解决什么问题
    把 7 个财务数据工具（6 个 AKShare 数据工具 + 1 个跨期对比计算工具）
    通过 MCP 协议统一暴露。Agent 只知道"有个叫 financial_data 的 Server，
    它能列出自己有哪些工具"，不知道底层是 AKShare 还是别的数据源。

核心设计决策
    1. Server 持有一个共享的 AKShareDataClient 实例传给所有工具。
       每个工具各建一个 client 会让公司名缓存失效（同一次研究里
       get_company_name 会被调 4~5 次），也会浪费连接。
    2. 工具集比 md 表格里的 6 个多一个 get_risk_snapshot，少一个
       search_news（新闻工具在 tools/news_search.py 里实现，仍注册在本 Server）。
       实际注册的是 7 个：5 个报表/行情工具 + 风险快照 + 新闻 + 跨期对比 = 8 个。
       # TODO: md 的工具表和 mcp_servers 章节对"6 个工具"的口径不一致
       # （表格里 search_news 算在 AKShare 工具内，但它其实是独立数据源）。
       # 这里按功能完整性注册全部 8 个，并在 health() 里分类标注来源，
       # 而不是为了凑数字砍掉风险快照这种高价值工具。
    3. 提供 evidence_from_tool_result()：把工具结果转成 Evidence 对象。
       这一步必须在最靠近数据源的地方做——只有这里知道
       "这个数字来自年报还是推算""披露日期是哪天""是不是合并口径"。
       让 Verifier 去猜这些元信息就等于放弃了证据体系。

为什么不用其他方案
    - 不给每个工具单独建一个 MCP Server：MCP Server 的粒度应该对应
      "一个数据域"，不是"一个函数"。8 个 Server 只会让配置爆炸。
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from harness.tracing import Tracer
from harness.types import Evidence, ToolResult
from mcp_servers.base_server import MCPServer
from tools.akshare_tools import AKShareDataClient, build_akshare_tools, get_data_client
from tools.compare_periods import ComparePeriodsTool
from tools.data_client import estimate_disclosure_date, parse_report_date
from tools.disclosure import GetAnalystCoverageTool, GetDisclosureEventsTool
from tools.news_search import SearchNewsTool
from tools.segment_analysis import GetOperatingEfficiencyTool, GetSegmentBreakdownTool

logger = logging.getLogger(__name__)

# 工具名 → 证据来源类型。决定该工具产出的数字在门禁里的可信等级。
TOOL_SOURCE_TYPE: dict[str, str] = {
    "get_financial_metrics": "derived",  # 由报表推算的指标
    "get_income_history": "annual_report",
    "get_balance_sheet": "annual_report",
    "get_cash_flow": "annual_report",
    "get_stock_price": "market_data",
    "get_risk_snapshot": "derived",
    "compare_periods": "derived",
    "search_news": "news",
    "get_segment_breakdown": "annual_report",
    "get_operating_efficiency": "derived",
    "get_disclosure_events": "announcement",
    # 第二层：机构判断。is_official=False，最高只能拿 ⚠️未验证
    "get_analyst_coverage": "research_report",
}

# 工具名 → 人类可读的来源名称模板
TOOL_SOURCE_NAME: dict[str, str] = {
    "get_financial_metrics": "{company} 核心财务指标（报表推算）",
    "get_income_history": "{company} 利润表（年报）",
    "get_balance_sheet": "{company} 资产负债表（年报）",
    "get_cash_flow": "{company} 现金流量表（年报）",
    "get_stock_price": "{company} 行情数据",
    "get_risk_snapshot": "{company} 风险规则检查（报表推算）",
    "compare_periods": "{company} 跨期对比（报表推算）",
    "search_news": "{company} 相关新闻",
    "get_segment_breakdown": "{company} 主营构成（年报分部数据）",
    "get_operating_efficiency": "{company} 运营效率指标（报表推算）",
    "get_disclosure_events": "{company} 公司公告",
    "get_analyst_coverage": "{company} 券商研报覆盖",
}


class FinancialDataMCPServer(MCPServer):
    """A 股金融数据工具集，通过 MCP 协议暴露。"""

    name = "financial_data"
    description = "A股金融数据工具集（AKShare 报表数据 + 本地确定性财务计算）"

    def __init__(
        self,
        client: AKShareDataClient | None = None,
        tracer: Tracer | None = None,
    ):
        super().__init__(tracer=tracer)
        self.client = client or get_data_client()
        self.register_tools(
            [
                *build_akshare_tools(self.client),
                SearchNewsTool(),
                ComparePeriodsTool(client=self.client),
                GetSegmentBreakdownTool(client=self.client),
                GetOperatingEfficiencyTool(client=self.client),
                GetDisclosureEventsTool(client=self.client),
                GetAnalystCoverageTool(client=self.client),
            ]
        )
        logger.info("FinancialDataMCPServer 已注册 %d 个工具", len(self.tool_names))

    # ---------------- 证据构造 ----------------

    def evidence_from_tool_result(
        self, result: ToolResult, as_of_date: date | None = None
    ) -> list[Evidence]:
        """把工具结果转成 Evidence 列表。

        一次工具调用可能产出多条证据：get_income_history 返回 3 期数据，
        每期是一条独立证据（各有自己的报告期和披露日期）。
        把它们合成一条会让时点检查失去意义——2022 年数据和 2024 年数据
        的披露日期差了两年。
        """
        if not result.success or not isinstance(result.data, dict):
            return []

        source_type = TOOL_SOURCE_TYPE.get(result.tool_name, "derived")
        payload = result.data
        company = str(payload.get("company") or payload.get("ticker") or "该公司")
        ticker = str(payload.get("ticker") or "")
        source_name = TOOL_SOURCE_NAME.get(result.tool_name, "{company} 数据").format(
            company=company
        )

        builders = {
            "get_financial_metrics": self._evidence_from_metrics,
            "get_income_history": self._evidence_from_periods,
            "get_balance_sheet": self._evidence_from_periods,
            "get_cash_flow": self._evidence_from_periods,
            "compare_periods": self._evidence_from_comparison,
            "get_risk_snapshot": self._evidence_from_risk,
            "get_stock_price": self._evidence_from_price,
            "search_news": self._evidence_from_news,
            "get_segment_breakdown": self._evidence_from_segments,
            "get_operating_efficiency": self._evidence_from_efficiency,
            "get_disclosure_events": self._evidence_from_disclosure,
            "get_analyst_coverage": self._evidence_from_analyst,
        }
        builder = builders.get(result.tool_name)
        if builder is None:
            return []

        try:
            return builder(payload, company, ticker, source_type, source_name, result.tool_name)
        except Exception as exc:  # noqa: BLE001 — 证据构造失败不能拖垮检索
            logger.warning("从 %s 结果构造证据失败: %s", result.tool_name, exc)
            return []

    # -- 各工具的证据构造实现 --

    @staticmethod
    def _evidence_from_metrics(
        payload: dict[str, Any],
        company: str,
        ticker: str,
        source_type: str,
        source_name: str,
        tool_name: str,
    ) -> list[Evidence]:
        metrics = payload.get("metrics") or {}
        numbers = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        period_end = parse_report_date(payload.get("as_of"))
        disclosure = parse_report_date(payload.get("disclosure_date")) or (
            estimate_disclosure_date(period_end) if period_end else date.today()
        )
        display = payload.get("metrics_display") or {}
        roe_scope = str(payload.get("roe_scope") or "").strip()
        source_note = str(payload.get("source") or "").strip()
        content = f"{company}({ticker}) 截至 {payload.get('as_of')} 的核心财务指标：" + "；".join(
            f"{key} {value}" for key, value in display.items()
        )
        # 口径必须写进正文：Verifier / Writer 只读 content，读不到 payload.source。
        # 摊薄 vs 平均净资产口径实测可差 2~4pp，漏写就会被评成"没说明口径"。
        if roe_scope:
            content += (
                f"。ROE 计算口径：{roe_scope}"
                "（期末摊薄与平均净资产口径可能相差数个百分点，不可直接混比）"
            )
        elif "ROE" in source_note:
            content += f"。{source_note}"
        return [
            Evidence(
                source_type=source_type,  # type: ignore[arg-type]
                source_name=source_name,
                disclosure_date=disclosure,
                period_end=period_end,
                content=content,
                numbers=numbers,
                ticker=ticker,
                company=company,
                tool_name=tool_name,
                metadata={
                    "unit_note": payload.get("unit_note", ""),
                    "roe_scope": roe_scope,
                    "source_detail": source_note,
                },
            )
        ]

    @staticmethod
    def _evidence_from_periods(
        payload: dict[str, Any],
        company: str,
        ticker: str,
        source_type: str,
        source_name: str,
        tool_name: str,
    ) -> list[Evidence]:
        evidences: list[Evidence] = []
        for period in payload.get("periods") or []:
            period_end = parse_report_date(period.get("date"))
            if period_end is None:
                continue
            numbers = {
                key: float(value)
                for key, value in period.items()
                if isinstance(value, (int, float)) and key != "date"
            }
            if not numbers:
                continue
            evidences.append(
                Evidence(
                    source_type=source_type,  # type: ignore[arg-type]
                    source_name=f"{company} {period_end.year}年报",
                    disclosure_date=estimate_disclosure_date(period_end),
                    period_end=period_end,
                    content=f"{company} {period_end.isoformat()} 报告期数据: "
                    + "；".join(f"{k}={v}" for k, v in period.items() if v is not None),
                    numbers=numbers,
                    ticker=ticker,
                    company=company,
                    tool_name=tool_name,
                    metadata={"source_detail": payload.get("source", "")},
                )
            )
        return evidences

    @staticmethod
    def _evidence_from_comparison(
        payload: dict[str, Any],
        company: str,
        ticker: str,
        source_type: str,
        source_name: str,
        tool_name: str,
    ) -> list[Evidence]:
        periods: list[str] = payload.get("periods") or []
        if not periods:
            return []
        latest_period = parse_report_date(periods[-1])
        if latest_period is None:
            return []

        comparisons = payload.get("metrics_comparison") or []
        dupont = payload.get("dupont") or {}
        roe_scope = str(
            dupont.get("roe_scope") or "归母净利润 / 期末归母股东权益（摊薄口径）"
        )
        scope_note = (
            f"ROE 计算口径：{roe_scope}。"
            "期末摊薄与平均净资产口径可能相差数个百分点，不可直接混比。"
        )

        evidences: list[Evidence] = []
        # 按报告期切开：跨期断言（"2025 ROE vs 2024 ROE"）的每个数字
        # 必须能在**自己那一年**的证据里对上。旧实现把三期合成一条、
        # numbers 只留最新值，2024 年的 ROE 在门禁里永远找不到，
        # 未验证率被顶过阈值后补搜把全部工具打开（F-022）。
        for index, period_str in enumerate(periods):
            period_end = parse_report_date(period_str)
            if period_end is None:
                continue
            numbers: dict[str, float] = {}
            lines: list[str] = []
            for comparison in comparisons:
                metric = comparison.get("metric")
                values = comparison.get("values") or []
                displays = comparison.get("values_display") or []
                if not metric or index >= len(values):
                    continue
                value = values[index]
                if isinstance(value, (int, float)):
                    numbers[metric] = float(value)
                display = displays[index] if index < len(displays) else value
                lines.append(f"{comparison.get('metric_label', metric)} {display}")
            if not numbers:
                continue
            evidences.append(
                Evidence(
                    source_type=source_type,  # type: ignore[arg-type]
                    source_name=f"{company} {period_end.year}年报跨期对比切片",
                    disclosure_date=estimate_disclosure_date(period_end),
                    period_end=period_end,
                    content=(
                        f"{scope_note}"
                        f"{company}({ticker}) {period_end.year}年报关键指标："
                        + "；".join(lines)
                    ),
                    numbers=numbers,
                    ticker=ticker,
                    company=company,
                    tool_name=tool_name,
                    metadata={"period": period_str, "roe_scope": roe_scope},
                )
            )

        trend_lines: list[str] = []
        summary_numbers: dict[str, float] = {}
        for comparison in comparisons:
            metric = comparison.get("metric")
            values = comparison.get("values") or []
            if not metric or not values:
                continue
            latest_value = values[-1]
            if isinstance(latest_value, (int, float)):
                summary_numbers[metric] = float(latest_value)
            trend_lines.append(
                f"{comparison.get('metric_label', metric)}: "
                f"{' → '.join(str(v) for v in comparison.get('values_display', []))}"
                f"（{comparison.get('trend_label', '')}）"
            )
        for key in ("roe", "net_margin", "asset_turnover", "equity_multiplier"):
            value = dupont.get(key)
            if isinstance(value, (int, float)):
                summary_numbers.setdefault(key, float(value))

        driver = str(dupont.get("driver") or "").strip()
        dupont_line = f"杜邦主导因子：{driver}。" if driver else ""
        calc_note = str(payload.get("calculation_note") or scope_note)
        evidences.append(
            Evidence(
                source_type=source_type,  # type: ignore[arg-type]
                source_name=source_name,
                disclosure_date=estimate_disclosure_date(latest_period),
                period_end=latest_period,
                content=(
                    f"{scope_note}{dupont_line}"
                    f"{company} 跨期对比（{periods[0]} ~ {periods[-1]}）：\n"
                    + "\n".join(trend_lines)
                    + (f"\n{calc_note}" if calc_note else "")
                ),
                numbers=summary_numbers,
                ticker=ticker,
                company=company,
                tool_name=tool_name,
                metadata={"periods": ", ".join(periods), "roe_scope": roe_scope},
            )
        )
        return evidences

    @staticmethod
    def _evidence_from_risk(
        payload: dict[str, Any],
        company: str,
        ticker: str,
        source_type: str,
        source_name: str,
        tool_name: str,
    ) -> list[Evidence]:
        period_end = parse_report_date(payload.get("as_of")) or date.today()
        numbers: dict[str, float] = {}
        lines: list[str] = []
        for alert in payload.get("alerts") or []:
            lines.append(f"[{alert.get('level')}] {alert.get('title')}: {alert.get('detail')}")
            for key, value in (alert.get("metrics") or {}).items():
                if isinstance(value, (int, float)):
                    numbers.setdefault(key, float(value))

        dupont = payload.get("dupont") or {}
        for key in ("roe", "net_margin", "asset_turnover", "equity_multiplier"):
            value = dupont.get(key)
            if isinstance(value, (int, float)):
                numbers.setdefault(key, float(value))

        content = (
            f"{company} 风险规则检查结果（共 {payload.get('alert_count', 0)} 项预警）:\n"
            + ("\n".join(lines) if lines else "未触发任何预警规则")
            + f"\n{payload.get('coverage_note', '')}"
        )
        return [
            Evidence(
                source_type=source_type,  # type: ignore[arg-type]
                source_name=source_name,
                disclosure_date=estimate_disclosure_date(period_end),
                period_end=period_end,
                content=content,
                numbers=numbers,
                ticker=ticker,
                company=company,
                tool_name=tool_name,
            )
        ]

    @staticmethod
    def _evidence_from_segments(
        payload: dict[str, Any],
        company: str,
        ticker: str,
        source_type: str,
        source_name: str,
        tool_name: str,
    ) -> list[Evidence]:
        """每个报告期一条证据，分部指标以 `segment_毛利率` 这样的键暴露。

        指标名带分部前缀而不是复用 `gross_margin`：整体毛利率和汽车业务毛利率
        是两个不同的数字，共用一个指标名会让门禁拿错值比对（F-018 同类问题）。
        """
        evidences: list[Evidence] = []
        for period_text, segments in (payload.get("segments_by_period") or {}).items():
            period_end = parse_report_date(period_text)
            if period_end is None:
                continue
            numbers: dict[str, float] = {}
            lines: list[str] = []
            for item in segments:
                name = str(item.get("segment", "")).strip()
                if not name:
                    continue
                for key, suffix in (("gross_margin", "毛利率"), ("revenue_share", "收入占比")):
                    value = item.get(key)
                    if isinstance(value, (int, float)):
                        numbers[f"{name}_{suffix}"] = float(value)
                revenue = item.get("revenue")
                if isinstance(revenue, (int, float)):
                    numbers[f"{name}_收入"] = float(revenue)
                lines.append(
                    f"{name}: 收入 {item.get('revenue_display')}"
                    f"（占比 {item.get('revenue_share_display')}），"
                    f"毛利率 {item.get('gross_margin_display')}"
                )
            if not numbers:
                continue
            evidences.append(
                Evidence(
                    source_type=source_type,  # type: ignore[arg-type]
                    source_name=f"{company} {period_end.year}年报分部数据（{payload.get('category')}）",
                    disclosure_date=estimate_disclosure_date(period_end),
                    period_end=period_end,
                    content=f"{company} {period_end.isoformat()} {payload.get('category')}:\n"
                    + "\n".join(lines),
                    numbers=numbers,
                    ticker=ticker,
                    company=company,
                    tool_name=tool_name,
                )
            )

        # 结构分解单独成一条证据：它描述的是"变动的成因"，不属于任何单期
        decomposition = payload.get("margin_decomposition")
        if decomposition and payload.get("periods"):
            latest = parse_report_date(payload["periods"][-1])
            if latest is not None:
                numbers = {
                    key: float(decomposition[key])
                    for key in ("overall_change_pp", "mix_effect_pp", "own_effect_pp")
                    if isinstance(decomposition.get(key), (int, float))
                }
                evidences.append(
                    Evidence(
                        source_type="derived",
                        source_name=f"{company} 毛利率变动结构分解",
                        disclosure_date=estimate_disclosure_date(latest),
                        period_end=latest,
                        content=str(decomposition.get("dominant", "")),
                        numbers=numbers,
                        ticker=ticker,
                        company=company,
                        tool_name=tool_name,
                    )
                )
        return evidences

    @staticmethod
    def _evidence_from_efficiency(
        payload: dict[str, Any],
        company: str,
        ticker: str,
        source_type: str,
        source_name: str,
        tool_name: str,
    ) -> list[Evidence]:
        periods: list[str] = payload.get("periods") or []
        if not periods:
            return []
        latest = parse_report_date(periods[-1])
        if latest is None:
            return []

        numbers: dict[str, float] = {}
        lines: list[str] = []
        for item in payload.get("indicators") or []:
            value = item.get("latest")
            if isinstance(value, (int, float)):
                numbers[item["metric"]] = float(value)
            lines.append(f"{item.get('label')}: {item.get('values')}")

        return [
            Evidence(
                source_type=source_type,  # type: ignore[arg-type]
                source_name=source_name,
                disclosure_date=estimate_disclosure_date(latest),
                period_end=latest,
                content=f"{company} 运营效率指标（{periods[0]} ~ {periods[-1]}）:\n"
                + "\n".join(lines),
                numbers=numbers,
                ticker=ticker,
                company=company,
                tool_name=tool_name,
            )
        ]

    @staticmethod
    def _evidence_from_price(
        payload: dict[str, Any],
        company: str,
        ticker: str,
        source_type: str,
        source_name: str,
        tool_name: str,
    ) -> list[Evidence]:
        end_date = parse_report_date(payload.get("end_date")) or date.today()
        numbers = {
            key: float(payload[key])
            for key in ("latest_close", "period_return", "daily_volatility")
            if isinstance(payload.get(key), (int, float))
        }
        source_note = str(payload.get("source") or "").strip()
        content = (
            f"{company} {payload.get('start_date')} ~ {payload.get('end_date')} 行情："
            f"最新收盘 {payload.get('latest_close')}，"
            f"区间涨跌 {payload.get('period_return_display')}"
        )
        if source_note:
            content += f"。数据来源：{source_note}"
        if payload.get("adjust_note"):
            content += f"。{payload['adjust_note']}"
        return [
            Evidence(
                source_type=source_type,  # type: ignore[arg-type]
                source_name=source_name,
                # 行情数据当日即可得，披露日期就是交易日
                disclosure_date=end_date,
                period_end=end_date,
                content=content,
                numbers=numbers,
                ticker=ticker,
                company=company,
                tool_name=tool_name,
                metadata={
                    "source_detail": source_note,
                    "provider": payload.get("provider"),
                    "adjust_note": payload.get("adjust_note"),
                },
            )
        ]

    @staticmethod
    def _evidence_from_disclosure(
        payload: dict[str, Any],
        company: str,
        ticker: str,
        source_type: str,
        source_name: str,
        tool_name: str,
    ) -> list[Evidence]:
        """公告证据。不提取数字——接口只给标题，从标题抠数字是过度解读。"""
        evidences: list[Evidence] = []

        relevant = payload.get("attribution_relevant_notices") or []
        if relevant:
            latest = parse_report_date(relevant[0].get("date")) or date.today()
            lines = [f"[{n.get('date')}] {n.get('title')}（{n.get('type')}）" for n in relevant[:15]]
            evidences.append(
                Evidence(
                    source_type=source_type,  # type: ignore[arg-type]
                    source_name=f"{company} 公司公告（与业绩归因相关）",
                    disclosure_date=latest,
                    content=f"{company} 近期与业绩归因相关的公告：\n" + "\n".join(lines),
                    numbers={},
                    ticker=ticker,
                    company=company,
                    tool_name=tool_name,
                    metadata={"notice_count": payload.get("notice_count", 0)},
                )
            )

        forecast = payload.get("earnings_forecast")
        if isinstance(forecast, dict) and forecast.get("exists"):
            announce = parse_report_date(forecast.get("announce_date")) or date.today()
            period_end = parse_report_date(forecast.get("period"))
            evidences.append(
                Evidence(
                    source_type=source_type,  # type: ignore[arg-type]
                    source_name=f"{company} {forecast.get('period')} 业绩预告",
                    disclosure_date=announce,
                    period_end=period_end,
                    content=(
                        f"预告类型：{forecast.get('forecast_type')}；"
                        f"指标：{forecast.get('indicator')}；"
                        f"变动：{forecast.get('change_desc')}"
                        f"（{forecast.get('change_pct')}%）\n"
                        f"公司披露的变动原因：{forecast.get('change_reason')}"
                    ),
                    # 预告里的数字是"预计值"而非最终数，不能用于校验实际业绩，
                    # 因此不进 numbers；它的价值在于那段官方的原因说明
                    numbers={},
                    ticker=ticker,
                    company=company,
                    tool_name=tool_name,
                )
            )
        return evidences

    @staticmethod
    def _evidence_from_analyst(
        payload: dict[str, Any],
        company: str,
        ticker: str,
        source_type: str,
        source_name: str,
        tool_name: str,
    ) -> list[Evidence]:
        """研报证据（第二层）。

        每家机构一条证据而不是汇总成一条：这样研报里引用时能落到
        具体机构（"东吴证券在报告中提到…"），符合第二层来源
        "必须注明机构观点"的要求。汇总成一条只能写"有机构认为"，
        既不可追溯，也容易被读成共识。
        """
        reports = payload.get("recent_reports") or []
        if not reports:
            return []

        by_institution: dict[str, list[dict[str, Any]]] = {}
        for report in reports:
            by_institution.setdefault(report.get("institution", "未知机构"), []).append(report)

        evidences: list[Evidence] = []
        for institution, items in list(by_institution.items())[:8]:
            latest = parse_report_date(items[0].get("date")) or date.today()
            evidences.append(
                Evidence(
                    source_type="research_report",
                    source_name=f"{institution}（研报，机构观点）",
                    disclosure_date=latest,
                    content=f"{institution} 近期关于{company}的研报标题：\n"
                    + "\n".join(f"[{i.get('date')}] {i.get('title')}" for i in items[:5]),
                    # 研报不提供可校验的数字：评级和盈利预测已在工具层剥离，
                    # 标题里的数字是机构测算值，不是已披露事实
                    numbers={},
                    ticker=ticker,
                    company=company,
                    tool_name=tool_name,
                    metadata={
                        "evidence_level": "institutional_view",
                        "report_count": len(items),
                    },
                )
            )

        themes = payload.get("themes") or []
        if themes:
            evidences.append(
                Evidence(
                    source_type="research_report",
                    source_name=f"{company} 卖方关注主题分布（{payload.get('report_count', 0)} 篇研报）",
                    disclosure_date=date.today(),
                    content="近期卖方研报的高频主题："
                    + "、".join(f"{t['theme']}({t['mentions']}篇)" for t in themes),
                    numbers={},
                    ticker=ticker,
                    company=company,
                    tool_name=tool_name,
                    metadata={"evidence_level": "institutional_view"},
                )
            )
        return evidences

    @staticmethod
    def _evidence_from_news(
        payload: dict[str, Any],
        company: str,
        ticker: str,
        source_type: str,
        source_name: str,
        tool_name: str,
    ) -> list[Evidence]:
        evidences: list[Evidence] = []
        for item in payload.get("news") or []:
            published = parse_report_date(item.get("published_at")) or date.today()
            evidences.append(
                Evidence(
                    source_type="news",
                    source_name=f"{item.get('source', '新闻')}：{item.get('title', '')[:40]}",
                    disclosure_date=published,
                    content=f"{item.get('title', '')}\n{item.get('summary', '')}",
                    # 新闻不提取数字：媒体转述的数字不可作为校验基准，
                    # 提取出来只会让门禁误判为"有来源支撑"
                    numbers={},
                    ticker=ticker,
                    company=company,
                    tool_name=tool_name,
                    metadata={"url": item.get("url") or "", "evidence_level": "unverified"},
                )
            )
        return evidences


__all__ = ["TOOL_SOURCE_TYPE", "FinancialDataMCPServer"]
