"""分部构成与运营效率工具。

解决什么问题
    三大报表只能告诉你"整体毛利率降了 1.70pp"，回答不了"为什么降"。
    对多业务公司，整体毛利率的变动可能**完全来自业务占比变化**——
    高毛利业务收入占比下降，即使每个分部自身毛利率一点没动，
    整体也会被拉低。不做分部拆解就分不清这两种情况，
    而它们的投资含义完全相反（一个是结构调整，一个是经营恶化）。

    同样，杜邦分析的第二个因子是资产周转率，此前是用"营收/期末总资产"
    自算的粗略值（F-007），而现成的、按平均资产计算的周转率和周转天数
    一直躺在 AKShare 的财务分析指标接口里没被接入。

核心设计决策
    1. **结构效应分解在代码里做，不交给 LLM。**
       `decompose_margin_change()` 用标准的两因素分解，把整体毛利率变动
       拆成"结构效应"和"各分部自身效应"，并指出影响最大的分部。
       这是确定性计算，LLM 只负责解释业务含义。
    2. 分部工具默认取"按产品分类"，同时允许切到"按地区分类"。
       按地区分类能直接回答"海外业务盈利能力如何"——
       实测比亚迪境外毛利率 19.46% vs 境内 16.66%，这个差距
       在整体报表里完全看不出来。
    3. 毛利率优先自算（收入-成本）/收入，接口自带的值只做兜底。
       口径可控，且与报表层的计算方式一致，避免同一份研报里
       出现两种口径的毛利率。
    4. 效率工具同时返回周转率和周转天数。周转天数在解释应收风险时
       比周转率直观得多——"应收账款周转天数从 29 天拉长到 45 天"
       比"周转率从 12.5 降到 8.1"更容易让人理解。

为什么不用其他方案
    - 不把分部数据塞进 get_income_history：那个工具的语义是"利润表"，
      分部构成不是利润表科目。混在一起会让返回体积翻倍，
      而单点事实类问题根本不需要分部数据。
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from data.schemas import (
    SegmentRow,
    decompose_margin_change,
    format_amount,
    format_ratio,
    growth_rate,
)
from tools.akshare_tools import AKShareDataClient, get_data_client
from tools.data_client import BaseTool, DataSourceError, normalize_ticker

logger = logging.getLogger(__name__)

CATEGORY_CHOICES = ("按产品分类", "按行业分类", "按地区分类")


class GetSegmentBreakdownTool(BaseTool):
    """分部构成分析工具。"""

    name = "get_segment_breakdown"
    description = (
        "获取指定 A 股公司的主营业务构成（分产品/分行业/分地区的收入、成本、毛利率、收入占比），"
        "并对比最近两期，把整体毛利率的变动拆解为「结构效应」（业务占比变化）"
        "和「各分部自身效应」（各业务毛利率变化），指出影响最大的分部。"
        "这是回答「毛利率为什么下降」「哪块业务在拖累盈利」「海外业务比国内赚钱吗」"
        "这类归因问题的关键数据——三大报表里没有这些信息。"
        "按地区分类可直接看出境内外盈利能力差异。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "A股股票代码，6位数字，例如 002594",
            },
            "category": {
                "type": "string",
                "description": (
                    "分类维度。按产品分类=看业务线，按行业分类=看行业归属，"
                    "按地区分类=看境内外。默认按产品分类"
                ),
                "enum": list(CATEGORY_CHOICES),
                "default": "按产品分类",
            },
            "periods": {
                "type": "integer",
                "description": "对比的年报期数，默认 2（当期 + 上期）；最大 4",
                "default": 2,
            },
        },
        "required": ["ticker"],
    }

    def __init__(self, client: AKShareDataClient | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.client = client or get_data_client()

    def fetch(
        self, ticker: str, category: str = "按产品分类", periods: int = 2
    ) -> dict[str, Any]:
        code = normalize_ticker(ticker)
        periods = max(2, min(int(periods), 4))
        if category not in CATEGORY_CHOICES:
            category = "按产品分类"

        rows = self.client.get_segment_rows(code, category=category)
        # 只取年报期。分部数据同时包含中报，混用会让"同比"变成"半年比"
        annual = [row for row in rows if row.period_end.month == 12]
        if not annual:
            logger.warning("%s 无年报分部数据，退回使用全部报告期", code)
            annual = rows

        period_ends = sorted({row.period_end for row in annual}, reverse=True)[:periods]
        if not period_ends:
            raise DataSourceError(f"{code} 未解析出任何分部数据")

        by_period: dict[date, list[SegmentRow]] = {
            period: [row for row in annual if row.period_end == period]
            for period in period_ends
        }

        latest = period_ends[0]
        previous = period_ends[1] if len(period_ends) > 1 else None

        effect = None
        if previous is not None:
            effect = decompose_margin_change(by_period[previous], by_period[latest])

        return {
            "company": self.client.get_company_name(code),
            "ticker": code,
            "category": category,
            "periods": [p.isoformat() for p in sorted(period_ends)],
            "segments_by_period": {
                period.isoformat(): [
                    row.to_display()
                    for row in sorted(
                        by_period[period], key=lambda r: r.revenue or 0, reverse=True
                    )
                ]
                for period in sorted(period_ends, reverse=True)
            },
            "segment_changes": self._build_changes(
                by_period.get(previous, []) if previous else [], by_period[latest]
            ),
            "margin_decomposition": effect.model_dump() if effect else None,
            "decomposition_note": (
                "整体毛利率变动 = 结构效应 + 各分部自身效应。"
                "结构效应为负说明高毛利业务的收入占比下降；"
                "自身效应为负说明各业务自身的盈利能力在恶化。"
                "两者的投资含义完全不同，不可混为一谈。"
            ),
            "source": "AKShare 东财主营构成（stock_zygc_em）· 年报口径",
            "unit_note": "金额单位为元；占比与毛利率为小数；变动以百分点(pp)表示",
        }

    @staticmethod
    def _build_changes(
        base: list[SegmentRow], current: list[SegmentRow]
    ) -> list[dict[str, Any]]:
        """逐分部列出收入增速、占比变化、毛利率变化。"""
        base_map = {row.segment_name: row for row in base}
        changes: list[dict[str, Any]] = []
        for row in sorted(current, key=lambda r: r.revenue or 0, reverse=True):
            previous = base_map.get(row.segment_name)
            changes.append(
                {
                    "segment": row.segment_name,
                    "revenue": row.revenue,
                    "revenue_display": format_amount(row.revenue),
                    "revenue_yoy": growth_rate(
                        row.revenue, previous.revenue if previous else None
                    ),
                    "revenue_share": row.revenue_share,
                    "revenue_share_display": format_ratio(row.revenue_share),
                    "share_change_pp": (
                        (row.revenue_share - previous.revenue_share) * 100
                        if previous
                        and row.revenue_share is not None
                        and previous.revenue_share is not None
                        else None
                    ),
                    "gross_margin": row.gross_margin,
                    "gross_margin_display": format_ratio(row.gross_margin),
                    "margin_change_pp": (
                        (row.gross_margin - previous.gross_margin) * 100
                        if previous
                        and row.gross_margin is not None
                        and previous.gross_margin is not None
                        else None
                    ),
                }
            )
        return changes


class GetOperatingEfficiencyTool(BaseTool):
    """运营效率指标工具（杜邦第二因子）。"""

    name = "get_operating_efficiency"
    description = (
        "获取指定 A 股公司的运营效率指标：应收账款周转率与周转天数、存货周转率与周转天数、"
        "固定资产周转率、总资产周转率、三项费用比重。"
        "适用于杜邦分析中「资产周转率」因子的分析，以及"
        "「应收账款回款是否变慢」「存货是否积压」这类运营质量问题。"
        "周转天数比周转率更直观：天数拉长意味着资金占用时间变久。"
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
                "description": "对比的年报期数，默认 3，最大 5",
                "default": 3,
            },
        },
        "required": ["ticker"],
    }

    def __init__(self, client: AKShareDataClient | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.client = client or get_data_client()

    LABELS: dict[str, str] = {
        "receivable_turnover": "应收账款周转率(次)",
        "receivable_days": "应收账款周转天数(天)",
        "inventory_turnover": "存货周转率(次)",
        "inventory_days": "存货周转天数(天)",
        "fixed_asset_turnover": "固定资产周转率(次)",
        "total_asset_turnover": "总资产周转率(次)",
        "total_asset_days": "总资产周转天数(天)",
        "current_asset_turnover": "流动资产周转率(次)",
        "equity_turnover": "股东权益周转率(次)",
        "three_expense_ratio": "三项费用比重",
        "cost_profit_ratio": "成本费用利润率",
    }

    def fetch(self, ticker: str, periods: int = 3) -> dict[str, Any]:
        code = normalize_ticker(ticker)
        periods = max(2, min(int(periods), 5))
        start_year = str(date.today().year - periods - 1)

        indicators = self.client.get_efficiency_indicators(code, start_year=start_year)
        if not indicators:
            raise DataSourceError(f"{code} 未获取到运营效率指标")

        # 只取年报期，理由同分部工具：混入季报会让同比变环比
        annual = {p: m for p, m in indicators.items() if p.month == 12}
        if not annual:
            annual = indicators
        period_ends = sorted(annual, reverse=True)[:periods]
        ordered = sorted(period_ends)

        series: list[dict[str, Any]] = []
        for field, label in self.LABELS.items():
            values = [annual[p].get(field) for p in ordered]
            if all(value is None for value in values):
                continue
            latest, previous = values[-1], values[-2] if len(values) > 1 else None
            series.append(
                {
                    "metric": field,
                    "label": label,
                    "values": values,
                    "latest": latest,
                    "change": (latest - previous)
                    if latest is not None and previous is not None
                    else None,
                    "change_pct": growth_rate(latest, previous),
                }
            )

        return {
            "company": self.client.get_company_name(code),
            "ticker": code,
            "periods": [p.isoformat() for p in ordered],
            "indicators": series,
            "interpretation": self._interpret(series),
            "source": "AKShare 财务分析指标（stock_financial_analysis_indicator）· 年报口径",
            "usage_note": (
                "周转率单位为「次」，周转天数单位为「天」，均非比率，不要当百分数解读。"
                "周转天数拉长 = 资金占用时间变久 = 运营效率下降。"
            ),
        }

    @staticmethod
    def _interpret(series: list[dict[str, Any]]) -> list[str]:
        """把关键变化翻译成一句话，减少 LLM 自行判断方向出错的机会。"""
        notes: list[str] = []
        for item in series:
            change = item.get("change")
            latest = item.get("latest")
            if change is None or latest is None:
                continue
            if item["metric"].endswith("_days") and abs(change) >= 3:
                direction = "拉长" if change > 0 else "缩短"
                notes.append(
                    f"{item['label']} {direction} {abs(change):.1f} 天至 {latest:.1f} 天"
                    + ("，资金占用时间变久" if change > 0 else "，回款/周转提速")
                )
            elif item["metric"].endswith("_turnover") and abs(change) >= 0.1:
                direction = "上升" if change > 0 else "下降"
                notes.append(f"{item['label']} {direction} {abs(change):.2f} 至 {latest:.2f}")
        return notes


__all__ = ["GetOperatingEfficiencyTool", "GetSegmentBreakdownTool"]
