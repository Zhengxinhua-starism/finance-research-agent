"""公告与机构覆盖工具（第一层与第二层来源）。

解决什么问题
    此前证据池只有两类东西：报表数字（第一层）和新闻（第三层）。
    中间是空的，而且第一层里**只有数字，没有文字**。

    这造成两个具体缺口：
    1. 知识库里明确写了归因排查要"检查是否有一次性减值、会计政策变更、
       合并范围变化"——但这些信息不在三大报表的数字里，在公告里。
       没有公告数据源，这条排查项永远无法执行。
    2. `news` 这一个类型同时装着券商研报和公众号传闻，两者都是
       priority 10、都最高只能拿 ⚠️未验证。这既低估了前者，也高估了后者。

核心设计决策
    1. **研报必须剥离目标价与评级。** `stock_research_report_em` 返回
       「东财评级」（买入/增持）和三年期的盈利预测（EPS/PE）。这两类内容
       如果进了证据池，Writer 完全可能引用它们——等于从后门绕过
       "本系统不提供投资建议"这条底线（F-017 刚建立的护栏）。
       所以在解析阶段就丢弃，只保留机构名、报告标题、日期。
       保留标题是因为它本身承载了分析判断（"销量环比改善，出口再创新高"），
       这才是第二层来源的价值所在。
    2. 研报归入新增的 `research_report` 类型（priority 40，第二层），
       `is_official=False`——**它最高只能拿 ⚠️未验证**。
       机构判断再权威也是判断不是事实，标成"已验证"等于把推测
       包装成客观依据。
    3. 公告工具**不提取数字**。接口只返回标题和链接，不含正文；
       从标题里正则抠数字是典型的过度解读。公告的价值在于
       "发生过什么事件"，而不是"数字是多少"。
    4. 业绩预告单独处理，因为它带「业绩变动原因」——**公司自己写的
       业绩变动解释**，是归因分析能拿到的最高质量的定性材料。
       注意它的覆盖率只有约 60%（3072/5000 家）：只有亏损、扭亏或
       变动超 50% 才强制预告，经营平稳的公司没有。拿不到不算异常。

为什么不接这些
    - 交易所互动平台：大量提问是"股价什么时候涨"，信噪比过低；
    - 盈利预测接口：是对未来的预测，与"整理已披露信息"的定位直接冲突；
    - 机构调研明细：AKShare 只给"哪些机构来调研了"的统计，没有纪要内容。
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any

from data.schemas import format_ratio
from tools.akshare_tools import AKShareDataClient, _call_with_timeout, get_data_client
from tools.data_client import BaseTool, DataSourceError, normalize_ticker, parse_report_date

logger = logging.getLogger(__name__)

# 公告类型 → 是否对基本面分析有价值。
# 「调研活动」在比亚迪的 3296 条公告里占 929 条，但内容是投资者关系活动记录表，
# 只有标题没有正文，对归因没有帮助，因此不列为重点。
IMPORTANT_NOTICE_TYPES = (
    "业绩预告",
    "业绩快报",
    "月度经营情况",  # 产销快报，销量数据的官方来源
    "定期报告",
    "重大事项",
    "资产重组",
    "对外投资",
    "会计政策",
    "资产减值",
    "股权激励",
    "分配方案",
    "增发",
    "回购",
)

# 从公告标题里识别归因相关的事件。命中即在返回里高亮，
# 因为这些正是"财报数字为什么异动"的可能解释。
ATTRIBUTION_KEYWORDS = (
    "减值",
    "会计政策",
    "会计估计",
    "重组",
    "并购",
    "处置",
    "计提",
    "补贴",
    "诉讼",
    "产销快报",
    "经营情况",
    "业绩预告",
    "业绩快报",
)


class GetDisclosureEventsTool(BaseTool):
    """公司公告与业绩预告（第一层：已披露事实）。"""

    name = "get_disclosure_events"
    description = (
        "获取指定 A 股公司最近 N 天的公司公告清单与业绩预告。"
        "公告是上市公司依法披露的正式文件，属于最高可信度来源。"
        "适用于回答「财报数字异动的原因是什么」——一次性减值、会计政策变更、"
        "资产重组、并购处置这些信息不在三大报表的数字里，只在公告里。"
        "业绩预告还包含公司自己撰写的「业绩变动原因」，是归因分析质量最高的定性材料。"
        "注意：本工具只返回公告标题与类型，不含公告正文；"
        "业绩预告仅在亏损、扭亏或业绩变动超 50% 时才强制披露，多数公司没有。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "A股股票代码，6位数字，例如 002594",
            },
            "days": {
                "type": "integer",
                "description": "回溯天数，默认 180（覆盖一个报告期），最大 730",
                "default": 180,
            },
            "forecast_period": {
                "type": "string",
                "description": (
                    "业绩预告的报告期，格式 YYYYMMDD，如 20251231。"
                    "留空则不查业绩预告（查询较慢，需拉全市场数据再筛选）"
                ),
            },
        },
        "required": ["ticker"],
    }

    def __init__(self, client: AKShareDataClient | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.client = client or get_data_client()

    def fetch(
        self, ticker: str, days: int = 180, forecast_period: str | None = None
    ) -> dict[str, Any]:
        code = normalize_ticker(ticker)
        days = max(7, min(int(days), 730))
        cutoff = date.today() - timedelta(days=days)

        notices = self._fetch_notices(code, cutoff)
        forecast = self._fetch_forecast(code, forecast_period) if forecast_period else None

        attribution_relevant = [
            item
            for item in notices
            if any(keyword in item["title"] for keyword in ATTRIBUTION_KEYWORDS)
        ]
        by_type: dict[str, int] = {}
        for item in notices:
            by_type[item["type"]] = by_type.get(item["type"], 0) + 1

        return {
            "company": self.client.get_company_name(code),
            "ticker": code,
            "days": days,
            "notice_count": len(notices),
            "notices_by_type": by_type,
            # 与归因相关的公告单独列出，这是本工具的核心产出
            "attribution_relevant_notices": attribution_relevant[:20],
            "recent_notices": notices[:25],
            "earnings_forecast": forecast,
            "source_tier": "① 已披露事实",
            "usage_note": (
                "公告属于第一层来源（上市公司依法披露），可作为事实引用。"
                "但本工具只返回标题，不含正文——只能据此判断「发生过某类事件」，"
                "不能推断事件的具体金额或影响程度。"
                "如需具体数字，须以财务报表数据为准。"
            ),
        }

    def _fetch_notices(self, code: str, cutoff: date) -> list[dict[str, Any]]:
        import akshare as ak

        try:
            frame = _call_with_timeout(ak.stock_individual_notice_report, security=code)
        except DataSourceError as exc:
            logger.warning("获取 %s 公告失败: %s", code, exc)
            raise

        if frame is None or frame.empty:
            return []

        items: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            published = parse_report_date(row.get("公告日期"))
            if published is None or published < cutoff:
                continue
            items.append(
                {
                    "title": str(row.get("公告标题", "")).strip(),
                    "type": str(row.get("公告类型", "")).strip() or "其他",
                    "date": published.isoformat(),
                    "url": str(row.get("网址", "")) or None,
                    "important": str(row.get("公告类型", "")).strip()
                    in IMPORTANT_NOTICE_TYPES,
                }
            )
        items.sort(key=lambda item: item["date"], reverse=True)
        return items

    def _fetch_forecast(self, code: str, period: str) -> dict[str, Any] | None:
        """业绩预告。拉全市场再筛选——接口不支持按标的查询。"""
        import akshare as ak

        digits = re.sub(r"\D", "", str(period))
        if len(digits) != 8:
            logger.warning("forecast_period 格式非法，跳过业绩预告查询: %s", period)
            return None

        try:
            frame = _call_with_timeout(ak.stock_yjyg_em, date=digits, timeout=60)
        except DataSourceError as exc:
            logger.warning("获取业绩预告失败: %s", exc)
            return None
        if frame is None or frame.empty:
            return None

        matched = frame[frame["股票代码"].astype(str).str.zfill(6) == code]
        if matched.empty:
            return {
                "period": digits,
                "exists": False,
                # 说清楚"没有"意味着什么，避免 LLM 把它读成"公司没披露业绩"
                "note": (
                    f"该公司在 {digits} 报告期没有业绩预告。这是正常情况——"
                    "只有亏损、扭亏或业绩变动超过 50% 时才强制披露业绩预告，"
                    "经营平稳的公司通常没有。不代表业绩存在问题或信息缺失。"
                ),
            }

        row = matched.iloc[0]
        return {
            "period": digits,
            "exists": True,
            "forecast_type": str(row.get("预告类型", "")),
            "indicator": str(row.get("预测指标", "")),
            "change_desc": str(row.get("业绩变动", "")),
            "change_pct": str(row.get("业绩变动幅度", "")),
            # 这是全工具最有价值的字段：公司自己写的业绩变动解释
            "change_reason": str(row.get("业绩变动原因", "")),
            "announce_date": str(row.get("公告日期", "")),
        }


class GetAnalystCoverageTool(BaseTool):
    """券商研报覆盖（第二层：机构判断）。"""

    name = "get_analyst_coverage"
    description = (
        "获取指定 A 股公司近期的券商研报覆盖情况：哪些机构在跟踪、发了什么主题的报告。"
        "属于第二层来源「机构判断」——是有署名、有方法论的专业观点，"
        "可信度高于新闻，但**是判断不是事实**，引用时必须注明「某机构认为」，"
        "最高只能标注为「⚠️未验证」。"
        "适用于了解市场对该公司的关注焦点、以及财报数据之外的行业视角。"
        "注意：本工具已剥离研报中的投资评级与目标价，避免把券商观点写成已披露事实。"
        "本系统会基于已核查的财报与行情独立给出倾向判断。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "A股股票代码，6位数字，例如 002594",
            },
            "days": {
                "type": "integer",
                "description": "回溯天数，默认 90，最大 365",
                "default": 90,
            },
        },
        "required": ["ticker"],
    }

    def __init__(self, client: AKShareDataClient | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.client = client or get_data_client()

    def fetch(self, ticker: str, days: int = 90) -> dict[str, Any]:
        import akshare as ak

        code = normalize_ticker(ticker)
        days = max(7, min(int(days), 365))
        cutoff = date.today() - timedelta(days=days)

        frame = _call_with_timeout(ak.stock_research_report_em, symbol=code)
        if frame is None or frame.empty:
            raise DataSourceError(f"{code} 未返回研报数据")

        reports: list[dict[str, Any]] = []
        institutions: dict[str, int] = {}
        for _, row in frame.iterrows():
            published = parse_report_date(row.get("日期"))
            if published is None or published < cutoff:
                continue
            institution = str(row.get("机构", "")).strip() or "未知机构"
            institutions[institution] = institutions.get(institution, 0) + 1
            reports.append(
                {
                    "title": str(row.get("报告名称", "")).strip(),
                    "institution": institution,
                    "date": published.isoformat(),
                    "industry": str(row.get("行业", "")).strip(),
                }
            )
            # 刻意不收录：「东财评级」（买入/增持）与「盈利预测-收益/市盈率」。
            # 这两类内容属于投资建议与未来预测，与本系统的定位冲突；
            # 一旦进入证据池，Writer 有可能引用它们，等于绕过
            # "不提供投资建议" 的护栏（F-017）。

        reports.sort(key=lambda item: item["date"], reverse=True)
        top_institutions = sorted(institutions.items(), key=lambda kv: kv[1], reverse=True)

        return {
            "company": self.client.get_company_name(code),
            "ticker": code,
            "days": days,
            "report_count": len(reports),
            "institution_count": len(institutions),
            "top_institutions": [
                {"institution": name, "reports": count} for name, count in top_institutions[:10]
            ],
            "recent_reports": reports[:20],
            "themes": self._extract_themes(reports),
            "source_tier": "② 机构判断",
            "excluded_fields": ["投资评级", "目标价", "盈利预测(EPS/PE)"],
            "usage_note": (
                "以上为券商研报的**标题与机构信息**，属于第二层来源（机构判断）。"
                "引用时必须写成「某机构在报告中提到…」，不得作为事实陈述，"
                "最高标注为「⚠️未验证」。"
                "本工具已剥离研报中的投资评级、目标价与盈利预测，"
                "倾向判断由本系统基于已核查财报与行情独立给出，不转述券商目标价。"
                "研报覆盖数量本身只反映市场关注度，不代表公司质地。"
            ),
        }

    @staticmethod
    def _extract_themes(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """从研报标题里统计高频主题词。

        用固定词表而不是分词+TF-IDF：研报标题短且句式固定，
        词表方式结果可解释、可审计，不会因为语料变化产生漂移。
        """
        vocabulary = (
            "销量", "出口", "海外", "毛利率", "盈利", "新品", "产能", "订单",
            "价格", "降本", "智驾", "电池", "储能", "份额", "渠道", "业绩",
            "点评", "深度", "季报", "年报", "中报",
        )
        counts: dict[str, int] = {}
        for report in reports:
            for word in vocabulary:
                if word in report["title"]:
                    counts[word] = counts.get(word, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        total = len(reports) or 1
        return [
            {"theme": word, "mentions": count, "share": format_ratio(count / total)}
            for word, count in ranked
        ]


__all__ = ["GetAnalystCoverageTool", "GetDisclosureEventsTool"]
