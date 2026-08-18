"""新闻搜索工具。

解决什么问题
    财报只能回答"发生了什么"，回答不了"为什么"。归因类问题（"毛利率为什么下降"）
    需要外部信息补充：原材料涨价、行业价格战、一次性减值。本工具提供这条信息通道。

核心设计决策
    1. 新闻在证据体系里的地位被**刻意压低**：source_type="news"，
       在 EvidenceGate 里即使数字对得上也只能标 ⚠️未验证，永远拿不到 ✅。
       原因很直接——媒体转述的数字经常有误，且新闻标题里的"预计""或将"
       是判断而非事实。让新闻只做"提供分析线索"，不做"证据支撑"。
    2. **AKShare 打底 + Tavily 按需补充**，而不是"Tavily 优先、失败降级"。
       两者是补充关系不是替代关系：
       - AKShare 东财个股新闻是按股票代码精确关联的，免费、无需 key、
         对 A 股标的的召回最直接，适合做默认数据源；
       - Tavily 是通用搜索，覆盖研报解读和行业报道，质量更好但按次计费，
         而且它靠关键词匹配公司，可能召回同名的无关内容。
       所以先拿免费且精确的那一路，只有条数不足（< news_min_results，默认 3）
       时才花钱调 Tavily 补齐，两路结果合并去重。
       这样做的收益：常态下零成本，窄时间窗或冷门标的才触发付费调用。
    3. 按天数过滤而不是按条数。归因分析关心的是"这段时间发生了什么"，
       返回 20 条三年前的新闻没有价值。日期解析失败的条目保留但排在最后——
       宁可多给一点噪音，也不要因为日期格式问题丢掉关键信息。
    4. 返回值里带 providers 和 notes，明确告诉调用方"数据从哪来、
       为什么只有这么几条"。新闻这种"可能合理地返回 0 条"的数据源，
       如果不区分"确实没新闻"和"接口挂了"，LLM 会把后者当成前者，
       进而得出"该公司近期无重大事件"的错误结论。

为什么不用其他方案
    - 不接付费财经数据商的新闻 API：成本和注册门槛都不适合简历项目。
    - 不做网页正文抓取：涉及反爬和解析维护，收益（几百字正文 vs 摘要）
      不足以支撑那部分复杂度。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import httpx

from config import get_config
from data.schemas import NewsItem
from tools.data_client import BaseTool, DataSourceError, normalize_ticker, parse_report_date

logger = logging.getLogger(__name__)

TAVILY_ENDPOINT = "https://api.tavily.com/search"

PROVIDER_AKSHARE = "akshare_eastmoney"
PROVIDER_TAVILY = "tavily"

# 无日期条目最多保留几条。这类条目无法参与时间窗过滤，
# 全留会让"最近 7 天"的查询混进几年前的内容。
MAX_UNDATED_ITEMS = 3


@dataclass
class NewsFetchResult:
    """新闻检索结果 + 过程元信息。

    把 items 和 providers/notes 一起返回，而不是只返回列表：
    调用方需要区分"确实没有新闻"和"数据源挂了"，两者对分析结论的
    影响完全不同（前者是事实，后者是数据缺口）。
    """

    items: list[NewsItem] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # 是否至少有一路数据源成功返回（哪怕返回 0 条）
    any_source_succeeded: bool = False


def fetch_stock_news(ticker: str, days: int = 7, query: str | None = None) -> list[NewsItem]:
    """获取个股新闻，返回条目列表。

    保持这个简单签名是为了兼容 AKShareDataClient.get_news；
    需要 providers/notes 的调用方用 fetch_stock_news_with_meta。
    """
    return fetch_stock_news_with_meta(ticker, days=days, query=query).items


def fetch_stock_news_with_meta(
    ticker: str, days: int = 7, query: str | None = None
) -> NewsFetchResult:
    """AKShare 打底，条数不足时用 Tavily 补充，两路合并去重。

    流程：
      1. 调 AKShare 东财个股新闻，按时间窗过滤；
      2. 若条数 < config.news_min_results 且配置了 TAVILY_API_KEY，
         调 Tavily 补充，与第一路结果合并去重；
      3. 两路都没成功（都抛异常）才抛 DataSourceError——
         "查到 0 条"是合法结果，不是错误。
    """
    config = get_config()
    days = max(1, days)
    cutoff = date.today() - timedelta(days=days)
    threshold = config.news_min_results

    result = NewsFetchResult()

    # ---- 第一路：AKShare（免费、按代码精确关联）----
    try:
        akshare_items = _filter_by_date(_fetch_from_akshare(ticker), cutoff)
        result.any_source_succeeded = True
        if akshare_items:
            result.items.extend(akshare_items)
            result.providers.append(PROVIDER_AKSHARE)
        else:
            result.notes.append(f"AKShare 东财新闻在最近 {days} 天内无匹配结果")
    except DataSourceError as exc:
        logger.warning("AKShare 新闻获取失败: %s", exc)
        result.notes.append(f"AKShare 新闻接口失败：{exc}")

    # ---- 第二路：条数不足时补 Tavily ----
    if len(result.items) < threshold:
        shortage_note = f"AKShare 仅得 {len(result.items)} 条（阈值 {threshold} 条）"
        if not config.tavily_api_key:
            result.notes.append(f"{shortage_note}，未配置 TAVILY_API_KEY，无法补充检索")
        else:
            try:
                tavily_items = _filter_by_date(
                    _fetch_from_tavily(
                        ticker, days=days, query=query, api_key=config.tavily_api_key
                    ),
                    cutoff,
                )
                result.any_source_succeeded = True
                added = _merge_dedup(result.items, tavily_items)
                if added:
                    result.providers.append(PROVIDER_TAVILY)
                    result.notes.append(f"{shortage_note}，已用 Tavily 补充 {added} 条")
                else:
                    result.notes.append(f"{shortage_note}，Tavily 补充检索未带来新增条目")
            except Exception as exc:  # noqa: BLE001 — 补充源失败不应让整个工具失败
                logger.warning("Tavily 补充检索失败: %s", exc)
                result.notes.append(f"{shortage_note}，Tavily 补充检索失败：{exc}")

    # 只有"所有尝试过的数据源都抛异常"才算工具失败。
    # AKShare 正常返回空列表属于合法结果（该股近期确实没新闻），
    # 把它当异常会让 LLM 收到误导性的错误信息。
    if not result.any_source_succeeded:
        raise DataSourceError("新闻数据源全部不可用：" + "；".join(result.notes))

    # 合并后按时间重排，保证两路结果交错时仍是倒序
    result.items.sort(key=lambda item: item.published_at or date.min, reverse=True)
    return result


def _filter_by_date(items: list[NewsItem], cutoff: date) -> list[NewsItem]:
    """按日期过滤。无日期的条目保留少量并排在最后。"""
    dated = [item for item in items if item.published_at and item.published_at >= cutoff]
    undated = [item for item in items if not item.published_at]
    dated.sort(key=lambda item: item.published_at or date.min, reverse=True)
    return dated + undated[:MAX_UNDATED_ITEMS]


def _merge_dedup(existing: list[NewsItem], incoming: list[NewsItem]) -> int:
    """把 incoming 合并进 existing（原地），返回实际新增条数。

    去重键优先用 URL，没有 URL 时退回归一化标题。同一条新闻被
    东财和 Tavily 同时收录是常态，标题往往只差一个来源前缀或标点，
    所以标题要去掉标点和空白再比。
    """
    seen = {key for item in existing for key in _dedup_keys(item)}
    added = 0
    for item in incoming:
        keys = _dedup_keys(item)
        if any(key in seen for key in keys):
            continue
        seen.update(keys)
        existing.append(item)
        added += 1
    return added


def _dedup_keys(item: NewsItem) -> list[str]:
    keys: list[str] = []
    if item.url:
        keys.append(f"url:{item.url.strip().rstrip('/')}")
    normalized_title = re.sub(r"[\s\W_]+", "", item.title).lower()
    if normalized_title:
        keys.append(f"title:{normalized_title}")
    return keys


def _fetch_from_tavily(
    ticker: str, days: int, query: str | None, api_key: str
) -> list[NewsItem]:
    search_query = query or f"{ticker} A股 财报 业绩 经营"
    payload = {
        "api_key": api_key,
        "query": search_query,
        "topic": "news",
        "days": max(1, min(days, 365)),
        "max_results": 10,
        "search_depth": "basic",
    }
    with httpx.Client(timeout=20.0) as client:
        response = client.post(TAVILY_ENDPOINT, json=payload)
        response.raise_for_status()
        body = response.json()

    items: list[NewsItem] = []
    for result in body.get("results", []):
        published = result.get("published_date")
        items.append(
            NewsItem(
                title=str(result.get("title") or "").strip() or "（无标题）",
                url=result.get("url"),
                published_at=_parse_iso_date(published),
                source="Tavily",
                summary=str(result.get("content") or "")[:500],
            )
        )
    return items


def _parse_iso_date(value: Any) -> date | None:
    if not value:
        return None
    text = str(value)
    for pattern in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[: len(pattern) + 2].strip(), pattern).date()
        except ValueError:
            continue
    return parse_report_date(text)


def _fetch_from_akshare(ticker: str) -> list[NewsItem]:
    """AKShare 东方财富个股新闻接口。"""
    try:
        import akshare as ak

        frame = ak.stock_news_em(symbol=normalize_ticker(ticker))
    except Exception as exc:  # noqa: BLE001
        raise DataSourceError(f"获取个股新闻失败: {type(exc).__name__}: {exc}") from exc

    if frame is None or frame.empty:
        return []

    items: list[NewsItem] = []
    for _, row in frame.head(20).iterrows():
        published = parse_report_date(row.get("发布时间"))
        items.append(
            NewsItem(
                title=str(row.get("新闻标题") or "").strip() or "（无标题）",
                url=str(row.get("新闻链接") or "") or None,
                published_at=published,
                source=str(row.get("文章来源") or "东方财富"),
                summary=str(row.get("新闻内容") or "")[:500],
            )
        )
    return items


class SearchNewsTool(BaseTool):
    """新闻搜索工具（ToolProtocol 实现）。"""

    name = "search_news"
    description = (
        "搜索指定 A 股公司最近 N 天的相关新闻和资讯，用于补充财报数据无法解释的背景信息"
        "（如原材料价格、行业竞争、政策变化、一次性事件）。"
        "重要：新闻属于非官方来源，其中的数字不能作为已验证事实引用，"
        "只能用于提供分析线索和假设方向。研报中引用新闻内容必须标注为「未验证」。"
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
                "description": "搜索最近多少天的新闻，默认 30，最大 365",
                "default": 30,
            },
            "query": {
                "type": "string",
                "description": (
                    "可选的补充检索关键词，例如「毛利率 原材料 价格」。"
                    "仅在东财新闻条数不足、触发 Tavily 补充检索时生效，用于聚焦特定议题"
                ),
            },
        },
        "required": ["ticker"],
    }

    def fetch(self, ticker: str, days: int = 30, query: str | None = None) -> dict[str, Any]:
        code = normalize_ticker(ticker)
        days = max(1, min(int(days), 365))
        outcome = fetch_stock_news_with_meta(code, days=days, query=query)

        return {
            "ticker": code,
            "days": days,
            "query": query,
            # 实际生效的数据源（可能是一路，也可能两路都用了）
            "providers": outcome.providers or ["none"],
            "news_count": len(outcome.items),
            "news": [item.to_display() for item in outcome.items[:10]],
            # 明确说明检索过程，避免 LLM 把"接口失败"当成"近期无新闻"
            "retrieval_notes": outcome.notes,
            "evidence_level": "unverified",
            "usage_note": (
                "新闻为非官方披露渠道，仅可作为分析线索。"
                "任何来自新闻的数字在研报中必须标注为「⚠️未验证」，"
                "不得与年报/季报数据同等对待。"
                + (
                    ""
                    if outcome.items
                    else "本次未检索到任何新闻，请勿据此推断「该公司近期无重大事件」，"
                    "应说明新闻信息缺失。"
                )
            ),
        }


__all__ = [
    "NewsFetchResult",
    "SearchNewsTool",
    "fetch_stock_news",
    "fetch_stock_news_with_meta",
]
