"""Agent 可调用的工具集合。

工具分两类：
- 数据获取类（akshare_tools）：从外部数据源拿原始财报；
- 计算分析类（compare_periods）：在本地做确定性计算。
两者都实现 ToolProtocol，通过 MCP Server 统一暴露给 Agent。
"""

from tools.akshare_tools import (
    AKShareDataClient,
    GetBalanceSheetTool,
    GetCashFlowTool,
    GetFinancialMetricsTool,
    GetIncomeHistoryTool,
    GetRiskSnapshotTool,
    GetStockPriceTool,
    build_akshare_tools,
    get_data_client,
)
from tools.cache import DiskCache, ToolResultCache, get_tool_cache, make_cache_key
from tools.compare_periods import ComparePeriodsTool
from tools.data_client import BaseTool, DataSourceError, FinancialDataClient, TickerNotFoundError
from tools.news_search import SearchNewsTool


def build_all_tools(client: AKShareDataClient | None = None) -> list[BaseTool]:
    """构造全部 7 个工具（6 个数据工具 + 1 个跨期对比工具）。"""
    shared = client or get_data_client()
    return [
        *build_akshare_tools(shared),
        SearchNewsTool(),
        ComparePeriodsTool(client=shared),
    ]


__all__ = [
    "AKShareDataClient",
    "BaseTool",
    "ComparePeriodsTool",
    "DataSourceError",
    "DiskCache",
    "FinancialDataClient",
    "GetBalanceSheetTool",
    "GetCashFlowTool",
    "GetFinancialMetricsTool",
    "GetIncomeHistoryTool",
    "GetRiskSnapshotTool",
    "GetStockPriceTool",
    "SearchNewsTool",
    "TickerNotFoundError",
    "ToolResultCache",
    "build_akshare_tools",
    "build_all_tools",
    "get_data_client",
    "get_tool_cache",
    "make_cache_key",
]
