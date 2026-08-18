"""MCP 工具协议层。

工具调用链路：Agent → MCPToolBroker → MCPServer → ToolRegistry → AKShare/Chroma
新增数据源只需实现一个新的 MCPServer 并加进 Broker，Agent 代码不动。
"""

from __future__ import annotations

from harness.tracing import Tracer
from mcp_servers.base_server import BrokerToolRegistry, MCPServer, MCPToolBroker
from mcp_servers.financial_data_server import FinancialDataMCPServer
from mcp_servers.knowledge_server import KnowledgeMCPServer, SearchKnowledgeTool


def build_mcp_servers(tracer: Tracer | None = None) -> list[MCPServer]:
    """构造项目的全部 MCP Server。"""
    return [
        FinancialDataMCPServer(tracer=tracer),
        KnowledgeMCPServer(tracer=tracer),
    ]


def build_broker(tracer: Tracer | None = None) -> MCPToolBroker:
    return MCPToolBroker(build_mcp_servers(tracer=tracer))


__all__ = [
    "BrokerToolRegistry",
    "FinancialDataMCPServer",
    "KnowledgeMCPServer",
    "MCPServer",
    "MCPToolBroker",
    "SearchKnowledgeTool",
    "build_broker",
    "build_mcp_servers",
]
