"""BrokerToolRegistry 必须走父类并发，而不是串行列表推导。"""

from __future__ import annotations

import time

from harness.types import ToolCall, ToolResult
from mcp_servers.base_server import BrokerToolRegistry


class _SlowBroker:
    def execute(self, call: ToolCall) -> ToolResult:
        time.sleep(0.25)
        return ToolResult.ok(call.id, call.name, data={}, content="ok")


def test_broker_registry_executes_tools_concurrently() -> None:
    registry = BrokerToolRegistry(broker=_SlowBroker())  # type: ignore[arg-type]
    calls = [ToolCall(name="sleep", arguments={"i": i}) for i in range(3)]
    started = time.perf_counter()
    results = registry.execute(calls)
    elapsed = time.perf_counter() - started
    assert all(item.success for item in results)
    assert elapsed < 0.55, f"并发应接近 0.25s，串行会到 0.75s，实际 {elapsed:.2f}s"
