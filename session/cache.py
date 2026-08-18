"""Redis 工具结果缓存（异步接口）。

解决什么问题
    与 tools/cache.py 的磁盘缓存互补，构成两级缓存：
    - Redis（本模块）：热数据，TTL 10 分钟，同一会话内的重复查询命中；
    - 磁盘（tools/cache.py）：冷数据，content-hash，跨会话跨进程持久化。
    查询顺序：Redis → 磁盘 → 调用 AKShare。

核心设计决策
    1. 本模块提供 **async** 接口（md 要求，且 FastAPI 路由是 async 的），
       而 tools/cache.py 里的 Redis 热层是 **sync** 的。两者不是重复实现：
       # TODO: md 把 ToolCache 定义为 async，但工具执行链路
       # （ToolRegistry → ToolProtocol.run → AKShare）整条是同步的，
       # 同步函数里无法 await。解决办法是两套客户端读写**同一套 key 格式**
       # （tool:{tool_name}:{content_hash}，由 tools.cache.make_cache_key 统一生成），
       # 缓存内容互相可见：API 层预热的缓存，工具层能读到，反之亦然。
       复用同一个 key 生成函数是这个设计成立的前提，不能各写各的。
    2. 所有方法在 Redis 不可用时静默降级为"未命中"，不抛异常。
       缓存是性能优化，不是功能依赖；让缓存故障演变成服务故障是错误的耦合。
    3. 连接用 redis.asyncio 的连接池，进程内共享。每次请求建连接会在
       并发下耗尽文件描述符。

为什么不用其他方案
    - 不用 fastapi-cache 之类的装饰器缓存库：本项目要缓存的是工具调用结果
      （在 Agent 内部，不在 HTTP 层），装饰器缓存的粒度对不上。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from config import get_config
from tools.cache import make_cache_key

logger = logging.getLogger(__name__)


class ToolCache:
    """异步 Redis 工具结果缓存。"""

    def __init__(self, redis: Any, ttl_seconds: int | None = None):
        self.redis = redis
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else get_config().cache_ttl_seconds

    @property
    def enabled(self) -> bool:
        return self.redis is not None

    async def get(self, tool_name: str, args_hash: str) -> str | None:
        """按工具名 + 参数哈希取缓存。args_hash 可以是完整 key 或纯哈希。"""
        if not self.enabled:
            return None
        key = self._normalize_key(tool_name, args_hash)
        try:
            return await self.redis.get(key)
        except Exception as exc:  # noqa: BLE001 — 缓存故障不能影响主流程
            logger.warning("Redis 读取失败 key=%s: %s", key, exc)
            return None

    async def set(self, tool_name: str, args_hash: str, result: str) -> None:
        if not self.enabled:
            return
        key = self._normalize_key(tool_name, args_hash)
        try:
            await self.redis.setex(key, self.ttl_seconds, result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis 写入失败 key=%s: %s", key, exc)

    async def get_json(self, tool_name: str, arguments: dict[str, Any]) -> Any | None:
        """按原始参数字典取缓存（与工具层的 key 生成规则完全一致）。"""
        if not self.enabled:
            return None
        key = make_cache_key(tool_name, arguments)
        try:
            raw = await self.redis.get(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis 读取失败 key=%s: %s", key, exc)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("缓存内容不是合法 JSON，已忽略: %s", key)
            return None

    async def set_json(self, tool_name: str, arguments: dict[str, Any], value: Any) -> None:
        if not self.enabled:
            return
        key = make_cache_key(tool_name, arguments)
        try:
            await self.redis.setex(
                key, self.ttl_seconds, json.dumps(value, ensure_ascii=False, default=str)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis 写入失败 key=%s: %s", key, exc)

    async def invalidate(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            await self.redis.delete(make_cache_key(tool_name, arguments))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis 删除失败: %s", exc)

    async def clear_tool(self, tool_name: str) -> int:
        """清空某个工具的全部缓存。用 scan_iter 而不是 keys：

        keys 在大 key 空间上是 O(N) 阻塞操作，生产环境会卡住整个 Redis。
        """
        if not self.enabled:
            return 0
        removed = 0
        try:
            async for key in self.redis.scan_iter(match=f"tool:{tool_name}:*", count=100):
                await self.redis.delete(key)
                removed += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("清理缓存失败: %s", exc)
        return removed

    async def stats(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        try:
            count = 0
            async for _ in self.redis.scan_iter(match="tool:*", count=100):
                count += 1
            return {"enabled": True, "cached_entries": count, "ttl_seconds": self.ttl_seconds}
        except Exception as exc:  # noqa: BLE001
            return {"enabled": True, "error": str(exc)}

    @staticmethod
    def _normalize_key(tool_name: str, args_hash: str) -> str:
        """兼容两种入参：完整 key（tool:xxx:vN:hash）或纯哈希。

        纯哈希形态要补上 schema 版本号，与 tools.cache.make_cache_key 保持一致——
        两层缓存共用同一套 key 格式是这个双客户端设计成立的前提，
        版本号只加一边会让两层各存各的，缓存命中率直接减半。
        """
        from tools.cache import TOOL_RESULT_SCHEMA_VERSION

        if args_hash.startswith("tool:"):
            return args_hash
        return f"tool:{tool_name}:v{TOOL_RESULT_SCHEMA_VERSION}:{args_hash}"


__all__ = ["ToolCache"]
