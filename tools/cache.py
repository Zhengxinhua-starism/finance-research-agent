"""工具结果缓存（content-hash 磁盘缓存 + 同步 Redis 热层）。

解决什么问题
    AKShare 是免费公开接口，调用不稳定且有隐性频率限制。开发和评测阶段
    同一个 (ticker, periods) 组合会被反复请求几十次：既慢（单次 2~5 秒），
    又容易触发对方限流导致整批评测失败。缓存把重复请求消掉。

核心设计决策
    1. 缓存键用 **content hash**（tool_name + 规范化参数的 SHA256），
       不用手工拼字符串。手工拼接的问题是参数顺序、类型（3 vs "3"）、
       默认值省略与否都会产生不同的键，缓存命中率会莫名其妙地低。
       规范化 JSON（sort_keys + 类型归一）保证语义相同的调用命中同一条目。
    2. 磁盘缓存写文件用"临时文件 + 原子替换"。直接写目标文件时，
       进程在写一半时被杀会留下半个 JSON，下次读取解析失败——
       而这种损坏是持久的，会一直失败到有人手动删文件。
    3. 提供同步 Redis 热层。
       # TODO: md 里 session/cache.py 的 ToolCache 定义为 async（redis.asyncio），
       # 但工具执行链路（ToolRegistry → ToolProtocol.run → AKShare）整条是同步的，
       # 在同步函数里没法 await。这里的处理是：tools 层用同步 redis 客户端做热层，
       # session/cache.py 保留 md 要求的 async 接口供 API/会话层使用，
       # 两者读写同一套 key 格式（tool:{tool_name}:{hash}），互相可见。
    4. Redis 不可用时静默降级到纯磁盘缓存。演示环境经常没有 Redis，
       让缓存层的可用性影响主流程的可用性是不可接受的。

为什么不用其他方案
    - 不用 functools.lru_cache：进程重启即失效，跨进程（API + 评测脚本）不共享，
      而且没有 TTL——财报数据会更新，永久缓存会读到过期数据。
    - 不用 diskcache 库：为一个 150 行能写完的功能引入依赖不划算，
      且它的 SQLite 后端在 Docker 卷上偶发锁问题。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from config import get_config

logger = logging.getLogger(__name__)

CACHE_KEY_PREFIX = "tool"

# 工具返回结构的版本号。**每次改动任何工具的返回字段，都必须 +1。**
#
# 为什么需要它（F-021）：缓存键原本只由「工具名 + 参数」决定，
# 而工具的返回**结构**不在键里。给 IncomeRow.to_display() 新增了
# 期间费用、利润构成等字段之后，参数没变，缓存键也没变，
# 于是磁盘缓存继续供应改动前的旧结构——新字段一个都到不了 LLM 面前。
# 表现为研报里写"缺乏期间费用明细"，而数据其实早就补齐了。
# 这类失效完全静默：不报错、不告警，只是功能没生效。
TOOL_RESULT_SCHEMA_VERSION = 3


def make_cache_key(tool_name: str, arguments: dict[str, Any]) -> str:
    """生成 content hash 缓存键：tool:{tool_name}:v{版本}:{sha256[:16]}。

    参数规范化规则：
      - 键排序，消除字典顺序差异；
      - None 值剔除，让"显式传 None"和"不传"等价；
      - 数字统一转 float 再格式化，消除 3 与 3.0 的差异。

    版本号进入键而不是进入值：进值的话要先读出来才能判断是否过期，
    等于每次都要读一遍旧数据；进键则旧条目直接不会被命中，
    并且会随 TTL 自然淘汰，不需要额外的清理逻辑。
    """
    normalized = _normalize_arguments(arguments)
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{CACHE_KEY_PREFIX}:{tool_name}:v{TOOL_RESULT_SCHEMA_VERSION}:{digest}"


def _normalize_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in sorted(arguments):
        value = arguments[key]
        if value is None:
            continue
        if isinstance(value, bool):
            normalized[key] = value
        elif isinstance(value, (int, float)):
            normalized[key] = f"{float(value):.6g}"
        elif isinstance(value, (list, tuple)):
            normalized[key] = [str(item) for item in value]
        else:
            normalized[key] = str(value)
    return normalized


class DiskCache:
    """content-hash 磁盘缓存。线程安全（用一把全局锁保护写入）。"""

    def __init__(self, cache_dir: str | Path | None = None, ttl_seconds: int | None = None):
        config = get_config()
        self.cache_dir = Path(cache_dir) if cache_dir else config.disk_cache_path
        self.ttl_seconds = (
            ttl_seconds if ttl_seconds is not None else config.disk_cache_ttl_seconds
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path_for(self, cache_key: str) -> Path:
        # 冒号在 Windows 文件名里非法，统一替换
        safe_name = cache_key.replace(":", "__") + ".json"
        return self.cache_dir / safe_name

    def get(self, cache_key: str) -> Any | None:
        path = self._path_for(cache_key)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("磁盘缓存损坏，已删除: %s (%s)", path.name, exc)
            path.unlink(missing_ok=True)
            return None

        stored_at = envelope.get("stored_at", 0)
        if self.ttl_seconds > 0 and time.time() - stored_at > self.ttl_seconds:
            logger.debug("磁盘缓存过期: %s", cache_key)
            path.unlink(missing_ok=True)
            return None
        return envelope.get("value")

    def set(self, cache_key: str, value: Any) -> None:
        envelope = {"stored_at": time.time(), "key": cache_key, "value": value}
        path = self._path_for(cache_key)
        try:
            serialized = json.dumps(envelope, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as exc:
            logger.warning("缓存值无法序列化，跳过写入 %s: %s", cache_key, exc)
            return

        with self._lock:
            try:
                # 原子写：先写临时文件再 replace，避免半个文件损坏缓存目录
                handle = tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    dir=self.cache_dir,
                    prefix=".tmp_",
                    delete=False,
                )
                try:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    handle.close()
                os.replace(handle.name, path)
            except OSError as exc:
                logger.warning("写入磁盘缓存失败 %s: %s", cache_key, exc)

    def clear(self) -> int:
        removed = 0
        for path in self.cache_dir.glob("tool__*.json"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    def stats(self) -> dict[str, Any]:
        files = list(self.cache_dir.glob("tool__*.json"))
        return {
            "entries": len(files),
            "size_bytes": sum(f.stat().st_size for f in files if f.exists()),
            "dir": str(self.cache_dir),
        }


class _SyncRedisLayer:
    """同步 Redis 热层。连接失败时整层禁用，不影响调用方。"""

    def __init__(self, redis_url: str, ttl_seconds: int, optional: bool = True):
        self.ttl_seconds = ttl_seconds
        self.enabled = False
        self._client: Any = None
        try:
            import redis as redis_sync

            self._client = redis_sync.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self._client.ping()
            self.enabled = True
            logger.info("工具缓存 Redis 热层已连接: %s", redis_url)
        except Exception as exc:  # noqa: BLE001 — 任何连接问题都降级
            if optional:
                logger.warning("Redis 不可用，工具缓存降级为纯磁盘模式: %s", exc)
            else:
                raise

    def get(self, cache_key: str) -> Any | None:
        if not self.enabled:
            return None
        try:
            raw = self._client.get(cache_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis 读取失败，本次跳过热层: %s", exc)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set(self, cache_key: str, value: Any) -> None:
        if not self.enabled:
            return
        try:
            self._client.setex(
                cache_key, self.ttl_seconds, json.dumps(value, ensure_ascii=False, default=str)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis 写入失败，仅保留磁盘缓存: %s", exc)


class ToolResultCache:
    """两级缓存：Redis 热层（10 分钟）→ 磁盘冷层（1 天）→ 回源。

    命中磁盘时会顺带回填 Redis，这样同一会话内的后续请求走更快的热层。
    """

    def __init__(
        self,
        disk_cache: DiskCache | None = None,
        redis_layer: _SyncRedisLayer | None = None,
    ):
        config = get_config()
        self.disk = disk_cache or DiskCache()
        self.redis = redis_layer or _SyncRedisLayer(
            redis_url=config.redis_url,
            ttl_seconds=config.cache_ttl_seconds,
            optional=config.redis_optional,
        )

    def get(self, tool_name: str, arguments: dict[str, Any]) -> tuple[Any | None, str | None]:
        """返回 (值, 命中层)。未命中返回 (None, None)。"""
        cache_key = make_cache_key(tool_name, arguments)

        value = self.redis.get(cache_key)
        if value is not None:
            logger.debug("缓存命中 [redis] %s", cache_key)
            return value, "redis"

        value = self.disk.get(cache_key)
        if value is not None:
            logger.debug("缓存命中 [disk] %s", cache_key)
            self.redis.set(cache_key, value)  # 回填热层
            return value, "disk"

        return None, None

    def set(self, tool_name: str, arguments: dict[str, Any], value: Any) -> None:
        cache_key = make_cache_key(tool_name, arguments)
        self.disk.set(cache_key, value)
        self.redis.set(cache_key, value)

    def stats(self) -> dict[str, Any]:
        return {"disk": self.disk.stats(), "redis_enabled": self.redis.enabled}


_SHARED_CACHE: list[ToolResultCache] = []


def get_tool_cache() -> ToolResultCache:
    """进程内共享的缓存实例。

    做成单例是因为 _SyncRedisLayer 会建连接池，每个工具各建一份
    会在 6 个工具 × N 个并发下把连接数打满。
    """
    if not _SHARED_CACHE:
        _SHARED_CACHE.append(ToolResultCache())
    return _SHARED_CACHE[0]
