"""Redis 会话状态管理。

解决什么问题
    一次完整研究要跑约 30~90 秒（四个节点均走 chat，门禁用代码校验）。
    HTTP 请求同步等这么久会超时，而且用户完全看不到进度。
    需要一个地方存"这次研究跑到哪一步了、结果是什么"，让客户端能轮询。

核心设计决策
    1. 用 Redis 而不是进程内字典或本地文件：
       - 进程内字典在多 worker（uvicorn --workers 4）下会出现
         "请求打到 worker A 创建会话，轮询打到 worker B 查不到"的问题；
       - 本地文件要自己处理并发写和过期清理；
       - Redis 的 TTL 自动过期正好匹配"会话是临时数据"的语义。
    2. 会话数据整体存一个 JSON string，而不是 Redis Hash。
       Hash 的优势是能局部更新字段，但本项目的更新都是"读-改-写"整体状态，
       用 Hash 反而要处理字段类型（Redis Hash 的值全是字符串，
       嵌套的 report dict 还得再序列化一次）。
    3. **每次 update 都刷新 TTL**。研究过程中会多次更新状态，
       如果只在创建时设 TTL，长任务可能在跑到一半时会话就过期了。
    4. Redis 不可用时降级为**进程内字典**，而不是直接失败。
       演示环境经常没有 Redis，单进程下内存降级完全够用；
       同时在 health 里明确暴露 "degraded" 状态，不隐瞒这个事实。

为什么不用其他方案
    - 不用 LangGraph checkpointer：它存的是图的内部状态（含 Pydantic 对象），
      序列化格式与代码版本强绑定，升级代码后旧会话读不出来。
      会话存储该存的是"面向用户的结果"，不是"框架的内部状态"。
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any

from config import get_config

logger = logging.getLogger(__name__)

SESSION_KEY_PREFIX = "session"

# 会话生命周期状态
SESSION_STATUS_PENDING = "pending"
SESSION_STATUS_RUNNING = "running"
SESSION_STATUS_COMPLETED = "completed"
SESSION_STATUS_FAILED = "failed"

# 线性四段只用于兜底。真实进度按「规划 → 检索/核查循环 → 撰写」单调递增，
# 不能按节点名回退：图会在 verifier 之后回到 retriever 做补搜。
_NODE_PROGRESS = {
    "planner": 0.15,
    "retriever": 0.32,
    "verifier": 0.48,
    "writer": 0.90,
}
# retriever 入口 retrieval_count=0/1/2；verifier 入口已含本轮，为 1/2/3。
_RETRIEVER_PROGRESS = (0.32, 0.58, 0.74)
_VERIFIER_PROGRESS = (0.48, 0.66, 0.82)


def describe_node_progress(node: str, retrieval_count: int = 0) -> tuple[float, str]:
    """把当前节点映射为进度条比例和中文说明。

    进度按执行路径单调递增。补搜回到 retriever 时，比例必须高于上一轮核查，
    否则线性四段 UI 会看起来像进度条倒退。
    """
    if not node:
        return 0.05, "已提交，等待规划"
    if node == "planner":
        return 0.15, "正在规划研究任务"
    if node == "retriever":
        round_no = max(int(retrieval_count), 0) + 1
        index = min(max(round_no - 1, 0), len(_RETRIEVER_PROGRESS) - 1)
        ratio = _RETRIEVER_PROGRESS[index]
        if round_no <= 1:
            return ratio, "正在检索财务数据与新闻（第 1 轮）"
        return ratio, f"证据不足，正在补搜（第 {round_no} 轮）"
    if node == "verifier":
        round_no = max(int(retrieval_count), 1)
        index = min(max(round_no - 1, 0), len(_VERIFIER_PROGRESS) - 1)
        ratio = _VERIFIER_PROGRESS[index]
        if round_no <= 1:
            return ratio, "正在核查证据"
        return ratio, f"正在核查证据（第 {round_no} 轮）"
    if node == "writer":
        return 0.90, "正在撰写研报"
    return _NODE_PROGRESS.get(node, 0.5), f"正在执行：{node}"


class SessionStore:
    """会话状态读写。redis=None 时自动降级为进程内存储。"""

    # 内存降级模式的共享存储。类属性而非实例属性：
    # FastAPI 每个请求会新建一个 SessionStore（通过 Depends），
    # 实例属性会导致每个请求都拿到空字典。
    _memory_store: dict[str, tuple[float, dict[str, Any]]] = {}

    def __init__(self, redis: Any = None, ttl_seconds: int | None = None):
        self.redis = redis
        self.ttl_seconds = (
            ttl_seconds if ttl_seconds is not None else get_config().session_ttl_seconds
        )

    @property
    def degraded(self) -> bool:
        return self.redis is None

    # ---------------- 写 ----------------

    async def create(
        self,
        session_id: str,
        question: str,
        company: str = "",
        ticker: str = "",
        as_of_date: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": session_id,
            "question": question,
            "company": company,
            "ticker": ticker,
            "as_of_date": as_of_date,
            "status": SESSION_STATUS_PENDING,
            "current_node": "",
            "node_path": [],
            "progress": 0.0,
            "progress_label": "等待开始",
            "created_at": datetime.now().astimezone().isoformat(),
            "updated_at": datetime.now().astimezone().isoformat(),
            "run_id": None,
            "result": None,
            "trace_summary": None,
            "error": None,
        }
        await self._write(session_id, payload)
        logger.info("会话已创建: %s", session_id)
        return payload

    async def update_status(
        self,
        session_id: str,
        status: str,
        current_node: str | None = None,
        error: str | None = None,
        progress: float | None = None,
        progress_label: str | None = None,
        retrieval_count: int | None = None,
    ) -> None:
        payload = await self.get(session_id)
        if payload is None:
            logger.warning("更新不存在的会话: %s", session_id)
            return
        payload["status"] = status
        if current_node is not None:
            payload["current_node"] = current_node
            node_path = payload.get("node_path") or []
            # 只在节点真正变化时追加，避免轮询更新把路径写满重复项
            if not node_path or node_path[-1] != current_node:
                node_path.append(current_node)
            payload["node_path"] = node_path
            if progress is None or progress_label is None:
                rounds = (
                    retrieval_count
                    if retrieval_count is not None
                    else int(payload.get("retrieval_count") or 0)
                )
                auto_progress, auto_label = describe_node_progress(current_node, rounds)
                if progress is None:
                    progress = auto_progress
                if progress_label is None:
                    progress_label = auto_label
        if progress is not None:
            previous = float(payload.get("progress") or 0.0)
            payload["progress"] = max(previous, float(progress))
        if progress_label is not None:
            payload["progress_label"] = progress_label
        if retrieval_count is not None:
            payload["retrieval_count"] = retrieval_count
        if error is not None:
            payload["error"] = error
        payload["updated_at"] = datetime.now().astimezone().isoformat()
        await self._write(session_id, payload)

    async def set_result(
        self,
        session_id: str,
        report: dict[str, Any],
        trace_summary: dict[str, Any],
        run_id: str | None = None,
    ) -> None:
        payload = await self.get(session_id)
        if payload is None:
            logger.warning("为不存在的会话写入结果: %s", session_id)
            return
        payload.update(
            {
                "status": SESSION_STATUS_COMPLETED,
                "result": report,
                "trace_summary": trace_summary,
                "run_id": run_id or payload.get("run_id"),
                "current_node": "writer",
                "progress": 1.0,
                "progress_label": "研究完成",
                "updated_at": datetime.now().astimezone().isoformat(),
            }
        )
        await self._write(session_id, payload)
        logger.info("会话结果已写入: %s", session_id)

    async def set_failed(self, session_id: str, error: str) -> None:
        await self.update_status(session_id, SESSION_STATUS_FAILED, error=error)

    async def delete(self, session_id: str) -> None:
        key = self._key(session_id)
        if self.redis is not None:
            try:
                await self.redis.delete(key)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("删除会话失败，回退内存存储: %s", exc)
        self._memory_store.pop(key, None)

    # ---------------- 读 ----------------

    async def get(self, session_id: str) -> dict[str, Any] | None:
        key = self._key(session_id)
        if self.redis is not None:
            try:
                raw = await self.redis.get(key)
                if raw is None:
                    return None
                return json.loads(raw)
            except json.JSONDecodeError:
                logger.error("会话数据损坏: %s", key)
                return None
            except Exception as exc:  # noqa: BLE001
                logger.warning("读取会话失败，回退内存存储: %s", exc)

        entry = self._memory_store.get(key)
        if entry is None:
            return None
        expires_at, payload = entry
        if time.time() > expires_at:
            self._memory_store.pop(key, None)
            return None
        return payload

    async def exists(self, session_id: str) -> bool:
        return await self.get(session_id) is not None

    async def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """列出最近的会话。仅用于调试面板，不做分页。"""
        sessions: list[dict[str, Any]] = []
        if self.redis is not None:
            try:
                async for key in self.redis.scan_iter(
                    match=f"{SESSION_KEY_PREFIX}:*", count=100
                ):
                    raw = await self.redis.get(key)
                    if raw:
                        try:
                            sessions.append(json.loads(raw))
                        except json.JSONDecodeError:
                            continue
                    if len(sessions) >= limit:
                        break
            except Exception as exc:  # noqa: BLE001
                logger.warning("列举会话失败: %s", exc)
        else:
            now = time.time()
            for expires_at, payload in list(self._memory_store.values()):
                if expires_at > now:
                    sessions.append(payload)
        sessions.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return sessions[:limit]

    # ---------------- 内部 ----------------

    async def _write(self, session_id: str, payload: dict[str, Any]) -> None:
        key = self._key(session_id)
        if self.redis is not None:
            try:
                # 每次写都重设 TTL：长任务不该在执行过程中过期
                await self.redis.setex(
                    key, self.ttl_seconds, json.dumps(payload, ensure_ascii=False, default=str)
                )
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("写入会话失败，降级为内存存储: %s", exc)
        self._memory_store[key] = (time.time() + self.ttl_seconds, payload)
        self._evict_expired()

    @classmethod
    def _evict_expired(cls, max_entries: int = 500) -> None:
        """内存降级模式下清理过期项，防止无限增长。"""
        now = time.time()
        expired = [key for key, (expires_at, _) in cls._memory_store.items() if expires_at <= now]
        for key in expired:
            cls._memory_store.pop(key, None)
        if len(cls._memory_store) > max_entries:
            oldest = sorted(cls._memory_store.items(), key=lambda item: item[1][0])
            for key, _ in oldest[: len(cls._memory_store) - max_entries]:
                cls._memory_store.pop(key, None)

    @staticmethod
    def _key(session_id: str) -> str:
        return f"{SESSION_KEY_PREFIX}:{session_id}"

    def health(self) -> dict[str, Any]:
        return {
            "backend": "redis" if self.redis is not None else "memory(degraded)",
            "ttl_seconds": self.ttl_seconds,
            "memory_entries": len(self._memory_store),
        }


__all__ = [
    "SESSION_STATUS_COMPLETED",
    "SESSION_STATUS_FAILED",
    "SESSION_STATUS_PENDING",
    "SESSION_STATUS_RUNNING",
    "SessionStore",
    "describe_node_progress",
]
