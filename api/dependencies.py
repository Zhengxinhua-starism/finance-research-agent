"""FastAPI 依赖注入。

解决什么问题
    Redis 连接池、LLM 客户端、MCP Server、embedding 模型这些都是
    "创建昂贵、必须复用"的对象。如果在路由函数里直接 new 出来，
    每个请求会重新加载 200MB 的模型、重建连接池——第二个并发请求就会 OOM。
    本模块把它们做成进程级单例，通过 Depends 注入。

核心设计决策
    1. 单例在 **lifespan 启动时创建**，不是首次请求时懒加载。
       懒加载会让第一个用户等 30 秒（加载 embedding + CrossEncoder），
       而且并发的首批请求会同时触发加载。启动时预热把这个代价
       移到部署阶段，用户侧永远是热的。
    2. Redis 连接失败**不阻止服务启动**（config.redis_optional）。
       Redis 只影响缓存和会话持久化，两者都有降级路径；
       为它拒绝启动会让"演示环境没装 Redis"变成"项目跑不起来"。
    3. 依赖函数返回的是共享实例而不是每次新建，唯一例外是 SessionStore
       ——它是无状态的薄封装，每次新建的开销可以忽略，
       而且这样能保证它总是拿到最新的 Redis 连接。
    4. get_research_pipeline 返回的 Pipeline 里含四个 Agent，
       它们在请求之间共享。唯一 per-request 的是 Tracer，
       由路由层创建后通过 bind_tracer 注入。

为什么不用其他方案
    - 不用全局变量 + import 副作用：那样 `import api.dependencies`
      就会触发模型加载，任何单元测试和 CLI 命令都要等 30 秒。
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from config import Config, get_config
from graph.graph import ResearchPipeline
from llm.router import LLMRouter
from mcp_servers.base_server import MCPServer
from session.cache import ToolCache
from session.session_store import SessionStore

logger = logging.getLogger(__name__)


class AppState:
    """进程级共享资源。由 lifespan 创建和销毁。"""

    def __init__(self) -> None:
        self.config: Config = get_config()
        self.redis: Any = None
        self.llm_router: LLMRouter | None = None
        self.pipeline: ResearchPipeline | None = None
        self.startup_errors: list[str] = []

    # ---------------- 生命周期 ----------------

    async def startup(self, warm_up: bool = True) -> None:
        self.config.ensure_runtime_dirs()
        await self._connect_redis()

        self.llm_router = LLMRouter(self.config)
        try:
            self.pipeline = ResearchPipeline()
            logger.info(
                "研究流程已初始化，可用工具: %s", self.pipeline.context.broker.tool_names
            )
        except Exception as exc:  # noqa: BLE001
            # Pipeline 初始化失败是致命的（没有它 /api/research 无法工作），
            # 但仍让服务起来：/api/health 会如实报告 error 状态，
            # 比进程直接退出更容易排查（能看到日志和健康检查）。
            message = f"研究流程初始化失败: {type(exc).__name__}: {exc}"
            logger.exception(message)
            self.startup_errors.append(message)

        if warm_up and self.pipeline is not None:
            self._warm_up_models()

    def _warm_up_models(self) -> None:
        """预热 embedding 与 CrossEncoder。

        失败只记录不抛出：模型下载失败时知识库检索会降级，
        但财务数据分析仍然可用，没必要因此让服务不可用。
        """
        try:
            from rag.rag_store import get_embedding_model

            get_embedding_model()
            logger.info("embedding 模型预热完成")
        except Exception as exc:  # noqa: BLE001
            message = f"embedding 模型预热失败（知识库检索将不可用）: {exc}"
            logger.warning(message)
            self.startup_errors.append(message)

        try:
            from rag.reranker import get_cross_encoder

            get_cross_encoder()
            logger.info("CrossEncoder 模型预热完成")
        except Exception as exc:  # noqa: BLE001
            logger.warning("CrossEncoder 预热失败（精排将被跳过）: %s", exc)

    async def _connect_redis(self) -> None:
        try:
            from redis.asyncio import Redis

            self.redis = Redis.from_url(
                self.config.redis_url,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
                health_check_interval=30,
            )
            await self.redis.ping()
            logger.info("Redis 已连接: %s", self.config.redis_url)
        except Exception as exc:  # noqa: BLE001
            self.redis = None
            message = f"Redis 连接失败，会话与缓存降级为内存模式: {exc}"
            if self.config.redis_optional:
                logger.warning(message)
                self.startup_errors.append(message)
            else:
                raise RuntimeError(message) from exc

    async def shutdown(self) -> None:
        if self.redis is not None:
            try:
                await self.redis.aclose()
            except Exception as exc:  # noqa: BLE001
                logger.warning("关闭 Redis 连接失败: %s", exc)
        if self.llm_router is not None:
            self.llm_router.close()
        if self.pipeline is not None:
            for server in self.pipeline.context.mcp_servers:
                server.registry.shutdown()
        logger.info("应用资源已释放")

    # ---------------- 健康检查 ----------------

    async def redis_status(self) -> str:
        if self.redis is None:
            return "disconnected(degraded to memory)"
        try:
            await self.redis.ping()
            return "connected"
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    def chroma_status(self) -> tuple[str, int]:
        try:
            from rag.rag_store import RagStore

            store = RagStore()
            count = store.count()
            if count == 0:
                return "connected(empty, run: python -m rag.prepare_data)", 0
            return "connected", count
        except Exception as exc:  # noqa: BLE001
            return f"error: {type(exc).__name__}: {exc}", 0


# 进程级单例。由 api/app.py 的 lifespan 负责初始化。
app_state = AppState()


# ============================================================
# FastAPI 依赖函数
# ============================================================


async def get_app_state() -> AppState:
    return app_state


async def get_redis() -> Any:
    """Redis 客户端。未连接时返回 None，调用方负责降级。"""
    return app_state.redis


async def get_session_store() -> SessionStore:
    return SessionStore(redis=app_state.redis)


async def get_tool_cache() -> ToolCache:
    return ToolCache(redis=app_state.redis)


async def get_llm_router() -> LLMRouter:
    if app_state.llm_router is None:
        app_state.llm_router = LLMRouter(app_state.config)
    return app_state.llm_router


async def get_mcp_servers() -> list[MCPServer]:
    pipeline = await get_research_pipeline()
    return pipeline.context.mcp_servers


async def get_research_pipeline() -> ResearchPipeline:
    """研究流程单例。未初始化时抛 RuntimeError，由路由层转成 503。"""
    if app_state.pipeline is None:
        raise RuntimeError(
            "研究流程未初始化。"
            + ("启动错误: " + "; ".join(app_state.startup_errors) if app_state.startup_errors else "")
        )
    return app_state.pipeline


async def get_research_graph() -> Any:
    """已编译的 LangGraph 实例（md 约定的依赖名）。"""
    pipeline = await get_research_pipeline()
    return pipeline.graph


async def lifespan_context() -> AsyncIterator[None]:
    await app_state.startup()
    try:
        yield
    finally:
        await app_state.shutdown()


__all__ = [
    "AppState",
    "app_state",
    "get_app_state",
    "get_llm_router",
    "get_mcp_servers",
    "get_redis",
    "get_research_graph",
    "get_research_pipeline",
    "get_session_store",
    "get_tool_cache",
    "lifespan_context",
]
