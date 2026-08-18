"""FastAPI 应用。

解决什么问题
    给 Agent 一个生产形态的服务外壳：能被任意客户端（Vue、
    评测脚本、curl）调用，有统一的错误格式、有自动生成的接口文档、
    有正确的资源生命周期管理。演示前端是 Vue3，构建后由本服务托管；
    CLI 仍直接调 Pipeline，用于调试。

为什么需要 API 层（而不是把界面和 Agent 塞进同一个进程）
    - Web 界面崩溃不应带走 Agent；
    - Agent 是被调用的能力，不是一个网页；
    - 前后端解耦后可以独立扩缩容。

架构
    Vue / curl → FastAPI /api/research → LangGraph 流程 → 三级标注研报 JSON
    Vue 通过 SSE（/api/session/{id}/events）订阅节点进度，不 import Agent 代码。

核心设计决策
    1. 用 lifespan 而不是 @app.on_event("startup")。后者在 FastAPI 0.109+
       已废弃，而且没法保证"启动失败时不接收流量"——lifespan 里抛异常
       会阻止服务监听端口，语义更正确。
    2. 全局异常处理器统一错误格式。默认的 FastAPI 错误响应是
       {"detail": "..."}，而参数校验错误是另一种结构，客户端要写两套解析。
       统一成 {error, error_type, detail} 后只需要一套。
    3. 未捕获异常**不返回堆栈**给客户端，只返回异常类型和一句话，
       完整堆栈进服务端日志。堆栈里可能包含文件路径和内部结构。
    4. CORS 允许所有来源。这是演示项目的取舍——生产环境必须收紧到
       具体域名，注释里写明了这一点，免得被当成疏忽。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.dependencies import app_state
from api.routes import router
from config import PROJECT_ROOT, get_config

logger = logging.getLogger(__name__)

API_DESCRIPTION = """
金融研报 Agent API — 自研 Harness + Agentic RAG + MCP 工具协议

## 能做什么
输入一个研究问题和 A 股代码，返回一份**带三级证据标注**的研报：
- ✅已验证：通过三级证据门禁（来源存在 + 数字一致 + 时点合规）
- ⚠️未验证：有来源但未通过数字校验，或来源为新闻等非官方渠道
- ❌拒答：找不到证据支撑，明确标注"无法判断"，不编造

## 技术架构
```
FastAPI → LangGraph 编排 → 四个 Agent 节点
                            ├─ Planner（deepseek-chat）问题拆解
                            ├─ Retriever（deepseek-chat）MCP 工具调用
                            ├─ Verifier（deepseek-chat）断言抽取 + 证据门禁
                            └─ Writer（deepseek-chat）三级标注研报

自研 Harness：ReAct 循环 + max_turns 保护 + 上下文压缩 + 证据门禁 + JSON trace
Agentic RAG：BM25 + 向量双路召回 → RRF 融合 → CrossEncoder 精排
MCP 协议：工具通过 JSON-RPC 2.0 语义暴露，新增数据源无需改 Agent 代码
```

## 使用建议
- 首次使用请先跑 `GET /api/health` 确认 Redis / Chroma / API Key 都就绪；
- 知识库为空时执行 `python -m rag.prepare_data`；
- 单次研究耗时 30~150 秒（补搜满轮可能更长），Web 前端用 `async_mode=true` + SSE。

## 免责声明
本服务可基于已披露信息给出研究观点（含买卖倾向），不构成持牌投资顾问服务，不保证收益。
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期：启动时初始化共享资源，关闭时释放。"""
    config = get_config()
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("正在启动 Finance Research Agent API...")
    await app_state.startup()
    if app_state.startup_errors:
        logger.warning(
            "启动时存在 %d 项降级：%s",
            len(app_state.startup_errors),
            "; ".join(app_state.startup_errors),
        )
    logger.info("API 已就绪，文档地址: http://%s:%d/docs", config.api_host, config.api_port)
    try:
        yield
    finally:
        logger.info("正在关闭 API 服务...")
        await app_state.shutdown()


app = FastAPI(
    title="Finance Research Agent API",
    description=API_DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "research", "description": "研究、会话、trace、评测、健康检查"},
    ],
)

# CORS：允许 Vite 开发服务器（localhost:5173）跨域调用。
# 生产环境把 Vue 构建产物挂在同源路径下，浏览器不会走跨域。
# 生产必须把 allow_origins 收紧到具体域名；
# "*" 加上 allow_credentials=True 在浏览器里本来就是非法组合，
# 这里显式设为 False 以避免误配。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(router)

WEB_DIST = PROJECT_ROOT / "web" / "dist"
_WEB_ASSETS = WEB_DIST / "assets"
if _WEB_ASSETS.is_dir():
    app.mount("/assets", StaticFiles(directory=_WEB_ASSETS), name="web-assets")


# ============================================================
# 统一错误处理
# ============================================================


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """参数校验失败 → 422，把 Pydantic 的错误列表转成可读的中文提示。"""
    problems = [
        f"{'.'.join(str(part) for part in error.get('loc', [])[1:])}: {error.get('msg', '')}"
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "请求参数校验失败",
            "error_type": "validation_error",
            "detail": "; ".join(problems),
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": str(exc.detail),
            "error_type": f"http_{exc.status_code}",
            "detail": "",
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底处理器。堆栈只进日志，不返回给客户端。"""
    logger.exception("未处理的异常: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "服务内部错误",
            "error_type": type(exc).__name__,
            "detail": "请查看服务端日志获取详细信息",
        },
    )


# ============================================================
# 根路由
# ============================================================


@app.get("/", summary="演示前端或服务信息", tags=["research"])
async def root() -> Any:
    index = WEB_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {
        "service": "Finance Research Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "web": "未找到 web/dist，请先在 web/ 目录执行 npm run build",
        "endpoints": {
            "POST /api/research": "发起研究",
            "GET /api/session/{session_id}": "查询会话状态与结果",
            "GET /api/session/{session_id}/events": "SSE 节点进度",
            "GET /api/trace/{run_id}": "获取运行 trace",
            "POST /api/eval": "运行自动化评测",
            "GET /api/health": "健康检查",
            "GET /api/tools": "列出 MCP 工具",
        },
        "disclaimer": "可给出研究观点（含买卖倾向），不构成持牌投顾服务",
    }


__all__ = ["app"]
