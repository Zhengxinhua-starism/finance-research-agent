"""FastAPI 路由。

解决什么问题
    把 LangGraph 研究流程暴露成 HTTP 接口，同时处理两个现实约束：
    (1) 一次研究要跑 30~150 秒，同步请求会打爆网关超时；
    (2) 研究流程是**同步阻塞**代码（AKShare 是 requests，
        sentence-transformers 是 CPU 密集），直接在 async 路由里跑
        会卡死整个事件循环，导致其他请求（包括健康检查）全部无响应。

核心设计决策
    1. 所有阻塞调用都用 `anyio.to_thread.run_sync` 丢到线程池。
       这是 async 框架里跑同步代码的唯一正确姿势。
       # TODO: md 只说"FastAPI 路由用 async"，没说同步的 Agent 流程怎么接。
       # 直接在 async def 里调 pipeline.run() 会阻塞事件循环——
       # 这是 FastAPI 项目最常见也最隐蔽的性能事故，必须显式处理。
    2. 提供同步和异步两种研究模式：
       - async_mode=false（默认）：等结果返回，适合 CLI 和评测脚本；
       - async_mode=true：立即返回 session_id，客户端通过
         GET /api/session/{id} 轮询，或 GET /api/session/{id}/events 订阅 SSE。
         适合 Vue 演示前端。
       两种模式共用同一段执行逻辑，只是包装方式不同。
    3. 后台任务用 asyncio.create_task 而不是 FastAPI 的 BackgroundTasks：
       BackgroundTasks 在响应发送**之后**才执行，而我们需要
       "立即返回 session_id 的同时任务已经在跑"。
    4. /api/health 不依赖任何可能失败的组件：它逐个探测并报告状态，
       自身永远返回 200。健康检查接口自己挂掉是运维噩梦。

为什么不用其他方案
    - 不用 Celery / RQ 做异步任务：需要额外的 worker 进程和消息队列，
      与"docker-compose 一键起"的目标冲突。asyncio 任务 + Redis 会话状态
      在单实例场景下完全够用，且失败语义更简单。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import date, datetime
from typing import Any

import anyio.to_thread
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from api.dependencies import (
    AppState,
    get_app_state,
    get_mcp_servers,
    get_research_pipeline,
    get_session_store,
)
from api.schemas import (
    EvalRequest,
    EvalResponse,
    HealthResponse,
    ReportPayload,
    ResearchRequest,
    ResearchResponse,
    SessionResponse,
    TraceResponse,
)
from config import get_config
from graph.graph import ResearchPipeline
from harness.tracing import Tracer, find_trace_file, load_trace
from mcp_servers.base_server import MCPServer
from session.session_store import (
    SESSION_STATUS_FAILED,
    SESSION_STATUS_RUNNING,
    SessionStore,
    describe_node_progress,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["research"])


# ============================================================
# 研究接口
# ============================================================


@router.post(
    "/research",
    response_model=ResearchResponse,
    summary="发起一次金融研报研究",
    description=(
        "输入研究问题和股票代码，Agent 会自动规划 → 取数 → 核查证据 → 生成三级标注研报。\n\n"
        "- `async_mode=false`（默认）：同步等待结果，耗时 30~150 秒；\n"
        "- `async_mode=true`：立即返回 session_id；轮询 `GET /api/session/{id}`，"
        "或订阅 `GET /api/session/{id}/events`（SSE）获取节点进度。"
    ),
)
async def create_research(
    request: ResearchRequest,
    pipeline: ResearchPipeline = Depends(get_research_pipeline),
    session_store: SessionStore = Depends(get_session_store),
) -> ResearchResponse:
    session_id = request.session_id or uuid.uuid4().hex[:16]
    as_of_date = request.resolved_as_of_date()

    await session_store.create(
        session_id=session_id,
        question=request.question,
        company=request.company_name or "",
        ticker=request.company_ticker,
        as_of_date=as_of_date.isoformat(),
    )

    if request.async_mode:
        # 不 await：任务在后台跑，立即返回让客户端去轮询。
        # 保留 task 引用防止被 GC 回收（Python 的已知陷阱：
        # create_task 返回的 task 如果没有强引用可能被中途回收）。
        task = asyncio.create_task(
            _run_research_task(pipeline, session_store, request, session_id, as_of_date)
        )
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)
        return ResearchResponse(
            run_id="",
            session_id=session_id,
            status="running",
            message=f"研究已在后台启动，请轮询 GET /api/session/{session_id} 获取结果",
        )

    outcome = await _execute_research(pipeline, session_store, request, session_id, as_of_date)
    return outcome


# 后台任务的强引用集合
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


async def _run_research_task(
    pipeline: ResearchPipeline,
    session_store: SessionStore,
    request: ResearchRequest,
    session_id: str,
    as_of_date: date,
) -> None:
    try:
        await _execute_research(pipeline, session_store, request, session_id, as_of_date)
    except Exception as exc:  # noqa: BLE001 — 后台任务的异常没人接，必须自己兜住
        logger.exception("后台研究任务失败 session=%s", session_id)
        await session_store.set_failed(session_id, f"{type(exc).__name__}: {exc}")


async def _execute_research(
    pipeline: ResearchPipeline,
    session_store: SessionStore,
    request: ResearchRequest,
    session_id: str,
    as_of_date: date,
) -> ResearchResponse:
    """执行研究并写入会话。同步流程在线程池里跑。"""
    await session_store.update_status(session_id, SESSION_STATUS_RUNNING, current_node="planner")
    tracer = Tracer()
    loop = asyncio.get_running_loop()

    def on_node(node: str, state: dict[str, Any]) -> None:
        if not node:
            return
        retrieval_count = int(state.get("retrieval_count") or 0)
        ratio, label = describe_node_progress(node, retrieval_count)
        future = asyncio.run_coroutine_threadsafe(
            session_store.update_status(
                session_id,
                SESSION_STATUS_RUNNING,
                current_node=node,
                progress=ratio,
                progress_label=label,
                retrieval_count=retrieval_count,
            ),
            loop,
        )
        try:
            future.result(timeout=3)
        except Exception:  # noqa: BLE001
            logger.warning("写入节点进度失败 node=%s session=%s", node, session_id)

    def run_blocking() -> dict[str, Any]:
        # pipeline.run 内部会调 AKShare（requests）和 sentence-transformers（CPU），
        # 全是阻塞操作，必须隔离在工作线程里，否则事件循环会被卡住
        return pipeline.run(
            question=request.question,
            ticker=request.company_ticker,
            company=request.company_name or "",
            as_of_date=as_of_date,
            session_id=session_id,
            tracer=tracer,
            on_node=on_node,
        )

    try:
        outcome = await anyio.to_thread.run_sync(run_blocking)
    except Exception as exc:  # noqa: BLE001
        logger.exception("研究执行失败 session=%s", session_id)
        await session_store.set_failed(session_id, f"{type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"研究流程执行失败: {type(exc).__name__}: {exc}",
        ) from exc

    state = outcome["state"]
    report_dict = state.get("report_dict") or {"markdown": state.get("report_markdown", "")}
    trace_summary = outcome["trace_summary"]
    errors = list(state.get("errors") or [])
    final_status = "completed" if state.get("status") == "completed" else "failed"

    await session_store.set_result(
        session_id=session_id,
        report=report_dict,
        trace_summary=trace_summary,
        run_id=outcome["run_id"],
    )
    if final_status == "failed":
        await session_store.update_status(
            session_id, SESSION_STATUS_FAILED, error="; ".join(errors[:3]) or "未知错误"
        )

    return ResearchResponse(
        run_id=outcome["run_id"],
        session_id=session_id,
        status=final_status,  # type: ignore[arg-type]
        report=ReportPayload(**_normalize_report(report_dict)),
        trace_summary=trace_summary,
        errors=errors,
        message="研究完成" if final_status == "completed" else "研究过程中出现错误，结果可能不完整",
    )


def _normalize_report(report_dict: dict[str, Any]) -> dict[str, Any]:
    """把 Writer 的输出对齐到 ReportPayload 的字段集。"""
    return {
        "title": report_dict.get("title", ""),
        "markdown": report_dict.get("markdown", ""),
        "conclusions": report_dict.get("conclusions", []),
        "analysis_body": report_dict.get("analysis_body", ""),
        "source_table": report_dict.get("source_table", []),
        "stats": report_dict.get("stats", {}),
        "refused": report_dict.get("refused", False),
    }


def session_payload_to_response(payload: dict[str, Any]) -> SessionResponse:
    """把 Redis/内存里的会话 dict 转成 API 契约。SSE 与 GET 共用。"""
    result = payload.get("result")
    return SessionResponse(
        session_id=payload["session_id"],
        status=payload.get("status", "unknown"),
        current_node=payload.get("current_node", ""),
        node_path=payload.get("node_path") or [],
        progress=float(payload.get("progress") or 0.0),
        progress_label=payload.get("progress_label") or "",
        question=payload.get("question", ""),
        company=payload.get("company", ""),
        ticker=payload.get("ticker", ""),
        created_at=payload.get("created_at", ""),
        updated_at=payload.get("updated_at", ""),
        run_id=payload.get("run_id"),
        result=ReportPayload(**_normalize_report(result)) if result else None,
        trace_summary=payload.get("trace_summary"),
        error=payload.get("error"),
    )


def format_sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


# ============================================================
# 会话查询
# ============================================================


@router.get(
    "/session/{session_id}",
    response_model=SessionResponse,
    summary="查询研究会话状态与结果",
)
async def get_session(
    session_id: str,
    session_store: SessionStore = Depends(get_session_store),
) -> SessionResponse:
    payload = await session_store.get(session_id)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话 {session_id} 不存在或已过期（TTL {get_config().session_ttl_seconds}s）",
        )
    return session_payload_to_response(payload)


@router.get(
    "/session/{session_id}/events",
    summary="SSE：节点级研究进度",
    description=(
        "推送 `progress` 事件（节点/百分比变化）以及结束时的 `done`。"
        "Vue 前端用 EventSource 订阅，避免 2 秒轮询。"
    ),
)
async def stream_session_events(
    session_id: str,
    session_store: SessionStore = Depends(get_session_store),
) -> StreamingResponse:
    if not await session_store.exists(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话 {session_id} 不存在或已过期（TTL {get_config().session_ttl_seconds}s）",
        )

    async def event_stream():
        last_signature: tuple[Any, ...] | None = None
        deadline = time.time() + 600
        while time.time() < deadline:
            payload = await session_store.get(session_id)
            if payload is None:
                yield format_sse("error", {"error": "会话已过期"})
                return
            signature = (
                payload.get("status"),
                payload.get("current_node"),
                payload.get("progress"),
                payload.get("progress_label"),
                payload.get("updated_at"),
            )
            if signature != last_signature:
                last_signature = signature
                body = session_payload_to_response(payload).model_dump()
                yield format_sse("progress", body)
                if payload.get("status") in {"completed", "failed"}:
                    yield format_sse("done", body)
                    return
            await asyncio.sleep(0.4)
        yield format_sse("error", {"error": "研究超时（超过 10 分钟）"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions", summary="列出最近的研究会话（调试用）")
async def list_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    session_store: SessionStore = Depends(get_session_store),
) -> dict[str, Any]:
    sessions = await session_store.list_sessions(limit=limit)
    return {
        "count": len(sessions),
        "backend": session_store.health()["backend"],
        "sessions": [
            {
                "session_id": item.get("session_id"),
                "question": item.get("question"),
                "status": item.get("status"),
                "created_at": item.get("created_at"),
                "run_id": item.get("run_id"),
            }
            for item in sessions
        ],
    }


# ============================================================
# Trace 查询
# ============================================================


@router.get(
    "/trace/{run_id}",
    response_model=TraceResponse,
    summary="获取一次运行的完整 trace",
    description="返回该次运行的全部事件（LLM 调用、工具调用、门禁检查、上下文压缩）。",
)
async def get_trace(run_id: str) -> TraceResponse:
    path = await anyio.to_thread.run_sync(find_trace_file, run_id)
    if path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到 run_id={run_id} 的 trace 文件。trace 保存在 {get_config().traces_dir}",
        )
    try:
        data = await anyio.to_thread.run_sync(load_trace, path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"读取 trace 文件失败: {exc}",
        ) from exc

    return TraceResponse(
        run_id=run_id,
        summary=data.get("summary", {}),
        final_result=data.get("final_result", ""),
        events=data.get("events", []),
    )


@router.get("/traces", summary="列出最近的 trace 文件")
async def list_traces(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    trace_dir = get_config().traces_dir
    if not trace_dir.exists():
        return {"count": 0, "traces": []}

    def scan() -> list[dict[str, Any]]:
        files = sorted(
            trace_dir.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )[:limit]
        return [
            {
                "file": path.name,
                "run_id": path.stem.split("_")[1] if "_" in path.stem else path.stem,
                "size_bytes": path.stat().st_size,
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            }
            for path in files
        ]

    traces = await anyio.to_thread.run_sync(scan)
    return {"count": len(traces), "traces": traces}


# ============================================================
# 评测
# ============================================================


@router.post(
    "/eval",
    response_model=EvalResponse,
    summary="运行自动化评测",
    description=(
        "对标准测试集跑 LLM-as-Judge 评测，输出四个维度的分数："
        "factual_accuracy（事实准确性）、refusal_calibration（拒答校准）、"
        "retrieval_efficiency（检索效率）、evidence_coverage（证据覆盖率）。\n\n"
        "注意：跑全部 9 道题需要 10~20 分钟，建议先用 test_case_ids 跑单题。"
    ),
)
async def run_eval(
    request: EvalRequest,
    pipeline: ResearchPipeline = Depends(get_research_pipeline),
) -> EvalResponse:
    # 延迟导入：评测模块会加载测试集文件，
    # 放在模块顶部会让 API 启动时就依赖 eval/test_cases.json 存在
    from eval.evaluate import Evaluator

    evaluator = Evaluator(pipeline=pipeline)

    def run_blocking() -> dict[str, Any]:
        return evaluator.run(
            case_ids=request.test_case_ids, save_report=request.save_report
        )

    try:
        outcome = await anyio.to_thread.run_sync(run_blocking)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("评测执行失败")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"评测执行失败: {type(exc).__name__}: {exc}",
        ) from exc

    return EvalResponse(
        results=outcome["results"],
        summary=outcome["summary"],
        report_path=outcome.get("report_path"),
    )


@router.get("/eval/cases", summary="列出全部评测用例")
async def list_eval_cases() -> dict[str, Any]:
    from eval.evaluate import load_test_cases

    try:
        cases = await anyio.to_thread.run_sync(load_test_cases)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return {
        "count": len(cases),
        "cases": [
            {
                "id": case["id"],
                "company": case.get("company"),
                "ticker": case.get("ticker"),
                "question": case.get("question"),
                "question_type": case.get("question_type"),
                "difficulty": case.get("difficulty"),
                "should_refuse": case.get("should_refuse", False),
            }
            for case in cases
        ],
    }


# ============================================================
# 健康检查与元信息
# ============================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="健康检查",
    description="逐个探测 Redis、Chroma、LLM 配置、MCP Server 的可用性。本接口永远返回 200。",
)
async def health(app: AppState = Depends(get_app_state)) -> HealthResponse:
    redis_status = await app.redis_status()
    chroma_status, document_count = await anyio.to_thread.run_sync(app.chroma_status)

    mcp_health: list[dict[str, Any]] = []
    llm_info: dict[str, Any] = {}
    pipeline_error: str | None = None
    try:
        pipeline = await get_research_pipeline()
        mcp_health = pipeline.context.broker.health()
        llm_info = pipeline.context.llm_router.describe()
    except RuntimeError as exc:
        pipeline_error = str(exc)

    config = get_config()
    llm_info["api_key_configured"] = bool(config.deepseek_api_key)

    if pipeline_error or not config.deepseek_api_key:
        overall = "error"
    elif redis_status != "connected" or document_count == 0:
        overall = "degraded"
    else:
        overall = "ok"

    return HealthResponse(
        status=overall,  # type: ignore[arg-type]
        redis=redis_status,
        chroma=chroma_status,
        llm=llm_info,
        mcp_servers=mcp_health,
        knowledge_base_documents=document_count,
        details={
            "startup_errors": app.startup_errors,
            "pipeline_error": pipeline_error,
            "trace_dir": str(config.traces_dir),
            "degraded_reason": _degraded_reason(
                redis_status, document_count, bool(config.deepseek_api_key)
            ),
        },
    )


def _degraded_reason(redis_status: str, document_count: int, has_api_key: bool) -> list[str]:
    reasons: list[str] = []
    if not has_api_key:
        reasons.append("未配置 DEEPSEEK_API_KEY，所有 LLM 调用都会失败")
    if redis_status != "connected":
        reasons.append("Redis 不可用，会话与缓存降级为内存模式（多进程部署下会失效）")
    if document_count == 0:
        reasons.append("知识库为空，请执行 python -m rag.prepare_data")
    return reasons


@router.get("/tools", summary="列出所有 MCP 工具（协议视图）")
async def list_tools(servers: list[MCPServer] = Depends(get_mcp_servers)) -> dict[str, Any]:
    return {
        "servers": [
            {
                "name": server.name,
                "description": server.description,
                "protocol_version": server.protocol_version,
                "tools": server.list_tools(),
            }
            for server in servers
        ]
    }


@router.post(
    "/mcp/{server_name}",
    summary="直接发送 JSON-RPC 2.0 请求到指定 MCP Server（调试用）",
    description=(
        '示例请求体：{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n\n'
        '调用工具：{"jsonrpc":"2.0","id":2,"method":"tools/call",'
        '"params":{"name":"get_financial_metrics","arguments":{"ticker":"002594"}}}'
    ),
)
async def mcp_jsonrpc(
    server_name: str,
    payload: dict[str, Any],
    servers: list[MCPServer] = Depends(get_mcp_servers),
) -> dict[str, Any]:
    server = next((item for item in servers if item.name == server_name), None)
    if server is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MCP Server {server_name} 不存在。可用: {[s.name for s in servers]}",
        )
    return await anyio.to_thread.run_sync(server.handle_request, payload)


@router.get("/config", summary="查看当前生效的关键配置（不含密钥）")
async def show_config() -> dict[str, Any]:
    config = get_config()
    return {
        "models": {"quick": config.quick_model, "deep": config.deep_model},
        "harness": {
            "max_turns": config.max_turns,
            "tool_timeout_seconds": config.tool_timeout_seconds,
            "context_max_tokens": config.context_max_tokens,
            "max_total_tokens": config.max_total_tokens,
        },
        "rag": {
            "bm25_top_k": config.bm25_top_k,
            "vector_top_k": config.vector_top_k,
            "rerank_top_k": config.rerank_top_k,
            "rrf_k": config.rrf_k,
            "embedding_model": config.embedding_model,
            "reranker_model": config.reranker_model,
        },
        "evidence_gate": {
            "number_tolerance": config.number_tolerance,
            "ratio_tolerance_pp": config.ratio_tolerance_pp,
            "growth_tolerance_pp": config.growth_tolerance_pp,
            "unverified_ratio_threshold": config.unverified_ratio_threshold,
            "max_retrieval_rounds": config.max_retrieval_rounds,
        },
        "cache": {
            "redis_ttl_seconds": config.cache_ttl_seconds,
            "disk_ttl_seconds": config.disk_cache_ttl_seconds,
        },
    }


__all__ = ["router"]
