"""LangGraph 编排：StateGraph 构建 + 条件边。

解决什么问题
    四个 Agent 之间不是简单的流水线：Verifier 判定证据不足时要**回到**
    Retriever 补搜，补搜完再验证，最多三轮。这种带条件回边的控制流
    如果用 while 循环手写，状态管理、轮次上限、每个分支的日志会缠在一起，
    而且改流程就要改控制流代码。StateGraph 把"流程拓扑"和"节点逻辑"分开。

核心设计决策
    1. **这里用框架，Harness 里自建**——这是整个项目最值得讲的架构判断。
       编排（谁在什么条件下跑）是标准化问题，LangGraph 的
       StateGraph + 条件边 + reducer 是成熟方案，自建没有额外价值；
       而运行时（ReAct 循环里的 max_turns、上下文压缩时机、证据门禁）
       是这个项目的差异化所在，藏进框架就没法讲、也没法改。
       分界线画在"通用 vs 领域特有"上，不是画在"简单 vs 复杂"上。
    2. 节点函数只做三件事：调 Agent、把结果转成状态增量、记 trace。
       业务逻辑全在 agents/ 里。这样节点函数保持在 30 行以内，
       流程图一眼能看懂，Agent 也能脱离 Graph 单独测试。
    3. 每个节点都用 try/except 包住，异常转成 errors 里的一条记录 +
       状态降级，而不是让整个 graph 崩掉。四个节点里任何一个挂掉，
       后面的节点仍能基于已有信息产出降级结果（Writer 会输出拒答研报）。
    4. 条件边的判定函数是**纯函数**（只读状态、返回字符串），
       不做任何计算和副作用。补搜与否的判断逻辑在 Verifier 里已经算好，
       这里只是读一个布尔值——把判断和路由混在一起是难以测试的写法。
    5. 补搜轮次有两道保险：Verifier 内部检查 retrieval_count，
       路由函数再检查一次。单点控制在 LLM 参与的系统里不够安全。

为什么不用其他方案
    - 不用 LangGraph 的 checkpointer 做持久化：会话状态走 Redis
      （见 session/），存的是渲染后的结果而不是原始状态。
      checkpointer 存的 pickle 结构与代码版本强绑定，升级后旧会话会读不出来。
"""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar
from datetime import date
from typing import Any, Callable, Literal

from langgraph.graph import END, START, StateGraph

from config import get_config
from harness.evidence_gate import EvidenceGate
from harness.tracing import NullTracer, Tracer
from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.verifier import VerifierAgent, VerifyResult
from agents.writer import WriterAgent
from graph.state import ResearchState, create_initial_state, state_summary
from llm.router import LLMRouter
from mcp_servers.base_server import MCPServer, MCPToolBroker

logger = logging.getLogger(__name__)


class ResearchGraphContext:
    """持有四个 Agent 及其共享依赖。

    做成一个对象而不是四个全局变量：LLM 客户端、MCP Server、
    embedding 模型都是重对象，必须在多次请求之间共享；
    同时 tracer 是 per-run 的，需要在每次运行时替换。
    """

    def __init__(
        self,
        llm_router: LLMRouter | None = None,
        mcp_servers: list[MCPServer] | None = None,
        tracer: Tracer | None = None,
    ):
        self.config = get_config()
        self.llm_router = llm_router or LLMRouter()
        self.tracer = tracer or NullTracer()

        if mcp_servers is None:
            from mcp_servers import build_mcp_servers

            mcp_servers = build_mcp_servers(tracer=self.tracer)
        self.mcp_servers = mcp_servers
        self.broker = MCPToolBroker(mcp_servers)

        self.planner = PlannerAgent(
            llm_client=self.llm_router.get("quick"),
            available_tools=self.broker.get_tool_definitions(),
            tracer=self.tracer,
        )
        self.retriever = RetrieverAgent(
            llm_client=self.llm_router.get("quick"),
            mcp_servers=mcp_servers,
            broker=self.broker,
            tracer=self.tracer,
        )
        self.verifier = VerifierAgent(
            llm_client=self.llm_router.get("quick"),
            evidence_gate=EvidenceGate(tracer=self.tracer),
            tracer=self.tracer,
        )
        self.writer = WriterAgent(llm_client=self.llm_router.get("quick"), tracer=self.tracer)

    def bind_tracer(self, tracer: Tracer) -> None:
        """把 per-run 的 tracer 注入所有组件。

        Agent 实例在请求之间复用，但 trace 必须按运行隔离，
        所以 tracer 是唯一需要在运行时替换的依赖。
        """
        self.tracer = tracer
        self.planner.tracer = tracer
        self.verifier.tracer = tracer
        self.verifier.gate.tracer = tracer
        self.writer.tracer = tracer
        self.retriever.tracer = tracer
        self.retriever.registry.tracer = tracer
        self.retriever.agent_loop.tracer = tracer
        self.retriever.agent_loop.compactor.tracer = tracer
        for server in self.mcp_servers:
            server.tracer = tracer
            server.registry.tracer = tracer

    def health(self) -> dict[str, Any]:
        return {
            "llm": self.llm_router.describe(),
            "mcp_servers": self.broker.health(),
            "tool_count": len(self.broker.tool_names),
        }


# 进度回调按「本次 run」隔离，避免 Pipeline 单例在并发请求下串台。
_node_start_cb: ContextVar[Callable[[str, ResearchState], None] | None] = ContextVar(
    "_node_start_cb", default=None
)


def _with_node_start(
    name: str, node_fn: Callable[[ResearchState], dict[str, Any]]
) -> Callable[[ResearchState], dict[str, Any]]:
    """节点入口挂钩：在节点真正开始工作前通知调用方。

    graph.stream(stream_mode='values') 是节点结束后才吐状态，
    用来报进度会永远慢一拍（planner 跑完才显示 planner，此时 retriever 已在跑）。
    """

    def wrapped(state: ResearchState) -> dict[str, Any]:
        callback = _node_start_cb.get()
        if callback is not None:
            try:
                callback(name, state)
            except Exception:  # noqa: BLE001
                logger.exception("on_node 回调异常，已忽略")
        return node_fn(state)

    wrapped.__name__ = getattr(node_fn, "__name__", name)
    return wrapped


# ============================================================
# 节点函数
# ============================================================


def make_planner_node(context: ResearchGraphContext) -> Callable[[ResearchState], dict[str, Any]]:
    def planner_node(state: ResearchState) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            plan = context.planner.plan(
                question=state["question"],
                ticker=state.get("ticker", ""),
                company=state.get("company", ""),
                as_of_date=state["as_of_date"].isoformat(),
            )
            context.tracer.log(
                "node_exit",
                "planner",
                output_summary=f"{plan.question_type} / {plan.required_data}",
                duration_ms=int((time.perf_counter() - started) * 1000),
                fallback_used=plan.fallback_used,
            )
            return {
                "plan": plan,
                "status": "retrieving",
                "current_node": "planner",
                "node_path": ["planner"],
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Planner 节点异常")
            # 规划失败不终止流程：用规则计划继续跑，总比整体失败强
            fallback = context.planner.rule_based_plan(state["question"])
            return {
                "plan": fallback,
                "status": "retrieving",
                "current_node": "planner",
                "node_path": ["planner"],
                "errors": [f"planner: {type(exc).__name__}: {exc}（已降级为规则计划）"],
            }

    return planner_node


def make_retriever_node(context: ResearchGraphContext) -> Callable[[ResearchState], dict[str, Any]]:
    def retriever_node(state: ResearchState) -> dict[str, Any]:
        plan = state.get("plan")
        if plan is None:
            return {
                "status": "verifying",
                "current_node": "retriever",
                "node_path": ["retriever"],
                "retrieval_count": 1,
                "errors": ["retriever: 缺少研究计划，跳过检索"],
            }

        round_number = state.get("retrieval_count", 0) + 1
        verify_result = state.get("verify_result")
        try:
            evidence, agent_result = context.retriever.retrieve(
                question=state["question"],
                plan=plan,
                ticker=state.get("ticker", ""),
                company=state.get("company", ""),
                as_of_date=state["as_of_date"],
                round_number=round_number,
                existing_evidence=state.get("evidence_pool") or [],
                gaps=getattr(verify_result, "coverage_gaps", []) or [],
                expand_query=state.get("expand_query"),
            )

            errors: list[str] = []
            if agent_result.status != "completed":
                errors.append(
                    f"retriever: 循环以 {agent_result.status} 结束"
                    f"（{agent_result.error or '无详情'}），证据可能不完整"
                )
            failed_tools = [r.tool_name for r in agent_result.tool_results if not r.success]
            if failed_tools:
                errors.append(f"retriever: 工具调用失败 {failed_tools}")

            return {
                "evidence_pool": evidence,
                "retrieval_count": 1,
                "retriever_summary": agent_result.final_text,
                "tool_calls_made": [r.tool_name for r in agent_result.tool_results],
                "status": "verifying",
                "current_node": "retriever",
                "node_path": ["retriever"],
                "errors": errors,
                "token_usage": agent_result.token_usage.as_dict(),
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Retriever 节点异常")
            return {
                "retrieval_count": 1,
                "status": "verifying",
                "current_node": "retriever",
                "node_path": ["retriever"],
                "errors": [f"retriever: {type(exc).__name__}: {exc}"],
            }

    return retriever_node


def make_verifier_node(context: ResearchGraphContext) -> Callable[[ResearchState], dict[str, Any]]:
    def verifier_node(state: ResearchState) -> dict[str, Any]:
        plan = state.get("plan")
        try:
            result = context.verifier.verify(
                question=state["question"],
                evidence_pool=state.get("evidence_pool") or [],
                plan_description=plan.describe() if plan else "",
                ticker=state.get("ticker", ""),
                company=state.get("company", ""),
                as_of_date=state["as_of_date"],
                retrieval_count=state.get("retrieval_count", 1),
            )
            return {
                "verify_result": result,
                "needs_more_retrieval": result.needs_more_retrieval,
                "expand_query": result.expand_query,
                "status": "retrieving" if result.needs_more_retrieval else "writing",
                "current_node": "verifier",
                "node_path": ["verifier"],
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Verifier 节点异常")
            # 核查失败时**不允许**跳过：直接给出空结果，
            # Writer 会因为没有已验证结论而输出拒答研报。
            # 放行未经核查的内容是这个系统绝对不能做的事。
            return {
                "verify_result": VerifyResult(
                    claims=[],
                    needs_more_retrieval=False,
                    unverified_ratio=1.0,
                    coverage_gaps=[f"证据核查环节异常：{type(exc).__name__}: {exc}"],
                    extraction_failed=True,
                ),
                "needs_more_retrieval": False,
                "status": "writing",
                "current_node": "verifier",
                "node_path": ["verifier"],
                "errors": [f"verifier: {type(exc).__name__}: {exc}"],
            }

    return verifier_node


def make_writer_node(context: ResearchGraphContext) -> Callable[[ResearchState], dict[str, Any]]:
    def writer_node(state: ResearchState) -> dict[str, Any]:
        plan = state.get("plan")
        verify_result = state.get("verify_result")
        try:
            report = context.writer.write(
                question=state["question"],
                company=state.get("company", ""),
                ticker=state.get("ticker", ""),
                as_of_date=state["as_of_date"],
                verify_result=verify_result,
                evidence_pool=state.get("evidence_pool") or [],
                analysis_framework=getattr(plan, "analysis_framework", "general"),
                extra_notes=_build_extra_notes(state),
                out_of_scope_reason=getattr(plan, "out_of_scope_reason", "") or "",
            )
            return {
                "report_markdown": report.markdown,
                "report_dict": report.to_dict(),
                "status": "completed",
                "current_node": "writer",
                "node_path": ["writer"],
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Writer 节点异常")
            fallback = (
                f"# 研报生成失败\n\n"
                f"问题：{state['question']}\n\n"
                f"错误：{type(exc).__name__}: {exc}\n\n"
                f"已获取 {len(state.get('evidence_pool') or [])} 条证据，"
                f"但研报渲染环节失败。请查看 trace 日志定位问题。"
            )
            return {
                "report_markdown": fallback,
                "report_dict": {"markdown": fallback, "refused": True},
                "status": "failed",
                "current_node": "writer",
                "node_path": ["writer"],
                "errors": [f"writer: {type(exc).__name__}: {exc}"],
            }

    return writer_node


def _build_extra_notes(state: ResearchState) -> list[str]:
    """把流程中的降级信息写进研报备注。

    工具失败、循环超限这些事故必须让读者看到——一份基于不完整数据
    的研报如果不说明"哪里不完整"，比没有研报更危险。
    """
    notes: list[str] = []
    errors = state.get("errors") or []
    if errors:
        notes.append(f"本次分析过程中出现 {len(errors)} 项异常：" + "；".join(errors[:5]))
    if state.get("retrieval_count", 0) > 1:
        notes.append(f"经过 {state['retrieval_count']} 轮检索补充证据")
    verify_result = state.get("verify_result")
    if verify_result is not None:
        notes.append(
            f"证据核查：共 {len(verify_result.claims)} 条结论，"
            f"已验证 {len(verify_result.verified_claims)} 条，"
            f"证据覆盖率 {verify_result.evidence_coverage:.0%}"
        )
    return notes


# ============================================================
# 条件边
# ============================================================


def should_start_research(state: ResearchState) -> Literal["research", "refuse"]:
    """Planner 之后的路由。

    投资建议类问题会正常走检索。仅当计划显式标记 out_of_scope
    （预留字段，当前投资建议不再使用）时才跳过取数。
    """
    plan = state.get("plan")
    if plan is not None and getattr(plan, "out_of_scope", False):
        return "refuse"
    return "research"


def should_retrieve_more(state: ResearchState) -> Literal["retrieve_more", "write_report"]:
    """Verifier 之后的路由。纯函数，只读状态。"""
    config = get_config()
    if not state.get("needs_more_retrieval"):
        return "write_report"
    # 第二道保险：即使 Verifier 说要补搜，超过轮次上限也强制收敛。
    # LLM 参与的判断不能作为唯一的循环退出条件。
    if state.get("retrieval_count", 0) >= config.max_retrieval_rounds:
        logger.info(
            "已达最大检索轮次 %d，强制进入写作", config.max_retrieval_rounds
        )
        return "write_report"
    return "retrieve_more"


# ============================================================
# 图构建与运行
# ============================================================


def build_research_graph(context: ResearchGraphContext | None = None) -> Any:
    """构建并编译研究流程图。"""
    context = context or ResearchGraphContext()

    graph = StateGraph(ResearchState)
    graph.add_node("planner", _with_node_start("planner", make_planner_node(context)))
    graph.add_node("retriever", _with_node_start("retriever", make_retriever_node(context)))
    graph.add_node("verifier", _with_node_start("verifier", make_verifier_node(context)))
    graph.add_node("writer", _with_node_start("writer", make_writer_node(context)))

    graph.add_edge(START, "planner")
    # 超出职责范围的问题直接跳到 writer 出拒答报告，不走检索与核查
    graph.add_conditional_edges(
        "planner",
        should_start_research,
        {"research": "retriever", "refuse": "writer"},
    )
    graph.add_edge("retriever", "verifier")
    graph.add_conditional_edges(
        "verifier",
        should_retrieve_more,
        {"retrieve_more": "retriever", "write_report": "writer"},
    )
    graph.add_edge("writer", END)

    compiled = graph.compile()
    # 保留 context 引用，方便调用方在运行前 bind_tracer
    compiled.research_context = context  # type: ignore[attr-defined]
    return compiled


class ResearchPipeline:
    """研究流程的对外门面。

    封装"建 tracer → 建初始状态 → 跑图 → 收尾 trace"这套固定动作，
    让 API 层和 CLI 层不用各写一遍。
    """

    def __init__(self, context: ResearchGraphContext | None = None):
        self.context = context or ResearchGraphContext()
        self.graph = build_research_graph(self.context)

    def run(
        self,
        question: str,
        ticker: str = "",
        company: str = "",
        as_of_date: date | None = None,
        run_id: str | None = None,
        session_id: str = "",
        tracer: Tracer | None = None,
        on_node: Callable[[str, ResearchState], None] | None = None,
    ) -> dict[str, Any]:
        """跑完一次完整研究，返回 {state, trace_summary, run_id}。"""
        config = get_config()
        config.ensure_runtime_dirs()

        run_id = run_id or uuid.uuid4().hex[:12]
        active_tracer = tracer or Tracer()
        active_tracer.start_trace(run_id, question)
        self.context.bind_tracer(active_tracer)

        # 公司名缺失时补一次。放在这里而不是 Planner 里：
        # 它是纯数据查询，不需要 LLM，且 Planner 和 Writer 都要用。
        if not company and ticker:
            company = self._resolve_company_name(ticker)

        initial_state = create_initial_state(
            run_id=run_id,
            question=question,
            ticker=ticker,
            company=company,
            as_of_date=as_of_date or date.today(),
            session_id=session_id,
        )

        started = time.perf_counter()
        final_state: ResearchState = initial_state
        token = _node_start_cb.set(on_node)
        try:
            # recursion_limit 是 LangGraph 的硬保护：节点执行次数上限。
            # 三轮补搜最多产生 1+3+3+1=8 次节点执行，留一倍余量。
            # 进度由节点入口挂钩上报（见 _with_node_start），不在 stream 循环里报。
            for chunk in self.graph.stream(
                initial_state, config={"recursion_limit": 20}, stream_mode="values"
            ):
                final_state = chunk  # type: ignore[assignment]
        except Exception as exc:  # noqa: BLE001
            logger.exception("研究流程执行失败")
            final_state = {
                **final_state,
                "status": "failed",
                "errors": [*(final_state.get("errors") or []), f"graph: {exc}"],
                "report_markdown": final_state.get("report_markdown")
                or f"# 研究失败\n\n{type(exc).__name__}: {exc}",
            }
        finally:
            _node_start_cb.reset(token)

        duration_ms = int((time.perf_counter() - started) * 1000)
        trace_summary = active_tracer.end_trace(
            final_result=final_state.get("report_markdown", "")[:500],
            total_duration_ms=duration_ms,
            status=final_state.get("status", "unknown"),
        )

        logger.info("研究完成 run_id=%s summary=%s", run_id, state_summary(final_state))
        return {
            "run_id": run_id,
            "state": final_state,
            "trace_summary": trace_summary.model_dump(),
            "trace_path": getattr(active_tracer, "saved_path", None),
            "duration_ms": duration_ms,
        }

    @staticmethod
    def _resolve_company_name(ticker: str) -> str:
        try:
            from tools.akshare_tools import get_data_client

            return get_data_client().get_company_name(ticker)
        except Exception as exc:  # noqa: BLE001
            logger.warning("解析公司名失败，使用代码代替: %s", exc)
            return ticker


_PIPELINE: list[ResearchPipeline] = []


def get_pipeline() -> ResearchPipeline:
    """进程内共享的 Pipeline 单例。

    Agent、LLM 客户端、MCP Server、embedding 模型都在 Pipeline 构造时创建，
    每个请求各建一个会导致重复加载数百 MB 的模型。
    """
    if not _PIPELINE:
        _PIPELINE.append(ResearchPipeline())
    return _PIPELINE[0]


__all__ = [
    "ResearchGraphContext",
    "ResearchPipeline",
    "build_research_graph",
    "get_pipeline",
    "should_retrieve_more",
    "should_start_research",
]
