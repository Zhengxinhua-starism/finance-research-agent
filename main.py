"""统一入口：CLI / FastAPI（托管 Vue 演示页）。

解决什么问题
    同一套 Agent 需要两种使用方式：开发时用 CLI 快速验证、
    演示时跑 API（Vue 前端同源托管）。两个入口如果各写一份
    启动逻辑，配置初始化和日志设置就会不一致。

核心设计决策
    1. **Vue 只通过 HTTP/SSE 调 FastAPI，绝不 import Agent 代码。**
       耦合之后 UI 构建会拖进 embedding 模型，而且前端崩溃会带走 Agent。
    2. 研究走 async_mode + SSE。一次研究 30~150 秒，节点切换由
       `/api/session/{id}/events` 推给前端，不靠 2 秒轮询。
    3. CLI 直接调 Pipeline，不经过 HTTP。本地调试不必多起一个服务进程。
    4. 重依赖（uvicorn、Pipeline）延迟导入。
       `python main.py --help` 不应该等 30 秒去加载 torch。

为什么不用其他方案
    - 不用第二套 Web（Gradio）：面试会被问为什么要两套界面。
      CLI 调试、Vue 演示，职责已经分清。
    - 不用 typer / click：子命令少，argparse 够用。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from config import get_config

logger = logging.getLogger(__name__)

BANNER = r"""
  ______ _                            ___                   _
 |  ____(_)                          / _ \                 | |
 | |__   _ _ __   __ _ _ __   ___ ___ /_\ | __ _  ___ _ __ | |_
 |  __| | | '_ \ / _` | '_ \ / __/ _ \  _  |/ _` |/ _ \ '_ \| __|
 | |    | | | | | (_| | | | | (_|  __/ | | | (_| |  __/ | | | |_
 |_|    |_|_| |_|\__,_|_| |_|\___\___\_| |_|\__, |\___|_| |_|\__|
                                             __/ |
  金融研报 Agent  ·  自研 Harness + Agentic RAG + MCP    |___/
"""


def setup_logging(level: str | None = None) -> None:
    config = get_config()
    logging.basicConfig(
        level=getattr(logging, (level or config.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def run_cli(args: argparse.Namespace) -> int:
    """命令行单次研究。"""
    from graph.graph import ResearchPipeline

    config = get_config()
    config.ensure_runtime_dirs()

    if not config.deepseek_api_key:
        print("错误：未配置 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 并填入 API Key。")
        return 1

    as_of = date.fromisoformat(args.as_of_date) if args.as_of_date else date.today()

    print(BANNER)
    print(f"问题：{args.question}")
    print(f"标的：{args.company}")
    print(f"基准日：{as_of.isoformat()}")
    print("-" * 70)

    started = time.perf_counter()
    pipeline = ResearchPipeline()

    def on_node(node_name: str, state: dict[str, Any]) -> None:
        if node_name:
            print(
                f"  [{node_name}] 证据 {len(state.get('evidence_pool') or [])} 条 | "
                f"检索轮次 {state.get('retrieval_count', 0)}"
            )

    outcome = pipeline.run(
        question=args.question,
        ticker=args.company,
        as_of_date=as_of,
        on_node=on_node,
    )

    state = outcome["state"]
    summary = outcome["trace_summary"]

    print("-" * 70)
    from agents.writer import WriterAgent

    print(WriterAgent.strip_report_html(state.get("report_markdown", "（未生成研报）")))
    print("-" * 70)
    print(
        f"run_id={outcome['run_id']} | 耗时 {(time.perf_counter() - started):.1f}s | "
        f"tokens={summary.get('total_tokens', 0)} | "
        f"LLM 调用 {summary.get('llm_call_count', 0)} 次 | "
        f"工具调用 {summary.get('tool_call_count', 0)} 次"
        f"（失败 {summary.get('tool_failure_count', 0)}）"
    )
    if outcome.get("trace_path"):
        print(f"trace: {outcome['trace_path']}")
    if state.get("errors"):
        print("\n过程中的异常：")
        for error in state["errors"]:
            print(f"  - {error}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".json":
            output_path.write_text(
                json.dumps(state.get("report_dict") or {}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            output_path.write_text(state.get("report_markdown", ""), encoding="utf-8")
        print(f"研报已保存到 {output_path}")

    return 0 if state.get("status") == "completed" else 2


def run_api(args: argparse.Namespace) -> int:
    import uvicorn

    config = get_config()
    print(BANNER)
    print(f"API 服务启动中... http://{config.api_host}:{args.port}")
    print(f"演示前端: http://localhost:{args.port}/  （需先 npm run build — web/dist）")
    print(f"接口文档: http://localhost:{args.port}/docs")
    print(f"健康检查: http://localhost:{args.port}/api/health")

    uvicorn.run(
        "api.app:app",
        host=config.api_host,
        port=args.port,
        reload=args.reload,
        log_level=config.log_level.lower(),
    )
    return 0


def run_prepare(args: argparse.Namespace) -> int:
    from rag.prepare_data import prepare_knowledge_base

    stats = prepare_knowledge_base(reset=args.reset)
    print("知识库构建完成：")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    return 0


def run_eval_cli(args: argparse.Namespace) -> int:
    from eval.evaluate import Evaluator

    evaluator = Evaluator()
    case_ids = [int(item) for item in args.cases.split(",")] if args.cases else None
    outcome = evaluator.run(case_ids=case_ids, save_report=True)
    print(json.dumps(outcome["summary"], ensure_ascii=False, indent=2))
    if outcome.get("report_path"):
        print(f"\n详细结果: {outcome['report_path']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finance-research-agent",
        description="金融研报 Agent — CLI 调试 / API + Vue 演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  python main.py --prepare-data
  python main.py --question "比亚迪2024年毛利率为什么下降" --company 002594
  python main.py --api
  python main.py --eval --cases 1,2,3

演示前端：先在 web/ 执行 npm install && npm run build，再启动 --api，
浏览器打开 http://localhost:<API_PORT>/ 。开发时可用 npm run dev（Vite 5173）。
        """,
    )

    mode = parser.add_argument_group("运行模式")
    mode.add_argument("--api", action="store_true", help="启动 FastAPI（托管 Vue 构建产物）")
    mode.add_argument("--prepare-data", action="store_true", help="初始化 RAG 知识库")
    mode.add_argument("--eval", action="store_true", help="运行自动化评测")

    cli_group = parser.add_argument_group("CLI 模式参数")
    cli_group.add_argument("-q", "--question", type=str, help="研究问题")
    cli_group.add_argument("-c", "--company", type=str, default="", help="A股代码，6位数字")
    cli_group.add_argument("--as-of-date", type=str, default="", help="分析基准日 YYYY-MM-DD")
    cli_group.add_argument("-o", "--output", type=str, help="研报输出路径（.md 或 .json）")

    server_group = parser.add_argument_group("服务参数")
    server_group.add_argument("--port", type=int, default=0, help="API 端口（默认读配置，常为 8000）")
    server_group.add_argument("--reload", action="store_true", help="API 热重载（开发用）")

    other_group = parser.add_argument_group("其他")
    other_group.add_argument("--reset", action="store_true", help="初始化知识库时先清空")
    other_group.add_argument("--cases", type=str, default="", help="评测用例 ID，逗号分隔")
    other_group.add_argument("--log-level", type=str, default="", help="日志级别")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level or None)
    config = get_config()

    if args.prepare_data:
        return run_prepare(args)
    if args.eval:
        return run_eval_cli(args)
    if args.api:
        args.port = args.port or config.api_port
        return run_api(args)
    if args.question:
        if not args.company:
            print("错误：CLI 模式需要同时提供 --company（6 位 A 股代码）")
            return 1
        return run_cli(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
