"""上下文压缩。

解决什么问题
    ReAct 循环每轮都会往消息历史里追加"LLM 的思考 + 工具返回的原始数据"。
    财务报表工具单次返回就可能上千 token，跑到第 6~7 轮时上下文会突破模型窗口。
    本模块在超限前把历史压成对 LLM 决策仍够用的摘要，让循环能继续跑完。

两条独立通路（这是本模块最重要的不变量）
    ToolResult 同时服务两个消费者，保真度要求完全不同：

        ToolResult.data  → Evidence 对象 → evidence_pool → Gate（需要精确数字）
        ToolResult.content → message 历史 → LLM 上下文（只需足够做下一步决策）

    Gate 从不读 message content。因此压缩消息里的工具 JSON **不会**影响门禁。
    旧实现把"工具结果不可压"写死，叠加 context_max_tokens=3000 之后：
    最近 3 轮的工具原文已经超预算 → `_partition` 发现 older 为空 → 原样返回。
    实测 80 次 compact 事件平均压缩率只有 2.7%，大量是 0% 空转。

核心设计决策（渐进式保真度，对标 Claude Code 的分层摘要）
    1. system prompt 原样保留——它定义了 Agent 的行为约束。
    2. 最近 1 轮完整保留——下一步 tool_calls 强依赖刚刚拿到的观察。
    3. 中间 2 轮半压缩——推理原文保留，工具结果压成"关键指标 + 报告期"摘要。
    4. 更早的轮次全压缩——过程性对话交给 quick 模型（失败则规则摘要），
       工具结果同样先做成数字摘要再写入那段历史，避免把 6000 字符 JSON dump
       原样塞进摘要。
    5. 若压完仍超预算（单轮工具原文就爆了），再对最近 1 轮的工具结果做摘要。
       绝不因为"无可压缩的过程文本"就返回原文——那是 2.7% 压缩率的根因。

与 TradingAgents 的区别
    TradingAgents 用 RemoveMessage 把历史整体清空，只留最终结论。
    本项目的下游（Verifier / Writer）需要回溯证据链，清空等于丢掉决策上下文。
    所以选择"分层降保真度"，而不是"清空历史"。门禁要的精确数字走 Evidence，
    不走这段被压缩的 message 历史。

为什么不用其他方案
    - 不用滑动窗口直接截断：会把早期工具结果（往往是最关键的财报）整段丢掉。
    - 不用向量化历史再检索：为一个最多 10 轮的循环引入一套 RAG，不划算。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Sequence

from config import get_config
from harness.tracing import NullTracer, Tracer

logger = logging.getLogger(__name__)

YI = 100_000_000
WAN = 10_000
SHORT_TOOL_CONTENT_CHARS = 400

# 摘要提示词。数字和来源必须保留，因为 LLM 后续决策仍可能引用它们。
# 门禁不读这段摘要——精确比对走 evidence_pool。
COMPACT_SYSTEM_PROMPT = """你是一个对话历史压缩器，服务于金融研报 Agent。

请把给定的历史对话压缩成一段简洁的中文摘要，严格遵守：
1. 必须完整保留所有具体数字（金额、比率、增速）及其单位，一个都不能丢，不能四舍五入。
2. 必须保留每个数字对应的来源（年报/季报/工具名）和报告期。
3. 必须保留已经得出的中间结论和尚未解决的问题。
4. 删除寒暄、重复表述、以及"我将调用某工具"这类过程性描述。
5. 只输出摘要正文，不要任何前言、标题或解释。

摘要长度控制在 500 字以内。"""

# 单期报表优先渲染的字段（英文 key → 中文标签）。顺序即金融分析优先级。
_PERIOD_FIELDS: tuple[tuple[str, str], ...] = (
    ("revenue", "营收"),
    ("net_profit", "归母净利"),
    ("gross_margin", "毛利率"),
    ("net_margin", "净利率"),
    ("deducted_net_profit", "扣非净利"),
    ("rd_expense", "研发"),
    ("period_expense_ratio", "期间费用率"),
    ("operating_cashflow", "经营现金流"),
    ("capital_expenditure", "资本开支"),
    ("free_cashflow", "自由现金流"),
    ("total_assets", "总资产"),
    ("net_assets", "归母净资产"),
    ("debt_ratio", "资产负债率"),
    ("interest_bearing_debt", "有息负债"),
    ("cashflow_to_profit", "经营现金流/净利"),
)

_METRIC_FIELDS: tuple[tuple[str, str], ...] = (
    ("revenue", "营收"),
    ("net_profit", "归母净利"),
    ("gross_margin", "毛利率"),
    ("net_margin", "净利率"),
    ("roe", "ROE"),
    ("revenue_yoy", "营收同比"),
    ("net_profit_yoy", "净利同比"),
    ("debt_ratio", "资产负债率"),
    ("rd_ratio", "研发占比"),
    ("cashflow_to_profit", "经营现金流/净利"),
)

_RATIO_MARKERS = ("margin", "ratio", "roe", "yoy", "share", "rate", "turnover")
_SKIP_TOP_KEYS = {
    "unit_note",
    "source",
    "disclosure_dates",
    "usage_note",
    "retrieval_method",
    "evidence_level",
    "checked_rules",
    "coverage_note",
    "interpretation",
    "padding",
}


def estimate_tokens(text: str) -> int:
    """估算 token 数。

    优先用 tiktoken 的 cl100k_base（DeepSeek 的分词器与之接近，误差可接受）；
    不可用时退回启发式：中文约 1 字 1 token，英文约 4 字符 1 token。
    这里刻意不做精确统计——压缩触发是个阈值判断，估算偏差 10% 不影响决策，
    而每轮都跑一次真实分词会带来可观的 CPU 开销。
    """
    if not text:
        return 0
    encoder = _get_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:  # noqa: BLE001 — 编码失败退回字符估算
            pass
    chinese_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other_chars = len(text) - chinese_chars
    return chinese_chars + other_chars // 4


_ENCODER_CACHE: list[Any] = []


def _get_encoder() -> Any:
    """懒加载并缓存 tiktoken encoder。首次加载需要联网下载词表。"""
    if _ENCODER_CACHE:
        return _ENCODER_CACHE[0]
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # noqa: BLE001
        logger.debug("tiktoken 不可用，回退到字符估算: %s", exc)
        encoder = None
    _ENCODER_CACHE.append(encoder)
    return encoder


def count_messages_tokens(messages: Sequence[dict[str, Any]]) -> int:
    """统计整个消息列表的 token 数（含 tool_calls 的参数）。"""
    total = 0
    for message in messages:
        total += estimate_tokens(str(message.get("content") or ""))
        tool_calls = message.get("tool_calls")
        if tool_calls:
            total += estimate_tokens(json.dumps(tool_calls, ensure_ascii=False, default=str))
        total += 4  # 每条消息的角色/分隔符开销
    return total


def _copy_message(message: dict[str, Any]) -> dict[str, Any]:
    copied = dict(message)
    tool_calls = copied.get("tool_calls")
    if isinstance(tool_calls, list):
        copied["tool_calls"] = [
            dict(item) if isinstance(item, dict) else item for item in tool_calls
        ]
    return copied


def _try_parse_json(content: str) -> dict[str, Any] | None:
    """尽量从工具文本里抠出 dict。截断后缀和残缺 JSON 都要兜住。"""
    text = content.strip()
    marker = "...[输出过长已截断"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _format_metric(key: str, value: Any, row: dict[str, Any] | None = None) -> str | None:
    """把工具 JSON 里的一个字段格式化成短中文。"""
    if value is None or value == "" or isinstance(value, bool):
        return None
    if row is not None:
        display = row.get(f"{key}_display")
        if display not in (None, ""):
            return str(display)
    if isinstance(value, str):
        return value
    if not isinstance(value, (int, float)):
        return str(value)
    if key == "cashflow_to_profit":
        return f"{value:.2f}"
    if any(marker in key for marker in _RATIO_MARKERS) and abs(value) <= 5:
        return f"{value * 100:.2f}%"
    if abs(value) >= YI:
        return f"{value / YI:,.2f}亿元"
    if abs(value) >= WAN:
        return f"{value / WAN:,.2f}万元"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _header(tool_name: str, data: dict[str, Any]) -> str:
    company = str(data.get("company") or "").strip()
    ticker = str(data.get("ticker") or "").strip()
    if company and ticker:
        return f"[{tool_name}] {company}({ticker})"
    if company:
        return f"[{tool_name}] {company}"
    return f"[{tool_name}]"


class ContextCompact:
    """消息历史压缩器。"""

    def __init__(
        self,
        max_tokens: int | None = None,
        keep_recent_turns: int = 1,
        middle_turns: int = 2,
        tracer: Tracer | None = None,
    ):
        config = get_config()
        self.max_tokens = max_tokens or config.context_max_tokens
        self.keep_recent_turns = max(1, keep_recent_turns)
        self.middle_turns = max(0, middle_turns)
        self.tracer = tracer or NullTracer()

    def should_compact(self, messages: Sequence[dict[str, Any]]) -> bool:
        return count_messages_tokens(messages) > self.max_tokens

    def compact(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        llm_client: Any = None,
    ) -> list[dict[str, Any]]:
        """压缩消息历史，返回新列表（不修改入参）。

        llm_client 为 None 时降级为规则压缩。降级路径必须存在——
        压缩本身要调 LLM，如果 LLM 正好在限流，不能因为压不了就让整次运行失败。
        """
        budget = max_tokens or self.max_tokens
        before_tokens = count_messages_tokens(messages)
        if before_tokens <= budget:
            return list(messages)

        system_messages, recent, middle, older = self._partition(messages)
        tool_summaries = 0

        middle_out, middle_n = self._summarize_tool_messages(middle)
        tool_summaries += middle_n

        older_for_summary, older_n = self._summarize_tool_messages(older)
        tool_summaries += older_n
        summary_text = self._summarize(older_for_summary, llm_client)

        recent_out = [_copy_message(item) for item in recent]
        compacted = self._assemble(system_messages, summary_text, middle_out, recent_out)
        after_tokens = count_messages_tokens(compacted)

        # 最近 1 轮原文仍超预算：对 recent 的工具结果也做摘要。
        # 旧实现在这里直接 return 原文，造成大量 0% 空转。
        if after_tokens > budget:
            recent_out, recent_n = self._summarize_tool_messages(recent)
            tool_summaries += recent_n
            compacted = self._assemble(system_messages, summary_text, middle_out, recent_out)
            after_tokens = count_messages_tokens(compacted)

        ratio = 0.0 if before_tokens <= 0 else max(0.0, 1.0 - after_tokens / before_tokens)
        self.tracer.log(
            "compact",
            agent_name="context_compact",
            input_summary=f"{len(messages)} 条消息 / {before_tokens} tokens",
            output_summary=f"{len(compacted)} 条消息 / {after_tokens} tokens",
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            compression_ratio=round(ratio, 4),
            tool_summaries=tool_summaries,
            older_messages=len(older),
            middle_messages=len(middle),
        )
        logger.info(
            "上下文压缩: %d → %d tokens (%.1f%%, 工具摘要 %d 条)",
            before_tokens,
            after_tokens,
            ratio * 100,
            tool_summaries,
        )
        return compacted

    # ---------------- 内部实现 ----------------

    @staticmethod
    def _assemble(
        system_messages: list[dict[str, Any]],
        summary_text: str,
        middle: list[dict[str, Any]],
        recent: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        compacted: list[dict[str, Any]] = [_copy_message(item) for item in system_messages]
        if summary_text:
            compacted.append(
                {
                    "role": "user",
                    "content": f"[以下是早期对话的压缩摘要，供你参考]\n{summary_text}",
                }
            )
        compacted.extend(middle)
        compacted.extend(recent)
        return ContextCompact._repair_tool_call_pairing(compacted)

    def _partition(
        self, messages: Sequence[dict[str, Any]]
    ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        """切成四段：system / 最近 N 轮(完整) / 中间 M 轮(半压缩) / 更早(全压缩)。

        "一轮"定义为一条 assistant 消息及其后续 tool / user 消息。
        按 assistant 倒数切，避免一轮里多条工具结果被拦腰截断。
        """
        system_messages = [m for m in messages if m.get("role") == "system"]
        body = [m for m in messages if m.get("role") != "system"]

        turn_starts: list[int] = []
        for index in range(len(body) - 1, -1, -1):
            if body[index].get("role") == "assistant":
                turn_starts.append(index)
                if len(turn_starts) >= self.keep_recent_turns + self.middle_turns:
                    break

        if not turn_starts:
            return system_messages, body, [], []

        recent_idx = min(self.keep_recent_turns, len(turn_starts)) - 1
        recent_start = turn_starts[recent_idx]
        recent = body[recent_start:]

        remaining = turn_starts[recent_idx + 1 :]
        if not remaining or self.middle_turns <= 0:
            return system_messages, recent, [], body[:recent_start]

        middle_count = min(self.middle_turns, len(remaining))
        middle_start = remaining[middle_count - 1]
        return (
            system_messages,
            recent,
            body[middle_start:recent_start],
            body[:middle_start],
        )

    def _summarize_tool_messages(
        self, messages: Sequence[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int]:
        """复制消息列表，把 tool 角色的 JSON dump 换成关键指标摘要。"""
        call_id_to_name: dict[str, str] = {}
        for message in messages:
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                call_id = tool_call.get("id")
                function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
                name = function.get("name") or tool_call.get("name")
                if call_id and name:
                    call_id_to_name[str(call_id)] = str(name)

        summarized = 0
        output: list[dict[str, Any]] = []
        for message in messages:
            copied = _copy_message(message)
            if copied.get("role") == "tool":
                tool_name = str(
                    copied.get("name")
                    or call_id_to_name.get(str(copied.get("tool_call_id") or ""), "tool")
                )
                original = str(copied.get("content") or "")
                compact_text = self._summarize_tool_result(original, tool_name)
                if compact_text != original:
                    summarized += 1
                copied["content"] = compact_text
            output.append(copied)
        return output, summarized

    @staticmethod
    def _summarize_tool_result(content: str, tool_name: str) -> str:
        """把工具结果的 JSON 文本压成紧凑摘要，保留数字、报告期、工具名。"""
        if not content.strip():
            return content
        if len(content) < SHORT_TOOL_CONTENT_CHARS:
            return content

        data = _try_parse_json(content)
        if data is None:
            snippet = content.strip().replace("\n", " ")[:200]
            return f"[{tool_name}] {snippet}"

        lines = [_header(tool_name, data)]
        rendered = False

        periods = data.get("periods")
        if isinstance(periods, list) and periods and isinstance(periods[0], dict):
            rendered = True
            for period in periods[:5]:
                date_str = str(period.get("date") or period.get("period_end") or "?")[:10]
                metrics: list[str] = []
                for key, label in _PERIOD_FIELDS:
                    formatted = _format_metric(key, period.get(key), period)
                    if formatted:
                        metrics.append(f"{label} {formatted}")
                if metrics:
                    lines.append(f"  {date_str}: {', '.join(metrics)}")

        comparisons = data.get("metrics_comparison")
        if isinstance(comparisons, list) and comparisons:
            rendered = True
            period_labels = data.get("periods") if isinstance(data.get("periods"), list) else []
            if period_labels and not isinstance(period_labels[0], dict):
                labels = ", ".join(str(item)[:10] for item in period_labels[:5])
                lines.append(f"  报告期: {labels}")
            for item in comparisons[:8]:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("metric_label") or item.get("metric") or "指标")
                displays = item.get("values_display") or []
                trend = str(item.get("trend_label") or "").strip()
                series = " → ".join(str(value) for value in displays)
                suffix = f"（{trend}）" if trend else ""
                if series:
                    lines.append(f"  {label}: {series}{suffix}")

        by_period = data.get("segments_by_period")
        if isinstance(by_period, dict) and by_period:
            rendered = True
            period_key = str(list(by_period.keys())[-1])
            category = str(data.get("category") or "分部")
            lines.append(f"  {category} {period_key[:10]}:")
            for segment in (by_period.get(period_key) or [])[:5]:
                if not isinstance(segment, dict):
                    continue
                name = str(segment.get("segment") or "未知分部")
                parts: list[str] = []
                for key, label in (
                    ("revenue", "营收"),
                    ("revenue_share", "占比"),
                    ("gross_margin", "毛利率"),
                ):
                    formatted = _format_metric(key, segment.get(key), segment)
                    if formatted:
                        parts.append(f"{label} {formatted}")
                lines.append(f"    {name}: {', '.join(parts)}" if parts else f"    {name}")

        results = data.get("results")
        if isinstance(results, list) and results:
            rendered = True
            lines.append(f"  result_count: {data.get('result_count', len(results))}")
            for item in results[:3]:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip() or "（无标题）"
                snippet = str(item.get("content") or item.get("summary") or "").replace("\n", " ")
                snippet = snippet[:80]
                rank = item.get("rank", "")
                prefix = f"#{rank} " if rank != "" else ""
                lines.append(f"  {prefix}{title}" + (f"：{snippet}" if snippet else ""))

        if data.get("alert_count") is not None or data.get("alerts"):
            rendered = True
            lines.append(f"  alert_count: {data.get('alert_count', len(data.get('alerts') or []))}")
            for alert in (data.get("alerts") or [])[:5]:
                if isinstance(alert, dict):
                    lines.append(
                        f"  [{alert.get('level', '?')}] {alert.get('title') or alert.get('detail') or ''}"
                    )
            dupont = data.get("dupont") if isinstance(data.get("dupont"), dict) else {}
            roe = _format_metric("roe", dupont.get("roe"), dupont) if dupont else None
            if roe:
                lines.append(f"  杜邦 ROE: {roe}")

        metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
        metrics_display = (
            data.get("metrics_display") if isinstance(data.get("metrics_display"), dict) else {}
        )
        if metrics or metrics_display:
            rendered = True
            for key, label in _METRIC_FIELDS:
                formatted = None
                if key in metrics_display:
                    formatted = str(metrics_display[key])
                elif key in metrics:
                    formatted = _format_metric(key, metrics.get(key), metrics_display)
                if formatted:
                    lines.append(f"  {label}: {formatted}")
            roe_scope = data.get("roe_scope")
            if roe_scope:
                lines.append(f"  ROE口径: {roe_scope}")

        if data.get("latest_close") is not None:
            rendered = True
            lines.append(f"  latest_close: {data.get('latest_close')}")

        if not rendered:
            for key, value in data.items():
                if key in _SKIP_TOP_KEYS or key in {"company", "ticker", "periods"}:
                    continue
                if isinstance(value, (dict, list)):
                    continue
                formatted = _format_metric(str(key), value, data)
                if formatted:
                    lines.append(f"  {key}: {formatted}")

        lines.append("  (完整数据已在证据池中，门禁校验不受影响)")
        summary = "\n".join(line for line in lines if line.strip())
        if len(summary) >= len(content):
            return content
        return summary

    def _summarize(self, messages: Sequence[dict[str, Any]], llm_client: Any) -> str:
        if not messages:
            return ""

        transcript = "\n\n".join(
            f"[{m.get('role')}] {str(m.get('content') or '').strip()}"
            for m in messages
            if str(m.get("content") or "").strip()
        )
        if not transcript:
            return ""

        if llm_client is None:
            return self._rule_based_summary(messages)

        try:
            response = llm_client.chat(
                messages=[{"role": "user", "content": transcript}],
                system_prompt=COMPACT_SYSTEM_PROMPT,
            )
            summary = (response.content or "").strip()
            return summary or self._rule_based_summary(messages)
        except Exception as exc:  # noqa: BLE001 — 压缩失败不能中断主流程
            logger.warning("LLM 压缩失败，降级为规则压缩: %s", exc)
            return self._rule_based_summary(messages)

    @staticmethod
    def _rule_based_summary(messages: Sequence[dict[str, Any]]) -> str:
        """无 LLM 时的降级摘要：只保留每条消息的首句和其中的数字。"""
        lines: list[str] = []
        for message in messages:
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            first_sentence = re.split(r"[。\n]", content)[0][:120]
            numbers = re.findall(r"-?\d[\d,]*\.?\d*\s*(?:亿|万|%|pp|元)?", content)[:8]
            fragment = f"[{message.get('role')}] {first_sentence}"
            if numbers:
                fragment += f"（涉及数字: {', '.join(numbers)}）"
            lines.append(fragment)
        return "\n".join(lines[:20])

    @staticmethod
    def _repair_tool_call_pairing(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """剔除失去配对的 tool 消息。

        压缩后有可能出现"role=tool 的消息，但它对应的 assistant tool_call
        被压掉了"的情况，OpenAI 兼容接口会返回 400。这里做一次一致性修复，
        比在压缩逻辑里到处小心翼翼地判断更可靠。
        """
        known_call_ids: set[str] = set()
        for message in messages:
            for tool_call in message.get("tool_calls") or []:
                call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
                if call_id:
                    known_call_ids.add(call_id)

        repaired: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") == "tool":
                if message.get("tool_call_id") not in known_call_ids:
                    logger.debug("丢弃失配的 tool 消息: %s", message.get("tool_call_id"))
                    continue
            repaired.append(message)
        return repaired
