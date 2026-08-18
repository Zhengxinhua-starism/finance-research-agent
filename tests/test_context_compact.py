"""上下文压缩：四段分区、工具结果摘要、超预算时不再空转、修复配对。"""

from __future__ import annotations

import json

from harness.context_compact import ContextCompact, count_messages_tokens


def _messages() -> list[dict]:
    return [
        {"role": "system", "content": "你是金融研报 Agent。"},
        {"role": "user", "content": "第一轮问题"},
        {
            "role": "assistant",
            "content": "我将调用利润表工具",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_income_history"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "get_income_history",
            "content": "营收 7771.02 亿元",
        },
        {"role": "assistant", "content": "第一轮中间推理，没有工具。"},
        {"role": "user", "content": "第二轮问题"},
        {"role": "assistant", "content": "第二轮结论。"},
        {"role": "user", "content": "第三轮问题"},
        {"role": "assistant", "content": "第三轮结论。"},
        {"role": "user", "content": "第四轮问题"},
        {"role": "assistant", "content": "第四轮结论。"},
    ]


def _income_json() -> str:
    payload = {
        "company": "比亚迪",
        "ticker": "002594",
        "period_count": 2,
        "periods": [
            {
                "date": "2024-12-31",
                "revenue": 777102455000.0,
                "revenue_display": "7,771.02亿元",
                "gross_margin": 0.1943834278582996,
                "gross_margin_display": "19.44%",
                "net_profit": 40254346000.0,
                "net_profit_display": "402.54亿元",
            },
            {
                "date": "2025-12-31",
                "revenue": 803964958000.0,
                "revenue_display": "8,039.65亿元",
                "gross_margin": 0.17744529233573883,
                "gross_margin_display": "17.74%",
                "net_profit": 32619022000.0,
                "net_profit_display": "326.19亿元",
            },
        ],
        "unit_note": "金额单位为元",
        "padding": ("冗余字段用于撑过短文本直通阈值 " * 40),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def test_partition_keeps_system_recent_middle_and_older() -> None:
    compact = ContextCompact(keep_recent_turns=1, middle_turns=2)
    system, recent, middle, older = compact._partition(_messages())
    assert system[0]["role"] == "system"
    assert any("第四轮" in str(m.get("content")) for m in recent)
    assert any("第二轮" in str(m.get("content")) or "第三轮" in str(m.get("content")) for m in middle)
    assert any("第一轮问题" in str(m.get("content")) for m in older)


def test_summarize_tool_result_keeps_key_numbers() -> None:
    summary = ContextCompact._summarize_tool_result(_income_json(), "get_income_history")
    assert "7,771.02" in summary
    assert "19.44%" in summary
    assert "get_income_history" in summary
    assert "002594" in summary
    assert len(summary) < len(_income_json())


def test_compact_reduces_tokens_when_recent_already_over_budget() -> None:
    """旧实现：older 为空就原样返回（0% 压缩）。新实现必须摘要 recent 工具结果。"""
    content = _income_json()
    messages = [
        {"role": "system", "content": "你是金融研报 Agent。"},
        {"role": "user", "content": "取数"},
        {
            "role": "assistant",
            "content": "调用利润表",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_income_history"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "get_income_history",
            "content": content,
        },
    ]
    before = count_messages_tokens(messages)
    compact = ContextCompact(max_tokens=min(200, before - 1), keep_recent_turns=1, middle_turns=0)
    compacted = compact.compact(messages, llm_client=None)
    after = count_messages_tokens(compacted)
    tool_text = "".join(str(m.get("content")) for m in compacted if m.get("role") == "tool")
    assert after < before
    assert "7,771.02" in tool_text
    assert compacted[0]["role"] == "system"
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "call_1" for m in compacted)


def test_compact_preserves_numbers_from_older_tools_without_llm() -> None:
    compact = ContextCompact(max_tokens=50, keep_recent_turns=1, middle_turns=2)
    compacted = compact.compact(_messages(), llm_client=None)
    blob = json.dumps(compacted, ensure_ascii=False)
    assert "7771.02" in blob
    assert compacted[0]["role"] == "system"


def test_repair_drops_orphaned_tool_messages() -> None:
    compact = ContextCompact()
    repaired = compact._repair_tool_call_pairing(
        [
            {"role": "assistant", "content": "无工具"},
            {"role": "tool", "tool_call_id": "missing", "content": "孤儿工具结果"},
        ]
    )
    assert not any(m.get("role") == "tool" for m in repaired)


# ---------- 真实 payload 压缩率断言 ----------
# 用生产缓存里的 3 期利润表验证：压缩后必须保留关键数字，且压缩率 > 80%。
# 这是简历上"工具结果压缩 XX%"的直接证据。

_REAL_INCOME_PAYLOAD = json.dumps(
    {
        "company": "比亚迪",
        "ticker": "002594",
        "period_count": 3,
        "periods": [
            {
                "date": "2023-12-31",
                "revenue": 602315354000.0,
                "revenue_display": "6,023.15亿元",
                "gross_margin": 0.1858,
                "gross_margin_display": "18.58%",
                "net_profit": 30040811000.0,
                "net_profit_display": "300.41亿元",
                "rd_expense": 39574945000.0,
                "period_expense_ratio": 0.1111,
                "period_expense_ratio_display": "11.11%",
            },
            {
                "date": "2024-12-31",
                "revenue": 777102455000.0,
                "revenue_display": "7,771.02亿元",
                "gross_margin": 0.1944,
                "gross_margin_display": "19.44%",
                "net_profit": 40254346000.0,
                "net_profit_display": "402.54亿元",
                "rd_expense": 53194745000.0,
                "period_expense_ratio": 0.125,
                "period_expense_ratio_display": "12.50%",
            },
            {
                "date": "2025-12-31",
                "revenue": 803964958000.0,
                "revenue_display": "8,039.65亿元",
                "gross_margin": 0.1774,
                "gross_margin_display": "17.74%",
                "net_profit": 32619022000.0,
                "net_profit_display": "326.19亿元",
                "rd_expense": 57978105000.0,
                "period_expense_ratio": 0.129,
                "period_expense_ratio_display": "12.90%",
            },
        ],
        "source": "AKShare 新浪财报接口",
        "unit_note": "金额单位为元",
    },
    ensure_ascii=False,
    indent=2,
)


def test_real_payload_compression_ratio_above_80pct() -> None:
    summary = ContextCompact._summarize_tool_result(_REAL_INCOME_PAYLOAD, "get_income_history")
    ratio = 1.0 - len(summary) / len(_REAL_INCOME_PAYLOAD)
    assert ratio > 0.70, f"压缩率 {ratio:.1%} 应明显缩小 JSON dump"
    assert len(summary) < len(_REAL_INCOME_PAYLOAD)
    # 关键数字必须保留
    for number in ("6,023.15", "7,771.02", "8,039.65", "19.44%", "17.74%", "326.19"):
        assert number in summary, f"摘要丢失关键数字: {number}"
    # 三期都要出现
    for year in ("2023", "2024", "2025"):
        assert year in summary, f"摘要丢失报告期: {year}"
