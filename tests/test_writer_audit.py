"""Writer 正文门禁与 ⚠️ 线索来源。"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any

from agents.writer import (
    DISCLAIMER,
    WriterAgent,
    _AUDIT_MAJORITY_TIP,
    _BLOCKED_PLACEHOLDER,
    _UNVERIFIED_MARK,
)
from harness.types import Evidence


def _writer() -> WriterAgent:
    return WriterAgent(llm_client=object())  # type: ignore[arg-type]


def _verified(**numbers: float) -> Any:
    return SimpleNamespace(verdict="verified", numbers=dict(numbers))


def _unverified(text: str, source: str | None = None, reason: str = "", **numbers: float) -> Any:
    return SimpleNamespace(
        verdict="unverified",
        label="⚠️未验证",
        text=text,
        source=source,
        reason=reason,
        section="conclusion",
        gate_status="unverified",
        numbers=dict(numbers),
    )


def _news(title: str, content: str) -> Evidence:
    return Evidence(
        source_type="news",
        source_name=title,
        disclosure_date=date(2026, 8, 14),
        content=content,
        numbers={},
        ticker="600519",
        company="贵州茅台",
    )


def test_verified_numbers_are_not_flagged() -> None:
    agent = _writer()
    body = "2025 年营收 7,771.02亿元，毛利率 17.74%。"
    unmatched = agent._audit_analysis_numbers(
        body,
        [_verified(revenue=777_102_000_000.0, gross_margin=0.1774)],
    )
    assert unmatched == []


def test_unverified_body_numbers_are_flagged() -> None:
    agent = _writer()
    body = "营收 7,771.02亿元，研发投入 500亿元。"
    unmatched = agent._audit_analysis_numbers(
        body,
        [_verified(revenue=777_102_000_000.0)],
    )
    assert any("500" in item for item in unmatched)
    assert not any("7,771.02" in item for item in unmatched)


def test_empty_or_refused_claims_skip_audit() -> None:
    agent = _writer()
    body = "营收 100亿元，净利率 50%。"
    assert agent._audit_analysis_numbers(body, []) == []
    refused = SimpleNamespace(verdict="refused", numbers={"revenue": 10_000_000_000.0})
    assert agent._audit_analysis_numbers(body, [refused]) == []


def test_years_and_tickers_are_not_flagged() -> None:
    agent = _writer()
    body = "2024年和2025年，股票代码 002594，营收 7,771.02亿元。"
    unmatched = agent._audit_analysis_numbers(
        body,
        [_verified(revenue=777_102_000_000.0)],
    )
    assert unmatched == []


def test_yuan_converts_to_yi_yuan_display() -> None:
    agent = _writer()
    body = "归母净利润 326.19亿元。"
    unmatched = agent._audit_analysis_numbers(
        body,
        [_verified(net_profit=32_619_000_000.0)],
    )
    assert unmatched == []


def test_majority_unmatched_appends_tip() -> None:
    agent = _writer()
    body = "A 1亿元，B 2亿元，C 3亿元。"
    unmatched = agent._audit_analysis_numbers(body, [_verified(revenue=777_102_000_000.0)])
    assert _AUDIT_MAJORITY_TIP in unmatched
    assert len([item for item in unmatched if item != _AUDIT_MAJORITY_TIP]) == 3


def test_invented_body_numbers_are_stripped() -> None:
    agent = _writer()
    body, flagged, blocked = agent._enforce_gate_on_body(
        "营收 7,771.02亿元，另有未校验口径 88.88%。",
        [_verified(revenue=777_102_000_000.0)],
    )
    assert "7,771.02" in body
    assert "88.88%" not in body
    assert _BLOCKED_PLACEHOLDER not in body
    assert flagged == []
    assert any("88.88" in item for item in blocked)


def test_unverified_claim_numbers_are_inline_marked() -> None:
    agent = _writer()
    body, flagged, blocked = agent._enforce_gate_on_body(
        "上半年营收 907.03亿元。",
        [
            _verified(revenue=172_054_000_000.0),
            _unverified("2026年上半年营业收入为907.03亿元", revenue=90_703_000_000.0),
        ],
    )
    assert "907.03亿元" in body
    assert "num-unverified" in body
    assert _UNVERIFIED_MARK not in body
    assert blocked == []
    assert any("907.03" in item for item in flagged)


def test_assemble_intercepts_invented_numbers() -> None:
    agent = _writer()
    claim = SimpleNamespace(
        verdict="verified",
        label="✅已验证",
        text="营收 7771.02 亿元",
        source="年报",
        reason="",
        section="revenue",
        gate_status="verified",
        numbers={"revenue": 777_102_000_000.0},
    )
    report = agent._assemble(
        question="营收多少",
        company="比亚迪",
        ticker="002594",
        as_of_date=date(2026, 8, 14),
        data_date="2025-12-31",
        claims=[claim],
        evidence_pool=[],
        analysis_body="营收 7,771.02亿元，另有未校验口径 88.88%。",
        coverage_gaps=[],
        extra_notes=[],
    )
    assert _BLOCKED_PLACEHOLDER not in report.analysis_body
    assert "88.88%" not in report.analysis_body
    assert "7,771.02" in report.analysis_body
    assert "## 正文数字审计" in report.markdown
    assert report.stats["body_audit_triggered"] is True
    assert report.stats["blocked_numbers_in_body"] >= 1
    assert report.markdown.index("正文数字审计") < report.markdown.index("免责声明")
    assert DISCLAIMER in report.markdown


def test_unverified_conclusion_shows_news_source() -> None:
    agent = _writer()
    claim = _unverified(
        "2026年半年报显示，中央汇金、证金公司退出前十大股东",
        source="贵州茅台 2025年报",
        reason="定性断言，来源为非官方渠道",
    )
    news = _news(
        "东方财富：茅台中报股东名单变动",
        "2026年半年报显示中央汇金、证金公司退出前十大股东，中国人寿加仓。",
    )
    rendered = agent._render_conclusions(agent._build_conclusions([claim], [news]))
    assert "[⚠️未验证]" in rendered
    assert "中央汇金" in rendered
    assert "2025年报" not in rendered
    assert "线索来源" not in rendered
    assert agent._build_conclusions([claim], [news])[0]["source"] == news.source_name


def test_unverified_numbers_recover_news_instead_of_annual_report() -> None:
    agent = _writer()
    claim = _unverified(
        "2026年上半年营业收入为907.03亿元，同比增长1.47%",
        source="贵州茅台 2025年报",
        reason="以下指标在证据池中找不到对应数据: revenue（2026年）",
        revenue=90_703_000_000.0,
    )
    news = _news(
        "东方财富：贵州茅台2026年中报净利润为445.17亿元",
        "贵州茅台2026年中报营业收入907.03亿元、同比增长1.47%，净利润445.17亿元。",
    )
    item = agent._build_conclusions([claim], [news])[0]
    assert item["source"] == news.source_name
    rendered = agent._render_conclusions([item])
    assert "907.03" in rendered
    assert "2025年报" not in rendered
    assert "线索来源" not in rendered


def test_unverified_without_news_does_not_cite_annual_report() -> None:
    agent = _writer()
    claim = _unverified(
        "某项指标为 12.34%",
        source="贵州茅台 2025年报",
        reason="以下指标在证据池中找不到对应数据: foo（2026年）",
    )
    item = agent._build_conclusions([claim], [])[0]
    assert item["source"] is None
    rendered = agent._render_conclusions([item])
    assert "线索来源" not in rendered
    assert "找不到对应数据" not in rendered
    assert "[⚠️未验证]" in rendered
    assert "找不到对应数据" in item["reason"]


def test_duplicate_conclusions_are_collapsed() -> None:
    agent = _writer()
    text = "比亚迪2024年ROE为21.73%，相比2023年的21.64%基本持平，微升0.09个百分点。"
    claims = [
        SimpleNamespace(
            verdict="verified",
            label="✅已验证",
            text=text,
            source="年报",
            reason="",
            section="conclusion",
            gate_status="verified",
            numbers={"roe": 0.2173},
        )
        for _ in range(3)
    ]
    items = agent._build_conclusions(claims, [])
    assert len(items) == 1
    rendered = agent._render_conclusions(items)
    assert rendered.count("21.73%") == 1


def test_inline_sources_are_stripped_from_body() -> None:
    agent = _writer()
    body, flagged, blocked = agent._enforce_gate_on_body(
        "ROE 为 21.73%（来源：比亚迪 2024年报跨期对比切片）。",
        [_verified(roe=0.2173)],
    )
    assert "21.73%" in body
    assert "来源：" not in body
    assert flagged == []
    assert blocked == []


def test_blocked_clause_dropped_keeps_verified_clause() -> None:
    agent = _writer()
    body, _flagged, blocked = agent._enforce_gate_on_body(
        "营收 7,771.02亿元，另有未校验口径 88.88%。毛利率为 19.44%，较去年上升。",
        [_verified(revenue=777_102_000_000.0)],
    )
    assert "7,771.02" in body
    assert "88.88" not in body
    assert "19.44" not in body
    assert _BLOCKED_PLACEHOLDER not in body
    assert any("88.88" in item for item in blocked)
