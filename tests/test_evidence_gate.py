"""Evidence Gate 确定性规则单测。

这些场景对应 failure_log 里反复出现的误判：跨期核对、前视偏差、
口径混用、新闻降级、比率容差。门禁是纯函数，不该只靠端到端评测覆盖。
"""

from __future__ import annotations

from datetime import date

from harness.evidence_gate import EvidenceGate
from harness.types import (
    GATE_LOOKAHEAD_BLOCKED,
    GATE_NUMBER_MISMATCH,
    GATE_UNVERIFIED,
    GATE_VERIFIED,
    Claim,
    ClaimNumber,
    Evidence,
)

AS_OF = date(2026, 4, 30)
GATE = EvidenceGate()


def _evidence(**overrides: object) -> Evidence:
    payload = {
        "source_type": "annual_report",
        "source_name": "比亚迪2025年年报",
        "disclosure_date": date(2026, 3, 28),
        "period_end": date(2025, 12, 31),
        "content": "比亚迪 2025 年营业收入 7771.02 亿元，毛利率 17.74%，ROE 13.2%。",
        "numbers": {"revenue": 7771.02e8, "gross_margin": 0.1774, "roe": 0.132},
        "ticker": "002594",
        "company": "比亚迪",
        "statement_scope": "consolidated",
    }
    payload.update(overrides)
    return Evidence(**payload)  # type: ignore[arg-type]


def _claim(*, text: str, numbers: list[ClaimNumber], **overrides: object) -> Claim:
    payload = {
        "text": text,
        "numbers": numbers,
        "as_of_date": AS_OF,
        "ticker": "002594",
        "source_hint": "比亚迪2025年年报",
        "statement_scope": "consolidated",
    }
    payload.update(overrides)
    return Claim(**payload)  # type: ignore[arg-type]


def test_single_period_metric_verified() -> None:
    result = GATE.check(
        _claim(
            text="比亚迪 2025 年营业收入为 7771.02 亿元",
            numbers=[ClaimNumber(metric="revenue", value=7771.02e8, period="2025")],
        ),
        [_evidence()],
    )
    assert result.status == GATE_VERIFIED


def test_cross_period_comparison_verified() -> None:
    """F-018：两个同名指标分属两年，必须按 period 分别核对。"""
    ev_2024 = _evidence(
        source_name="比亚迪2024年年报",
        disclosure_date=date(2025, 3, 28),
        period_end=date(2024, 12, 31),
        content="比亚迪 2024 年毛利率 19.44%。",
        numbers={"gross_margin": 0.1944},
    )
    ev_2025 = _evidence(
        content="比亚迪 2025 年毛利率 17.74%。",
        numbers={"gross_margin": 0.1774},
    )
    result = GATE.check(
        _claim(
            text="毛利率由 2024 年的 19.44% 降至 2025 年的 17.74%",
            numbers=[
                ClaimNumber(metric="gross_margin", value=0.1944, kind="ratio", period="2024"),
                ClaimNumber(metric="gross_margin", value=0.1774, kind="ratio", period="2025"),
            ],
            source_hint=None,
        ),
        [ev_2024, ev_2025],
    )
    assert result.status == GATE_VERIFIED


def test_lookahead_blocked() -> None:
    result = GATE.check(
        _claim(
            text="比亚迪 2025 年营业收入为 7771.02 亿元",
            numbers=[ClaimNumber(metric="revenue", value=7771.02e8, period="2025")],
            as_of_date=date(2025, 12, 31),
        ),
        [_evidence()],
    )
    assert result.status == GATE_LOOKAHEAD_BLOCKED
    assert result.verdict == "refused"


def test_number_mismatch() -> None:
    result = GATE.check(
        _claim(
            text="比亚迪 2025 年营业收入为 5000 亿元",
            numbers=[ClaimNumber(metric="revenue", value=5000e8, period="2025")],
        ),
        [_evidence()],
    )
    assert result.status == GATE_NUMBER_MISMATCH
    assert result.number_diffs


def test_accounting_scope_mismatch() -> None:
    result = GATE.check(
        _claim(
            text="比亚迪 2025 年营业收入为 7771.02 亿元",
            numbers=[ClaimNumber(metric="revenue", value=7771.02e8, period="2025")],
            statement_scope="consolidated",
        ),
        [_evidence(statement_scope="parent")],
    )
    assert result.status == GATE_UNVERIFIED
    assert "口径" in result.reason


def test_news_source_downgraded_even_if_number_matches() -> None:
    news = _evidence(
        source_type="news",
        source_name="某媒体：比亚迪营收",
        content="报道称比亚迪 2025 年营业收入 7771.02 亿元。",
        numbers={"revenue": 7771.02e8},
    )
    result = GATE.check(
        _claim(
            text="比亚迪 2025 年营业收入为 7771.02 亿元",
            numbers=[ClaimNumber(metric="revenue", value=7771.02e8, period="2025")],
            source_hint="某媒体",
        ),
        [news],
    )
    assert result.status == GATE_UNVERIFIED
    assert "非官方" in result.reason or "新闻" in result.reason or "news" in result.reason


def test_qualitative_official_source_verified() -> None:
    result = GATE.check(
        _claim(
            text="比亚迪处于新能源汽车行业",
            numbers=[],
            source_hint="比亚迪2025年年报",
        ),
        [
            _evidence(
                content="比亚迪处于新能源汽车行业，主营新能源车及电池。",
                numbers={},
            )
        ],
    )
    assert result.status == GATE_VERIFIED


def test_qualitative_news_source_unverified() -> None:
    result = GATE.check(
        _claim(
            text="比亚迪处于新能源汽车行业",
            numbers=[],
            source_hint="某媒体报道",
        ),
        [
            _evidence(
                source_type="news",
                source_name="某媒体报道",
                content="比亚迪处于新能源汽车行业。",
                numbers={},
            )
        ],
    )
    assert result.status == GATE_UNVERIFIED


def test_ratio_uses_pp_tolerance_not_relative_5pct() -> None:
    """0.218 vs 0.228：相对误差约 4.6% 会过 5% 门槛，但差 1.0pp，应被比率容差拦住。"""
    result = GATE.check(
        _claim(
            text="比亚迪 2025 年毛利率为 22.8%",
            numbers=[ClaimNumber(metric="gross_margin", value=0.228, kind="ratio", period="2025")],
        ),
        [_evidence(content="比亚迪 2025 年毛利率 21.8%。", numbers={"gross_margin": 0.218})],
    )
    assert result.status == GATE_NUMBER_MISMATCH
