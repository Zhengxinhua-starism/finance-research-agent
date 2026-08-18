"""列名三段式匹配：精确优先，短别名不能误命中「公告日期」。"""

from __future__ import annotations

from tools.data_client import find_column


def test_exact_alias_wins() -> None:
    columns = ["营业收入", "营业总收入", "一、营业总收入"]
    assert find_column(columns, ["营业总收入", "营业收入"]) == "营业总收入"


def test_strip_prefix_then_exact() -> None:
    columns = ["一、营业总收入", "营业成本"]
    assert find_column(columns, ["营业总收入"]) == "一、营业总收入"


def test_report_date_aliases_prefer_报告日_not_公告日期() -> None:
    """F-001：别名不能含孤立的「日期」，否则包含匹配会命中公告日期。"""
    from tools.akshare_tools import REPORT_DATE_ALIASES

    assert "日期" not in REPORT_DATE_ALIASES
    columns = ["报告日", "公告日期", "更新日期"]
    assert find_column(columns, REPORT_DATE_ALIASES) == "报告日"
