"""会话进度映射：节点名 → 比例与中文说明。"""

from __future__ import annotations

import asyncio

from session.session_store import SESSION_STATUS_RUNNING, SessionStore, describe_node_progress


def test_empty_node_is_submitted() -> None:
    ratio, label = describe_node_progress("")
    assert ratio == 0.05
    assert "等待规划" in label


def test_four_nodes_increase_monotonically() -> None:
    planner, _ = describe_node_progress("planner")
    retriever, retriever_label = describe_node_progress("retriever", retrieval_count=0)
    verifier, _ = describe_node_progress("verifier", retrieval_count=1)
    writer, _ = describe_node_progress("writer")
    assert 0 < planner < retriever < verifier < writer < 1
    assert "第 1 轮" in retriever_label


def test_extra_retrieval_round_does_not_drop_progress() -> None:
    first, first_label = describe_node_progress("retriever", retrieval_count=0)
    second, second_label = describe_node_progress("retriever", retrieval_count=1)
    assert second > first
    assert "第 2 轮" in second_label
    assert "第 1 轮" in first_label


def test_writer_is_near_complete() -> None:
    ratio, label = describe_node_progress("writer")
    assert ratio == 0.90
    assert "撰写" in label


def test_verifier_to_retriever_loop_does_not_drop() -> None:
    """补搜路径：verifier → retriever → verifier，比例只增不减。"""
    planner, _ = describe_node_progress("planner")
    r1, r1_label = describe_node_progress("retriever", retrieval_count=0)
    v1, _ = describe_node_progress("verifier", retrieval_count=1)
    r2, r2_label = describe_node_progress("retriever", retrieval_count=1)
    v2, _ = describe_node_progress("verifier", retrieval_count=2)
    r3, _ = describe_node_progress("retriever", retrieval_count=2)
    v3, _ = describe_node_progress("verifier", retrieval_count=3)
    writer, _ = describe_node_progress("writer")
    assert planner < r1 < v1 < r2 < v2 < r3 < v3 < writer < 1
    assert "第 1 轮" in r1_label
    assert "补搜" in r2_label
    assert "第 2 轮" in r2_label


def test_session_store_keeps_high_water_progress() -> None:
    async def _run() -> None:
        store = SessionStore(redis=None)
        await store.create("sid-progress", "q")
        await store.update_status(
            "sid-progress",
            SESSION_STATUS_RUNNING,
            current_node="verifier",
            progress=0.48,
            progress_label="正在核查证据",
        )
        await store.update_status(
            "sid-progress",
            SESSION_STATUS_RUNNING,
            current_node="retriever",
            progress=0.32,
            progress_label="正在检索",
        )
        payload = await store.get("sid-progress")
        assert payload is not None
        assert payload["current_node"] == "retriever"
        assert payload["progress"] >= 0.48

    asyncio.run(_run())
