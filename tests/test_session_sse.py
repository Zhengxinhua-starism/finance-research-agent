"""SSE 事件格式：不启动 FastAPI，只测序列化。"""

from __future__ import annotations

from api.routes import format_sse, session_payload_to_response


def test_format_sse_has_event_and_data_lines() -> None:
    text = format_sse("progress", {"current_node": "retriever", "progress": 0.4})
    assert text.startswith("event: progress\n")
    assert '"current_node": "retriever"' in text
    assert text.endswith("\n\n")


def test_session_payload_maps_progress_fields() -> None:
    response = session_payload_to_response(
        {
            "session_id": "abc",
            "status": "running",
            "current_node": "planner",
            "node_path": ["planner"],
            "progress": 0.15,
            "progress_label": "正在规划研究任务",
            "question": "营收多少",
            "company": "比亚迪",
            "ticker": "002594",
            "created_at": "2026-08-18T09:00:00",
            "updated_at": "2026-08-18T09:00:01",
            "run_id": None,
            "result": None,
            "trace_summary": None,
            "error": None,
        }
    )
    assert response.progress == 0.15
    assert response.progress_label == "正在规划研究任务"
    assert response.current_node == "planner"
