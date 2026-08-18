"""Redis 会话与缓存层。

两个模块的分工：
- session_store：会话状态（研究进度、最终结果），TTL 1 小时；
- cache：工具结果热缓存（异步接口），TTL 10 分钟，与 tools/cache.py 的磁盘冷缓存互补。
"""

from session.cache import ToolCache
from session.session_store import (
    SESSION_STATUS_COMPLETED,
    SESSION_STATUS_FAILED,
    SESSION_STATUS_PENDING,
    SESSION_STATUS_RUNNING,
    SessionStore,
)

__all__ = [
    "SESSION_STATUS_COMPLETED",
    "SESSION_STATUS_FAILED",
    "SESSION_STATUS_PENDING",
    "SESSION_STATUS_RUNNING",
    "SessionStore",
    "ToolCache",
]
