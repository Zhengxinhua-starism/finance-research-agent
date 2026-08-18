"""FastAPI 服务层。

分层：schemas（契约）→ dependencies（资源注入）→ routes（端点）→ app（应用装配）
Vue 只通过 HTTP/SSE 调用本层，不直接 import Agent 代码，保证前后端解耦。
"""

from api.app import app

__all__ = ["app"]
