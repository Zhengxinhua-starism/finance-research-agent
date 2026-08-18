# ============================================================
# 金融研报 Agent 镜像
#
# 设计说明：
# 1. 分层顺序按「变更频率从低到高」排列：系统依赖 → Python 依赖 → 模型 → 代码。
#    代码改动只会让最后一层失效，不会触发几分钟的依赖安装和模型下载。
# 2. 模型在构建期预下载进镜像（md 要求）。否则每次容器启动都要联网拉
#    200MB 模型，冷启动要 1~3 分钟，网络不通时服务直接不可用。
# 3. 用非 root 用户运行。容器逃逸时降低影响面，也是多数企业镜像扫描的硬性要求。
# 4. python:3.11-slim 而不是 alpine：sentence-transformers 依赖的
#    numpy/torch 在 musl libc 上没有预编译 wheel，alpine 会触发源码编译，
#    构建时间从 5 分钟涨到 40 分钟。
# ============================================================

FROM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim

# 构建期与运行期都需要的环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # 模型缓存目录固定，方便挂载卷复用
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers \
    # 容器内的 Redis 走服务名，而不是 localhost
    REDIS_URL=redis://redis:6379/0

WORKDIR /app

# ---- 层 1：系统依赖（几乎不变）----
# curl 用于 HEALTHCHECK；build-essential 供个别包源码编译后即删
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- 层 2：Python 依赖（改动较少）----
COPY requirements.txt .
# 先装 CPU 版 torch，避免默认拉取带 CUDA 的 2GB+ 包
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ---- 层 3：预下载模型（依赖不变则不重跑）----
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')" \
    && python -c "from sentence_transformers import CrossEncoder; \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# ---- 层 4：应用代码（改动最频繁）----
COPY . .
COPY --from=web /web/dist /app/web/dist

# 运行期需要写入的目录
RUN mkdir -p /app/traces /app/data/knowledge_base /app/data/tool_cache /app/eval/results \
    && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# 健康检查用 /api/health：它不依赖任何可能失败的组件，永远返回 200，
# 通过响应体里的 status 字段区分 ok / degraded / error
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
