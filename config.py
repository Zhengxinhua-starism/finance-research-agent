"""全局配置模块。

解决什么问题
    项目里有 6 类外部依赖（DeepSeek、Redis、Chroma、AKShare、embedding 模型、
    reranker 模型）和 3 类阈值参数（RAG 检索、Harness 保护、证据门禁容差）。
    如果散落在各模块里用 os.getenv 读取，会出现"同一个阈值两个地方不一致"的问题，
    而且没法在测试时统一覆盖。本模块把它们收敛成单一配置对象。

核心设计决策
    1. 用 pydantic-settings 的 BaseSettings 而不是裸 dict / configparser：
       它自带类型校验和 .env 加载，配置写错类型（比如 max_turns="ten"）会在
       进程启动时立刻报错，而不是在 Agent 跑到第 8 轮时才炸。
    2. 单例通过 get_config() + lru_cache 暴露，而不是模块级 `config = Config()`。
       模块级实例化会让"导入这个模块"产生副作用：没有 .env 时 import 就失败，
       连 `python -c "import config"` 这种冒烟测试都跑不了。
    3. API key 不设为必填字段，而是给空默认值 + 提供 require_deepseek_api_key()
       做惰性校验。这样离线单测（只测 RRF 融合、证据门禁这类纯逻辑）不需要真 key。

为什么不用其他方案
    - 不用 dynaconf / hydra：这个项目只有单一环境，多环境配置分层是过度设计。
    - 不用 dataclass + 手写解析：等于重新实现一遍 pydantic 的类型强制转换。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（config.py 所在目录），用于把相对路径锚定到项目而不是当前工作目录。
# 否则从 eval/ 目录执行脚本时，"data/knowledge_base" 会指向 eval/data/knowledge_base。
PROJECT_ROOT = Path(__file__).resolve().parent


class Config(BaseSettings):
    """全局配置。所有字段均可通过同名大写环境变量覆盖，例如 MAX_TURNS=5。"""

    # pydantic-settings v2：用 model_config 替代 v1 的内部 class Config。
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------- DeepSeek LLM ----------------
    # 空默认 + require_deepseek_api_key() 惰性校验：纯逻辑单测不需要真 key。
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    quick_model: str = "deepseek-chat"
    deep_model: str = "deepseek-reasoner"

    llm_timeout_seconds: int = 120
    # 指数退避 2s/4s/8s：2 * 2^n，最多 llm_max_retries 次。
    llm_max_retries: int = 3
    llm_backoff_base_seconds: float = 2.0
    llm_temperature: float = 0.0

    # ---------------- Redis ----------------
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 600  # 工具结果缓存 10 分钟
    session_ttl_seconds: int = 3600  # 会话状态 1 小时
    # Redis 不可用时是否降级为"无缓存直接调用"而不是抛错。
    # 面试演示环境经常没装 Redis，硬依赖会导致整个项目跑不起来。
    redis_optional: bool = True

    # ---------------- Chroma / RAG 模型 ----------------
    chroma_persist_dir: str = "data/knowledge_base"
    chroma_collection_name: str = "finance_knowledge"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ---------------- RAG 检索参数 ----------------
    bm25_top_k: int = 20
    vector_top_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60
    chunk_size: int = 400  # 字符数，中文语料按字符切更稳
    chunk_overlap: int = 80

    # ---------------- Harness 运行时 ----------------
    max_turns: int = 10
    tool_timeout_seconds: int = 30
    # 触发压缩的阈值，不是模型窗口。DeepSeek 有 64K–128K，但 3000 只够
    # 放下约 1.5 条完整工具 JSON，会让"最近 1 轮原文"经常直接超预算。
    # 8000 ≈ system + 1 轮完整工具结果 + 中间轮次的指标摘要。
    context_max_tokens: int = 8000
    # 单次 Agent 运行的 token 硬上限，超过则 status="token_limit"
    max_total_tokens: int = 120_000

    # ---------------- 证据门禁 ----------------
    number_tolerance: float = 0.05  # 绝对值 5% 容差
    ratio_tolerance_pp: float = 0.5  # 比率类指标容差（百分点）
    growth_tolerance_pp: float = 1.0  # 同比增速容差（百分点）
    # 补搜触发阈值
    unverified_ratio_threshold: float = 0.3
    max_retrieval_rounds: int = 3

    # ---------------- 工具 / 数据源 ----------------
    akshare_timeout_seconds: int = 30
    disk_cache_dir: str = "data/tool_cache"
    disk_cache_ttl_seconds: int = 86400  # 冷数据缓存 1 天
    # 日 K 行情源优先级。腾讯/新浪直连且绕过系统代理；东财字段更全但易被代理劫持，放最后。
    price_source_priority: str = "tencent,sina,eastmoney"
    # 可选。新闻工具默认走免费的 AKShare 东财接口，只有条数不足
    # news_min_results 时才调 Tavily 补充；留空则不做补充检索。
    tavily_api_key: str = ""
    news_min_results: int = 3

    # ---------------- 服务 ----------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    trace_dir: str = "traces"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ---------------- 评测 ----------------
    eval_test_cases_path: str = "eval/test_cases.json"
    eval_judge_tier: Literal["quick", "deep"] = "deep"

    @field_validator("number_tolerance")
    @classmethod
    def _validate_tolerance(cls, value: float) -> float:
        if not 0 < value < 1:
            raise ValueError("number_tolerance 必须是 0~1 之间的小数（0.05 表示 5%）")
        return value

    @field_validator("max_turns")
    @classmethod
    def _validate_max_turns(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_turns 至少为 1")
        return value

    # ---------------- 路径工具 ----------------
    def resolve_path(self, relative_or_absolute: str) -> Path:
        """把配置里的路径锚定到项目根目录（绝对路径原样返回）。"""
        path = Path(relative_or_absolute)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def chroma_dir(self) -> Path:
        return self.resolve_path(self.chroma_persist_dir)

    @property
    def traces_dir(self) -> Path:
        return self.resolve_path(self.trace_dir)

    @property
    def disk_cache_path(self) -> Path:
        return self.resolve_path(self.disk_cache_dir)

    def ensure_runtime_dirs(self) -> None:
        """创建运行时需要写入的目录。在服务/CLI 启动时调用一次即可。"""
        for directory in (self.chroma_dir, self.traces_dir, self.disk_cache_path):
            directory.mkdir(parents=True, exist_ok=True)

    def require_deepseek_api_key(self) -> str:
        """需要真实调用 LLM 时才校验 key，缺失时给出可操作的错误信息。"""
        key = self.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")
        if not key:
            raise RuntimeError(
                "缺少 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 并填入 DeepSeek API Key，"
                "或直接设置环境变量 DEEPSEEK_API_KEY。"
            )
        return key


@lru_cache(maxsize=1)
def get_config() -> Config:
    """获取全局唯一配置实例。lru_cache 保证 .env 只解析一次。"""
    return Config()


def reload_config() -> Config:
    """清空缓存后重新加载配置。仅测试或热更新场景使用。"""
    get_config.cache_clear()
    return get_config()
