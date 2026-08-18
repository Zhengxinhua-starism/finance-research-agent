# Cursor Prompt：金融研报 Agent 完整项目生成

> 实现偏离：演示界面是 Vue3（FastAPI 同源托管 + SSE），已去掉 Gradio；调试走 CLI。下文里的 `--ui` / Gradio 以仓库现状为准。

你是一个资深 AI Agent 工程师，请基于以下架构设计，生成完整可运行的 Python 项目代码。

---

## 项目背景

这是一个**求职简历项目**，用于面试 AI Agent / 大模型应用开发岗位（深圳/广州，金融科技方向）。

代码设计要满足以下面试考察点：
1. **架构判断力**：能说清"为什么自建这块、为什么用框架那块"
2. **Agent 原理理解**：能白板画出 ReAct 循环、工具调用流程
3. **处理非确定性**：有超时保护、失败分类、幻觉拦截
4. **评测意识**：有自动化评测 pipeline，不是只有 demo
5. **可观测性**：每次运行有完整 trace 日志
6. **生产级工程**：有 API 服务层、Docker 部署、Redis 缓存

代码风格要求：
- 每个核心模块文件头部写一段注释，说明：这个模块解决什么问题、核心设计决策是什么、为什么这样选
- 类型注解完整（TypedDict / Pydantic BaseModel / Protocol）
- 错误处理清晰，不吞异常
- 变量命名语义明确，不缩写

---

## 技术栈

- Python 3.11+
- LangGraph（多 Agent 编排，StateGraph + 条件边）
- FastAPI（RESTful API 服务层）
- Redis（会话状态缓存 + 工具结果缓存，TTL 10min）
- DeepSeek API（OpenAI 兼容接口）
  - quick 层：deepseek-chat（工具调用、数据获取）
  - deep 层：deepseek-reasoner（规划、验证、综合分析）
- Chroma（向量数据库，本地运行）
- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2（embedding，轻量优先）
- sentence-transformers/ms-marco-MiniLM-L-6-v2（CrossEncoder Rerank，轻量优先）
- rank_bm25（BM25 关键词检索）
- AKShare（A 股财务数据，免费）
- MCP（Model Context Protocol，工具协议标准化）
- Gradio（Demo UI，调 FastAPI 接口）
- Pydantic（数据类 + 结构化输出）
- httpx（LLM API 调用）
- Docker + docker-compose（一键部署）

---

## 项目文件结构

```
finance_research_agent/
│
├── config.py                    # 全局配置（API key、模型名、Redis、阈值）
├── main.py                      # 入口：CLI + FastAPI + Gradio UI 启动
├── requirements.txt
├── .env.example
├── docker-compose.yml           # 一键启动（FastAPI + Redis + Chroma）
├── Dockerfile                   # FastAPI 服务镜像
├── README.md                    # 项目说明 + 启动命令 + 架构简介
│
├── harness/                     # 自建 Agent 运行时（不依赖 LangGraph）
│   ├── agent_loop.py            # ReAct 主循环 + max_turns 保护
│   ├── tool_registry.py         # 工具注册 + dispatch map
│   ├── evidence_gate.py         # 证据门禁（三级检查）
│   ├── context_compact.py       # 上下文压缩（保留关键事实）
│   ├── tracing.py               # JSON trace 日志
│   └── types.py                 # AgentResult, ToolResult, ToolCall 等数据类
│
├── llm/                         # LLM 调用层
│   ├── client.py                # LLMClient 统一接口（httpx 调 DeepSeek）
│   ├── router.py                # 两层路由（quick/deep）
│   └── schemas.py               # LLMResponse, TokenUsage 数据类
│
├── tools/                       # Agent 可调用的工具（实现 ToolProtocol）
│   ├── data_client.py           # FinancialDataClient Protocol 定义
│   ├── akshare_tools.py         # 6 个 AKShare 工具实现
│   ├── compare_periods.py       # 跨期对比工具
│   ├── news_search.py           # 新闻搜索工具
│   └── cache.py                 # content-hash 磁盘缓存
│
├── rag/                         # Agentic RAG 知识库
│   ├── hybrid_retriever.py      # BM25 + 向量双路检索 + RRF 融合
│   ├── reranker.py              # CrossEncoder Rerank 重排序
│   ├── rag_store.py             # Chroma 向量存储
│   ├── chunker.py               # 文本切分
│   └── prepare_data.py          # 预填充知识库脚本
│
├── mcp_servers/                 # MCP 工具协议服务
│   ├── base_server.py           # MCP Server 基类
│   ├── financial_data_server.py # AKShare 数据工具 MCP Server
│   └── knowledge_server.py      # RAG 知识库 MCP Server
│
├── agents/                      # LangGraph Agent 节点
│   ├── planner.py               # Planner Agent（deep model）
│   ├── retriever.py             # Retriever Agent（quick model + MCP 工具调用）
│   ├── verifier.py              # Verifier Agent（deep model + evidence gate）
│   └── writer.py                # Writer Agent（quick model，三级输出）
│
├── graph/                       # LangGraph 编排
│   ├── state.py                 # ResearchState TypedDict 定义
│   └── graph.py                 # StateGraph 构建 + 条件边
│
├── api/                         # FastAPI 服务层
│   ├── app.py                   # FastAPI 应用 + 路由注册
│   ├── routes.py                # /api/research, /api/trace, /api/eval 路由
│   ├── schemas.py               # 请求/响应 Pydantic 模型
│   └── dependencies.py          # FastAPI Depends（Redis、LLM Router 注入）
│
├── session/                     # Redis 会话管理
│   ├── session_store.py         # 会话状态读写（Redis）
│   └── cache.py                 # 工具结果缓存（Redis，TTL 10min）
│
├── eval/                        # 评测体系
│   ├── test_cases.json          # 9 道标准测试题
│   ├── evaluate.py              # LLM-as-Judge 自动评测脚本
│   └── failure_log.md           # 真实失败案例记录模板
│
├── traces/                      # 运行 trace 日志（自动生成，gitignore）
│
├── data/
│   ├── schemas.py               # FinancialMetrics, IncomeRow 等 Pydantic 模型
│   └── knowledge_base/          # Chroma 持久化目录（gitignore）
│
└── docs/
    ├── interview_qa.md          # 面试话术文档
    └── architecture.md          # 架构说明
```

---

## 金融业务领域知识

> 本节描述项目的金融业务逻辑。所有 Agent、工具、评测都必须围绕这些业务规则实现，不能生成空壳代码。

### 目标用户与场景

```
用户：散户投资者、初级分析师
场景：用户输入一个研究问题（如"比亚迪 2024 年毛利率为什么下降"），
      Agent 自动拉取财务数据 + 检索知识库，输出一份带证据标注的简报。
定位：信息处理工具，不是选股建议。输出的是"事实 + 证据"，不是"买入/卖出"。
```

### 研报输出结构

Writer Agent 输出的研报必须是这个结构（Markdown 格式）：

```markdown
# {公司名} — {问题主题}
> 分析日期：{as_of_date} | 数据截至：{data_date}

## 核心结论
- [✅已验证] 结论1（来源：2024年报 P12，营收 5021 亿元）
- [⚠️未验证] 结论2（未找到直接数据来源）
- [❌拒答] 无法判断的部分（原因：...）

## 详细分析
### 收入结构
- 2024年营收 X 亿元，同比 +Y%（来源：利润表）
- 其中：主营业务占比 Z%

### 盈利能力
- 毛利率 X%（去年 Y%，变动 Z pp）
- 净利率 X%
- ROE X%（杜邦拆解：利润率 × 周转率 × 杠杆）

### 风险提示
- 应收账款增速 X% vs 营收增速 Y%（背离预警）
- 经营性现金流 vs 净利润比值：Z（<0.8 为预警）

## 数据来源
| 数据项 | 来源 | 披露日期 | 状态 |
|--------|------|---------|------|
| 营收 5021 亿 | 2024年报 | 2025-03-28 | ✅已验证 |
| 毛利率下降 | 利润表推算 | 2025-03-28 | ✅已验证 |
```

三级标注规则（必须严格执行）：
- `✅已验证`：Evidence Gate 三级全通过（来源存在 + 数字一致 + 时点合规）
- `⚠️未验证`：有合理来源但未通过数字校验，或来源为新闻（非官方披露）
- `❌拒答`：找不到任何来源支撑，必须明确标注"无法判断"，**禁止编造**

### 核心财务指标体系

工具返回的数据必须包含以下指标，Writer 必须引用这些指标：

**盈利能力指标：**
| 指标 | 计算公式 | 数据来源 | 预警阈值 |
|------|---------|---------|---------|
| 毛利率 | (营收-营业成本)/营收 | 利润表 | 同比下降 >3pp |
| 净利率 | 净利润/营收 | 利润表 | 同比下降 >2pp |
| ROE | 净利润/净资产 | 利润表+资产负债表 | <8% 为偏低 |
| 扣非净利润增速 | (本期-上期)/上期 | 利润表 | 与营收增速背离 >10pp |

**成长性指标：**
| 指标 | 计算公式 | 数据来源 |
|------|---------|---------|
| 营收增速 | (本期-上期)/上期 | 利润表（至少 3 期） |
| 净利润增速 | 同上 | 利润表 |
| 研发费用占比 | 研发费用/营收 | 利润表 |

**风险指标：**
| 指标 | 计算公式 | 数据来源 | 预警阈值 |
|------|---------|---------|---------|
| 应收账款增速 vs 营收增速 | — | 资产负债表 vs 利润表 | 应收增速 > 营收增速 × 1.5 |
| 经营现金流/净利润 | — | 现金流量表 | <0.8 利润质量存疑 |
| 资产负债率 | 总负债/总资产 | 资产负债表 | >70% 高杠杆 |
| 商誉占净资产比 | 商誉/净资产 | 资产负债表 | >30% 减值风险 |

**杜邦分析（ROE 拆解）：**
```
ROE = 净利率 × 资产周转率 × 权益乘数

当 ROE 变动时，Agent 必须拆解出哪个因子驱动：
- 净利率下降 → 盈利恶化（关注成本控制）
- 周转率下降 → 效率恶化（关注资产利用）
- 杠杆上升推高 ROE → 质量下降（关注债务风险）
```

### 工具返回数据格式

每个 AKShare 工具必须返回结构化数据，格式示例：

```python
# get_financial_metrics 返回示例
{
    "company": "比亚迪",
    "ticker": "002594",
    "as_of": "2024-12-31",
    "metrics": {
        "revenue": 777100000000,        # 营业收入（元）
        "net_profit": 40200000000,      # 归母净利润
        "deducted_net_profit": 38500000000,  # 扣非净利润
        "gross_margin": 0.218,          # 毛利率
        "net_margin": 0.052,            # 净利率
        "roe": 0.195,                   # ROE
        "revenue_yoy": 0.182,           # 营收同比增速
        "net_profit_yoy": 0.341,        # 净利润同比增速
    }
}

# get_income_history 返回示例（3 期）
{
    "company": "比亚迪",
    "periods": [
        {"date": "2024-12-31", "revenue": 7771亿, "cost": 6075亿, "net_profit": 402亿, "rd_expense": 500亿},
        {"date": "2023-12-31", "revenue": 6023亿, "cost": 4910亿, "net_profit": 300亿, "rd_expense": 399亿},
        {"date": "2022-12-31", "revenue": 4241亿, "cost": 3518亿, "net_profit": 166亿, "rd_expense": 187亿},
    ]
}

# get_balance_sheet 返回关键字段
{
    "total_assets": ...,
    "total_liabilities": ...,
    "net_assets": ...,
    "accounts_receivable": ...,    # 应收账款
    "goodwill": ...,               # 商誉
    "debt_ratio": 0.774,           # 资产负债率
}

# get_cash_flow 返回关键字段
{
    "operating_cashflow": ...,     # 经营活动现金流
    "investing_cashflow": ...,     # 投资活动现金流
    "financing_cashflow": ...,     # 筹资活动现金流
    "cashflow_to_profit": 1.12,    # 经营现金流/净利润
}
```

### Planner Agent 的业务逻辑

Planner 不是简单地把问题传给 Retriever，而是做**金融维度的问题拆解**：

```python
# 输入问题："比亚迪 2024 年毛利率为什么下降"

# Planner 输出：
ResearchPlan(
    question_type="causal",           # 归因类问题
    required_data=[
        "get_income_history",         # 需要利润表（至少 3 期算趋势）
        "get_balance_sheet",          # 需要资产负债表（成本结构）
        "get_financial_metrics",      # 需要核心指标（毛利率直接值）
    ],
    research_steps=[
        "获取 2022-2024 三期利润表，计算毛利率变动趋势",
        "拆解成本端：营业成本增速 vs 营收增速",
        "检查是否有一次性减值、会计政策变更等非经营因素",
        "对比同行业（如有知识库数据）判断行业性还是个体性",
    ],
    expand_keywords=["毛利率", "成本控制", "原材料", "价格战", "规模效应"],
    analysis_framework="dupont",      # 告诉 Writer 用杜邦分析框架
)
```

问题类型与所需工具的对应关系：

```
single_fact（单点事实）:
  "比亚迪 2024 年 ROE 是多少"
  → get_financial_metrics 即可

cross_period（跨期对比）:
  "平安银行近三年不良率变化趋势"
  → get_income_history（3 期） + 计算同比变动

causal（归因分析）:
  "茅台净利率为什么这么高"
  → get_income_history + get_financial_metrics + 知识库检索（商业模式）

risk（风险评估）:
  "比亚迪应收账款风险大吗"
  → get_balance_sheet + get_cash_flow + 预警阈值检查
```

### Evidence Gate 的金融业务规则

证据门禁在检查金融数据时，必须执行以下业务规则：

```python
class EvidenceGate:
    def check_financial_claim(self, claim, evidence_pool):
        """
        金融事实检查（在通用三级门禁基础上增加）：

        1. 来源检查：是否来自年报/季报/公告（优先级：年报 > 季报 > 新闻）
        2. 数字检查：
           - 绝对值：允许 5% 误差（四舍五入导致）
           - 比率：允许 0.5pp 误差（如毛利率 21.8% vs 22.0%）
           - 同比增速：允许 1pp 误差
        3. 时点检查：
           - 2024年报数据必须在 2025-04-30 前披露（A股披露截止日）
           - 如果 as_of_date 是 2024-06-30，只能用 2024 中报及之前的数据
        4. 会计一致性检查（新增）：
           - 同一指标不能混用"合并报表"和"母公司报表"数据
           - "净利润"默认指"归母净利润"，不是"净利润总额"
        """
```

### 跨期对比工具的业务逻辑

```python
# tools/compare_periods.py

class ComparePeriodsTool:
    """
    跨期对比不是简单地把两期数据放一起，而是计算：
    1. 绝对值变动：本期 - 上期
    2. 同比变动率：(本期 - 上期) / 上期
    3. 结构变动：各业务板块占比变化
    4. 异常检测：变动率 > ±30% 的指标自动标记为"显著变动"
    """
    def run(self, ticker: str, periods: int = 3) -> dict:
        # 返回格式：
        return {
            "metrics_comparison": [
                {
                    "metric": "revenue",
                    "values": [4241亿, 6023亿, 7771亿],
                    "yoy_changes": ["", "+42.0%", "+29.0%"],
                    "trend": "growth_decelerating",  # 增速放缓
                    "anomaly": False,
                },
                {
                    "metric": "gross_margin",
                    "values": ["17.1%", "18.5%", "21.8%"],
                    "yoy_changes": ["", "+1.4pp", "+3.3pp"],
                    "trend": "improving",
                    "anomaly": False,
                },
            ],
            "anomalies": [],  # 如有异常变动，列出
            "periods": ["2022-12-31", "2023-12-31", "2024-12-31"],
        }
```

### 知识库预填充内容（RAG）

知识库不是空的，需要预填充以下金融领域知识：

```
prepare_data.py 预填充内容：

1. 行业分类与商业模式模板
   - 银行业：息差收入为主，关注不良率、拨备覆盖率、净息差
   - 新能源汽车：制造业逻辑，关注毛利率、交付量、研发投入
   - 白酒行业：消费品逻辑，关注预收款、库存、吨价

2. 财务分析框架
   - 杜邦分析：ROE = 净利率 × 周转率 × 杠杆
   - 现金流分析：经营现金流 vs 净利润判断利润质量
   - 成长性分析：营收增速 + 利润增速 + 研发投入

3. 风险预警规则（文本描述）
   - "应收账款增速远超营收增速，可能存在虚增收入风险"
   - "经营现金流持续低于净利润，利润质量存疑"
   - "商誉占净资产比过高，存在减值风险"
   - "短期借款大幅增加，偿债压力上升"

4. A 股信息披露规则
   - 年报披露截止：次年 4 月 30 日
   - 中报披露截止：当年 8 月 31 日
   - 季报披露截止：季后 1 个月内
   - 业绩预告：亏损/扭亏/大幅变动需提前预告
```

每条知识存储为 Document 对象：
```python
Document(
    content="银行业核心分析框架：...",
    metadata={
        "sector": "banking",
        "type": "analysis_framework",    # 框架/规则/事实
        "source": "手工整理",
    }
)
```

### 评测题的真实预期答案

test_cases.json 里的 expected_facts 必须是真实金融数据（需要运行后校准，先给出预期方向）：

```json
{
  "id": 1,
  "company": "比亚迪",
  "ticker": "002594",
  "question": "比亚迪 2024 年 ROE 是多少，相比 2023 年有什么变化",
  "question_type": "cross_period",
  "difficulty": "easy",
  "expected_facts": [
    "2024年ROE约19-20%",
    "2023年ROE约22-23%",
    "ROE同比下降约2-3个百分点",
    "需要杜邦拆解说明下降原因（净利率/周转率/杠杆哪个因素）"
  ],
  "expected_tools": ["get_financial_metrics", "compare_periods"],
  "expected_retrieval_count": 1,
  "should_refuse": false
}
```

---

## 各模块详细要求

### harness/agent_loop.py

参考：learn-claude-code 课程 s01（Agent Loop）

核心：while 循环实现 ReAct 模式，LLM 判断是否调用工具。

```python
class AgentLoop:
    def __init__(self, llm_client, tool_registry, max_turns=10, tracer=None):
        ...

    def run(self, system_prompt: str, messages: list, context: dict = None) -> AgentResult:
        """
        ReAct 循环：
        1. 调用 LLM（带 tools 定义）
        2. 如果返回 tool_calls → 执行工具 → 把结果加入 messages → 回到 1
        3. 如果返回纯文本（无 tool_calls）→ 结束，返回 AgentResult
        4. 超过 max_turns → 强制结束，status="max_turns"

        每次循环：
        - 记录 trace event（工具名、参数、耗时）
        - 检查 token 是否超限（status="token_limit"）
        """
```

必须实现的 status 枚举：
- `"completed"` — LLM 主动结束（正常）
- `"max_turns"` — 循环次数耗尽（异常，需记录）
- `"token_limit"` — 上下文超限（异常）
- `"error"` — 未预期的错误

---

### harness/tool_registry.py

参考：learn-claude-code 课程 s02（Tool Use）

```python
class ToolProtocol(Protocol):
    name: str
    description: str
    parameters: dict  # JSON Schema 格式
    def run(self, **kwargs) -> str | dict: ...

class ToolRegistry:
    def register(self, tool: ToolProtocol): ...
    def get_definitions(self) -> list[dict]: ...  # 给 LLM 的 function calling 格式
    def execute(self, tool_calls: list[ToolCall]) -> list[ToolResult]: ...
```

工具执行要求：
- 每个工具调用加 30 秒超时
- 失败时返回 ToolResult(success=False, error_type="...")
- error_type 分类：`"tool_not_found"` / `"param_error"` / `"execution_error"` / `"timeout"`

---

### harness/evidence_gate.py

参考：ai-hedge-fund 的弃权语义（abstain ≠ neutral）

```python
class EvidenceGate:
    def check(self, claim: Claim, evidence_pool: list[Evidence]) -> GateResult:
        """
        三级门禁：
        1. 来源检查：claim 是否有对应的 evidence source
        2. 数字检查：claim 中的数字是否与 source 原文一致（允许 5% 误差）
        3. 时点检查：source.disclosure_date <= claim.as_of_date（前视偏差拦截）
        """
```

GateResult status 枚举：
- `"verified"` — 三级全部通过
- `"unverified"` — 无来源文档
- `"number_mismatch"` — 数字与原文不一致
- `"lookahead_blocked"` — 披露日期晚于分析日期

---

### harness/context_compact.py

参考：learn-claude-code 课程 s08（Context Compact）

```python
class ContextCompact:
    def compact(self, messages: list, max_tokens: int = 3000, llm_client=None) -> list:
        """
        四步压缩策略：
        1. 保留 system prompt（不压缩）
        2. 保留最近 3 轮对话（不压缩）
        3. 保留所有工具调用结果（数字/来源，不压缩）
        4. 把更早的对话用 quick LLM 压缩成一段摘要
        """
```

与 TradingAgents 的区别（写进文件头部注释）：
> TradingAgents 用 RemoveMessage 清空全部历史；本模块压缩但保留关键事实和工具结果，只压缩过程性对话。

---

### harness/tracing.py

```python
class TraceEvent(BaseModel):
    timestamp: str
    event_type: Literal["llm_call", "tool_call", "gate_check", "compact", "node_enter", "node_exit"]
    agent_name: str
    input_summary: str   # 截断前 200 字符
    output_summary: str
    duration_ms: int
    token_usage: dict    # {"input": int, "output": int}
    metadata: dict = {}

class Tracer:
    def start_trace(self, run_id: str, question: str): ...
    def log_event(self, event: TraceEvent): ...
    def end_trace(self, final_result: str, total_tokens: int, total_duration_ms: int): ...
    def export_json(self) -> str: ...  # 保存到 traces/run_{timestamp}.json
```

---

### llm/client.py

```python
class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        """DeepSeek OpenAI 兼容接口，用 httpx 调用"""

    def chat(self, messages: list, system_prompt: str, tools: list[dict] = None) -> LLMResponse:
        """普通 chat，支持 function calling"""

    def chat_structured(self, messages: list, schema: type[BaseModel]) -> BaseModel | str:
        """结构化输出，失败降级为自由文本 + JSON 解析"""
```

DeepSeek API 地址：`https://api.deepseek.com/v1`（OpenAI 兼容格式）
加重试：指数退避（2s, 4s, 8s，最多 3 次）

---

### llm/router.py

```python
class LLMRouter:
    def __init__(self, config):
        self.quick = LLMClient(model="deepseek-chat", ...)
        self.deep  = LLMClient(model="deepseek-reasoner", ...)

    def get(self, tier: Literal["quick", "deep"]) -> LLMClient: ...
```

---

### rag/hybrid_retriever.py（核心新增模块）

```python
class HybridRetriever:
    """
    混合检索：BM25 关键词检索 + Chroma 向量检索，用 RRF 融合两路结果。
    
    为什么不用单路向量检索：
    - 向量检索擅长语义相似，但对精确数字/公司名称不敏感
    - BM25 擅长关键词精确匹配，但缺乏语义理解
    - RRF 融合后召回率比单路向量检索高约 10-15 个百分点
    
    RRF 公式：score(d) = sum(1 / (k + rank_i(d)))，k=60
    """
    def __init__(self, vector_store: RagStore, documents: list[Document], bm25_index=None):
        ...

    def retrieve(self, query: str, top_k: int = 10, alpha: float = 0.5) -> list[RetrievalResult]:
        """
        1. BM25 检索 → bm25_results（按 BM25 分数排序）
        2. Chroma 向量检索 → vector_results（按余弦相似度排序）
        3. RRF 融合：对两路结果计算 RRF 分数，合并去重
        4. 返回 top_k 结果，每条带 source、score、retrieval_method 标注
        """
```

RRF 实现参考：
```python
def reciprocal_rank_fusion(results_list: list[list[str]], k: int = 60) -> dict[str, float]:
    """
    results_list: 多路检索结果，每路是一个文档 ID 列表（按相关性排序）
    返回：{doc_id: rrf_score}，分数越高越相关
    """
    scores = {}
    for results in results_list:
        for rank, doc_id in enumerate(results, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores
```

---

### rag/reranker.py（核心新增模块）

```python
class CrossEncoderReranker:
    """
    对 HybridRetriever 返回的 top_k 结果做 CrossEncoder 精排。
    
    为什么加 Rerank：
    - 向量检索（Bi-Encoder）速度快但精度有限
    - CrossEncoder 对 (query, document) 对做联合注意力，精度更高但速度慢
    - 先粗筛（top 20）再精排（top 5），兼顾速度和质量
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        ...

    def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
        """
        1. 用 CrossEncoder 对每个 (query, doc.content) 对打分
        2. 按分数降序排列
        3. 返回 top_k，每条带 rerank_score 字段
        """
```

---

### rag/rag_store.py

```python
class RagStore:
    """Chroma 向量存储封装"""
    def __init__(self, collection_name: str = "finance_knowledge", persist_dir: str = "data/knowledge_base"):
        ...

    def add(self, documents: list[Document]): ...
    def search(self, query: str, top_k: int = 20) -> list[RetrievalResult]: ...
    def get_all_documents(self) -> list[Document]: ...  # 给 BM25 索引用
```

---

### mcp_servers/base_server.py（核心新增模块）

```python
class MCPServer:
    """
    MCP（Model Context Protocol）工具服务基类。
    
    为什么用 MCP：
    - 标准化工具接入协议，新增工具只需实现 MCP 接口
    - 解耦工具实现与 Agent 调用，支持远程工具
    - 业界标准协议（Anthropic 提出），面试高频考点
    
    工具通过 JSON-RPC 2.0 协议暴露：
    - tools/list：列出可用工具
    - tools/call：调用指定工具
    """
    name: str
    description: str

    def list_tools(self) -> list[dict]:
        """返回工具列表（JSON Schema 格式）"""
        ...

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """调用工具，返回结果"""
        ...
```

---

### mcp_servers/financial_data_server.py

```python
class FinancialDataMCPServer(MCPServer):
    """
    AKShare 金融数据 MCP Server。
    封装 6 个 AKShare 工具，通过 MCP 协议暴露给 Agent。
    Agent 通过 MCP 调用，而不是直接 import akshare_tools。
    """
    name = "financial_data"
    description = "A股金融数据工具集（AKShare）"

    # 内部持有 ToolRegistry，注册 6 个 AKShare 工具
    # list_tools() 返回 6 个工具的 JSON Schema
    # call_tool() 转发到 ToolRegistry.execute()
```

---

### mcp_servers/knowledge_server.py

```python
class KnowledgeMCPServer(MCPServer):
    """
    RAG 知识库 MCP Server。
    封装 HybridRetriever + Reranker，通过 MCP 协议暴露检索能力。
    """
    name = "knowledge_base"
    description = "金融研报知识库检索（BM25+向量混合检索+Rerank）"

    # list_tools() 返回 1 个工具：search_knowledge
    # call_tool("search_knowledge", {"query": "...", "top_k": 5})
    # → 调用 HybridRetriever + Reranker → 返回结果
```

---

### tools/akshare_tools.py

实现 6 个 ToolProtocol 子类：

| 工具名 | AKShare 函数 | 输入 | 输出 |
|--------|------------|------|------|
| `get_financial_metrics` | `ak.stock_financial_abstract_ths` | ticker | 最新财务指标 |
| `get_income_history` | `ak.stock_financial_report_sina` | ticker, periods | 最近 N 期利润表 |
| `get_balance_sheet` | `ak.stock_financial_report_sina` | ticker, periods | 资产负债表 |
| `get_cash_flow` | `ak.stock_financial_report_sina` | ticker, periods | 现金流量表 |
| `get_stock_price` | `ak.stock_zh_a_hist` | ticker, days | 日K线 |
| `search_news` | Tavily API 或 AKShare 新闻接口 | ticker, days | 新闻列表 |

每个工具：
- 实现 ToolProtocol（name, description, parameters, run）
- AKShare 调用加 try/except + 30s 超时
- 失败返回 ToolResult(success=False, error_type="execution_error")
- 成功结果写入 Redis 缓存（TTL 10min，key 格式：`tool:{tool_name}:{content_hash}`）

---

### agents/planner.py（deep model）

```python
class ResearchPlan(BaseModel):
    question_type: Literal["single_fact", "cross_period", "causal", "risk"]
    required_data: list[str]      # 需要哪些工具
    research_steps: list[str]     # 分析步骤
    expand_keywords: list[str]    # 补搜关键词
```

---

### agents/verifier.py（deep model）

```python
class VerifyResult(BaseModel):
    claims: list[VerifiedClaim]
    needs_more_retrieval: bool
    expand_query: str | None
    unverified_ratio: float

# 补搜触发条件：unverified_ratio > 0.3 且 retrieval_count < 3
```

---

### agents/retriever.py（重要更新：通过 MCP 调用工具）

```python
class RetrieverAgent:
    """
    Retriever 通过 MCP 协议调用工具，而不是直接 import 工具模块。
    
    工具调用链路：
    Retriever → MCP Client → MCP Server → ToolRegistry → AKShare/Chroma
    
    好处：
    - 新增数据源只需实现新的 MCP Server，不用改 Retriever 代码
    - 工具定义统一由 MCP Server 管理
    """
    def __init__(self, llm_client, mcp_servers: list[MCPServer], tracer):
        # 收集所有 MCP Server 的工具定义，传给 LLM 做 function calling
        ...

    def retrieve(self, plan: ResearchPlan, state: ResearchState) -> ResearchState:
        """
        1. 根据 plan.required_data 确定调用哪些工具
        2. 通过 MCP 协议调用工具（MCP Server 内部路由到具体工具）
        3. 工具结果写入 evidence_pool
        4. 结果同时写入 Redis 缓存
        """
```

---

### graph/graph.py（LangGraph StateGraph）

参考：TradingAgents graph/setup.py

```python
def build_research_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("planner",    planner_node)
    graph.add_node("retriever",  retriever_node)
    graph.add_node("verifier",   verifier_node)
    graph.add_node("writer",     writer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "verifier")
    graph.add_conditional_edges("verifier", should_retrieve_more, {
        "retrieve_more": "retriever",
        "write_report":  "writer",
    })
    graph.add_edge("writer", END)
    return graph
```

---

### api/app.py（核心新增模块）

```python
"""
FastAPI 服务层。

为什么需要 API 层（不只是 Gradio）：
- 面试 JD 几乎都要求 FastAPI 经验
- Gradio 只做前端展示，真实业务场景需要 RESTful API
- 支持后续接入前端（Vue/React）或其他调用方

架构：
  客户端 → FastAPI /api/research → 触发 LangGraph → 返回研报 JSON
  Gradio → 调 FastAPI 接口 → 渲染结果（Gradio 不直接调 Agent）
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Finance Research Agent API",
    description="金融研报 Agent API — 自研 Harness + Agentic RAG + MCP 工具协议",
    version="1.0.0",
)

# CORS（Gradio 跨域调用）
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```

---

### api/routes.py

```python
"""
三个核心路由：

POST /api/research
  请求：{"question": "...", "company_ticker": "000001", "as_of_date": "2025-12-31"}
  响应：{"run_id": "...", "status": "completed", "report": {...}, "trace_summary": {...}}

GET /api/trace/{run_id}
  响应：完整 trace JSON（所有 TraceEvent）

POST /api/eval
  请求：{"test_case_ids": [1, 2, 3]} 或 {}（跑全部）
  响应：{"results": [...], "summary": {"factual_accuracy": 0.89, ...}}

GET /api/health
  响应：{"status": "ok", "redis": "connected", "chroma": "connected"}
"""
```

---

### api/schemas.py

```python
class ResearchRequest(BaseModel):
    question: str
    company_ticker: str
    as_of_date: str  # YYYY-MM-DD

class ResearchResponse(BaseModel):
    run_id: str
    status: str
    report: dict        # 三级标注研报
    trace_summary: dict # token、耗时、工具调用次数

class EvalRequest(BaseModel):
    test_case_ids: list[int] | None = None  # None = 跑全部

class EvalResponse(BaseModel):
    results: list[dict]
    summary: dict       # 四维度平均分
```

---

### api/dependencies.py

```python
"""
FastAPI 依赖注入。
Redis 连接、LLM Router、MCP Servers 在这里初始化，通过 Depends 注入路由。
"""
from redis.asyncio import Redis

async def get_redis() -> Redis: ...
async def get_llm_router() -> LLMRouter: ...
async def get_mcp_servers() -> list[MCPServer]: ...
async def get_research_graph(): ...
```

---

### session/session_store.py

```python
"""
Redis 会话管理。

为什么用 Redis 而不是本地文件：
- 支持多进程/多实例部署
- TTL 自动过期，不用手动清理
- 生产环境标配，面试高频考点

存储内容：
- session:{session_id} → {"question": "...", "status": "running", "current_node": "retriever", ...}
- 研究完成后 status 改为 "completed"，report 写入 result 字段
"""
class SessionStore:
    def __init__(self, redis: Redis):
        ...

    async def create(self, session_id: str, question: str, company: str) -> None: ...
    async def update_status(self, session_id: str, status: str, current_node: str = None) -> None: ...
    async def get(self, session_id: str) -> dict | None: ...
    async def set_result(self, session_id: str, report: dict, trace_summary: dict) -> None: ...
```

---

### session/cache.py

```python
"""
Redis 工具结果缓存。

与 tools/cache.py 的磁盘缓存互补：
- Redis：热数据缓存（TTL 10min，同一会话内重复查询命中）
- 磁盘：冷数据缓存（content-hash，跨会话持久化）

查询顺序：Redis → 磁盘 → 调 AKShare
"""
class ToolCache:
    def __init__(self, redis: Redis, ttl_seconds: int = 600):
        ...

    async def get(self, tool_name: str, args_hash: str) -> str | None: ...
    async def set(self, tool_name: str, args_hash: str, result: str) -> None: ...
```

---

### eval/evaluate.py

评测维度（4个）：
1. `factual_accuracy`：LLM-as-Judge，传入输出 + expected_facts，打分 0-1
2. `refusal_calibration`：该拒答时拒答了吗
3. `retrieval_efficiency`：从 trace 提取 retrieval_count
4. `evidence_coverage`：confirmed 结论占总结论比例

---

### eval/test_cases.json

9 道题，覆盖：
- 3 家公司（平安银行 000001、比亚迪 002594、贵州茅台 600519）
- 3 种难度（easy/medium/hard）
- 4 种问题类型（single_fact/cross_period/causal/risk）
- 包含 1 道拒答测试题（不存在的公司）

---

### main.py

三种启动方式：
1. `python main.py --question "..." --company "000001"`（CLI）
2. `python main.py --api`（FastAPI，端口 8000）
3. `python main.py --ui`（Gradio，端口 7860，调 FastAPI 接口）
4. `docker-compose up`（一键启动 FastAPI + Redis + Chroma）

Gradio 界面改为调 FastAPI 接口：
- 输入：研究问题 + 公司下拉框 + 日期选择器
- 点击提交 → POST /api/research → 轮询直到完成 → 显示结果
- 输出左栏：结构化研报（Markdown，三级标注）
- 输出右栏：trace 表格（工具名、耗时、结果摘要）
- 底部：token 消耗统计

---

### docker-compose.yml

```yaml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - redis
    volumes:
      - ./data/knowledge_base:/app/data/knowledge_base
      - ./traces:/app/traces

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  ui:
    build: .
    command: python main.py --ui
    ports:
      - "7860:7860"
    depends_on:
      - api

volumes:
  redis_data:
```

---

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 预下载 embedding 和 reranker 模型到镜像
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### config.py

```python
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    # DeepSeek
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    quick_model: str = "deepseek-chat"
    deep_model: str = "deepseek-reasoner"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 600  # 10 分钟

    # Chroma
    chroma_persist_dir: str = "data/knowledge_base"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # RAG
    bm25_top_k: int = 20
    vector_top_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60

    # Harness
    max_turns: int = 10
    tool_timeout_seconds: int = 30
    context_max_tokens: int = 3000

    # 证据门禁
    number_tolerance: float = 0.05  # 5% 数字容差

    class Config:
        env_file = ".env"
```

---

### requirements.txt

```
# LLM
httpx>=0.27
openai>=1.0

# Agent 框架
langgraph>=0.2
langchain-core>=0.3

# 向量数据库
chromadb>=0.5

# Embedding + Rerank
sentence-transformers>=3.0
rank-bm25>=0.2

# 金融数据
akshare>=1.14

# Web 框架
fastapi>=0.115
uvicorn>=0.32
gradio>=5.0

# Redis
redis>=5.0

# 数据类
pydantic>=2.0
pydantic-settings>=2.0

# 工具
python-dotenv>=1.0
tiktoken>=0.8
```

---

## 重要注意事项

1. AKShare 接口不稳定：所有调用必须加 try/except + 超时
2. DeepSeek API 限流：加指数退避重试
3. 不要 mock 数据：调真实 API
4. .env 不要提交：创建 .env.example
5. Chroma embedding：优先用 paraphrase-multilingual-MiniLM-L12-v2（约 120MB）
6. CrossEncoder Rerank：用 cross-encoder/ms-marco-MiniLM-L-6-v2（约 80MB）
7. Redis：开发时可用 `redis-server` 本地启动，生产用 docker-compose
8. 用 requirements.txt，不用 poetry
9. FastAPI 路由用 async，Redis 用 redis.asyncio
10. Gradio 只调 FastAPI 接口，不直接调 Agent（解耦前后端）
11. MCP Server 用本地进程间调用（不用 HTTP），保持简单
12. Docker 镜像预下载模型，避免每次启动都下载

请生成所有文件的完整代码。
