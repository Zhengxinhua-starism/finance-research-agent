# 金融研报 Agent

一个面向 A 股的研究型 Agent：输入一个研究问题，自动拉取财务数据、检索知识库、
核查证据，输出一份**每条结论都带证据标注**的研报。

> 定位：信息处理工具，不是选股建议。输出的是「事实 + 证据」，不是「买入/卖出」。

```
✅已验证  通过三级证据门禁（来源存在 + 数字一致 + 时点合规）
⚠️未验证  有来源但未通过数字校验，或来源为新闻等非官方渠道
❌拒答    找不到证据支撑，明确标注"无法判断"，禁止编造
```

---

## 这个项目在解决什么

让 LLM 读财报并不难，难的是**让它不撒谎**。三类致命错误：

| 错误类型 | 例子 | 本项目的应对 |
|---------|------|-------------|
| 编造数据 | "2024 年营收 8000 亿"，无任何来源 | 证据门禁第 1 级：无来源 → ⚠️未验证 |
| 抄错数字 | 来源写 7771 亿，输出写 7717 亿 | 证据门禁第 2 级：数字比对，分档容差 |
| 前视偏差 | 用 2025-03 披露的年报支撑"截至 2024-06 的判断" | 证据门禁第 3 级：披露日期 ≤ 分析基准日 |

另外两个工程问题：**非确定性**（工具挂了、模型不收敛、上下文爆炸）
和**不可观测**（出了问题不知道为什么）。这两块由自建 Harness 处理。

---

## 架构

```
                        FastAPI (/api/research)
                                 │
                    ┌────────────▼────────────┐
                    │   LangGraph StateGraph  │   ← 编排：用框架
                    └────────────┬────────────┘
        ┌─────────────┬──────────┼──────────┬─────────────┐
        ▼             ▼          ▼          ▼             │
    Planner  ──→  Retriever ──→ Verifier ──→ Writer       │
   (chat)         (chat)       (chat)       (chat)        │
        │             │            │            ▲          │
        │             │            └─ 证据不足 ──│──────────┘
        │             │               （最多补搜 3 轮）     │
        └─ 索取投资建议 ─────────────────────────┘
           （直接拒答，不调 LLM、不取数）
        │             ▼
        │      ┌──────────────┐
        │      │ MCP Broker   │  ← 工具协议：JSON-RPC 2.0 语义
        │      └──────┬───────┘
        │        ┌────┴─────┐
        │        ▼          ▼
        │  financial_data  knowledge_base
        │   (AKShare)      (Agentic RAG)
        │        │              │
        │        │         BM25 ┼ 向量
        │        │              ├─ RRF 融合 (k=60)
        │        │              └─ CrossEncoder 精排
        ▼        ▼
┌──────────────────────────────────────────────┐
│  自建 Harness（不依赖 LangGraph）             │   ← 运行时：自建
│  ├─ agent_loop     ReAct 循环 + max_turns    │
│  ├─ tool_registry  注册/校验/超时/错误归类     │
│  ├─ evidence_gate  三级证据门禁                │
│  ├─ context_compact 压缩过程、保留事实         │
│  └─ tracing        JSON trace 全链路          │
└──────────────────────────────────────────────┘
```

### 为什么编排用框架、运行时自建

这是本项目最核心的架构判断，分界线画在**通用 vs 领域特有**，不是简单 vs 复杂：

- **编排**（谁在什么条件下跑、状态怎么合并、条件回边）是标准化问题，
  LangGraph 的 StateGraph + reducer 是成熟方案，自建没有额外价值。
- **运行时**（max_turns 何时触发、上下文怎么压、证据怎么验）是这个项目的
  差异化所在。藏进框架配置里，既讲不清也改不动——
  而这三件事恰恰决定了研报可不可信。

---

## 关键设计

### 1. 三级证据门禁（`harness/evidence_gate.py`

```
第 1 级 来源检查 → 证据池里有没有支撑这条断言的来源？
第 2 级 时点检查 → 来源的披露日期是否不晚于分析基准日？（前视偏差拦截）
第 3 级 口径检查 → 合并报表 vs 母公司报表有没有混用？
第 4 级 数字检查 → 断言里的数字与来源是否一致？
```

**数字容差按语义分三档**，不是一刀切 5%：

| 数字类型 | 容差 | 为什么 |
|---------|------|--------|
| 绝对值（营收/净利润） | 5% 相对误差 | 容忍单位换算和四舍五入 |
| 比率（毛利率/ROE） | 0.5 个百分点 | 21.8% 与 22.9% 相对误差仅 5%，但投研结论完全不同 |
| 同比增速 | 1 个百分点 | 同上 |

**弃权语义**（借鉴 ai-hedge-fund）：`abstain ≠ neutral`。
"查不到证据"不等于"证据表明不成立"，更不等于"中性判断"。
所以 `GateResult` 没有 True/False，只有四种状态，调用方无法把"查不到"误读成"没问题"。

### 2. 混合检索（`rag/hybrid_retriever.py`

单路向量检索在金融语料上有两个明确短板：查"商誉减值"会被语义相近的
"资产减值准备"顶掉；查"002594"这种代码在 embedding 空间里没有语义。
BM25 补上这两点，用 **RRF 融合**：

```
score(d) = Σ 1 / (k + rank_i(d))，k = 60
```

用**名次**而不是分数融合，天然免疫两路分数分布不同的问题
（BM25 分数无上界且随语料规模变化，余弦相似度在 [0,1]）。
分数加权方案里的 alpha 每次换语料都要重调，RRF 不用。

之后再过一遍 CrossEncoder 精排：粗排 20 条（Bi-Encoder，快）→ 精排 5 条
（CrossEncoder 对 query-document 做联合注意力，准）。

### 3. 上下文压缩（`harness/context_compact.py`）

给模型看的对话可以分层摘要；给门禁用的证据对象保留精确数字。
两条通路分开：压 message 不会让门禁对不上数。
TradingAgents 式清空全部历史会丢掉证据链，这里不采用。

### 4. 金融计算不交给 LLM

毛利率、ROE、杜邦拆解、同比增速、七条风险规则，全部在
`data/schemas.py` 和 `tools/compare_periods.py` 里用代码实现。
LLM 只负责**解释**"为什么变化"，不负责**计算**"变化了多少"。

理由：LLM 做算术不可靠且错得没有规律，在评测里表现为随机失分；
而这些公式是确定的，交给代码后同一份数据永远算出同一个结果。

---

## 快速开始

```bash
git clone https://github.com/<your-username>/finance-research-agent.git
cd finance-research-agent
```

推送后把上面的地址换成你的仓库 URL。不要提交 `.env`，只提交 `.env.example`。

### 环境要求

- Python 3.11 或 3.12（3.13+ 上 chromadb / sentence-transformers 尚无稳定 wheel）
- DeepSeek API Key（[申请地址](https://platform.deepseek.com/api_keys)）
- Redis（可选，没有会自动降级为内存模式）

### 本地运行

```bash
# 1. 建虚拟环境并安装依赖
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu   # 先装 CPU 版，否则会拉 2.5GB 的 CUDA 包
pip install -r requirements.txt

# 2. 配置密钥
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

# 3. 初始化知识库（首次必须，约 3 分钟，会下载 120MB embedding + 80MB reranker 模型）
python main.py --prepare-data

# 4. 跑一个问题试试
python main.py -q "比亚迪2024年毛利率为什么下降" -c 002594
```

> **Windows 用户注意**：如果项目路径较长（超过约 100 字符），
> 把 venv 建在项目目录下会在安装 torch 时报 `WinError 206 文件名或扩展名太长`——
> torch 的 license 文件嵌套极深，加上项目路径会突破 260 字符的 MAX_PATH 上限。
> 两种解法：启用系统长路径支持（改注册表 `LongPathsEnabled`，需管理员），
> 或者把 venv 建到短路径下，例如：
>
> ```powershell
> py -3.12 -m venv D:\venvs\fra
> D:\venvs\fra\Scripts\Activate.ps1
> ```
>
> 后者不需要管理员权限。`.vscode/settings.json` 里的
> `python.defaultInterpreterPath` 指向的就是这个位置，按需修改。

### 启动服务

```bash
# 演示前端（Vue3）：先构建，再由 FastAPI 同源托管
cd web
npm install
# 若 API_PORT 不是 8000，把 web/.env.development 里的 VITE_API_TARGET 改成对应地址
npm run build
cd ..

# 终端 1：API + 前端（同一端口）
python main.py --api
# 打开 http://localhost:8000/   （若 .env 里 API_PORT=8010 则用 8010）
# 接口文档 http://localhost:8000/docs

# 前端热更新开发（可选）：另开终端
# cd web && npm run dev
# 浏览器打开 http://localhost:5173 ，Vite 把 /api 代理到 FastAPI

# 调试请用 CLI，不要再开第二套 Web
python main.py -q "比亚迪2024年毛利率为什么下降" -c 002594
```

### Docker 一键部署

```bash
cp .env.example .env   # 填入 DEEPSEEK_API_KEY
docker-compose up -d --build

# 首次需初始化知识库
docker-compose exec api python -m rag.prepare_data
```

启动后：演示前端 `http://localhost:8000/`，API 文档 `http://localhost:8000/docs`。

### 运行评测

```bash
# 全量 9 道题（10~20 分钟）
python main.py --eval

# 只跑指定用例
python main.py --eval --cases 1,2,8
```

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/research` | 发起研究。`async_mode=true` 立即返回 session_id |
| GET | `/api/session/{id}` | 查询研究进度与结果 |
| GET | `/api/session/{id}/events` | SSE：节点进度（`progress` / `done`） |
| GET | `/api/trace/{run_id}` | 获取完整 trace（所有 LLM/工具/门禁事件） |
| POST | `/api/eval` | 运行自动化评测 |
| GET | `/api/health` | 健康检查（Redis / Chroma / LLM / MCP） |
| GET | `/api/tools` | 列出 MCP 工具（协议视图） |
| POST | `/api/mcp/{server}` | 直接发 JSON-RPC 2.0 请求（调试） |

```bash
curl -X POST http://localhost:8000/api/research \
  -H "Content-Type: application/json" \
  -d '{
    "question": "比亚迪2024年ROE是多少，相比2023年有什么变化",
    "company_ticker": "002594",
    "as_of_date": "2025-06-30"
  }'
```

---

## 工具清单

通过 MCP 协议语义接入（进程内实现，非远程传输），金融数据 12 个 + 知识库 1 个，共 13 个。

**financial_data server（节选）**

| 工具 | 用途 |
|------|------|
| `get_financial_metrics` | 最新核心指标（毛利率/净利率/ROE/增速/负债率/现金流质量） |
| `get_income_history` | 多期利润表 + 自动计算同比 |
| `get_balance_sheet` | 多期资产负债表 + 负债率/商誉占比 |
| `get_cash_flow` | 多期现金流量表 + 利润质量比值 |
| `get_stock_price` | 日 K 线 + 区间涨跌/波动率（腾讯前复权 → 新浪 → 东财） |
| `get_risk_snapshot` | 七条风险规则一次性检查 |
| `compare_periods` | 跨期对比 + 趋势判断 + 杜邦分解 |
| `search_news` | 新闻检索：AKShare 东财打底，条数不足（<3）时用 Tavily 补充并合并去重。证据等级最低，只能标 ⚠️ |

**knowledge_base server（1 个）**

| 工具 | 用途 |
|------|------|
| `search_knowledge` | 行业框架 / 分析方法 / 风险规则 / 披露制度检索 |

---

## 评测（全量 9 题，2026-08-14）

压缩修复后跑次 `eval_20260814_115957`：

| 维度 | 得分 |
|------|------|
| 事实准确性 | 0.964 |
| 归因质量 | 0.833 |
| 拒答校准 | 1.000 |
| 证据覆盖 | 0.895 |
| 检索效率 | 0.889 |
| **加权总分** | **0.923** |

通过率 **9/9**（阈值 0.60）。归因分随 Judge 波动，后续改动后应重跑。明细：`eval/results/eval_20260814_115957.md`。

运行：`python main.py --eval`；门禁等纯逻辑单测：`pytest`。

---

## 项目结构

```
finance_research_agent/
├── harness/            自建 Agent 运行时（不依赖 LangGraph）
│   ├── agent_loop.py       ReAct 循环 + max_turns/token 保护
│   ├── tool_registry.py    注册 + 参数校验 + 30s 超时 + 错误归类
│   ├── evidence_gate.py    三级证据门禁 + 金融业务规则
│   ├── context_compact.py  压缩过程性对话，保留工具结果
│   ├── tracing.py          结构化 JSON trace
│   └── types.py            全层数据契约
├── llm/                DeepSeek 客户端 + 两层路由（quick/deep）
├── tools/              AKShare 工具 + 跨期对比 + 两级缓存
├── rag/                切分 / Chroma / BM25+RRF / CrossEncoder / 建库脚本
├── mcp_servers/        MCP 协议层（JSON-RPC 2.0 语义 + 进程内传输）
├── agents/             Planner / Retriever / Verifier / Writer
├── graph/              LangGraph StateGraph + 条件边
├── api/                FastAPI（schemas / dependencies / routes / app）
├── session/            Redis 会话 + 异步工具缓存
├── data/               财务数据模型 + 指标计算规则 + 知识库持久化目录
├── eval/               9 道测试题 + LLM-as-Judge 评测 + 失败案例记录
├── docs/               架构说明 + 面试话术
└── traces/             运行 trace（自动生成）
```

---

## 模型路由

| Agent | 层级 | 模型 | 为什么 |
|-------|------|------|--------|
| Planner | deep | deepseek-reasoner | 问题拆解决定后续所有节点的行为，错了全白费 |
| Retriever | quick | deepseek-chat | 高频调用；且 reasoner 不支持 function calling |
| Verifier | deep | deepseek-reasoner | 断言抽取错配最难发现，代价最高 |
| Writer | quick | deepseek-chat | 模板固定的模式化任务 |
| Judge | deep | deepseek-reasoner | 评测需要严格推理和可重复性 |

---

## 可观测性

每次运行落盘一份 `traces/run_{run_id}_{时间戳}.json`：

```json
{
  "summary": {
    "run_id": "a3f2...", "total_duration_ms": 48210, "total_tokens": 18432,
    "llm_call_count": 7, "tool_call_count": 5, "tool_failure_count": 1,
    "gate_check_count": 12, "gate_blocked_count": 2,
    "node_path": ["planner", "retriever", "verifier", "writer"]
  },
  "events": [
    {"event_type": "tool_call", "agent_name": "get_income_history",
     "duration_ms": 2341, "success": true, "metadata": {"from_cache": false}},
    {"event_type": "gate_check", "agent_name": "evidence_gate",
     "output_summary": "number_mismatch: 数字与来源不一致", "success": false}
  ]
}
```

能直接回答："哪一步慢"、"哪个工具挂了"、"哪句话被门禁拦了、为什么"。

---

## 已知局限

诚实列在这里，详细分析见 [`eval/failure_log.md`](eval/failure_log.md)：

1. **披露日期是估算值**——用法定截止日（年报=次年 4/30）而非真实披露日。
   偏差方向是保守的（宁可误拦，不可漏拦），但会误拦真实披露日到法定截止日之间的数据。
2. **只取年报，不取季报**——因为指标体系全是同比口径，混入季报会让同比变环比。
3. **杜邦分析用期末总资产**——非期初期末平均值，资产快速扩张的公司有偏差，口径已在研报中声明。
4. **CrossEncoder 是英文模型**——`ms-marco-MiniLM-L-6-v2` 对中文精排弱于
   `bge-reranker-base`，这里为了体积（80MB vs 1.1GB）做了取舍。
5. **AKShare 接口不稳定**——已加超时、重试、两级缓存、列名模糊匹配，但仍可能失败；
   失败时研报会显式标注"该数据获取失败"，不会静默跳过。
6. **数据覆盖面尚不完整**——10 个分析维度里，分部结构、运营效率、同业对比
   三个维度当前为空；销量/单价/折旧摊销明细则是 AKShare 的硬边界（拿不到）。
   这意味着制造业的**量价拆解做不了**，归因只能停在成本与收入的增速对比。
   完整盘点、能力边界和待办优先级见 [`docs/data_coverage.md`](docs/data_coverage.md)。

---

## 文档

- [`docs/architecture.md`](docs/architecture.md) — 架构详解与设计权衡
- [`docs/data_coverage.md`](docs/data_coverage.md) — **数据覆盖度盘点**：按分析维度列出覆盖状态、能力边界与待办
- [`docs/interview_qa.md`](docs/interview_qa.md) — 面试问答准备
- [`eval/failure_log.md`](eval/failure_log.md) — 真实失败案例与校准记录

---

## 免责声明

本项目输出的是公开财报数据的整理与核查结果，**不构成任何投资建议**。
数据来源为 AKShare 抓取的公开信息，可能存在延迟或错误。
投资决策请以上市公司正式披露文件为准。
