<script setup>
import { computed, onMounted, onUnmounted, ref } from "vue";
import { marked } from "marked";
import {
  COMPANIES,
  EXAMPLES,
  fetchHealth,
  fetchTrace,
  startResearch,
  subscribeSession,
} from "./api.js";

const NODE_ORDER = ["planner", "retriever", "verifier", "writer"];
const NODES = [
  { id: "planner", label: "规划" },
  { id: "retriever", label: "检索" },
  { id: "verifier", label: "核查" },
  { id: "writer", label: "撰写" },
];

const question = ref("比亚迪2024年ROE是多少，相比2023年有什么变化");
const ticker = ref("002594");
const asOfDate = ref(new Date().toISOString().slice(0, 10));
const running = ref(false);
const errorText = ref("");
const health = ref(null);
const session = ref(null);
const traceEvents = ref([]);
let unsubscribe = null;

const progressFloor = ref(0);

const progressPct = computed(() => {
  const raw = session.value?.status === "completed" ? 1 : session.value?.progress || 0;
  const value = Math.max(progressFloor.value, raw);
  return Math.round((value * 100 + Number.EPSILON) * 10) / 10;
});

const reportHtml = computed(() => {
  const markdown = session.value?.result?.markdown;
  if (!markdown) return "";
  return decorateReport(marked.parse(markdown, { async: false }));
});

const stats = computed(() => session.value?.result?.stats || {});
const summary = computed(() => session.value?.trace_summary || {});

function applySession(payload) {
  session.value = payload;
  const next = payload?.status === "completed" ? 1 : payload?.progress || 0;
  if (next > progressFloor.value) progressFloor.value = next;
}

function decorateReport(html) {
  const doc = new DOMParser().parseFromString(`<div class="md-root">${html}</div>`, "text/html");
  const root = doc.body.querySelector(".md-root");
  if (!root) return html;
  root.querySelectorAll("li").forEach((li) => {
    const text = li.textContent || "";
    if (text.includes("✅") || text.includes("[已验证]")) li.classList.add("claim-verified");
    else if (text.includes("❌") || text.includes("[拒答]")) li.classList.add("claim-refused");
    else if (text.includes("⚠️") || text.includes("[未验证]")) li.classList.add("claim-unverified");
  });
  root.querySelectorAll("tr").forEach((tr) => {
    const text = tr.textContent || "";
    if (text.includes("✅")) tr.classList.add("row-verified");
    else if (text.includes("❌")) tr.classList.add("row-refused");
    else if (text.includes("⚠️")) tr.classList.add("row-unverified");
  });
  return root.innerHTML;
}

function displayStage() {
  const current = session.value?.current_node || "";
  const path = session.value?.node_path || [];
  let farthest = NODE_ORDER.indexOf(current);
  for (const node of path) {
    farthest = Math.max(farthest, NODE_ORDER.indexOf(node));
  }
  return farthest >= 0 ? NODE_ORDER[farthest] : current;
}

function nodeState(id) {
  if (session.value?.status === "completed") return "done";
  const display = displayStage();
  const idx = NODE_ORDER.indexOf(id);
  const displayIdx = NODE_ORDER.indexOf(display);
  if (id === display && session.value?.status === "running") return "active";
  if (displayIdx > idx) return "done";
  return "idle";
}

async function loadHealth() {
  try {
    health.value = await fetchHealth();
  } catch (exc) {
    health.value = { status: "error", detail: String(exc) };
  }
}

function applyExample(item) {
  question.value = item.question;
  ticker.value = item.ticker;
}

async function submit() {
  errorText.value = "";
  session.value = null;
  traceEvents.value = [];
  progressFloor.value = 0;
  if (unsubscribe) {
    unsubscribe();
    unsubscribe = null;
  }
  const code = (ticker.value || "").replace(/\D/g, "").slice(0, 6);
  if (!question.value.trim()) {
    errorText.value = "请输入研究问题";
    return;
  }
  if (code.length !== 6) {
    errorText.value = "请输入 6 位 A 股代码";
    return;
  }
  running.value = true;
  try {
    const created = await startResearch({
      question: question.value.trim(),
      company_ticker: code,
      as_of_date: asOfDate.value || null,
    });
    unsubscribe = subscribeSession(created.session_id, {
      onProgress: (payload) => {
        applySession(payload);
      },
      onDone: async (payload) => {
        applySession(payload);
        running.value = false;
        if (payload.run_id) {
          const trace = await fetchTrace(payload.run_id);
          traceEvents.value = (trace?.events || []).slice(-40);
        }
      },
      onError: (message) => {
        errorText.value = message;
        running.value = false;
      },
    });
  } catch (exc) {
    errorText.value = String(exc.message || exc);
    running.value = false;
  }
}

onMounted(loadHealth);
onUnmounted(() => {
  if (unsubscribe) unsubscribe();
});
</script>

<template>
  <div class="shell">
    <header class="top">
      <div>
        <p class="kicker">Finance Research Agent</p>
        <h1>金融研报</h1>
      </div>
      <div class="health" :data-status="health?.status || 'unknown'">
        <span class="dot" />
        <span>{{ health?.status || "检测中" }}</span>
        <button class="link" type="button" @click="loadHealth">刷新</button>
      </div>
    </header>

    <section class="composer">
      <label>
        研究问题
        <textarea v-model="question" rows="2" placeholder="例如：贵州茅台净利率为什么这么高" />
      </label>
      <div class="row">
        <label>
          公司
          <select v-model="ticker">
            <option v-for="item in COMPANIES" :key="item.ticker" :value="item.ticker">
              {{ item.ticker }} {{ item.name }}
            </option>
          </select>
        </label>
        <label>
          代码（可改）
          <input v-model="ticker" maxlength="6" />
        </label>
        <label>
          分析基准日
          <input v-model="asOfDate" type="date" />
        </label>
        <button class="primary" type="button" :disabled="running" @click="submit">
          {{ running ? "研究中…" : "开始研究" }}
        </button>
      </div>
      <div class="examples">
        <button
          v-for="item in EXAMPLES"
          :key="item.question"
          type="button"
          class="chip"
          @click="applyExample(item)"
        >
          {{ item.question }}
        </button>
      </div>
    </section>

    <section v-if="session || running" class="pipeline">
      <div class="pipeline-head">
        <strong>{{ session?.progress_label || "已提交，等待规划" }}</strong>
        <span>{{ progressPct }}%</span>
      </div>
      <div class="bar"><i :style="{ width: progressPct + '%' }" /></div>
      <ol class="nodes">
        <li v-for="node in NODES" :key="node.id" :data-state="nodeState(node.id)">
          {{ node.label }}
        </li>
      </ol>
      <p v-if="session?.node_path?.length" class="path">
        {{ (session.node_path || []).join(" → ") }}
      </p>
    </section>

    <p v-if="errorText" class="error">{{ errorText }}</p>
    <p v-if="session?.status === 'failed'" class="error">{{ session.error || "研究失败" }}</p>

    <main class="workspace">
      <article class="report">
        <div v-if="reportHtml" class="markdown" v-html="reportHtml" />
        <p v-else class="placeholder">研报将显示在这里。结论带三级标注：已验证 / 未验证 / 拒答。</p>
      </article>
      <aside class="meta">
        <div class="card">
          <h2>结论统计</h2>
          <ul>
            <li>已验证 {{ stats.verified ?? "—" }}</li>
            <li>未验证 {{ stats.unverified ?? "—" }}</li>
            <li>拒答 {{ stats.refused ?? "—" }}</li>
          </ul>
        </div>
        <div class="card">
          <h2>本次运行</h2>
          <ul>
            <li>run_id {{ session?.run_id || "—" }}</li>
            <li>耗时 {{ ((summary.total_duration_ms || 0) / 1000).toFixed(1) }}s</li>
            <li>Token {{ summary.total_tokens ?? 0 }}</li>
            <li>LLM {{ summary.llm_call_count ?? 0 }} / 工具 {{ summary.tool_call_count ?? 0 }}</li>
            <li>门禁 {{ summary.gate_check_count ?? 0 }}（拦截 {{ summary.gate_blocked_count ?? 0 }}）</li>
          </ul>
        </div>
        <div class="card trace">
          <h2>Trace</h2>
          <table v-if="traceEvents.length">
            <thead>
              <tr>
                <th>事件</th>
                <th>节点</th>
                <th>耗时</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(event, index) in traceEvents" :key="index">
                <td>{{ event.event_type }}</td>
                <td>{{ event.agent_name }}</td>
                <td>{{ event.duration_ms }}ms</td>
              </tr>
            </tbody>
          </table>
          <p v-else class="placeholder">完成后显示关键事件</p>
        </div>
      </aside>
    </main>
  </div>
</template>
