export const COMPANIES = [
  { ticker: "002594", name: "比亚迪" },
  { ticker: "600519", name: "贵州茅台" },
  { ticker: "000001", name: "平安银行" },
  { ticker: "000858", name: "五粮液" },
  { ticker: "601318", name: "中国平安" },
  { ticker: "300750", name: "宁德时代" },
];

export const EXAMPLES = [
  { question: "比亚迪2024年ROE是多少，相比2023年有什么变化", ticker: "002594" },
  { question: "比亚迪近三年毛利率变化趋势如何", ticker: "002594" },
  { question: "贵州茅台的净利率为什么这么高", ticker: "600519" },
  { question: "平安银行的资产质量有哪些风险点", ticker: "000001" },
];

export async function fetchHealth() {
  const response = await fetch("/api/health");
  if (!response.ok) throw new Error(`健康检查失败 ${response.status}`);
  return response.json();
}

export async function startResearch(payload) {
  const response = await fetch("/api/research", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, async_mode: true }),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error || body.detail || `提交失败 ${response.status}`);
  }
  return body;
}

export async function fetchTrace(runId) {
  const response = await fetch(`/api/trace/${runId}`);
  if (!response.ok) return null;
  return response.json();
}

export function subscribeSession(sessionId, { onProgress, onDone, onError }) {
  const source = new EventSource(`/api/session/${sessionId}/events`);
  source.addEventListener("progress", (event) => {
    onProgress(JSON.parse(event.data));
  });
  source.addEventListener("done", (event) => {
    onDone(JSON.parse(event.data));
    source.close();
  });
  source.addEventListener("error", (event) => {
    if (!event.data) return;
    try {
      onError(JSON.parse(event.data).error || "SSE 错误");
    } catch {
      onError("SSE 错误");
    }
    source.close();
  });
  return () => source.close();
}
