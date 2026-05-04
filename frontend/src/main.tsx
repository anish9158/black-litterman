import { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend,
} from "recharts";
import { Bot, TrendingUp, Send, RefreshCw } from "lucide-react";
import "./style.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const REQUEST_TIMEOUT_MS = 90000;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type ChatMessage = { role: "user" | "assistant"; text: string };

type NotebookModel = {
  name: string;
  annual_return: number;
  annual_volatility: number | null;
  sharpe_ratio: number | null;
  max_drawdown: number | null;
  excess_return_vs_benchmark: number;
};
type NotebookWeight = { ticker: string; weight: number };
type NotebookSensitivitySeries = { ticker: string; weights: number[] };
type NotebookResults = {
  benchmark: { name: string; annual_return: number; description: string };
  models: NotebookModel[];
  weights: NotebookWeight[];
  sensitivity: { multipliers: number[]; series: NotebookSensitivitySeries[] };
  notes: string;
};

type ApiError = { detail?: string };

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

async function fetchWithTimeout(
  input: RequestInfo | URL,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    window.clearTimeout(timeout);
  }
}

async function getJson<TResponse>(endpoint: string): Promise<TResponse> {
  const res = await fetchWithTimeout(
    `${API_URL}${endpoint}`,
    { method: "GET" },
    REQUEST_TIMEOUT_MS,
  );
  const data = (await res.json()) as TResponse | ApiError;
  if (!res.ok) throw new Error((data as ApiError).detail || "Request failed");
  return data as TResponse;
}

// ---------------------------------------------------------------------------
// Colour palette
// ---------------------------------------------------------------------------

const LINE_COLORS = [
  "#4f7cff", "#f97316", "#22c55e", "#a78bfa", "#f43f5e",
  "#06b6d4", "#eab308", "#ec4899", "#14b8a6", "#8b5cf6",
];

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

function App() {
  // --- Notebook pre-computed results ---
  const [nbResults, setNbResults] = useState<NotebookResults | null>(null);
  const [nbLoading, setNbLoading] = useState(false);

  async function loadNotebookResults() {
    setNbLoading(true);
    try {
      const data = await getJson<NotebookResults>("/notebook-results");
      setNbResults(data);
    } catch (e) {
      alert(e instanceof Error ? e.message : "Failed to load notebook results");
    } finally {
      setNbLoading(false);
    }
  }

  useEffect(() => { loadNotebookResults(); }, []);

  // --- RAG Chatbot ---
  const [question, setQuestion] = useState(
    "How does this Black-Litterman model outperform the NIFTY 50 benchmark?",
  );
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [ragHistory, setRagHistory] = useState<Array<{ user: string; assistant: string }>>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  // ---------------------------------------------------------------------------
  // Streaming RAG Chatbot
  // ---------------------------------------------------------------------------

  const handleChatKey = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        ask();
      }
    },
    [question, chatLoading],
  );

  async function ask() {
    if (!question.trim()) return;
    const userMsg = question.trim();
    setChatHistory((h) => [...h, { role: "user", text: userMsg }]);
    setQuestion("");
    setChatLoading(true);
    setChatHistory((h) => [...h, { role: "assistant", text: "" }]);

    try {
      const response = await fetchWithTimeout(
        `${API_URL}/ask-rag-stream`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: userMsg, k: 5, history: ragHistory }),
        },
        REQUEST_TIMEOUT_MS,
      );

      if (!response.ok || !response.body) {
        throw new Error("Stream request failed");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let fullText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const rawData = line.slice(6).trim();
          if (rawData === "[DONE]") break;
          try {
            const parsed = JSON.parse(rawData) as { token?: string; answer?: string };
            const token = parsed.token ?? parsed.answer ?? "";
            if (token) {
              fullText += token;
              setChatHistory((h) => {
                const updated = [...h];
                updated[updated.length - 1] = { role: "assistant", text: fullText };
                return updated;
              });
            }
          } catch {
            // ignore malformed SSE frames
          }
        }
      }

      setRagHistory((h) => [...h, { user: userMsg, assistant: fullText }]);
    } catch (error) {
      const msg = error instanceof Error ? error.message : "Request failed";
      setChatHistory((h) => {
        const updated = [...h];
        updated[updated.length - 1] = { role: "assistant", text: `Error: ${msg}` };
        return updated;
      });
    } finally {
      setChatLoading(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <main>
      {/* Hero */}
      <section className="hero">
        <div>
          <p className="eyebrow">AI Portfolio Intelligence</p>
          <h1>Black-Litterman Portfolio Optimizer</h1>
          <p className="sub">
            XGBoost views · Rolling-window backtest · Fama-French 5 factors ·
            Streaming RAG chatbot · NIFTY 50 benchmark outperformance
          </p>
        </div>
        <div className="status">{`Backend: ${API_URL}`}</div>
      </section>

      {/* ── Notebook Results (pre-computed, instant) ── */}
      <section className="card nbResults" style={{ marginBottom: 18 }}>
        <div className="cardTitle">
          <TrendingUp size={20} /> Backtest Results — Black-Litterman vs NIFTY 50
          <span className="streamBadge" style={{ marginLeft: 10 }}>Pre-computed</span>
        </div>
        <p className="muted" style={{ fontSize: 13, marginBottom: 12 }}>
          5-year rolling-window backtest (2019–2024) · XGBoost views · All 50 NIFTY stocks ·
          Benchmark: <strong>NIFTY 50 (^NSEI)</strong> — market-cap-weighted index of the
          50 largest companies listed on the NSE of India
        </p>

        {nbLoading && <p className="muted">Loading…</p>}

        {nbResults && (
          <>
            {/* Performance Metrics Table */}
            <div style={{ overflowX: "auto", marginBottom: 24 }}>
              <table className="nbTable">
                <thead>
                  <tr>
                    <th>Metric</th>
                    {nbResults.models.map((m) => (
                      <th key={m.name} className={m.name.startsWith("All 50") ? "highlight" : ""}>
                        {m.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[
                    {
                      label: "Annual Return",
                      key: "annual_return",
                      fmt: (v: number | null) => v != null ? `${(v * 100).toFixed(1)}%` : "—",
                    },
                    {
                      label: "Annual Volatility",
                      key: "annual_volatility",
                      fmt: (v: number | null) => v != null ? `${(v * 100).toFixed(1)}%` : "—",
                    },
                    {
                      label: "Sharpe Ratio",
                      key: "sharpe_ratio",
                      fmt: (v: number | null) => v != null ? v.toFixed(2) : "—",
                    },
                    {
                      label: "Max Drawdown",
                      key: "max_drawdown",
                      fmt: (v: number | null) => v != null ? `${(v * 100).toFixed(1)}%` : "—",
                    },
                    {
                      label: "Excess vs Benchmark",
                      key: "excess_return_vs_benchmark",
                      fmt: (v: number | null) =>
                        v != null ? `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%` : "—",
                    },
                  ].map((row) => (
                    <tr key={row.label}>
                      <td>{row.label}</td>
                      {nbResults.models.map((m) => (
                        <td
                          key={m.name}
                          className={m.name.startsWith("All 50") ? "highlight" : ""}
                        >
                          {row.fmt(
                            (m as unknown as Record<string, number | null>)[row.key] as number | null,
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Side-by-side: Weights + Sensitivity */}
            <div className="nbCharts">
              <div>
                <h3 className="chartTitle">Final Allocation (view multiplier s = 1.5)</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart
                    data={nbResults.weights.map((w) => ({
                      ticker: w.ticker.replace(".NS", ""),
                      weight: parseFloat((w.weight * 100).toFixed(1)),
                    }))}
                    layout="vertical"
                    margin={{ left: 80, right: 20, top: 4, bottom: 4 }}
                  >
                    <XAxis type="number" unit="%" tick={{ fontSize: 11 }} />
                    <YAxis type="category" dataKey="ticker" tick={{ fontSize: 11 }} width={80} />
                    <Tooltip formatter={(v) => `${v}%`} />
                    <Bar dataKey="weight" fill="#4f7cff" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div>
                <h3 className="chartTitle">Sensitivity — Allocation vs View Multiplier s</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart
                    data={nbResults.sensitivity.multipliers.map((s, i) => {
                      const pt: Record<string, number> = { s };
                      nbResults.sensitivity.series.forEach((ser) => {
                        pt[ser.ticker.replace(".NS", "")] = parseFloat(
                          (ser.weights[i] * 100).toFixed(1),
                        );
                      });
                      return pt;
                    })}
                    margin={{ left: 10, right: 20, top: 4, bottom: 20 }}
                  >
                    <XAxis
                      dataKey="s"
                      label={{ value: "Multiplier s", position: "insideBottom", offset: -10 }}
                      tick={{ fontSize: 11 }}
                    />
                    <YAxis unit="%" tick={{ fontSize: 11 }} />
                    <Tooltip formatter={(v) => `${v}%`} />
                    <Legend />
                    {nbResults.sensitivity.series.map((ser, i) => (
                      <Line
                        key={ser.ticker}
                        type="monotone"
                        dataKey={ser.ticker.replace(".NS", "")}
                        stroke={LINE_COLORS[i % LINE_COLORS.length]}
                        dot={false}
                        strokeWidth={2}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <p className="muted" style={{ fontSize: 12, marginTop: 16 }}>
              {nbResults.notes}
            </p>
          </>
        )}
      </section>

      {/* ── RAG Chatbot ── */}
      <section className="card chat">
        <div className="cardTitle">
          <Bot size={20} /> Ask the Model — RAG Chatbot
          <span className="streamBadge">Streaming</span>
        </div>
        <p className="muted" style={{ fontSize: 13, marginBottom: 12 }}>
          Grounded on the notebook methodology, backtest results, and NIFTY 50 market context.
          Only answers questions about this project — off-topic questions are refused.
          Responses stream token-by-token.
        </p>

        <div className="chatThread">
          {chatHistory.length === 0 && (
            <p className="muted" style={{ fontSize: 13 }}>
              Try: "How does this model beat the NIFTY 50?" · "Explain the Black-Litterman model" ·
              "What is the Sharpe ratio?" · "Why was ICICIBANK given the highest weight?"
            </p>
          )}
          {chatHistory.map((msg, i) => (
            <div key={i} className={`chatBubble ${msg.role}`}>
              <span className="chatRole">{msg.role === "user" ? "You" : "Assistant"}</span>
              <pre className="chatText">{msg.text}</pre>
            </div>
          ))}
          {chatLoading && chatHistory[chatHistory.length - 1]?.text === "" && (
            <div className="streamCursor" />
          )}
          <div ref={chatEndRef} />
        </div>

        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleChatKey}
          placeholder="Ask about the model, backtest results, benchmark outperformance… (Enter to send)"
        />
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <button onClick={ask} disabled={chatLoading || !question.trim()}>
            {chatLoading ? <RefreshCw className="spin" size={16} /> : <Send size={16} />}
            Ask
          </button>
          {chatHistory.length > 0 && (
            <button
              onClick={() => { setChatHistory([]); setRagHistory([]); }}
              style={{ background: "#263247" }}
            >
              Clear Chat
            </button>
          )}
        </div>
      </section>
    </main>
  );
}

const root = document.getElementById("root");
if (root) createRoot(root).render(<App />);
