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
import { Bot, PieChart, RefreshCw, Send, TrendingUp, BarChart2, Activity } from "lucide-react";
import "./style.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const REQUEST_TIMEOUT_MS = 60000;
const MAX_RETRIES = 1;
const BACKTEST_POLL_INTERVAL_MS = 3000;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type AllocationPoint = { ticker: string; weight: number };
type CurvePoint = { date: string; value: number };

type OptimizeResponse = {
  tickers: string[];
  allocation: AllocationPoint[];
  metrics: Record<string, number>;
  cumulative_returns: CurvePoint[];
  generated_at: string;
};

type ChatMessage = { role: "user" | "assistant"; text: string };
type RagResponse = {
  answer: string;
  sources: Array<Record<string, unknown>>;
  history: Array<{ user: string; assistant: string }>;
};

type BacktestMetrics = {
  annual_return: number;
  annual_volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
  benchmark_annual_return: number;
  final_cumulative_return: number;
};

type BacktestResult = {
  tickers: string[];
  cumulative_returns: CurvePoint[];
  metrics: BacktestMetrics;
  weights_history: Array<{ date: string; weights: Record<string, number> }>;
};

type SensitivityRow = { multiplier: number; weights: Record<string, number> };
type SensitivityResponse = { tickers: string[]; sensitivity: SensitivityRow[] };
type FFFactors = { factors: Record<string, number> };
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

async function postJsonWithRetry<TResponse>(
  endpoint: string,
  body: unknown,
): Promise<TResponse> {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt += 1) {
    try {
      const res = await fetchWithTimeout(
        `${API_URL}${endpoint}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
        REQUEST_TIMEOUT_MS,
      );
      const data = (await res.json()) as TResponse | ApiError;
      if (!res.ok) {
        throw new Error((data as ApiError).detail || "Request failed");
      }
      return data as TResponse;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error("Unknown request error");
      if (attempt === MAX_RETRIES) break;
      await new Promise((resolve) => window.setTimeout(resolve, 600));
    }
  }
  throw lastError || new Error("Request failed");
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
// Colour palette for multi-line charts
// ---------------------------------------------------------------------------

const LINE_COLORS = [
  "#4f7cff", "#f97316", "#22c55e", "#a78bfa", "#f43f5e",
  "#06b6d4", "#eab308", "#ec4899", "#14b8a6", "#8b5cf6",
];

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

function App() {
  // --- Optimizer ---
  const [tickers, setTickers] = useState(
    "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS,SBIN.NS,LT.NS,AXISBANK.NS",
  );
  const [period, setPeriod] = useState("1y");
  const [useXgb, setUseXgb] = useState(false);
  const [result, setResult] = useState<OptimizeResponse | null>(null);
  const [loading, setLoading] = useState(false);

  // --- Backtest ---
  const [btPeriod, setBtPeriod] = useState("5y");
  const [btUseViews, setBtUseViews] = useState(true);
  const [btLoading, setBtLoading] = useState(false);
  const [btResult, setBtResult] = useState<BacktestResult | null>(null);
  const [btError, setBtError] = useState("");
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // --- Sensitivity ---
  const [sensLoading, setSensLoading] = useState(false);
  const [sensData, setSensData] = useState<SensitivityResponse | null>(null);

  // --- FF Factors ---
  const [ffLoading, setFfLoading] = useState(false);
  const [ffData, setFfData] = useState<FFFactors | null>(null);

  // --- RAG Chatbot ---
  const [question, setQuestion] = useState(
    "Explain how this Black-Litterman RAG app works.",
  );
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [ragHistory, setRagHistory] = useState<Array<{ user: string; assistant: string }>>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  // ---------------------------------------------------------------------------
  // Optimizer
  // ---------------------------------------------------------------------------

  async function optimize() {
    setLoading(true);
    setResult(null);
    try {
      const data = await postJsonWithRetry<OptimizeResponse>("/optimize-portfolio", {
        tickers: tickers.split(",").map((v) => v.trim()).filter(Boolean),
        period,
        use_xgb_views: useXgb,
      });
      setResult(data);
    } catch (error) {
      alert(error instanceof Error ? error.message : "Optimization failed");
    } finally {
      setLoading(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Backtest polling
  // ---------------------------------------------------------------------------

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  async function startBacktest() {
    stopPolling();
    setBtLoading(true);
    setBtResult(null);
    setBtError("");
    try {
      const { job_id } = await postJsonWithRetry<{ job_id: string }>("/backtest", {
        tickers: tickers.split(",").map((v) => v.trim()).filter(Boolean),
        period: btPeriod,
        use_views: btUseViews,
      });

      pollRef.current = setInterval(async () => {
        try {
          const status = await getJson<{
            status: string;
            result?: BacktestResult;
            error?: string;
          }>(`/backtest/${job_id}`);

          if (status.status === "done" && status.result) {
            stopPolling();
            setBtResult(status.result);
            setBtLoading(false);
          } else if (status.status === "error") {
            stopPolling();
            setBtError(status.error || "Backtest failed");
            setBtLoading(false);
          }
        } catch {
          stopPolling();
          setBtError("Polling failed");
          setBtLoading(false);
        }
      }, BACKTEST_POLL_INTERVAL_MS);
    } catch (error) {
      setBtError(error instanceof Error ? error.message : "Backtest start failed");
      setBtLoading(false);
    }
  }

  // Cleanup on unmount
  useEffect(() => () => stopPolling(), [stopPolling]);

  // ---------------------------------------------------------------------------
  // Sensitivity
  // ---------------------------------------------------------------------------

  async function runSensitivity() {
    setSensLoading(true);
    setSensData(null);
    try {
      const data = await postJsonWithRetry<SensitivityResponse>("/sensitivity", {
        tickers: tickers.split(",").map((v) => v.trim()).filter(Boolean),
        period,
      });
      setSensData(data);
    } catch (error) {
      alert(error instanceof Error ? error.message : "Sensitivity failed");
    } finally {
      setSensLoading(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Fama-French
  // ---------------------------------------------------------------------------

  async function fetchFF() {
    setFfLoading(true);
    setFfData(null);
    try {
      const data = await getJson<FFFactors>("/ff-factors");
      setFfData(data);
    } catch (error) {
      alert(error instanceof Error ? error.message : "FF factors failed");
    } finally {
      setFfLoading(false);
    }
  }

  // ---------------------------------------------------------------------------
  // RAG Chatbot
  // ---------------------------------------------------------------------------

  async function ask() {
    if (!question.trim()) return;
    const userMsg = question.trim();
    setChatHistory((h) => [...h, { role: "user", text: userMsg }]);
    setQuestion("");
    setChatLoading(true);
    try {
      const data = await postJsonWithRetry<RagResponse>("/ask-rag", {
        question: userMsg,
        k: 5,
        history: ragHistory,
      });
      setChatHistory((h) => [...h, { role: "assistant", text: data.answer }]);
      setRagHistory(data.history);
    } catch (error) {
      setChatHistory((h) => [
        ...h,
        { role: "assistant", text: error instanceof Error ? error.message : "RAG failed" },
      ]);
    } finally {
      setChatLoading(false);
    }
  }

  function handleChatKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!chatLoading) ask();
    }
  }

  // ---------------------------------------------------------------------------
  // Sensitivity chart data
  // ---------------------------------------------------------------------------

  const sensitivityChartData = sensData
    ? sensData.sensitivity.map((row) => ({
        multiplier: row.multiplier,
        ...row.weights,
      }))
    : [];

  const top10Tickers = sensData
    ? (() => {
        const lastRow = sensData.sensitivity[sensData.sensitivity.length - 1];
        return Object.entries(lastRow.weights)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 10)
          .map(([t]) => t);
      })()
    : [];

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  const alloc = result?.allocation || [];
  const metrics = result?.metrics || {};

  return (
    <main>
      {/* Hero */}
      <section className="hero">
        <div>
          <p className="eyebrow">Live Portfolio Intelligence</p>
          <h1>Black-Litterman Portfolio Optimizer + RAG Chatbot</h1>
          <p className="sub">
            Full-stack faithful rebuild of the notebook — XGBoost views, rolling backtest,
            Fama-French factors, sensitivity analysis, and a grounded RAG chatbot.
          </p>
        </div>
        <div className="status">{`Backend: ${API_URL}`}</div>
      </section>

      {/* ---- Optimizer ---- */}
      <section className="grid">
        <div className="card wide">
          <div className="cardTitle">
            <PieChart size={20} /> Portfolio Optimizer
          </div>
          <label>Tickers (comma-separated)</label>
          <textarea value={tickers} onChange={(e) => setTickers(e.target.value)} />
          <label>Data period</label>
          <select value={period} onChange={(e) => setPeriod(e.target.value)}>
            <option>6mo</option>
            <option>1y</option>
            <option>2y</option>
            <option>5y</option>
          </select>
          <div className="checkRow">
            <input
              id="xgb-toggle"
              type="checkbox"
              checked={useXgb}
              onChange={(e) => setUseXgb(e.target.checked)}
            />
            <label htmlFor="xgb-toggle" className="checkLabel">
              Use XGBoost Views (slower, more accurate)
            </label>
          </div>
          <button onClick={optimize} disabled={loading}>
            {loading ? <RefreshCw className="spin" size={16} /> : null}
            Run Optimization
          </button>
        </div>

        <div className="card">
          <div className="cardTitle">Metrics</div>
          {Object.keys(metrics).length === 0 ? (
            <p className="muted">Run optimization to see metrics.</p>
          ) : (
            <div className="metrics">
              {Object.entries(metrics).map(([key, value]) => (
                <div key={key}>
                  <span>{key.split("_").join(" ")}</span>
                  <strong>{Number(value).toFixed(4)}</strong>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      {result && (
        <section className="grid">
          <div className="card">
            <div className="cardTitle">Allocation</div>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={alloc}>
                <XAxis dataKey="ticker" hide />
                <YAxis />
                <Tooltip formatter={(v) => (typeof v === "number" ? v.toFixed(4) : v)} />
                <Bar dataKey="weight" fill="#4f7cff" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="card">
            <div className="cardTitle">Cumulative Returns</div>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={result.cumulative_returns}>
                <XAxis dataKey="date" hide />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="value" dot={false} stroke="#4f7cff" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      {/* ---- Backtest ---- */}
      <section className="card" style={{ marginBottom: 18 }}>
        <div className="cardTitle">
          <TrendingUp size={20} /> Rolling Window Backtest
        </div>
        <p className="muted" style={{ marginBottom: 12, fontSize: 13 }}>
          Runs a full walk-forward backtest on the tickers above. Uses XGBoost views when enabled.
          Results appear when the job completes (may take 1–3 minutes for many tickers).
        </p>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <div>
            <label>Backtest period</label>
            <select value={btPeriod} onChange={(e) => setBtPeriod(e.target.value)} style={{ minHeight: 40, width: 100 }}>
              <option>1y</option>
              <option>2y</option>
              <option>3y</option>
              <option>5y</option>
            </select>
          </div>
          <div className="checkRow" style={{ marginTop: 18 }}>
            <input
              id="bt-views"
              type="checkbox"
              checked={btUseViews}
              onChange={(e) => setBtUseViews(e.target.checked)}
            />
            <label htmlFor="bt-views" className="checkLabel">Use Views</label>
          </div>
          <button onClick={startBacktest} disabled={btLoading} style={{ marginTop: 18 }}>
            {btLoading ? <RefreshCw className="spin" size={16} /> : <Activity size={16} />}
            {btLoading ? "Running…" : "Run Backtest"}
          </button>
        </div>
        {btError && <p style={{ color: "#f87171", marginTop: 8 }}>{btError}</p>}

        {btResult && (
          <>
            <div style={{ marginTop: 20 }}>
              <div className="cardTitle" style={{ fontSize: 15 }}>Backtest Cumulative Returns</div>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={btResult.cumulative_returns}>
                  <XAxis dataKey="date" hide />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="value" dot={false} stroke="#22c55e" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div style={{ marginTop: 16 }}>
              <div className="cardTitle" style={{ fontSize: 15 }}>Backtest Metrics</div>
              <div className="metrics">
                {Object.entries(btResult.metrics).map(([k, v]) => (
                  <div key={k}>
                    <span>{k.split("_").join(" ")}</span>
                    <strong>{Number(v).toFixed(4)}</strong>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </section>

      {/* ---- Sensitivity Analysis ---- */}
      <section className="card" style={{ marginBottom: 18 }}>
        <div className="cardTitle">
          <BarChart2 size={20} /> Sensitivity Analysis
        </div>
        <p className="muted" style={{ marginBottom: 12, fontSize: 13 }}>
          Sweeps the view multiplier from 0.5× to 1.5× to show how portfolio weights respond to
          view confidence changes. Uses the tickers and period from the optimizer above.
        </p>
        <button onClick={runSensitivity} disabled={sensLoading}>
          {sensLoading ? <RefreshCw className="spin" size={16} /> : null}
          Run Sensitivity Sweep
        </button>

        {sensData && sensitivityChartData.length > 0 && (
          <>
            <div style={{ marginTop: 16, fontSize: 13, color: "#b8c0d9", marginBottom: 6 }}>
              Top 10 tickers at multiplier = 1.5×
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={sensitivityChartData}>
                <XAxis dataKey="multiplier" tickFormatter={(v: number) => v.toFixed(1)} />
                <YAxis />
                <Tooltip formatter={(v) => (typeof v === "number" ? v.toFixed(4) : v)} />
                <Legend />
                {top10Tickers.map((t, i) => (
                  <Line
                    key={t}
                    type="monotone"
                    dataKey={t}
                    dot={false}
                    stroke={LINE_COLORS[i % LINE_COLORS.length]}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </>
        )}
      </section>

      {/* ---- Fama-French Factors ---- */}
      <section className="card" style={{ marginBottom: 18 }}>
        <div className="cardTitle">
          <BarChart2 size={20} /> Fama-French 5 Factors
        </div>
        <p className="muted" style={{ marginBottom: 12, fontSize: 13 }}>
          Computes SMB, HML, RMW, CMA, and Mkt-RF from NIFTY50 constituents using monthly data.
        </p>
        <button onClick={fetchFF} disabled={ffLoading}>
          {ffLoading ? <RefreshCw className="spin" size={16} /> : null}
          Load FF Factors
        </button>
        {ffData && (
          <div className="metrics" style={{ marginTop: 14 }}>
            {Object.entries(ffData.factors).map(([k, v]) => (
              <div key={k}>
                <span>{k}</span>
                <strong style={{ color: Number(v) >= 0 ? "#22c55e" : "#f87171" }}>
                  {Number(v).toFixed(6)}
                </strong>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* ---- RAG Chatbot ---- */}
      <section className="card chat">
        <div className="cardTitle">
          <Bot size={20} /> RAG Chatbot
        </div>
        <p className="muted" style={{ fontSize: 13, marginBottom: 12 }}>
          Grounded on the notebook methodology, backtest results, and indexed documents.
          Conversation history is maintained across turns.
        </p>

        <div className="chatThread">
          {chatHistory.length === 0 && (
            <p className="muted" style={{ fontSize: 13 }}>
              Ask anything about the Black-Litterman model, backtest results, or portfolio methodology.
            </p>
          )}
          {chatHistory.map((msg, i) => (
            <div key={i} className={`chatBubble ${msg.role}`}>
              <span className="chatRole">{msg.role === "user" ? "You" : "Assistant"}</span>
              <pre className="chatText">{msg.text}</pre>
            </div>
          ))}
          {chatLoading && (
            <div className="chatBubble assistant">
              <span className="chatRole">Assistant</span>
              <p className="muted" style={{ margin: 0, fontSize: 13 }}>
                <RefreshCw className="spin" size={12} style={{ marginRight: 6 }} />
                Thinking…
              </p>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleChatKey}
          placeholder="Ask about backtest metrics, model assumptions, Fama-French factors… (Enter to send)"
        />
        <button onClick={ask} disabled={chatLoading || !question.trim()}>
          {chatLoading ? <RefreshCw className="spin" size={16} /> : <Send size={16} />}
          Ask
        </button>
        {chatHistory.length > 0 && (
          <button
            onClick={() => { setChatHistory([]); setRagHistory([]); }}
            style={{ marginLeft: 10, background: "#263247" }}
          >
            Clear Chat
          </button>
        )}
      </section>
    </main>
  );
}

const root = document.getElementById("root");
if (!root) throw new Error("Root element not found");
createRoot(root).render(<App />);
