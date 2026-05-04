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
import {
  Bot,
  PieChart,
  RefreshCw,
  Send,
  TrendingUp,
  BarChart2,
  Activity,
  Sparkles,
  Newspaper,
  MessageSquare,
  Download,
} from "lucide-react";
import "./style.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const REQUEST_TIMEOUT_MS = 90000;
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
  narrative?: string;
};

type ShapFeature = {
  feature: string;
  shap_value: number;
  mean_abs_shap: number;
  feature_value: number;
};
type ExplainResponse = {
  tickers: string[];
  explanations: Record<string, ShapFeature[]>;
};

type SentimentItem = {
  ticker: string;
  score: number;
  signal: "bullish" | "neutral" | "bearish";
  headlines_count: number;
  headlines: string[];
  error?: string;
};
type SentimentResponse = {
  sentiment: Record<string, SentimentItem>;
  portfolio_score: number;
  portfolio_signal: string;
};

type ChatMessage = { role: "user" | "assistant"; text: string };

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
// Colour palette
// ---------------------------------------------------------------------------

const LINE_COLORS = [
  "#4f7cff", "#f97316", "#22c55e", "#a78bfa", "#f43f5e",
  "#06b6d4", "#eab308", "#ec4899", "#14b8a6", "#8b5cf6",
];

const SIGNAL_COLOR: Record<string, string> = {
  bullish: "#22c55e",
  neutral: "#eab308",
  bearish: "#f43f5e",
};

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

  // --- SHAP Explainability ---
  const [shapLoading, setShapLoading] = useState(false);
  const [shapData, setShapData] = useState<ExplainResponse | null>(null);
  const [shapTicker, setShapTicker] = useState("");

  // --- Sentiment ---
  const [sentLoading, setSentLoading] = useState(false);
  const [sentData, setSentData] = useState<SentimentResponse | null>(null);

  // --- NL Optimize ---
  const [nlPrompt, setNlPrompt] = useState("");
  const [nlLoading, setNlLoading] = useState(false);
  const [nlResult, setNlResult] = useState<OptimizeResponse | null>(null);

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
  // Optimizer
  // ---------------------------------------------------------------------------

  async function optimize() {
    setLoading(true);
    setResult(null);
    setShapData(null);
    setSentData(null);
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
  // SHAP Explainability
  // ---------------------------------------------------------------------------

  async function explainViews() {
    setShapLoading(true);
    setShapData(null);
    try {
      const data = await postJsonWithRetry<ExplainResponse>("/explain-views", {
        tickers: tickers.split(",").map((v) => v.trim()).filter(Boolean),
        period,
      });
      setShapData(data);
      if (data.tickers.length > 0) setShapTicker(data.tickers[0]);
    } catch (error) {
      alert(error instanceof Error ? error.message : "Explain views failed");
    } finally {
      setShapLoading(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Sentiment
  // ---------------------------------------------------------------------------

  async function runSentiment() {
    setSentLoading(true);
    setSentData(null);
    try {
      const data = await postJsonWithRetry<SentimentResponse>("/sentiment", {
        tickers: tickers.split(",").map((v) => v.trim()).filter(Boolean),
      });
      setSentData(data);
    } catch (error) {
      alert(error instanceof Error ? error.message : "Sentiment failed");
    } finally {
      setSentLoading(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Natural Language Optimizer
  // ---------------------------------------------------------------------------

  async function nlOptimize() {
    if (!nlPrompt.trim()) return;
    setNlLoading(true);
    setNlResult(null);
    try {
      const data = await postJsonWithRetry<OptimizeResponse>("/nl-optimize", {
        prompt: nlPrompt.trim(),
      });
      setNlResult(data);
    } catch (error) {
      alert(error instanceof Error ? error.message : "NL optimize failed");
    } finally {
      setNlLoading(false);
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
  // Streaming RAG Chatbot
  // ---------------------------------------------------------------------------

  async function ask() {
    if (!question.trim()) return;
    const userMsg = question.trim();
    setChatHistory((h) => [...h, { role: "user", text: userMsg }]);
    setQuestion("");
    setChatLoading(true);

    // Add empty placeholder for assistant message (will be updated token by token)
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
            const parsed = JSON.parse(rawData) as {
              token?: string;
              done?: boolean;
              sources?: unknown[];
              history?: Array<{ user: string; assistant: string }>;
              error?: string;
            };
            if (parsed.token) {
              fullText += parsed.token;
              const snapshot = fullText;
              setChatHistory((h) => {
                const updated = [...h];
                updated[updated.length - 1] = { role: "assistant", text: snapshot };
                return updated;
              });
            }
            if (parsed.done && parsed.history) {
              setRagHistory(parsed.history);
            }
          } catch {
            // malformed chunk — skip
          }
        }
      }
    } catch (error) {
      const errMsg = error instanceof Error ? error.message : "RAG stream failed";
      setChatHistory((h) => {
        const updated = [...h];
        updated[updated.length - 1] = { role: "assistant", text: errMsg };
        return updated;
      });
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
  // PDF Report Download
  // ---------------------------------------------------------------------------

  async function downloadReport(sourceResult: OptimizeResponse) {
    try {
      const res = await fetchWithTimeout(
        `${API_URL}/generate-report`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            tickers: sourceResult.tickers,
            allocation: sourceResult.allocation,
            metrics: sourceResult.metrics,
            narrative: sourceResult.narrative || "",
            shap_data: shapData?.explanations || null,
          }),
        },
        REQUEST_TIMEOUT_MS,
      );
      if (!res.ok) throw new Error("Report generation failed");
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "portfolio_report.pdf";
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      alert(error instanceof Error ? error.message : "Report download failed");
    }
  }

  // ---------------------------------------------------------------------------
  // Derived data for charts
  // ---------------------------------------------------------------------------

  const sensitivityChartData = sensData
    ? sensData.sensitivity.map((row) => ({ multiplier: row.multiplier, ...row.weights }))
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

  const shapBarData = shapData && shapTicker && shapData.explanations[shapTicker]
    ? shapData.explanations[shapTicker]
        .slice(0, 8)
        .map((f) => ({ feature: f.feature, shap: parseFloat(f.shap_value.toFixed(6)) }))
    : [];

  const alloc = result?.allocation || [];
  const metrics = result?.metrics || {};

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
            XGBoost views · SHAP explainability · Sentiment analysis · Natural language queries ·
            Streaming RAG chatbot · PDF reports · Fama-French 5 factors
          </p>
        </div>
        <div className="status">{`Backend: ${API_URL}`}</div>
      </section>

      {/* ── Natural Language Optimizer ── */}
      <section className="card" style={{ marginBottom: 18 }}>
        <div className="cardTitle">
          <MessageSquare size={20} /> Ask the AI to Optimise for You
        </div>
        <p className="muted" style={{ fontSize: 13, marginBottom: 12 }}>
          Describe your portfolio in plain English. The AI will parse your request and run the optimiser automatically.
        </p>
        <textarea
          value={nlPrompt}
          onChange={(e) => setNlPrompt(e.target.value)}
          placeholder={`Try: "Optimise ICICIBANK, HDFCBANK, AXISBANK over 2 years with XGBoost views"\nOr: "What if I only used IT stocks for 6 months?"`}
          style={{ minHeight: 72 }}
        />
        <button onClick={nlOptimize} disabled={nlLoading || !nlPrompt.trim()}>
          {nlLoading ? <RefreshCw className="spin" size={16} /> : <Sparkles size={16} />}
          {nlLoading ? "Parsing & Optimising…" : "Run with AI"}
        </button>
        {nlResult && (
          <div style={{ marginTop: 16 }}>
            {nlResult.narrative && (
              <div className="narrative">{nlResult.narrative}</div>
            )}
            <div className="metrics" style={{ marginTop: 12 }}>
              {Object.entries(nlResult.metrics).map(([k, v]) => (
                <div key={k}>
                  <span>{k.split("_").join(" ")}</span>
                  <strong style={{ color: Number(v) < 0 ? "#f87171" : "#22c55e" }}>
                    {Number(v).toFixed(4)}
                  </strong>
                </div>
              ))}
            </div>
            <button
              onClick={() => downloadReport(nlResult)}
              style={{ marginTop: 12, background: "#263247" }}
            >
              <Download size={16} /> Download PDF Report
            </button>
          </div>
        )}
      </section>

      {/* ── Portfolio Optimizer ── */}
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
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <button onClick={optimize} disabled={loading}>
              {loading ? <RefreshCw className="spin" size={16} /> : <Activity size={16} />}
              {loading ? "Optimising…" : "Run Optimization"}
            </button>
            <button onClick={explainViews} disabled={shapLoading} style={{ background: "#7c3aed" }}>
              {shapLoading ? <RefreshCw className="spin" size={16} /> : <Sparkles size={16} />}
              {shapLoading ? "Computing SHAP…" : "Explain with SHAP"}
            </button>
            <button onClick={runSentiment} disabled={sentLoading} style={{ background: "#0e7490" }}>
              {sentLoading ? <RefreshCw className="spin" size={16} /> : <Newspaper size={16} />}
              {sentLoading ? "Loading…" : "News Sentiment"}
            </button>
          </div>
        </div>

        <div className="card">
          <div className="cardTitle">Metrics</div>
          {Object.keys(metrics).length === 0 ? (
            <p className="muted">Run optimization to see metrics.</p>
          ) : (
            <>
              <div className="metrics">
                {Object.entries(metrics).map(([key, value]) => (
                  <div key={key}>
                    <span>{key.split("_").join(" ")}</span>
                    <strong style={{ color: Number(value) < 0 ? "#f87171" : "#e2e8f0" }}>
                      {Number(value).toFixed(4)}
                    </strong>
                  </div>
                ))}
              </div>
              {result && (
                <button
                  onClick={() => downloadReport(result)}
                  style={{ marginTop: 14, background: "#1e3a5f", fontSize: 13 }}
                >
                  <Download size={14} /> Download PDF Report
                </button>
              )}
            </>
          )}
        </div>
      </section>

      {/* AI Narrative */}
      {result?.narrative && (
        <section className="card" style={{ marginBottom: 18 }}>
          <div className="cardTitle"><Sparkles size={20} /> AI Portfolio Analysis</div>
          <p className="narrative">{result.narrative}</p>
        </section>
      )}

      {/* Allocation + Returns charts */}
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

      {/* ── SHAP Explainability ── */}
      {shapData && (
        <section className="card" style={{ marginBottom: 18 }}>
          <div className="cardTitle"><Sparkles size={20} /> XGBoost Feature Importances (SHAP)</div>
          <p className="muted" style={{ fontSize: 13, marginBottom: 12 }}>
            Each bar shows how much a feature pushed the predicted return view up (positive) or down (negative) for the selected stock.
          </p>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
            {shapData.tickers.map((t) => (
              <button
                key={t}
                onClick={() => setShapTicker(t)}
                style={{
                  padding: "6px 12px",
                  fontSize: 12,
                  marginTop: 0,
                  background: shapTicker === t ? "#7c3aed" : "#263247",
                }}
              >
                {t}
              </button>
            ))}
          </div>
          {shapBarData.length > 0 && (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={shapBarData} layout="vertical">
                <XAxis type="number" tickFormatter={(v: number) => v.toFixed(5)} />
                <YAxis type="category" dataKey="feature" width={120} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => (typeof v === "number" ? v.toFixed(6) : v)} />
                <Bar dataKey="shap" fill="#7c3aed" />
              </BarChart>
            </ResponsiveContainer>
          )}
        </section>
      )}

      {/* ── News Sentiment ── */}
      {sentData && (
        <section className="card" style={{ marginBottom: 18 }}>
          <div className="cardTitle">
            <Newspaper size={20} /> News Sentiment
            <span style={{ marginLeft: "auto", fontSize: 13, color: SIGNAL_COLOR[sentData.portfolio_signal] || "#e2e8f0" }}>
              Portfolio: {sentData.portfolio_signal.toUpperCase()} ({sentData.portfolio_score > 0 ? "+" : ""}{sentData.portfolio_score.toFixed(3)})
            </span>
          </div>
          <div className="sentGrid">
            {Object.entries(sentData.sentiment).map(([ticker, item]) => (
              <div key={ticker} className="sentCard">
                <div className="sentTicker">{ticker}</div>
                <div className="sentSignal" style={{ color: SIGNAL_COLOR[item.signal] || "#e2e8f0" }}>
                  {item.signal.toUpperCase()}
                </div>
                <div className="sentScore">{item.score > 0 ? "+" : ""}{item.score.toFixed(3)}</div>
                <div className="sentCount">{item.headlines_count} headlines</div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── Backtest ── */}
      <section className="card" style={{ marginBottom: 18 }}>
        <div className="cardTitle">
          <TrendingUp size={20} /> Rolling Window Backtest
        </div>
        <p className="muted" style={{ marginBottom: 12, fontSize: 13 }}>
          Walk-forward backtest replicating the notebook methodology. At 5y with All-50 tickers the original results showed +41% annual return vs ~21% NIFTY benchmark (Sharpe 2.36).
        </p>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <div>
            <label>Period</label>
            <select value={btPeriod} onChange={(e) => setBtPeriod(e.target.value)} style={{ minHeight: 40, width: 100 }}>
              <option>1y</option>
              <option>2y</option>
              <option>3y</option>
              <option>5y</option>
            </select>
          </div>
          <div className="checkRow" style={{ marginTop: 18 }}>
            <input id="bt-views" type="checkbox" checked={btUseViews} onChange={(e) => setBtUseViews(e.target.checked)} />
            <label htmlFor="bt-views" className="checkLabel">Use XGBoost Views</label>
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
              <div className="cardTitle" style={{ fontSize: 15 }}>Cumulative Returns vs Benchmark</div>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={btResult.cumulative_returns}>
                  <XAxis dataKey="date" hide />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="value" dot={false} stroke="#22c55e" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="metrics" style={{ marginTop: 16 }}>
              {Object.entries(btResult.metrics).map(([k, v]) => (
                <div key={k}>
                  <span>{k.split("_").join(" ")}</span>
                  <strong style={{ color: Number(v) < 0 ? "#f87171" : "#22c55e" }}>{Number(v).toFixed(4)}</strong>
                </div>
              ))}
            </div>
          </>
        )}
      </section>

      {/* ── Sensitivity Analysis ── */}
      <section className="card" style={{ marginBottom: 18 }}>
        <div className="cardTitle">
          <BarChart2 size={20} /> Sensitivity Analysis
        </div>
        <p className="muted" style={{ marginBottom: 12, fontSize: 13 }}>
          Sweeps the view multiplier (0.5× → 1.5×) to show how allocation responds to view confidence. In the notebook, ICICIBANK and BAJFINANCE consistently gain weight as confidence increases.
        </p>
        <button onClick={runSensitivity} disabled={sensLoading}>
          {sensLoading ? <RefreshCw className="spin" size={16} /> : null}
          Run Sensitivity Sweep
        </button>
        {sensData && sensitivityChartData.length > 0 && (
          <ResponsiveContainer width="100%" height={320} style={{ marginTop: 16 }}>
            <LineChart data={sensitivityChartData}>
              <XAxis dataKey="multiplier" tickFormatter={(v: number) => `${v.toFixed(1)}×`} />
              <YAxis />
              <Tooltip formatter={(v) => (typeof v === "number" ? v.toFixed(4) : v)} />
              <Legend />
              {top10Tickers.map((t, i) => (
                <Line key={t} type="monotone" dataKey={t} dot={false} stroke={LINE_COLORS[i % LINE_COLORS.length]} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </section>

      {/* ── Fama-French Factors ── */}
      <section className="card" style={{ marginBottom: 18 }}>
        <div className="cardTitle">
          <BarChart2 size={20} /> Fama-French 5 Factors
        </div>
        <p className="muted" style={{ marginBottom: 12, fontSize: 13 }}>
          Mkt-RF, SMB, HML, RMW, CMA — computed from NIFTY50 constituents using monthly prices and fundamentals.
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

      {/* ── RAG Chatbot ── */}
      <section className="card chat">
        <div className="cardTitle">
          <Bot size={20} /> RAG Chatbot
          <span className="streamBadge">Streaming</span>
        </div>
        <p className="muted" style={{ fontSize: 13, marginBottom: 12 }}>
          Grounded on the notebook methodology, backtest results, model explainer, and NIFTY50 market context.
          Responses stream token-by-token. Ask about benchmark outperformance, SHAP features, or model details.
        </p>

        <div className="chatThread">
          {chatHistory.length === 0 && (
            <p className="muted" style={{ fontSize: 13 }}>
              Try: "How does this model beat the NIFTY 50?" or "Explain the SHAP results for ICICIBANK" or "What is the Sharpe ratio?"
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
          placeholder="Ask about the model, backtest results, benchmark, or Fama-French factors… (Enter to send)"
        />
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <button onClick={ask} disabled={chatLoading || !question.trim()}>
            {chatLoading ? <RefreshCw className="spin" size={16} /> : <Send size={16} />}
            Ask
          </button>
          {chatHistory.length > 0 && (
            <button onClick={() => { setChatHistory([]); setRagHistory([]); }} style={{ background: "#263247" }}>
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
