import { useMemo, useState } from "react";
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
} from "recharts";
import { Bot, PieChart, RefreshCw, Send } from "lucide-react";
import "./style.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const REQUEST_TIMEOUT_MS = 15000;
const MAX_RETRIES = 1;

type AllocationPoint = { ticker: string; weight: number };
type CurvePoint = { date: string; value: number };
type OptimizeResponse = {
  tickers: string[];
  allocation: AllocationPoint[];
  metrics: Record<string, number>;
  cumulative_returns: CurvePoint[];
  generated_at: string;
};
type RagResponse = { answer: string; sources: Array<Record<string, unknown>> };
type ApiError = { detail?: string };

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
      lastError =
        error instanceof Error ? error : new Error("Unknown request error");
      if (attempt === MAX_RETRIES) {
        break;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 600));
    }
  }
  throw lastError || new Error("Request failed");
}

function App() {
  const [tickers, setTickers] = useState(
    "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS,SBIN.NS,LT.NS,AXISBANK.NS",
  );
  const [period, setPeriod] = useState("1y");
  const [result, setResult] = useState<OptimizeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [question, setQuestion] = useState(
    "Explain how this Black-Litterman RAG app works.",
  );
  const [answer, setAnswer] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  const backendStatusText = useMemo(() => `Backend: ${API_URL}`, []);

  async function optimize() {
    setLoading(true);
    setResult(null);
    try {
      const data = await postJsonWithRetry<OptimizeResponse>(
        "/optimize-portfolio",
        {
          tickers: tickers
            .split(",")
            .map((value) => value.trim())
            .filter(Boolean),
          period,
        },
      );
      setResult(data);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Optimization failed";
      alert(message);
    } finally {
      setLoading(false);
    }
  }

  async function ask() {
    setChatLoading(true);
    setAnswer("");
    try {
      const data = await postJsonWithRetry<RagResponse>("/ask-rag", {
        question,
        k: 5,
      });
      setAnswer(data.answer);
    } catch (error) {
      const message = error instanceof Error ? error.message : "RAG failed";
      alert(message);
    } finally {
      setChatLoading(false);
    }
  }

  const alloc = result?.allocation || [];
  const metrics = result?.metrics || {};

  return (
    <main>
      <section className="hero">
        <div>
          <p className="eyebrow">Live Portfolio Intelligence</p>
          <h1>Black-Litterman Portfolio Optimizer + RAG Chatbot</h1>
          <p className="sub">
            A full-stack web version of the notebook, with a FastAPI backend and
            Vercel-ready React frontend.
          </p>
        </div>
        <div className="status">{backendStatusText}</div>
      </section>

      <section className="grid">
        <div className="card wide">
          <div className="cardTitle">
            <PieChart size={20} /> Portfolio Optimizer
          </div>
          <label>Tickers</label>
          <textarea value={tickers} onChange={(e) => setTickers(e.target.value)} />
          <label>Data period</label>
          <select value={period} onChange={(e) => setPeriod(e.target.value)}>
            <option>6mo</option>
            <option>1y</option>
            <option>2y</option>
            <option>5y</option>
          </select>
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
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={alloc}>
                <XAxis dataKey="ticker" hide />
                <YAxis />
                <Tooltip />
                <Bar dataKey="weight" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="card">
            <div className="cardTitle">Cumulative Returns</div>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={result.cumulative_returns}>
                <XAxis dataKey="date" hide />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="value" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      <section className="card chat">
        <div className="cardTitle">
          <Bot size={20} /> RAG Chatbot
        </div>
        <textarea value={question} onChange={(e) => setQuestion(e.target.value)} />
        <button onClick={ask} disabled={chatLoading}>
          {chatLoading ? <RefreshCw className="spin" size={16} /> : <Send size={16} />}
          Ask
        </button>
        {answer && <pre className="answer">{answer}</pre>}
      </section>
    </main>
  );
}

const root = document.getElementById("root");
if (!root) {
  throw new Error("Root element not found");
}

createRoot(root).render(<App />);
