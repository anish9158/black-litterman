import { useEffect, useState } from "react";
import {
  TrendingUp,
  Brain,
  Database,
  LineChart as LineChartIcon,
  RefreshCw,
  ListOrdered,
  Activity,
} from "lucide-react";
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const LINE_COLORS = ["#6366f1", "#f97316", "#22c55e", "#a78bfa", "#f43f5e"];

const AI_NOVELTY = [
  {
    icon: Brain,
    title: "XGBoost Views",
    color: "text-orange-400",
    description:
      "Trains a per-stock XGBoost regressor on RSI, SMA20/50, momentum, P/E, P/B, and polynomial features. Predicted returns become the 'views' (Q) in the Black-Litterman update, replacing subjective analyst forecasts with data-driven ML signals.",
  },
  {
    icon: TrendingUp,
    title: "Black-Litterman Model",
    color: "text-indigo-400",
    description:
      "Bayesian posterior blend of CAPM market equilibrium (Π) and XGBoost views (Q) via the full two-inverse formula. Covariance shrinkage (Ledoit-Wolf) stabilises the matrix inversion. CVXPY solves the constrained optimisation.",
  },
  {
    icon: Database,
    title: "Source-Grounded Portfolio Assistant",
    color: "text-green-400",
    description:
      "FAISS vector store with HuggingFace sentence-transformers embeddings. Groq-backed chat model generates answers only from retrieved chunks via a strict system prompt; `/ask-rag-stream` uses SSE; the UI shows sources, L² distances, and per-turn token and latency metadata when returned by the API.",
  },
  {
    icon: LineChartIcon,
    title: "SHAP Explainability",
    color: "text-purple-400",
    description:
      "TreeExplainer computes per-feature SHAP values for each XGBoost model, explaining which technical/fundamental signals drove each stock's predicted return. Exposed where optimisation and reporting flows surface SHAP payloads.",
  },
];

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
  models: NotebookModel[];
  weights: NotebookWeight[];
  sensitivity: { multipliers: number[]; series: NotebookSensitivitySeries[] };
  notes: string;
};

type PipelineTrace = {
  name: string;
  stages: { stage: string; description: string }[];
};

const HERO_STATS = [
  { label: "Sharpe Ratio", value: "2.36", sub: "All-50 model", color: "text-indigo-400" },
  { label: "Annual Return", value: "41.0%", sub: "vs 20.9% benchmark", color: "text-green-400" },
  { label: "Excess Return", value: "+20.1%", sub: "over NIFTY 50", color: "text-green-400" },
  { label: "Max Drawdown", value: "−10.3%", sub: "controlled risk", color: "text-orange-400" },
];

const CHART_CAPTIONS = [
  "Cumulative returns: BL portfolio (All-50) vs NIFTY 50 benchmark — July 2020–July 2025 (~5-year window)",
  "Sensitivity heatmap: portfolio allocation vs view multiplier s (0.5 → 1.5)",
  "Top-ticker weights vs view multiplier s — ICICIBANK and BAJFINANCE gain weight as conviction increases",
];

function fmt(v: number | null, type: "pct" | "num"): string {
  if (v == null) return "—";
  return type === "pct" ? `${(v * 100).toFixed(1)}%` : v.toFixed(2);
}

export default function Dashboard() {
  const [nb, setNb] = useState<NotebookResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [pipeline, setPipeline] = useState<PipelineTrace | null>(null);
  const [pipelineLoaded, setPipelineLoaded] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetch(`${API_URL}/notebook-results`)
      .then((r) => r.json())
      .then((d) => setNb(d as NotebookResults))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetch(`${API_URL}/ai-pipeline-trace`)
      .then((r) => {
        if (!r.ok) throw new Error("trace");
        return r.json();
      })
      .then((d) => setPipeline(d as PipelineTrace))
      .catch(() => setPipeline(null))
      .finally(() => setPipelineLoaded(true));
  }, []);

  return (
    <ScrollArea className="h-screen">
      <div className="px-6 py-6 max-w-6xl mx-auto space-y-8">
        {/* Page header */}
        <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:gap-3">
          <div>
            <h1 className="text-xl font-bold text-foreground">Backtest Results &amp; AI Architecture</h1>
            <p className="text-xs text-muted-foreground mt-0.5">
              AI-Powered Black-Litterman Portfolio Intelligence Platform — stack overview and cached notebook metrics.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2 sm:ml-auto">
            <Badge variant="indigo">Jul 2020 – Jul 2025</Badge>
            <Badge variant="outline" className="text-[10px]">
              NIFTY 50 benchmark
            </Badge>
          </div>
        </div>

        {/* Hero stats */}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {HERO_STATS.map((s) => (
            <Card key={s.label}>
              <CardContent className="pt-5">
                <p className="text-xs text-muted-foreground">{s.label}</p>
                <p className={`text-2xl font-bold mt-1 ${s.color}`}>{s.value}</p>
                <p className="text-[11px] text-muted-foreground mt-0.5">{s.sub}</p>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* AI architecture cards */}
        <div>
          <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">
            AI Engineering Architecture
          </h2>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {AI_NOVELTY.map(({ icon: Icon, title, color, description }) => (
              <Card key={title} className="hover:border-primary/40 transition-colors">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 text-sm">
                    <Icon size={16} className={color} />
                    {title}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground leading-relaxed">{description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Pipeline trace */}
        <div>
          <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-2">
            <ListOrdered size={14} />
            AI Pipeline Trace
          </h2>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-mono text-primary">
                {pipeline?.name ?? "AI-Powered Black-Litterman Portfolio Intelligence Pipeline"}
              </CardTitle>
              <p className="text-[11px] text-muted-foreground">
                Stages from <span className="font-mono">GET {API_URL}/ai-pipeline-trace</span> — reference only; execution
                follows individual API routes.
              </p>
            </CardHeader>
            <CardContent className="space-y-0">
              {!pipelineLoaded && <p className="text-xs text-muted-foreground">Loading pipeline…</p>}
              {pipelineLoaded && !pipeline && (
                <p className="text-xs text-muted-foreground">Pipeline trace unavailable (API not reachable).</p>
              )}
              {pipeline?.stages.map((s, i) => (
                <div key={s.stage}>
                  {i > 0 && (
                    <div className="flex justify-center py-1.5 text-muted-foreground text-xs select-none" aria-hidden>
                      ↓
                    </div>
                  )}
                  <div className="rounded-md border border-border bg-muted/20 px-3 py-2">
                    <p className="text-[10px] font-mono font-semibold text-indigo-400 tracking-wide">{s.stage}</p>
                    <p className="text-xs text-muted-foreground leading-snug mt-1">{s.description}</p>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* AI observability (documented behaviour, not live metrics) */}
        <div>
          <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-2">
            <Activity size={14} />
            AI Observability
          </h2>
          <Card>
            <CardContent className="pt-4 text-xs text-muted-foreground leading-relaxed space-y-2">
              <p className="text-foreground font-medium text-sm">What the codebase exposes today</p>
              <ul className="list-disc pl-4 space-y-1.5">
                <li>
                  <span className="text-foreground/90">Token usage tracking:</span> non-streaming RAG returns provider
                  token counts in JSON when Groq metadata is present; streaming chat returns estimated token fields in the
                  terminal SSE payload for UI display.
                </li>
                <li>
                  <span className="text-foreground/90">Latency tracking:</span> wall-clock milliseconds recorded for RAG
                  handlers and returned to the client (non-stream and stream).
                </li>
                <li>
                  <span className="text-foreground/90">Source attribution:</span> each chat turn includes retrieved chunk
                  metadata and previews alongside the assistant reply.
                </li>
                <li>
                  <span className="text-foreground/90">FAISS retrieval distance:</span> LangChain exposes L² embedding
                  distance per retrieved chunk; the chat UI labels and visualises relative match strength within the batch.
                </li>
                <li>
                  <span className="text-foreground/90">Prompt hash audit logging:</span> each audited LLM call writes a
                  SHA-256 hash of prompt material (not raw secrets) to <span className="font-mono">backend/logs/llm_calls.jsonl</span>.
                </li>
                <li>
                  <span className="text-foreground/90">Stage-level LLM logging:</span> stages such as{" "}
                  <span className="font-mono">rag_chat</span>, <span className="font-mono">rag_chat_stream</span>,{" "}
                  <span className="font-mono">portfolio_narrative</span>, <span className="font-mono">rag_starter_suggestions</span>
                  , and <span className="font-mono">rag_follow_up_suggestions</span> append NDJSON rows with optional route,
                  tokens, latency, run_id, and status fields.
                </li>
              </ul>
            </CardContent>
          </Card>
        </div>

        {/* Notebook charts */}
        <div>
          <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">
            Notebook Output Charts
          </h2>
          <div className="flex flex-col gap-4">
            {[1, 2, 3].map((n) => (
              <Card key={n} className="overflow-hidden">
                <img
                  src={`/charts/chart${n}.png`}
                  alt={CHART_CAPTIONS[n - 1]}
                  className="w-full object-contain bg-white"
                />
                <CardContent className="py-2.5">
                  <p className="text-xs text-muted-foreground">{CHART_CAPTIONS[n - 1]}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Metrics table */}
        <div>
          <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">
            Performance Comparison
          </h2>
          {loading && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground py-8">
              <RefreshCw size={14} className="animate-spin" /> Loading…
            </div>
          )}
          {nb && (
            <Card>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Metric</TableHead>
                    {nb.models.map((m) => (
                      <TableHead key={m.name} className={m.name.startsWith("All 50") ? "text-indigo-400 font-bold" : ""}>
                        {m.name}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {[
                    { label: "Annual Return", key: "annual_return", type: "pct" as const },
                    { label: "Volatility", key: "annual_volatility", type: "pct" as const },
                    { label: "Sharpe Ratio", key: "sharpe_ratio", type: "num" as const },
                    { label: "Max Drawdown", key: "max_drawdown", type: "pct" as const },
                    { label: "Excess vs Benchmark", key: "excess_return_vs_benchmark", type: "pct" as const },
                  ].map((row) => (
                    <TableRow key={row.label}>
                      <TableCell className="text-muted-foreground">{row.label}</TableCell>
                      {nb.models.map((m) => {
                        const v = (m as unknown as Record<string, number | null>)[row.key] as number | null;
                        const isAlloc = m.name.startsWith("All 50");
                        const isNeg = v != null && v < 0;
                        return (
                          <TableCell
                            key={m.name}
                            className={cn(
                              isAlloc ? "text-indigo-300 font-semibold" : "",
                              isNeg ? "text-red-400" : v != null && v > 0 && row.key !== "annual_volatility" ? "text-green-400" : "",
                            )}
                          >
                            {fmt(v, row.type)}
                          </TableCell>
                        );
                      })}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Card>
          )}
        </div>

        {/* Interactive recharts */}
        {nb && (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Final Allocation (s = 1.5)</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart
                    data={nb.weights.map((w) => ({
                      ticker: w.ticker.replace(".NS", ""),
                      weight: parseFloat((w.weight * 100).toFixed(1)),
                    }))}
                    layout="vertical"
                    margin={{ left: 70, right: 16, top: 4, bottom: 4 }}
                  >
                    <XAxis type="number" unit="%" tick={{ fontSize: 10, fill: "#64748b" }} />
                    <YAxis type="category" dataKey="ticker" tick={{ fontSize: 10, fill: "#94a3b8" }} width={70} />
                    <Tooltip formatter={(v) => `${v}%`} contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 12 }} />
                    <Bar dataKey="weight" fill="#6366f1" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Sensitivity — Weight vs Multiplier s</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart
                    data={nb.sensitivity.multipliers.map((s, i) => {
                      const pt: Record<string, number> = { s };
                      nb.sensitivity.series.forEach((ser) => {
                        pt[ser.ticker.replace(".NS", "")] = parseFloat((ser.weights[i] * 100).toFixed(1));
                      });
                      return pt;
                    })}
                    margin={{ left: 0, right: 16, top: 4, bottom: 20 }}
                  >
                    <XAxis dataKey="s" tick={{ fontSize: 10, fill: "#64748b" }} label={{ value: "s", position: "insideBottom", offset: -8, fill: "#64748b", fontSize: 11 }} />
                    <YAxis unit="%" tick={{ fontSize: 10, fill: "#64748b" }} />
                    <Tooltip formatter={(v) => `${v}%`} contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 12 }} />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                    {nb.sensitivity.series.map((ser, i) => (
                      <Line key={ser.ticker} type="monotone" dataKey={ser.ticker.replace(".NS", "")} stroke={LINE_COLORS[i % LINE_COLORS.length]} dot={false} strokeWidth={2} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        )}

        <p className="text-xs text-muted-foreground pb-6">{nb?.notes}</p>
      </div>
    </ScrollArea>
  );
}
