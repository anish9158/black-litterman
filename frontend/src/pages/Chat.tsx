import { useCallback, useEffect, useRef, useState } from "react";
import { Send, RefreshCw, ChevronDown, ChevronUp, Lock, Zap } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { Skeleton } from "@/components/ui/skeleton";
import ChatMarkdown from "@/components/ChatMarkdown";
import { cn } from "@/lib/utils";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const REQUEST_TIMEOUT_MS = 90000;

// Charts that can be triggered by keywords in user messages
const CHART_TRIGGERS: { keywords: string[]; chart: string; caption: string }[] = [
  {
    keywords: ["backtest", "cumulative", "returns chart", "performance chart", "benchmark chart"],
    chart: "/charts/chart1.png",
    caption: "Cumulative returns: BL portfolio vs NIFTY 50 benchmark",
  },
  {
    keywords: ["sensitivity heatmap", "heatmap", "heat map", "sensitivity analysis", "view multiplier"],
    chart: "/charts/chart2.png",
    caption: "Sensitivity heatmap: allocation vs view multiplier",
  },
  {
    keywords: ["sensitivity chart", "sensitivity line", "ticker weight", "weight chart", "show chart"],
    chart: "/charts/chart3.png",
    caption: "Top-ticker weights vs view multiplier s",
  },
];

const SUGGESTED_QUESTIONS = [
  "How does this model beat the NIFTY 50?",
  "Explain the XGBoost view generation",
  "What is the Sharpe ratio from the backtest?",
  "Why does ICICIBANK get the highest allocation?",
  "Show the backtest chart",
  "What is the Black-Litterman model?",
];

type Source = {
  source: string;
  category: string;
  preview: string;
  char_count: number;
  score: number;
};

type TokenUsage = {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
};

type Message = {
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
  token_usage?: TokenUsage;
  latency_ms?: number;
  follow_up_questions?: string[];
  inline_chart?: { src: string; caption: string };
};

function detectChart(text: string): { src: string; caption: string } | undefined {
  const lower = text.toLowerCase();
  for (const t of CHART_TRIGGERS) {
    if (t.keywords.some((kw) => lower.includes(kw))) return { src: t.chart, caption: t.caption };
  }
}

async function fetchWithTimeout(input: RequestInfo | URL, init: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    window.clearTimeout(timeout);
  }
}

/** FAISS L2 embedding distance among this batch — lower = closer semantic match */
function relevanceBarPct(batchDistances: number[], d: number): number {
  if (batchDistances.length === 0) return 50;
  const mn = Math.min(...batchDistances);
  const mx = Math.max(...batchDistances);
  if (mx <= mn + 1e-8) return 100;
  return Math.round(Math.max(4, Math.min(100, (100 * (mx - d)) / (mx - mn))));
}

function SourcesPanel({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);
  if (!sources.length) return null;
  const distances = sources.map((s) => s.score);
  return (
    <div className="mt-2 rounded-md border border-border bg-muted/20 text-xs">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-1.5 text-muted-foreground hover:text-foreground transition-colors text-left"
      >
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        <span>Sources used ({sources.length})</span>
      </button>
      {open && (
        <div className="flex flex-col gap-2 px-3 pb-3">
          <p className="text-[10px] leading-snug text-muted-foreground">
            Retrieval distance is the FAISS L2 embedding distance (same value LangChain exposes as score). Lower
            numbers mean the chunk embedding is nearer to your question — a stronger semantic match among the retrieved
            set.
          </p>
          {sources.map((s, i) => (
            <div key={i} className="flex flex-col gap-0.5">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-indigo-400">{s.source}</span>
                <Badge variant="outline" className="text-[9px] px-1 py-0">{s.category}</Badge>
                <span className="ml-auto whitespace-nowrap text-muted-foreground font-mono" title="Embedding L2 distance (lower is better match)">
                  L²&nbsp;distance: <span className="text-foreground">{s.score.toFixed(4)}</span>
                </span>
              </div>
              {/* Relative relevance within this retrieval batch */}
              <div className="h-1 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-indigo-500"
                  title="Relative similarity within this retrieval batch (full bar = closest match)"
                  style={{ width: `${relevanceBarPct(distances, s.score)}%` }}
                />
              </div>
              <p className="line-clamp-2 text-muted-foreground">{s.preview}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function TokenBadge({ usage, latencyMs }: { usage: TokenUsage; latencyMs: number }) {
  if (!usage.total_tokens && !latencyMs) return null;
  return (
    <div className="mt-1.5 flex items-center gap-2 text-[10px] text-muted-foreground">
      {usage.total_tokens > 0 && (
        <>
          <Zap size={9} className="text-yellow-500" />
          <span>↑ {usage.prompt_tokens} · ↓ {usage.completion_tokens} tokens</span>
        </>
      )}
      {latencyMs > 0 && <span className="ml-1">{latencyMs.toLocaleString()} ms</span>}
    </div>
  );
}

export default function Chat({ threadId: _threadId }: { threadId: string }) {
  const [messages, setMessages] = useState<Message[]>(() => {
    try {
      const saved = localStorage.getItem("bl_chat_messages");
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [ragHistory, setRagHistory] = useState<Array<{ user: string; assistant: string }>>([]);
  const [input, setInput] = useState("");
  const [loading, setChatLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem("bl_chat_messages", JSON.stringify(messages.slice(-40)));
    }
  }, [messages]);

  const ask = useCallback(
    async (questionOverride?: string) => {
      const q = (questionOverride ?? input).trim();
      if (!q || loading) return;
      setInput("");
      setChatLoading(true);

      const inlineChart = detectChart(q);
      const userMsg: Message = { role: "user", text: q };
      const assistantPlaceholder: Message = { role: "assistant", text: "", inline_chart: inlineChart };
      setMessages((m) => [...m, userMsg, assistantPlaceholder]);

      try {
        const res = await fetchWithTimeout(`${API_URL}/ask-rag-stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q, k: 5, history: ragHistory }),
        });

        if (!res.ok || !res.body) throw new Error("Stream failed");

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let fullText = "";
        let finalMeta: Partial<Message> = {};

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const raw = line.slice(6).trim();
            if (raw === "[DONE]") break;
            try {
              const evt = JSON.parse(raw) as {
                token?: string;
                done?: boolean;
                sources?: Source[];
                token_usage?: TokenUsage;
                latency_ms?: number;
                follow_up_questions?: string[];
                history?: Array<{ user: string; assistant: string }>;
              };
              if (evt.token) {
                fullText += evt.token;
                setMessages((m) => {
                  const updated = [...m];
                  updated[updated.length - 1] = { ...updated[updated.length - 1], text: fullText };
                  return updated;
                });
              }
              if (evt.done) {
                finalMeta = {
                  sources: evt.sources ?? [],
                  token_usage: evt.token_usage,
                  latency_ms: evt.latency_ms,
                  follow_up_questions: evt.follow_up_questions ?? [],
                };
                if (evt.history) setRagHistory(evt.history);
              }
            } catch {
              // malformed SSE frame
            }
          }
        }

        setMessages((m) => {
          const updated = [...m];
          updated[updated.length - 1] = { ...updated[updated.length - 1], ...finalMeta };
          return updated;
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Request failed";
        setMessages((m) => {
          const updated = [...m];
          updated[updated.length - 1] = { ...updated[updated.length - 1], text: `Error: ${msg}` };
          return updated;
        });
      } finally {
        setChatLoading(false);
      }
    },
    [input, loading, ragHistory]
  );

  const handleKey = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        ask();
      }
    },
    [ask]
  );

  return (
    <div className="flex h-screen flex-col">
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-border px-6 py-3 bg-card shrink-0">
        <h1 className="text-sm font-semibold text-foreground">BL Portfolio Assistant</h1>
        <Badge variant="streaming">Streaming</Badge>
        <Badge variant="outline" className="gap-1 text-[10px]">
          <Lock size={9} /> Topic-guarded
        </Badge>
        <div className="ml-auto text-[11px] text-muted-foreground font-mono">{API_URL}</div>
      </div>

      {/* Chat thread */}
      <ScrollArea className="flex-1 px-6 py-4">
        {messages.length === 0 && (
          <div className="flex h-full flex-col items-center justify-center gap-6 py-16">
            <div className="text-center">
              <p className="text-lg font-semibold text-foreground">Ask anything about the project</p>
              <p className="text-sm text-muted-foreground mt-1">
                Grounded on your notebook, backtest results, and model documentation
              </p>
            </div>
            <div className="flex flex-wrap justify-center gap-2 max-w-xl">
              {SUGGESTED_QUESTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => ask(q)}
                  className="rounded-full border border-border bg-accent/40 px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground hover:border-primary/50 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="flex flex-col gap-5 pb-4">
          {messages.map((msg, i) => (
            <div key={i} className={cn("flex flex-col gap-1", msg.role === "user" ? "items-end" : "items-start")}>
              <span className={cn("text-[10px] font-semibold uppercase tracking-wider", msg.role === "user" ? "text-indigo-400" : "text-green-400")}>
                {msg.role === "user" ? "You" : "Assistant"}
              </span>

              {/* Inline chart (triggered by user question) */}
              {msg.role === "assistant" && msg.inline_chart && (
                <Card className="w-full max-w-2xl overflow-hidden mb-1">
                  <img src={msg.inline_chart.src} alt={msg.inline_chart.caption} className="w-full object-contain" />
                  <CardContent className="py-2 px-3">
                    <p className="text-xs text-muted-foreground">{msg.inline_chart.caption}</p>
                  </CardContent>
                </Card>
              )}

              <div
                className={cn(
                  "rounded-xl px-4 py-2.5 text-sm leading-relaxed max-w-2xl",
                  msg.role === "user"
                    ? "bg-primary/20 text-foreground"
                    : "bg-card border border-border text-foreground"
                )}
              >
                {msg.text ? (
                  msg.role === "assistant" ? (
                    <ChatMarkdown text={msg.text} />
                  ) : (
                    <pre className="font-sans whitespace-pre-wrap">{msg.text}</pre>
                  )
                ) : loading && i === messages.length - 1 ? (
                  <div className="flex gap-1 items-center py-1">
                    <Skeleton className="h-3 w-24" />
                    <Skeleton className="h-3 w-16" />
                    <Skeleton className="h-3 w-20" />
                  </div>
                ) : null}
              </div>

              {/* Sources + token usage for assistant messages */}
              {msg.role === "assistant" && msg.text && (
                <div className="w-full max-w-2xl">
                  {msg.sources && <SourcesPanel sources={msg.sources} />}
                  {msg.token_usage && msg.latency_ms !== undefined && (
                    <TokenBadge usage={msg.token_usage} latencyMs={msg.latency_ms} />
                  )}
                  {msg.follow_up_questions && msg.follow_up_questions.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {msg.follow_up_questions.map((q, j) => (
                        <button
                          key={j}
                          onClick={() => ask(q)}
                          className="rounded-full border border-primary/30 bg-primary/10 px-2.5 py-1 text-[11px] text-primary hover:bg-primary/20 transition-colors"
                        >
                          {q}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      {/* Input area */}
      <div className="border-t border-border bg-card px-6 py-4 shrink-0">
        {/* Suggested questions (shown when chat has messages) */}
        {messages.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-1.5">
            {SUGGESTED_QUESTIONS.slice(0, 4).map((q) => (
              <button
                key={q}
                onClick={() => ask(q)}
                className="rounded-full border border-border bg-accent/30 px-2.5 py-1 text-[11px] text-muted-foreground hover:text-foreground hover:border-primary/50 transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        )}
        <div className="flex gap-2 items-end">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask about the model, backtest, allocation… (Enter to send, Shift+Enter for newline)"
            className="min-h-[44px] max-h-32 text-sm bg-background"
            disabled={loading}
          />
          <div className="flex flex-col gap-2 shrink-0">
            <Button onClick={() => ask()} disabled={loading || !input.trim()} size="icon">
              {loading ? <RefreshCw size={15} className="animate-spin" /> : <Send size={15} />}
            </Button>
            {messages.length > 0 && (
              <Button
                variant="ghost"
                size="icon"
                onClick={() => { setMessages([]); setRagHistory([]); localStorage.removeItem("bl_chat_messages"); }}
                title="Clear chat"
              >
                <RefreshCw size={13} />
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
