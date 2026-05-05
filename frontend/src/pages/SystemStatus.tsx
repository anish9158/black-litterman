import { useEffect, useState, useCallback } from "react";
import { Activity, RefreshCw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

type CheckBlock = Record<string, unknown>;

type SystemCheckResponse = {
  status: "ok" | "degraded" | "error";
  checks: Record<string, CheckBlock>;
  warnings: string[];
};

function PassFail({ ok }: { ok?: boolean }) {
  if (ok === undefined) return <span className="text-muted-foreground">—</span>;
  return <Badge variant={ok ? "positive" : "negative"}>{ok ? "Pass" : "Fail"}</Badge>;
}

export default function SystemStatus() {
  const [data, setData] = useState<SystemCheckResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setFetchError(null);
    fetch(`${API_URL}/system-check`)
      .then(async (r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<SystemCheckResponse>;
      })
      .then(setData)
      .catch(() => setFetchError("Could not reach the API. Is the backend running on port 8000?"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const statusLabel = data?.status ?? "—";
  const statusBadgeClass =
    data?.status === "ok"
      ? "text-green-400 border-green-700"
      : data?.status === "degraded"
        ? "text-amber-400 border-amber-700"
        : data?.status === "error"
          ? "text-red-400 border-red-800"
          : "";

  const rows = data
    ? [
        {
          key: "notebook_results",
          title: "Dashboard data",
          desc: "Cached backtest summaries served to /notebook-results.",
          ok: Boolean(data.checks.notebook_results?.ok),
          extra: data.checks.notebook_results,
        },
        {
          key: "rag_vectorstore",
          title: "RAG index",
          desc: "FAISS load + one similarity search (no LLM call).",
          ok: Boolean(data.checks.rag_vectorstore?.ok),
          extra: data.checks.rag_vectorstore,
        },
        {
          key: "groq_llm",
          title: "LLM API key",
          desc: "GROQ_API_KEY or GROQ_TOKEN needed for chat generation.",
          ok: Boolean(data.checks.groq_llm?.ok),
          extra: data.checks.groq_llm,
        },
      ]
    : [];

  return (
    <ScrollArea className="h-screen">
      <div className="px-6 py-6 max-w-3xl mx-auto space-y-6">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-xl font-bold text-foreground flex items-center gap-2">
            <Activity size={22} className="text-primary" />
            System check
          </h1>
          {data && (
            <Badge variant="outline" className={cn("uppercase text-[10px]", statusBadgeClass)}>
              {statusLabel}
            </Badge>
          )}
          <Button variant="outline" size="sm" onClick={load} disabled={loading} className="ml-auto shrink-0">
            <RefreshCw size={14} className={cn("mr-1.5", loading && "animate-spin")} />
            Refresh
          </Button>
        </div>

        <p className="text-sm text-muted-foreground leading-relaxed">
          Validates core pieces of the stack without Yahoo Finance calls or streaming chat. The first run after a cold server start can take 15–30s while the embedding model loads. For full end-to-end coverage (optimise + RAG), run{" "}
          <code className="text-xs bg-muted px-1 rounded">scripts/smoke-check.ps1</code> with your backend base URL.
        </p>

        {fetchError && (
          <Card className="border-red-900/60">
            <CardContent className="py-4 text-sm text-red-400">{fetchError}</CardContent>
          </Card>
        )}

        {data?.warnings && data.warnings.length > 0 && (
          <Card className="border-amber-900/40">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-amber-200">Warnings</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="text-sm text-muted-foreground list-disc pl-5 space-y-1">
                {data.warnings.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}

        <div className="space-y-3">
          {loading && !data ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground py-8">
              <RefreshCw size={14} className="animate-spin" /> Running checks…
            </div>
          ) : (
            rows.map(({ key, title, desc, ok, extra }) => (
              <Card key={key}>
                <CardHeader className="flex flex-row items-start justify-between gap-4 pb-2">
                  <div>
                    <CardTitle className="text-base">{title}</CardTitle>
                    <p className="text-xs text-muted-foreground mt-1">{desc}</p>
                  </div>
                  <PassFail ok={ok} />
                </CardHeader>
                <CardContent>
                  <pre className="text-[11px] text-muted-foreground bg-muted/50 rounded-md p-3 overflow-x-auto font-mono">
                    {JSON.stringify(extra, null, 2)}
                  </pre>
                </CardContent>
              </Card>
            ))
          )}
        </div>

        {!loading && data?.status === "ok" && (
          <p className="text-xs text-green-400 pb-6">Core checks passed. Optional: exercise chat and optimise flows manually.</p>
        )}
      </div>
    </ScrollArea>
  );
}
