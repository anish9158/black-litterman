import { useEffect, useState } from "react";
import { Search, RefreshCw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ScrollArea } from "@/components/ui/scroll-area";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

type Chunk = {
  id: number;
  source: string;
  category: string;
  preview: string;
  char_count: number;
};

const CATEGORY_VARIANT: Record<string, "default" | "indigo" | "positive" | "outline"> = {
  notebook: "indigo",
  sector_research: "positive",
  market_report: "default",
  company_filing: "outline",
};

export default function KnowledgeBase() {
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState("");
  const [catFilter, setCatFilter] = useState("all");

  useEffect(() => {
    setLoading(true);
    fetch(`${API_URL}/knowledge-base`)
      .then((r) => r.json())
      .then((d: { chunks: Chunk[] }) => setChunks(d.chunks ?? []))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const categories = ["all", ...Array.from(new Set(chunks.map((c) => c.category)))];

  const filtered = chunks.filter((c) => {
    const matchCat = catFilter === "all" || c.category === catFilter;
    const matchSearch =
      !search ||
      c.source.toLowerCase().includes(search.toLowerCase()) ||
      c.preview.toLowerCase().includes(search.toLowerCase());
    return matchCat && matchSearch;
  });

  return (
    <ScrollArea className="h-screen">
      <div className="px-6 py-6 max-w-5xl mx-auto space-y-5">
        {/* Header */}
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-bold text-foreground">Knowledge Base</h1>
          <Badge variant="indigo">{chunks.length} chunks indexed</Badge>
        </div>
        <p className="text-sm text-muted-foreground">
          All document chunks currently loaded into the FAISS vector store. These are the sources the RAG chatbot retrieves from.
        </p>

        {/* Filters */}
        <div className="flex flex-wrap gap-2 items-center">
          <div className="relative flex-1 min-w-[180px] max-w-sm">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search source or content…"
              className="w-full rounded-md border border-input bg-transparent pl-8 pr-3 py-1.5 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            />
          </div>
          <div className="flex gap-1.5 flex-wrap">
            {categories.map((c) => (
              <button
                key={c}
                onClick={() => setCatFilter(c)}
                className={`rounded-full border px-2.5 py-0.5 text-xs transition-colors ${
                  catFilter === c
                    ? "border-primary bg-primary/20 text-primary"
                    : "border-border text-muted-foreground hover:text-foreground"
                }`}
              >
                {c}
              </button>
            ))}
          </div>
        </div>

        {loading && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground py-8">
            <RefreshCw size={14} className="animate-spin" /> Building index and loading chunks…
          </div>
        )}

        {!loading && (
          <Card>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-10">#</TableHead>
                  <TableHead>Source</TableHead>
                  <TableHead>Category</TableHead>
                  <TableHead>Preview</TableHead>
                  <TableHead className="text-right">Chars</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filtered.map((chunk) => (
                  <TableRow key={chunk.id}>
                    <TableCell className="text-muted-foreground font-mono text-xs">{chunk.id}</TableCell>
                    <TableCell className="font-mono text-xs text-indigo-400 whitespace-nowrap">{chunk.source}</TableCell>
                    <TableCell>
                      <Badge variant={CATEGORY_VARIANT[chunk.category] ?? "outline"} className="text-[9px] px-1.5">
                        {chunk.category}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground max-w-sm">
                      <span className="line-clamp-2">{chunk.preview}</span>
                    </TableCell>
                    <TableCell className="text-right text-xs text-muted-foreground tabular-nums">
                      {chunk.char_count.toLocaleString()}
                    </TableCell>
                  </TableRow>
                ))}
                {filtered.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={5} className="text-center text-muted-foreground py-8">
                      No chunks match your filter.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </Card>
        )}
      </div>
    </ScrollArea>
  );
}
