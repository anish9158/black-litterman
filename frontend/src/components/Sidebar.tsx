import { NavLink } from "react-router-dom";
import { MessageSquare, BarChart2, Database, TrendingUp, Copy, Check, Activity } from "lucide-react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";

const AI_STACK = ["XGBoost", "FAISS", "LangChain", "Groq", "SHAP"];

const NAV_ITEMS = [
  { to: "/chat", icon: MessageSquare, label: "Chat" },
  { to: "/dashboard", icon: BarChart2, label: "Dashboard" },
  { to: "/knowledge-base", icon: Database, label: "Knowledge Base" },
  { to: "/system-check", icon: Activity, label: "System Check" },
];

export default function Sidebar({ threadId }: { threadId: string }) {
  const [copied, setCopied] = useState(false);

  function copyThread() {
    navigator.clipboard.writeText(threadId).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <aside className="flex h-screen w-56 flex-col border-r border-border bg-card px-3 py-4 shrink-0">
      {/* Logo */}
      <div className="flex items-center gap-2 px-2 mb-5">
        <TrendingUp className="text-primary shrink-0" size={20} />
        <div>
          <p className="text-sm font-bold leading-tight text-foreground">BL Portfolio</p>
          <p className="text-[10px] text-muted-foreground leading-tight">Optimizer</p>
        </div>
      </div>

      <Separator className="mb-4" />

      {/* Navigation */}
      <nav className="flex flex-col gap-1 mb-4">
        <p className="px-2 mb-1 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
          Navigation
        </p>
        {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm transition-colors",
                isActive
                  ? "bg-primary/15 text-primary font-medium"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground"
              )
            }
          >
            <Icon size={16} />
            {label}
          </NavLink>
        ))}
      </nav>

      <Separator className="mb-4" />

      {/* AI Stack */}
      <div className="px-2 mb-4">
        <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
          AI Stack
        </p>
        <div className="flex flex-wrap gap-1.5">
          {AI_STACK.map((s) => (
            <Badge key={s} variant="indigo" className="text-[10px] px-1.5 py-0">
              {s}
            </Badge>
          ))}
        </div>
      </div>

      <Separator className="mb-4" />

      {/* Key result */}
      <div className="px-2 mb-4">
        <p className="mb-1 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
          Key Result
        </p>
        <p className="text-xs text-foreground font-semibold">Sharpe 2.36</p>
        <p className="text-[11px] text-green-400">+20.1% vs NIFTY 50</p>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      <Separator className="mb-3" />

      {/* Thread ID footer */}
      <div className="px-2">
        <p className="mb-1 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
          Thread ID
        </p>
        <button
          onClick={copyThread}
          className="flex w-full items-center gap-1.5 rounded-md px-2 py-1.5 text-[10px] font-mono text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
          title="Click to copy thread ID"
        >
          <span className="truncate">{threadId.slice(0, 20)}…</span>
          {copied ? <Check size={11} className="text-green-400 shrink-0" /> : <Copy size={11} className="shrink-0" />}
        </button>
      </div>
    </aside>
  );
}
