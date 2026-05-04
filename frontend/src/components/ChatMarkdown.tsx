import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { cn } from "@/lib/utils";
import "katex/dist/katex.min.css";

/** LLMs often emit LaTeX with \( … \) / \[ … \]. Normalize to formats remark-math accepts. */
export function preprocessLatexDelimiters(text: string): string {
  let s = text;
  s = s.replace(/\\\[([\s\S]*?)\\\]/g, (_, inner: string) => `$$\n${inner.trim()}\n$$\n`);
  s = s.replace(/\\\(([\s\S]*?)\\\)/g, (_, inner: string) => `$${inner.trim()}$`);
  return s;
}

export default function ChatMarkdown({ text, className }: { text: string; className?: string }) {
  const md = preprocessLatexDelimiters(text);

  const components: Components = {
    p(props) {
      const { node: _node, children } = props;
      return <p className="mb-3 last:mb-0 whitespace-pre-wrap text-[inherit]">{children}</p>;
    },
    strong(props) {
      const { node: _node, children } = props;
      return <strong className="font-semibold text-indigo-200">{children}</strong>;
    },
    em(props) {
      const { node: _node, children } = props;
      return <em className="italic text-muted-foreground">{children}</em>;
    },
    ul(props) {
      const { node: _node, children } = props;
      return <ul className="mb-3 list-disc space-y-1 pl-6 last:mb-0">{children}</ul>;
    },
    ol(props) {
      const { node: _node, children } = props;
      return <ol className="mb-3 list-decimal space-y-1 pl-6 last:mb-0">{children}</ol>;
    },
    li(props) {
      const { node: _node, children } = props;
      return <li className="leading-relaxed">{children}</li>;
    },
    h1(props) {
      const { node: _node, children } = props;
      return <h1 className="mb-2 text-base font-semibold">{children}</h1>;
    },
    h2(props) {
      const { node: _node, children } = props;
      return <h2 className="mb-2 mt-4 text-sm font-semibold">{children}</h2>;
    },
    h3(props) {
      const { node: _node, children } = props;
      return <h3 className="mb-1 mt-3 text-sm font-medium">{children}</h3>;
    },
    pre(props) {
      const { node: _node, children } = props;
      return (
        <pre className="my-3 overflow-x-auto rounded-md border border-border bg-muted/40 p-3 font-mono text-[12px]">
          {children}
        </pre>
      );
    },
    code(props) {
      const { node: _node, className, children } = props;
      const isFence = /\blanguage-/.test(String(className || ""));
      const content = flattenText(children as ReactNode);
      if (!isFence && content.length < 280 && !content.includes("\n")) {
        return <code className="rounded bg-muted/80 px-1 py-0.5 font-mono text-[12px]">{children}</code>;
      }
      return <code className={cn("font-mono text-[12px]", className)}>{children}</code>;
    },
    blockquote(props) {
      const { node: _node, children } = props;
      return (
        <blockquote className="my-3 border-l-2 border-primary/60 pl-3 italic text-muted-foreground">
          {children}
        </blockquote>
      );
    },
    a(props) {
      const { node: _node, href, children } = props;
      return (
        <a
          href={href ?? "#"}
          className="text-primary underline underline-offset-2"
          target="_blank"
          rel="noreferrer noopener"
        >
          {children}
        </a>
      );
    },
    hr() {
      return <hr className="my-4 border-border" />;
    },
  };

  return (
    <div className={cn("markdown-body text-sm leading-relaxed break-words [&_.katex]:text-[0.92em]", className)}>
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]} components={components}>
        {md}
      </ReactMarkdown>
    </div>
  );
}

function flattenText(children: ReactNode): string {
  if (typeof children === "string" || typeof children === "number") return String(children);
  if (Array.isArray(children)) return children.map(flattenText).join("");
  return "";
}
