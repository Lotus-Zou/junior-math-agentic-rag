import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { normalizeLatexDelimiters } from "../latex";

export function MarkdownAnswer({ content }: { content: string }) {
  return (
    <div className="markdown-answer assistant-answer">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          a: ({ children, ...props }) => (
            <a {...props} target="_blank" rel="noreferrer">
              {children}
            </a>
          ),
          code: ({ children, className, ...props }) => (
            <code {...props} className={className || "inline-code"}>
              {children}
            </code>
          ),
        }}
      >
        {normalizeLatexDelimiters(content)}
      </ReactMarkdown>
    </div>
  );
}
