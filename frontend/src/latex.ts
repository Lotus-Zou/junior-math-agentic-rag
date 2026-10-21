import { unified } from "unified";
import remarkParse from "remark-parse";
import { visit } from "unist-util-visit";

const parser = unified().use(remarkParse);

export function normalizeLatexDelimiters(content: string): string {
  const protectedRanges: [number, number][] = [];
  visit(parser.parse(content), (node) => {
    if (node.type !== "code" && node.type !== "inlineCode") return;
    const start = node.position?.start.offset;
    const end = node.position?.end.offset;
    if (start !== undefined && end !== undefined) protectedRanges.push([start, end]);
  });
  const normalize = (text: string) => text
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, math: string) => `\n$$\n${math}\n$$\n`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_match, math: string) => `$${math}$`);
  let offset = 0;
  let result = "";
  for (const [start, end] of protectedRanges) {
    result += normalize(content.slice(offset, start)) + content.slice(start, end);
    offset = end;
  }
  return result + normalize(content.slice(offset));
}
