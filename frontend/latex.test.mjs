import test from "node:test";
import assert from "node:assert/strict";
import { normalizeLatexDelimiters } from "./src/latex.ts";

test("normalizes inline and display math without modifying code", () => {
  assert.equal(normalizeLatexDelimiters(String.raw`Solve \(x=4\).`), "Solve $x=4$.");
  assert.equal(normalizeLatexDelimiters(String.raw`\[a^2+b^2=c^2\]`), "\n$$\na^2+b^2=c^2\n$$\n");
  const code = "`\\(literal\\)`\n\n```text\n\\[literal\\]\n```";
  assert.equal(normalizeLatexDelimiters(code), code);
  assert.equal(normalizeLatexDelimiters("Already $x=4$."), "Already $x=4$.");
});
