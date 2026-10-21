import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig(({ mode }) => ({
  base: mode === "showcase" ? "/junior-math-agentic-rag/" : "/static/",
  define: { __SHOWCASE_MODE__: JSON.stringify(mode === "showcase") },
  plugins: [react(), tailwindcss()],
  build: {
    outDir: mode === "showcase" ? "dist" : "../static",
    emptyOutDir: true,
    sourcemap: false,
    target: "es2023",
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom", "zustand", "@tanstack/react-query"],
          markdown: [
            "react-markdown",
            "remark-gfm",
            "remark-math",
            "rehype-katex",
            "katex",
          ],
          interface: [
            "@radix-ui/react-dialog",
            "@radix-ui/react-dropdown-menu",
            "@radix-ui/react-tooltip",
            "lucide-react",
            "motion",
          ],
        },
      },
    },
  },
}));
