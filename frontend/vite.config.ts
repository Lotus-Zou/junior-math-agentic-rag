import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  base: "/static/",
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "../static",
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
});
