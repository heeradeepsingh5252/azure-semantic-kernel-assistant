import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/health": "http://127.0.0.1:8001",
      "/ingest": "http://127.0.0.1:8001",
      "/chat": "http://127.0.0.1:8001",
      "/documents": "http://127.0.0.1:8001",
    },
  },
});
