import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  plugins: [],
  server: {
    port: 5173,
    proxy: {
      "/v1": { target: process.env.VITE_API_PROXY_TARGET || "http://localhost:8000", changeOrigin: true },
      "/health": { target: process.env.VITE_API_PROXY_TARGET || "http://localhost:8000", changeOrigin: true }
    }
  }
});
