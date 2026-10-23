import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vitest/config'

// The browser only ever talks to its own origin at /api; Vite forwards it to FastAPI. That avoids
// CORS entirely (no cross-origin surface to configure or get wrong) and keeps secrets such as the
// TMDB key on the server: they are never in this bundle.
const api = {
  '/api': {
    target: process.env.CINEMATCH_API ?? 'http://127.0.0.1:8000',
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/api/, ''),
  },
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy: api },
  preview: { proxy: api },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
  },
})
