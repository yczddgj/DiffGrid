import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '')
  const target = env.DIFFGRID_PROXY_TARGET || 'http://127.0.0.1:5000'
  const proxy = {
    '/api': { target, changeOrigin: true },
    '/static': { target, changeOrigin: true }
  }
  return {
    plugins: [vue(), tailwindcss()],
    server: { proxy },
    preview: { proxy }
  }
})
