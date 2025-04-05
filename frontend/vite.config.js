import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000, // Default frontend port
    open: true, // Automatically open browser
    // Optional: Proxy API requests to backend to avoid CORS issues during development
    // proxy: {
    //   '/api': {
    //     target: 'http://localhost:8000', // Your backend address
    //     changeOrigin: true,
    //     // rewrite: (path) => path.replace(/^\/api/, '') // Remove /api prefix if backend doesn't expect it
    //   }
    // }
  }
})
