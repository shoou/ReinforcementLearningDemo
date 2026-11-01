# Codex project rules

- Do not scan, read, summarize, edit, or analyze `node_modules/`.
- Do not scan generated directories: `dist/`, `.vite/`, `.nuxt/`, `.output/`, `.venv*/`, `coverage/`.
- Treat these directories as third-party or build artifacts.
- When analyzing frontend code, focus only on:
  - `src/`
  - `public/`
  - `package.json`
  - `vite.config.*`
  - `vue.config.*`
  - `tsconfig*.json`
  - `.env.example`
- If dependency source code is needed, ask before reading anything under `node_modules/`.
