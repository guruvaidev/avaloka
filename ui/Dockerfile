# Dockerfile.frontend — production image for the develop-1.6 TanStack Start SSR UI.
#
# Multi-stage: build the nitro node-server output in a full build image, then run
# only the built output on a slim node runtime. The container listens on $PORT
# (default 3000). nginx APP_UPSTREAM proxies to it; the backend routes proxy to
# AVALOKA_API_UPSTREAM (a separate service), so this image serves SSR only.
#
# HARD DEPENDENCY: the vite/nitro build must use the "node-server" preset so that
# `npm run build` emits .output/server/index.mjs (a runnable Node server), NOT a
# Cloudflare Worker. The Lovable wrapper (@lovable.dev/vite-tanstack-config)
# currently defaults nitro to a cloudflare target — that MUST be overridden to
# node-server or the final CMD has nothing to run. See the nitro preset note.
#
# RUNTIME ENV (read live via process.env at container start, NOT baked at build):
#   SUPABASE_URL, SUPABASE_ANON_KEY, API_BASE, MCP_API_BASE, MCP_TOOL_BASE
# In-cluster these must NOT be localhost — point them at in-cluster Service URLs
# (or "" for same-origin via nginx). Set them with `kubectl set env` / Helm values,
# never hard-coded here.

# ---- build stage ----
FROM node:22-alpine AS builder
WORKDIR /app

# Install deps against the lockfile for a reproducible build.
COPY package*.json ./
RUN npm ci

# Copy source and build. Requires nitro preset "node-server" (see header note).
COPY . .
RUN NITRO_PRESET=node-server npm run build

# ---- runtime stage ----
FROM node:22-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
# nitro node-server honors PORT/HOST. Bind all interfaces so k8s can reach it.
ENV PORT=3000
ENV HOST=0.0.0.0

# Ship only the built server output — no source, no dev toolchain.
# NOTE: if startup fails with "Cannot find module 'X'", this nitro config
# externalized some deps; also copy node_modules (see the node_modules note).
COPY --from=builder /app/.output ./.output

# Run as the built-in non-root node user.
USER node

EXPOSE 3000

CMD ["node", ".output/server/index.mjs"]