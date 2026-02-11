# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Install build dependencies (only what's needed)
RUN apk add --no-cache python3 make g++

# Set npm configuration for faster installs
RUN npm config set fetch-retries 3 && \
    npm config set fetch-retry-mintimeout 20000 && \
    npm config set fetch-retry-maxtimeout 120000

# Copy package files first (better layer caching)
COPY package*.json ./

# Install dependencies with optimizations
# --prefer-offline: Use npm cache if available
# --no-audit: Skip security audit (saves time)
# --progress=false: Less verbose output
# --loglevel=error: Only show errors
RUN npm ci --prefer-offline --no-audit --progress=false --loglevel=error || \
    (echo "First install failed, cleaning cache and retrying..." && \
     npm cache clean --force && \
     npm ci --prefer-offline --no-audit --progress=false --loglevel=error)

# Copy source code
COPY . .

# Build with optimizations (skip type checking for faster builds)
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
# Use fast build script that skips type checking
RUN npm run build:fast

# Production stage
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production

# Install wget for healthcheck
RUN apk add --no-cache wget

# Create non-root user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Copy necessary files from builder
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

# Use the standalone server
CMD ["node", "server.js"]

