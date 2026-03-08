# ── Stage 1: Build React frontend ────────────────────────────────────────────
FROM node:20-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python app ───────────────────────────────────────────────────────
FROM python:3.13-slim
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY config.py demand_calculator.py demand_environment.py \
     inventory_manager.py order_processor.py performance_tracker.py \
     agent_environment.py montecarlo_simulator.py reward.py main.py app.py ./
COPY server/ ./server/
COPY agent/ ./agent/
COPY client/ ./client/

# Copy built React app
COPY --from=frontend-build /frontend/dist ./static/

RUN useradd -m user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["uvicorn", "server.inventory_env:app", "--host", "0.0.0.0", "--port", "7860"]
