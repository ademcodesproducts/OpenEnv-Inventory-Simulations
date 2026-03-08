FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py demand_calculator.py demand_environment.py \
     inventory_manager.py order_processor.py performance_tracker.py \
     agent_environment.py montecarlo_simulator.py reward.py main.py \
     models.py inventory_env_client.py openenv.yaml ./
COPY server/ ./server/
COPY agent/ ./agent/
COPY client/ ./client/

RUN useradd -m user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
ENV ENABLE_WEB_INTERFACE=true

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
