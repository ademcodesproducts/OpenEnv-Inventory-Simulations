FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py demand_calculator.py demand_environment.py \
     inventory_manager.py order_processor.py performance_tracker.py \
     agent_environment.py montecarlo_simulator.py ./
COPY server/ ./server/

RUN useradd -m user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["uvicorn", "server.inventory_env:app", "--host", "0.0.0.0", "--port", "7860"]
