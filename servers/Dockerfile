FROM python:3.11-slim-bullseye

WORKDIR /app

# put a requirements.txt in ./servers
COPY requirements.txt .

# mcp with HTTP transport support
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Each server defines its own CMD in docker-compose
# FastMCP binds/serves HTTP from inside Python, so no extra entrypoint needed
