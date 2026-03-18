#!/usr/bin/env bash
# ==============================================================================
# Aegean Consensus - Quick Start Script
# ==============================================================================
# Usage:
#   ./scripts/start.sh           # default: start with Python directly (dev mode)
#   ./scripts/start.sh docker    # start with Docker Compose
#   ./scripts/start.sh seed      # run data initialization only
#   ./scripts/start.sh stop      # stop Docker services
# ==============================================================================

set -e
cd "$(dirname "$0")/.."

MODE=${1:-dev}

print_banner() {
  echo ""
  echo "================================================="
  echo "  Aegean Consensus Platform"
  echo "  Multi-Agent Consensus + Financial Risk VAN"
  echo "================================================="
  echo ""
}

check_env() {
  if [ ! -f ".env" ]; then
    echo "[WARN]  .env not found. Creating from env.example..."
    cp env.example .env
    echo "[OK] Created .env - please edit it and set your OPENAI_API_KEY"
    echo "   vim .env"
    echo ""
    echo "   NOTE: Without an API key, validators use rule-based pre-screen only."
    echo "         The system still works but LLM deep analysis is disabled."
    echo ""
  fi
  source .env 2>/dev/null || true
}

start_dev() {
  print_banner
  check_env

  echo ">>  Mode: Development (direct Python)"
  echo ">>  Checking Python environment..."

  # Find python
  PYTHON=""
  for py in python3 python3.11 python3.12 python3.10; do
    if command -v $py &>/dev/null; then
      PYTHON=$py
      break
    fi
  done

  if [ -z "$PYTHON" ]; then
    echo "[ERR] Python not found. Install Python 3.9+"
    exit 1
  fi

  echo ">>  Python: $($PYTHON --version)"

  # Check venv
  if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo ">>  Virtualenv: .venv activated"
  fi

  # Install if needed
  if ! $PYTHON -c 'import fastapi' 2>/dev/null; then
    echo ">>  Installing dependencies..."
    $PYTHON -m pip install -r requirements.txt -q
    $PYTHON -m pip install -e . -q
  fi

  echo ">>  Starting Aegean API..."
  echo ">>  API docs: http://localhost:${AEGEAN_PORT:-8000}/docs"
  echo ""

  $PYTHON main.py \
    --host "${AEGEAN_HOST:-0.0.0.0}" \
    --port "${AEGEAN_PORT:-8000}" \
    --reload
}

start_docker() {
  print_banner
  check_env

  echo ">>  Mode: Docker Compose"

  if ! command -v docker &>/dev/null; then
    echo "[ERR] Docker not found. Install Docker Desktop: https://docs.docker.com/get-docker/"
    exit 1
  fi

  if ! command -v docker-compose &>/dev/null && ! docker compose version &>/dev/null 2>&1; then
    echo "[ERR] docker-compose not found"
    exit 1
  fi

  echo ">>  Building image..."
  docker compose build --quiet

  echo ">>  Starting services..."
  docker compose up -d

  echo ""
  echo "[OK] Aegean is running!"
  echo "   API:  http://localhost:${AEGEAN_PORT:-8000}"
  echo "   Docs: http://localhost:${AEGEAN_PORT:-8000}/docs"
  echo ""
  echo "   Logs:  docker compose logs -f aegean"
  echo "   Stop:  ./scripts/start.sh stop"
}

run_seed() {
  print_banner
  check_env

  PYTHON=""
  for py in python3 python3.11 python3.12; do
    if command -v $py &>/dev/null; then PYTHON=$py; break; fi
  done

  [ -d ".venv" ] && source .venv/bin/activate

  echo ">>  Step 1/3: Seeding built-in risk knowledge (18 docs)..."
  $PYTHON -c "
import asyncio, sys
sys.path.insert(0, 'src')
from aegean.memory.global_memory import GlobalMemorySystem
from aegean.risk.data_seed import RiskKnowledgeSeeder
async def run():
    mem = GlobalMemorySystem()
    seeder = RiskKnowledgeSeeder(mem)
    n = await seeder.seed_all(skip_if_exists=False)
    print(f'  Seeded {n} documents')
asyncio.run(run())
"

  echo ">>  Step 2/3: Generating synthetic risk cases..."
  $PYTHON scripts/generate_synthetic.py --count 0

  echo ">>  Step 3/3: OFAC sanctions list (requires internet)..."
  $PYTHON scripts/fetch_ofac.py --limit 200 2>/dev/null || echo "  (OFAC download skipped - check internet connection)"

  echo ""
  echo "[OK] Knowledge base seeded and ready!"
}

stop_docker() {
  echo ">>  Stopping Docker services..."
  docker compose down
  echo "[OK] Stopped"
}

# Router
case $MODE in
  dev|python|"")
    start_dev
    ;;
  docker)
    start_docker
    ;;
  seed)
    run_seed
    ;;
  stop)
    stop_docker
    ;;
  *)
    echo "Usage: $0 [dev|docker|seed|stop]"
    exit 1
    ;;
esac

