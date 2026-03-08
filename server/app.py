"""
FastAPI application for the Inventory Reasoning Environment (OpenEnv).

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openenv.core.env_server.http_server import create_app

from models import InventoryAction, InventoryObservation
from server.inventory_environment import InventoryEnvironment

app = create_app(
    InventoryEnvironment,
    InventoryAction,
    InventoryObservation,
    env_name="inventory_env",
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
