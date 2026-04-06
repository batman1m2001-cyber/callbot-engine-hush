"""Customer store — CRUD for customer script_data (JSON files)."""

import json
import logging
from pathlib import Path

from server import config

LOGGER = logging.getLogger(__name__)


def _customer_path(customer_id: str) -> Path:
    return Path(config.CUSTOMER_INFO_DIR) / f"{customer_id}.json"


def get(customer_id: str) -> dict | None:
    """Load script_data for a customer. Returns None if not found."""
    path = _customer_path(customer_id)
    if not path.exists():
        LOGGER.warning(f"Customer {customer_id} not found at {path}")
        return None
    with open(path) as f:
        return json.load(f)


def save(customer_id: str, data: dict):
    """Save script_data for a customer."""
    path = _customer_path(customer_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    LOGGER.info(f"Customer {customer_id} saved to {path}")


def delete(customer_id: str) -> bool:
    """Delete a customer's script_data. Returns True if existed."""
    path = _customer_path(customer_id)
    if path.exists():
        path.unlink()
        return True
    return False
