"""Vercel entrypoint.

The FastAPI application lives in ``prototype/backend/api.py`` so the same code
runs locally and on Vercel. Everything is rewritten to this single function
(see vercel.json), and FastAPI serves both the JSON API and the static UI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "prototype" / "backend"

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# Loaded by path rather than ``import api`` so the name cannot collide with the
# ``api/`` directory that Vercel uses for functions.
_spec = importlib.util.spec_from_file_location("pd_backend_api", BACKEND / "api.py")
if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
    raise ImportError(f"Could not load the FastAPI app from {BACKEND / 'api.py'}")

_module = importlib.util.module_from_spec(_spec)
sys.modules["pd_backend_api"] = _module
_spec.loader.exec_module(_module)

app = _module.app
