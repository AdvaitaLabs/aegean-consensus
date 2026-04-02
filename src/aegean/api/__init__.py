"""
FastAPI service for Aegean consensus.
"""

from aegean.api.app import create_app
from aegean.api import risk_api
from aegean.api import investment_api

__all__ = ["create_app", "risk_api", "investment_api"]

