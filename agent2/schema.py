"""
agent2/schema.py — Pydantic schema for Agent 2 output.
"""

from typing import Literal

from pydantic import BaseModel, Field


class TradingSignal(BaseModel):
    ticker: str
    direction: Literal["long", "short", "neutral"]
    strategy_tag: Literal["momentum", "mean_reversion", "event_driven", "macro", "none"]
    confidence: float = Field(ge=0.0, le=1.0)
    cot: str
