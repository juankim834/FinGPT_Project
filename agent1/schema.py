"""
agent1/schema.py — Pydantic schema for Agent 1 output.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


SentimentLabel = Literal[
    "strongly bullish",
    "bullish",
    "neutral",
    "bearish",
    "strongly bearish",
]


class NewsFingerprint(BaseModel):
    source: str
    published_at: str
    headline: str
    companies_named: list[str]
    event_keywords: list[str]
    sentiment_label: SentimentLabel
    sentiment_score: float = Field(ge=-2.0, le=2.0)
    sentiment_confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("companies_named")
    @classmethod
    def companies_must_be_nonempty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("no company identified - skip article")
        return value

    @field_validator("event_keywords")
    @classmethod
    def keywords_must_be_lowercase(cls, value: list[str]) -> list[str]:
        return [keyword.lower() for keyword in value]
