"""Pydantic response/request models for the REST API."""
from __future__ import annotations

from pydantic import BaseModel
from typing import Optional


class PersonOut(BaseModel):
    person_id: str
    first_seen: float
    last_seen: float
    embedding_count: int = 0
    best_image_path: Optional[str] = None
    has_image: bool = False


class BagOut(BaseModel):
    bag_id: str
    owner_person_id: Optional[str] = None
    first_seen: float
    last_seen: float
    image_path: Optional[str] = None
    is_abandoned: bool = False
    has_image: bool = False


class OwnershipOut(BaseModel):
    bag_id: str
    person_id: str
    confidence: float
    first_linked: float
    last_confirmed: float
    link_count: int = 1


class EventOut(BaseModel):
    id: int
    timestamp: float
    event_type: str
    bag_id: Optional[str] = None
    person_id: Optional[str] = None
    details: Optional[dict] = None


class StatsOut(BaseModel):
    persons: int
    bags: int
    abandoned_bags: int
    ownership_links: int


class SearchResult(BaseModel):
    person_id: str
    similarity: float


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class Paginated(BaseModel):
    items: list
    total: int
    page: int
    per_page: int
