"""Ownership API router."""
from __future__ import annotations

from fastapi import APIRouter, Request, Query

from ..schemas import OwnershipOut

router = APIRouter()


def _get_pool(request: Request):
    return request.app.state.pool


@router.get("", response_model=list[OwnershipOut])
async def list_ownership(
    request: Request,
    bag_id: str | None = Query(None),
    person_id: str | None = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
):
    pool = _get_pool(request)
    offset = (page - 1) * per_page
    async with pool.acquire() as conn:
        if bag_id:
            rows = await conn.fetch(
                "SELECT bag_id, person_id, confidence, first_linked, last_confirmed, link_count "
                "FROM ownership WHERE bag_id = $1 "
                "ORDER BY confidence DESC LIMIT $2 OFFSET $3",
                bag_id, per_page, offset,
            )
        elif person_id:
            rows = await conn.fetch(
                "SELECT bag_id, person_id, confidence, first_linked, last_confirmed, link_count "
                "FROM ownership WHERE person_id = $1 "
                "ORDER BY confidence DESC LIMIT $2 OFFSET $3",
                person_id, per_page, offset,
            )
        else:
            rows = await conn.fetch(
                "SELECT bag_id, person_id, confidence, first_linked, last_confirmed, link_count "
                "FROM ownership ORDER BY last_confirmed DESC LIMIT $1 OFFSET $2",
                per_page, offset,
            )
    return [
        OwnershipOut(
            bag_id=r["bag_id"],
            person_id=r["person_id"],
            confidence=r["confidence"],
            first_linked=r["first_linked"],
            last_confirmed=r["last_confirmed"],
            link_count=r["link_count"],
        )
        for r in rows
    ]
