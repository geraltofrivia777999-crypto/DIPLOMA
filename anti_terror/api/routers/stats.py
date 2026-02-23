"""Stats API router."""
from __future__ import annotations

from fastapi import APIRouter, Request

from ..schemas import StatsOut

router = APIRouter()


def _get_pool(request: Request):
    return request.app.state.pool


@router.get("", response_model=StatsOut)
async def get_stats(request: Request):
    pool = _get_pool(request)
    async with pool.acquire() as conn:
        persons = await conn.fetchval("SELECT COUNT(*) FROM persons")
        bags = await conn.fetchval("SELECT COUNT(*) FROM bags")
        abandoned = await conn.fetchval("SELECT COUNT(*) FROM bags WHERE is_abandoned = TRUE")
        links = await conn.fetchval("SELECT COUNT(*) FROM ownership")
    return StatsOut(
        persons=persons,
        bags=bags,
        abandoned_bags=abandoned,
        ownership_links=links,
    )
