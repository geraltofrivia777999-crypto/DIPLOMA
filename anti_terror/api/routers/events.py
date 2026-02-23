"""Events API router with SSE support."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request, Query
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from ..schemas import EventOut

router = APIRouter()


def _get_pool(request: Request):
    return request.app.state.pool


@router.get("", response_model=list[EventOut])
async def list_events(
    request: Request,
    event_type: str | None = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
):
    pool = _get_pool(request)
    offset = (page - 1) * per_page
    async with pool.acquire() as conn:
        if event_type:
            rows = await conn.fetch(
                "SELECT id, timestamp, event_type, bag_id, person_id, details "
                "FROM events WHERE event_type = $1 "
                "ORDER BY timestamp DESC LIMIT $2 OFFSET $3",
                event_type, per_page, offset,
            )
        else:
            rows = await conn.fetch(
                "SELECT id, timestamp, event_type, bag_id, person_id, details "
                "FROM events ORDER BY timestamp DESC LIMIT $1 OFFSET $2",
                per_page, offset,
            )
    results = []
    for r in rows:
        details = r["details"]
        if isinstance(details, str):
            details = json.loads(details)
        results.append(EventOut(
            id=r["id"],
            timestamp=r["timestamp"],
            event_type=r["event_type"],
            bag_id=r["bag_id"],
            person_id=r["person_id"],
            details=details,
        ))
    return results


@router.get("/stream")
async def event_stream(request: Request):
    """SSE endpoint — polls for new events every 2 seconds."""
    pool = _get_pool(request)
    last_id = 0

    async def generate():
        nonlocal last_id
        # Get initial last ID
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT COALESCE(MAX(id), 0) AS max_id FROM events")
            last_id = row["max_id"]

        while True:
            if await request.is_disconnected():
                break
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT id, timestamp, event_type, bag_id, person_id, details "
                    "FROM events WHERE id > $1 ORDER BY id ASC LIMIT 20",
                    last_id,
                )
            for r in rows:
                last_id = r["id"]
                details = r["details"]
                if isinstance(details, str):
                    details = json.loads(details)
                event_data = {
                    "id": r["id"],
                    "timestamp": r["timestamp"],
                    "event_type": r["event_type"],
                    "bag_id": r["bag_id"],
                    "person_id": r["person_id"],
                    "details": details,
                }
                yield {"event": "alert", "data": json.dumps(event_data)}
            await asyncio.sleep(2)

    return EventSourceResponse(generate())
