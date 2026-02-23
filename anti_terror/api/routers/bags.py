"""Bags API router."""
from __future__ import annotations

from fastapi import APIRouter, Request, Query
from fastapi.responses import Response, JSONResponse

from ..schemas import BagOut

router = APIRouter()


def _get_pool(request: Request):
    return request.app.state.pool


@router.get("", response_model=list[BagOut])
async def list_bags(
    request: Request,
    abandoned: bool = Query(False),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
):
    pool = _get_pool(request)
    offset = (page - 1) * per_page
    async with pool.acquire() as conn:
        if abandoned:
            rows = await conn.fetch(
                "SELECT bag_id, owner_person_id, first_seen, last_seen, image_path, "
                "is_abandoned, (image IS NOT NULL) AS has_image "
                "FROM bags WHERE is_abandoned = TRUE "
                "ORDER BY last_seen DESC LIMIT $1 OFFSET $2",
                per_page, offset,
            )
        else:
            rows = await conn.fetch(
                "SELECT bag_id, owner_person_id, first_seen, last_seen, image_path, "
                "is_abandoned, (image IS NOT NULL) AS has_image "
                "FROM bags ORDER BY last_seen DESC LIMIT $1 OFFSET $2",
                per_page, offset,
            )
    return [
        BagOut(
            bag_id=r["bag_id"],
            owner_person_id=r["owner_person_id"],
            first_seen=r["first_seen"],
            last_seen=r["last_seen"],
            image_path=r["image_path"],
            is_abandoned=r["is_abandoned"],
            has_image=r["has_image"],
        )
        for r in rows
    ]


@router.get("/{bag_id}", response_model=BagOut)
async def get_bag(request: Request, bag_id: str):
    pool = _get_pool(request)
    async with pool.acquire() as conn:
        r = await conn.fetchrow(
            "SELECT bag_id, owner_person_id, first_seen, last_seen, image_path, "
            "is_abandoned, (image IS NOT NULL) AS has_image "
            "FROM bags WHERE bag_id = $1", bag_id,
        )
    if r is None:
        return JSONResponse(status_code=404, content={"detail": "Bag not found"})
    return BagOut(
        bag_id=r["bag_id"],
        owner_person_id=r["owner_person_id"],
        first_seen=r["first_seen"],
        last_seen=r["last_seen"],
        image_path=r["image_path"],
        is_abandoned=r["is_abandoned"],
        has_image=r["has_image"],
    )


@router.get("/{bag_id}/image")
async def get_bag_image(request: Request, bag_id: str):
    pool = _get_pool(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT image FROM bags WHERE bag_id = $1", bag_id,
        )
    if row is None or row["image"] is None:
        return JSONResponse(status_code=404, content={"detail": "Image not found"})
    return Response(content=bytes(row["image"]), media_type="image/jpeg")
