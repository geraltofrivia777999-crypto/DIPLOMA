"""Persons API router."""
from __future__ import annotations

import io
import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, Query
from fastapi.responses import Response, JSONResponse

from ..schemas import PersonOut, SearchResult

router = APIRouter()


def _get_pool(request: Request):
    return request.app.state.pool


@router.get("", response_model=list[PersonOut])
async def list_persons(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
):
    pool = _get_pool(request)
    offset = (page - 1) * per_page
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT person_id, first_seen, last_seen, embedding_count, best_image_path, "
            "(best_image IS NOT NULL) AS has_image "
            "FROM persons ORDER BY last_seen DESC LIMIT $1 OFFSET $2",
            per_page, offset,
        )
    return [
        PersonOut(
            person_id=r["person_id"],
            first_seen=r["first_seen"],
            last_seen=r["last_seen"],
            embedding_count=r["embedding_count"],
            best_image_path=r["best_image_path"],
            has_image=r["has_image"],
        )
        for r in rows
    ]


@router.get("/{person_id}", response_model=PersonOut)
async def get_person(request: Request, person_id: str):
    pool = _get_pool(request)
    async with pool.acquire() as conn:
        r = await conn.fetchrow(
            "SELECT person_id, first_seen, last_seen, embedding_count, best_image_path, "
            "(best_image IS NOT NULL) AS has_image "
            "FROM persons WHERE person_id = $1", person_id,
        )
    if r is None:
        return JSONResponse(status_code=404, content={"detail": "Person not found"})
    return PersonOut(
        person_id=r["person_id"],
        first_seen=r["first_seen"],
        last_seen=r["last_seen"],
        embedding_count=r["embedding_count"],
        best_image_path=r["best_image_path"],
        has_image=r["has_image"],
    )


@router.get("/{person_id}/image")
async def get_person_image(request: Request, person_id: str):
    pool = _get_pool(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT best_image FROM persons WHERE person_id = $1", person_id,
        )
    if row is None or row["best_image"] is None:
        return JSONResponse(status_code=404, content={"detail": "Image not found"})
    return Response(content=bytes(row["best_image"]), media_type="image/jpeg")


@router.post("/search", response_model=list[SearchResult])
async def search_by_face(request: Request, file: UploadFile = File(...)):
    """Upload a JPEG face photo and find matching persons via pgvector k-NN."""
    image_bytes = await file.read()

    # Decode image
    import cv2
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"detail": "Invalid image"})

    # Extract face embedding using InsightFace on CPU
    try:
        from insightface.app import FaceAnalysis
        fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        fa.prepare(ctx_id=-1, det_size=(640, 640))
        faces = fa.get(img)
        if not faces:
            return JSONResponse(status_code=400, content={"detail": "No face detected"})
        query_emb = faces[0].normed_embedding.astype(np.float32)
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Face detection failed: {e}"})

    # pgvector k-NN search — asyncpg needs vector as string '[0.1,0.2,...]'
    pool = _get_pool(request)
    vec_str = "[" + ",".join(str(float(x)) for x in query_emb) + "]"
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT person_id, 1 - (embedding <=> $1::vector) AS similarity "
            "FROM identity_embeddings "
            "WHERE is_centroid = TRUE "
            "ORDER BY embedding <=> $1::vector "
            "LIMIT 5",
            vec_str,
        )

    return [
        SearchResult(person_id=r["person_id"], similarity=float(r["similarity"]))
        for r in rows
    ]
