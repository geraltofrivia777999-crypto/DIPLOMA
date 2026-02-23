"""FastAPI application with asyncpg lifespan and Jinja2 templates."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse

from .settings import settings
from .routers import persons, bags, ownership, events, stats, auth as auth_router

_HERE = Path(__file__).parent


def _parse_preview_streams() -> dict[str, int]:
    """Parse 'CAM_01=8081,CAM_02=8082' into {camera_id: port}."""
    streams = {}
    for part in settings.preview_streams.split(","):
        part = part.strip()
        if "=" in part:
            cam_id, port = part.split("=", 1)
            streams[cam_id.strip()] = int(port.strip())
    return streams


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create asyncpg pool on startup
    dsn = settings.postgres_dsn
    app.state.pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
    app.state.cameras = _parse_preview_streams()
    yield
    await app.state.pool.close()


app = FastAPI(
    title="AntiTerror Dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

# Static files & templates
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")
templates = Jinja2Templates(directory=str(_HERE / "templates"))

# API routers
app.include_router(auth_router.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(persons.router, prefix="/api/v1/persons", tags=["persons"])
app.include_router(bags.router, prefix="/api/v1/bags", tags=["bags"])
app.include_router(ownership.router, prefix="/api/v1/ownership", tags=["ownership"])
app.include_router(events.router, prefix="/api/v1/events", tags=["events"])
app.include_router(stats.router, prefix="/api/v1/stats", tags=["stats"])


# ── Helper to get DB pool from request ───────────────────────────
def get_pool(request: Request) -> asyncpg.Pool:
    return request.app.state.pool


# ── MJPEG stream proxy ───────────────────────────────────────────

@app.get("/api/v1/cameras")
async def list_cameras(request: Request):
    """List configured camera streams."""
    cameras = request.app.state.cameras
    return [{"camera_id": cam_id, "port": port} for cam_id, port in cameras.items()]


@app.get("/api/v1/cameras/{camera_id}/stream")
async def camera_stream(request: Request, camera_id: str):
    """Proxy MJPEG stream from the pipeline's PreviewServer.

    Uses raw socket-level streaming with automatic reconnection
    to handle pipeline restarts and slow frames.
    """
    import asyncio

    cameras = request.app.state.cameras
    if camera_id not in cameras:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"detail": f"Camera {camera_id} not found"})

    port = cameras[camera_id]

    async def proxy_mjpeg():
        while True:
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                # Send minimal HTTP GET
                writer.write(b"GET / HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
                await writer.drain()

                # Skip HTTP response headers
                while True:
                    line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                    if line in (b"\r\n", b"\n", b""):
                        break

                # Stream body chunks
                while True:
                    chunk = await asyncio.wait_for(reader.read(8192), timeout=5.0)
                    if not chunk:
                        break
                    yield chunk

                writer.close()
            except (ConnectionRefusedError, OSError):
                # Pipeline not running yet — send a 1x1 black JPEG and wait
                await asyncio.sleep(2)
                continue
            except asyncio.TimeoutError:
                # No data for 5s — reconnect
                continue
            except GeneratorExit:
                return

    return StreamingResponse(
        proxy_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Dashboard HTML routes ────────────────────────────────────────

@app.get("/")
async def dashboard(request: Request):
    pool = get_pool(request)
    async with pool.acquire() as conn:
        person_count = await conn.fetchval("SELECT COUNT(*) FROM persons")
        bag_count = await conn.fetchval("SELECT COUNT(*) FROM bags")
        abandoned = await conn.fetchval("SELECT COUNT(*) FROM bags WHERE is_abandoned = TRUE")
        recent_events = await conn.fetch(
            "SELECT id, timestamp, event_type, bag_id, person_id, details "
            "FROM events ORDER BY timestamp DESC LIMIT 10"
        )
    cameras = request.app.state.cameras
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "person_count": person_count,
        "bag_count": bag_count,
        "abandoned_count": abandoned,
        "recent_events": [dict(e) for e in recent_events],
        "cameras": cameras,
    })


@app.get("/persons")
async def persons_page(request: Request):
    pool = get_pool(request)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT person_id, first_seen, last_seen, embedding_count, best_image_path, "
            "(best_image IS NOT NULL) AS has_image "
            "FROM persons ORDER BY last_seen DESC LIMIT 100"
        )
    return templates.TemplateResponse("persons.html", {
        "request": request,
        "persons": [dict(r) for r in rows],
    })


@app.get("/persons/{person_id}")
async def person_detail_page(request: Request, person_id: str):
    pool = get_pool(request)
    async with pool.acquire() as conn:
        person = await conn.fetchrow(
            "SELECT person_id, first_seen, last_seen, embedding_count, best_image_path, "
            "(best_image IS NOT NULL) AS has_image "
            "FROM persons WHERE person_id = $1", person_id
        )
        bags_rows = await conn.fetch(
            "SELECT bag_id, first_seen, last_seen, is_abandoned "
            "FROM bags WHERE owner_person_id = $1", person_id
        )
        events_rows = await conn.fetch(
            "SELECT id, timestamp, event_type, bag_id, details "
            "FROM events WHERE person_id = $1 ORDER BY timestamp DESC LIMIT 50",
            person_id
        )
    return templates.TemplateResponse("person_detail.html", {
        "request": request,
        "person": dict(person) if person else None,
        "bags": [dict(b) for b in bags_rows],
        "events": [dict(e) for e in events_rows],
    })


@app.get("/bags")
async def bags_page(request: Request, abandoned: bool = False):
    pool = get_pool(request)
    async with pool.acquire() as conn:
        if abandoned:
            rows = await conn.fetch(
                "SELECT bag_id, owner_person_id, first_seen, last_seen, is_abandoned, "
                "(image IS NOT NULL) AS has_image "
                "FROM bags WHERE is_abandoned = TRUE ORDER BY last_seen DESC LIMIT 100"
            )
        else:
            rows = await conn.fetch(
                "SELECT bag_id, owner_person_id, first_seen, last_seen, is_abandoned, "
                "(image IS NOT NULL) AS has_image "
                "FROM bags ORDER BY last_seen DESC LIMIT 100"
            )
    return templates.TemplateResponse("bags.html", {
        "request": request,
        "bags": [dict(r) for r in rows],
        "filter_abandoned": abandoned,
    })


@app.get("/search")
async def search_page(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})


@app.get("/alerts")
async def alerts_page(request: Request):
    return templates.TemplateResponse("alerts.html", {"request": request})
