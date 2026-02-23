"""PostgreSQL database backend with pgvector support.

Drop-in replacement for anti_terror.database.Database with additional
capabilities: BYTEA image storage, vector embeddings via pgvector,
and bag centroid persistence.

Uses psycopg3 synchronous connections with a connection pool
(the pipeline runs synchronously).
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from ..database import PersonRecord, BagRecord, OwnershipRecord


class PostgresDatabase:
    """PostgreSQL backend — API-compatible with Database (SQLite)."""

    def __init__(self, dsn: str, min_size: int = 2, max_size: int = 10):
        self.dsn = dsn
        self.pool = ConnectionPool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            open=True,
            configure=self._configure_conn,
        )
        self._ensure_schema()
        logger.info(f"PostgresDatabase connected to {dsn.split('@')[-1]}")

    @staticmethod
    def _configure_conn(conn):
        """Register pgvector type on each new connection."""
        register_vector(conn)

    def _ensure_schema(self):
        """Apply schema.sql if tables don't exist, and run migrations."""
        schema_path = Path(__file__).parent / "schema.sql"
        ddl = schema_path.read_text()
        with self.pool.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(ddl)
            conn.commit()

            # Migration: if bag_embeddings has old dimension (2148), recreate it
            try:
                row = conn.execute(
                    "SELECT atttypmod FROM pg_attribute "
                    "WHERE attrelid = 'bag_embeddings'::regclass "
                    "AND attname = 'embedding'"
                ).fetchone()
                if row and row[0] != 384:
                    logger.info(f"Migrating bag_embeddings from dim={row[0]} to 384 (DINOv2)")
                    conn.execute("DROP TABLE IF EXISTS bag_embeddings CASCADE")
                    conn.execute("""
                        CREATE TABLE bag_embeddings (
                            id SERIAL PRIMARY KEY,
                            bag_id TEXT NOT NULL UNIQUE,
                            embedding vector(384) NOT NULL,
                            updated_at DOUBLE PRECISION NOT NULL
                        )
                    """)
                    conn.commit()
            except Exception as e:
                logger.debug(f"bag_embeddings migration check: {e}")
                conn.rollback()

    # ── helpers ──────────────────────────────────────────────────────

    def _exec(self, sql: str, params=None, *, fetch: str = "none"):
        """Execute SQL and optionally fetch results."""
        with self.pool.connection() as conn:
            cur = conn.execute(sql, params)
            if fetch == "one":
                row = cur.fetchone()
                conn.commit()
                return row
            elif fetch == "all":
                rows = cur.fetchall()
                conn.commit()
                return rows
            conn.commit()
            return cur

    # ── Person operations ────────────────────────────────────────────

    def add_person(self, person_id: str, image_path: str | None = None) -> PersonRecord:
        now = time.time()
        self._exec("""
            INSERT INTO persons (person_id, first_seen, last_seen, embedding_count, best_image_path)
            VALUES (%s, %s, %s, 1, %s)
            ON CONFLICT (person_id) DO UPDATE SET
                last_seen = %s,
                embedding_count = persons.embedding_count + 1
        """, (person_id, now, now, image_path, now))
        return self.get_person(person_id)

    def get_person(self, person_id: str) -> PersonRecord | None:
        row = self._exec(
            "SELECT person_id, first_seen, last_seen, embedding_count, best_image_path "
            "FROM persons WHERE person_id = %s", (person_id,), fetch="one"
        )
        if row:
            return PersonRecord(*row)
        return None

    def get_all_persons(self) -> List[PersonRecord]:
        rows = self._exec(
            "SELECT person_id, first_seen, last_seen, embedding_count, best_image_path "
            "FROM persons ORDER BY last_seen DESC", fetch="all"
        )
        return [PersonRecord(*r) for r in rows]

    # ── Bag operations ───────────────────────────────────────────────

    def add_bag(self, bag_id: str, owner_person_id: str | None = None,
                image_path: str | None = None) -> BagRecord:
        now = time.time()
        self._exec("""
            INSERT INTO bags (bag_id, owner_person_id, first_seen, last_seen, image_path)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (bag_id) DO UPDATE SET
                last_seen = %s,
                owner_person_id = COALESCE(%s, bags.owner_person_id)
        """, (bag_id, owner_person_id, now, now, image_path, now, owner_person_id))
        return self.get_bag(bag_id)

    def get_bag(self, bag_id: str) -> BagRecord | None:
        row = self._exec(
            "SELECT bag_id, owner_person_id, first_seen, last_seen, image_path, is_abandoned "
            "FROM bags WHERE bag_id = %s", (bag_id,), fetch="one"
        )
        if row:
            return BagRecord(
                bag_id=row[0], owner_person_id=row[1],
                first_seen=row[2], last_seen=row[3],
                image_path=row[4], is_abandoned=bool(row[5])
            )
        return None

    def get_bags_by_owner(self, person_id: str) -> List[BagRecord]:
        rows = self._exec(
            "SELECT bag_id, owner_person_id, first_seen, last_seen, image_path, is_abandoned "
            "FROM bags WHERE owner_person_id = %s", (person_id,), fetch="all"
        )
        return [
            BagRecord(bag_id=r[0], owner_person_id=r[1], first_seen=r[2],
                      last_seen=r[3], image_path=r[4], is_abandoned=bool(r[5]))
            for r in rows
        ]

    def set_bag_abandoned(self, bag_id: str, is_abandoned: bool = True):
        self._exec("UPDATE bags SET is_abandoned = %s WHERE bag_id = %s",
                    (is_abandoned, bag_id))

    # ── Ownership operations ─────────────────────────────────────────

    def link_bag_to_person(self, bag_id: str, person_id: str, confidence: float = 1.0):
        now = time.time()
        with self.pool.connection() as conn:
            conn.execute("""
                INSERT INTO ownership (bag_id, person_id, confidence, first_linked, last_confirmed, link_count)
                VALUES (%s, %s, %s, %s, %s, 1)
                ON CONFLICT (bag_id, person_id) DO UPDATE SET
                    confidence = GREATEST(ownership.confidence, %s),
                    last_confirmed = %s,
                    link_count = ownership.link_count + 1
            """, (bag_id, person_id, confidence, now, now, confidence, now))
            conn.execute("""
                UPDATE bags SET owner_person_id = (
                    SELECT person_id FROM ownership
                    WHERE bag_id = %s
                    ORDER BY confidence DESC, link_count DESC
                    LIMIT 1
                ) WHERE bag_id = %s
            """, (bag_id, bag_id))
            conn.commit()

    def get_bag_owner(self, bag_id: str) -> str | None:
        bag = self.get_bag(bag_id)
        return bag.owner_person_id if bag else None

    def get_ownership_history(self, bag_id: str) -> List[OwnershipRecord]:
        rows = self._exec(
            "SELECT bag_id, person_id, confidence, first_linked, last_confirmed, link_count "
            "FROM ownership WHERE bag_id = %s ORDER BY confidence DESC, link_count DESC",
            (bag_id,), fetch="all"
        )
        return [OwnershipRecord(*r) for r in rows]

    # ── Event logging ────────────────────────────────────────────────

    def log_event(self, event_type: str, bag_id: str | None = None,
                  person_id: str | None = None, details: dict | None = None):
        self._exec("""
            INSERT INTO events (timestamp, event_type, bag_id, person_id, details)
            VALUES (%s, %s, %s, %s, %s)
        """, (time.time(), event_type, bag_id, person_id,
              json.dumps(details) if details else None))

    def get_recent_events(self, limit: int = 100) -> List[dict]:
        rows = self._exec("""
            SELECT id, timestamp, event_type, bag_id, person_id, details
            FROM events ORDER BY timestamp DESC LIMIT %s
        """, (limit,), fetch="all")
        events = []
        for r in rows:
            events.append({
                'id': r[0], 'timestamp': r[1], 'event_type': r[2],
                'bag_id': r[3], 'person_id': r[4],
                'details': r[5] if isinstance(r[5], dict) else (json.loads(r[5]) if r[5] else None)
            })
        return events

    # ── Embedding storage ────────────────────────────────────────────

    def store_embedding(self, entity_type: str, entity_id: str,
                        embedding: torch.Tensor, quality: float = 1.0):
        vec = embedding.cpu().numpy().astype(np.float32)
        # Store as generic embeddings table — for legacy compat we keep in identity_embeddings
        # if it's a face; this method is used sparingly.
        self._exec("""
            INSERT INTO identity_embeddings (person_id, embedding, quality, timestamp, is_centroid)
            VALUES (%s, %s, %s, %s, FALSE)
        """, (entity_id, vec, quality, time.time()))

    def get_embeddings(self, entity_type: str, entity_id: str) -> List[torch.Tensor]:
        rows = self._exec("""
            SELECT embedding FROM identity_embeddings
            WHERE person_id = %s AND is_centroid = FALSE
            ORDER BY quality DESC LIMIT 10
        """, (entity_id,), fetch="all")
        return [torch.from_numpy(np.array(r[0], dtype=np.float32)) for r in rows]

    # ── Identity Gallery Persistence ─────────────────────────────────

    def save_identity(self, person_id: str, embeddings: List[torch.Tensor],
                      qualities: List[float], centroid: torch.Tensor | None = None):
        now = time.time()
        paired = sorted(zip(embeddings, qualities), key=lambda x: x[1], reverse=True)

        with self.pool.connection() as conn:
            # Delete old embeddings for this person to avoid unbounded growth
            conn.execute("DELETE FROM identity_embeddings WHERE person_id = %s", (person_id,))
            conn.commit()

            for i, (emb, quality) in enumerate(paired[:10]):
                vec = emb.cpu().numpy().astype(np.float32)
                # Offset timestamp slightly per embedding to avoid unique constraint
                ts = now + i * 0.0001
                conn.execute("""
                    INSERT INTO identity_embeddings (person_id, embedding, quality, timestamp, is_centroid)
                    VALUES (%s, %s, %s, %s, FALSE)
                """, (person_id, vec, quality, ts))

            if centroid is not None:
                vec = centroid.cpu().numpy().astype(np.float32)
                conn.execute("""
                    INSERT INTO identity_embeddings (person_id, embedding, quality, timestamp, is_centroid)
                    VALUES (%s, %s, 1.0, %s, TRUE)
                """, (person_id, vec, now + 0.01))

            conn.commit()

    def load_gallery(self, max_identities: int = 500, ttl_hours: float = 8.0) -> Dict[str, dict]:
        cutoff = time.time() - (ttl_hours * 3600)
        persons = self._exec("""
            SELECT person_id, last_seen FROM persons
            WHERE last_seen > %s ORDER BY last_seen DESC LIMIT %s
        """, (cutoff, max_identities), fetch="all")

        gallery: Dict[str, dict] = {}
        for person_id, last_seen in persons:
            rows = self._exec("""
                SELECT embedding, quality, is_centroid FROM identity_embeddings
                WHERE person_id = %s ORDER BY is_centroid DESC, quality DESC LIMIT 30
            """, (person_id,), fetch="all")

            embeddings, qualities, centroid = [], [], None
            for emb_data, quality, is_centroid in rows:
                tensor = torch.from_numpy(np.array(emb_data, dtype=np.float32))
                if is_centroid:
                    centroid = tensor
                else:
                    embeddings.append(tensor)
                    qualities.append(quality)

            if embeddings:
                gallery[person_id] = {
                    'embeddings': embeddings,
                    'qualities': qualities,
                    'centroid': centroid,
                    'last_seen': last_seen,
                }

        logger.info(f"Loaded {len(gallery)} identities from database")
        return gallery

    def save_gallery_batch(self, identities: Dict[str, 'FaceIdentity']) -> None:
        for person_id, identity in identities.items():
            self.save_identity(
                person_id=person_id,
                embeddings=identity.embeddings,
                qualities=identity.qualities,
                centroid=identity.centroid,
            )

    # ── Track ID History ─────────────────────────────────────────────

    def save_track_mapping(self, person_id: str, track_id: int,
                           start_time: float, end_time: float | None = None,
                           session_id: str | None = None):
        self._exec("""
            INSERT INTO track_id_history (person_id, track_id, start_time, end_time, session_id)
            VALUES (%s, %s, %s, %s, %s)
        """, (person_id, track_id, start_time, end_time, session_id))

    def get_recent_track_history(self, window_hours: float = 1.0) -> List[Tuple]:
        cutoff = time.time() - (window_hours * 3600)
        rows = self._exec("""
            SELECT person_id, track_id, start_time, end_time
            FROM track_id_history WHERE start_time > %s ORDER BY start_time DESC
        """, (cutoff,), fetch="all")
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    # ── Session Management ───────────────────────────────────────────

    def start_session(self, camera_id: str | None = None) -> str:
        session_id = str(uuid.uuid4())[:8]
        self._exec("""
            INSERT INTO sessions (session_id, start_time, camera_id)
            VALUES (%s, %s, %s)
        """, (session_id, time.time(), camera_id))
        logger.info(f"Started session {session_id}")
        return session_id

    def end_session(self, session_id: str):
        self._exec("UPDATE sessions SET end_time = %s WHERE session_id = %s",
                    (time.time(), session_id))

    def get_last_session_id(self) -> str | None:
        row = self._exec(
            "SELECT session_id FROM sessions ORDER BY start_time DESC LIMIT 1",
            fetch="one"
        )
        return row[0] if row else None

    # ── Bag Ownership Persistence ────────────────────────────────────

    def save_bag_ownership(self, bag_id: str, bag_track_id: int,
                           owner_person_id: str | None, confidence: float,
                           is_validated: bool, first_seen: float,
                           last_seen: float, session_id: str | None = None):
        self._exec("""
            INSERT INTO bag_ownership_snapshots
            (bag_id, bag_track_id, owner_person_id, confidence, is_validated,
             first_seen, last_seen, session_id, snapshot_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (bag_id, bag_track_id, owner_person_id, confidence,
              is_validated, first_seen, last_seen, session_id, time.time()))

    def load_bag_ownerships(self, session_id: str | None = None,
                            max_age_hours: float = 8.0) -> List[dict]:
        cutoff = time.time() - (max_age_hours * 3600)
        if session_id:
            rows = self._exec("""
                SELECT * FROM bag_ownership_snapshots
                WHERE session_id = %s AND snapshot_time > %s
                ORDER BY snapshot_time DESC
            """, (session_id, cutoff), fetch="all")
        else:
            rows = self._exec("""
                SELECT * FROM bag_ownership_snapshots
                WHERE snapshot_time > %s ORDER BY snapshot_time DESC
            """, (cutoff,), fetch="all")

        ownerships = []
        seen_bags = set()
        for r in rows:
            bag_id = r[1]
            if bag_id in seen_bags:
                continue
            seen_bags.add(bag_id)
            ownerships.append({
                'bag_id': bag_id,
                'bag_track_id': r[2],
                'owner_person_id': r[3],
                'confidence': r[4],
                'is_validated': bool(r[5]),
                'first_seen': r[6],
                'last_seen': r[7],
            })

        logger.info(f"Loaded {len(ownerships)} bag ownerships from database")
        return ownerships

    def save_ownerships_batch(self, ownerships: Dict[int, 'BagOwnership'],
                              session_id: str | None = None):
        for bag_track_id, ownership in ownerships.items():
            if ownership.bag_id and ownership.owner_person_id:
                self.save_bag_ownership(
                    bag_id=ownership.bag_id,
                    bag_track_id=bag_track_id,
                    owner_person_id=ownership.owner_person_id,
                    confidence=ownership.confidence,
                    is_validated=ownership.validated,
                    first_seen=ownership.first_seen,
                    last_seen=ownership.last_owner_seen,
                    session_id=session_id,
                )

    # ── Cleanup ──────────────────────────────────────────────────────

    def cleanup_old_data(self, max_age_hours: float = 24.0) -> dict:
        cutoff = time.time() - (max_age_hours * 3600)
        deleted = {}
        with self.pool.connection() as conn:
            cur = conn.execute("DELETE FROM identity_embeddings WHERE timestamp < %s", (cutoff,))
            deleted['identity_embeddings'] = cur.rowcount
            cur = conn.execute("DELETE FROM track_id_history WHERE start_time < %s", (cutoff,))
            deleted['track_id_history'] = cur.rowcount
            cur = conn.execute("DELETE FROM bag_ownership_snapshots WHERE snapshot_time < %s", (cutoff,))
            deleted['bag_ownership_snapshots'] = cur.rowcount
            cur = conn.execute("DELETE FROM events WHERE timestamp < %s", (cutoff,))
            deleted['events'] = cur.rowcount
            conn.commit()

        if sum(deleted.values()) > 0:
            logger.info(f"Cleaned up old data: {deleted}")
        return deleted

    # ── Statistics ───────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self.pool.connection() as conn:
            person_count = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
            bag_count = conn.execute("SELECT COUNT(*) FROM bags").fetchone()[0]
            abandoned_count = conn.execute("SELECT COUNT(*) FROM bags WHERE is_abandoned = TRUE").fetchone()[0]
            link_count = conn.execute("SELECT COUNT(*) FROM ownership").fetchone()[0]
        return {
            'persons': person_count,
            'bags': bag_count,
            'abandoned_bags': abandoned_count,
            'ownership_links': link_count,
        }

    # ── NEW: Face image storage ──────────────────────────────────────

    def save_face_image(self, person_id: str, jpeg_bytes: bytes):
        """Save best face image as BYTEA."""
        self._exec(
            "UPDATE persons SET best_image = %s WHERE person_id = %s",
            (jpeg_bytes, person_id)
        )

    def get_face_image(self, person_id: str) -> bytes | None:
        row = self._exec(
            "SELECT best_image FROM persons WHERE person_id = %s",
            (person_id,), fetch="one"
        )
        return bytes(row[0]) if row and row[0] else None

    # ── NEW: Bag image storage ───────────────────────────────────────

    def save_bag_image(self, bag_id: str, jpeg_bytes: bytes):
        """Save bag image as BYTEA."""
        self._exec(
            "UPDATE bags SET image = %s WHERE bag_id = %s",
            (jpeg_bytes, bag_id)
        )

    def get_bag_image(self, bag_id: str) -> bytes | None:
        row = self._exec(
            "SELECT image FROM bags WHERE bag_id = %s",
            (bag_id,), fetch="one"
        )
        return bytes(row[0]) if row and row[0] else None

    # ── NEW: Face embedding search via pgvector ──────────────────────

    def search_by_face_embedding(self, query: np.ndarray, top_k: int = 5) -> List[dict]:
        """k-NN face search using pgvector cosine distance.

        Args:
            query: 512-dim face embedding (numpy float32)
            top_k: number of results

        Returns:
            List of {person_id, similarity} dicts
        """
        vec = query.astype(np.float32)
        rows = self._exec("""
            SELECT person_id, 1 - (embedding <=> %s::vector) AS similarity
            FROM identity_embeddings
            WHERE is_centroid = TRUE
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (vec, vec, top_k), fetch="all")
        return [{'person_id': r[0], 'similarity': float(r[1])} for r in rows]

    # ── NEW: Bag centroid persistence ────────────────────────────────

    def save_bag_centroid(self, bag_id: str, centroid_tensor: torch.Tensor):
        """Persist bag centroid embedding in bag_embeddings table."""
        vec = centroid_tensor.cpu().numpy().astype(np.float32)
        self._exec("""
            INSERT INTO bag_embeddings (bag_id, embedding, updated_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (bag_id) DO UPDATE SET
                embedding = %s, updated_at = %s
        """, (bag_id, vec, time.time(), vec, time.time()))

    def get_all_bag_centroids(self) -> Dict[str, np.ndarray]:
        """Load all bag centroids from DB."""
        rows = self._exec(
            "SELECT bag_id, embedding FROM bag_embeddings", fetch="all"
        )
        return {r[0]: np.array(r[1], dtype=np.float32) for r in rows}

    # ── Close ────────────────────────────────────────────────────────

    def close(self):
        self.pool.close()
