-- AntiTerror PostgreSQL schema
-- Requires: CREATE EXTENSION IF NOT EXISTS vector;

-- Persons table
CREATE TABLE IF NOT EXISTS persons (
    person_id   TEXT PRIMARY KEY,
    first_seen  DOUBLE PRECISION NOT NULL,
    last_seen   DOUBLE PRECISION NOT NULL,
    embedding_count INTEGER DEFAULT 0,
    best_image_path TEXT,
    best_image  BYTEA
);

-- Bags table
CREATE TABLE IF NOT EXISTS bags (
    bag_id          TEXT PRIMARY KEY,
    owner_person_id TEXT REFERENCES persons(person_id),
    first_seen      DOUBLE PRECISION NOT NULL,
    last_seen       DOUBLE PRECISION NOT NULL,
    image_path      TEXT,
    image           BYTEA,
    is_abandoned    BOOLEAN DEFAULT FALSE
);

-- Ownership history
CREATE TABLE IF NOT EXISTS ownership (
    id              SERIAL PRIMARY KEY,
    bag_id          TEXT NOT NULL REFERENCES bags(bag_id),
    person_id       TEXT NOT NULL REFERENCES persons(person_id),
    confidence      DOUBLE PRECISION DEFAULT 1.0,
    first_linked    DOUBLE PRECISION NOT NULL,
    last_confirmed  DOUBLE PRECISION NOT NULL,
    link_count      INTEGER DEFAULT 1,
    UNIQUE(bag_id, person_id)
);

-- Events
CREATE TABLE IF NOT EXISTS events (
    id          SERIAL PRIMARY KEY,
    timestamp   DOUBLE PRECISION NOT NULL,
    event_type  TEXT NOT NULL,
    bag_id      TEXT,
    person_id   TEXT,
    details     JSONB
);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);

-- Identity embeddings (face gallery persistence)
CREATE TABLE IF NOT EXISTS identity_embeddings (
    id          SERIAL PRIMARY KEY,
    person_id   TEXT NOT NULL,
    embedding   vector(512) NOT NULL,
    quality     DOUBLE PRECISION DEFAULT 1.0,
    timestamp   DOUBLE PRECISION NOT NULL,
    is_centroid BOOLEAN DEFAULT FALSE,
    UNIQUE(person_id, is_centroid, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_identity_embeddings_person ON identity_embeddings(person_id);

-- IVFFlat index for k-NN face search
-- Created after initial data load for better list estimation
-- CREATE INDEX IF NOT EXISTS idx_identity_embeddings_vector
--     ON identity_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Bag embeddings — DINOv2-small produces 384-dim vectors
CREATE TABLE IF NOT EXISTS bag_embeddings (
    id          SERIAL PRIMARY KEY,
    bag_id      TEXT NOT NULL UNIQUE,
    embedding   vector(384) NOT NULL,
    updated_at  DOUBLE PRECISION NOT NULL
);

-- Track ID history
CREATE TABLE IF NOT EXISTS track_id_history (
    id          SERIAL PRIMARY KEY,
    person_id   TEXT NOT NULL REFERENCES persons(person_id),
    track_id    INTEGER NOT NULL,
    start_time  DOUBLE PRECISION NOT NULL,
    end_time    DOUBLE PRECISION,
    session_id  TEXT
);
CREATE INDEX IF NOT EXISTS idx_track_history_person ON track_id_history(person_id);

-- Sessions
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    start_time  DOUBLE PRECISION NOT NULL,
    end_time    DOUBLE PRECISION,
    camera_id   TEXT,
    metadata    JSONB
);

-- Bag ownership snapshots
CREATE TABLE IF NOT EXISTS bag_ownership_snapshots (
    id                SERIAL PRIMARY KEY,
    bag_id            TEXT NOT NULL,
    bag_track_id      INTEGER,
    owner_person_id   TEXT,
    confidence        DOUBLE PRECISION DEFAULT 0.0,
    is_validated      BOOLEAN DEFAULT FALSE,
    first_seen        DOUBLE PRECISION,
    last_seen         DOUBLE PRECISION,
    session_id        TEXT,
    snapshot_time     DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bag_ownership_session ON bag_ownership_snapshots(session_id);
