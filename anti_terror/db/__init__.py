"""Database backends for AntiTerror system.

Provides PostgreSQL backend with pgvector support alongside the original SQLite.
"""

from .postgres_database import PostgresDatabase

__all__ = ["PostgresDatabase"]
