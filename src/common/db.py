"""Database connection pool and helpers."""

from __future__ import annotations

import os

import psycopg
from psycopg_pool import ConnectionPool

_pool: ConnectionPool | None = None


def get_dsn() -> str:
    return (
        f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}"
        f"@{os.environ.get('PGHOST', 'localhost')}:{os.environ.get('PGPORT', '5432')}"
        f"/{os.environ.get('PGDATABASE', 'vtech_market')}"
    )


def get_pool(min_size: int = 2, max_size: int = 10) -> ConnectionPool:
    """Return a singleton connection pool (created on first call)."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=get_dsn(),
            min_size=min_size,
            max_size=max_size,
            open=True,
        )
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def execute_sql_file(path: str) -> None:
    """Execute a raw SQL file against the database."""
    with open(path) as f:
        sql = f.read()
    with psycopg.connect(get_dsn()) as conn:
        conn.execute(sql)
