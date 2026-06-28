"""Neo4j async driver factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from neo4j import AsyncDriver, AsyncGraphDatabase


def create_neo4j_driver(
    *,
    uri: str,
    username: str,
    password: str,
    database: str,
) -> AsyncDriver:
    return AsyncGraphDatabase.driver(
        uri,
        auth=(username, password),
        database=database,
    )


@asynccontextmanager
async def neo4j_driver_scope(
    *,
    uri: str,
    username: str,
    password: str,
    database: str,
) -> AsyncIterator[AsyncDriver]:
    driver = create_neo4j_driver(
        uri=uri,
        username=username,
        password=password,
        database=database,
    )
    try:
        yield driver
    finally:
        await driver.close()
