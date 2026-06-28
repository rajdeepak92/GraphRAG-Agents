from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from neo4j import AsyncDriver

from multi_agentic_graph_rag.config.settings import load_settings
from multi_agentic_graph_rag.infrastructure.neo4j.client import neo4j_driver_scope
from multi_agentic_graph_rag.infrastructure.neo4j.schema import ensure_neo4j_schema


@dataclass(frozen=True)
class RequiredConstraint:
    label: str
    property_name: str


REQUIRED_CONSTRAINTS = (
    RequiredConstraint(label="Project", property_name="project_id"),
    RequiredConstraint(label="Document", property_name="document_id"),
    RequiredConstraint(label="DocumentVersion", property_name="document_version_id"),
    RequiredConstraint(label="Chunk", property_name="chunk_id"),
    RequiredConstraint(label="Fact", property_name="fact_id"),
    RequiredConstraint(label="Requirement", property_name="requirement_id"),
    RequiredConstraint(label="IngestionRun", property_name="run_id"),
)


def _secret_value(value: Any) -> str:
    """Return plain string from pydantic SecretStr or normal string."""

    if hasattr(value, "get_secret_value"):
        return str(value.get_secret_value())

    return str(value)


async def _load_constraints(
    driver: AsyncDriver,
    *,
    database: str,
) -> set[tuple[str, str]]:
    """Return unique constraints as (label, property_name)."""

    query = """
    SHOW CONSTRAINTS
    YIELD name, type, labelsOrTypes, properties
    RETURN name, type, labelsOrTypes, properties
    """

    discovered: set[tuple[str, str]] = set()

    async with driver.session(database=database) as session:
        result = await session.run(query)

        async for record in result:
            constraint_type = str(record["type"]).upper()
            labels_or_types = record["labelsOrTypes"] or []
            properties = record["properties"] or []

            if "UNIQUE" not in constraint_type and "UNIQUENESS" not in constraint_type:
                continue

            for label in labels_or_types:
                for property_name in properties:
                    discovered.add((str(label), str(property_name)))

    return discovered


async def main() -> None:
    settings = load_settings()

    neo4j_settings = settings.neo4j

    uri = str(neo4j_settings.uri)
    username = str(neo4j_settings.username)
    password = _secret_value(neo4j_settings.password)
    database = str(neo4j_settings.database)

    async with neo4j_driver_scope(
        uri=uri,
        username=username,
        password=password,
        database=database,
    ) as driver:
        await driver.verify_connectivity()

        await ensure_neo4j_schema(driver, database=database)

        discovered_constraints = await _load_constraints(driver, database=database)

    missing_constraints = [
        required
        for required in REQUIRED_CONSTRAINTS
        if (required.label, required.property_name) not in discovered_constraints
    ]

    if missing_constraints:
        missing = ", ".join(
            f"{constraint.label}.{constraint.property_name}" for constraint in missing_constraints
        )
        raise RuntimeError(f"FAIL Neo4j schema smoke test. Missing constraints: {missing}")

    print("PASS Neo4j schema smoke test.")
    print("Verified constraints:")
    for constraint in REQUIRED_CONSTRAINTS:
        print(f"  - {constraint.label}.{constraint.property_name}")


if __name__ == "__main__":
    asyncio.run(main())
