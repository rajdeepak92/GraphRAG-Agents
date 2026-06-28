"""PostgreSQL health checks for the Phase 4 control plane."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

REQUIRED_ALEMBIC_REVISION = "95d3d37e4060"

REQUIRED_TABLES = {
    "projects",
    "documents",
    "document_versions",
    "source_files",
    "ingestion_runs",
    "run_steps",
    "chunk_manifests",
    "chunks",
    "discovery_runs",
    "discovery_batches",
    "facts",
    "fact_evidence",
    "requirements",
    "requirement_fact_links",
    "requirement_relations",
    "artifacts",
    "projection_jobs",
    "outbox_events",
}

REQUIRED_UNIQUE_CONSTRAINTS = {
    ("projects", ("project_key",)),
    ("documents", ("project_id", "logical_document_name")),
    ("document_versions", ("document_id", "supplied_version")),
    ("chunks", ("document_version_id", "chunk_ordinal")),
    ("chunks", ("document_version_id", "chunk_content_hash", "chunk_ordinal")),
    ("facts", ("fact_id",)),
    ("requirements", ("requirement_id",)),
    ("run_steps", ("run_id", "step_name", "attempt")),
}


@dataclass(frozen=True)
class HealthCheck:
    name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class PostgresHealthReport:
    checks: tuple[HealthCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


async def run_postgres_health_check(
    engine: AsyncEngine,
    *,
    required_revision: str = REQUIRED_ALEMBIC_REVISION,
) -> PostgresHealthReport:
    checks: list[HealthCheck] = []

    try:
        async with engine.connect() as conn:
            result = await conn.scalar(text("SELECT 1"))
            checks.append(
                HealthCheck(
                    name="PostgreSQL",
                    passed=result == 1,
                    message="connection OK" if result == 1 else "connection failed",
                )
            )

            revision_rows = await conn.execute(text("SELECT version_num FROM alembic_version"))
            revisions = set(revision_rows.scalars().all())
            checks.append(
                HealthCheck(
                    name="Alembic",
                    passed=required_revision in revisions,
                    message=(
                        "required revision present"
                        if required_revision in revisions
                        else f"required revision missing: {required_revision}"
                    ),
                )
            )

            table_rows = await conn.execute(
                text(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_type = 'BASE TABLE'
                    """
                )
            )
            existing_tables = set(table_rows.scalars().all())
            missing_tables = sorted(REQUIRED_TABLES - existing_tables)

            checks.append(
                HealthCheck(
                    name="Tables",
                    passed=not missing_tables,
                    message=(
                        f"{len(REQUIRED_TABLES)} required tables present"
                        if not missing_tables
                        else "missing tables: " + ", ".join(missing_tables)
                    ),
                )
            )

            constraint_rows = await conn.execute(
                text(
                    """
                    SELECT
                        rel.relname AS table_name,
                        array_agg(att.attname ORDER BY cols.ordinality) AS columns
                    FROM pg_constraint con
                    JOIN pg_class rel
                        ON rel.oid = con.conrelid
                    JOIN pg_namespace nsp
                        ON nsp.oid = rel.relnamespace
                    JOIN unnest(con.conkey) WITH ORDINALITY AS cols(attnum, ordinality)
                        ON true
                    JOIN pg_attribute att
                        ON att.attrelid = rel.oid
                       AND att.attnum = cols.attnum
                    WHERE nsp.nspname = 'public'
                      AND con.contype = 'u'
                    GROUP BY rel.relname, con.oid
                    """
                )
            )

            unique_index_rows = await conn.execute(
                text(
                    """
                    SELECT
                        tbl.relname AS table_name,
                        array_agg(att.attname ORDER BY keys.ordinality) AS columns
                    FROM pg_index idx
                    JOIN pg_class ind
                        ON ind.oid = idx.indexrelid
                    JOIN pg_class tbl
                        ON tbl.oid = idx.indrelid
                    JOIN pg_namespace nsp
                        ON nsp.oid = tbl.relnamespace
                    JOIN unnest(idx.indkey) WITH ORDINALITY AS keys(attnum, ordinality)
                        ON true
                    JOIN pg_attribute att
                        ON att.attrelid = tbl.oid
                       AND att.attnum = keys.attnum
                    WHERE nsp.nspname = 'public'
                      AND idx.indisunique = true
                      AND idx.indisprimary = false
                    GROUP BY tbl.relname, ind.relname
                    """
                )
            )

            existing_unique_shapes = {
                (row["table_name"], tuple(row["columns"])) for row in constraint_rows.mappings()
            }

            existing_unique_shapes.update(
                {(row["table_name"], tuple(row["columns"])) for row in unique_index_rows.mappings()}
            )

            missing_constraints = sorted(REQUIRED_UNIQUE_CONSTRAINTS - existing_unique_shapes)

            checks.append(
                HealthCheck(
                    name="Constraints",
                    passed=not missing_constraints,
                    message=(
                        "required unique constraints/indexes present"
                        if not missing_constraints
                        else "missing unique constraints/indexes: "
                        + "; ".join(
                            f"{table}({', '.join(columns)})"
                            for table, columns in missing_constraints
                        )
                    ),
                )
            )

    except Exception as exc:
        checks.append(
            HealthCheck(
                name="PostgreSQL",
                passed=False,
                message=f"health check failed: {exc}",
            )
        )
        return PostgresHealthReport(checks=tuple(checks))

    checks.append(await _check_transaction_rollback(engine))

    return PostgresHealthReport(checks=tuple(checks))


async def _check_transaction_rollback(engine: AsyncEngine) -> HealthCheck:
    project_id = uuid4()
    project_key = f"__healthcheck_rollback_{uuid4().hex}"

    try:
        async with engine.connect() as conn:
            tx = await conn.begin()

            try:
                await conn.execute(
                    text(
                        """
                        INSERT INTO projects (
                            project_id,
                            project_key,
                            name,
                            created_at
                        )
                        VALUES (
                            :project_id,
                            :project_key,
                            :name,
                            now()
                        )
                        """
                    ),
                    {
                        "project_id": project_id,
                        "project_key": project_key,
                        "name": "rollback health check",
                    },
                )

                inside_count = await conn.scalar(
                    text(
                        """
                        SELECT COUNT(*)
                        FROM projects
                        WHERE project_key = :project_key
                        """
                    ),
                    {"project_key": project_key},
                )
            finally:
                await tx.rollback()

            outside_count = await conn.scalar(
                text(
                    """
                    SELECT COUNT(*)
                    FROM projects
                    WHERE project_key = :project_key
                    """
                ),
                {"project_key": project_key},
            )

        passed = inside_count == 1 and outside_count == 0

        return HealthCheck(
            name="Transaction rollback",
            passed=passed,
            message="rollback OK" if passed else "rollback failed",
        )

    except Exception as exc:
        return HealthCheck(
            name="Transaction rollback",
            passed=False,
            message=f"rollback check failed: {exc}",
        )


def format_postgres_health_report(report: PostgresHealthReport) -> list[str]:
    return [
        f"{check.name}: {'PASS' if check.passed else 'FAIL'} - {check.message}"
        for check in report.checks
    ]
