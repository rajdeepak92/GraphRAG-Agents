"""PostgreSQL error translation.

Infrastructure errors must be translated before crossing into application/domain code.
"""

from __future__ import annotations

from sqlalchemy.exc import DBAPIError, IntegrityError

from multi_agentic_graph_rag.domain.errors import DomainError, VersionConflictError


def translate_postgres_error(exc: Exception) -> DomainError:
    """Translate SQLAlchemy/driver exceptions into domain errors."""

    if isinstance(exc, IntegrityError):
        message = str(exc.orig).lower()

        if "uq_document_versions_document_id_supplied_version" in message:
            return VersionConflictError(
                "Document version already exists for this logical document."
            )

        return DomainError(f"PostgreSQL integrity error: {exc.orig}")

    if isinstance(exc, DBAPIError):
        return DomainError(f"PostgreSQL database error: {exc.orig}")

    return DomainError(str(exc))
