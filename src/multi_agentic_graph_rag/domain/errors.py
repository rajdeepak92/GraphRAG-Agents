"""Domain-level errors.

This module must not import infrastructure packages.
"""

from __future__ import annotations


class DomainError(ValueError):
    """Base class for deterministic domain validation errors."""


class IdentifierError(DomainError):
    """Raised when a public traceability identifier is invalid."""


class VersionConflictError(DomainError):
    """Raised when a document version conflicts with an existing checksum."""


class InvalidRunTransitionError(DomainError):
    """Raised when an ingestion run attempts an invalid state transition."""


class TraceabilityError(DomainError):
    """Raised when a fact or requirement lacks valid source traceability."""
