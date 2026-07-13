"""Cross-version assertion lifecycle: supersede / retire / new resolution.

Given the assertions extracted for a new document version and the assertions that
were active in the document's *prior* active knowledge version, this decides, per
semantic slot (``assertion_lineage_key``):

* same lineage + same exact ``assertion_key``  -> unchanged revision
* same lineage + different exact ``assertion_key`` -> changed revision
* prior lineage absent from the new version     -> retired
* new lineage absent from the prior version      -> new assertion

Ambiguous collisions — more than one new assertion mapping to the same lineage in
a single version — are detected and left unlinked (both kept active, flagged)
rather than silently merged. Pure and deterministic for unit testing; the Neo4j
writes are applied by the store from :class:`LifecycleResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from multi_agentic_graph_rag.domain.schemas import AssertionRecord


@dataclass(frozen=True)
class PriorAssertion:
    """Minimal identity of an assertion active in the prior knowledge version."""

    assertion_id: str
    assertion_key: str
    assertion_lineage_key: str


@dataclass(frozen=True)
class LifecycleUpdate:
    """A status change to apply to a *prior*-version assertion node."""

    assertion_id: str
    status: str  # "superseded" | "retired"
    superseded_by_assertion_id: str | None = None


@dataclass
class LifecycleResult:
    """Coordinate lifecycle result behavior within the services boundary."""

    assertions: list[AssertionRecord] = field(default_factory=list)
    prior_updates: list[LifecycleUpdate] = field(default_factory=list)
    ambiguous_lineage_keys: list[str] = field(default_factory=list)


def reconcile_assertion_lifecycle(
    *,
    new_assertions: list[AssertionRecord],
    prior_assertions: list[PriorAssertion],
) -> LifecycleResult:
    """Reconcile assertion lifecycle deterministically within the active scope.

    Args:
        new_assertions (list[AssertionRecord]): New assertions required by the operation's typed
                                                contract.
        prior_assertions (list[PriorAssertion]): Prior assertions required by the operation's typed
                                                 contract.

    Returns:
        LifecycleResult: The typed result produced by the operation.
    """
    new_by_lineage: dict[str, list[AssertionRecord]] = {}
    for assertion in new_assertions:
        new_by_lineage.setdefault(assertion.assertion_lineage_key, []).append(assertion)

    prior_by_lineage: dict[str, PriorAssertion] = {}
    for item in prior_assertions:
        prior_by_lineage.setdefault(item.assertion_lineage_key, item)

    ambiguous = sorted(key for key, group in new_by_lineage.items() if len(group) > 1)
    ambiguous_set = set(ambiguous)

    resolved: list[AssertionRecord] = []
    prior_updates: list[LifecycleUpdate] = []
    matched_prior: set[str] = set()

    for assertion in new_assertions:
        lineage = assertion.assertion_lineage_key
        prior = prior_by_lineage.get(lineage)
        if lineage in ambiguous_set:
            # Ambiguous slot: keep every occurrence active and unlinked.
            resolved.append(
                assertion.model_copy(update={"status": "active", "revision_type": "new"})
            )
            if prior is not None and prior.assertion_id not in matched_prior:
                matched_prior.add(prior.assertion_id)
                prior_updates.append(
                    LifecycleUpdate(assertion_id=prior.assertion_id, status="superseded")
                )
            continue
        if prior is None:
            resolved.append(
                assertion.model_copy(update={"status": "active", "revision_type": "new"})
            )
            continue
        matched_prior.add(prior.assertion_id)
        unchanged = assertion.assertion_key == prior.assertion_key
        resolved.append(
            assertion.model_copy(
                update={
                    "status": "active",
                    "previous_assertion_id": prior.assertion_id,
                    "revision_type": "unchanged" if unchanged else "changed",
                }
            )
        )
        prior_updates.append(
            LifecycleUpdate(
                assertion_id=prior.assertion_id,
                status="superseded",
                superseded_by_assertion_id=assertion.assertion_id,
            )
        )

    # Retire any prior lineage that has no assertion in the new version at all.
    for lineage, prior_assertion in prior_by_lineage.items():
        if prior_assertion.assertion_id in matched_prior:
            continue
        if lineage in new_by_lineage:
            continue
        prior_updates.append(
            LifecycleUpdate(assertion_id=prior_assertion.assertion_id, status="retired")
        )

    return LifecycleResult(
        assertions=resolved,
        prior_updates=prior_updates,
        ambiguous_lineage_keys=ambiguous,
    )
