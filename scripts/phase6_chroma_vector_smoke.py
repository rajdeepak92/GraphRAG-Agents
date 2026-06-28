from __future__ import annotations

import math
import re
from uuid import uuid4

from multi_agentic_graph_rag.config.settings import load_settings
from multi_agentic_graph_rag.domain.vectors import VectorRecord
from multi_agentic_graph_rag.infrastructure.chroma.client import create_persistent_chroma_client
from multi_agentic_graph_rag.infrastructure.chroma.vector_store import ChromaVectorStore

_KEYWORDS = (
    "session",
    "sessions",
    "timeout",
    "expire",
    "expires",
    "inactivity",
    "authenticated",
    "authentication",
    "password",
    "account",
    "security",
    "backup",
)


class DeterministicSmokeEmbedder:
    """Small deterministic embedding adapter for Phase 6 Chroma smoke testing.

    This intentionally avoids sentence-transformers because local Hugging Face
    execution is currently blocked by Windows Application Control on SciPy .pyd files.
    """

    def fingerprint(self) -> str:
        return "emb-phase6-smoke-deterministic-v1"

    def embed_documents(self, texts: tuple[str, ...]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        tokens = _tokenize(text)
        vector = [float(tokens.count(keyword)) for keyword in _KEYWORDS]
        return _l2_normalize(vector)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))

    if norm == 0:
        return vector

    return [value / norm for value in vector]


def main() -> None:
    settings = load_settings()

    client = create_persistent_chroma_client(settings.paths.chroma_persist_dir)
    store = ChromaVectorStore(
        client=client,
        collection_name=f"{settings.chroma.collection_name}_phase6_smoke",
    )

    embedder = DeterministicSmokeEmbedder()

    suffix = uuid4().hex[:10]
    document_version_id = str(uuid4())

    target_text = "Authenticated sessions expire after 30 minutes of inactivity."
    distractor_text = "Daily backup jobs run at midnight for account recovery."

    target_record = VectorRecord(
        chunk_id=f"CHUNK-PHASE6-SMOKE-TARGET-{suffix}",
        document_version_id=document_version_id,
        normalized_text=target_text,
        content_hash=("a" * 64),
        ordinal=1,
        page_start=1,
        page_end=1,
        section_path=("Security", "Session Management"),
        embedding_fingerprint=embedder.fingerprint(),
    )

    distractor_record = VectorRecord(
        chunk_id=f"CHUNK-PHASE6-SMOKE-DISTRACTOR-{suffix}",
        document_version_id=document_version_id,
        normalized_text=distractor_text,
        content_hash=("b" * 64),
        ordinal=2,
        page_start=2,
        page_end=2,
        section_path=("Operations", "Backup"),
        embedding_fingerprint=embedder.fingerprint(),
    )

    records = (target_record, distractor_record)
    embeddings = embedder.embed_documents(tuple(record.normalized_text for record in records))

    indexed_count = store.upsert_chunks(records=records, embeddings=embeddings)

    query_embedding = embedder.embed_query("session timeout after inactivity")
    results = store.search_chunks(
        query_embedding=query_embedding,
        n_results=2,
        document_version_id=document_version_id,
    )

    if indexed_count != 2:
        raise RuntimeError(f"Expected 2 indexed chunks, got {indexed_count}.")

    if not results:
        raise RuntimeError("Expected at least one vector search result.")

    if results[0].chunk_id != target_record.chunk_id:
        raise RuntimeError(
            f"Expected top result {target_record.chunk_id}, got {results[0].chunk_id}."
        )

    print("PASS Phase 6 Chroma vector smoke test succeeded.")
    print(f"Collection: {settings.chroma.collection_name}_phase6_smoke")
    print(f"Document version: {document_version_id}")
    print(f"Indexed chunks: {indexed_count}")
    print(f"Top result: {results[0].chunk_id}")
    print(f"Top result distance: {results[0].distance}")


if __name__ == "__main__":
    main()
