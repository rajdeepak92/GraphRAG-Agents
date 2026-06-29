# Tracker

| Timestamp | Phase | Status | Notes |
| --- | --- | --- | --- |
| 2026-06-29 | Phase 0 | Complete | Replaced placeholder source/docs/scripts/migrations with simplified ingestion-first layout. |
| 2026-06-29 | Phase 1 | Complete | Added config loader, provider sections, cache policy, `version`, `config-check`, and `doctor`. |
| 2026-06-29 | Phase 2 | Complete | Added strict schemas and Python-owned `RUN-*`, `CHUNK-*`, `FACT-*`, and `REQ-*` IDs. |
| 2026-06-29 | Phase 3 | Complete | Added text/Markdown/DOCX/PDF parsing, chunking, checksum, fingerprints, and manifest writing. |
| 2026-06-29 | Phase 4 | Complete | Added PostgreSQL, Neo4j, and Chroma adapters with real modes plus local trace smoke modes. |
| 2026-06-29 | Phase 5 | Complete | Added LangGraph-backed `IngestionDocumentAgent` orchestration and structured node logging. |
| 2026-06-29 | Phase 6 | Complete | Added bounded requirement discovery with strict source quote verification. |
| 2026-06-29 | Phase 7 | Complete | Added permanent fact/requirement builder, artifact JSON writing, and store projection. |
| 2026-06-29 | Phase 8 | Partial | Same checksum/version is idempotent by stable IDs and upserts; `run status`, `run resume`, and `artifact verify` exist. Full checkpoint resume is deferred. |
| 2026-06-29 | Phase 9 | Complete | README publishes the future user-story contract without implementing a user-story agent. |
