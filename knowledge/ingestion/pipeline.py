"""
ingestion/pipeline.py — Document Ingestion Orchestrator
========================================================

WHY THIS EXISTS:
    This is the central pipeline that transforms raw financial data into
    searchable vector embeddings. It orchestrates the full lifecycle:

    Raw Text → Clean → Chunk → Embed → Store

    Without a unified pipeline, each tool/loader would need to independently
    handle cleaning, chunking, and embedding — leading to inconsistent
    processing and duplicated logic.

PIPELINE STAGES:
    1. LOAD — Convert source data into a Document (via loaders.py).
    2. CLEAN — Remove noise, normalize text (via cleaners.py).
    3. CHUNK — Split into embedding-sized segments (via chunker.py).
    4. EMBED — Generate vector representations (via embeddings.py).
    5. STORE — Persist in vector database (via vector_store.py).

HOW IT CONNECTS:
    - agent/nodes.py calls ingest_tool_output() after every tool execution.
    - retrieval/retriever.py queries the vector store populated by this pipeline.
    - main.py initializes the pipeline at startup.

SCALABILITY:
    - Async ingestion (future): tool outputs are ingested in background.
    - Deduplication (future): skip re-ingesting identical content.
    - Batch processing: ingest_documents() handles multiple docs efficiently.
"""

from __future__ import annotations

from typing import Any, Optional

from knowledge.ingestion.chunker import TextChunker
from knowledge.ingestion.cleaners import clean_text
from knowledge.ingestion.loaders import (
    load_earnings_transcript,
    load_news_article,
    load_pdf_document,
    load_research_note,
    load_tool_output,
)
from knowledge.retrieval.embeddings import EmbeddingPipeline
from knowledge.retrieval.schemas import Chunk, Document, DocumentType
from knowledge.retrieval.vector_store import VectorStoreBase
from utils.logger import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """
    Orchestrates the full document ingestion workflow.

    Stateless processor — receives dependencies via constructor injection.
    This makes testing easy: inject mock vector store + mock embedder.
    """

    def __init__(
        self,
        vector_store: VectorStoreBase,
        embedding_pipeline: EmbeddingPipeline,
        chunker: Optional[TextChunker] = None,
    ) -> None:
        """
        Args:
            vector_store: Where to store embedded chunks.
            embedding_pipeline: How to generate embeddings.
            chunker: Text chunking strategy (default config if None).
        """
        self._vector_store = vector_store
        self._embedder = embedding_pipeline
        self._chunker = chunker or TextChunker()

    def ingest_document(self, document: Document) -> int:
        """
        Ingest a single document through the full pipeline.

        Args:
            document: The document to process.

        Returns:
            Number of chunks stored.
        """
        logger.info(
            f"Ingesting document: {document.id} "
            f"(type={document.source_type}, ticker={document.ticker})"
        )

        # 1. Clean the text
        cleaned_content = clean_text(document.content)
        if not cleaned_content:
            logger.warning(f"Document {document.id} is empty after cleaning — skipping")
            return 0

        # Create a cleaned copy of the document
        cleaned_doc = document.model_copy(update={"content": cleaned_content})

        # 2. Chunk the document
        chunks = self._chunker.chunk_document(cleaned_doc)
        if not chunks:
            logger.warning(f"Document {document.id} produced no chunks — skipping")
            return 0

        # 3. Embed the chunks
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self._embedder.embed_batch(chunk_texts)

        if len(embeddings) != len(chunks):
            logger.error(
                f"Embedding count mismatch: {len(embeddings)} embeddings "
                f"for {len(chunks)} chunks"
            )
            return 0

        # 4. Store in vector database
        ids = [chunk.id for chunk in chunks]
        documents_text = [chunk.content for chunk in chunks]
        metadatas = [chunk.to_vector_metadata() for chunk in chunks]

        try:
            self._vector_store.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents_text,
                metadatas=metadatas,
            )
        except Exception as e:
            logger.error(f"Vector store insertion failed: {e}")
            return 0

        logger.info(
            f"Ingested {len(chunks)} chunks from document {document.id} "
            f"(total in store: {self._vector_store.count()})"
        )
        return len(chunks)

    def ingest_documents(self, documents: list[Document]) -> int:
        """
        Batch ingest multiple documents.

        Args:
            documents: List of documents to process.

        Returns:
            Total number of chunks stored across all documents.
        """
        total_chunks = 0
        for doc in documents:
            try:
                count = self.ingest_document(doc)
                total_chunks += count
            except Exception as e:
                logger.error(
                    f"Failed to ingest document {doc.id}: "
                    f"{type(e).__name__}: {e}"
                )
                continue

        logger.info(
            f"Batch ingestion complete: {total_chunks} chunks "
            f"from {len(documents)} documents"
        )
        return total_chunks

    def ingest_tool_output(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str,
        ticker: str = "",
    ) -> int:
        """
        Convenience method to ingest a Phase 1 tool output.

        Called by agent/nodes.py after every successful tool execution.
        This is how the agent builds long-term memory automatically.

        Args:
            tool_name: Name of the tool.
            tool_input: Arguments passed to the tool.
            tool_output: Raw output string.
            ticker: Stock ticker (auto-extracted if not provided).

        Returns:
            Number of chunks stored.
        """
        if not tool_output or len(tool_output.strip()) < 20:
            logger.debug(f"Tool output too short to ingest: {tool_name}")
            return 0

        document = load_tool_output(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            ticker=ticker,
        )

        return self.ingest_document(document)

    def ingest_research_note(
        self,
        content: str,
        ticker: str = "",
        title: str = "",
    ) -> int:
        """
        Store an agent-generated analysis for future retrieval.

        Called at the end of a successful agent run to persist the
        agent's synthesized analysis. Future runs can retrieve this
        as context, building institutional knowledge.

        Args:
            content: The research note/analysis text.
            ticker: Related stock ticker.
            title: Note title.

        Returns:
            Number of chunks stored.
        """
        document = load_research_note(
            content=content,
            ticker=ticker,
            title=title,
        )
        return self.ingest_document(document)

    def ingest_pdf(
        self,
        file_path: str,
        ticker: str = "",
        title: str = "",
        source_type: DocumentType = DocumentType.SEC_FILING,
        source_name: str = "sec_filing_pdf",
    ) -> int:
        """
        Ingest a PDF's embedded text layer (10-K/10-Q, investor deck).

        Phase 7.2. Raises ValueError (propagated, not swallowed) if the file
        can't be read or has no extractable text — the caller decides what
        to tell the user, since this may be an interactive CLI flag or an
        API upload.
        """
        document = load_pdf_document(
            file_path=file_path,
            ticker=ticker,
            title=title,
            source_type=source_type,
            source_name=source_name,
        )
        return self.ingest_document(document)

    def ingest_earnings_transcript(
        self,
        content: str,
        ticker: str = "",
        title: str = "",
        fiscal_period: str = "",
    ) -> int:
        """Ingest a plain-text earnings call transcript. Phase 7.3."""
        document = load_earnings_transcript(
            content=content,
            ticker=ticker,
            title=title,
            fiscal_period=fiscal_period,
        )
        return self.ingest_document(document)
