"""
Main RAG Pipeline - Orchestrates the entire retrieval-augmented generation process.
"""

import time
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from ..core.config import settings
from ..core.logging import logger
from .chunker import BaseChunker, get_chunker
from .document_processor import DocumentProcessor, ProcessedDocument
from .generator import BaseGenerator, GenerationResult, get_generator
from .retriever import BaseRetriever, VectorRetriever, HybridRetriever, EnhancedQueryResult


class RAGConfig(BaseModel):
    """Configuration for the RAG pipeline."""

    # Chunking
    chunking_strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    retrieval_strategy: str = "vector"  # vector, hybrid
    max_retrieved_docs: int = 5
    similarity_threshold: float = 0.7

    # Generation
    generation_provider: str = "openai"
    generation_model: str = "gpt-4-1106-preview"
    max_tokens: int = 1000
    temperature: float = 0.7

    # Vector Database
    vector_db_type: str = "qdrant"
    embedding_model: str = "text-embedding-ada-002"


class RAGResult(BaseModel):
    """Result of a complete RAG operation."""

    query: str
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_result: EnhancedQueryResult
    generation_result: GenerationResult
    total_time: float
    metadata: Dict[str, Any] = {}


class RAGPipeline:
    """Main RAG pipeline that orchestrates all components."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()

        # Initialize components
        self.document_processor = DocumentProcessor()
        self.chunker = get_chunker(
            strategy=self.config.chunking_strategy,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        # Initialize retriever
        self.retriever = self._initialize_retriever()

        # Initialize generator
        self.generator = get_generator(
            provider=self.config.generation_provider,
            model=self.config.generation_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )

        logger.info("RAG Pipeline initialized successfully")

    def _initialize_retriever(self) -> Optional[BaseRetriever]:
        """Initialize the retriever based on configuration."""
        try:
            vector_retriever = VectorRetriever(
                embedding_model=self.config.embedding_model,
                vector_db_type=self.config.vector_db_type
            )

            if self.config.retrieval_strategy == "hybrid":
                return HybridRetriever(vector_retriever)
            else:
                return vector_retriever
        
        except Exception as e:
            logger.warning(f"Failed to initialize retriever, falling back to pure LLM mode: {e}")
            return None

    async def add_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a document to the RAG system."""
        try:
            logger.info(f"Adding document: {file_path}")

            # Process document
            document = await self.document_processor.process_file(file_path, metadata)

            # Chunk document
            chunks = self.chunker.chunk_document(document)

            # Add chunks to retriever
            success = await self.retriever.add_chunks(chunks)

            if success:
                logger.info(
                    f"Successfully added document {document.id} with {len(chunks)} chunks"
                )
            else:
                logger.error(f"Failed to add document chunks to retriever")

            return success

        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
            return False

    async def add_text(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add raw text content to the RAG system."""
        try:
            logger.info("Adding text content")

            # Process text
            document = await self.document_processor.process_text(content, metadata)

            # Chunk document
            chunks = self.chunker.chunk_document(document)

            # Add chunks to retriever
            success = await self.retriever.add_chunks(chunks)

            if success:
                logger.info(f"Successfully added text with {len(chunks)} chunks")

            return success

        except Exception as e:
            logger.error(f"Error adding text content: {e}")
            return False

    async def add_url(
        self,
        url: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add content from a URL to the RAG system."""
        try:
            logger.info(f"Adding content from URL: {url}")

            # Process URL
            document = await self.document_processor.process_url(url, metadata)

            # Chunk document
            chunks = self.chunker.chunk_document(document)

            # Add chunks to retriever
            success = await self.retriever.add_chunks(chunks)

            if success:
                logger.info(f"Successfully added URL content with {len(chunks)} chunks")

            return success

        except Exception as e:
            logger.error(f"Error adding URL content {url}: {e}")
            return False

    async def query(
        self,
        query: str,
        max_docs: Optional[int] = None,
        **kwargs
    ) -> RAGResult:
        """Perform a complete RAG query."""
        start_time = time.time()

        try:
            logger.info(f"Processing query: {query}")

            max_docs = max_docs or self.config.max_retrieved_docs

            # Retrieve relevant documents
            retrieval_start = time.time()
            retrieval_result = await self.retriever.search(
                query,
                limit=max_docs,
                score_threshold=self.config.similarity_threshold,
                **kwargs
            )
            retrieval_time = time.time() - retrieval_start

            logger.info(f"Retrieved {len(retrieval_result.results)} documents in {retrieval_time:.2f}s")

            # Generate response
            generation_start = time.time()
            generation_result = await self.generator.generate(
                query,
                retrieval_context=retrieval_result,
                **kwargs
            )
            generation_time = time.time() - generation_start

            logger.info(f"Generated response in {generation_time:.2f}s")

            # Prepare sources
            sources = [
                {
                    "content": result.chunk.content,
                    "document_id": result.chunk.document_id,
                    "chunk_index": result.chunk.chunk_index,
                    "score": result.score,
                    "metadata": result.chunk.metadata
                }
                for result in retrieval_result.results
            ]

            total_time = time.time() - start_time

            return RAGResult(
                query=query,
                answer=generation_result.generated_text,
                sources=sources,
                retrieval_result=retrieval_result,
                generation_result=generation_result,
                total_time=total_time,
                metadata={
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "config": self.config.dict(),
                    "num_sources": len(sources)
                }
            )

        except Exception as e:
            logger.error(f"RAG query failed: {e}")

            # Return error result
            return RAGResult(
                query=query,
                answer=f"Sorry, I encountered an error while processing your query: {str(e)}",
                sources=[],
                retrieval_result=EnhancedQueryResult(query=query, results=[], total_time=0.0),
                generation_result=GenerationResult(
                    generated_text="",
                    model=self.config.generation_model,
                    metadata={"error": str(e)}
                ),
                total_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    async def query_stream(
        self,
        query: str,
        max_docs: Optional[int] = None,
        **kwargs
    ):
        """Perform a streaming RAG query."""
        import json
        
        logger.info(f"Processing stream query: {query}")

        max_docs = max_docs or self.config.max_retrieved_docs
        retrieval_result = None

        # 1. Retrieve relevant documents (Only if retriever is available)
        if self.retriever:
            try:
                retrieval_result = await self.retriever.search(
                    query,
                    limit=max_docs,
                    score_threshold=self.config.similarity_threshold,
                    **kwargs
                )
            except Exception as e:
                logger.error(f"Retrieval failed during stream: {e}")
                retrieval_result = None
        else:
            logger.info("Retriever not available, skipping retrieval step.")

        # 2. Check retrieval results and fallback
        if not retrieval_result or not retrieval_result.results:
            if self.retriever:
                logger.warning("No relevant documents found. Proceeding without context.")
            retrieval_result = None
        else:
            # 3. Yield metadata block
            top_sources = [
                {
                    "source": res.chunk.document_id,
                    "score": round(res.score, 4),
                    "content_preview": res.chunk.content[:100] + "..." if res.chunk.content else ""
                }
                for res in retrieval_result.results[:3]
            ]

            metadata_packet = {
                "type": "metadata",
                "count": len(retrieval_result.results),
                "sources": top_sources
            }
            yield json.dumps(metadata_packet)

        # 4. Generate streaming response
        async for chunk in self.generator.generate_stream(
            query,
            retrieval_context=retrieval_result,
            **kwargs
        ):
            yield chunk

    async def batch_query(
        self,
        queries: List[str],
        max_docs: Optional[int] = None,
        **kwargs
    ) -> List[RAGResult]:
        """Process multiple queries in batch."""
        logger.info(f"Processing {len(queries)} queries in batch")

        results = []
        for query in queries:
            result = await self.query(query, max_docs, **kwargs)
            results.append(result)

        logger.info(f"Completed batch processing of {len(queries)} queries")
        return results

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the RAG system."""
        try:
            success = await self.retriever.delete_document(document_id)
            if success:
                logger.info(f"Successfully deleted document {document_id}")
            else:
                logger.warning(f"Failed to delete document {document_id}")
            return success

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the RAG system."""
        try:
            retriever_stats = await self.retriever.get_stats()

            return {
                "pipeline_config": self.config.dict(),
                "retriever_stats": retriever_stats,
                "components": {
                    "document_processor": self.document_processor.__class__.__name__,
                    "chunker": self.chunker.__class__.__name__,
                    "retriever": self.retriever.__class__.__name__,
                    "generator": self.generator.__class__.__name__,
                }
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}

    def update_config(self, **kwargs) -> None:
        """Update pipeline configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config key: {key}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on all components."""
        health_status = {
            "status": "healthy",
            "components": {},
            "timestamp": time.time()
        }

        try:
            # Test retriever
            test_query = "health check test query"
            retrieval_result = await self.retriever.search(test_query, limit=1)
            health_status["components"]["retriever"] = {
                "status": "healthy",
                "search_time": retrieval_result.total_time
            }

        except Exception as e:
            health_status["components"]["retriever"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"

        try:
            # Test generator
            generation_result = await self.generator.generate("test prompt")
            health_status["components"]["generator"] = {
                "status": "healthy",
                "generation_time": generation_result.generation_time
            }

        except Exception as e:
            health_status["components"]["generator"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"

        return health_status


# Global pipeline instance for easy access
_pipeline_instance = None


def get_rag_pipeline(config: Optional[RAGConfig] = None) -> RAGPipeline:
    """Get or create a global RAG pipeline instance."""
    global _pipeline_instance

    if _pipeline_instance is None or config is not None:
        _pipeline_instance = RAGPipeline(config)

    return _pipeline_instance


async def initialize_default_pipeline() -> RAGPipeline:
    """Initialize a default RAG pipeline with settings from config."""
    config = RAGConfig(
        chunking_strategy="recursive",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_retrieved_docs=settings.max_retrieved_docs,
        similarity_threshold=settings.similarity_threshold,
        generation_provider=settings.llm_provider,
        generation_model=settings.default_llm_model,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        vector_db_type=settings.vector_db_type,
        embedding_model=settings.embedding_model
    )

    pipeline = RAGPipeline(config)

    # Perform health check
    health = await pipeline.health_check()
    if health["status"] != "healthy":
        logger.warning(f"Pipeline health check failed: {health}")
    else:
        logger.info("RAG pipeline initialized and healthy")

    return pipeline