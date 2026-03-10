"""
Document retrieval components for RAG pipeline.
Enhanced with multi-modal support, query expansion, and reranking capabilities.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

from ..core.config import settings
from ..core.logging import logger
from .document_processor import DocumentChunk
from .embedder import get_embedder, MultiModalEmbedder



class RetrievalResult(BaseModel):
    """Result of a retrieval operation."""

    chunk: DocumentChunk
    score: float
    metadata: Dict[str, Any] = {}
    rerank_score: Optional[float] = None
    retrieval_type: str = "vector"  # vector, keyword, hybrid, multimodal


class EnhancedQueryResult(BaseModel):
    """Enhanced result of a query operation with multi-modal support."""

    query: str
    results: List[RetrievalResult]
    total_time: float
    metadata: Dict[str, Any] = {}
    expanded_queries: List[str] = []
    reranked: bool = False


class QueryExpander:
    """Query expansion using synonyms and related terms."""

    def __init__(self, use_llm: bool = False, llm_model: Optional[str] = None):
        self.use_llm = use_llm
        self.llm_model = llm_model
        self._synonym_dict = None

    async def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """Expand query with related terms."""
        expansions = [query]  # Always include original

        if self.use_llm and self.llm_model:
            llm_expansions = await self._llm_expand_query(query, max_expansions)
            expansions.extend(llm_expansions)
        else:
            # Simple keyword expansion
            keyword_expansions = self._keyword_expand_query(query, max_expansions)
            expansions.extend(keyword_expansions)

        return list(set(expansions))[:max_expansions + 1]

    async def _llm_expand_query(self, query: str, max_expansions: int) -> List[str]:
        """Expand query using LLM."""
        try:
            if "openai" in self.llm_model.lower():
                from openai import AsyncOpenAI

                # 构建OpenAI客户端配置
                client_kwargs = {"api_key": settings.openai_api_key}
                if settings.openai_base_url:
                    client_kwargs["base_url"] = settings.openai_base_url
                client = AsyncOpenAI(**client_kwargs)

                prompt = f"""Generate {max_expansions} semantically related queries for: "{query}"
                Rules:
                1. Keep the same intent and meaning
                2. Use synonyms and alternative phrasings
                3. Return only the queries, one per line
                4. No explanations or numbering"""

                response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.7
                )

                expansions = response.choices[0].message.content.strip().split('\n')
                return [exp.strip() for exp in expansions if exp.strip()][:max_expansions]

        except Exception as e:
            logger.warning(f"LLM query expansion failed: {e}")
            return []

        return []

    def _keyword_expand_query(self, query: str, max_expansions: int) -> List[str]:
        """Simple keyword-based expansion."""
        # Basic synonym mapping - in production, use a proper thesaurus
        simple_synonyms = {
            'document': ['file', 'paper', 'text'],
            'search': ['find', 'locate', 'retrieve'],
            'data': ['information', 'content', 'details'],
            'process': ['handle', 'execute', 'run'],
            'create': ['generate', 'build', 'make'],
            'analyze': ['examine', 'study', 'review'],
        }

        words = query.lower().split()
        expansions = []

        for word in words:
            if word in simple_synonyms:
                synonyms = simple_synonyms[word]
                for synonym in synonyms[:max_expansions]:
                    expanded = query.lower().replace(word, synonym)
                    if expanded != query.lower():
                        expansions.append(expanded)

        return expansions[:max_expansions]


class Reranker:
    """Rerank retrieval results using various strategies."""

    def __init__(
        self,
        strategy: str = "cross_encoder",
        model_name: Optional[str] = None
    ):
        self.strategy = strategy
        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self._model = None

    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Rerank results based on strategy."""
        if not results:
            return results

        if self.strategy == "cross_encoder":
            return await self._cross_encoder_rerank(query, results, top_k)
        elif self.strategy == "llm_rerank":
            return await self._llm_rerank(query, results, top_k)
        elif self.strategy == "combined_score":
            return self._combined_score_rerank(query, results, top_k)
        else:
            logger.warning(f"Unknown rerank strategy: {self.strategy}")
            return results

    async def _cross_encoder_rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int]
    ) -> List[RetrievalResult]:
        """Rerank using cross-encoder model."""
        try:
            if not self._model:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                logger.info(f"Loaded cross-encoder model: {self.model_name}")

            # Prepare pairs for cross-encoder
            pairs = [(query, result.chunk.content) for result in results]

            # Get relevance scores
            scores = self._model.predict(pairs)

            # Update results with rerank scores
            reranked_results = []
            for result, score in zip(results, scores):
                result.rerank_score = float(score)
                reranked_results.append(result)

            # Sort by rerank score
            reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)

            return reranked_results[:top_k] if top_k else reranked_results

        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed: {e}")
            return results

    async def _llm_rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int]
    ) -> List[RetrievalResult]:
        """Rerank using LLM scoring."""
        try:
            from openai import AsyncOpenAI

            # 构建OpenAI客户端配置
            client_kwargs = {"api_key": settings.openai_api_key}
            if settings.openai_base_url:
                client_kwargs["base_url"] = settings.openai_base_url
            client = AsyncOpenAI(**client_kwargs)

            # Score each result
            scored_results = []
            for i, result in enumerate(results):
                prompt = f"""Rate the relevance of this text to the query on a scale of 0-10.

Query: {query}

Text: {result.chunk.content[:500]}...

Provide only a number between 0-10 where:
- 10 = extremely relevant
- 5 = somewhat relevant
- 0 = not relevant

Score:"""

                response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.1
                )

                try:
                    score = float(response.choices[0].message.content.strip())
                    result.rerank_score = score / 10.0  # Normalize to 0-1
                except:
                    result.rerank_score = result.score  # Fallback to original score

                scored_results.append(result)

            # Sort by rerank score
            scored_results.sort(key=lambda x: x.rerank_score, reverse=True)
            return scored_results[:top_k] if top_k else scored_results

        except Exception as e:
            logger.warning(f"LLM reranking failed: {e}")
            return results

    def _combined_score_rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int]
    ) -> List[RetrievalResult]:
        """Rerank using combined scoring factors."""
        for result in results:
            # Combine original score with other factors
            content_length_score = min(len(result.chunk.content) / 1000, 1.0)

            # Check for query terms in content
            query_terms = set(query.lower().split())
            content_terms = set(result.chunk.content.lower().split())
            term_overlap = len(query_terms & content_terms) / len(query_terms)

            # Combined score
            result.rerank_score = (
                0.6 * result.score +  # Original similarity
                0.2 * content_length_score +  # Content length factor
                0.2 * term_overlap  # Query term overlap
            )

        # Sort by combined score
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results[:top_k] if top_k else results


class BaseRetriever(ABC):
    """Base class for document retrievers."""

    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to the retriever."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 5,
        **kwargs
    ) -> EnhancedQueryResult:
        """Search for relevant chunks."""
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        pass


class EnhancedMultiModalRetriever(BaseRetriever):
    """Enhanced multi-modal retrieval with query expansion and reranking."""

    def __init__(
        self,
        embedder: Optional[MultiModalEmbedder] = None,
        vector_db_type: str = "qdrant",
        enable_query_expansion: bool = True,
        enable_reranking: bool = True,
        query_expander: Optional[QueryExpander] = None,
        reranker: Optional[Reranker] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vector_db_type = vector_db_type
        self.enable_query_expansion = enable_query_expansion
        self.enable_reranking = enable_reranking

        # Initialize embedder
        self.embedder = embedder or get_embedder(
            embedder_type="multimodal",
            use_unified_space=True  # Use CLIP for both text and images
        )

        # Initialize query expansion and reranking
        self.query_expander = query_expander or QueryExpander()
        self.reranker = reranker or Reranker()

        # Vector database components
        self._vector_db = None
        self._initialize_vector_db()

        logger.info(
            f"Initialized Enhanced Multi-Modal Retriever - "
            f"Vector DB: {vector_db_type}, "
            f"Query Expansion: {enable_query_expansion}, "
            f"Reranking: {enable_reranking}"
        )

    def _initialize_vector_db(self):
        """Initialize the vector database with multi-modal support."""
        if self.vector_db_type == "qdrant":
            self._initialize_qdrant_multimodal()
        elif self.vector_db_type == "chromadb":
            self._initialize_chromadb()
        else:
            raise ValueError(f"Unsupported vector database: {self.vector_db_type}")

    def _initialize_qdrant_multimodal(self):
        """Initialize Qdrant with multi-modal vector configuration."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct

            self._vector_db = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port
            )

            # Check if collection exists
            collections = self._vector_db.get_collections().collections
            collection_names = [c.name for c in collections]

            if settings.qdrant_collection not in collection_names:
                # Create collection with unified vector space
                if self.embedder.is_unified_space():
                    # Single vector space for both text and images
                    vector_config = VectorParams(
                        size=self.embedder.get_text_dimension(),
                        distance=Distance.COSINE
                    )
                else:
                    # Multiple vector spaces
                    vector_config = {
                        "text": VectorParams(
                            size=self.embedder.get_text_dimension(),
                            distance=Distance.COSINE
                        ),
                        "image": VectorParams(
                            size=self.embedder.get_image_dimension(),
                            distance=Distance.COSINE
                        )
                    }

                self._vector_db.create_collection(
                    collection_name=settings.qdrant_collection,
                    vectors_config=vector_config
                )
                logger.info(f"Created multi-modal Qdrant collection: {settings.qdrant_collection}")

            logger.info("Initialized Qdrant multi-modal vector database")

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise

    def _initialize_chromadb(self):
        """Initialize ChromaDB."""
        try:
            import chromadb

            self._vector_db = chromadb.Client()
            self._collection = self._vector_db.get_or_create_collection(
                name=settings.qdrant_collection  # Reuse setting
            )

            logger.info("Initialized ChromaDB vector database")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to the vector database with multi-modal embeddings."""
        if not chunks:
            return True

        try:
            # Separate chunks by type
            text_chunks = [c for c in chunks if c.chunk_type in ["text", "table"]]
            image_chunks = [c for c in chunks if c.chunk_type == "image_description"]

            # Generate embeddings
            all_embeddings = []
            all_chunk_data = []

            # Text embeddings
            if text_chunks:
                text_content = [chunk.content for chunk in text_chunks]
                text_result = await self.embedder.embed_texts(text_content)

                for i, chunk in enumerate(text_chunks):
                    all_embeddings.append(text_result.embeddings[i])
                    all_chunk_data.append({
                        'chunk': chunk,
                        'vector_type': 'text',
                        'embedding_type': text_result.embedding_type
                    })

            # Image embeddings (for image descriptions using text embedder)
            if image_chunks:
                image_content = [chunk.content for chunk in image_chunks]
                image_result = await self.embedder.embed_texts(image_content)

                for i, chunk in enumerate(image_chunks):
                    all_embeddings.append(image_result.embeddings[i])
                    all_chunk_data.append({
                        'chunk': chunk,
                        'vector_type': 'image',
                        'embedding_type': image_result.embedding_type
                    })

            # Store in vector database
            if self.vector_db_type == "qdrant":
                await self._add_to_qdrant_multimodal(all_chunk_data, all_embeddings)
            elif self.vector_db_type == "chromadb":
                await self._add_to_chromadb_multimodal(all_chunk_data, all_embeddings)

            logger.info(f"Added {len(chunks)} multi-modal chunks to vector database")
            return True

        except Exception as e:
            logger.error(f"Failed to add chunks to vector database: {e}")
            return False

    async def _add_to_qdrant_multimodal(
        self,
        chunk_data: List[Dict],
        embeddings: List[List[float]]
    ):
        """Add chunks to Qdrant with multi-modal support."""
        from qdrant_client.models import PointStruct

        points = []
        for data, embedding in zip(chunk_data, embeddings):
            chunk = data['chunk']

            # Prepare payload
            payload = {
                "document_id": chunk.document_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type,
                "source_element": chunk.source_element,
                "metadata": chunk.metadata,
                "vector_type": data['vector_type'],
                "embedding_type": data['embedding_type']
            }

            if self.embedder.is_unified_space():
                # Single vector space
                point = PointStruct(
                    id=chunk.id,
                    vector=embedding,
                    payload=payload
                )
            else:
                # Multiple vector spaces
                vector_name = data['vector_type']
                point = PointStruct(
                    id=chunk.id,
                    vector={vector_name: embedding},
                    payload=payload
                )

            points.append(point)

        self._vector_db.upsert(
            collection_name=settings.qdrant_collection,
            points=points
        )

    async def _add_to_chromadb_multimodal(
        self,
        chunk_data: List[Dict],
        embeddings: List[List[float]]
    ):
        """Add chunks to ChromaDB with multi-modal support."""
        ids = [data['chunk'].id for data in chunk_data]
        documents = [data['chunk'].content for data in chunk_data]

        metadatas = []
        for data in chunk_data:
            chunk = data['chunk']
            metadata = {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type,
                "vector_type": data['vector_type'],
                "embedding_type": data['embedding_type'],
                **chunk.metadata
            }
            metadatas.append(metadata)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    async def search(
        self,
        query: str,
        limit: int = 5,
        **kwargs
    ) -> EnhancedQueryResult:
        """Enhanced search with query expansion and reranking."""
        start_time = time.time()

        try:
            # Step 1: Query expansion
            expanded_queries = [query]
            if self.enable_query_expansion:
                expanded_queries = await self.query_expander.expand_query(query)
                logger.debug(f"Expanded query to {len(expanded_queries)} variants")

            # Step 2: Multi-query retrieval
            all_results = []
            for expanded_query in expanded_queries:
                query_results = await self._single_query_search(
                    expanded_query, limit * 2, **kwargs  # Get more results for reranking
                )
                all_results.extend(query_results)

            # Step 3: Deduplicate and merge results
            unique_results = self._deduplicate_results(all_results)

            # Step 4: Reranking
            if self.enable_reranking and unique_results:
                unique_results = await self.reranker.rerank(
                    query, unique_results, top_k=limit
                )
                reranked = True
            else:
                # Sort by original score and limit
                unique_results.sort(key=lambda x: x.score, reverse=True)
                unique_results = unique_results[:limit]
                reranked = False

            total_time = time.time() - start_time

            return EnhancedQueryResult(
                query=query,
                results=unique_results,
                total_time=total_time,
                expanded_queries=expanded_queries,
                reranked=reranked,
                metadata={
                    "vector_db_type": self.vector_db_type,
                    "embedder_unified_space": self.embedder.is_unified_space(),
                    "query_expansion_enabled": self.enable_query_expansion,
                    "reranking_enabled": self.enable_reranking,
                    "total_raw_results": len(all_results),
                    "unique_results": len(unique_results)
                }
            )

        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            return EnhancedQueryResult(
                query=query,
                results=[],
                total_time=time.time() - start_time,
                expanded_queries=[query],
                metadata={"error": str(e)}
            )


    async def _single_query_search(
        self,
        query: str,
        limit: int,
        **kwargs
    ) -> List[RetrievalResult]:
        """Perform single query search in vector database."""
        try:
            # Generate query embedding
            query_embedding = await self.embedder.embed_query(query)

            # Search in vector database
            if self.vector_db_type == "qdrant":
                results = await self._search_qdrant_multimodal(query_embedding, limit, **kwargs)
            elif self.vector_db_type == "chromadb":
                results = await self._search_chromadb_multimodal(query_embedding, limit, **kwargs)
            else:
                results = []

            return results

        except Exception as e:
            logger.error(f"Single query search failed: {e}")
            return []

    async def _search_qdrant_multimodal(
        self,
        query_embedding: List[float],
        limit: int,
        **kwargs
    ) -> List[RetrievalResult]:
        """Search in Qdrant with multi-modal support."""
        if self.embedder.is_unified_space():
            # Single vector space search
            search_result = self._vector_db.search(
                collection_name=settings.qdrant_collection,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=kwargs.get("score_threshold", 0.0)
            )
        else:
            # Multi-vector search (search in text vectors primarily)
            search_result = self._vector_db.search(
                collection_name=settings.qdrant_collection,
                query_vector=("text", query_embedding),
                limit=limit,
                score_threshold=kwargs.get("score_threshold", 0.0)
            )

        results = []
        for hit in search_result:
            chunk = DocumentChunk(
                id=hit.id,
                document_id=hit.payload["document_id"],
                content=hit.payload["content"],
                chunk_index=hit.payload["chunk_index"],
                chunk_type=hit.payload.get("chunk_type", "text"),
                source_element=hit.payload.get("source_element"),
                metadata=hit.payload.get("metadata", {})
            )

            result = RetrievalResult(
                chunk=chunk,
                score=hit.score,
                retrieval_type="multimodal",
                metadata={
                    "source": "qdrant",
                    "vector_type": hit.payload.get("vector_type", "text"),
                    "embedding_type": hit.payload.get("embedding_type", "unknown")
                }
            )
            results.append(result)

        return results

    async def _search_chromadb_multimodal(
        self,
        query_embedding: List[float],
        limit: int,
        **kwargs
    ) -> List[RetrievalResult]:
        """Search in ChromaDB with multi-modal support."""
        search_result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )

        results = []
        if search_result["ids"]:
            for i in range(len(search_result["ids"][0])):
                chunk = DocumentChunk(
                    id=search_result["ids"][0][i],
                    document_id=search_result["metadatas"][0][i]["document_id"],
                    content=search_result["documents"][0][i],
                    chunk_index=search_result["metadatas"][0][i]["chunk_index"],
                    chunk_type=search_result["metadatas"][0][i].get("chunk_type", "text"),
                    metadata=search_result["metadatas"][0][i]
                )

                # ChromaDB returns distances, convert to similarity scores
                distance = search_result["distances"][0][i]
                score = 1.0 / (1.0 + distance)

                result = RetrievalResult(
                    chunk=chunk,
                    score=score,
                    retrieval_type="multimodal",
                    metadata={
                        "source": "chromadb",
                        "distance": distance,
                        "vector_type": search_result["metadatas"][0][i].get("vector_type", "text"),
                        "embedding_type": search_result["metadatas"][0][i].get("embedding_type", "unknown")
                    }
                )
                results.append(result)

        return results

    def _deduplicate_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Deduplicate results by chunk ID, keeping the highest score."""
        seen_chunks = {}

        for result in results:
            chunk_id = result.chunk.id
            if chunk_id not in seen_chunks or result.score > seen_chunks[chunk_id].score:
                seen_chunks[chunk_id] = result

        return list(seen_chunks.values())

    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        try:
            if self.vector_db_type == "qdrant":
                from qdrant_client.models import Filter, FieldCondition, MatchValue

                self._vector_db.delete(
                    collection_name=settings.qdrant_collection,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=document_id)
                            )
                        ]
                    )
                )

            elif self.vector_db_type == "chromadb":
                logger.warning("ChromaDB document deletion not implemented for multi-modal")
                return False

            logger.info(f"Deleted document {document_id} from vector database")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get enhanced retriever statistics."""
        try:
            if self.vector_db_type == "qdrant":
                info = self._vector_db.get_collection(settings.qdrant_collection)
                stats = {
                    "total_chunks": info.vectors_count,
                    "vector_dimension": info.config.params.vectors.size if self.embedder.is_unified_space() else "multi",
                    "distance_metric": info.config.params.vectors.distance if self.embedder.is_unified_space() else "multi",
                    "unified_space": self.embedder.is_unified_space(),
                    "query_expansion": self.enable_query_expansion,
                    "reranking": self.enable_reranking,
                }

            elif self.vector_db_type == "chromadb":
                stats = {
                    "total_chunks": self._collection.count(),
                    "unified_space": self.embedder.is_unified_space(),
                    "query_expansion": self.enable_query_expansion,
                    "reranking": self.enable_reranking,
                }

            return stats

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}


# Legacy VectorRetriever with enhanced compatibility
class VectorRetriever(BaseRetriever):
    """Legacy vector-based retrieval for backward compatibility."""

    def __init__(
        self,
        embedding_model: str = "text-embedding-ada-002",
        vector_db_type: str = "qdrant",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding_model = embedding_model
        self.vector_db_type = vector_db_type
        self._embedder = None
        self._vector_db = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize legacy embedding and vector database components."""
        self._initialize_embedder()
        self._initialize_vector_db()

    def _initialize_embedder(self):
        """Initialize the embedding model."""
        try:
            if "openai" in self.embedding_model.lower():
                from openai import AsyncOpenAI

                # 构建OpenAI客户端配置
                client_kwargs = {"api_key": settings.openai_api_key}
                if settings.openai_base_url:
                    client_kwargs["base_url"] = settings.openai_base_url
                    logger.info(f"Using custom OpenAI base URL for embeddings: {settings.openai_base_url}")

                self._embedder = AsyncOpenAI(**client_kwargs)
                self._embed_method = self._openai_embed
            else:
                # Use sentence transformers for other models
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedding_model)
                self._embed_method = self._sentence_transformer_embed

            logger.info(f"Initialized legacy embedding model: {self.embedding_model}")

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def _initialize_vector_db(self):
        """Initialize the vector database."""
        if self.vector_db_type == "qdrant":
            self._initialize_qdrant()
        elif self.vector_db_type == "chromadb":
            self._initialize_chromadb()
        elif self.vector_db_type == "faiss":
            self._initialize_faiss()
        else:
            raise ValueError(f"Unsupported vector database: {self.vector_db_type}")

    def _initialize_qdrant(self):
        """Initialize legacy Qdrant vector database."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._vector_db = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port
            )
            
            # Ensure collection exists
            collections = self._vector_db.get_collections().collections
            collection_names = [c.name for c in collections]

            if settings.qdrant_collection not in collection_names:
                self._vector_db.create_collection(
                    collection_name=settings.qdrant_collection,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                logger.info(f"Created legacy Qdrant collection: {settings.qdrant_collection}")

            logger.info("Initialized legacy Qdrant vector database")

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise

    def _initialize_chromadb(self):
        """Initialize ChromaDB vector database."""
        try:
            import chromadb

            self._vector_db = chromadb.Client()
            self._collection = self._vector_db.get_or_create_collection(
                name=settings.qdrant_collection
            )

            logger.info("Initialized ChromaDB vector database")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _initialize_faiss(self):
        """Initialize FAISS vector database."""
        try:
            import faiss

            self._dimension = 384  # OpenAI embedding dimension
            self._vector_db = faiss.IndexFlatIP(self._dimension)
            self._id_to_chunk = {}

            logger.info("Initialized FAISS vector database")

        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            raise

    async def _openai_embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI."""
        try:
            response = await self._embedder.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [item.embedding for item in response.data]

        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise

    async def _sentence_transformer_embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using SentenceTransformers."""
        try:
            embeddings = self._embedder.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()

        except Exception as e:
            logger.error(f"SentenceTransformer embedding failed: {e}")
            raise

    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to the vector database."""
        if not chunks:
            return True

        try:
            # Generate embeddings for chunks
            texts = [chunk.content for chunk in chunks]
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = await self._embed_method(texts)
            logger.info(f"Generated {len(embeddings)} embeddings")

            # Add to vector database
            if self.vector_db_type == "qdrant":
                logger.info("Adding to Qdrant...")
                await self._add_to_qdrant(chunks, embeddings)
            elif self.vector_db_type == "chromadb":
                logger.info("Adding to ChromaDB...")
                await self._add_to_chromadb(chunks, embeddings)
            elif self.vector_db_type == "faiss":
                logger.info(f"Adding to FAISS (dimension={self._dimension})...")
                await self._add_to_faiss(chunks, embeddings)

            logger.info(f"Added {len(chunks)} chunks to legacy vector database")
            return True

        except Exception as e:
            import traceback
            logger.error(f"Failed to add chunks to vector database: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    async def _add_to_qdrant(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Add chunks to Qdrant."""
        try:
            from qdrant_client.models import PointStruct

            logger.info(f"Preparing {len(chunks)} points for Qdrant...")

            # 检查embedding维度
            if embeddings and len(embeddings) > 0:
                embedding_dim = len(embeddings[0])
                logger.info(f"Embedding dimension: {embedding_dim}")

                # 检查是否与collection配置匹配
                try:
                    collection_info = self._vector_db.get_collection(settings.qdrant_collection)
                    expected_dim = collection_info.config.params.vectors.size
                    if embedding_dim != expected_dim:
                        raise ValueError(f"Embedding dimension mismatch: got {embedding_dim}, collection expects {expected_dim}")
                except Exception as e:
                    logger.warning(f"Could not verify collection dimension: {e}")

            points = []
            for chunk, embedding in zip(chunks, embeddings):
                point = PointStruct(
                    id=chunk.id,
                    vector=embedding,
                    payload={
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                        "metadata": chunk.metadata
                    }
                )
                points.append(point)

            logger.info(f"Upserting {len(points)} points to Qdrant collection '{settings.qdrant_collection}'...")
            result = self._vector_db.upsert(
                collection_name=settings.qdrant_collection,
                points=points
            )
            logger.info(f"Successfully added {len(points)} points to Qdrant. Operation result: {result}")

        except Exception as e:
            import traceback
            logger.error(f"Qdrant upsert operation failed: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _add_to_chromadb(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Add chunks to ChromaDB."""
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.content for chunk in chunks],
            metadatas=[{
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata
            } for chunk in chunks]
        )

    async def _add_to_faiss(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Add chunks to FAISS."""
        try:
            import numpy as np

            logger.info(f"Converting {len(embeddings)} embeddings to numpy array...")
            embeddings_array = np.array(embeddings, dtype=np.float32)
            logger.info(f"Embedding array shape: {embeddings_array.shape}, expected dimension: {self._dimension}")

            # 检查维度是否匹配
            if embeddings_array.shape[1] != self._dimension:
                raise ValueError(f"Embedding dimension mismatch: got {embeddings_array.shape[1]}, expected {self._dimension}")

            start_id = self._vector_db.ntotal
            logger.info(f"Adding embeddings to FAISS index (current size: {start_id})...")
            self._vector_db.add(embeddings_array)
            logger.info(f"Successfully added to FAISS, new size: {self._vector_db.ntotal}")

            for i, chunk in enumerate(chunks):
                self._id_to_chunk[start_id + i] = chunk

            logger.info(f"Stored {len(chunks)} chunks in ID mapping")

        except Exception as e:
            import traceback
            logger.error(f"FAISS add operation failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def search(
        self,
        query: str,
        limit: int = 5,
        **kwargs
    ) -> EnhancedQueryResult:
        """Search for relevant chunks."""
        start_time = time.time()

        try:
            # Generate query embedding
            query_embedding = (await self._embed_method([query]))[0]

            # Search in vector database
            if self.vector_db_type == "qdrant":
                results = await self._search_qdrant(query_embedding, limit, **kwargs)
            elif self.vector_db_type == "chromadb":
                results = await self._search_chromadb(query_embedding, limit, **kwargs)
            elif self.vector_db_type == "faiss":
                results = await self._search_faiss(query_embedding, limit, **kwargs)
            else:
                results = []

            total_time = time.time() - start_time

            return EnhancedQueryResult(
                query=query,
                results=results,
                total_time=total_time,
                metadata={
                    "vector_db_type": self.vector_db_type,
                    "embedding_model": self.embedding_model,
                    "limit": limit,
                    "legacy_retriever": True
                }
            )

        except Exception as e:
            logger.error(f"Legacy search failed: {e}")
            return EnhancedQueryResult(
                query=query,
                results=[],
                total_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    async def _search_qdrant(
        self,
        query_embedding: List[float],
        limit: int,
        **kwargs
    ) -> List[RetrievalResult]:
        """Search in Qdrant."""
        search_result = self._vector_db.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=kwargs.get("score_threshold", 0.0)
        )

        results = []
        for hit in search_result:
            chunk = DocumentChunk(
                id=hit.id,
                document_id=hit.payload["document_id"],
                content=hit.payload["content"],
                chunk_index=hit.payload["chunk_index"],
                metadata=hit.payload.get("metadata", {})
            )

            result = RetrievalResult(
                chunk=chunk,
                score=hit.score,
                retrieval_type="vector",
                metadata={"source": "qdrant"}
            )
            results.append(result)

        return results

    async def _search_chromadb(
        self,
        query_embedding: List[float],
        limit: int,
        **kwargs
    ) -> List[RetrievalResult]:
        """Search in ChromaDB."""
        search_result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )

        results = []
        for i in range(len(search_result["ids"][0])):
            chunk = DocumentChunk(
                id=search_result["ids"][0][i],
                document_id=search_result["metadatas"][0][i]["document_id"],
                content=search_result["documents"][0][i],
                chunk_index=search_result["metadatas"][0][i]["chunk_index"],
                metadata=search_result["metadatas"][0][i]
            )

            distance = search_result["distances"][0][i]
            score = 1.0 / (1.0 + distance)

            result = RetrievalResult(
                chunk=chunk,
                score=score,
                retrieval_type="vector",
                metadata={"source": "chromadb", "distance": distance}
            )
            results.append(result)

        return results

    async def _search_faiss(
        self,
        query_embedding: List[float],
        limit: int,
        **kwargs
    ) -> List[RetrievalResult]:
        """Search in FAISS."""
        import numpy as np

        query_array = np.array([query_embedding], dtype=np.float32)
        scores, indices = self._vector_db.search(query_array, limit)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx in self._id_to_chunk:
                chunk = self._id_to_chunk[idx]
                result = RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    retrieval_type="vector",
                    metadata={"source": "faiss", "index": int(idx)}
                )
                results.append(result)

        return results

    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        try:
            if self.vector_db_type == "qdrant":
                from qdrant_client.models import Filter, FieldCondition, MatchValue

                self._vector_db.delete(
                    collection_name=settings.qdrant_collection,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=document_id)
                            )
                        ]
                    )
                )

            elif self.vector_db_type == "chromadb":
                logger.warning("ChromaDB document deletion not implemented")
                return False

            elif self.vector_db_type == "faiss":
                logger.warning("FAISS document deletion not implemented")
                return False

            logger.info(f"Deleted document {document_id} from vector database")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        try:
            if self.vector_db_type == "qdrant":
                info = self._vector_db.get_collection(settings.qdrant_collection)
                return {
                    "total_chunks": info.vectors_count,
                    "vector_dimension": info.config.params.vectors.size,
                    "distance_metric": info.config.params.vectors.distance,
                }

            elif self.vector_db_type == "chromadb":
                return {
                    "total_chunks": self._collection.count(),
                }

            elif self.vector_db_type == "faiss":
                return {
                    "total_chunks": self._vector_db.ntotal,
                    "vector_dimension": self._dimension,
                }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")

        return {}


# Factory functions and enhanced hybrid retriever
def get_retriever(
    retriever_type: str = "enhanced_multimodal",
    **kwargs
) -> BaseRetriever:
    """Factory function to get an enhanced retriever."""
    if retriever_type == "enhanced_multimodal":
        return EnhancedMultiModalRetriever(**kwargs)
    elif retriever_type == "vector":
        return VectorRetriever(**kwargs)
    elif retriever_type == "hybrid":
        # Create hybrid with enhanced multimodal as base
        vector_retriever = EnhancedMultiModalRetriever(**kwargs)
        return EnhancedHybridRetriever(vector_retriever, **kwargs)
    else:
        logger.warning(f"Unknown retriever type '{retriever_type}', using enhanced_multimodal")
        return EnhancedMultiModalRetriever(**kwargs)


def get_legacy_retriever(
    retriever_type: str = "vector",
    **kwargs
) -> BaseRetriever:
    """Factory function for legacy retrievers."""
    if retriever_type == "vector":
        return VectorRetriever(**kwargs)
    elif retriever_type == "hybrid":
        vector_retriever = VectorRetriever(**kwargs)
        return HybridRetriever(vector_retriever, **kwargs)
    else:
        logger.warning(f"Unknown legacy retriever type '{retriever_type}', using vector")
        return VectorRetriever(**kwargs)


class EnhancedHybridRetriever(BaseRetriever):
    """Enhanced hybrid retrieval combining vector and keyword search with reranking."""

    def __init__(
        self,
        vector_retriever: EnhancedMultiModalRetriever,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
        enable_reranking: bool = True,
        reranker: Optional[Reranker] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vector_retriever = vector_retriever
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        self.enable_reranking = enable_reranking
        self.reranker = reranker or Reranker()
        self._keyword_index = {}

    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add chunks to both vector and keyword indices."""
        vector_success = await self.vector_retriever.add_chunks(chunks)

        for chunk in chunks:
            words = set(chunk.content.lower().split())
            for word in words:
                if word not in self._keyword_index:
                    self._keyword_index[word] = []
                self._keyword_index[word].append(chunk)

        return vector_success

    async def search(
        self,
        query: str,
        limit: int = 5,
        **kwargs
    ) -> EnhancedQueryResult:
        """Enhanced hybrid search with vector and keyword results."""
        start_time = time.time()

        try:
            # Get vector search results
            vector_results = await self.vector_retriever.search(
                query, limit * 2, **kwargs
            )

            # Get keyword search results
            keyword_results = self._keyword_search(query, limit * 2)

            # Combine results
            combined_results = self._combine_results(
                vector_results.results,
                keyword_results,
                limit
            )

            # Apply reranking
            if self.enable_reranking and combined_results:
                combined_results = await self.reranker.rerank(
                    query, combined_results, top_k=limit
                )
                reranked = True
            else:
                combined_results = combined_results[:limit]
                reranked = False

            total_time = time.time() - start_time

            return EnhancedQueryResult(
                query=query,
                results=combined_results,
                total_time=total_time,
                expanded_queries=vector_results.expanded_queries,
                reranked=reranked,
                metadata={
                    "retrieval_type": "enhanced_hybrid",
                    "vector_weight": self.vector_weight,
                    "keyword_weight": self.keyword_weight,
                    "vector_results": len(vector_results.results),
                    "keyword_results": len(keyword_results)
                }
            )

        except Exception as e:
            logger.error(f"Enhanced hybrid search failed: {e}")
            return EnhancedQueryResult(
                query=query,
                results=[],
                total_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    def _keyword_search(self, query: str, limit: int) -> List[RetrievalResult]:
        """Simple keyword-based search."""
        query_words = set(query.lower().split())
        chunk_scores = {}

        for word in query_words:
            if word in self._keyword_index:
                for chunk in self._keyword_index[word]:
                    if chunk.id not in chunk_scores:
                        chunk_scores[chunk.id] = {"chunk": chunk, "score": 0}
                    chunk_scores[chunk.id]["score"] += 1

        sorted_chunks = sorted(
            chunk_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        results = []
        for item in sorted_chunks[:limit]:
            result = RetrievalResult(
                chunk=item["chunk"],
                score=item["score"] / len(query_words),
                retrieval_type="keyword",
                metadata={"source": "keyword"}
            )
            results.append(result)

        return results

    def _combine_results(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        limit: int
    ) -> List[RetrievalResult]:
        """Combine and rerank vector and keyword results."""
        all_chunks = {}

        # Add vector results
        for result in vector_results:
            chunk_id = result.chunk.id
            all_chunks[chunk_id] = {
                "chunk": result.chunk,
                "vector_score": result.score,
                "keyword_score": 0.0
            }

        # Add keyword results
        for result in keyword_results:
            chunk_id = result.chunk.id
            if chunk_id in all_chunks:
                all_chunks[chunk_id]["keyword_score"] = result.score
            else:
                all_chunks[chunk_id] = {
                    "chunk": result.chunk,
                    "vector_score": 0.0,
                    "keyword_score": result.score
                }

        # Calculate hybrid scores
        hybrid_results = []
        for chunk_data in all_chunks.values():
            hybrid_score = (
                self.vector_weight * chunk_data["vector_score"] +
                self.keyword_weight * chunk_data["keyword_score"]
            )

            result = RetrievalResult(
                chunk=chunk_data["chunk"],
                score=hybrid_score,
                retrieval_type="enhanced_hybrid",
                metadata={
                    "source": "enhanced_hybrid",
                    "vector_score": chunk_data["vector_score"],
                    "keyword_score": chunk_data["keyword_score"]
                }
            )
            hybrid_results.append(result)

        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        return hybrid_results[:limit]

    async def delete_document(self, document_id: str) -> bool:
        """Delete document from both indices."""
        vector_success = await self.vector_retriever.delete_document(document_id)

        # Remove from keyword index
        for word, word_chunks in self._keyword_index.items():
            self._keyword_index[word] = [
                c for c in word_chunks if c.document_id != document_id
            ]

        return vector_success

    async def get_stats(self) -> Dict[str, Any]:
        """Get hybrid retriever statistics."""
        vector_stats = await self.vector_retriever.get_stats()
        keyword_stats = {
            "keyword_vocabulary_size": len(self._keyword_index),
            "total_keyword_entries": sum(
                len(chunks) for chunks in self._keyword_index.values()
            )
        }

        return {
            **vector_stats,
            **keyword_stats,
            "retrieval_type": "enhanced_hybrid"
        }


# Keep the original HybridRetriever for backward compatibility
class HybridRetriever(BaseRetriever):
    """Hybrid retrieval combining multiple retrieval strategies."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vector_retriever = vector_retriever
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        self._keyword_index = {}  # Simple keyword index

    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add chunks to both vector and keyword indices."""
        # Add to vector retriever
        vector_success = await self.vector_retriever.add_chunks(chunks)

        # Add to keyword index
        for chunk in chunks:
            words = set(chunk.content.lower().split())
            for word in words:
                if word not in self._keyword_index:
                    self._keyword_index[word] = []
                self._keyword_index[word].append(chunk)

        return vector_success

    async def search(
        self,
        query: str,
        limit: int = 5,
        **kwargs
    ) -> EnhancedQueryResult:
        """Hybrid search combining vector and keyword results."""
        start_time = time.time()

        try:
            # Get vector search results
            vector_results = await self.vector_retriever.search(
                query, limit * 2, **kwargs  # Get more results for reranking
            )

            # Get keyword search results
            keyword_results = self._keyword_search(query, limit * 2)

            # Combine and rerank results
            combined_results = self._combine_results(
                vector_results.results,
                keyword_results,
                limit
            )

            total_time = time.time() - start_time

            return EnhancedQueryResult(
                query=query,
                results=combined_results,
                total_time=total_time,
                metadata={
                    "retrieval_type": "hybrid",
                    "vector_weight": self.vector_weight,
                    "keyword_weight": self.keyword_weight
                }
            )

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return EnhancedQueryResult(
                query=query,
                results=[],
                total_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    def _keyword_search(self, query: str, limit: int) -> List[RetrievalResult]:
        """Simple keyword-based search."""
        query_words = set(query.lower().split())
        chunk_scores = {}

        # Score chunks based on keyword matches
        for word in query_words:
            if word in self._keyword_index:
                for chunk in self._keyword_index[word]:
                    if chunk.id not in chunk_scores:
                        chunk_scores[chunk.id] = {"chunk": chunk, "score": 0}
                    chunk_scores[chunk.id]["score"] += 1

        # Sort by score and convert to RetrievalResult
        sorted_chunks = sorted(
            chunk_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        results = []
        for item in sorted_chunks[:limit]:
            result = RetrievalResult(
                chunk=item["chunk"],
                score=item["score"] / len(query_words),  # Normalize
                metadata={"source": "keyword"}
            )
            results.append(result)

        return results

    def _combine_results(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        limit: int
    ) -> List[RetrievalResult]:
        """Combine and rerank vector and keyword results."""
        # Create a map of all unique chunks
        all_chunks = {}

        # Add vector results
        for result in vector_results:
            chunk_id = result.chunk.id
            all_chunks[chunk_id] = {
                "chunk": result.chunk,
                "vector_score": result.score,
                "keyword_score": 0.0
            }

        # Add keyword results
        for result in keyword_results:
            chunk_id = result.chunk.id
            if chunk_id in all_chunks:
                all_chunks[chunk_id]["keyword_score"] = result.score
            else:
                all_chunks[chunk_id] = {
                    "chunk": result.chunk,
                    "vector_score": 0.0,
                    "keyword_score": result.score
                }

        # Calculate hybrid scores
        hybrid_results = []
        for chunk_data in all_chunks.values():
            hybrid_score = (
                self.vector_weight * chunk_data["vector_score"] +
                self.keyword_weight * chunk_data["keyword_score"]
            )

            result = RetrievalResult(
                chunk=chunk_data["chunk"],
                score=hybrid_score,
                metadata={
                    "source": "hybrid",
                    "vector_score": chunk_data["vector_score"],
                    "keyword_score": chunk_data["keyword_score"]
                }
            )
            hybrid_results.append(result)

        # Sort by hybrid score and return top results
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        return hybrid_results[:limit]

    async def delete_document(self, document_id: str) -> bool:
        """Delete document from both indices."""
        vector_success = await self.vector_retriever.delete_document(document_id)

        # Remove from keyword index
        chunks_to_remove = []
        for word_chunks in self._keyword_index.values():
            chunks_to_remove.extend([
                chunk for chunk in word_chunks
                if chunk.document_id == document_id
            ])

        for chunk in chunks_to_remove:
            for word, word_chunks in self._keyword_index.items():
                self._keyword_index[word] = [
                    c for c in word_chunks if c.document_id != document_id
                ]

        return vector_success

    async def get_stats(self) -> Dict[str, Any]:
        """Get hybrid retriever statistics."""
        vector_stats = await self.vector_retriever.get_stats()
        keyword_stats = {
            "keyword_vocabulary_size": len(self._keyword_index),
            "total_keyword_entries": sum(
                len(chunks) for chunks in self._keyword_index.values()
            )
        }

        return {
            **vector_stats,
            **keyword_stats,
            "retrieval_type": "hybrid"
        }