"""
Multi-modal Embedding Components for RAG Pipeline.
Supports text and image embeddings with unified vector output format.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np

from pydantic import BaseModel, Field

from ..core.config import settings
from ..core.logging import logger


class EmbeddingResult(BaseModel):
    """Result of an embedding operation."""

    embeddings: List[List[float]]
    model: str
    embedding_type: str  # text, image, multimodal
    dimension: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time: float = 0.0


class BaseEmbedder(ABC):
    """Base class for embedding models."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
        self.dimension = None

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Generate embeddings for text inputs."""
        pass

    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query text."""
        pass

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimension


class TextEmbedder(BaseEmbedder):
    """Text embedding using sentence-transformers or OpenAI API."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        provider: str = "sentence_transformers",
        **kwargs
    ):
        """Initialize text embedder.

        Args:
            model_name: Name of the embedding model
            provider: Provider type (sentence_transformers, openai, anthropic)
        """
        super().__init__(model_name, **kwargs)
        self.provider = provider
        self._model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the embedding model."""
        try:
            if self.provider == "sentence_transformers":
                self._initialize_sentence_transformer()
            elif self.provider == "openai":
                self._initialize_openai()
            elif self.provider == "anthropic":
                raise NotImplementedError("Anthropic embeddings not yet available")
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            logger.info(f"Initialized text embedder: {self.provider}/{self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize text embedder: {e}")
            raise

    def _initialize_sentence_transformer(self):
        """Initialize sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()

        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer: {e}")
            raise

    def _initialize_openai(self):
        """Initialize OpenAI embeddings."""
        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
            # OpenAI text-embedding-ada-002 has 1536 dimensions
            if "ada-002" in self.model_name:
                self.dimension = 1536
            elif "text-embedding-3-small" in self.model_name:
                self.dimension = 1536
            elif "text-embedding-3-large" in self.model_name:
                self.dimension = 3072
            else:
                self.dimension = 1536  # Default

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
            raise

    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Generate embeddings for multiple texts."""
        start_time = time.time()

        try:
            if self.provider == "sentence_transformers":
                embeddings = await self._embed_with_sentence_transformer(texts)
            elif self.provider == "openai":
                embeddings = await self._embed_with_openai(texts)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            return EmbeddingResult(
                embeddings=embeddings,
                model=self.model_name,
                embedding_type="text",
                dimension=self.dimension,
                processing_time=time.time() - start_time,
                metadata={
                    "provider": self.provider,
                    "input_count": len(texts),
                    "total_tokens": sum(len(text.split()) for text in texts)
                }
            )

        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            raise

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        result = await self.embed_texts([query])
        return result.embeddings[0]

    async def _embed_with_sentence_transformer(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sentence transformers."""
        try:
            # Encode texts
            embeddings = self._model.encode(
                texts,
                convert_to_tensor=False,
                normalize_embeddings=True
            )
            return embeddings.tolist()

        except Exception as e:
            logger.error(f"SentenceTransformer embedding failed: {e}")
            raise

    async def _embed_with_openai(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            response = await self._client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            return [item.embedding for item in response.data]

        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise


class ImageEmbedder(BaseEmbedder):
    """Image embedding using CLIP model."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        **kwargs
    ):
        """Initialize image embedder.

        Args:
            model_name: CLIP model name
            device: Device to run the model on (cpu, cuda, auto)
        """
        super().__init__(model_name, **kwargs)
        self.device = device
        self._model = None
        self._processor = None
        self._initialize_clip_model()

    def _initialize_clip_model(self):
        """Initialize CLIP model for image embeddings."""
        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch

            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)

            # Get embedding dimension
            self.dimension = self._model.config.projection_dim

            logger.info(f"Initialized CLIP model: {self.model_name} on {self.device}")

        except Exception as e:
            logger.error(f"Failed to initialize CLIP model: {e}")
            raise

    async def embed_images(self, image_paths: List[str]) -> EmbeddingResult:
        """Generate embeddings for images from file paths."""
        start_time = time.time()

        try:
            from PIL import Image
            import torch

            images = []
            for path in image_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    images.append(image)
                except Exception as e:
                    logger.warning(f"Failed to load image {path}: {e}")
                    # Add a placeholder (could be a blank image)
                    images.append(Image.new('RGB', (224, 224), color='white'))

            # Process images
            inputs = self._processor(images=images, return_tensors="pt").to(self.device)

            # Generate embeddings
            with torch.no_grad():
                image_features = self._model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            embeddings = image_features.cpu().numpy().tolist()

            return EmbeddingResult(
                embeddings=embeddings,
                model=self.model_name,
                embedding_type="image",
                dimension=self.dimension,
                processing_time=time.time() - start_time,
                metadata={
                    "provider": "clip",
                    "device": self.device,
                    "input_count": len(images)
                }
            )

        except Exception as e:
            logger.error(f"Image embedding failed: {e}")
            raise

    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Generate text embeddings using CLIP (for multimodal search)."""
        start_time = time.time()

        try:
            import torch

            # Process texts
            inputs = self._processor(text=texts, return_tensors="pt", padding=True).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                text_features = self._model.get_text_features(**inputs)
                # Normalize embeddings
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            embeddings = text_features.cpu().numpy().tolist()

            return EmbeddingResult(
                embeddings=embeddings,
                model=self.model_name,
                embedding_type="text_via_clip",
                dimension=self.dimension,
                processing_time=time.time() - start_time,
                metadata={
                    "provider": "clip",
                    "device": self.device,
                    "input_count": len(texts)
                }
            )

        except Exception as e:
            logger.error(f"CLIP text embedding failed: {e}")
            raise

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single text query using CLIP."""
        result = await self.embed_texts([query])
        return result.embeddings[0]


class MultiModalEmbedder:
    """Multi-modal embedder that handles both text and images."""

    def __init__(
        self,
        text_model: str = "all-MiniLM-L6-v2",
        text_provider: str = "sentence_transformers",
        image_model: str = "openai/clip-vit-base-patch32",
        use_unified_space: bool = False
    ):
        """Initialize multi-modal embedder.

        Args:
            text_model: Text embedding model
            text_provider: Text provider (sentence_transformers, openai)
            image_model: Image embedding model (CLIP)
            use_unified_space: If True, use CLIP for both text and images
        """
        self.use_unified_space = use_unified_space

        if use_unified_space:
            # Use CLIP for both text and images (shared embedding space)
            self.image_embedder = ImageEmbedder(image_model)
            self.text_embedder = self.image_embedder  # Same model
            logger.info("Initialized unified CLIP embedder for text and images")
        else:
            # Separate models for text and images
            self.text_embedder = TextEmbedder(text_model, text_provider)
            self.image_embedder = ImageEmbedder(image_model)
            logger.info("Initialized separate text and image embedders")

    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Generate text embeddings."""
        return await self.text_embedder.embed_texts(texts)

    async def embed_images(self, image_paths: List[str]) -> EmbeddingResult:
        """Generate image embeddings."""
        return await self.image_embedder.embed_images(image_paths)

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query."""
        return await self.text_embedder.embed_query(query)

    def get_text_dimension(self) -> int:
        """Get text embedding dimension."""
        return self.text_embedder.get_dimension()

    def get_image_dimension(self) -> int:
        """Get image embedding dimension."""
        return self.image_embedder.get_dimension()

    def is_unified_space(self) -> bool:
        """Check if using unified embedding space."""
        return self.use_unified_space


def get_embedder(
    embedder_type: str = "text",
    model_name: Optional[str] = None,
    provider: str = "sentence_transformers",
    **kwargs
) -> Union[TextEmbedder, ImageEmbedder, MultiModalEmbedder]:
    """Factory function to create embedders.

    Args:
        embedder_type: Type of embedder (text, image, multimodal)
        model_name: Model name (uses defaults if not provided)
        provider: Provider for text embeddings
        **kwargs: Additional arguments

    Returns:
        Appropriate embedder instance
    """
    if embedder_type == "text":
        model_name = model_name or "all-MiniLM-L6-v2"
        return TextEmbedder(model_name, provider, **kwargs)

    elif embedder_type == "image":
        model_name = model_name or "openai/clip-vit-base-patch32"
        return ImageEmbedder(model_name, **kwargs)

    elif embedder_type == "multimodal":
        text_model = kwargs.get("text_model", "all-MiniLM-L6-v2")
        image_model = kwargs.get("image_model", "openai/clip-vit-base-patch32")
        use_unified = kwargs.get("use_unified_space", False)

        return MultiModalEmbedder(
            text_model=text_model,
            text_provider=provider,
            image_model=image_model,
            use_unified_space=use_unified
        )

    else:
        raise ValueError(f"Unsupported embedder type: {embedder_type}")


# Qdrant-compatible multi-vector configuration helper
def get_qdrant_vector_config(embedder: MultiModalEmbedder) -> Dict[str, Any]:
    """Generate Qdrant vector configuration for multi-modal embeddings.

    Args:
        embedder: Multi-modal embedder instance

    Returns:
        Dictionary with Qdrant vector configuration
    """
    if embedder.is_unified_space():
        # Single vector space for both text and images
        return {
            "size": embedder.get_text_dimension(),
            "distance": "Cosine"
        }
    else:
        # Multiple vector spaces
        return {
            "text": {
                "size": embedder.get_text_dimension(),
                "distance": "Cosine"
            },
            "image": {
                "size": embedder.get_image_dimension(),
                "distance": "Cosine"
            }
        }