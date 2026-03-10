"""
Text chunking strategies for RAG pipeline.
Enhanced with table-aware chunking, semantic segmentation, and multi-modal support.
"""

import re
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from ..core.config import settings
from ..core.logging import logger
from .document_processor import DocumentChunk, ProcessedDocument, ParsedTable, ParsedImage


class EnhancedChunker(ABC):
    """Enhanced base class for text chunking strategies with table/image awareness."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        preserve_tables: bool = True,
        preserve_images: bool = True,
        **kwargs
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_tables = preserve_tables
        self.preserve_images = preserve_images

    @abstractmethod
    def chunk_document(self, document: ProcessedDocument) -> List[DocumentChunk]:
        """Chunk a document into smaller pieces with enhanced awareness."""
        pass

    def _create_chunk(
        self,
        document_id: str,
        content: str,
        chunk_index: int,
        start_pos: int = 0,
        end_pos: int = 0,
        chunk_type: str = "text",
        source_element: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentChunk:
        """Create an enhanced DocumentChunk object."""
        return DocumentChunk(
            document_id=document_id,
            content=content.strip(),
            chunk_index=chunk_index,
            start_pos=start_pos,
            end_pos=end_pos,
            chunk_type=chunk_type,
            source_element=source_element,
            metadata=metadata or {}
        )

    def _extract_enhanced_elements(self, document: ProcessedDocument) -> Dict[str, Any]:
        """Extract enhanced elements (tables, images) from document processing stats."""
        enhanced_data = {
            'tables': [],
            'images': [],
            'enhanced_result': None
        }

        if 'enhanced_result' in document.processing_stats:
            enhanced_result = document.processing_stats['enhanced_result']
            enhanced_data['tables'] = enhanced_result.get('tables', [])
            enhanced_data['images'] = enhanced_result.get('images', [])
            enhanced_data['enhanced_result'] = enhanced_result

        return enhanced_data


class TableAwareRecursiveChunker(EnhancedChunker):
    """Enhanced recursive chunker that preserves table and image boundaries."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        table_max_size: int = 2000,  # Max size for table chunks
        **kwargs
    ):
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        self.table_max_size = table_max_size
        self.separators = separators or [
            "\n=== TABLES ===\n",  # Table section separator
            "\n=== IMAGES ===\n",  # Image section separator
            "\n\n",  # Double newline (paragraph breaks)
            "\n",    # Single newline
            ". ",    # Sentence endings
            "! ",    # Exclamation sentences
            "? ",    # Question sentences
            "; ",    # Semicolons
            ", ",    # Commas
            " ",     # Spaces
            ""       # Character level (fallback)
        ]

    def chunk_document(self, document: ProcessedDocument) -> List[DocumentChunk]:
        """Chunk document with table and image awareness."""
        if not document.content.strip():
            return []

        logger.debug(f"Chunking document {document.id} with table-aware recursive strategy")

        # Extract enhanced elements
        enhanced_data = self._extract_enhanced_elements(document)

        # Split content into sections
        content_sections = self._split_by_sections(document.content)
        all_chunks = []
        chunk_index = 0

        for section_type, section_content in content_sections:
            if section_type == "text":
                # Regular text chunking
                text_chunks = self._split_text_recursive(section_content, self.separators)
                for chunk_text in text_chunks:
                    if chunk_text.strip():
                        chunk = self._create_chunk(
                            document_id=document.id,
                            content=chunk_text,
                            chunk_index=chunk_index,
                            chunk_type="text",
                            metadata={
                                "chunking_strategy": "table_aware_recursive",
                                "chunk_size": self.chunk_size,
                                "overlap": self.chunk_overlap,
                                "section_type": "text"
                            }
                        )
                        all_chunks.append(chunk)
                        chunk_index += 1

            elif section_type == "tables" and self.preserve_tables:
                # Handle tables separately
                table_chunks = self._chunk_tables(
                    document.id, section_content, enhanced_data['tables'], chunk_index
                )
                all_chunks.extend(table_chunks)
                chunk_index += len(table_chunks)

            elif section_type == "images" and self.preserve_images:
                # Handle images separately
                image_chunks = self._chunk_images(
                    document.id, section_content, enhanced_data['images'], chunk_index
                )
                all_chunks.extend(image_chunks)
                chunk_index += len(image_chunks)

        logger.info(
            f"Created {len(all_chunks)} table-aware chunks for document {document.id}"
        )
        return all_chunks

    def _split_by_sections(self, content: str) -> List[Tuple[str, str]]:
        """Split content by sections (text, tables, images)."""
        sections = []

        # Check for table and image sections
        if "=== TABLES ===" in content and "=== IMAGES ===" in content:
            parts = content.split("=== TABLES ===")
            text_part = parts[0].strip()
            remaining = "=== TABLES ===" + parts[1]

            table_image_parts = remaining.split("=== IMAGES ===")
            table_part = table_image_parts[0].replace("=== TABLES ===", "").strip()
            image_part = table_image_parts[1].strip() if len(table_image_parts) > 1 else ""

            if text_part:
                sections.append(("text", text_part))
            if table_part:
                sections.append(("tables", table_part))
            if image_part:
                sections.append(("images", image_part))

        elif "=== TABLES ===" in content:
            parts = content.split("=== TABLES ===")
            text_part = parts[0].strip()
            table_part = parts[1].strip()

            if text_part:
                sections.append(("text", text_part))
            if table_part:
                sections.append(("tables", table_part))

        elif "=== IMAGES ===" in content:
            parts = content.split("=== IMAGES ===")
            text_part = parts[0].strip()
            image_part = parts[1].strip()

            if text_part:
                sections.append(("text", text_part))
            if image_part:
                sections.append(("images", image_part))
        else:
            # No special sections, treat as all text
            sections.append(("text", content))

        return sections

    def _chunk_tables(
        self,
        document_id: str,
        table_section: str,
        tables_metadata: List[Dict],
        start_index: int
    ) -> List[DocumentChunk]:
        """Create chunks for tables, ensuring they're not broken."""
        chunks = []
        table_texts = table_section.split("Table ")[1:]  # Skip first empty split

        for i, table_text in enumerate(table_texts):
            if not table_text.strip():
                continue

            # Clean up table text
            table_content = "Table " + table_text.strip()

            # Check if table is too large and needs splitting
            if len(table_content) <= self.table_max_size:
                # Table fits in one chunk
                chunk = self._create_chunk(
                    document_id=document_id,
                    content=table_content,
                    chunk_index=start_index + i,
                    chunk_type="table",
                    source_element=f"table_{i}",
                    metadata={
                        "chunking_strategy": "table_aware",
                        "is_complete_table": True,
                        "table_index": i,
                        "original_table_metadata": tables_metadata[i] if i < len(tables_metadata) else None
                    }
                )
                chunks.append(chunk)
            else:
                # Table is too large, split by rows but mark as partial
                logger.warning(f"Table {i} is large ({len(table_content)} chars), splitting by rows")
                table_chunks = self._split_large_table(
                    document_id, table_content, start_index + i, i
                )
                chunks.extend(table_chunks)

        return chunks

    def _split_large_table(
        self,
        document_id: str,
        table_content: str,
        chunk_index: int,
        table_index: int
    ) -> List[DocumentChunk]:
        """Split a large table while preserving structure."""
        chunks = []
        lines = table_content.split('\n')

        # Identify header and separator
        header_lines = []
        data_lines = []
        in_data = False

        for line in lines:
            if '---' in line and '|' in line:  # Separator line
                header_lines.append(line)
                in_data = True
            elif not in_data:
                header_lines.append(line)
            else:
                data_lines.append(line)

        header_text = '\n'.join(header_lines)

        # Split data lines into chunks
        current_chunk = []
        current_size = len(header_text)

        for line in data_lines:
            line_size = len(line) + 1  # +1 for newline

            if current_size + line_size > self.table_max_size and current_chunk:
                # Create chunk with current data
                chunk_content = header_text + '\n' + '\n'.join(current_chunk)
                chunk = self._create_chunk(
                    document_id=document_id,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    chunk_type="table",
                    source_element=f"table_{table_index}_part_{len(chunks)}",
                    metadata={
                        "chunking_strategy": "table_aware",
                        "is_complete_table": False,
                        "table_index": table_index,
                        "table_part": len(chunks),
                        "has_header": True
                    }
                )
                chunks.append(chunk)

                # Reset for next chunk
                current_chunk = [line]
                current_size = len(header_text) + line_size
                chunk_index += 1
            else:
                current_chunk.append(line)
                current_size += line_size

        # Add final chunk if there's remaining data
        if current_chunk:
            chunk_content = header_text + '\n' + '\n'.join(current_chunk)
            chunk = self._create_chunk(
                document_id=document_id,
                content=chunk_content,
                chunk_index=chunk_index,
                chunk_type="table",
                source_element=f"table_{table_index}_part_{len(chunks)}",
                metadata={
                    "chunking_strategy": "table_aware",
                    "is_complete_table": False,
                    "table_index": table_index,
                    "table_part": len(chunks),
                    "has_header": True
                }
            )
            chunks.append(chunk)

        return chunks

    def _chunk_images(
        self,
        document_id: str,
        image_section: str,
        images_metadata: List[Dict],
        start_index: int
    ) -> List[DocumentChunk]:
        """Create chunks for image descriptions."""
        chunks = []
        image_texts = image_section.split("Image ")[1:]  # Skip first empty split

        for i, image_text in enumerate(image_texts):
            if not image_text.strip():
                continue

            image_content = "Image " + image_text.strip()

            chunk = self._create_chunk(
                document_id=document_id,
                content=image_content,
                chunk_index=start_index + i,
                chunk_type="image_description",
                source_element=f"image_{i}",
                metadata={
                    "chunking_strategy": "table_aware",
                    "image_index": i,
                    "original_image_metadata": images_metadata[i] if i < len(images_metadata) else None
                }
            )
            chunks.append(chunk)

        return chunks

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the provided separators."""
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Character-level splitting (fallback)
            return self._split_by_character(text)

        # Skip section separators for regular text
        if separator in ["\n=== TABLES ===\n", "\n=== IMAGES ===\n"]:
            return self._split_text_recursive(text, remaining_separators)

        splits = text.split(separator)
        chunks = []
        current_chunk = ""

        for split in splits:
            # If this split would make the chunk too large, process current chunk
            test_chunk = current_chunk + (separator if current_chunk else "") + split

            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    # Current chunk is ready, process it
                    if len(current_chunk) > self.chunk_size:
                        # Current chunk is still too large, split recursively
                        sub_chunks = self._split_text_recursive(
                            current_chunk, remaining_separators
                        )
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(current_chunk)

                # Start new chunk with current split
                current_chunk = split

        # Add remaining chunk
        if current_chunk:
            if len(current_chunk) > self.chunk_size:
                sub_chunks = self._split_text_recursive(
                    current_chunk, remaining_separators
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(current_chunk)

        # Add overlap between chunks
        return self._add_overlap(chunks)

    def _split_by_character(self, text: str) -> List[str]:
        """Split text by characters when all other separators fail."""
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks."""
        if not chunks or self.chunk_overlap == 0:
            return chunks

        overlapped_chunks = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = overlapped_chunks[-1]
            current_chunk = chunks[i]

            # Add overlap from previous chunk
            if len(prev_chunk) >= self.chunk_overlap:
                overlap = prev_chunk[-self.chunk_overlap:]
                overlapped_chunk = overlap + " " + current_chunk
            else:
                overlapped_chunk = current_chunk

            overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks


# Enhanced semantic chunker
class EnhancedSemanticChunker(EnhancedChunker):
    """Enhanced semantic-aware chunking with table/image preservation."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        similarity_threshold: float = 0.7,
        **kwargs
    ):
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        self.similarity_threshold = similarity_threshold
        self._sentence_model = None

    def chunk_document(self, document: ProcessedDocument) -> List[DocumentChunk]:
        """Chunk document using semantic similarity with table/image awareness."""
        if not document.content.strip():
            return []

        logger.debug(f"Chunking document {document.id} with enhanced semantic strategy")

        # Extract enhanced elements first
        enhanced_data = self._extract_enhanced_elements(document)

        # Split content into sections
        content_sections = self._split_by_sections(document.content)
        all_chunks = []
        chunk_index = 0

        for section_type, section_content in content_sections:
            if section_type == "text":
                # Apply semantic chunking to text sections
                text_chunks = self._semantic_chunk_text(section_content)
                for chunk_text in text_chunks:
                    if chunk_text.strip():
                        chunk = self._create_chunk(
                            document_id=document.id,
                            content=chunk_text,
                            chunk_index=chunk_index,
                            chunk_type="text",
                            metadata={
                                "chunking_strategy": "enhanced_semantic",
                                "similarity_threshold": self.similarity_threshold,
                                "section_type": "text"
                            }
                        )
                        all_chunks.append(chunk)
                        chunk_index += 1

            elif section_type == "tables" and self.preserve_tables:
                # Handle tables using table-aware methods
                table_chunks = self._chunk_tables(
                    document.id, section_content, enhanced_data['tables'], chunk_index
                )
                all_chunks.extend(table_chunks)
                chunk_index += len(table_chunks)

            elif section_type == "images" and self.preserve_images:
                # Handle images
                image_chunks = self._chunk_images(
                    document.id, section_content, enhanced_data['images'], chunk_index
                )
                all_chunks.extend(image_chunks)
                chunk_index += len(image_chunks)

        logger.info(
            f"Created {len(all_chunks)} enhanced semantic chunks for document {document.id}"
        )
        return all_chunks

    def _split_by_sections(self, content: str) -> List[Tuple[str, str]]:
        """Split content by sections (text, tables, images)."""
        sections = []

        # Check for table and image sections
        if "=== TABLES ===" in content and "=== IMAGES ===" in content:
            parts = content.split("=== TABLES ===")
            text_part = parts[0].strip()
            remaining = "=== TABLES ===" + parts[1]

            table_image_parts = remaining.split("=== IMAGES ===")
            table_part = table_image_parts[0].replace("=== TABLES ===", "").strip()
            image_part = table_image_parts[1].strip() if len(table_image_parts) > 1 else ""

            if text_part:
                sections.append(("text", text_part))
            if table_part:
                sections.append(("tables", table_part))
            if image_part:
                sections.append(("images", image_part))

        elif "=== TABLES ===" in content:
            parts = content.split("=== TABLES ===")
            text_part = parts[0].strip()
            table_part = parts[1].strip()

            if text_part:
                sections.append(("text", text_part))
            if table_part:
                sections.append(("tables", table_part))

        elif "=== IMAGES ===" in content:
            parts = content.split("=== IMAGES ===")
            text_part = parts[0].strip()
            image_part = parts[1].strip()

            if text_part:
                sections.append(("text", text_part))
            if image_part:
                sections.append(("images", image_part))
        else:
            # No special sections, treat as all text
            sections.append(("text", content))

        return sections

    def _semantic_chunk_text(self, text: str) -> List[str]:
        """Apply semantic chunking to text content."""
        # Split into sentences
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        # Group sentences semantically
        semantic_groups = self._group_sentences_semantically(sentences)

        # Convert groups to text chunks
        chunks = []
        for group in semantic_groups:
            chunk_text = " ".join(group)
            if chunk_text.strip():
                chunks.append(chunk_text)

        return chunks

    def _chunk_tables(
        self,
        document_id: str,
        table_section: str,
        tables_metadata: List[Dict],
        start_index: int
    ) -> List[DocumentChunk]:
        """Create chunks for tables."""
        chunks = []
        table_texts = table_section.split("Table ")[1:]

        for i, table_text in enumerate(table_texts):
            if not table_text.strip():
                continue

            table_content = "Table " + table_text.strip()
            chunk = self._create_chunk(
                document_id=document_id,
                content=table_content,
                chunk_index=start_index + i,
                chunk_type="table",
                source_element=f"table_{i}",
                metadata={
                    "chunking_strategy": "enhanced_semantic",
                    "is_complete_table": True,
                    "table_index": i,
                    "original_table_metadata": tables_metadata[i] if i < len(tables_metadata) else None
                }
            )
            chunks.append(chunk)

        return chunks

    def _chunk_images(
        self,
        document_id: str,
        image_section: str,
        images_metadata: List[Dict],
        start_index: int
    ) -> List[DocumentChunk]:
        """Create chunks for image descriptions."""
        chunks = []
        image_texts = image_section.split("Image ")[1:]

        for i, image_text in enumerate(image_texts):
            if not image_text.strip():
                continue

            image_content = "Image " + image_text.strip()
            chunk = self._create_chunk(
                document_id=document_id,
                content=image_content,
                chunk_index=start_index + i,
                chunk_type="image_description",
                source_element=f"image_{i}",
                metadata={
                    "chunking_strategy": "enhanced_semantic",
                    "image_index": i,
                    "original_image_metadata": images_metadata[i] if i < len(images_metadata) else None
                }
            )
            chunks.append(chunk)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentence_endings = re.compile(r'[.!?]+')
        sentences = sentence_endings.split(text)

        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def _group_sentences_semantically(self, sentences: List[str]) -> List[List[str]]:
        """Group sentences based on semantic similarity."""
        if not self._sentence_model:
            self._load_sentence_model()

        if not self._sentence_model or len(sentences) <= 1:
            return self._group_by_size(sentences)

        try:
            embeddings = self._sentence_model.encode(sentences)
            groups = []
            current_group = [sentences[0]]
            current_length = len(sentences[0])

            for i in range(1, len(sentences)):
                sentence = sentences[i]
                sentence_length = len(sentence)

                if current_length + sentence_length > self.chunk_size:
                    groups.append(current_group)
                    current_group = [sentence]
                    current_length = sentence_length
                else:
                    group_embedding = embeddings[i-len(current_group):i].mean(axis=0)
                    sentence_embedding = embeddings[i]
                    similarity = self._cosine_similarity(group_embedding, sentence_embedding)

                    if similarity >= self.similarity_threshold:
                        current_group.append(sentence)
                        current_length += sentence_length
                    else:
                        groups.append(current_group)
                        current_group = [sentence]
                        current_length = sentence_length

            if current_group:
                groups.append(current_group)

            return groups

        except Exception as e:
            logger.warning(f"Semantic grouping failed: {e}, falling back to size-based")
            return self._group_by_size(sentences)

    def _group_by_size(self, sentences: List[str]) -> List[List[str]]:
        """Fallback grouping by size only."""
        groups = []
        current_group = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length <= self.chunk_size:
                current_group.append(sentence)
                current_length += sentence_length
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [sentence]
                current_length = sentence_length

        if current_group:
            groups.append(current_group)

        return groups

    def _load_sentence_model(self):
        """Load sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer model for enhanced semantic chunking")
        except ImportError:
            logger.warning(
                "sentence-transformers not available, "
                "enhanced semantic chunking will fall back to size-based"
            )
            self._sentence_model = None

    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Legacy base class for backward compatibility
class BaseChunker(ABC):
    """Legacy base class for text chunking strategies."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk_document(self, document: ProcessedDocument) -> List[DocumentChunk]:
        """Chunk a document into smaller pieces."""
        pass

    def _create_chunk(
        self,
        document_id: str,
        content: str,
        chunk_index: int,
        start_pos: int = 0,
        end_pos: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentChunk:
        """Create a DocumentChunk object."""
        return DocumentChunk(
            document_id=document_id,
            content=content.strip(),
            chunk_index=chunk_index,
            start_pos=start_pos,
            end_pos=end_pos,
            metadata=metadata or {}
        )


class RecursiveChunker(BaseChunker):
    """
    Recursive text chunking that tries to preserve semantic boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        self.separators = separators or [
            "\n\n",  # Double newline (paragraph breaks)
            "\n",    # Single newline
            ". ",    # Sentence endings
            "! ",    # Exclamation sentences
            "? ",    # Question sentences
            "; ",    # Semicolons
            ", ",    # Commas
            " ",     # Spaces
            ""       # Character level (fallback)
        ]

    def chunk_document(self, document: ProcessedDocument) -> List[DocumentChunk]:
        """Chunk document using recursive strategy."""
        if not document.content.strip():
            return []

        logger.debug(f"Chunking document {document.id} with recursive strategy")

        chunks = []
        text_chunks = self._split_text(document.content)

        for i, chunk_text in enumerate(text_chunks):
            if chunk_text.strip():
                chunk = self._create_chunk(
                    document_id=document.id,
                    content=chunk_text,
                    chunk_index=i,
                    metadata={
                        "chunking_strategy": "recursive",
                        "chunk_size": self.chunk_size,
                        "overlap": self.chunk_overlap
                    }
                )
                chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks for document {document.id}")
        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Split text recursively using different separators."""
        return self._split_text_recursive(text, self.separators)

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the provided separators."""
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Character-level splitting (fallback)
            return self._split_by_character(text)

        splits = text.split(separator)
        chunks = []
        current_chunk = ""

        for split in splits:
            # If this split would make the chunk too large, process current chunk
            test_chunk = current_chunk + (separator if current_chunk else "") + split

            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    # Current chunk is ready, process it
                    if len(current_chunk) > self.chunk_size:
                        # Current chunk is still too large, split recursively
                        sub_chunks = self._split_text_recursive(
                            current_chunk, remaining_separators
                        )
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(current_chunk)

                # Start new chunk with current split
                current_chunk = split

        # Add remaining chunk
        if current_chunk:
            if len(current_chunk) > self.chunk_size:
                sub_chunks = self._split_text_recursive(
                    current_chunk, remaining_separators
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(current_chunk)

        # Add overlap between chunks
        return self._add_overlap(chunks)

    def _split_by_character(self, text: str) -> List[str]:
        """Split text by characters when all other separators fail."""
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks."""
        if not chunks or self.chunk_overlap == 0:
            return chunks

        overlapped_chunks = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = overlapped_chunks[-1]
            current_chunk = chunks[i]

            # Add overlap from previous chunk
            if len(prev_chunk) >= self.chunk_overlap:
                overlap = prev_chunk[-self.chunk_overlap:]
                overlapped_chunk = overlap + " " + current_chunk
            else:
                overlapped_chunk = current_chunk

            overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks


class SemanticChunker(BaseChunker):
    """
    Semantic-aware chunking that uses sentence embeddings to group
    semantically similar sentences together.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        similarity_threshold: float = 0.7,
        **kwargs
    ):
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        self.similarity_threshold = similarity_threshold
        self._sentence_model = None

    def chunk_document(self, document: ProcessedDocument) -> List[DocumentChunk]:
        """Chunk document using semantic similarity."""
        if not document.content.strip():
            return []

        logger.debug(f"Chunking document {document.id} with semantic strategy")

        # Split into sentences
        sentences = self._split_into_sentences(document.content)
        if not sentences:
            return []

        # Group sentences semantically
        semantic_groups = self._group_sentences_semantically(sentences)

        # Create chunks from groups
        chunks = []
        for i, group in enumerate(semantic_groups):
            chunk_text = " ".join(group)
            if chunk_text.strip():
                chunk = self._create_chunk(
                    document_id=document.id,
                    content=chunk_text,
                    chunk_index=i,
                    metadata={
                        "chunking_strategy": "semantic",
                        "sentence_count": len(group),
                        "similarity_threshold": self.similarity_threshold
                    }
                )
                chunks.append(chunk)

        logger.info(f"Created {len(chunks)} semantic chunks for document {document.id}")
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - could be improved with spaCy or NLTK
        sentence_endings = re.compile(r'[.!?]+')
        sentences = sentence_endings.split(text)

        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter very short sentences
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def _group_sentences_semantically(self, sentences: List[str]) -> List[List[str]]:
        """Group sentences based on semantic similarity."""
        if not self._sentence_model:
            self._load_sentence_model()

        if not self._sentence_model or len(sentences) <= 1:
            # Fallback to size-based grouping
            return self._group_by_size(sentences)

        try:
            # Compute embeddings for all sentences
            embeddings = self._sentence_model.encode(sentences)

            # Group sentences based on similarity
            groups = []
            current_group = [sentences[0]]
            current_length = len(sentences[0])

            for i in range(1, len(sentences)):
                sentence = sentences[i]
                sentence_length = len(sentence)

                # Check if adding this sentence would exceed chunk size
                if current_length + sentence_length > self.chunk_size:
                    groups.append(current_group)
                    current_group = [sentence]
                    current_length = sentence_length
                else:
                    # Check semantic similarity with current group
                    group_embedding = embeddings[i-len(current_group):i].mean(axis=0)
                    sentence_embedding = embeddings[i]

                    similarity = self._cosine_similarity(group_embedding, sentence_embedding)

                    if similarity >= self.similarity_threshold:
                        current_group.append(sentence)
                        current_length += sentence_length
                    else:
                        groups.append(current_group)
                        current_group = [sentence]
                        current_length = sentence_length

            # Add final group
            if current_group:
                groups.append(current_group)

            return groups

        except Exception as e:
            logger.warning(f"Semantic grouping failed: {e}, falling back to size-based")
            return self._group_by_size(sentences)

    def _group_by_size(self, sentences: List[str]) -> List[List[str]]:
        """Fallback grouping by size only."""
        groups = []
        current_group = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length <= self.chunk_size:
                current_group.append(sentence)
                current_length += sentence_length
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [sentence]
                current_length = sentence_length

        if current_group:
            groups.append(current_group)

        return groups

    def _load_sentence_model(self):
        """Load sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer model for semantic chunking")
        except ImportError:
            logger.warning(
                "sentence-transformers not available, "
                "semantic chunking will fall back to size-based"
            )
            self._sentence_model = None

    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class FixedSizeChunker(BaseChunker):
    """Simple fixed-size chunking with overlap."""

    def chunk_document(self, document: ProcessedDocument) -> List[DocumentChunk]:
        """Chunk document into fixed-size pieces."""
        if not document.content.strip():
            return []

        logger.debug(f"Chunking document {document.id} with fixed-size strategy")

        chunks = []
        text = document.content
        step_size = self.chunk_size - self.chunk_overlap

        for i, start in enumerate(range(0, len(text), step_size)):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunk = self._create_chunk(
                    document_id=document.id,
                    content=chunk_text,
                    chunk_index=i,
                    start_pos=start,
                    end_pos=end,
                    metadata={
                        "chunking_strategy": "fixed_size",
                        "chunk_size": self.chunk_size,
                        "overlap": self.chunk_overlap
                    }
                )
                chunks.append(chunk)

            # Break if we've reached the end
            if end >= len(text):
                break

        logger.info(f"Created {len(chunks)} fixed-size chunks for document {document.id}")
        return chunks


def get_chunker(strategy: str = "table_aware_recursive", **kwargs) -> EnhancedChunker:
    """Factory function to get an enhanced chunker based on strategy."""
    enhanced_chunkers = {
        "table_aware_recursive": TableAwareRecursiveChunker,
        "enhanced_semantic": EnhancedSemanticChunker,
        "table_aware": TableAwareRecursiveChunker,  # Alias
        "semantic": EnhancedSemanticChunker,  # Alias for enhanced version
    }

    legacy_chunkers = {
        "recursive": RecursiveChunker,
        "legacy_semantic": SemanticChunker,
        "fixed": FixedSizeChunker,
    }

    # Check enhanced chunkers first
    if strategy in enhanced_chunkers:
        chunker_class = enhanced_chunkers[strategy]
    elif strategy in legacy_chunkers:
        chunker_class = legacy_chunkers[strategy]
        logger.warning(f"Using legacy chunking strategy '{strategy}'. Consider using enhanced versions.")
    else:
        logger.warning(f"Unknown chunking strategy '{strategy}', using table_aware_recursive")
        chunker_class = TableAwareRecursiveChunker

    # Use settings defaults if not provided
    chunk_size = kwargs.pop("chunk_size", settings.chunk_size)
    chunk_overlap = kwargs.pop("chunk_overlap", settings.chunk_overlap)

    return chunker_class(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs
    )


def get_enhanced_chunker(
    strategy: str = "table_aware_recursive",
    **kwargs
) -> EnhancedChunker:
    """Factory function specifically for enhanced chunkers."""
    enhanced_chunkers = {
        "table_aware_recursive": TableAwareRecursiveChunker,
        "enhanced_semantic": EnhancedSemanticChunker,
        "table_aware": TableAwareRecursiveChunker,
        "semantic": EnhancedSemanticChunker,
    }

    if strategy not in enhanced_chunkers:
        logger.warning(f"Unknown enhanced chunking strategy '{strategy}', using table_aware_recursive")
        strategy = "table_aware_recursive"

    chunk_size = kwargs.get("chunk_size", settings.chunk_size)
    chunk_overlap = kwargs.get("chunk_overlap", settings.chunk_overlap)

    return enhanced_chunkers[strategy](
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs
    )


# Legacy factory function for backward compatibility
def get_legacy_chunker(strategy: str = "recursive", **kwargs) -> BaseChunker:
    """Legacy factory function for backward compatibility."""
    chunkers = {
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
        "fixed": FixedSizeChunker,
    }

    if strategy not in chunkers:
        logger.warning(f"Unknown legacy chunking strategy '{strategy}', using recursive")
        strategy = "recursive"

    chunk_size = kwargs.get("chunk_size", settings.chunk_size)
    chunk_overlap = kwargs.get("chunk_overlap", settings.chunk_overlap)

    return chunkers[strategy](
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs
    )