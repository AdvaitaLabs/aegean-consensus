"""
Knowledge base management utilities.

Provides:
- Batch document import
- Category management
- Document preprocessing
- Knowledge base maintenance
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import asyncio
from dataclasses import dataclass

from aegean.memory.knowledge_base import KnowledgeBase


@dataclass
class ImportResult:
    """Result of batch import operation."""
    total_files: int
    successful: int
    failed: int
    doc_ids: List[str]
    errors: List[Dict[str, str]]


class KnowledgeBaseManager:
    """
    Manager for knowledge base operations.
    
    Features:
    - Batch document import
    - Document preprocessing
    - Category management
    - Maintenance operations
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        """
        Initialize knowledge base manager.
        
        Args:
            knowledge_base: KnowledgeBase instance to manage
        """
        self.kb = knowledge_base
    
    async def import_directory(
        self,
        directory: str,
        category: str,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True,
        chunk_size: int = 1000,
        metadata_extractor: Optional[callable] = None
    ) -> ImportResult:
        """
        Import all documents from a directory.
        
        Args:
            directory: Directory path
            category: Category for all documents
            file_extensions: List of file extensions to import (e.g., ['.txt', '.md'])
            recursive: Whether to search subdirectories
            chunk_size: Maximum characters per document chunk
            metadata_extractor: Optional function to extract metadata from file
            
        Returns:
            ImportResult with statistics
        """
        path = Path(directory)
        if not path.exists():
            raise ValueError(f"Directory not found: {directory}")
        
        # Default file extensions
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.pdf', '.docx']
        
        # Find all files
        if recursive:
            files = []
            for ext in file_extensions:
                files.extend(path.rglob(f"*{ext}"))
        else:
            files = []
            for ext in file_extensions:
                files.extend(path.glob(f"*{ext}"))
        
        # Import files
        doc_ids = []
        errors = []
        
        for file_path in files:
            try:
                # Read file content
                content = await self._read_file(file_path)
                
                # Extract metadata
                metadata = {}
                if metadata_extractor:
                    metadata = metadata_extractor(file_path)
                else:
                    metadata = {
                        "filename": file_path.name,
                        "filepath": str(file_path),
                        "extension": file_path.suffix
                    }
                
                # Chunk if necessary
                chunks = self._chunk_text(content, chunk_size)
                
                # Import each chunk
                for i, chunk in enumerate(chunks):
                    chunk_metadata = metadata.copy()
                    if len(chunks) > 1:
                        chunk_metadata["chunk_index"] = i
                        chunk_metadata["total_chunks"] = len(chunks)
                    
                    doc_id = await self.kb.add_document(
                        content=chunk,
                        category=category,
                        metadata=chunk_metadata
                    )
                    doc_ids.append(doc_id)
                
            except Exception as e:
                errors.append({
                    "file": str(file_path),
                    "error": str(e)
                })
        
        return ImportResult(
            total_files=len(files),
            successful=len(files) - len(errors),
            failed=len(errors),
            doc_ids=doc_ids,
            errors=errors
        )
    
    async def import_file(
        self,
        file_path: str,
        category: str,
        chunk_size: int = 1000,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Import a single file.
        
        Args:
            file_path: Path to file
            category: Document category
            chunk_size: Maximum characters per chunk
            metadata: Optional metadata
            
        Returns:
            List of document IDs
        """
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"File not found: {file_path}")
        
        # Read content
        content = await self._read_file(path)
        
        # Prepare metadata
        file_metadata = metadata or {}
        file_metadata.update({
            "filename": path.name,
            "filepath": str(path),
            "extension": path.suffix
        })
        
        # Chunk and import
        chunks = self._chunk_text(content, chunk_size)
        doc_ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = file_metadata.copy()
            if len(chunks) > 1:
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)
            
            doc_id = await self.kb.add_document(
                content=chunk,
                category=category,
                metadata=chunk_metadata
            )
            doc_ids.append(doc_id)
        
        return doc_ids
    
    async def import_text(
        self,
        text: str,
        category: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000
    ) -> List[str]:
        """
        Import text directly.
        
        Args:
            text: Text content
            category: Document category
            metadata: Optional metadata
            chunk_size: Maximum characters per chunk
            
        Returns:
            List of document IDs
        """
        chunks = self._chunk_text(text, chunk_size)
        doc_ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            if len(chunks) > 1:
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)
            
            doc_id = await self.kb.add_document(
                content=chunk,
                category=category,
                metadata=chunk_metadata
            )
            doc_ids.append(doc_id)
        
        return doc_ids
    
    async def delete_category(self, category: str) -> int:
        """
        Delete all documents in a category.
        
        Args:
            category: Category to delete
            
        Returns:
            Number of documents deleted
        """
        # Get all documents in category
        # Note: This is a simplified implementation
        # In production, you'd want to query by category
        count = 0
        
        # TODO: Implement category-based deletion
        # For now, return 0
        return count
    
    async def update_document(
        self,
        doc_id: str,
        content: Optional[str] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update a document.
        
        Args:
            doc_id: Document ID
            content: New content (optional)
            category: New category (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if updated, False if not found
        """
        # Get existing document
        doc = await self.kb.get_document(doc_id)
        if not doc:
            return False
        
        # Delete old document
        await self.kb.delete_document(doc_id)
        
        # Add updated document
        await self.kb.add_document(
            content=content or doc.content,
            category=category or doc.category,
            metadata=metadata or doc.metadata,
            doc_id=doc_id
        )
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        stats = self.kb.get_stats()
        categories = self.kb.list_categories()
        
        return {
            **stats,
            "categories": categories,
            "category_count": len(categories)
        }
    
    # ==================== Helper Methods ====================
    
    async def _read_file(self, file_path: Path) -> str:
        """
        Read file content based on extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            File content as string
        """
        extension = file_path.suffix.lower()
        
        if extension in ['.txt', '.md']:
            # Plain text
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif extension == '.pdf':
            # PDF (requires PyPDF2 or pdfplumber)
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = []
                    for page in reader.pages:
                        text.append(page.extract_text())
                    return '\n'.join(text)
            except ImportError:
                raise ImportError("PyPDF2 not installed. Run: pip install PyPDF2")
        
        elif extension == '.docx':
            # Word document (requires python-docx)
            try:
                from docx import Document
                doc = Document(file_path)
                text = []
                for para in doc.paragraphs:
                    text.append(para.text)
                return '\n'.join(text)
            except ImportError:
                raise ImportError("python-docx not installed. Run: pip install python-docx")
        
        else:
            # Try as plain text
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Split by paragraphs
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            para_size = len(para)
            
            if current_size + para_size > chunk_size and current_chunk:
                # Save current chunk
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            if para_size > chunk_size:
                # Paragraph too large, split by sentences
                sentences = para.split('. ')
                for sent in sentences:
                    sent_size = len(sent)
                    if current_size + sent_size > chunk_size and current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                        current_chunk = []
                        current_size = 0
                    current_chunk.append(sent)
                    current_size += sent_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks


class CategoryManager:
    """
    Manager for knowledge base categories.
    
    Provides category-level operations and statistics.
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        """
        Initialize category manager.
        
        Args:
            knowledge_base: KnowledgeBase instance
        """
        self.kb = knowledge_base
    
    def list_categories(self) -> List[str]:
        """List all categories."""
        return self.kb.list_categories()
    
    async def get_category_stats(self, category: str) -> Dict[str, Any]:
        """
        Get statistics for a category.
        
        Args:
            category: Category name
            
        Returns:
            Dictionary with category statistics
        """
        # TODO: Implement category-specific statistics
        return {
            "category": category,
            "document_count": 0,
            "total_size": 0
        }
    
    async def merge_categories(
        self,
        source_category: str,
        target_category: str
    ) -> int:
        """
        Merge one category into another.
        
        Args:
            source_category: Category to merge from
            target_category: Category to merge into
            
        Returns:
            Number of documents moved
        """
        # TODO: Implement category merging
        return 0
    
    async def rename_category(
        self,
        old_name: str,
        new_name: str
    ) -> int:
        """
        Rename a category.
        
        Args:
            old_name: Current category name
            new_name: New category name
            
        Returns:
            Number of documents updated
        """
        # TODO: Implement category renaming
        return 0

