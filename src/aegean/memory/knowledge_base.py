"""
Knowledge Base for storing and retrieving static knowledge.

Uses vector database (Milvus/Pinecone) for semantic search.
Supports RAG (Retrieval-Augmented Generation) for prompt enhancement.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import hashlib


@dataclass
class Document:
    """Document stored in knowledge base."""
    doc_id: str
    content: str
    category: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class RetrievalResult:
    """Result from knowledge base retrieval."""
    documents: List[Document]
    scores: List[float]
    query: str
    total_results: int


class KnowledgeBase:
    """
    Knowledge base for storing and retrieving static knowledge.
    
    Features:
    - Document storage with embeddings
    - Semantic search via vector similarity
    - Category-based filtering
    - RAG support for prompt enhancement
    
    Storage backends:
    - Milvus (recommended for production)
    - In-memory (for development/testing)
    - Pinecone (cloud alternative)
    """
    
    def __init__(
        self,
        backend: str = "memory",
        embedding_model: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize knowledge base.
        
        Args:
            backend: Storage backend ('memory', 'milvus', 'pinecone')
            embedding_model: Model for generating embeddings
            config: Backend-specific configuration
        """
        self.backend = backend
        self.embedding_model = embedding_model
        self.config = config or {}
        
        # In-memory storage (for development)
        self._documents: Dict[str, Document] = {}
        self._embeddings: Dict[str, List[float]] = {}
        
        # Initialize backend
        self._init_backend()
    
    def _init_backend(self):
        """Initialize storage backend."""
        if self.backend == "memory":
            # Already initialized above
            pass
        elif self.backend == "milvus":
            self._init_milvus()
        elif self.backend == "pinecone":
            self._init_pinecone()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _init_milvus(self):
        """Initialize Milvus connection."""
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            
            # Connect to Milvus
            connections.connect(
                alias="default",
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 19530)
            )
            
            # Define schema
            fields = [
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.config.get("dim", 768)),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            ]
            schema = CollectionSchema(fields=fields, description="Knowledge base documents")
            
            # Create or get collection
            collection_name = self.config.get("collection_name", "knowledge_base")
            self.collection = Collection(name=collection_name, schema=schema)
            
            # Create index
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            
        except ImportError:
            raise ImportError("pymilvus not installed. Run: pip install pymilvus")
    
    def _init_pinecone(self):
        """Initialize Pinecone connection."""
        try:
            import pinecone
            
            pinecone.init(
                api_key=self.config.get("api_key"),
                environment=self.config.get("environment", "us-west1-gcp")
            )
            
            index_name = self.config.get("index_name", "knowledge-base")
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=self.config.get("dim", 768),
                    metric="cosine"
                )
            
            self.index = pinecone.Index(index_name)
            
        except ImportError:
            raise ImportError("pinecone-client not installed. Run: pip install pinecone-client")
    
    async def add_document(
        self,
        content: str,
        category: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Add a document to knowledge base.
        
        Args:
            content: Document content
            category: Document category (e.g., 'regulation', 'knowledge', 'case')
            metadata: Additional metadata
            doc_id: Optional document ID (auto-generated if not provided)
            
        Returns:
            Document ID
        """
        # Generate doc_id if not provided
        if doc_id is None:
            doc_id = self._generate_doc_id(content)
        
        # Generate embedding
        embedding = await self._generate_embedding(content)
        
        # Create document
        doc = Document(
            doc_id=doc_id,
            content=content,
            category=category,
            metadata=metadata or {},
            embedding=embedding
        )
        
        # Store based on backend
        if self.backend == "memory":
            self._documents[doc_id] = doc
            self._embeddings[doc_id] = embedding
        elif self.backend == "milvus":
            self._add_to_milvus(doc)
        elif self.backend == "pinecone":
            self._add_to_pinecone(doc)
        
        return doc_id
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        min_score: float = 0.0
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            category: Optional category filter
            min_score: Minimum similarity score
            
        Returns:
            RetrievalResult with documents and scores
        """
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)
        
        # Search based on backend
        if self.backend == "memory":
            results = self._search_memory(query_embedding, top_k, category, min_score)
        elif self.backend == "milvus":
            results = self._search_milvus(query_embedding, top_k, category, min_score)
        elif self.backend == "pinecone":
            results = self._search_pinecone(query_embedding, top_k, category, min_score)
        else:
            results = RetrievalResult(documents=[], scores=[], query=query, total_results=0)
        
        return results
    
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from knowledge base.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        if self.backend == "memory":
            if doc_id in self._documents:
                del self._documents[doc_id]
                del self._embeddings[doc_id]
                return True
            return False
        elif self.backend == "milvus":
            self.collection.delete(expr=f'doc_id == "{doc_id}"')
            return True
        elif self.backend == "pinecone":
            self.index.delete(ids=[doc_id])
            return True
        
        return False
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        if self.backend == "memory":
            return self._documents.get(doc_id)
        elif self.backend == "milvus":
            results = self.collection.query(expr=f'doc_id == "{doc_id}"', output_fields=["*"])
            if results:
                return self._milvus_to_document(results[0])
        elif self.backend == "pinecone":
            results = self.index.fetch(ids=[doc_id])
            if doc_id in results["vectors"]:
                return self._pinecone_to_document(results["vectors"][doc_id])
        
        return None
    
    def list_categories(self) -> List[str]:
        """List all document categories."""
        if self.backend == "memory":
            return list(set(doc.category for doc in self._documents.values()))
        # TODO: Implement for other backends
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        if self.backend == "memory":
            return {
                "total_documents": len(self._documents),
                "categories": len(self.list_categories()),
                "backend": self.backend
            }
        # TODO: Implement for other backends
        return {"backend": self.backend}
    
    # ==================== Helper Methods ====================
    
    def _generate_doc_id(self, content: str) -> str:
        """Generate document ID from content hash."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Uses configured embedding model or falls back to simple hash-based embedding.
        """
        if self.embedding_model:
            # Use provided embedding model
            if hasattr(self.embedding_model, 'encode'):
                # Sentence-transformers style
                embedding = self.embedding_model.encode(text)
                return embedding.tolist()
            elif callable(self.embedding_model):
                # Custom function
                return await self.embedding_model(text)
        
        # Fallback: simple hash-based embedding (for testing only)
        import hashlib
        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Convert to 768-dim vector (standard BERT size)
        embedding = []
        for i in range(768):
            embedding.append(float(hash_bytes[i % len(hash_bytes)]) / 255.0)
        return embedding
    
    def _search_memory(
        self,
        query_embedding: List[float],
        top_k: int,
        category: Optional[str],
        min_score: float
    ) -> RetrievalResult:
        """Search in-memory storage."""
        import numpy as np
        
        # Filter by category
        candidates = [
            (doc_id, doc) for doc_id, doc in self._documents.items()
            if category is None or doc.category == category
        ]
        
        if not candidates:
            return RetrievalResult(documents=[], scores=[], query="", total_results=0)
        
        # Calculate cosine similarity
        query_vec = np.array(query_embedding)
        scores = []
        
        for doc_id, doc in candidates:
            doc_vec = np.array(self._embeddings[doc_id])
            similarity = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            )
            scores.append((doc, similarity))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by min_score and take top_k
        results = [(doc, score) for doc, score in scores if score >= min_score][:top_k]
        
        documents = [doc for doc, _ in results]
        result_scores = [score for _, score in results]
        
        return RetrievalResult(
            documents=documents,
            scores=result_scores,
            query="",
            total_results=len(results)
        )
    
    def _add_to_milvus(self, doc: Document):
        """Add document to Milvus."""
        data = [
            [doc.doc_id],
            [doc.embedding],
            [doc.content],
            [doc.category]
        ]
        self.collection.insert(data)
        self.collection.flush()
    
    def _search_milvus(
        self,
        query_embedding: List[float],
        top_k: int,
        category: Optional[str],
        min_score: float
    ) -> RetrievalResult:
        """Search in Milvus."""
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        expr = f'category == "{category}"' if category else None
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["doc_id", "content", "category"]
        )
        
        documents = []
        scores = []
        
        for hits in results:
            for hit in hits:
                if hit.distance <= (1.0 - min_score):  # L2 distance
                    doc = Document(
                        doc_id=hit.entity.get("doc_id"),
                        content=hit.entity.get("content"),
                        category=hit.entity.get("category"),
                        metadata={}
                    )
                    documents.append(doc)
                    scores.append(1.0 - hit.distance)
        
        return RetrievalResult(
            documents=documents,
            scores=scores,
            query="",
            total_results=len(documents)
        )
    
    def _add_to_pinecone(self, doc: Document):
        """Add document to Pinecone."""
        self.index.upsert(
            vectors=[(
                doc.doc_id,
                doc.embedding,
                {
                    "content": doc.content,
                    "category": doc.category,
                    **doc.metadata
                }
            )]
        )
    
    def _search_pinecone(
        self,
        query_embedding: List[float],
        top_k: int,
        category: Optional[str],
        min_score: float
    ) -> RetrievalResult:
        """Search in Pinecone."""
        filter_dict = {"category": category} if category else None
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True
        )
        
        documents = []
        scores = []
        
        for match in results["matches"]:
            if match["score"] >= min_score:
                doc = Document(
                    doc_id=match["id"],
                    content=match["metadata"].get("content", ""),
                    category=match["metadata"].get("category", ""),
                    metadata=match["metadata"]
                )
                documents.append(doc)
                scores.append(match["score"])
        
        return RetrievalResult(
            documents=documents,
            scores=scores,
            query="",
            total_results=len(documents)
        )
    
    def _milvus_to_document(self, entity: Dict) -> Document:
        """Convert Milvus entity to Document."""
        return Document(
            doc_id=entity["doc_id"],
            content=entity["content"],
            category=entity["category"],
            metadata={}
        )
    
    def _pinecone_to_document(self, vector: Dict) -> Document:
        """Convert Pinecone vector to Document."""
        return Document(
            doc_id=vector["id"],
            content=vector["metadata"].get("content", ""),
            category=vector["metadata"].get("category", ""),
            metadata=vector["metadata"]
        )

