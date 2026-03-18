"""
Memory system for Aegean consensus protocol.

Provides:
- Knowledge base (static knowledge via RAG)
- Experience base (historical decisions and feedback)
- Global memory system (unified interface)
- Prompt enhancement (RAG-enhanced prompts)
- Knowledge base management (batch import, maintenance)
"""

from aegean.memory.knowledge_base import KnowledgeBase, Document, RetrievalResult
from aegean.memory.experience_base import ExperienceBase, ConsensusRecord, AgentPerformance
from aegean.memory.global_memory import GlobalMemorySystem, MemoryContext
from aegean.memory.prompt_enhancer import PromptEnhancer, PromptTemplate, MemoryAwareAgent
from aegean.memory.knowledge_manager import KnowledgeBaseManager, CategoryManager, ImportResult

__all__ = [
    # Core components
    "KnowledgeBase",
    "ExperienceBase",
    "GlobalMemorySystem",
    
    # Data models
    "Document",
    "RetrievalResult",
    "ConsensusRecord",
    "AgentPerformance",
    "MemoryContext",
    
    # Prompt enhancement
    "PromptEnhancer",
    "PromptTemplate",
    "MemoryAwareAgent",
    
    # Management tools
    "KnowledgeBaseManager",
    "CategoryManager",
    "ImportResult",
]

