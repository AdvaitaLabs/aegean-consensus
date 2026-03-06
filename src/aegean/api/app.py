"""
FastAPI application for Aegean consensus protocol.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aegean.core.agent import AgentRegistry
from aegean.api import group_chat_api


def create_app(
    agent_registry: AgentRegistry = None,
    storage_backend = None,
    enable_cors: bool = True
) -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Args:
        agent_registry: AgentRegistry instance (creates default if None)
        storage_backend: Optional storage backend for persistence
        enable_cors: Whether to enable CORS middleware
        
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Aegean Consensus API",
        description="Multi-agent consensus protocol with weighted voting",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Enable CORS if requested
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Initialize agent registry if not provided
    if agent_registry is None:
        agent_registry = AgentRegistry()
    
    # Initialize GroupChatService
    group_chat_api.init_service(agent_registry, storage_backend)
    
    # Include routers
    app.include_router(group_chat_api.router)
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "Aegean Consensus API",
            "version": "0.1.0",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    return app

