"""
FastAPI application for Aegean consensus protocol.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aegean.core.agent import AgentRegistry
from aegean.api import group_chat_api
from aegean.api import risk_api
from aegean.api import investment_api
from aegean.api import setu_api
from aegean.services.setu_service import SetuService, build_setu_config_from_env


def create_app(
    agent_registry: AgentRegistry = None,
    storage_backend = None,
    enable_cors: bool = True,
    memory_system = None,
    llm_client = None,
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

    # Initialize Setu adapter service bound to its dedicated group
    setu_service = SetuService(
        group_service=group_chat_api.get_service(),
        config=build_setu_config_from_env(),
    )
    setu_api.init_setu_service(setu_service)
    
    # Initialize risk service
    risk_coordinator = risk_api.init_risk_service(
        memory_system=memory_system,
        llm_client=llm_client,
    )

    # Initialize investment service
    investment_api.init_investment_service(
        agent_registry=agent_registry,
        memory_system=memory_system,
        llm_client=llm_client,
        risk_coordinator=risk_coordinator,
    )

    # Include routers
    app.include_router(group_chat_api.router)
    app.include_router(setu_api.router)
    app.include_router(risk_api.router)
    app.include_router(investment_api.router)
    
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

