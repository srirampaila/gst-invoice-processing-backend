"""
Main FastAPI application module for GST Intelligence System.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.api.routers.system import router as system_router
from src.api.routers.invoices import router as invoices_router
from src.api.routers.reviews import router as reviews_router
from src.api.routers.gstr1 import router as gstr1_router
from src.api.routers.evaluation import router as evaluation_router

load_dotenv()

import os
from contextlib import asynccontextmanager
from src.utils.paths import ensure_data_directories
from src.utils.config_validator import validate_environment

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure data directories exist and validate deployment configuration on startup."""
    ensure_data_directories()
    validate_environment()
    yield

app = FastAPI(
    title="GST Invoice Intelligence API",
    description="Deterministic GST Validation, Multi-Factor Confidence Scoring, and GSTR-1 Draft Preparation API",
    version="1.0.0",
    lifespan=lifespan
)

# Parse configurable CORS origins (comma-separated), falling back to safe local dev origins
raw_cors = os.getenv("CORS_ORIGINS", "")
if raw_cors.strip():
    allowed_origins = [origin.strip() for origin in raw_cors.split(",") if origin.strip()]
else:
    allowed_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

# Enable CORS for frontend with explicit allowed origins (never wildcard with credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(system_router)
app.include_router(invoices_router)
app.include_router(reviews_router)
app.include_router(gstr1_router)
app.include_router(evaluation_router)

@app.get("/")
def root():
    return {
        "name": "GST Invoice Intelligence System API Bridge",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
        "endpoints": [
            "/api/health",
            "/api/providers",
            "/api/invoices/samples",
            "/api/invoices/upload",
            "/api/invoices/process",
            "/api/reviews/all",
            "/api/gstr1/draft",
            "/api/gstr1/exports/{export_type}",
            "/api/evaluation/metrics",
            "/api/evaluation/performance",
            "/api/architecture/summary"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("APP_ENV", "development").lower() == "development"
    uvicorn.run("src.api.app:app", host=host, port=port, reload=reload)

