from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import os

from config import settings
from logger import setup_logger
from routers import scan, confirm, price

# Setup logger
logger = setup_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup and shutdown"""
    # Startup
    logger.info("Starting HogIntel API Server")
    yield
    # Shutdown
    logger.info("Shutting down HogIntel API Server")

app = FastAPI(
    title="HogIntel API",
    description="Pig Weight & Price Estimation API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(scan.router, prefix="/api/v1", tags=["scan"])
app.include_router(confirm.router, prefix="/api/v1", tags=["confirm"])
app.include_router(price.router, prefix="/api/v1", tags=["price"])

@app.get("/")
async def root():
    return {
        "message": "HogIntel API Server",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "hogintel-api"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )