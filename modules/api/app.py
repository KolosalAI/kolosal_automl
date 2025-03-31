from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import os
import sys

# Import routers
from modules.api.engine import router as engine_router
from modules.api.preprocessor import router as preprocessor_router
from modules.api.inference import router as inference_router
from modules.api.monitoring import router as monitoring_router
from modules.api.optimizer import router as optimizer_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ml-api")

# Create FastAPI app
app = FastAPI(
    title="ML Training and Inference API",
    description="API for ML model training, optimization, and inference",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(engine_router)
app.include_router(preprocessor_router)
app.include_router(inference_router)
app.include_router(monitoring_router)
app.include_router(optimizer_router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "ML API is running",
        "docs_url": "/docs",
        "version": "1.0.0"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "components": {
            "api": "running",
            "database": "connected"  # You may want to add actual DB checks here
        }
    }

if __name__ == "__main__":
    # Run the application using uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("modules.api.app:app", host="0.0.0.0", port=port, reload=True)