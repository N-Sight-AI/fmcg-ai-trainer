from fastapi import FastAPI
from .api.training import training_router

def create_app():
    app = FastAPI(
        title="FMCG AI Trainer API", 
        version="1.0.0",
        description="AI Training API for FMCG recommendations and similar customer analysis",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    app.include_router(training_router, prefix="/api/v1")
    return app

app = create_app()
