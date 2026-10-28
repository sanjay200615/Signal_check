from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from app.api.routes_health import router as health_router
from app.api.routes_cases import router as cases_router
from app.api.routes_text import router as text_router

app = FastAPI(
    title="Evidence-Based Cross-Channel Deception Verification",
    description="Initial text evidence-verification prototype.",
    version="0.1.0",
)

frontend_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:4173,http://127.0.0.1:4173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(health_router)
app.include_router(cases_router)
app.include_router(text_router)
