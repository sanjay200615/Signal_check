from fastapi import APIRouter
from app.core.readiness import readiness

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict[str, str]:
    """Report whether the API process is available."""
    state = readiness.snapshot()
    components_ready = state.analysis_ready and not state.last_error
    return {
        "status": "ready" if components_ready else "degraded",
        "index_loaded": str(state.index_loaded).lower(),
        "embedding_model_loaded": str(state.embedding_model_loaded).lower(),
        "nli_model_loaded": str(state.nli_model_loaded).lower(),
        "analysis_ready": str(state.analysis_ready).lower(),
        "last_error": state.last_error or "",
    }
