from app.core.readiness import ReadinessState


def test_live_pipeline_readiness_does_not_require_local_index() -> None:
    state = ReadinessState()

    state.mark_ready(
        index_loaded=False,
        embedding_model_loaded=False,
        nli_model_loaded=True,
    )

    snapshot = state.snapshot()
    assert snapshot.analysis_ready is True
    assert snapshot.index_loaded is False
    assert snapshot.embedding_model_loaded is False


def test_readiness_error_clears_analysis_ready() -> None:
    state = ReadinessState()
    state.mark_ready(
        index_loaded=True,
        embedding_model_loaded=True,
        nli_model_loaded=True,
    )

    state.mark_error("model unavailable")

    assert state.snapshot().analysis_ready is False
