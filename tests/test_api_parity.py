from __future__ import annotations

import importlib.util

import pytest
from fastapi.testclient import TestClient


def _zonos2_available() -> bool:
    return importlib.util.find_spec("mlx_audio.tts.models.zonos2") is not None


@pytest.fixture(scope="module")
def client() -> TestClient:
    from mlx_zonos2.server import api_server

    api_server._ENGINE = None
    api_server._SERVER_CONFIG = None
    return TestClient(api_server.app)


def test_stream_rejected(client: TestClient) -> None:
    if not _zonos2_available():
        pytest.skip("mlx-audio zonos2 module not installed")
    response = client.post(
        "/tts/generate",
        json={"text": "hello", "stream": True, "max_tokens": 8},
    )
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


def test_health_without_engine(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code in {200, 503}