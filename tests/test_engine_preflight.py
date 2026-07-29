from pathlib import Path

import pytest

from mlx_zonos2.adapter.engine import Zonos2EngineError, resolve_local_dac_path


def test_resolve_local_dac_path_requires_model_weights(tmp_path: Path) -> None:
    snapshot = (
        tmp_path
        / "models--mlx-community--descript-audio-codec-44khz"
        / "snapshots"
        / "revision"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")

    with pytest.raises(Zonos2EngineError, match="Automatic download is disabled"):
        resolve_local_dac_path(tmp_path)


def test_resolve_local_dac_path_returns_complete_snapshot(tmp_path: Path) -> None:
    snapshot = (
        tmp_path
        / "models--mlx-community--descript-audio-codec-44khz"
        / "snapshots"
        / "revision"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").touch()

    assert resolve_local_dac_path(tmp_path) == str(snapshot.resolve())
