"""
End-to-end test: start a real Uvicorn server on a random port, hit /predict,
then shut everything down. The test succeeds even when
models/random_forest_model/model.pkl is missing, because we stub open() /
pickle.load in *both* the server process and the pytest process.

Run only E2E tests:
    pytest -m e2e
Skip E2E tests:
    pytest -m "not e2e"
"""
from __future__ import annotations

import builtins
import importlib
import io
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import requests


# ----------------------------------------------------------------------------- helpers
def _free_port() -> int:
    """Ask the OS for an unused TCP port and return it."""
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_port(host: str, port: int, timeout: float = 10.0) -> None:
    """Block until the TCP port is open (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.25)
    raise RuntimeError(f"Port {port} did not open within {timeout} s")


def _write_stub_app(tmpdir: Path) -> Path:
    """
    Create stub_app.py that patches pickle.load/open *before* importing
    mushroom_ml.api, so the server process never needs the real .pkl file.
    """
    code = textwrap.dedent(
        """
        import builtins, io, os, pickle

        # ---- patch first ------------------------------------------------------
        class _Dummy:
            feature_names_in_: list[str] = []
            def predict(self, df): return [1]

        pickle.load = lambda f: _Dummy()

        _real_open = builtins.open
        def _fake_open(path, *a, **k):
            if isinstance(path, (str, os.PathLike)) and str(path).endswith("model.pkl"):
                return io.BytesIO(b"dummy")
            return _real_open(path, *a, **k)
        builtins.open = _fake_open

        # ---- now import the real API -----------------------------------------
        import mushroom_ml.api as base

        app = base.app
        """
    )
    file = tmpdir / "stub_app.py"
    file.write_text(code)
    return file


# --------------------------------------------------------------------------- the test
@pytest.mark.e2e
def test_live_server_predict(tmp_path: Path):
    port = _free_port()
    app_file = _write_stub_app(tmp_path)

    # Start Uvicorn in a separate process
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", f"{app_file.stem}:app", "--port", str(port)],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_port("localhost", port, timeout=10.0)

        # -------- patch THIS pytest process so it can import api.py ------------
        import pickle as _pickle

        _real_load = _pickle.load
        _real_open = builtins.open

        def _fake_open2(path, *a, **k):
            if isinstance(path, (str, os.PathLike)) and str(path).endswith("model.pkl"):
                return io.BytesIO(b"dummy")
            return _real_open(path, *a, **k)

        builtins.open = _fake_open2
        _pickle.load = lambda f: object()

        import mushroom_ml.api as api_module  # noqa: WPS433
        importlib.reload(api_module)

        # -------- restore original functions so other tests stay clean ---------
        builtins.open = _real_open
        _pickle.load = _real_load

        # ------------------- call the live server ------------------------------
        features = [opts[0] for opts in api_module.FEATURE_OPTIONS.values()]
        resp = requests.post(
            f"http://localhost:{port}/predict",
            json=features,
            timeout=5,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["input"] == features
        assert data["prediction"] == 1
    finally:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=5)
