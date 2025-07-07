"""
Unit-test the /predict endpoint without the real model file.

The real model is loaded at *import time* in mushroom_ml.api, so we monkey-patch
`pickle.load` and `open` **before** importing the module.
"""
import builtins, io, importlib
from fastapi.testclient import TestClient
import pytest


@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    # --- Stub out the RandomForest model and the pickle/open machinery ----------
    class DummyModel:
        feature_names_in_: list[str] = []
        def predict(self, df):
            return [1]                     # always “edible”

    import pickle
    monkeypatch.setattr(pickle, "load", lambda file_obj: DummyModel())

    # Any attempt to open the real .pkl file now succeeds and returns dummy bytes
    monkeypatch.setattr(
        builtins,
        "open",
        lambda *a, **k: io.BytesIO(b"dummy"),   # noqa: S301 – fine in tests
    )


def test_api_predict_returns_number():
    # Import AFTER patching so api.py sees our stubs
    from mushroom_ml import api as api_module
    importlib.reload(api_module)

    client = TestClient(api_module.app)

    # Build a syntactically correct feature list: first option of every feature
    features = [opts[0] for opts in api_module.FEATURE_OPTIONS.values()]

    # POST because we send a JSON body
    response = client.post("/predict", json=features)
    assert response.status_code == 200

    payload = response.json()
    assert payload["input"] == features
    assert isinstance(payload["prediction"], (int, float))
    assert payload["prediction"] == 1
