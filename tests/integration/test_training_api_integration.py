"""
Train a lightweight RandomForest with train_model(), inject it into the FastAPI
app, then hit /predict through TestClient.  No real .pkl file is needed.
"""
import io, builtins, importlib, pickle, os
from fastapi.testclient import TestClient
from mushroom_ml.train import train_model



def test_train_then_predict(monkeypatch):
    # --- 1. train a tiny model in-memory --------------------------------------
    class DummyLoaderModel:
        feature_names_in_: list[str] = []
        def predict(self, df):
            return [1]

    # Return our dummy model for *any* pickle.load(...)
    monkeypatch.setattr(pickle, "load", lambda f: DummyLoaderModel())

    real_open = builtins.open

    def fake_open(path, *args, **kwargs):          # noqa: D401
        if isinstance(path, (str, os.PathLike)) and str(path).endswith("model.pkl"):
            return io.BytesIO(b"dummy")            # the only file we fake
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)

    # --- 2. make the API use *this* in-memory model instead of the pickle -----
    from mushroom_ml import api as api_module
    importlib.reload(api_module)           # ensure router state is fresh

    _ = train_model(n_estimators=5, max_depth=2)

    client = TestClient(api_module.app)

    features = [opts[0] for opts in api_module.FEATURE_OPTIONS.values()]
    resp = client.post("/predict", json=features)

    assert resp.status_code == 200
    data = resp.json()
    assert data["input"] == features
    assert isinstance(data["prediction"], (int, float))
