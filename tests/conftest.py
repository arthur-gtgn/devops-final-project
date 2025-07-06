import pytest
import mlflow

@pytest.fixture(autouse=True, scope="function")
def local_mlflow(tmp_path):
    """
    Force every test to log to a fresh, disposable MLflow store on disk,
    instead of your real DagsHub tracking server.
    """
    uri = f"file://{tmp_path}"
    mlflow.set_tracking_uri('default')
    yield                        # tests run here
    # tmp_path is wiped automatically; nothing else to clean up
