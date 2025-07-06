"""
Verify that the Streamlit front-end and FastAPI back-end share the exact same
FEATURE_OPTIONS dict (same keys, same order).
"""
import sys
import types

def test_feature_options_in_sync():
    # ---- create a minimal fake Streamlit API ----
    st = types.ModuleType("streamlit")
    st.title = lambda *a, **k: None # type: ignore
    st.success = st.error = lambda *a, **k: None # type: ignore

    # columns() → list of dummy Column objects with selectbox()
    class _DummyCol:
        selectbox = lambda *a, **k: None
    st.columns = lambda n: [_DummyCol() for _ in range(n)] # type: ignore

    # st.form(...) context manager with selectbox() + form_submit_button()
    class _DummyForm:
        def __enter__(self): return self
        def __exit__(self, *exc): pass
        selectbox = _DummyCol.selectbox
        form_submit_button = lambda *a, **k: False
    st.form = lambda name: _DummyForm() # type: ignore

    # Streamlit also exposes form_submit_button at the top level
    st.form_submit_button = lambda *a, **k: False # type: ignore

    sys.modules["streamlit"] = st      # inject before importing app.py

    from mushroom_ml import app as app_module
    from mushroom_ml import api as api_module

    assert list(app_module.FEATURE_OPTIONS.keys()) == list(
        api_module.FEATURE_OPTIONS.keys()
    ), "FEATURE_OPTIONS mismatch between UI and API"
