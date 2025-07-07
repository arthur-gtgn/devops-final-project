from fastapi import FastAPI, Body
from typing import List
import pickle
import pandas as pd
from sklearn.datasets import load_digits

app = FastAPI()

# Enable CORS for local development
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(http://)?(localhost|127\\.0\\.0\\.1)(:\\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = pickle.load(open("models/random_forest_model/model.pkl", "rb"))

from fastapi import Query

FEATURE_OPTIONS = {
    "cap_shape": ['b','c','x','f','k','s'],
    "cap_surface": ['f','g','y','s'],
    "cap_color": ['n','b','c','g','r','p','u','e','w','y'],
    "bruises": ['t','f'],
    "odor": ['a','l','c','y','f','m','n','p','s'],
    "gill_attachment": ['a','d','f','n'],
    "gill_spacing": ['c','w','d'],
    "gill_size": ['b','n'],
    "gill_color": ['k','n','b','h','g','r','o','p','u','e','w','y'],
    "stalk_shape": ['e','t'],
    "stalk_root": ['b','c','u','e','z','r','?'],
    "stalk_surface_above_ring": ['f','y','k','s'],
    "stalk_surface_below_ring": ['f','y','k','s'],
    "stalk_color_above_ring": ['n','b','c','g','o','p','e','w','y'],
    "stalk_color_below_ring": ['n','b','c','g','o','p','e','w','y'],
    "veil_type": ['p','u'],
    "veil_color": ['n','o','w','y'],
    "ring_number": ['n','o','t'],
    "ring_type": ['c','e','f','l','n','p','s','z'],
    "spore_print_color": ['k','n','b','h','r','o','u','w','y'],
    "population": ['a','c','n','s','v','y'],
    "habitat": ['g','l','m','p','u','w','d']
}

@app.post("/predict")
def predict(
    features: List[str] = Body(
        ...,
        min_items=len(FEATURE_OPTIONS),
        max_items=len(FEATURE_OPTIONS),
        example=[list(FEATURE_OPTIONS.values())[0][0] for _ in range(len(FEATURE_OPTIONS))],
        description="List of feature values in the same order as defined in FEATURE_OPTIONS",
        title="Mushroom Features"
    )
):
    """
    Features must be a list of 22 characters in the order of FEATURE_OPTIONS.keys()
    """

    df = pd.DataFrame([features], columns=FEATURE_OPTIONS.keys()) # type: ignore
    df = pd.get_dummies(df) # type: ignore

    df.columns = [c.replace("_", "-") for c in df.columns]

    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    y = model.predict(df)

    return {"input": features, "prediction": float(y[0])}
